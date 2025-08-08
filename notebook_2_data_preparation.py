# Databricks notebook source
# MAGIC %md
# MAGIC # Pizza Company VOC Analysis - Data Preparation & Chunking
# MAGIC
# MAGIC This notebook prepares the VOC data for vector search by creating meaningful chunks and generating embeddings.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import re
from typing import List

# Configuration
CATALOG_NAME = "users"
SCHEMA_NAME = "kevin_ippen"
SOURCE_TABLE = "voc_pizza_table"
CHUNKS_TABLE = "voc_pizza_chunks"

# Chunking parameters
MAX_CHUNK_LENGTH = 300  # Characters per chunk
MIN_CHUNK_LENGTH = 50   # Minimum viable chunk size
OVERLAP_SIZE = 50       # Character overlap between chunks

spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

print("✅ Environment configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Source Data

# COMMAND ----------

# Load the cleaned VOC data
df_source = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.{SOURCE_TABLE}")

print(f"Source data loaded: {df_source.count()} records")
df_source.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text Preprocessing Functions

# COMMAND ----------

def clean_text(text: str) -> str:
    """Clean and normalize text for better chunking."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep punctuation for context
    text = re.sub(r'[^\w\s\.,!?;:-]', '', text)
    
    return text

def create_chunks(text: str, max_length: int = MAX_CHUNK_LENGTH, 
                 min_length: int = MIN_CHUNK_LENGTH, 
                 overlap: int = OVERLAP_SIZE) -> List[str]:
    """
    Create overlapping chunks from text.
    For short comments, return the full text as a single chunk.
    """
    text = clean_text(text)
    
    if len(text) <= max_length:
        return [text] if len(text) >= min_length else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # If we're not at the end, try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence boundary (. ! ?)
            sentence_end = max(
                text.rfind('. ', start, end),
                text.rfind('! ', start, end),
                text.rfind('? ', start, end)
            )
            
            if sentence_end > start:
                end = sentence_end + 1
            else:
                # Fall back to word boundary
                word_end = text.rfind(' ', start, end)
                if word_end > start:
                    end = word_end
        
        chunk = text[start:end].strip()
        if len(chunk) >= min_length:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, end - overlap)
        
        # Prevent infinite loop
        if end >= len(text):
            break
    
    return chunks

# Register UDF for chunking
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

clean_text_udf = udf(clean_text, StringType())
create_chunks_udf = udf(create_chunks, ArrayType(StringType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Chunks with Metadata

# COMMAND ----------

# Define the UDF to create chunks
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

def create_chunks(comments):
    # Example logic for creating chunks
    # Ensure max() is used correctly
    chunks = []
    max_length = 100  # Example max length for a chunk
    for i in range(0, len(comments), max_length):
        chunks.append(comments[i:i + max_length])
    return chunks

create_chunks_udf = udf(create_chunks, ArrayType(StringType()))

# Create chunks with full context metadata
df_with_chunks = df_source.withColumn(
    "chunks", create_chunks_udf(col("comments"))
).filter(size(col("chunks")) > 0)

# Explode chunks to create one row per chunk
df_chunks = df_with_chunks.select(
    col("record_id").alias("source_record_id"),
    col("comments").alias("original_comment"),
    col("satisfaction"),
    col("order_source"),
    col("service_method"),
    col("customer_type"),
    col("order_date"),
    col("service_time"),
    col("comment_length"),
    col("word_count"),
    posexplode(col("chunks")).alias("chunk_index", "chunk_text")
).withColumn("chunk_id", concat(col("source_record_id"), lit("_"), col("chunk_index")))

display(df_chunks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enhance Chunks with Context

# COMMAND ----------

# Add contextual information to each chunk
df_chunks_enhanced = df_chunks.withColumn(
    "context_metadata", 
    concat_ws(" | ",
        concat(lit("Satisfaction: "), col("satisfaction")),
        concat(lit("Service: "), col("service_method")),
        concat(lit("Customer: "), col("customer_type")),
        concat(lit("Source: "), col("order_source"))
    )
).withColumn(
    "chunk_with_context",
    concat(
        col("chunk_text"),
        lit("\n[Context: "),
        col("context_metadata"),
        lit("]")
    )
).withColumn(
    "chunk_length", length(col("chunk_text"))
).withColumn(
    "chunk_word_count", size(split(col("chunk_text"), "\\s+"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality Assessment of Chunks

# COMMAND ----------

print("=== CHUNK QUALITY ASSESSMENT ===")

# Basic chunk statistics
df_chunks_enhanced.select(
    count("*").alias("total_chunks"),
    avg("chunk_length").alias("avg_chunk_length"),
    min("chunk_length").alias("min_chunk_length"),
    max("chunk_length").alias("max_chunk_length"),
    avg("chunk_word_count").alias("avg_word_count")
).show()

# Distribution by satisfaction
print("\n=== CHUNKS BY SATISFACTION ===")
df_chunks_enhanced.groupBy("satisfaction") \
    .agg(count("*").alias("chunk_count")) \
    .orderBy(col("chunk_count").desc()) \
    .show()

# Sample chunks
print("\n=== SAMPLE CHUNKS ===")
df_chunks_enhanced.select("chunk_id", "chunk_text", "satisfaction", "service_method") \
    .limit(3) \
    .show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Chunks to Unity Catalog

# COMMAND ----------

# Write chunks table
df_chunks_enhanced.write \
    .mode("overwrite") \
    .option("delta.autoOptimize.optimizeWrite", "true") \
    .option("delta.autoOptimize.autoCompact", "true") \
    .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.{CHUNKS_TABLE}")

print(f"✅ Chunks saved to {CATALOG_NAME}.{SCHEMA_NAME}.{CHUNKS_TABLE}")

# Add table properties
spark.sql(f"""
    ALTER TABLE {CATALOG_NAME}.{SCHEMA_NAME}.{CHUNKS_TABLE} 
    SET TBLPROPERTIES (
        'comment' = 'Chunked VOC comments with metadata for vector search',
        'chunk_max_length' = '{MAX_CHUNK_LENGTH}',
        'chunk_overlap' = '{OVERLAP_SIZE}'
    )
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Analytical Views

# COMMAND ----------

# Create a view for easy analysis
spark.sql(f"""
    CREATE OR REPLACE VIEW {CATALOG_NAME}.{SCHEMA_NAME}.voc_analysis_view AS
    SELECT 
        satisfaction,
        service_method,
        customer_type,
        order_source,
        COUNT(*) as comment_count,
        AVG(chunk_length) as avg_chunk_length,
        AVG(service_time) as avg_service_time
    FROM {CATALOG_NAME}.{SCHEMA_NAME}.{CHUNKS_TABLE}
    GROUP BY satisfaction, service_method, customer_type, order_source
""")

print("✅ Analysis view created")

# Test the view
print("\n=== SAMPLE ANALYSIS ===")
spark.sql(f"""
    SELECT satisfaction, service_method, comment_count, avg_service_time
    FROM {CATALOG_NAME}.{SCHEMA_NAME}.voc_analysis_view
    WHERE comment_count > 50
    ORDER BY comment_count DESC
    LIMIT 10
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Sample Data for Testing

# COMMAND ----------

# Create a small sample for testing (useful during development)
df_sample = df_chunks_enhanced.sample(0.1, seed=42).limit(100)

df_sample.write \
    .mode("overwrite") \
    .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.{CHUNKS_TABLE}_sample")

print("✅ Sample dataset created for testing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC ✅ **Completed:**
# MAGIC - Text cleaning and chunking strategy implemented
# MAGIC - Chunks created with contextual metadata
# MAGIC - Data quality assessment performed
# MAGIC - Chunks saved to Unity Catalog
# MAGIC - Analytical views created
# MAGIC
# MAGIC **Next:** Run notebook `03_vector_search_setup.py` to create embeddings and set up vector search.
