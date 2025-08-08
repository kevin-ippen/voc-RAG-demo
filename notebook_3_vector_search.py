# Databricks notebook source
# MAGIC %md
# MAGIC # Pizza Company VOC Analysis - Vector Search Setup
# MAGIC 
# MAGIC This notebook sets up Databricks Vector Search for semantic search over VOC chunks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Vector Search imports
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import time

# Configuration
CATALOG_NAME = "pizza_voc"
SCHEMA_NAME = "customer_feedback"
CHUNKS_TABLE = "voc_comments_chunks"

# Vector Search configuration
VECTOR_SEARCH_ENDPOINT = "pizza_voc_endpoint"
VECTOR_INDEX_NAME = "voc_chunks_index"
EMBEDDING_MODEL = "databricks-bge-large-en"  # Databricks-hosted embedding model

# Initialize clients
spark = SparkSession.builder.getOrCreate()
vector_client = VectorSearchClient()
w = WorkspaceClient()

spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

print("✅ Vector Search client initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Source Data

# COMMAND ----------

# Load and inspect the chunks table
df_chunks = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.{CHUNKS_TABLE}")

print(f"Total chunks available: {df_chunks.count()}")
print("\nChunk schema:")
df_chunks.printSchema()

# Show sample data
print("\n=== SAMPLE CHUNKS FOR INDEXING ===")
df_chunks.select(
    "chunk_id", 
    "chunk_text", 
    "satisfaction", 
    "service_method",
    "chunk_length"
).limit(3).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Endpoint

# COMMAND ----------

# Check if endpoint exists, create if not
try:
    endpoint = vector_client.get_endpoint(VECTOR_SEARCH_ENDPOINT)
    print(f"✅ Endpoint '{VECTOR_SEARCH_ENDPOINT}' already exists")
except Exception as e:
    print(f"Creating new endpoint '{VECTOR_SEARCH_ENDPOINT}'...")
    endpoint = vector_client.create_endpoint(
        name=VECTOR_SEARCH_ENDPOINT,
        endpoint_type="STANDARD"  # Use STANDARD for production, SERVERLESS for dev
    )
    print(f"✅ Endpoint '{VECTOR_SEARCH_ENDPOINT}' created")

# Wait for endpoint to be ready
print("Waiting for endpoint to be ready...")
while endpoint.endpoint_status.state not in ["ONLINE", "PROVISIONING_SUCCESS"]:
    time.sleep(30)
    endpoint = vector_client.get_endpoint(VECTOR_SEARCH_ENDPOINT)
    print(f"Endpoint status: {endpoint.endpoint_status.state}")

print("✅ Endpoint is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data for Vector Index

# COMMAND ----------

# Select and prepare columns for vector indexing
# We'll use chunk_text as primary content and include metadata for filtering
df_for_indexing = df_chunks.select(
    col("chunk_id").alias("id"),  # Primary key
    col("chunk_text").alias("text"),  # Text to embed
    col("chunk_with_context").alias("text_with_context"),  # Enhanced text
    col("source_record_id"),
    col("satisfaction"),
    col("service_method"), 
    col("customer_type"),
    col("order_source"),
    col("order_date"),
    col("service_time"),
    col("chunk_length"),
    col("chunk_word_count")
)

# Ensure we have clean data for indexing
df_for_indexing = df_for_indexing.filter(
    col("text").isNotNull() & 
    (length(trim(col("text"))) > 0) &
    col("id").isNotNull()
)

print(f"Prepared {df_for_indexing.count()} chunks for vector indexing")

# Write to a dedicated table for vector search
INDEX_SOURCE_TABLE = f"{CHUNKS_TABLE}_for_index"

df_for_indexing.write \
    .mode("overwrite") \
    .option("delta.autoOptimize.optimizeWrite", "true") \
    .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.{INDEX_SOURCE_TABLE}")

print(f"✅ Index source table created: {CATALOG_NAME}.{SCHEMA_NAME}.{INDEX_SOURCE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Index

# COMMAND ----------

# Full table name for the vector index
source_table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{INDEX_SOURCE_TABLE}"
index_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{VECTOR_INDEX_NAME}"

# Check if index exists
try:
    existing_index = vector_client.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=index_name)
    print(f"Index '{index_name}' already exists. Deleting to recreate...")
    vector_client.delete_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=index_name)
    time.sleep(10)  # Wait for deletion
except Exception as e:
    print(f"Index doesn't exist yet. Creating new index...")

# Create the vector index
print(f"Creating vector index '{index_name}'...")
index = vector_client.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    source_table_name=source_table_name,
    index_name=index_name,
    primary_key="id",
    embedding_source_column="text",  # Column to create embeddings from
    embedding_model_endpoint_name=EMBEDDING_MODEL,
    pipeline_type="TRIGGERED"  # Use TRIGGERED for manual sync, CONTINUOUS for auto-sync
)

print(f"✅ Vector index '{index_name}' created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sync Vector Index

# COMMAND ----------

# Trigger initial sync
print("Starting vector index sync...")
vector_client.sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=index_name
)

# Monitor sync progress
print("Monitoring sync progress...")
max_wait_time = 1800  # 30 minutes max wait
start_time = time.time()

while True:
    index_status = vector_client.get_index(VECTOR_SEARCH_ENDPOINT, index_name)
    status = index_status.status.detailed_state
    
    print(f"Index status: {status}")
    
    if status == "ONLINE":
        print("✅ Vector index sync completed successfully!")
        break
    elif status in ["FAILED", "OFFLINE"]:
        print(f"❌ Vector index sync failed with status: {status}")
        break
    elif time.time() - start_time > max_wait_time:
        print("⚠️  Sync taking longer than expected. Check status in UI.")
        break
    
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Vector Search

# COMMAND ----------

# Test the vector search with sample queries
test_queries = [
    "Pizza was not cooked properly",
    "Great service and fast delivery", 
    "The staff was very friendly",
    "Order took too long to arrive",
    "Pizza taste was amazing"
]

print("=== TESTING VECTOR SEARCH ===")

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    
    try:
        results = vector_client.similarity_search(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=index_name,
            query_text=query,
            columns=["id", "text", "satisfaction", "service_method"],
            num_results=3
        )
        
        if results and 'result' in results and 'data_array' in results['result']:
            for i, result in enumerate(results['result']['data_array'][:3]):
                print(f"  {i+1}. [{result[2]}] {result[1][:100]}...")
        else:
            print("  No results found")
            
    except Exception as e:
        print(f"  Error: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Utility Functions

# COMMAND ----------

def search_voc_comments(query: str, num_results: int = 5, satisfaction_filter: str = None):
    """
    Search VOC comments using vector similarity.
    
    Args:
        query: Search query text
        num_results: Number of results to return
        satisfaction_filter: Optional filter by satisfaction level
    
    Returns:
        List of matching chunks with metadata
    """
    try:
        # Build filters if specified
        filters = None
        if satisfaction_filter:
            filters = {"satisfaction": satisfaction_filter}
        
        results = vector_client.similarity_search(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=index_name,
            query_text=query,
            columns=[
                "id", "text", "satisfaction", "service_method", 
                "customer_type", "order_source", "service_time"
            ],
            num_results=num_results,
            filters=filters
        )
        
        if results and 'result' in results and 'data_array' in results['result']:
            return results['result']['data_array']
        else:
            return []
            
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

def get_satisfaction_insights(topic: str, top_k: int = 10):
    """Get insights about a topic across different satisfaction levels."""
    
    print(f"=== INSIGHTS FOR TOPIC: '{topic}' ===")
    
    satisfaction_levels = ["Highly Satisfied", "Satisfied", "Not Satisfied", "Accuracy", "Taste", "Appearance"]
    
    for satisfaction in satisfaction_levels:
        print(f"\n--- {satisfaction} ---")
        results = search_voc_comments(topic, num_results=3, satisfaction_filter=satisfaction)
        
        if results:
            for result in results:
                print(f"  • {result[1][:80]}...")
        else:
            print("  No matching comments found")

# Test the utility functions
print("=== TESTING UTILITY FUNCTIONS ===")

# Test basic search
results = search_voc_comments("delivery time", num_results=3)
print(f"\nFound {len(results)} results for 'delivery time'")

# Test satisfaction insights
get_satisfaction_insights("pizza quality", top_k=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Index Summary

# COMMAND ----------

# Get final index statistics
index_info = vector_client.get_index(VECTOR_SEARCH_ENDPOINT, index_name)

print("=== VECTOR INDEX SUMMARY ===")
print(f"Index Name: {index_name}")
print(f"Endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"Status: {index_info.status.detailed_state}")
print(f"Embedding Model: {EMBEDDING_MODEL}")
print(f"Source Table: {source_table_name}")

# Show index properties
if hasattr(index_info, 'index_spec'):
    print(f"Primary Key: {index_info.index_spec.primary_key}")
    print(f"Embedding Source: {index_info.index_spec.embedding_source_column}")

print("\n✅ Vector Search setup completed successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC ✅ **Completed:**
# MAGIC - Vector Search endpoint created and configured
# MAGIC - Vector index built with embeddings for VOC chunks
# MAGIC - Search functionality tested and validated
# MAGIC - Utility functions created for easy searching
# MAGIC 
# MAGIC **Next:** Run notebook `04_rag_model_serving.py` to set up model serving for the complete RAG pipeline.