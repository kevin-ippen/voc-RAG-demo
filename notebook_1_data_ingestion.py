# Databricks notebook source
# MAGIC %md
# MAGIC # Pizza Company VOC Analysis - Data Ingestion & Exploration
# MAGIC
# MAGIC This notebook ingests customer feedback data into Unity Catalog and performs initial exploration for our RAG system.
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Unity Catalog enabled workspace
# MAGIC - DBR 14.3 LTS ML or higher

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import re

# Configuration
CATALOG_NAME = "users"
SCHEMA_NAME = "kevin_ippen" 
TABLE_NAME = "voc_pizza_table"

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

print(f"Target table: {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Unity Catalog Structure

# COMMAND ----------

# Create catalog if it doesn't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")

# Create schema if it doesn't exist  
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")

# Set current catalog and schema
spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

print("✅ Unity Catalog structure created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Clean Data

# COMMAND ----------

# Define schema for the CSV data
voc_schema = StructType([
    StructField("comments", StringType(), True),
    StructField("order_source", StringType(), True),
    StructField("satisfaction", StringType(), True),
    StructField("order_date", StringType(), True),
    StructField("royalty_sales", StringType(), True),
    StructField("service_method", StringType(), True),
    StructField("customer_type", StringType(), True),
    StructField("service_time", DoubleType(), True)
])

# Load CSV data
# Update the path to your CSV file location in DBFS or external storage
csv_path = "/Volumes/users/kevin_ippen/voc-data-dpz/voc_comments_redesign_6_13_2025(All Redesign Comments).csv"

df_raw = spark.read \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .option("encoding", "UTF-8") \
    .schema(voc_schema) \
    .csv(csv_path)

print(f"Raw data loaded: {df_raw.count()} rows")

# COMMAND ----------

# Clean and prepare data
df_cleaned = df_raw \
    .filter(col("comments").isNotNull()) \
    .filter(length(trim(col("comments"))) > 0) \
    .withColumn("comments", trim(col("comments"))) \
    .withColumn("order_date", to_date(col("order_date"), "M/d/yyyy")) \
    .withColumn("comment_length", length(col("comments"))) \
    .withColumn("word_count", size(split(col("comments"), "\\s+"))) \
    .withColumn("record_id", monotonically_increasing_id())

# Add data quality flags
df_cleaned = df_cleaned \
    .withColumn("is_short_comment", col("comment_length") < 10) \
    .withColumn("is_long_comment", col("comment_length") > 500)

print(f"Cleaned data: {df_cleaned.count()} rows with valid comments")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Assessment

# COMMAND ----------

# Basic statistics
print("=== DATA QUALITY SUMMARY ===")
df_cleaned.select(
    count("*").alias("total_records"),
    avg("comment_length").alias("avg_comment_length"),
    min("comment_length").alias("min_comment_length"),
    max("comment_length").alias("max_comment_length"),
    sum(col("is_short_comment").cast("int")).alias("short_comments"),
    sum(col("is_long_comment").cast("int")).alias("long_comments")
).show()

# Satisfaction distribution
print("\n=== SATISFACTION DISTRIBUTION ===")
df_cleaned.groupBy("satisfaction") \
    .agg(count("*").alias("count")) \
    .orderBy(col("count").desc()) \
    .show()

# Service method breakdown
print("\n=== SERVICE METHOD BREAKDOWN ===")
df_cleaned.groupBy("service_method") \
    .agg(count("*").alias("count")) \
    .orderBy(col("count").desc()) \
    .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Unity Catalog

# COMMAND ----------

# Write to Delta table in Unity Catalog
df_cleaned.write \
    .mode("overwrite") \
    .option("delta.autoOptimize.optimizeWrite", "true") \
    .option("delta.autoOptimize.autoCompact", "true") \
    .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}")

print(f"✅ Data saved to {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}")

# Add table comment
spark.sql(f"""
    ALTER TABLE {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME} 
    SET TBLPROPERTIES ('comment' = 'Pizza company voice of customer comments for RAG analysis')
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Data Preview

# COMMAND ----------

# Show sample records
sample_df = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}")

print("=== SAMPLE RECORDS ===")
sample_df.select("record_id", "comments", "satisfaction", "service_method", "comment_length") \
    .limit(5) \
    .show(truncate=False)

# Show some examples by satisfaction level
print("\n=== EXAMPLES BY SATISFACTION LEVEL ===")
for satisfaction_level in ["Highly Satisfied", "Not Satisfied", "Accuracy"]:
    print(f"\n--- {satisfaction_level} ---")
    sample_df.filter(col("satisfaction") == satisfaction_level) \
        .select("comments") \
        .limit(2) \
        .show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC ✅ **Completed:**
# MAGIC - Data loaded into Unity Catalog
# MAGIC - Basic data quality assessment
# MAGIC - Table structure optimized for Delta Lake
# MAGIC
# MAGIC **Next:** Run notebook `02_data_preparation_and_chunking.py` to prepare the data for vector search.
