# Databricks notebook source
# MAGIC %md
# MAGIC # Create Vector Search Index and populate with pre-computed embeddings
# MAGIC
# MAGIC [Mosaic AI Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html)
# MAGIC
# MAGIC ### Working with images, video, or non-text data
# MAGIC 1. Pre-compute the embeddings and use a Delta Sync Index with self-managed embeddings.
# MAGIC 2. Don’t store binary formats such as images as metadata, as this adversely affects latency. Instead, store the path of the file as metadata.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

endpoint_name = "one-env-shared-endpoint-15" # name of Vector Search endpoint to host the index - create one if necessary
source_table_name = "shared.cv_similarity_search.images_with_embeddings_df" # name of table in UC with pre computed embeddings
index_name = "shared.cv_similarity_search.images_with_embeddings_df_index" # name of the index we are creating
primary_key = "file_path" # this can be set as a unique id or the file path
embedding_vector_column = "clip_embeddings" # the column with the embeddings - ArrayType(FloatType())

# COMMAND ----------

# Read the table from source_table_name
images_with_embeddings_df = spark.table(source_table_name)

# Show Sample Data
display(images_with_embeddings_df)

# COMMAND ----------

unique_file_paths = images_with_embeddings_df.select(primary_key).distinct()
display(unique_file_paths)

# COMMAND ----------

client = VectorSearchClient()

index = client.create_delta_sync_index(
  endpoint_name=endpoint_name,
  source_table_name=source_table_name,
  index_name=index_name,
  pipeline_type="TRIGGERED",
  primary_key=primary_key,
  embedding_dimension=512,
  embedding_vector_column=embedding_vector_column,
)
