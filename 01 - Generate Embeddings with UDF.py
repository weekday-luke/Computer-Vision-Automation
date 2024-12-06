# Databricks notebook source
# MAGIC %md
# MAGIC ## Register the model with MLflow
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Working with Images in a Unity Catalog Volume using Python
# MAGIC
# MAGIC This snippet demonstrates how to read and display images from a folder using the Python `Pillow` library. The code identifies `.jpg` files in a specified directory, reads the first image, and displays it.

# COMMAND ----------

from PIL import Image
import os

# Get the list of all .jpg files in the folder
folder_path = "/Volumes/shutterstock_free_sample_dataset_1000_high_resolution_images_metadata/sample_datasets/set1_image_files/medium/"
jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# Read the first image in the folder
first_image_path = os.path.join(folder_path, jpg_files[0])
first_image = Image.open(first_image_path)

# Display the first image
display(first_image)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading Images into a Spark DataFrame
# MAGIC
# MAGIC This snippet demonstrates how to load image files from the Unity Catalog Volume into a Spark DataFrame. It uses the `binaryFile` data source to read image files and includes their metadata and file paths.

# COMMAND ----------

from pyspark.sql.functions import input_file_name

# Define folder path
image_folder_path = "/Volumes/shutterstock_free_sample_dataset_1000_high_resolution_images_metadata/sample_datasets/set1_image_files/medium/"  # Replace with your folder path

# Load images into Spark DataFrame
images_df = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.jpg")  # Replace with your desired file extension, e.g., *.png
    .option("recursiveFileLookup", "true")
    .load(image_folder_path)
    .withColumn("file_path", input_file_name())
)

# Show the DataFrame schema and optionally show the DataFrame
images_df.printSchema()
#images_df.display()

# COMMAND ----------

unique_file_paths_count = images_df.select("file_path").distinct().count()
print(f"Number of unique file paths: {unique_file_paths_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generating CLIP Embeddings for Images in PySpark
# MAGIC
# MAGIC This snippet demonstrates how to define a PySpark User Defined Function (UDF) to generate embeddings for images using OpenAI's CLIP model. The embeddings can be used for downstream tasks like image search, classification, or clustering.
# MAGIC

# COMMAND ----------

import torch
from transformers import CLIPProcessor, CLIPModel
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define UDF for embedding generation
def generate_clip_embedding(image_bytes):
    try:
        # Decode and preprocess the image
        from PIL import Image
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = clip_model.get_image_features(**inputs).squeeze().tolist()
        
        return embeddings
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Register the UDF
embedding_udf = udf(generate_clip_embedding, ArrayType(FloatType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Applying CLIP UDF to Generate Image Embeddings
# MAGIC
# MAGIC This snippet applies the previously defined User Defined Function (UDF) to a Spark DataFrame to generate embeddings for images. The embeddings are added as a new column alongside the image file paths.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

# Apply the UDF to generate embeddings and add a unique ID
images_with_embeddings_df = images_df.select(
    "file_path",
    embedding_udf("content").alias("clip_embeddings")
).withColumn("id", monotonically_increasing_id())

# COMMAND ----------

# Count total rows and unique rows before the UDF was applied
total_rows_before = images_df.count()
unique_rows_before = images_df.select("file_path").distinct().count()

# Count total rows and unique rows after the UDF was applied
total_rows_after = images_with_embeddings_df.count()
unique_rows_after = images_with_embeddings_df.select("file_path").distinct().count()

# Display the results
result_df = spark.createDataFrame(
    [(total_rows_before, unique_rows_before, total_rows_after, unique_rows_after)],
    ["total_rows_before", "unique_rows_before", "total_rows_after", "unique_rows_after"]
)

display(result_df)

# COMMAND ----------

unique_file_paths_df = images_with_embeddings_df.select("file_path").distinct()
display(unique_file_paths_df)

# COMMAND ----------

# Create the schema if it doesn't exist
spark.sql("CREATE SCHEMA IF NOT EXISTS shared.cv_similarity_search")

# Save the DataFrame to a Delta table in Unity Catalog
images_with_embeddings_df.write.format("delta").saveAsTable("shared.cv_similarity_search.images_with_embeddings_df")

# Enable change data feed on the Delta table
spark.sql(
    "ALTER TABLE shared.cv_similarity_search.images_with_embeddings_df SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

# COMMAND ----------


