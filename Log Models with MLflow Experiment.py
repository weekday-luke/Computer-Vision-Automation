# Databricks notebook source
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

import mlflow.pyfunc
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
import torch
import pandas as pd
from mlflow.models.signature import infer_signature
import time

class CLIPEmbeddingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load the CLIP model and processor when the MLflow model is loaded.
        """
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def predict(self, context, model_input):
        """
        Generate embeddings for image binary data.

        Args:
            context: MLflow context (unused here).
            model_input: Pandas DataFrame with a 'content' column containing binary image data.

        Returns:
            Pandas Series with embeddings as lists of floats.
        """
        results = []
        for image_bytes in model_input["content"]:
            try:
                # Decode and preprocess the image
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                
                # Generate embeddings
                with torch.no_grad():
                    embedding = self.model.get_image_features(**inputs).squeeze().tolist()
                results.append(embedding)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append(None)
        
        return pd.Series(results)

# Metadata for the model
metadata = {
    "model_name": "openai/clip-vit-base-patch32",
    "processor_name": "openai/clip-vit-base-patch32",
    "description": "CLIP model for generating image embeddings.",
    "source": "Hugging Face Transformers",
    "type": "image-embedding"
}

# Set the MLflow experiment name
mlflow.set_experiment("/Users/luke.gardner@databricks.com/Computer Vision Models")

# Prepare sample input
sample_input = images_df.select("content").limit(1).toPandas()

# Test the CLIPEmbeddingModel locally
sample_model = CLIPEmbeddingModel()
sample_model.load_context(None)
sample_output = sample_model.predict(None, sample_input)

# Infer the signature
signature = infer_signature(sample_input, sample_output)

# Start an MLflow run
with mlflow.start_run(run_name="OpenAI - CLIP") as run:
    # Log metadata as tags
    for key, value in metadata.items():
        mlflow.set_tag(key, value)

    # Log the model
    mlflow.pyfunc.log_model(
        artifact_path="clip_embedding_model",
        python_model=CLIPEmbeddingModel(),
        registered_model_name="users.luke_gardner.clip_embedding_model",
        signature=signature
    )
    
    # Measure runtime performance
    start_time = time.time()
    _ = sample_model.predict(None, sample_input)
    end_time = time.time()
    
    # Log parameters and metrics
    mlflow.log_param("num_inputs", len(sample_input))
    mlflow.log_metric("embedding_generation_time", (end_time - start_time) / len(sample_input))
    mlflow.log_metric("embedding_length", len(sample_output.iloc[0]) if sample_output.iloc[0] else 0)


# COMMAND ----------

pip install git+https://github.com/facebookresearch/segment-anything.git

# COMMAND ----------

import mlflow.pyfunc
import torch
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
from io import BytesIO
import pandas as pd
from mlflow.models.signature import infer_signature
import time

class SegmentAnythingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load the Segment Anything model and predictor when the MLflow model is loaded.
        """
        self.model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SamPredictor(self.model)

    def predict(self, context, model_input):
        """
        Generate segmentation masks for image binary data.

        Args:
            context: MLflow context (unused here).
            model_input: Pandas DataFrame with a 'content' column containing binary image data.

        Returns:
            Pandas Series with segmentation masks (e.g., numpy arrays).
        """
        results = []
        for image_bytes in model_input["content"]:
            try:
                # Decode the image
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                self.predictor.set_image(image)

                # Perform segmentation
                masks, _, _ = self.predictor.predict(point_coords=None, point_labels=None, box=None)
                results.append(masks)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append(None)
        
        return pd.Series(results)

# Metadata for the model
metadata = {
    "model_name": "SAM (Segment Anything Model)",
    "checkpoint": "sam_vit_h_4b8939.pth",
    "description": "Segment Anything model for image segmentation.",
    "source": "Meta AI",
    "type": "image-segmentation"
}

# Set the MLflow experiment name
mlflow.set_experiment("/Users/luke.gardner@databricks.com/Computer Vision Models")

# Prepare sample input
sample_input = images_df.select("content").limit(1).toPandas()

# Test the SegmentAnythingModel locally
sample_model = SegmentAnythingModel()
sample_model.load_context(None)
sample_output = sample_model.predict(None, sample_input)

# Infer the signature
signature = infer_signature(sample_input, sample_output)

# Start an MLflow run
with mlflow.start_run(run_name="Meta - Segment Anything Model") as run:
    # Log metadata as tags
    for key, value in metadata.items():
        mlflow.set_tag(key, value)

    # Log the model
    mlflow.pyfunc.log_model(
        artifact_path="segment_anything_model",
        python_model=SegmentAnythingModel(),
        registered_model_name="users.luke_gardner.segment_anything_model",
        signature=signature
    )
    
    # Measure runtime performance
    start_time = time.time()
    _ = sample_model.predict(None, sample_input)
    end_time = time.time()
    
    # Log parameters and metrics
    mlflow.log_param("num_inputs", len(sample_input))
    mlflow.log_metric("segmentation_runtime", (end_time - start_time) / len(sample_input))
    mlflow.log_metric("num_successful_masks", sum(1 for mask in sample_output if mask is not None))
    mlflow.log_metric("num_failed_masks", sum(1 for mask in sample_output if mask is None))


# COMMAND ----------


