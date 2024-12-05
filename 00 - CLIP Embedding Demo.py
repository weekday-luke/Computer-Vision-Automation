# Databricks notebook source
# MAGIC %md
# MAGIC #CLIP Embedding Demo
# MAGIC Create embeddings for text prompts and images to be used in a single vector search index.
# MAGIC
# MAGIC ###OpenAI CLIP
# MAGIC HuggingFace - https://huggingface.co/openai/clip-vit-base-patch32
# MAGIC
# MAGIC Medium Article - https://towardsdatascience.com/quick-fire-guide-to-multi-modal-ml-with-openais-clip-2dad7e398ac0
# MAGIC
# MAGIC YouTube Tutorial - https://www.youtube.com/watch?v=989aKUVBfbk

# COMMAND ----------

!pip install torch transformers datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ###Setup
# MAGIC Load Images and Model

# COMMAND ----------

#load images
from datasets import load_dataset

imagenette = load_dataset(
  'frgfm/imagenette',
  'full_size',
  split='train',
  ignore_verifications=False
)

imagenette[0]["image"]

# COMMAND ----------

#load model
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import torch

device = "cuda" if torch.cuda.is_available() else \
          ("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "openai/clip-vit-base-patch32"


model = CLIPModel.from_pretrained(model_id).to(device)
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create Embeddings
# MAGIC The next steps include:
# MAGIC 1. Embedding a prompt (plain text) using the open source model
# MAGIC 2. Embed an image using the same model
# MAGIC 3. Embed several images in a batch process

# COMMAND ----------

# MAGIC %md
# MAGIC ###Text Embeddings

# COMMAND ----------

prompt = "a dog in the snow"

#tokenize the prompt

inputs = tokenizer(prompt, return_tensors="pt")
inputs

# COMMAND ----------

text_emb = model.get_text_features(**inputs)
text_emb.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ###Image Embeddings

# COMMAND ----------

#resizing the image with proceessor
#expected shape is torch.Size([1, 3, 224, 224])
image = processor(text=None,
                  images = imagenette[0]['image'],
                  return_tensors="pt")['pixel_values'].to(device)
image.shape

# COMMAND ----------

import matplotlib.pyplot as plt

#resize the image and show it
#the pixels have been modified which is why the image looks distorted
plt.imshow(image.squeeze(0).T)

# COMMAND ----------

#after this line you will have a 512 dimension embedding vector
image_emb = model.get_image_features(image)
image_emb.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ###Batch Processing
# MAGIC We'll embed about 100 images to compare to our initial prompt.

# COMMAND ----------

#get a subset of 100 images for this experiment
import numpy as np

np.random.seed(0)
sample_idx = np.random.randint(0, len(imagenette)+1,100).tolist()
images = [imagenette[i]['image'] for i in sample_idx]
len(images)

# COMMAND ----------

# DBTITLE 1,Image Batch Embedding Generator
from tqdm.auto import tqdm

batch_size = 16
image_arr = None

for i in tqdm(range(0, len(images), batch_size)):
  #select batch of images
  batch = images[i:i+batch_size]
  #process and resize images
  batch = processor(text=None,
                  images = batch,
                  return_tensors="pt",
                  padding=True,
                  is_train=False)['pixel_values'].to(device)
  
  #get image embeddings
  batch_emb = model.get_image_features(pixel_values=batch)
  #convert to numpy array
  batch_emb = batch_emb.squeeze(0)
  batch_emb = batch_emb.cpu().detach().numpy()
  #add to larger array of all image embeddings
  if image_arr is None:
    image_arr = batch_emb
  else:
    image_arr = np.concatenate((image_arr, batch_emb), axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ###Calculate Scores

# COMMAND ----------

#normalize the values in the image array
image_arr = image_arr.T / np.linalg.norm(image_arr, axis=1)

#get the text embedding from ealier
text_emb = text_emb.cpu().detach().numpy()

#calculate the scores and get the top 5
scores = np.dot(text_emb, image_arr)

top_k = 5
idx = np.argsort(-scores[0])[:top_k]

idx

# COMMAND ----------

# MAGIC %md
# MAGIC ###Show Results

# COMMAND ----------

# show the images and their scores
for i in idx:
  print(scores[0][i])
  plt.imshow(images[i])
  plt.show()
