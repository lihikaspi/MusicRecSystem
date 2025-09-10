import os
import pandas as pd
import numpy as np

DATA_DIR = "final_project/project_data/YambdaData50m/"
OUTPUT_DIR = "final_project/processed_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading parquet files...")
likes = pd.read_parquet(os.path.join(DATA_DIR, "likes.parquet"))
listens = pd.read_parquet(os.path.join(DATA_DIR, "listens.parquet"))
dislikes = pd.read_parquet(os.path.join(DATA_DIR, "dislikes.parquet"))
unlikes = pd.read_parquet(os.path.join(DATA_DIR, "unlikes.parquet"))
undislikes = pd.read_parquet(os.path.join(DATA_DIR, "undislikes.parquet"))
embeddings = pd.read_parquet(os.path.join(DATA_DIR, "embeddings.parquet")).to_numpy()

