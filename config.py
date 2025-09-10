# config.py

# Dataset settings
DATASET_SIZE = "50m"  # Options: "50m", "500m", "5b"
DATA_DIR = f"final_project/project_data/YambdaData/{DATASET_SIZE}/"
OUTPUT_DIR = "final_project/processed_data/"

# Filenames
INTERACTIONS_FILE = "interactions.parquet"
TRAIN_FILE = "train.parquet"
VAL_FILE = "val.parquet"
TEST_FILE = "test.parquet"
GRAPH_FILE = "graph.pt"
EMBEDDINGS_FILE = "embeddings.parquet"
ALBUM_MAPPING_FILE = "album_mapping.parquet"
ARTIST_MAPPING_FILE = "artist_mapping.parquet"

# Listen count scaling factor
LISTEN_COUNT_FACTOR = 0.01
