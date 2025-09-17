# Dataset settings
DATASET_SIZE = "50m"  # Options: "50m", "500m", "5b"
DATASET_TYPE = "flat" # options: "flat", "sequential"

# --------
# PIPELINE
# --------

# Ordered pipeline (name → file)
STAGE_FILES = [
    ("download", "download_yambda.py"),
    ("gnn_prep", "run_GNN_prep.py"),
    ("train_gnn", "train_GNN.py"),
    ("ann_search", "run_ANN_search.py"),
]

# ---------------------
# PATHS AND DIRECTORIES
# ---------------------

# Save directories
DATA_DIR = f"project_data/YambdaData{DATASET_SIZE}/"
PROCESSED_DIR = "processed_data"
RECS_DIR = "recs/"

# Raw interaction files
RAW_LISTENS_FILE = f"{DATA_DIR}/listens.parquet"
RAW_LIKES_FILE = f"{DATA_DIR}/likes.parquet"
RAW_DISLIKES_FILE = f"{DATA_DIR}/dislikes.parquet"
RAW_UNLIKES_FILE = f"{DATA_DIR}/unlikes.parquet"
RAW_UNDISLIKES_FILE = f"{DATA_DIR}/undislikes.parquet"
RAW_MULTI_EVENT_FILE = f"{DATA_DIR}/multi_event.parquet"

# Raw embeddings
EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings.parquet"

# Mappings
ALBUM_MAPPING_FILE = f"{DATA_DIR}/album_mapping.parquet"
ARTIST_MAPPING_FILE = f"{DATA_DIR}/artist_mapping.parquet"

# Processed files
INTERACTIONS_FILE = f"{PROCESSED_DIR}/interactions.parquet"
TRAIN_FILE = f"{PROCESSED_DIR}/train.parquet"
VAL_FILE = f"{PROCESSED_DIR}/val.parquet"
TEST_FILE = f"{PROCESSED_DIR}/test.parquet"
EDGES_FILE = f"{PROCESSED_DIR}/train_edges.parquet"
GRAPH_FILE = f"{PROCESSED_DIR}/graph.pt"

# List of single-event files
RAW_DATA_FILES = [
    RAW_LISTENS_FILE,
    RAW_LIKES_FILE,
    RAW_DISLIKES_FILE,
    RAW_UNLIKES_FILE,
    RAW_UNDISLIKES_FILE
]

# ------------------
# DATA PREPROCESSING
# ------------------

# Mapping edge type names to integer IDs
EDGE_TYPE_MAPPING = {
    "listen": 1,
    "like": 2,
    "dislike": 3,
    "unlike": 4,
    "undislike": 5
}

# user interaction threshold
INTERACTION_THRESHOLD = 5

# event type Weights
WEIGHTS = {
    "listens": 1.0,
    "likes": 3.0,
    "dislikes": -3.0,
    "unlikes": -1.0,
    "undislikes": -1.0
}

# list of event types and their opposites
OPPOSITE_EVENT_TYPES = [
    {"table": "listens", "opposite": None},
    {"table": "likes", "opposite": "unlikes"},
    {"table": "unlikes", "opposite": None},
    {"table": "dislikes", "opposite": "undislikes"},
    {"table": "undislikes", "opposite": None},
]

# Train/val/test split ratios
SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.0,
    "test": 0.2
}

SPLIT_PATHS = {
    "train": TRAIN_FILE,
    "val": VAL_FILE,
    "test": TEST_FILE
}

# -------------------
# GNN HYPERPARAMETERS
# -------------------



# -------------------
# ANN HYPERPARAMETERS
# -------------------

# ANN top-k results
TOP_K = 10








