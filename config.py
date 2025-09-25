import numpy as np
import torch

# -------
# DATASET
# -------

# Dataset settings
DATASET_SIZE = "50m"  # Options: "50m", "500m", "5b"
DATASET_TYPE = "flat" # options: "flat", "sequential"

# Flag to download full dataset or only essential files
# True = download everything, False = only multi_event + embeddings + mappings
DOWNLOAD_FULL_DATASET = False

# --------
# PIPELINE
# --------

# Ordered pipeline (name → file)
STAGE_FILES = [
    ("download", "download_data.py"),
    ("gnn_prep", "run_GNN_prep.py"),
    ("gnn_train", "run_GNN_train.py"),
    ("ann_search", "run_ANN_search.py"),
]

# ---------------------
# PATHS AND DIRECTORIES
# ---------------------

# Save directories
DATA_DIR = f"project_data/YambdaData{DATASET_SIZE}/"
PROCESSED_DIR = "processed_data"
GNN_MODEL = "GNN_model"
RECS_DIR = "recs"

# Data additional CSVs
DATA_COLS_FILE = f"{DATA_DIR}/yambda_columns.csv"
DATA_STATS_FILE = f"{DATA_DIR}/YambdaStats_{DATASET_SIZE}.csv"

# Raw interaction files
RAW_LISTENS_FILE = f"{DATA_DIR}/listens.parquet"
RAW_LIKES_FILE = f"{DATA_DIR}/likes.parquet"
RAW_DISLIKES_FILE = f"{DATA_DIR}/dislikes.parquet"
RAW_UNLIKES_FILE = f"{DATA_DIR}/unlikes.parquet"
RAW_UNDISLIKES_FILE = f"{DATA_DIR}/undislikes.parquet"
RAW_MULTI_EVENT_FILE = f"{DATA_DIR}/multi_event.parquet"

# Raw embeddings
AUDIO_EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings.parquet"

# Mappings
ALBUM_MAPPING_FILE = f"{DATA_DIR}/album_mapping.parquet"
ARTIST_MAPPING_FILE = f"{DATA_DIR}/artist_mapping.parquet"

# Processed files
INTERACTIONS_FILE = f"{PROCESSED_DIR}/interactions.parquet"
TRAIN_SET_FILE = f"{PROCESSED_DIR}/train.parquet"
VAL_SET_FILE = f"{PROCESSED_DIR}/val.parquet"
TEST_SET_FILE = f"{PROCESSED_DIR}/test.parquet"
TRAIN_EDGES_FILE = f"{PROCESSED_DIR}/train_edges.parquet"
TRAIN_GRAPH_FILE = f"{PROCESSED_DIR}/graph.pt"

# List of single-event files
RAW_DATA_FILES = [
    RAW_LISTENS_FILE,
    RAW_LIKES_FILE,
    RAW_DISLIKES_FILE,
    RAW_UNLIKES_FILE,
    RAW_UNDISLIKES_FILE
]

# GNN save paths
TRAINED_GNN = f"{GNN_MODEL}/best_model.pth"
USER_EMBEDDINGS_GNN = f"{GNN_MODEL}/user_embeddings.pt"
SONG_EMBEDDINGS_GNN = f"{GNN_MODEL}/song_embeddings.pt"


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

# Train/val/test split ratios
SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

SPLIT_PATHS = {
    "train": TRAIN_SET_FILE,
    "val": VAL_SET_FILE,
    "test": TEST_SET_FILE
}

# -------------------
# GNN HYPERPARAMETERS
# -------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# GNN class parameters
EMBED_DIM = 128               # node embedding dimension
NUM_LAYERS = 3                # number of LightGCN layers
INIT_STD = 0.1                # initial std for embedding initialization
LAMBDA_ALIGN = 0.1            # weight for content/audio alignment loss
FREEZE_AUDIO = True           # keep audio embeddings fixed


# Edge weight MLP
EDGE_MLP_HIDDEN_DIM = 32  # hidden dimension for edge weight MLP
EDGE_MLP_INPUT_DIM = 4    # edge_type, edge_count, edge_avg_played_ratio, edge_weight_init

# training
LR = 0.001                    # learning rate
NUM_EPOCHS = 50               # max epochs
BATCH_SIZE = 1024             # BPR training batch size
WEIGHT_DECAY = 1e-4           # optional L2 regularization on embeddings

# GNN eval parameters
K_HIT = 50                     # top-K for evaluation matrics

# Event label mapping for evaluation
EVAL_EVENT_MAP = {
    "like": 2,        # strong positive
    "listen": 1,      # weak positive
    "unlike": 0,      # neutral (revoked like)
    "dislike": -1,    # explicit negative
    "undislike": 0    # neutral (revoked dislike)
}

# -------------------
# ANN HYPERPARAMETERS
# -------------------

# ANN top-k results
TOP_K = 10








