# Dataset settings
DATASET_SIZE = "50m"  # Options: "50m", "500m", "5b"
DATASET_TYPE = "flat" # options: "flat", "sequential"

# save directories
DATA_DIR = f"final_project/project_data/YambdaData{DATASET_SIZE}/"
PROCESSED_DIR = "final_project/processed_data/"

# Mapping edge type names to integer IDs
EDGE_TYPE_MAPPING = {
    "listen": 0,
    "like": 1,
    "dislike": 2,
    "unlike": 3,
    "undislike": 4
}
