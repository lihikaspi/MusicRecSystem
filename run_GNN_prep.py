from GNN_prep.data_preprocessing import main as run_data_prep
from GNN_prep.split_data import main as run_split
from GNN_prep.build_graph import main as build_graph
import os
from config import PROCESSED_DIR


def main():
    interactions_dir = os.path.join(PROCESSED_DIR, "interactions.parquet")
    train_dir = os.path.join(PROCESSED_DIR, "train.parquet")
    val_dir = os.path.join(PROCESSED_DIR, "val.parquet")
    test_dir = os.path.join(PROCESSED_DIR, "test.parquet")
    graph_dir = os.path.join(PROCESSED_DIR, "graph.pt")

    # Stage 2 script 1: data prep
    if not os.path.exists(interactions_dir):
        print("interactions parquet not found. Running data preparation script...")
        run_data_prep()
    else:
        print("interactions parquet exists. Skipping data preparation.")

    # Stage 2 script 2: split
    if not os.path.exists(train_dir) or not os.path.exists(val_dir) or not os.path.exists(test_dir):
        print("Train/val/test splits not found. Running split script...")
        run_split()
    else:
        print("Splits exist. Skipping split.")

    # Stage 2 script 3: build graph
    if not os.path.exists(graph_dir):
        print("graph not found. Running graph building script...")
        build_graph()
    else:
        print("graph exists. Skipping graph building.")

    print("all files exists! ready for the GNN training")


if __name__ == "__main__":
    main()