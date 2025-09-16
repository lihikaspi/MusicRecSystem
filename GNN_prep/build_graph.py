import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from MusicRecSystem.config import PROCESSED_DIR, EDGE_TYPE_MAPPING

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ----------------------------
    # Load data
    # ----------------------------
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    embeddings = pd.read_parquet(os.path.join(PROCESSED_DIR, "embeddings.parquet")).to_numpy()

    # Ensure IDs are 0-indexed and continuous
    train["user_idx"] = pd.Categorical(train["user_id"]).codes
    train["item_idx"] = pd.Categorical(train["item_id"]).codes

    num_users = train["user_idx"].nunique()
    num_items = train["item_idx"].nunique()

    # ----------------------------
    # Build HeteroData
    # ----------------------------
    data = HeteroData()

    # Users: learnable embeddings
    data["user"].num_nodes = num_users

    # Items: song embeddings (audio + optional metadata)
    data["item"].x = torch.tensor(embeddings, dtype=torch.float)

    # ----------------------------
    # Prepare edges
    # ----------------------------
    # Convert edge types to integers
    train["edge_type"] = train["interaction_type"].map(EDGE_TYPE_MAPPING)

    # Edge index (user -> item)
    edge_index = torch.tensor([
        train["user_idx"].values,
        train["item_idx"].values
    ], dtype=torch.long)

    # Edge type tensor
    edge_type = torch.tensor(train["edge_type"].values, dtype=torch.long)

    # Edge weight tensor
    # Example: listen count (or 1/-1 for like/dislike)
    edge_weight = torch.tensor(train["weight"].values, dtype=torch.float)

    # Optional: normalize or include timestamp
    if "timestamp" in train.columns:
        # Convert UNIX timestamp to float (e.g., days since min timestamp)
        min_ts = train["timestamp"].min()
        edge_recency = ((train["timestamp"] - min_ts) / (train["timestamp"].max() - min_ts)).astype(np.float32)
        edge_recency = torch.tensor(edge_recency.values, dtype=torch.float)
        data["user", "interacts", "item"].edge_recency = edge_recency

    # Assign to HeteroData
    data["user", "interacts", "item"].edge_index = edge_index
    data["user", "interacts", "item"].edge_type = edge_type
    data["user", "interacts", "item"].edge_weight = edge_weight

    # ----------------------------
    # Save graph
    # ----------------------------
    torch.save(data, os.path.join(PROCESSED_DIR, "graph.pt"))
    print(f"Graph built: {num_users} users, {num_items} items, {edge_index.shape[1]} edges")
    print(f"Edge types: {train['interaction_type'].unique()}")

if __name__ == '__main__':
    main()
