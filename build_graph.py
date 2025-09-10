import os
import torch
import pandas as pd
from torch_geometric.data import HeteroData

OUTPUT_DIR = "final_project/processed_data/"

# Load train interactions parquet
train = pd.read_parquet(os.path.join(OUTPUT_DIR, "train.parquet"))

# Load audio embeddings parquet (if saved as numpy array, use np.load)
embeddings = pd.read_parquet(os.path.join(OUTPUT_DIR, "embeddings.parquet")).to_numpy()

# ----------------------------
# Build bipartite user–item graph
# ----------------------------
data = HeteroData()

# Node counts
num_users = train["user_idx"].nunique()
num_items = train["item_idx"].nunique()

data["user"].num_nodes = num_users
data["item"].x = torch.tensor(embeddings, dtype=torch.float)

# Edges (user -> item)
edge_index = torch.tensor([
    train["user_idx"].values,
    train["item_idx"].values
], dtype=torch.long)

data["user", "interacts", "item"].edge_index = edge_index
data["user", "interacts", "item"].edge_weight = torch.tensor(
    train["weight"].values, dtype=torch.float
)

# ----------------------------
# Save graph
# ----------------------------
torch.save(data, os.path.join(OUTPUT_DIR, "graph.pt"))

print(f"✅ Graph built: {num_users} users, {num_items} items, {edge_index.shape[1]} edges")
