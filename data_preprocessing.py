import os
import pandas as pd
from config import DATA_DIR, PROCESSED_DIR

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Loading parquet files...")
# Load interactions files
likes = pd.read_parquet(os.path.join(DATA_DIR, "likes.parquet"))
listens = pd.read_parquet(os.path.join(DATA_DIR, "listens.parquet"))
dislikes = pd.read_parquet(os.path.join(DATA_DIR, "dislikes.parquet"))
unlikes = pd.read_parquet(os.path.join(DATA_DIR, "unlikes.parquet"))
undislikes = pd.read_parquet(os.path.join(DATA_DIR, "undislikes.parquet"))

# Load mappings
album_mapping = pd.read_parquet(os.path.join(DATA_DIR, "album_mapping.parquet"))
artist_mapping = pd.read_parquet(os.path.join(DATA_DIR, "artist_mapping.parquet"))

# edge weights
print("Assigning event type weights")
# TODO: decide on weights for each event type
likes["weight"] = 2
listens["weight"] = 1
dislikes["weight"] = 0
unlikes["weight"] = 0
undislikes["weight"] = 0

# TODO: give less weight to accidental dislikes (based on the timestamps of the dislike and undislike events)

# Keep only required columns
cols = ["user_id", "item_id", "timestamp", "weight"]
interactions = pd.concat([
    likes[cols],
    listens[cols],
    dislikes[cols],
    unlikes[cols],
    undislikes[cols]
], ignore_index=True)

# calculate listen counts per song for each user
listen_counts = listens.groupby(["user_id", "item_id"]).size().reset_index(name="listen_count")

# Drop missing values
interactions = interactions.dropna(subset=["user_id", "item_id", "timestamp"])

# Drop duplicates (keep latest interaction)
interactions = interactions.sort_values("timestamp").drop_duplicates(
    subset=["user_id", "item_id"], keep="last"
)

# Merge album and artist mapping
interactions = interactions.merge(album_mapping, on="item_id", how="left")
interactions = interactions.merge(artist_mapping, on="item_id", how="left")

# merge listen count
interactions = interactions.merge(listen_counts, on=["user_id","item_id"], how="left")
interactions["listen_count"] = interactions["listen_count"].fillna(0)

# Encode IDs
user2idx = {u: i for i, u in enumerate(interactions["user_id"].unique())}
item2idx = {i: j for j, i in enumerate(interactions["item_id"].unique())}

interactions["user_idx"] = interactions["user_id"].map(user2idx)
interactions["item_idx"] = interactions["item_id"].map(item2idx)

# edge weight with listen count
# TODO: decide on weight formula
"""
options from ChatGPT:
Log-scaled listen counts: log(1 + listen_count) to reduce extreme influence from very active users.
Cap listen counts: set a max (e.g., 20) so super-heavy listeners don’t dominate.
Separate edge feature: keep weight as main edge weight, and store listen_count as a secondary edge feature. This is useful for GNNs that support multiple edge attributes.
"""
interactions["edge_weight"] = interactions["weight"] + 0.01 * interactions["listen_count"]

# Save processed interactions
interactions.to_parquet(os.path.join(PROCESSED_DIR, "interactions.parquet"), index=False)

print(f"✅ Saved {len(interactions)} interactions")
print(f"Users: {len(user2idx)}, Items: {len(item2idx)}")




