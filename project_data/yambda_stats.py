import os
import pandas as pd

# ----------------------------
# Dataset settings
# ----------------------------
dataset_size = "50m"  # choose "50m", "500m", "5b"
base_folder = os.path.join(os.getcwd(), f"YambdaData{dataset_size}")

# ----------------------------
# Interaction types
# ----------------------------
interactions = ["likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"]

# Columns
user_col = "uid"
item_col = "item_id"

# Sets to track all unique users and songs
all_users = set()
all_items = set()

# List to collect stats per interaction
stats_list = []

# ----------------------------
# Explore each interaction type
# ----------------------------
for interaction in interactions:
    file_path = os.path.join(base_folder, f"{interaction}.parquet")
    if not os.path.exists(file_path):
        print(f"{interaction} not found, skipping...")
        continue

    print(f"\n--- {interaction} ---")
    df = pd.read_parquet(file_path)

    # Unique counts
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    n_rows = len(df)

    # Update total sets
    all_users.update(df[user_col].unique())
    all_items.update(df[item_col].unique())

    # Distribution per user
    user_counts = df.groupby(user_col).size()
    user_stats = user_counts.describe()

    # Distribution per item
    item_counts = df.groupby(item_col).size()
    item_stats = item_counts.describe()

    # Save stats into a dictionary
    stats = {
        "interaction": interaction,
        "total_interactions": n_rows,
        "unique_users": n_users,
        "unique_items": n_items,
        "user_mean": user_stats["mean"],
        "user_std": user_stats["std"],
        "user_min": user_stats["min"],
        "user_25%": user_stats["25%"],
        "user_50%": user_stats["50%"],
        "user_75%": user_stats["75%"],
        "user_max": user_stats["max"],
        "item_mean": item_stats["mean"],
        "item_std": item_stats["std"],
        "item_min": item_stats["min"],
        "item_25%": item_stats["25%"],
        "item_50%": item_stats["50%"],
        "item_75%": item_stats["75%"],
        "item_max": item_stats["max"]
    }
    stats_list.append(stats)

# ----------------------------
# Create DataFrame and save to CSV
# ----------------------------
stats_df = pd.DataFrame(stats_list)

# Add total unique users and items across all interactions
stats_df.loc[len(stats_df)] = {
    "interaction": "total_all_interactions",
    "total_interactions": sum(stats_df["total_interactions"]),
    "unique_users": len(all_users),
    "unique_items": len(all_items),
    "user_mean": "",
    "user_std": "",
    "user_min": "",
    "user_25%": "",
    "user_50%": "",
    "user_75%": "",
    "user_max": "",
    "item_mean": "",
    "item_std": "",
    "item_min": "",
    "item_25%": "",
    "item_50%": "",
    "item_75%": "",
    "item_max": ""
}

# Save CSV
csv_path = os.path.join(base_folder, f"YambdaStats_{dataset_size}.csv")
stats_df.to_csv(csv_path, index=False)
print(f"\n✅ Stats saved to CSV: {csv_path}")
