import os
import pandas as pd
from math import ceil

OUTPUT_DIR = "final_project/processed_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load merged interactions parquet
interactions = pd.read_parquet(os.path.join(OUTPUT_DIR, "interactions.parquet"))

# Time-aware per-user split
train, val, test = [], [], []
count_removed_user = 0

for user, df in interactions.groupby("user_idx"):
    df = df.sort_values("timestamp")
    n = len(df)
    if n < 5:
        count_removed_user += 1
        continue  # skip users with very few interactions

    train_end = ceil(0.8 * n)
    val_end = ceil(0.9 * n)

    train.append(df.iloc[:train_end])
    val.append(df.iloc[train_end:val_end])
    test.append(df.iloc[val_end:])

train = pd.concat(train)
val = pd.concat(val)
test = pd.concat(test)

# Save splits as parquet
train.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), index=False)
val.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"), index=False)
test.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"), index=False)

print(f"✅ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"{count_removed_user} users were removed during the splitting process")
