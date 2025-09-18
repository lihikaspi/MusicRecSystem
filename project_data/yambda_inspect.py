import pandas as pd
import os
import json
import numpy as np
from config import (
    DOWNLOAD_FULL_DATASET, RAW_LISTENS_FILE, RAW_LIKES_FILE, RAW_DISLIKES_FILE,
    RAW_UNLIKES_FILE, RAW_UNDISLIKES_FILE, RAW_MULTI_EVENT_FILE, EMBEDDINGS_FILE,
    ALBUM_MAPPING_FILE, ARTIST_MAPPING_FILE, DATA_COLS_FILE, DATA_STATS_FILE
)

# Decide which files to include
if DOWNLOAD_FULL_DATASET:
    files_to_process = [
        RAW_LISTENS_FILE, RAW_LIKES_FILE, RAW_DISLIKES_FILE,
        RAW_UNLIKES_FILE, RAW_UNDISLIKES_FILE, RAW_MULTI_EVENT_FILE,
        EMBEDDINGS_FILE, ALBUM_MAPPING_FILE, ARTIST_MAPPING_FILE
    ]
else:
    # Only essential files
    files_to_process = [
        RAW_MULTI_EVENT_FILE, EMBEDDINGS_FILE, ALBUM_MAPPING_FILE, ARTIST_MAPPING_FILE
    ]

# Helper function to convert JSON-serializable
def make_json_serializable(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            new_d[k] = v.tolist()  # convert numpy array to list
        else:
            new_d[k] = v
    return new_d

# Prepare summary
summary_list = []

for file_path in files_to_process:
    if os.path.exists(file_path):
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}")
        df = pd.read_parquet(file_path)

        # Column names
        columns = df.columns.tolist()

        # First row
        first_row = df.head(1).to_dict(orient="records")[0]
        first_row = make_json_serializable(first_row)

        # Convert to JSON string
        first_row_json = json.dumps(first_row, ensure_ascii=False)

        # Add to summary
        summary_list.append({
            "file": file_name,
            "columns": ", ".join(columns),
            "first_row": first_row_json,
            "num_rows": len(df)  # optional extra stat
        })
    else:
        print(f"File not found, skipping: {file_path}")

# Save summary CSVs
summary_df = pd.DataFrame(summary_list)
summary_df.to_csv(DATA_COLS_FILE, index=False)
print(f"Column summary saved to {DATA_COLS_FILE}")

