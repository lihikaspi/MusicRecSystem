import pandas as pd
import os
import json
import numpy as np
from config import config

# Decide which files to include
if config.dataset.download_full:
    files_to_process = config.paths.raw_data_files
    files_to_process.append(config.paths.raw_multi_event_file)
    files_to_process.append(config.paths.audio_embeddings_file)
    files_to_process.append(config.paths.album_mapping_file)
    files_to_process.append(config.paths.artist_mapping_file)
else:
    # Only essential files
    files_to_process = [
        config.paths.raw_multi_event_file, config.paths.audio_embeddings_file,
        config.path.album_mapping_file, config.paths.artist_mapping_file
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
summary_df.to_csv(config.paths.data_cols_file, index=False)
print(f"Column summary saved to {config.paths.data_cols_file}")

