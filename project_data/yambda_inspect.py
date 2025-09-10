import pandas as pd
import os
import json
import numpy as np

# ------------------------------
# Settings
# ------------------------------
parquet_folder = "/home/student/project_data/YambdaData50m"  # <-- folder where your parquet files are
output_csv = "/home/student/project_data/YambdaData50m/yambda_columns.csv"  # <-- output CSV
num_preview_rows = 1  # only the first row


# ------------------------------
# Helper function to convert JSON-serializable
# ------------------------------
def make_json_serializable(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            new_d[k] = v.tolist()  # convert numpy array to list
        else:
            new_d[k] = v
    return new_d


# ------------------------------
# Prepare summary
# ------------------------------
summary_list = []

for file_name in os.listdir(parquet_folder):
    if file_name.endswith(".parquet"):
        print(f"Starting {file_name}")
        file_path = os.path.join(parquet_folder, file_name)
        df = pd.read_parquet(file_path)

        # Column names
        columns = df.columns.tolist()

        # First row
        first_row = df.head(num_preview_rows).to_dict(orient="records")[0]
        first_row = make_json_serializable(first_row)

        # Convert to JSON string
        first_row_json = json.dumps(first_row, ensure_ascii=False)

        # Add to summary
        summary_list.append({
            "file": file_name,
            "columns": ", ".join(columns),
            "first_row": first_row_json
        })

# ------------------------------
# Save summary
# ------------------------------
summary_df = pd.DataFrame(summary_list)
summary_df.to_csv(output_csv, index=False)
print(f"Summary saved to {output_csv}")
