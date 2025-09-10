# Yambda Dataset Downloader

This folder contains a Python utility for downloading and saving the [Yandex Yambda dataset](https://huggingface.co/datasets/yandex/yambda) in Parquet format.  
The scripts allow you to download the dataset, inspect the files, and compute basic statistics. You can select the dataset size (`50m`, `500m`, or `5b`) and save all interaction and metadata files locally.

---

## Files Structure

- `project_data/` : folder containing the download and exploration scripts
  - `Yambda_download.py` : script that downloads the Yambda dataset and saves it as Parquet files.
  - `yambda_inspect.py` : script that saves a CSV file containing the column names and the first row of each Parquet file.
  - `yambda_stats.py` : script that saves a CSV file containing basic statistics of each interactions file.
  - `YambdaData50m/` : folder containing the downloaded Parquet files of the 50m dataset and exploration CSV files.  
    *NOTE:* the Parquet files were not uploaded to this Git repository.
    - `listens.parquet` : user → track interactions where a user listened to a track (without explicit feedback).
    - `likes.parquet` : user → track interactions where a user explicitly "liked" a track.
    - `dislikes.parquet` : user → track interactions where a user explicitly "disliked" a track.
    - `unlikes.parquet` : records of users removing a previous "like" from a track.
    - `undislikes.parquet` : records of users removing a previous "dislike" from a track.
    - `multi_event.parquet` : a unified table including multiple types of interactions (`listen`, `like`, `unlike`, `dislike`, `undislike`). This file may contain `null` values and is not cleaned.
    - `embeddings.parquet` : pre-computed audio embeddings for the tracks.
    - `album_mapping.parquet` : mapping between album IDs and the items (tracks) they contain.
    - `artist_mapping.parquet` : mapping between artist IDs and the items (tracks) they created.
    - `yambda_columns.csv` : CSV file saved by `yambda_inspect.py` containing column names and first rows.
    - `YambdaStats_50m.csv` : CSV file saved by `yambda_stats.py` containing basic statistics of each interactions file.

---

## How It Works

1. The scripts use Hugging Face's `datasets` library to download Yambda.
2. The user specifies:
   - Dataset **type** (`flat` or `sequential`)  
   - Dataset **size** (`50m`, `500m`, or `5b`)  
3. All interaction files and metadata are saved as `.parquet` in the chosen folder (e.g., `YambdaData50m/`).

---

## Usage

### 1. Download Yambda Dataset
Downloads the dataset and saves all Parquet files.

```bash
# Install dependencies
pip install datasets pyarrow

# Run the download script
python Yambda_download.py
```

By default, the script downloads the **flat** representation of the Yambda dataset. You can change the dataset size and type by editing the script variables:

```python
dataset_size = "50m"  # options: "50m", "500m", "5b"
dataset_type = "flat"  # options: "flat", "sequential"
```

---

### 2. Inspect Parquet Files
Generates a CSV containing the column names and the first row of each downloaded Parquet file.

```bash
python yambda_inspect.py
```

Output:
- `yambda_columns.csv` in the `YambdaData50m/` folder.

---

### 3. Compute Basic Statistics
Generates a CSV with basic statistics for each interactions file, such as:
- Number of unique users
- Number of unique tracks
- Total number of interactions

```bash
python yambda_stats.py
```

Output:
- `YambdaStats_50m.csv` in the `YambdaData50m/` folder.

---

## Notes

- The **multi_event.parquet** file may contain missing values for some columns. The separate interaction files (e.g., `likes`, `listens`) are already cleaned.
- Dataset sizes:
  - `50m`: lightweight, easier for experiments
  - `500m`: medium scale
  - `5b`: full dataset, very large and resource-intensive
- All CSV outputs are generated inside the same folder as the Parquet files.
