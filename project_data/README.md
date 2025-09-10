# Yambda Dataset Downloader

This repository contains Python utilities for downloading, inspecting, and analyzing the [Yandex Yambda dataset](https://huggingface.co/datasets/yandex/yambda) in Parquet format.  
The dataset is downloaded using the script in the repo root, and the scripts in `project_data/` allow you to inspect the files and compute basic statistics.  
The dataset size, type, and save directory are configured via a Python `config.py` file in the repo root.

---

## Files Structure

- `config.py` : global configuration file for dataset parameters and save directory.  
- `yambda_download.py` : script that downloads the Yambda dataset and saves it as Parquet files.  
- `project_data/` : folder containing exploration scripts and downloaded files  
  - `yambda_inspect.py` : script that saves a CSV file containing the column names and the first row of each Parquet file.  
  - `yambda_stats.py` : script that saves a CSV file containing basic statistics of each interactions file.  
  - `YambdaData50m/` : folder containing the downloaded Parquet files and exploration CSV files.  
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

## Configuration

A `config.py` file in the repo root sets the dataset parameters and save directory. Example:

```python
# config.py
DATASET_SIZE = "50m"        # Options: "50m", "500m", "5b"
DATASET_TYPE = "flat"       # Options: "flat", "sequential"
DATA_DIR = f"project_data/YambdaData{DATASET_SIZE}/"
```

All scripts (`yambda_download.py`, `yambda_inspect.py`, `yambda_stats.py`) import this file and use its global variables.  
You can edit this file to change the dataset parameters or save directory.

---

## How It Works

1. The download script uses Hugging Face's `datasets` library to fetch Yambda.  
2. Parameters like dataset type, size, and save directory are read from `config.py`.  
3. All interaction files and metadata are saved as `.parquet` in the directory specified in the config.  
4. Exploration scripts in `project_data/` can inspect and generate statistics for the downloaded files.  

---

## Usage

### 1. Download Yambda Dataset  
Run from the repo root to download the dataset and save all Parquet files.  

```bash
python yambda_download.py
```

---

### 2. Inspect Parquet Files  
Run from inside `project_data/`. Generates a CSV containing the column names and the first row of each downloaded Parquet file.  

```bash
python yambda_inspect.py
```

Output:  
- `yambda_columns.csv` in the save directory specified in `config.py`.

---

### 3. Compute Basic Statistics  
Run from inside `project_data/`. Generates a CSV with basic statistics for each interactions file, such as:  
- Number of unique users  
- Number of unique tracks  
- Total number of interactions  

```bash
python yambda_stats.py
```

Output:  
- `YambdaStats_50m.csv` in the save directory specified in `config.py`.  

---

## Notes

- The **multi_event.parquet** file may contain missing values for some columns. The separate interaction files (e.g., `likes`, `listens`) are already cleaned.  
- Dataset sizes:  
  - `50m`: lightweight, easier for experiments  
  - `500m`: medium scale  
  - `5b`: full dataset, very large and resource-intensive  
- All CSV outputs are generated in the save directory specified in `config.py`.  
