# Yambda Dataset Downloader

This folder contains a Python utility for downloading and saving the [Yandex Yambda dataset](https://huggingface.co/datasets/yandex/yambda) in Parquet format.  
The scripts allow you to download the dataset, inspect the files, and compute basic statistics. You can select the dataset size (`50m`, `500m`, or `5b`) and save all interaction and metadata files locally.

---

## Files Structure

- `project_data/` : folder containing the download and exploration scripts
  - `Yambda_download.py` : script that downloads the Yambda dataset and saves it as Parquet files.
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

## How It Works

1. The scripts use Hugging Face's `datasets` library to download Yambda.
2. The user specifies:
   - Dataset **type** (`flat` or `sequential`)  
   - Dataset **size** (`50m`, `500m`, or `5b`)  
3. All interaction files and metadata are saved as `.parquet` in the chosen folder (e.g., `YambdaData50m/`).

---

## Usage

```bash
# Install dependencies
pip install datasets pyarrow

# Run the download script
python Yambda_download.py
