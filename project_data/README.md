# Yambda Dataset Downloader

This folder contains a Python utility for downloading and saving the [Yandex Yambda dataset](https://huggingface.co/datasets/yandex/yambda) in Parquet format.  
The script allows you to select the dataset size (`50m`, `500m`, or `5b`) and saves all interaction and metadata files locally.

---

## Files Structure

- `project_data/` : folder containing the download and exploration scripts
  - `Yambda_download.py` : script that downloads the Yambda dataset and saves it as parquet files
  - `yambda_inspect.py` : script that saves a csv file containing the column names and the first row of each parquet file 
  - `yambda_stats.py` : script that saves a csv file containing basic statistics of each interactions file
  - `YambdaData50m/` : folder containing the downloaded parquet files and exploration csv files
    - `listens.parquet` : user → track interactions where a user listened to a track (without explicit feedback).
    - `likes.parquet` : user → track interactions where a user explicitly "liked" a track.
    - `dislikes.parquet` : user → track interactions where a user explicitly "disliked" a track.
    - `unlikes,parquet` : records of users removing a previous "like" from a track.
    - `undislikes.parquet` : records of users removing a previous "dislike" from a track.
    - `multi_event.parquet` : a unified table that includes multiple types of interactions in a single file (`listen`, `like`, `unlike`, `dislike`, `undislike`). This data is not cleaned and contains `null` values.
    - `embeddings.parquet` : pre-computed audio embeddings for the tracks.
    - `album_mapping.parquet` : mapping between album IDs and the items (tracks) they contain.
    - `artist_mapping.parquet` : mapping between artist IDs and the items (tracks) they created.
    - `yambda_columns.csv` : the csv file saved from the `yambda_inspect.py` file
    - `YambdaStast_50m.csv` : the csv file saved from the `yambda_stats.py` file

---

## How It Works

1. The script uses Hugging Face's `datasets` library to download Yambda.
2. The user specifies:
   - The dataset **type** (`flat` or `sequential`)  
   - The dataset **size** (`50m`, `500m`, or `5b`)  
3. All interaction files and metadata are saved as `.parquet` in the chosen folder (e.g., `YambdaData500m/`).

---

## 🚀 Usage

```bash
# Install dependencies
pip install datasets pyarrow

# Run the script
python download_yambda.py
