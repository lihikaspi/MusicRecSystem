import os
import shutil
from datasets import Dataset, load_dataset
from typing import Literal
from config import DATASET_SIZE, DATASET_TYPE, DATA_DIR

# ----------------------------
# Wrapper Class for Yambda
# ----------------------------
class YambdaDataset:
    INTERACTIONS = frozenset([
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ])

    def __init__(self, dataset_type: Literal["flat", "sequential"] = "flat",
                 dataset_size: Literal["50m", "500m", "5b"] = "50m"):
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

    def interaction(self, event_type: str) -> Dataset:
        assert event_type in YambdaDataset.INTERACTIONS
        return self._download(f"{self.dataset_type}/{self.dataset_size}", event_type)

    def audio_embeddings(self) -> Dataset:
        return self._download("", "embeddings")

    def album_item_mapping(self) -> Dataset:
        return self._download("", "album_item_mapping")

    def artist_item_mapping(self) -> Dataset:
        return self._download("", "artist_item_mapping")

    def _download(self, data_dir: str, file: str) -> Dataset:
        data = load_dataset("yandex/yambda", data_dir=data_dir, data_files=f"{file}.parquet")
        return data["train"]

# ----------------------------
# Delete old folder if it exists
# ----------------------------
if os.path.exists(DATA_DIR):
    print(f"Deleting old folder: {DATA_DIR}")
    shutil.rmtree(DATA_DIR)

os.makedirs(DATA_DIR, exist_ok=True)
print(f"Created new folder: {DATA_DIR}")

# ----------------------------
# Download datasets
# ----------------------------
dataset = YambdaDataset(dataset_type=DATASET_TYPE, dataset_size=DATASET_SIZE)

# Interactions
likes = dataset.interaction("likes")
listens = dataset.interaction("listens")
multi_event = dataset.interaction("multi_event")
dislikes = dataset.interaction("dislikes")
unlikes = dataset.interaction("unlikes")
undislikes = dataset.interaction("undislikes")

# Metadata + embeddings
embeddings = dataset.audio_embeddings()
album_mapping = dataset.album_item_mapping()
artist_mapping = dataset.artist_item_mapping()

# ----------------------------
# Save all datasets to disk
# ----------------------------
datasets_to_save = {
    "likes": likes,
    "listens": listens,
    "multi_event": multi_event,
    "dislikes": dislikes,
    "unlikes": unlikes,
    "undislikes": undislikes,
    "embeddings": embeddings,
    "album_mapping": album_mapping,
    "artist_mapping": artist_mapping
}

for name, ds in datasets_to_save.items():
    file_path = os.path.join(DATA_DIR, f"{name}.parquet")
    print(f"Saving {name} to {file_path} ...")
    ds.to_parquet(file_path)

print(f"✅ All datasets saved to folder: {DATA_DIR}")