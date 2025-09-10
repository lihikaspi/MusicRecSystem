import os
import shutil
from datasets import Dataset, load_dataset
from typing import Literal

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
# User configurable variables
# ----------------------------
dataset_size = "50m"  # choose "50m", "500m", or "5b"
dataset_type = "flat"  # options: "flat", "sequential"
save_dir = os.path.join(os.getcwd(), f"YambdaData{dataset_size}")

# ----------------------------
# Delete old folder if it exists
# ----------------------------
if os.path.exists(save_dir):
    print(f"Deleting old folder: {save_dir}")
    shutil.rmtree(save_dir)

os.makedirs(save_dir, exist_ok=True)
print(f"Created new folder: {save_dir}")

# ----------------------------
# Download datasets
# ----------------------------
dataset = YambdaDataset(dataset_type=dataset_type, dataset_size=dataset_size)

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
    file_path = os.path.join(save_dir, f"{name}.parquet")
    print(f"Saving {name} to {file_path} ...")
    ds.to_parquet(file_path)

print(f"✅ All datasets saved to folder: {save_dir}")