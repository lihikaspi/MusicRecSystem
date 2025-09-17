from datasets import Dataset, load_dataset
from typing import Literal

# ----------------------------
# Wrapper Class for Yambda
# ----------------------------
class YambdaDataset:
    """
    This class was taken from Hugging Face: https://huggingface.co/datasets/yandex/yambda
    """
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
