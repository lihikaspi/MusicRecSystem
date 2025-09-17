import os
import shutil
from project_data.download_yambda import YambdaDataset
from config import DATASET_SIZE, DATASET_TYPE, DATA_DIR, DOWNLOAD_FULL_DATASET


def main():
    # Delete old folder if it exists
    if os.path.exists(DATA_DIR):
        print(f"Deleting old folder: {DATA_DIR}")
        shutil.rmtree(DATA_DIR)

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Created new folder: {DATA_DIR}")

    # Download datasets
    dataset = YambdaDataset(dataset_type=DATASET_TYPE, dataset_size=DATASET_SIZE)

    datasets_to_save = {}

    if DOWNLOAD_FULL_DATASET:
        # Interactions
        for event in ["likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"]:
            datasets_to_save[event] = dataset.interaction(event)
    else:
        # Only download multi_event
        datasets_to_save["multi_event"] = dataset.interaction("multi_event")

    # embeddings + mappings (always download)
    datasets_to_save["embeddings"] = dataset.audio_embeddings()
    datasets_to_save["album_mapping"] = dataset.album_item_mapping()
    datasets_to_save["artist_mapping"] = dataset.artist_item_mapping()

    # Save all datasets to disk
    for name, ds in datasets_to_save.items():
        file_path = os.path.join(DATA_DIR, f"{name}.parquet")
        print(f"Saving {name} to {file_path} ...")
        ds.to_parquet(file_path)

    print(f"All datasets saved to folder: {DATA_DIR}")


if __name__ == "__main__":
    main()
