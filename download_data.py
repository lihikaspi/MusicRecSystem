import os
import shutil
from project_data.download_yambda import YambdaDataset
from config import config


def main():
    data_dir = config.paths.data_dir

    # Delete old folder if it exists
    if os.path.exists(data_dir):
        print(f"Deleting old folder: {data_dir}")
        shutil.rmtree(data_dir)

    os.makedirs(data_dir, exist_ok=True)
    print(f"Created new folder: {data_dir}")

    # Download datasets
    dataset = YambdaDataset(dataset_type=config.dataset.dataset_type,
                            dataset_size=config.dataset.dataset_size)

    datasets_to_save = {}

    if config.dataset.download_full:
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
        file_path = os.path.join(data_dir, f"{name}.parquet")
        print(f"Saving {name} to {file_path} ...")
        ds.to_parquet(file_path)

    print(f"All datasets saved to folder: {data_dir}")


if __name__ == "__main__":
    main()
