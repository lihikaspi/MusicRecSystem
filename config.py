from dataclasses import dataclass, field
import torch
from typing import List, Tuple, Dict
import os

# -------------------
# DATASET CONFIG
# -------------------
@dataclass
class DatasetConfig:
    dataset_size: str = "50m"          # Options: "50m", "500m", "5b"
    dataset_type: str = "flat"         # Options: "flat", "sequential"
    download_full: bool = False        # True = download everything, False = only essentials


# -------------------
# PIPELINE CONFIG
# -------------------
@dataclass
class PipelineConfig:
    stage_files: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("download", "download_data.py"),
        ("gnn_prep", "run_GNN_prep.py"),
        ("gnn_train", "run_GNN_train.py"),
        ("ann_search", "run_ANN_search.py"),
    ])


# -------------------
# PATHS CONFIG
# -------------------
@dataclass
class PathsConfig:
    dataset_size: str = "50m"
    data_dir: str = field(init=False)
    processed_dir: str = "processed_data"
    gnn_models_dir: str = "models/GNN"
    ann_models_dir: str = "models/ANN"

    data_cols_file: str = field(init=False)
    data_stats_file: str = field(init=False)

    raw_listens_file: str = field(init=False)
    raw_likes_file: str = field(init=False)
    raw_dislikes_file: str = field(init=False)
    raw_unlikes_file: str = field(init=False)
    raw_undislikes_file: str = field(init=False)
    raw_multi_event_file: str = field(init=False)

    audio_embeddings_file: str = field(init=False)

    album_mapping_file: str = field(init=False)
    artist_mapping_file: str = field(init=False)

    interactions_file: str = field(init=False)
    train_set_file: str = field(init=False)
    val_set_file: str = field(init=False)
    test_set_file: str = field(init=False)
    cold_start_songs_file: str = field(init=False)
    train_edges_file: str = field(init=False)
    train_graph_file: str = field(init=False)

    raw_data_files: List[str] = field(init=False)
    split_ratios_file: str = field(init=False)

    trained_gnn: str = field(init=False)
    user_embeddings_gnn: str = field(init=False)
    song_embeddings_gnn: str = field(init=False)

    ann_index: str = field(init=False)
    ann_song_ids: str = field(init=False)

    def __post_init__(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.gnn_models_dir, exist_ok=True)
        os.makedirs(self.ann_models_dir, exist_ok=True)

        self.data_dir = f"project_data/YambdaData{self.dataset_size}/"
        self.data_cols_file = f"{self.data_dir}/yambda_columns.csv"
        self.data_stats_file = f"{self.data_dir}/YambdaStats_{self.dataset_size}.csv"

        self.raw_listens_file = f"{self.data_dir}/listens.parquet"
        self.raw_likes_file = f"{self.data_dir}/likes.parquet"
        self.raw_dislikes_file = f"{self.data_dir}/dislikes.parquet"
        self.raw_unlikes_file = f"{self.data_dir}/unlikes.parquet"
        self.raw_undislikes_file = f"{self.data_dir}/undislikes.parquet"
        self.raw_multi_event_file = f"{self.data_dir}/multi_event.parquet"

        self.audio_embeddings_file = f"{self.data_dir}/embeddings.parquet"

        self.album_mapping_file = f"{self.data_dir}/album_mapping.parquet"
        self.artist_mapping_file = f"{self.data_dir}/artist_mapping.parquet"

        self.interactions_file = f"{self.processed_dir}/interactions.parquet"
        self.train_set_file = f"{self.processed_dir}/train.parquet"
        self.val_set_file = f"{self.processed_dir}/val.parquet"
        self.test_set_file = f"{self.processed_dir}/test.parquet"
        self.cold_start_songs_file = f"{self.processed_dir}/cold_start_songs.parquet"
        self.train_edges_file = f"{self.processed_dir}/train_edges.parquet"
        self.train_graph_file = f"{self.processed_dir}/graph.pt"

        self.raw_data_files = [
            self.raw_listens_file,
            self.raw_likes_file,
            self.raw_dislikes_file,
            self.raw_unlikes_file,
            self.raw_undislikes_file
        ]

        self.split_paths = {
            "train": self.train_set_file,
            "val": self.val_set_file,
            "test": self.test_set_file,
        }

        self.trained_gnn = f"{self.gnn_models_dir}/best_model.pth"
        self.user_embeddings_gnn = f"{self.gnn_models_dir}/user_embeddings.pt"
        self.song_embeddings_gnn = f"{self.gnn_models_dir}/song_embeddings.pt"

        self.ann_index = f"{self.ann_models_dir}/index.faiss"
        self.ann_song_ids = f"{self.ann_models_dir}/song_ids.npy"


# -------------------
# PREPROCESSING CONFIG
# -------------------
@dataclass
class PreprocessingConfig:
    edge_type_mapping: Dict[str, int] = field(default_factory=lambda: {
        "listen": 1,
        "like": 2,
        "dislike": 3,
        "unlike": 4,
        "undislike": 5
    })
    interaction_threshold: int = 500
    weights: Dict[str, float] = field(default_factory=lambda: {
        "listens": 0.7,
        "likes": 1.0,
        "dislikes": -1,
        "unlikes": -0.5,
        "undislikes": 0.5
    })
    split_ratios: Dict[str, float] = field(default_factory=lambda: {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    })


# -------------------
# GNN CONFIG
# -------------------
@dataclass
class GNNConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    embed_dim: int = 128
    num_layers: int = 3
    init_std: float = 0.1
    lambda_align: float = 0.1
    freeze_audio: bool = True

    edge_mlp_hidden_dim: int = 32
    edge_mlp_input_dim: int = 4

    lr: float = 0.001
    num_epochs: int = 50
    batch_size: int = 64
    weight_decay: float = 1e-4
    num_workers: int = 4
    eval_every: int = 5
    neg_samples_per_pos = 5

    k_hit: int = 10

    eval_event_map: Dict[str, int] = field(default_factory=lambda: {
        "like": 2,
        "listen": 1,
        "unlike": 0,
        "dislike": -1,
        "undislike": 0
    })


# -------------------
# ANN CONFIG
# -------------------
@dataclass
class ANNConfig:
    top_k: int = 10
    nprobe: int = 32
    nlist: int = 4096


# -------------------
# COMBINED CONFIG
# -------------------
@dataclass
class Config:
    dataset: DatasetConfig = DatasetConfig()
    pipeline: PipelineConfig = PipelineConfig()
    paths: PathsConfig = PathsConfig(dataset_size=DatasetConfig().dataset_size)
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    gnn: GNNConfig = GNNConfig()
    ann: ANNConfig = ANNConfig()


# single global config object
config = Config()
