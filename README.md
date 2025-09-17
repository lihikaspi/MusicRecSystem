# Music Recommendation System

A graph-based music recommendation system leveraging GNNs and ANN search for scalable retrieval.

Data Analysis and Presentation - Final project     
Created By: Lihi Kaspi, Harel Oved & Niv Maman

---

## Table of Contents

1. [Overview](#overview)
2. [The Yandex Yambda Dataset](#the-yandex-yambda-dataset)
3. [Setup and Requirements](#setup-and-requirements)
4. [File Structure](#file-structure)
5. [Configuration](#configuration)
6. [Project Pipeline](#project-pipeline)
   - [Stage 1: Downloading the Dataset](#stage-1-downloading-the-dataset)
   - [Stage 2: Preparing the Data for the GNN](#stage-2-preparing-the-data-for-the-gnn)
     - [1. Data Preprocessing](#1-data-preprocessing)
     - [2. Split Data](#2-split-data)
     - [3. Build Graph](#3-build-graph)
   - [Stage 3: GNN Modeling and Training](#stage-3-gnn-modeling-and-training)
   - [Stage 4: ANN Search and Retrieval](#stage-4-ann-search-and-retrieval)
     - [1. ANN Indexing and Retrieval](#1-ann-indexing-and-retrieval)
     - [2. Retrieval Evaluation](#2-retrieval-evaluation)
7. [Notes](#notes)

---

## Overview

This project implements a music recommendation system using the Yandex Yambda dataset.
The workflow includes downloading and preprocessing the dataset, constructing a user-song graph, training a GNN, and performing ANN-based recommendation retrieval.

---

## The Yandex Yambda Dataset

The Yandex Yambda dataset is a large-scale music dataset containing billions of user-track interactions from millions of users. 
It includes both implicit (listens) and explicit (likes/dislikes) feedback and precomputed audio embeddings for millions of tracks. 
The dataset is available in multiple scales (50M, 500M, 5B interactions) in Parquet format.  
More details about the dataset can be found here: [Yandex Yambda dataset](https://huggingface.co/datasets/yandex/yambda)

The files contain:
- **`multi_event.parquet`:** a unified table including multiple types of interactions (`listen`, `like`, `unlike`, `dislike`, `undislike`). This file may contain `null` values and is not cleaned.
- **single-event records:** cleaned files for each interaction type
- **audio embeddings:** pre-computed audio embedding per track
- **song mappings:** song-album and song-artist mappings

---

## Setup and Requirements

Ensure your environment has enough disk space for the dataset size you plan to download (`50m`, `500m`, or `5b`).
The 50m dataset takes approximately 18GB.

**GPU Required**: This project requires a CUDA-compatible GPU for GNN training and ANN indexing. CPU-only runs are not supported.

> Python version: Python 3.10+

Install required Python packages:

```bash
pip install -r requirements.txt
```

### Installing PyTorch

The requirements.txt includes PyTorch with CUDA 11.8 support by default. For other CUDA versions, modify the PyTorch versions in `requirements.txt`:

- **CUDA 12.1**: Change `+cu118` to `+cu121` 
- **CUDA 12.8**: Change `+cu118` to `+cu128`

Check your CUDA version:

```bash
nvidia-smi
```

Example for manual CUDA 12.8 installation:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Installing PyTorch Geometric

> Version must match PyTorch

PyTorch Geometric is included in requirements.txt with the necessary dependencies:
- `torch-geometric>=2.3.0`
- `torch-scatter>=2.1.0` 
- `torch-sparse>=0.6.17`

For additional installation guidance, follow the official guide:
[pytorch-geometric installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Verify Installation

Test your setup:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

import torch_geometric
print(f"PyG version: {torch_geometric.__version__}")
```

---

## File Structure

> **Note:** Parquet files were not uploaded to this Git repository.

### Core Scripts
- `config.py`
- `run_all.py`
- `yambda_download.py` 
- `run_GNN_prep.py`
- `train_GNN.py`
- `run_ANN_search.py`
- `requirements.txt`

### Data Processing (`GNN_prep/`)
- `data_preprocessing.py`
- `split_data.py`
- `build_graph.py`

### GNN Modeling (`GNN_model/`)
- `GNN_class.py`

### ANN Search (`ANN_search/`)
- `ANN_index_recs.py`
- `ANN_eval.py`

### Processed Data (`processed_data/`)
- `interactions.parquet`
- `train.parquet` / `val.parquet` / `test.parquet`
- `graph.pt`

### Project Data (`project_data/`)
- `yambda_inspect.py`
- `yambda_stats.py`
- #### Raw Dataset (`YambdaData50m/`)
   - `listens.parquet` / `likes.parquet` / `dislikes.parquet` / `unlikes.parquet` / `undislikes.parquet`
   - `multi_event.parquet`
   - `embeddings.parquet`
   - `album_mapping.parquet` / `artist_mapping.parquet`
   - `yambda_columns.csv`
   - `YambdaStats_50m.csv`


---

## Configuration

The `config.py` file sets the save directories, dataset parameters, number of retrieved ANN results, and also defines the pipeline stages

---

## Project Pipeline

You can run the entire project or individual stages via the top-level runner `run_all.py` (recommended):

```bash
# Run all stages in order
python run_all.py

# Run a specific stage by number (1–4)
python run_all.py --stage 3

# Run a specific stage by name
python run_all.py --stage train_gnn
```

The stages correspond to the scripts defined in `config.py`:

1. `download` → `download_yambda.py`
2. `gnn_prep` → `run_GNN_prep.py`
3. `train_gnn` → `train_GNN.py`
4. `ann_search` → `run_ANN_search.py`

Optionally, for development or debugging, you can also run individual stage scripts directly:

```bash
python download_yambda.py
python run_GNN_prep.py
python train_GNN.py
python run_ANN_search.py
```

### Stage 1: Downloading the Dataset

The code downloads the Yambda dataset from Hugging Face's `datasets` library using the provided wrapper class.
All the interaction files and metadata available in this dataset are saved in parquet format.

```bash
# via the top-level runner (recommended)
python run_all.py --stage download

# directly
python download_data.py
```

The dataset size, type and save directory are set in the `config.py` file. Example:

```python
# config.py
DATASET_SIZE = "50m"        # Options: "50m", "500m", "5b"
DATASET_TYPE = "flat"       # Options: "flat", "sequential"
DATA_DIR = f"project_data/YambdaData{DATASET_SIZE}/"
```

Additional scripts:

#### Inspect Columns
Saves a CSV with column names and the first row of each file:

```bash
python project_data/yambda_inspect.py
```

#### Basic Statistics
Saves a CSV with basic statistics for each interaction file:

```bash
python project_data/yambda_stats.py
```

### Stage 2: Preparing the Data for the GNN

Run the GNN preparation pipeline:

```bash
# via the top-level runner (recommended)
python run_all.py --stage gnn_prep

# directly
python run_GNN_prep.py
```

#### 1. Data Preprocessing

- Defines weights for each event type.
- Calculates listen counts per song for each user.
- Encodes user and track IDs.
- Saves a new Parquet file `interactions.parquet` with all event records.

#### 2. Split Data

- Splits user interactions (users with ≥5 interactions) into train, validation, and test datasets.
- Saves `train.parquet`, `val.parquet`, and `test.parquet`.

#### 3. Build Graph

- Builds a bipartite graph with users and songs as nodes and interactions as weighted edges.
- Saves the graph as a PyTorch file `graph.pt`.


All outputs are saved to the directory specified in `config.py` as `PROCESSED_DIR`:

```python
# config.py
PROCESSED_DIR = "final_project/processed_data/"
```

### Stage 3: GNN modeling and training

Run the GNN training pipeline:

```bash
# via the top-level runner (recommended)
python run_all.py --stage train_gnn

# directly
python train_GNN.py
```

### Stage 4: ANN search and retrieval

Run the ANN search and retrieval pipeline:

```bash
# via the top-level runner (recommended)
python run_all.py --stage ann_search

# directly
python run_ANN_search.py
```

#### 1. ANN Indexing and Retrieval



#### 2. Retrieval Evaluation



---

## Notes

- Dataset size options:
  - `50m`: lightweight, suitable for experiments.
  - `500m`: medium-scale dataset.
  - `5b`: full dataset, very large and resource-intensive.


