# Music Recommendation System

A graph-based music recommendation system leveraging GNNs and ANN search for scalable retrieval.

Data Analysis and Presentation - Final project     
Created By: Lihi Kaspi, Harel Oved & Niv Maman

---

## Table of Contents

1. [Overview](#overview)
2. [The Yandex Yambda Dataset](#the-yandex-yambda-dataset)
3. [Setup and Requirements](#setup-and-requirements)
   - [Installing PyTorch](#installing-pytorch)
   - [Installing PyTorch Geometric](#installing-pytorch-geometric)
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

## Setup and Requirements:

Ensure your environment has enough disk space for the dataset size you plan to download (`50m`, `500m`, or `5b`).
The 50m datset takes approximately 18GB.

GPU strongly recommended for training and ANN indexing. CPU-only runs may be very slow.    
Install required Python packages:

> Python version used: Python 3.10+

```bash
pip install numpy pandas torch faiss-gpu typing-extensions scikit-learn
```

### Installing PyTorch

For GPU, make sure to install the correct version that matches your CUDA setup.
Example for CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Installing PyTorch Geometric

> version must match PyTorch

```bash
pip install torch-scatter torch-sparse torch-geometric
```

PyTorch Geometric requires additional dependencies that must match your PyTorch and CUDA versions.
Follow the official guide to install all necessary packages:
[pytorch-geometric installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)




---

## File Structure

> **Note:** Parquet files were not uploaded to this Git repository.

### Core Scripts
- `config.py`
- `yambda_download.py` 
- `run_GNN_prep.py`
- `train_GNN.py`
- `run_ANN_search.py`

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

a `config.py` file sets the save directories and dataset parameters for the download

---

## Project Pipeline

### Stage 1: Downloading the Dataset

The code downloads the Yambda dataset from Hugging Face's `datasets` library using the provided wrapper class.
All the interaction files and metadata available in this dataset are saved in parquet format.

```bash
python yambda_download.py
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

Run the preparation pipeline:

```bash
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

Run the modeling and training pipeline:

```bash
python train_GNN.py
```

### Stage 4: ANN search and retrieval

Run the ANN search and retrieval pipeline:

```bash
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


