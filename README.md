# Music Recommendation System

A graph-based music recommendation system leveraging GNNs and ANN search for scalable retrieval.

Data Analysis and Presentation - Final project     
Created By: Lihi Kaspi, Harel Oved & Niv Maman

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [The Yandex Yambda Dataset](#the-yandex-yambda-dataset)
4. [Setup and Requirements](#setup-and-requirements)
5. [File Structure](#file-structure)
6. [Configuration](#configuration)
7. [Project Pipeline](#project-pipeline)
   - [Stage 1: Downloading the Dataset](#stage-1-downloading-the-dataset)
   - [Stage 2: Preparing the Data for the GNN](#stage-2-preparing-the-data-for-the-gnn)
   - [Stage 3: Training and Evaluating the GNN](#stage-3-training-and-evaluating-the-gnn)
   - [Stage 4: Retrieving Recommendations using ANN Index](#stage-4-retrieving-recommendations-using-ann-index)

---

## Overview

This project implements a large-scale music recommendation system using the Yandex Yambda dataset.
It combines Graph Neural Networks (GNNs) with Approximate Nearest Neighbor (ANN) search to deliver efficient and accurate 
recommendations across millions of user–song interactions.
The workflow includes dataset preprocessing, user–song graph construction, GNN-based embedding training, and ANN-powered 
retrieval and evaluation.

---

## Quick Start

```bash
# Clone repository and navigate to project directory
cd music-recommendation-system

# Install dependencies (requires Python 3.10+)
pip install -r requirements.txt

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run full pipeline (requires GPU)
python run_all.py

# Or run individual stages
python run_all.py --stage 1  # Download dataset
python run_all.py --stage 2  # Prepare data for GNN
python run_all.py --stage 3  # Train and evaluate GNN
python run_all.py --stage 4  # ANN search and evaluation
```

---

## The Yandex Yambda Dataset

The Yandex Yambda dataset is a large-scale music dataset containing billions of user-track interactions from millions of users. 
It includes both implicit (listens) and explicit (likes/dislikes) feedback and precomputed audio embeddings for millions of tracks. 
The dataset is available in multiple scales (50M, 500M, 5B interactions) in Parquet format.  
More details about the dataset can be found here: [Yandex Yambda dataset](https://huggingface.co/datasets/yandex/yambda)

### Dataset Components

- **multi-event file**: Unified table with multiple interaction types (`listen`, `like`, `unlike`, `dislike`, `undislike`).
- **Single-event files**: Files containing interactions of a singular type
- **Audio embeddings**: Precomputed audio embeddings per track
- **Song mappings**: Song-album and song-artist relationship mappings

### Dataset Scales

The dataset come in three scales, each containing user–song interactions along with likes and dislikes:

- **50m**: 10,000 users, 934,057 songs, almost 50m interactions in total.
- **500m**: 100,000 users, 3,004,578 songs, almost 500m interactions in total.
- **5b**: 1,000,000 users, 9,390,623 songs, almost 5b interactions in total.

Each dataset size allows experimentation at a different scale, from small tests to large-scale model training.

---

## Setup and Requirements

Before running the project, ensure your environment meets the following requirements:
- **Disk Space:** Enough space for the dataset you plan to download.
- **GPU:** A CUDA-compatible GPU is required for GNN training and ANN indexing. CPU-only runs are not supported.
- **Python:** Python 3.10 or higher.

### Installing Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

### Installing PyTorch

The `requirements.txt` includes PyTorch 2.0.1 with CUDA 11.7 support by default.     
For other CUDA versions, modify the PyTorch versions in `requirements.txt`.

### Installing PyTorch Geometric

> Version must match PyTorch

PyTorch Geometric is included in `requirements.txt` with the necessary dependencies:
- `torch-geometric==2.6.1`
- `torch-scatter==2.1.2+pt20cu117`
- `torch-sparse==0.6.18+pt20cu117`

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

> **Note:** data files were not uploaded to this Git repository.

### Core Scripts
```
├── config.py                   # Configuration and hyperparameters
├── run_all.py                  # Main pipeline runner
├── download_data.py            # Dataset download script
├── run_GNN_prep.py             # GNN data preparation
├── run_GNN_train.py            # GNN training and evaluation
├── run_ANN_search.py           # ANN search and evaluation
└── requirements.txt            # Python dependencies
```

### Data Processing (`GNN_prep/`)
```
├── event_processor.py          # Interaction data preprocessing
├── edge_assembler.py           # Graph edge creation and aggregation
└── build_graph.py              # Graph construction for GNN
```

### GNN Modeling (`GNN_model/`)
```
├── GNN_class.py                # Graph Neural Network implementation
├── train_GNN.py                # GNN training class
└── eval_GNN.py                 # GNN evaluation class
```

### ANN Search (`ANN_search/`)
```
├── ANN_index.py                # ANN indexing and recommendation
└── ANN_eval.py                 # Recommendation evaluation metrics
```

### Trained Models (`models/`)

```
├── GNN/
    ├── best_model.pth          # best GNN models based on validation evaluation
    ├── user_embeddings.pt      # final user embeddings
    └── song_embessings.pt      # final song embeddings
└── ANN/
    ├── index.faiss             # ANN index
    └── song_ids.npy            # song IDs
    
```

### Processed Data (`processed_data/`)
```
├── interactions.parquet        # Processed interactions data (optional)
├── train.parquet               # Training set
├── val.parquet                 # Validation set
├── test.parquet                # Test set
├── train_edges.parquet         # Training edges (optional)
└── graph.pt                    # PyTorch graph object
```

### Project Data (`project_data/`)
```
├── download_yambda.py          # Yambda dataset wrapper
├── yambda_inspect.py           # Dataset inspection utility
├── yambda_stats.py             # Dataset statistics generator
└── YambdaData50m/              # Raw dataset directory
    ├── multi_event.parquet     # unified table with all the interaction types
    ├── listens.parquet         # listens event interactions (optional)
    ├── likes.parquet           # likes event interactions (optional)
    ├── dislikes.parquet        # dislikes event interactions (optional)
    ├── unlikes.parquet         # unlikes event interactions (optional)
    ├── undislikes.parquet      # undislikes event interactions (optional)
    ├── embeddings.parquet      # pre-computed audio embeddings
    ├── album_mapping.parquet   # song-album mapping
    ├── artist_mapping.parquet  # song-artist mapping
    ├── yambda_columns.csv      # column names of each data file and first row (optional)
    └── YambdaStats_50m.csv     # basic statistics for each interaction file (optional)
```

---

## Configuration

All configuration parameters are defined using Python dataclasses in `config.py` and organized into logical sections:

- **Dataset parameters:** Size, type, download options
- **Pipeline:** Order of the stages for the main pipeline runner
- **File paths:** Input/output directories
- **Preprocessing parameters:** Interaction threshold, event-type weights, split ratios
- **GNN hyperparameters:** PyTorch device, embeddings dimension, number of layers, training hyperparameters, etc.
- **ANN parameters:** Top-K recommendations to retrieve

The pipeline contains a single global `config` object used throughout all the scripts.

Example key configurations:
```python
# Dataset
config.dataset.dataset_size         # "50m", "500m", "5b"
config.dataset.dataset_type         # "flat" or "sequential"
config.dataset.download_full        # True/False

# Preprocessing
config.preprocessing.interaction_threshold     
config.preprocessing.split_ratios   # dict with train, val and test

# GNN
config.gnn.device                   # "cuda" if available
config.gnn.embed_dim              
config.gnn.layers_num

# ANN
config.ann.top_k                    # number of recommendations to retrieve
```

---

## Project Pipeline

### Running the Pipeline

**Full pipeline (recommended):**
```bash
python run_all.py
```

**Individual stages:**
```bash
python run_all.py --stage 1        # or --stage download
python run_all.py --stage 2        # or --stage gnn_prep
python run_all.py --stage 3        # or --stage gnn_train
python run_all.py --stage 4        # or --stage ann_search
```

**Direct script execution (for debugging):**
```bash
python download_data.py
python run_GNN_prep.py
python run_GNN_train.py
python run_ANN_search.py
```

### Stage 1: Downloading the Dataset

The code downloads the Yambda dataset from Hugging Face's `datasets` library using the provided wrapper class found in `project_data/download_yambda.py`.   
All the interaction files and metadata available in this dataset are saved in parquet format.

```bash
# via the top-level runner (recommended)
python run_all.py --stage download

# directly
python download_data.py
```

As default, the code only download the multi_event, embeddings and mapping files.     
To download the entire dataset (including the single-event files) update the `confid.py` file.   
The dataset size, type and save directory are also set in the `config.py` file. Example:

```python
config.dataset.dataset_size         # "50m", "500m", "5b"
config.dataset.dataset_type         # "flat" or "sequential"
config.dataset.download_full        # True/False
config.paths.data_dir               # raw data folder
```

Additional scripts:

```bash
# Saves a CSV with column names and the first row of each file:
python project_data/yambda_inspect.py

# Saves a CSV with basic statistics for each interaction file
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

Process the interactions file (`EventProcessor` defined in `event_processor.py`):

- Filter out users with less interaction than a given threshold and songs without audio embeddings.
- Encode user and song IDs to fit GNN requirements.
- Split the interactions into train, validation, and test sets according to a given ratio.

To save the intermediate filtered file replace the code with the following:

```python
# run_GNN_prep.py

# processor.filter_events(config.preprocessing.interaction_threshold)
processor.filter_events(config.preprocessing.interaction_threshold, config.paths.interactions_file)
```

Create the graph properties information (`EdgeAssembler` defined in `edge_assembler.py`):

- Aggregate the interactions into user-song-event records with interactions counter and average played ratio (for listen event).
- Add event-type fixed weights to act as initial values for the learnable weights in the GNN.
- Add the artist and album IDs according to the mappings and encode them.
- Turn the event names into categories.

To save the intermediate ready-to-build edges file replace the code with the following:

```python
# run_GNN_prep.py

# aggregator.assemble_edges()
aggregator.assemble_edges(config.paths.train_edges_file)
```

#### 2. Build Graph

Construct the graph for the GNN (`GraphBuilder` defined in `build_graph.py`):

- Build the graph structure: bipartite graph with users and songs as nodes and the aggregated event-type interactions as edges.
- Edges attributes: event type, interactions counter, average played ratio. 
- Edge weights: initial values to act as learnable weights.
- User nodes attributes: encoded user ID to act as initial value for the learnable embedding
- Song nodes attributes: pre-computed embeddings as node features and encoded album and artist IDs to act as initial value for the learnable embedding
- Saves the graph as a PyTorch file `graph.pt`.

The save paths and hyperparameters such as the interaction threshold and split ratio can be found in the `config.py` file. 
Example:

```python
config.preprocessing.interaction_threshold  
config.preprocessing.split_ratios
config.paths.train_graph_file
```

### Stage 3: Training and Evaluating the GNN

Run the GNN training and evaluation pipeline:

```bash
# via the top-level runner (recommended)
python run_all.py --stage gnn_train

# directly
python run_GNN_train.py
```

#### 1. Training

Construct the model (`LightGCN` defined in `GNN_class.py`) and train it on the prepared graph (`GNNTrainer` defined in `train_GNN.py`):

- Model: **LightGCN with weighted edges** to capture different event types.
- Fixed audio embeddings for songs; learnable embeddings for users, artists, albums.
- Edge weights initialized with fixed values set in the previous stage and updated during training.
- Loss: BPR loss with optional L2 regularization.
- Optimizer: Adam with hyperparameters from `config.py`.
- Training loop:
  - Batches of users and negative samples (if used).  
  - Validation using the `GNNEvaluator` class after each epoch.  
  - Checkpoint saved when validation NDCG@K improves.
- The best model checkpoint is saved to `config.paths.trained_gnn` during training based on validation NDCG@K.

#### 2. Evaluation

Evaluate the model using the validation set during training, and the test set on the final model (`GNNEvaluator` defined in `eval_GNN.py`):

- The `GNNEvaluator` class is used for validation and test evaluation.
- Handles collapsing multiple events per `(user, item)` to the latest timestamp.
- Maps events to relevance labels:
  - `like=2`, `listen=1`, `unlike/undislike=0`, `dislike=-1`
- Metrics:
  - **NDCG@K** (graded relevance)  
  - **Hit@K** (like-only and like+listen)  
  - **AUC** (likes+listens vs dislikes)  
  - **Dislike-FPR@K**
- Evaluation automatically sets the model to `eval()` mode and disables gradients.

The save paths and hyperparameters for the GNN can be found in the `config.py` file. 
Example:

```python
config.gnn.device
config.gnn.embed_dim
config.gnn.num_layers

config.paths.trained_gnn

# TODO: update when done
# LR = 0.001
# BATCH_SIZE = 1024
# NUM_EPOCHS = 20
# K_HIT = 50
```

### Stage 4: Retrieving Recommendations using ANN Index

Run the ANN search and retrieval pipeline:

```bash
# via the top-level runner (recommended)
python run_all.py --stage ann_search

# directly
python run_ANN_search.py
```

#### 1. ANN Indexing and Retrieval



#### 2. Retrieval Evaluation



