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
     - [1. Data Preprocessing](#1-data-preprocessing)
     - [2. Build Graph](#2-build-graph)
   - [Stage 3: Training and Evaluating the GNN](#stage-3-training-and-evaluating-the-gnn)
     - [1. Training](#1-training)
     - [2. Evaluation](#2-evaluation)
   - [Stage 4: ANN Search and Retrieval](#stage-4-ann-search-and-retrieval)
     - [1. ANN Indexing and Retrieval](#1-ann-indexing-and-retrieval)
     - [2. Retrieval Evaluation](#2-retrieval-evaluation)

---

## Overview

This project implements a scalable music recommendation system using the Yandex Yambda dataset. 
The system combines Graph Neural Networks (GNNs) with Approximate Nearest Neighbor (ANN) search to provide efficient and accurate music recommendations.

The workflow includes downloading and preprocessing the dataset, constructing a user-song bipartite graph, training a GNN to learn user and song embeddings, and performing ANN-based recommendation retrieval with comprehensive evaluation.

**Key Features:**
- Handles large-scale music datasets (50M to 5B interactions)
- Graph-based representation of user-song interactions
- GNN-powered embedding learning
- Fast ANN-based recommendation retrieval
- Comprehensive evaluation metrics

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

- **`multi_event.parquet`**: Unified table with multiple interaction types (`listen`, `like`, `unlike`, `dislike`, `undislike`). May contain null values.
- **Single-event records**: Cleaned files for each interaction type
- **Audio embeddings**: Pre-computed audio embeddings per track
- **Song mappings**: Song-album and song-artist relationship mappings

### Dataset Scales

- **50m**: Lightweight version, suitable for experiments and development
- **500m**: Medium-scale dataset for comprehensive testing
- **5b**: Full dataset, production-scale but very resource-intensive

---

## Setup and Requirements

Ensure your environment has enough disk space for the dataset size you plan to download (`50m`, `500m`, or `5b`).

**GPU Required**: This project requires a CUDA-compatible GPU for GNN training and ANN indexing. CPU-only runs are not supported.

> Python version: Python 3.10+

### Installing Dependencies

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
├── eval_GNN.py                 # GNN evaluation class
├── user_embeddings.pt          # User embeddings from the GNN
├── song_embeddings.pt          # Song embeddings from the GNN
└── best_model.pt               # Best model checkpoint
```

### ANN Search (`ANN_search/`)
```
├── ANN_index_recs.py           # ANN indexing and recommendation
└── ANN_eval.py                 # Recommendation evaluation metrics
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
    ├── yambda_columns.csv      # column names of each data file and first row
    └── YambdaStats_50m.csv     # basic statistics for each interaction file
```

---

## Configuration

The `config.py` file contains all configuration parameters:

- **Dataset parameters**: Size, type, download options
- **Processing parameters**: Interaction thresholds, event-type weights, split ratios
- **GNN hyperparameters**: TBD
- **ANN parameters**: TBD
- **File paths**: Input/output directories

Example key configurations:
```python
# Dataset
DATASET_SIZE = "50m"              # Options: "50m", "500m", "5b"
DATASET_TYPE = "flat"             # Options: "flat", "sequential"
DOWNLOAD_FULL_DATASET = False     # Download single-event files

# Processing
INTERACTION_THRESHOLD = 5         # Minimum interactions per user
SPLIT_RATIOS = {"train": 0.8, "val": 0.0, "test": 0.2}

# more examples about the GNN and ANN TBD
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
# config.py
DATASET_SIZE = "50m"        # Options: "50m", "500m", "5b"
DATASET_TYPE = "flat"       # Options: "flat", "sequential"
DOWNLOAD_FULL_DATASET = False
DATA_DIR = f"project_data/YambdaData{DATASET_SIZE}/"
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

# processor.filter_events(INTERACTION_THRESHOLD)
processor.filter_events(INTERACTION_THRESHOLD, INTERACTIONS_FILE)
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
aggregator.assemble_edges(EDGES_FILE)
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
INTERACTION_THRESHOLD = 5
SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.0,
    "test": 0.2
}
PROCESSED_DIR = "processed_data"
TRAIN_GRAPH_FILE = f"{PROCESSED_DIR}/graph.pt"
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
- The best model checkpoint is saved to `TRAINED_GNN` during training based on validation NDCG@K.

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
EMBED_DIM = 128
NUM_LAYERS = 3
LR = 0.001
BATCH_SIZE = 1024
NUM_EPOCHS = 20
K_HIT = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GNN_MODEL = "GNN_model"
TRAINED_GNN = f"{GNN_MODEL}/best_model.pth"
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



