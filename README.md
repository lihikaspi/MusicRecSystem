# Music Recommendation System
#### Data Analysis and Presentation - Final project
#### Created By: Lihi Kaspi, Harel Oved & Niv Maman

## Table of Contents

1. [Overview](#overview)
2. [Setup and Requirements](#setup-and-requirements)
   - [Installing PyTorch](#installing-pytorch)
   - [Installing PyTorch Geometric](#installing-pytorch-geometric)
3. [Files Structure](#files-structure)
4. [Configuration](#configuration)
5. [Project Pipeline](#project-pipeline)
   - [Stage 1: Downloading the Dataset](#stage-1-downloading-the-dataset)
   - [Stage 2: Preparing the Data for the GNN](#stage-2-preparing-the-data-for-the-gnn)
     - [1. data_preprocessing.py](#1-data_preprocessingpy)
     - [2. split_data.py](#2-split_datapy)
     - [3. build_graph.py](#3-build_graphpy)
   - [Stage 3: GNN Modelling and Training](#stage-3-gnn-modelling-and-training)
   - [Stage 4: ANN Search and Retrieval](#stage-4-ann-search-and-retrieval)
6. [Notes](#notes)

---

## Overview


---

## Setup and Requirements:

Ensure your environment has enough disk space for the dataset size you plan to download (`50m`, `500m`, or `5b`).

Install the required Python packages. For GPU setups:

```bash
pip install numpy pandas torch faiss-gpu typing-extensions scikit-learn
```

### Installing PyTorch

For GPU, make sure to install the correct version that matches your CUDA setup.
Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Installing PyTorch Geometric 

PyTorch Geometric requires additional dependencies that must match your PyTorch and CUDA versions.
Follow the official guide to install all necessary packages:
[pytorch-geometric installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)


---

## Files Structure

- `config.py` : script containing global variables such as data directories
- `yambda_download.py` : script that downloads the Yambda dataset and saves it as Parquet files
- `run_GNN_prep.py` : script that runs the stage 2 scripts 
- `GNN_prep/` : folder containing the stage 2 (GNN preparation) script files
  - `data_preprocessing.py` : script that prepares the data files for the graph building
  - `split_data.py` : script that splits the `interactions.parquet` file into train/val/test sets
  - `build_graph.py` : script that builds the graphs used for the GNN
- `processed_data/` : folder containing the processed data files
  - `interactions.parquet` : files containing all the interactions with respective weights
  - `train.parquet` : train set
  - `val.parquet` : validation set
  - `test.parquet` : test set
  - `graph.pt` : bipartite graph of users and songs with weighted edges created using the train set
- `project_data/` : folder containing the original data files and exploration scripts

---

## Configuration:

a `config.py` file sets the save directories and dataset parameters for the download

---

## Project Pipeline

### Stage 1: Downloading the Dataset

The code downloads the Yambda dataset from Hugging Face's `datasets` library. 
Further information about the download process can be found in the `project_data/README.md` file

```bash
python yambda_download.py
```

### Stage 2: Preparing the Data for the GNN

Run the preparation pipeline:

```bash
python run_GNN_prep.py
```

#### 1. `GNN_prep/data_preprocessing.py`

- Defines weights for each event type.
- Calculates listen counts per song for each user.
- Encodes user and track IDs.
- Saves a new Parquet file `interactions.parquet` with all event records.

#### 2. `GNN_prep/split_data.py`

- Splits user interactions (users with ≥5 interactions) into train, validation, and test datasets.
- Saves `train.parquet`, `val.parquet`, and `test.parquet`.

#### 3. `GNN_prep/build_graph.py`

- Builds a bipartite graph with users and songs as nodes and interactions as weighted edges.
- Saves the graph as a PyTorch file graph.pt.


All outputs are saved to the directory specified in `config.py` as `PROCESSED_DIR`. Example:

```python
# config.py
PROCESSED_DIR = "final_project/processed_data/"
```

### Stage 3: GNN modelling and training


### Stage 4: ANN search retrieval

---

## Notes

- The root README focuses on the high-level pipeline and processed data.
- Detailed information about raw/downloaded Yambda files is in `project_data/README.md`.
- Dataset size options:
  - `50m`: lightweight, suitable for experiments.
  - `500m`: medium-scale dataset.
  - `5b`: full dataset, very large and resource-intensive.


