# Music Recommendation System


---

## Files Structure

- `final_project/` : folder containing all the project scripts and data
  - `config.py` : script containing global variables such as data directories
  - `data_preprocessing.py` : script that prepares the data files for the graph building
  - `split_data.py` : script that splits the `interactions.parquet` file into train/val/test sets
  - `build_graph.py` : script that builds the graphs used for the GNN
  - `processed_data/` : folder containing the processed data files
    - `interactions.parquet` : files containing all the interactions with respective weights
    - `train.parquet` : train set
    - `val.parquet` : validation set
    - `test.parquet` : test set
    - `graph.pt` : bipartite graph of users and songs with weighted edges created using the train set
  - `project_data/` : folder containing the original data files and the script to download them

---
