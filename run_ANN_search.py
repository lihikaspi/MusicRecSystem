import os
from config import config
from ANN_search.ANN_index import ANNIndex
from ANN_search.ANN_eval import RecEvaluator


def check_prev_files():
    """
    check for the files created in the previous stage.
    if at least one file is missing raises FileNotFoundError
    """
    needed = [config.paths.user_embeddings_gnn, config.paths.song_embeddings_gnn,
              config.paths.cold_start_songs_file]
    fail = False
    for file in needed:
      if not os.path.exists(file):
        print("Couldn't find file: {}".format(file))
        fail = True
    if fail:
      raise FileNotFoundError("Needed files are missing, run previous stage to create the needed files!")
    else:
      print("All needed files are present! starting indexing ... ")


def main():
    index = ANNIndex(config)
    recs = index.retrieve_recs()
    # evaluator = RecEvaluator(recs, config)
    # evaluator.eval()


if __name__ == "__main__":
    check_prev_files()
    main()
