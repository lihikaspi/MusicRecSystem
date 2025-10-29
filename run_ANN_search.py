import os
from config import config
from ANN_search.ANN_index import retrieve_recs
from ANN_search.ANN_eval import eval_recs

def check_prev_files():
    """
    check for the files created in the previous stage.
    if at least one file is missing raises FileNotFoundError
    """
    needed = [config.paths.song_embeddings_gnn, config.paths.audio_embeddings_gnn,
              config.paths.song_embeddings_gnn]
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
  rec_song_ids = retrieve_recs(config)
  eval_recs(rec_song_ids, config)


if __name__ == "__main__":
  check_prev_files()
  main()
