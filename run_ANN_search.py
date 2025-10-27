import os
from config import config

def check_prev_files():
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
  pass


if __name__ == "__main__":
  check_prev_files()
  main()
