# TODO: NEED TO ADAPT TO OUT FILES!!

import faiss
import numpy as np

song_emb = np.load("song_emb.npy").astype("float32")
faiss.normalize_L2(song_emb)

index = faiss.IndexFlatIP(song_emb.shape[1])
index.add(song_emb)

user_emb = user_embeddings[user_id].reshape(1, -1).astype("float32")
faiss.normalize_L2(user_emb)

D, I = index.search(user_emb, k=100)  # retrieve top 100 candidates
recommended_songs = I[0]
