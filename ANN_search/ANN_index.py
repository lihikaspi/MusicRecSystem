import faiss
import torch
import numpy as np
import pandas as pd
from config import Config


def load_embeddings(config: Config):
    user_embs = torch.load(config.paths.user_embeddings_gnn).numpy()

    song_embs_gnn = torch.load(config.paths.song_embeddings_gnn).numpy()
    song_ids_gnn = np.arange(song_embs_gnn.shape[0])

    audio_df = pd.read_parquet(config.paths.cold_start_songs_file)
    song_ids_audio = audio_df['item_idx'].to_numpy()
    song_embs_audio = np.vstack(audio_df['item_normalized_embed'].to_numpy())

    all_song_embs = np.vstack([song_embs_gnn, song_embs_audio])
    all_song_ids = np.concatenate([song_ids_gnn, song_ids_audio])

    return user_embs, all_song_embs, all_song_ids


def build_faiss_index(song_embs, nlist, nprobe):
    faiss.normalize_L2(song_embs)
    d = song_embs.shape[1]

    # Create IVF index
    quantizer = faiss.IndexFlatIP(d)  # the quantizer for the IVF
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train the index
    print("Training IVF index...")
    index.train(song_embs)
    print("Adding song embeddings to index...")
    index.add(song_embs)

    # Set search parameter
    index.nprobe = nprobe
    print(f"IVF FAISS index built with {index.ntotal} songs, nlist={nlist}, nprobe={nprobe}")
    return index


def save_index(index, song_ids, index_path, song_ids_path):
    faiss.write_index(index, index_path)
    np.save(song_ids_path, song_ids)
    print(f"Index saved to {index_path}, song IDs saved to {song_ids_path}")


def load_index(index_path, ids_path):
    index = faiss.read_index(index_path)
    song_ids = np.load(ids_path)
    return index, song_ids


def recommend_topk(index, user_embs, song_ids, k):
    faiss.normalize_L2(user_embs)
    D, I = index.search(user_embs, k)
    recommended_song_ids = song_ids[I]
    recommended_scores = D
    return recommended_song_ids, recommended_scores


def build_ann_index(config: Config):
    user_embs, song_embs, song_ids = load_embeddings(config)
    index = build_faiss_index(song_embs, config.ann.nlist, config.ann.nprobe )
    save_index(index, song_ids, config.paths.ann_index, config.paths.ann_song_ids)
    return recommend_topk(index, user_embs, song_ids, config.ann.top_k)

