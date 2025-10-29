import faiss
import torch
import numpy as np
import pandas as pd
from config import Config


def load_embeddings(config: Config):
    """Load user and song embeddings, combine GNN and audio-based songs."""
    user_embs = torch.load(config.paths.user_embeddings_gnn).numpy().astype('float32')

    song_embs_gnn = torch.load(config.paths.song_embeddings_gnn).numpy().astype('float32')
    song_ids_gnn = np.arange(song_embs_gnn.shape[0])

    audio_df = pd.read_parquet(config.paths.cold_start_songs_file)
    song_ids_audio = audio_df['item_idx'].to_numpy()
    song_embs_audio = np.vstack(audio_df['item_normalized_embed'].to_numpy()).astype('float32')

    all_song_embs = np.vstack([song_embs_gnn, song_embs_audio])
    all_song_ids = np.concatenate([song_ids_gnn, song_ids_audio])

    print(f"Loaded {len(user_embs)} users and {len(all_song_embs)} songs.")
    return user_embs, all_song_embs, all_song_ids


def build_faiss_index(song_embs: np.ndarray, nlist: int, nprobe: int):
    """Build and train an IVF Flat FAISS index for cosine similarity search."""
    faiss.normalize_L2(song_embs)
    d = song_embs.shape[1]

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    print("Training FAISS IVF index...")
    index.train(song_embs)
    print("Adding embeddings to index...")
    index.add(song_embs)

    index.nprobe = nprobe
    print(f"Index built: ntotal={index.ntotal}, nlist={nlist}, nprobe={nprobe}")
    return index


def save_index(index, song_ids, index_path: str, song_ids_path: str):
    """Save FAISS index and song ID mapping."""
    faiss.write_index(index, index_path)
    np.save(song_ids_path, song_ids)
    print(f"Saved index to {index_path} and song IDs to {song_ids_path}")


def load_index(index_path, song_ids_path):
    """Load FAISS index and song IDs."""
    index = faiss.read_index(index_path)
    song_ids = np.load(song_ids_path)
    return index, song_ids


def recommend_topk(index, user_embs, song_ids, k: int):
    """Get top-k song recommendations for each user embedding."""
    faiss.normalize_L2(user_embs)
    D, I = index.search(user_embs, k)
    rec_song_ids = song_ids[I]  # shape: (num_users, k)
    rec_scores = D
    return rec_song_ids, rec_scores


def retrieve_recs(config: Config):
    """End-to-end pipeline: load embs → build index → recommend."""
    user_embs, song_embs, song_ids = load_embeddings(config)
    index = build_faiss_index(song_embs, nlist=config.ann.nlist, nprobe=config.ann.nprobe)
    save_index(index, song_ids, config.paths.ann_index, config.paths.ann_song_ids)
    rec_song_ids, rec_scores = recommend_topk(index,user_embs,song_ids, config.ann.top_k)

    print(f"Generated top-{config.ann.top_k} recommendations for {len(user_embs)} users.")
    return rec_song_ids
