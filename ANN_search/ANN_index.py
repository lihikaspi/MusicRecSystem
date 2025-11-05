import faiss
import numpy as np
import pandas as pd
from config import Config


class ANNIndex:
    def __init__(self, config: Config):
        self.user_embeddings_gnn = config.paths.user_embeddings_gnn
        self.song_embeddings_gnn = config.paths.song_embeddings_gnn
        self.cold_start_songs_file = config.paths.cold_start_songs_file
        self.ann_index_path = config.paths.ann_index
        self.ann_song_ids_path = config.paths.ann_song_ids

        self.nlist = config.ann.nlist
        self.nprobe = config.ann.nprobe
        self.top_k = config.ann.top_k

        self.index = None
        self.song_ids = None
        self.user_embs = None
        self.user_ids = None
        self.song_embs = None

    def load_embeddings(self):
        """Load user and song embeddings, combining GNN and audio-based ones."""
        # ---- Load user embeddings ----
        user_data = np.load(self.user_embeddings_gnn)
        self.user_embs = user_data['embeddings']
        self.user_ids = user_data['original_ids']

        # ---- Load song embeddings ----
        song_data = np.load(self.song_embeddings_gnn)
        song_embs_gnn = song_data['embeddings']
        song_ids_gnn = song_data['original_ids']

        # ---- Load cold-start / audio embeddings ----
        audio_df = pd.read_parquet(self.cold_start_songs_file)
        song_ids_audio = audio_df['item_idx'].to_numpy()
        song_embs_audio = np.vstack(audio_df['item_normalized_embed'].to_numpy()).astype('float32')

        # ---- Combine ----
        self.song_embs = np.vstack([song_embs_gnn, song_embs_audio])
        self.song_ids = np.concatenate([song_ids_gnn, song_ids_audio])

        print(f"Loaded {len(self.user_embs)} users and {len(self.song_embs)} songs.")


    def build_index(self):
        """Build and train a FAISS IVF Flat index."""
        if self.song_embs is None:
            raise ValueError("Song embeddings not loaded. Call load_embeddings() first.")

        faiss.normalize_L2(self.song_embs)
        d = self.song_embs.shape[1]

        quantizer = faiss.IndexFlatIP(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)

        print("Training FAISS IVF index...")
        self.index.train(self.song_embs)
        print("Adding embeddings to index...")
        self.index.add(self.song_embs)

        self.index.nprobe = self.nprobe
        print(f"Index built: ntotal={self.index.ntotal}, nlist={self.nlist}, nprobe={self.nprobe}")


    def save(self):
        """Save the FAISS index and song ID mapping."""
        if self.index is None:
            raise ValueError("No index to save. Build or load an index first.")

        faiss.write_index(self.index, self.ann_index_path)
        np.save(self.ann_song_ids_path, self.song_ids)
        print(f"Saved index to {self.ann_index_path} and song IDs to {self.ann_song_ids_path}")


    def load(self):
        """Load FAISS index and song IDs."""
        self.index = faiss.read_index(self.ann_index_path)
        self.song_ids = np.load(self.ann_song_ids_path)
        print(f"Loaded index from {self.ann_index_path} with {self.index.ntotal} vectors.")


    def recommend(self, k: int = None):
        """Get top-k song recommendations for each user embedding."""
        if self.index is None or self.song_ids is None:
            raise ValueError("Index not loaded or built.")
        if self.user_embs is None:
            raise ValueError("User embeddings not loaded.")

        k = k or self.top_k
        faiss.normalize_L2(self.user_embs)

        D, I = self.index.search(self.user_embs, k)
        rec_song_ids = self.song_ids[I]
        rec_scores = D

        results = [
            {"user_id": uid, "recommended_song_ids": recs.tolist()}
            for uid, recs in zip(self.user_ids, rec_song_ids)
        ]

        print(f"Generated top-{k} recommendations for {len(self.user_embs)} users.")
        return results, rec_scores


    def retrieve_recs(self):
        """End-to-end pipeline: load embeddings → build index → recommend."""
        self.load_embeddings()
        self.build_index()
        self.save()
        results, _ = self.recommend(self.top_k)
        return results
