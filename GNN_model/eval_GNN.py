import torch
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict
from torch_geometric.data import HeteroData


class GNNEvaluator:
    def __init__(self, model: torch.nn.Module, graph: HeteroData, device, event_map: dict):
        """
        Args:
            model: trained GNN model
            graph: PyG HeteroData graph (needed for full user/item embeddings)
            device: cpu or cuda
        """
        self.model = model.to(device)
        self.graph = graph
        self.device = device
        self.event_map = event_map

        # Cache for embeddings
        self._cached_embeddings = None

    def _load_and_process(self, parquet_path: str) -> pd.DataFrame:
        """
        Load parquet and collapse multiple events to the latest per (user, item).
        """
        table = pq.read_table(parquet_path,
                              columns=['user_idx', 'item_idx', 'event_type', 'timestamp'])
        df = table.to_pandas()

        # More efficient groupby operation
        df = (df.sort_values("timestamp")
              .groupby(["user_idx", "item_idx"], as_index=False)
              .last())

        # Vectorized mapping
        df["label"] = df["event_type"].map(self.event_map).fillna(0).astype(np.int8)

        return df

    def _get_embeddings(self):
        """
        Run full-graph propagation once and cache embeddings.
        """
        if self._cached_embeddings is None:
            self.model.eval()
            with torch.no_grad():
                user_emb, item_emb, _ = self.model()
                self._cached_embeddings = (user_emb.cpu(), item_emb.cpu())
        return self._cached_embeddings

    def evaluate(self, parquet_path: str, k: int = 10) -> Dict[str, float]:
        """
        Evaluate model on validation/test parquet.
        Returns a dictionary of metrics.
        """
        df = self._load_and_process(parquet_path)
        user_emb, item_emb = self._get_embeddings()

        metrics = {
            "ndcg@k": [],
            "hit_like@k": [],
            "hit_like_listen@k": [],
            "auc": [],
            "dislike_fpr@k": []
        }

        unique_users = df['user_idx'].unique()

        for uid, group in unique_users:
            user_data = df[df['user_idx'] == uid]

            u = user_emb[uid:uid + 1]
            scores = torch.mm(u, item_emb.T).squeeze(0).numpy()

            # Sort top-k
            if len(scores) > k:
                topk_idx = np.argpartition(-scores, k)[:k]
            else:
                topk_idx = np.argsort(-scores)[:k]

            topk_items = set(topk_idx)

            # Ground truth labels
            gt_items = group["item_idx"].values
            gt_labels = group["label"].values

            # Build relevance vector
            relevance = np.zeros(len(scores), dtype=int)
            relevance[gt_items] = gt_labels

            # ---- NDCG ----
            topk_relevance = relevance[topk_idx]
            dcg = np.sum((2 ** np.maximum(topk_relevance, 0) - 1) /
                         np.log2(np.arange(2, len(topk_idx) + 2)))

            sorted_rels = np.sort(np.maximum(relevance[gt_items], 0))[::-1][:k]
            idcg = np.sum((2 ** sorted_rels - 1) /
                          np.log2(np.arange(2, len(sorted_rels) + 2)))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics["ndcg@k"].append(ndcg)

            # ---- Hit@K (like-only) ----
            like_items = gt_items[gt_labels == 2]
            hit_like = len(set(like_items) & topk_items) > 0
            metrics["hit_like@k"].append(float(hit_like))

            # ---- Hit@K (like+listen) ----
            pos_items = gt_items[gt_labels > 0]
            hit_like_listen = len(set(pos_items) & topk_items) > 0
            metrics["hit_like_listen@k"].append(float(hit_like_listen))

            # ---- AUC ----
            pos_mask = relevance > 0
            neg_mask = relevance < 0
            if pos_mask.any() and neg_mask.any():
                try:
                    y_true = np.concatenate([np.ones(pos_mask.sum()),
                                             np.zeros(neg_mask.sum())])
                    y_score = np.concatenate([scores[pos_mask], scores[neg_mask]])
                    auc = roc_auc_score(y_true, y_score)
                    metrics["auc"].append(auc)
                except:
                    pass  # Skip if AUC computation fails

            # ---- Dislike FPR ----
            dislike_items = gt_items[gt_labels < 0]
            if len(dislike_items) > 0:
                dislike_in_topk = len(set(dislike_items) & topk_items) > 0
                metrics["dislike_fpr@k"].append(float(dislike_in_topk))

        # Average across users
        return {m: float(np.mean(v)) if len(v) > 0 else 0.0 for m, v in metrics.items()}
