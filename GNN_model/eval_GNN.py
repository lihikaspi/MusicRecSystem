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

    def _load_and_process(self, parquet_path: str) -> pd.DataFrame:
        """
        Load parquet and collapse multiple events to the latest per (user, item).
        """
        df = pq.read_table(parquet_path).to_pandas()

        # Keep only last interaction per (user,item)
        df = df.sort_values("timestamp").groupby(
            ["user_idx", "item_idx"], as_index=False
        ).last()

        # Map event_type to relevance label
        df["label"] = df["event_type"].map(self.event_map).fillna(0).astype(int)

        return df

    def _get_embeddings(self):
        """
        Run full-graph propagation once and cache embeddings.
        """
        self.model.eval()
        with torch.no_grad():
            user_emb, item_emb, _ = self.model(self.graph)
        return user_emb, item_emb

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

        for uid, group in df.groupby("user_idx"):
            u = user_emb[uid].unsqueeze(0)  # [1, d]
            scores = torch.matmul(u, item_emb.T).squeeze(0).cpu().numpy()

            # Sort top-k
            topk_idx = np.argpartition(-scores, k)[:k]
            topk_items = set(topk_idx)

            # Ground truth labels
            gt_items = group["item_idx"].values
            gt_labels = group["label"].values

            # Build relevance vector
            relevance = np.zeros(len(scores), dtype=int)
            relevance[gt_items] = gt_labels

            # ---- NDCG ----
            dcg = 0.0
            idcg = 0.0
            for rank, item in enumerate(topk_idx, start=1):
                rel = max(relevance[item], 0)  # negatives count as 0
                if rel > 0:
                    dcg += (2**rel - 1) / np.log2(rank + 1)
            # ideal DCG from sorted ground truth
            sorted_rels = sorted([max(r, 0) for r in relevance[gt_items]], reverse=True)[:k]
            for rank, rel in enumerate(sorted_rels, start=1):
                idcg += (2**rel - 1) / np.log2(rank + 1)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics["ndcg@k"].append(ndcg)

            # ---- Hit@K (like-only) ----
            like_items = gt_items[gt_labels == 2]
            hit_like = any(i in topk_items for i in like_items)
            metrics["hit_like@k"].append(float(hit_like))

            # ---- Hit@K (like+listen) ----
            pos_items = gt_items[gt_labels > 0]
            hit_like_listen = any(i in topk_items for i in pos_items)
            metrics["hit_like_listen@k"].append(float(hit_like_listen))

            # ---- AUC ----
            pos_mask = relevance > 0
            neg_mask = relevance < 0
            if pos_mask.any() and neg_mask.any():
                y_true = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
                y_score = np.concatenate([scores[pos_mask], scores[neg_mask]])
                auc = roc_auc_score(y_true, y_score)
                metrics["auc"].append(auc)

            # ---- Dislike FPR ----
            dislike_items = gt_items[gt_labels < 0]
            dislike_in_topk = any(i in topk_items for i in dislike_items)
            if len(dislike_items) > 0:
                metrics["dislike_fpr@k"].append(float(dislike_in_topk))

        # Average across users
        return {m: float(np.mean(v)) if len(v) > 0 else 0.0 for m, v in metrics.items()}
