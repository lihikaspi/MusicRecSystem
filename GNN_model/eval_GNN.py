import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict
from torch_geometric.data import HeteroData
from config import Config


class GNNEvaluator:
    def __init__(self, model: torch.nn.Module, graph: HeteroData, eval_set: str, config: Config):
        """
        Args:
            model: trained GNN model
            graph: PyG HeteroData graph (needed for full user/item embeddings)
            device: cpu or cuda
        """
        self.device = config.gnn.device
        self.model = model.to(self.device)
        self.graph = graph
        self.scores_path = getattr(config.paths, f"{eval_set}_scores_file")
        self.top_k = config.gnn.k_hit

        # Cache for embeddings
        self._cached_embeddings = None


    def _load_and_process(self) -> pd.DataFrame:
        """Load the pre-computed scores"""
        df = pd.read_parquet(self.scores_path)
        df = df.rename(columns={"user_id": "user_idx", "item_id": "item_idx"})
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


    def evaluate(self) -> Dict[str, float]:
        """
        Same interface as before, now using **adjusted_score** as ground-truth relevance.
        Also returns a new metric: **novelty@k** (fraction of top-k that are unseen).
        """
        k = self.top_k
        df = self._load_and_process()
        user_emb, item_emb = self._get_embeddings()

        metrics = {
            "ndcg@k": [],
            "hit_like@k": [],
            "hit_like_listen@k": [],
            "auc": [],
            "dislike_fpr@k": [],
            "novelty@k": []
        }

        for uid, group in df.groupby('user_idx'):
            u = user_emb[uid:uid + 1]
            pred = torch.mm(u, item_emb.T).squeeze(0).cpu().numpy()

            # ---- top-k indices (argpartition is fastest) ----
            topk_idx = np.argpartition(-pred, k)[:k]
            topk_set = set(topk_idx)

            # ---- ground-truth vectors ----
            gt_items = group["item_idx"].values
            gt_adj = group["adjusted_score"].values
            gt_seen = group["seen_in_train"].values.astype(bool)

            # full relevance vector (size = #items)
            relevance = np.zeros(len(pred), dtype=float)
            relevance[gt_items] = gt_adj

            # ----- NDCG@k (graded) -----
            top_rel = relevance[topk_idx]
            dcg = np.sum((2 ** np.maximum(top_rel, 0) - 1) / np.log2(np.arange(2, k + 2)))
            ideal = np.sort(np.maximum(gt_adj, 0))[::-1][:k]
            idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
            ndcg = dcg / (idcg if idcg > 0 else 1.0)
            metrics["ndcg@k"].append(ndcg)

            # ----- Hit@k (like-equivalent) -----
            like_items = gt_items[gt_adj > 1.0]  # >1 ≈ explicit like
            metrics["hit_like@k"].append(float(len(set(like_items) & topk_set) > 0))

            # ----- Hit@k (like+listen) -----
            pos_items = gt_items[gt_adj > 0.5]
            metrics["hit_like_listen@k"].append(float(len(set(pos_items) & topk_set) > 0))

            # ----- AUC (pos vs neg) -----
            pos_mask = relevance > 0
            neg_mask = relevance < 0
            if pos_mask.any() and neg_mask.any():
                y_true = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
                y_score = np.concatenate([pred[pos_mask], pred[neg_mask]])
                metrics["auc"].append(roc_auc_score(y_true, y_score))

            # ----- Dislike FPR@k -----
            dislike_items = gt_items[gt_adj < 0]
            if len(dislike_items):
                metrics["dislike_fpr@k"].append(float(len(set(dislike_items) & topk_set) > 0))

            # ----- Novelty@k (fraction unseen) -----
            unseen_in_topk = sum(1 for i in topk_idx if i not in gt_items)  # never interacted
            metrics["novelty@k"].append(unseen_in_topk / k)

        # ---- average over users ----
        return {m: float(np.mean(v)) if len(v) else 0.0 for m, v in metrics.items()}
