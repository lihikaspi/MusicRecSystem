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
        self._orig_id_to_graph_idx = None

    def _load_and_process(self) -> pd.DataFrame:
        """Load the pre-computed scores AND vectorize the ID mapping."""
        df = pd.read_parquet(self.scores_path)
        df = df.rename(columns={"user_id": "user_idx", "item_id": "item_idx"})

        # --- KEY CHANGE 1: VECTORIZED ID MAPPING ---
        # Get the mapper (this ensures embeddings are cached first)
        self._get_embeddings()
        mapper = self._orig_id_to_graph_idx

        # Map all item IDs to graph indices at once.
        df['graph_idx'] = df['item_idx'].map(mapper)

        # Drop rows for items that are not in our model's embedding table
        df = df.dropna(subset=['graph_idx'])

        # Convert to int for numpy indexing
        df['graph_idx'] = df['graph_idx'].astype(np.int64)
        # -------------------------------------------

        return df

    def _get_embeddings(self):
        """
        Run full-graph propagation once and cache embeddings.

        --- THIS IS YOUR ORIGINAL OOM-SAFE FUNCTION ---
        It correctly uses forward_cpu() to avoid GPU OOM.
        """
        if self._cached_embeddings is None:
            self.model.eval()
            with torch.no_grad():
                # Use the CPU forward pass
                user_emb, item_emb, _ = self.model.forward_cpu()
                # Cache them on the CPU
                self._cached_embeddings = (user_emb.cpu(), item_emb.cpu())

            if self._orig_id_to_graph_idx is None:
                # Get the original IDs tensor (size=num_items) stored in the model
                item_orig_ids_tensor = self.model.item_original_ids.cpu()

                # Create a mapping from original_id -> graph_idx
                # The graph_idx is just the position in the tensor
                self._orig_id_to_graph_idx = {
                    orig_id.item(): graph_idx
                    for graph_idx, orig_id in enumerate(item_orig_ids_tensor)
                }

        return self._cached_embeddings

    def evaluate(self) -> Dict[str, float]:
        """
        Same interface as before, now using **adjusted_score** as ground-truth relevance.
        Also returns a new metric: **novelty@k** (fraction of top-k that are unseen).
        """
        print(">>> starting evaluation")
        k = self.top_k

        # This now also maps the IDs
        df = self._load_and_process()

        # These embeddings are on the CPU
        user_emb, item_emb = self._get_embeddings()

        mapper = self._orig_id_to_graph_idx

        metrics = {
            "ndcg@k": [],
            "hit_like@k": [],
            "hit_like_listen@k": [],
            "auc": [],
            "dislike_fpr@k": [],
            "novelty@k": []
        }

        for uid, group in df.groupby('user_idx'):
            # This is the unavoidable CPU dot product.
            # It's still the fastest way given your OOM constraint.
            u = user_emb[uid:uid + 1]
            pred = torch.mm(u, item_emb.T).squeeze(0).cpu().numpy()

            # ---- top-k indices (argpartition is fastest) ----
            topk_idx = np.argpartition(-pred, k)[:k]
            topk_set = set(topk_idx)

            # ---- ground-truth vectors ----

            # --- KEY CHANGE 1 (Continued): USE PRE-MAPPED INDICES ---
            # We already did the slow mapping. Just get the values.
            gt_items = group["graph_idx"].values
            gt_adj = group["adjusted_score"].values
            gt_seen = group["seen_in_train"].values.astype(bool)

            # THIS ENTIRE SLOW BLOCK IS NO LONGER NEEDED
            # gt_orig_ids = group["item_idx"].values
            # ...
            # valid_gt_items_graph_idx = []
            # ...
            # for orig_id, adj_score in zip(gt_orig_ids, gt_adj):
            # ...
            # gt_items = np.array(valid_gt_items_graph_idx, dtype=np.int64)
            # gt_adj = np.array(valid_gt_adj, dtype=np.float64)
            # --------------------------------------------------------

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

            # --- KEY CHANGE 2: OPTIMIZED AUC ---
            # Get the small index arrays of pos/neg items
            pos_items_auc = gt_items[gt_adj > 0]
            neg_items_auc = gt_items[gt_adj < 0]

            # Only proceed if we have both
            if len(pos_items_auc) > 0 and len(neg_items_auc) > 0:
                y_true = np.concatenate([
                    np.ones(len(pos_items_auc)),
                    np.zeros(len(neg_items_auc))
                ])

                # Slice the 'pred' vector using *only* the relevant indices
                y_score = np.concatenate([
                    pred[pos_items_auc],
                    pred[neg_items_auc]
                ])

                metrics["auc"].append(roc_auc_score(y_true, y_score))
            # ---------------------------------

            # ----- Dislike FPR@k -----
            dislike_items = gt_items[gt_adj < 0]
            if len(dislike_items):
                metrics["dislike_fpr@k"].append(float(len(set(dislike_items) & topk_set) > 0))

            # ----- Novelty@k (fraction unseen) -----
            unseen_in_topk = sum(1 for i in topk_idx if i not in gt_items)  # never interacted
            metrics["novelty@k"].append(unseen_in_topk / k)

        # ---- average over users ----
        return {m: float(np.mean(v)) if len(v) else 0.0 for m, v in metrics.items()}