import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Dict, List
import json
from config import Config


class RecEvaluator:
    """
    Evaluator class for ANN and baseline recommendations.

    This version avoids dense arrays and max-ID sized allocations.
    All relevance lookups are dictionary-based for memory safety on
    non‑contiguous original item IDs.
    """
    def __init__(self, recs, config: Config):
        # Expecting keys: 'gnn', 'content', 'popular', 'random', 'cf'
        self.gnn_recs = recs.get("gnn", {})
        self.content_recs = recs.get("content", {})
        self.popular_recs = recs.get("popular", {})
        self.random_recs = recs.get("random", {})
        self.cf_recs = recs.get("cf", {})

        self.top_k = config.ann.top_k
        self.relevance_scores = config.paths.test_scores_file
        self.eval_dir = config.paths.eval_dir  # directory (not a single file)

        # Load and preprocess once
        df = pd.read_parquet(self.relevance_scores)
        df = df.rename(columns={"user_id": "user_idx", "item_id": "item_idx"})

        # Fast per-user grouping
        self._user_groups = {uid: g for uid, g in df.groupby("user_idx")}
        # Score lookup (original IDs)
        self._scores_df = df.set_index(["user_idx", "item_idx"])['score']

    # -------------------------------
    # Baselines wrappers
    # -------------------------------
    def _popular_baseline(self):
        popular_eval = self._eval(self.popular_recs)
        save_path = f"{self.eval_dir}/popular_eval_results.json"
        self._save_eval_results(popular_eval, save_path)

    def _random_baseline(self):
        random_eval = self._eval(self.random_recs)
        save_path = f"{self.eval_dir}/random_eval_results.json"
        self._save_eval_results(random_eval, save_path)

    def _cf_baseline(self):
        cf_eval = self._eval(self.cf_recs)
        save_path = f"{self.eval_dir}/cf_eval_results.json"
        self._save_eval_results(cf_eval, save_path)

    def _content_baseline(self):
        content_eval = self._eval(self.content_recs)
        save_path = f"{self.eval_dir}/content_eval_results.json"
        self._save_eval_results(content_eval, save_path)

    def _eval_gnn_recs(self):
        gnn_eval = self._eval(self.gnn_recs)
        save_path = f"{self.eval_dir}/gnn_eval_results.json"
        self._save_eval_results(gnn_eval, save_path)

    def _eval_baselines(self):
        self._popular_baseline()
        self._random_baseline()
        self._cf_baseline()
        self._content_baseline()

    # -------------------------------
    # Core helpers
    # -------------------------------
    @staticmethod
    def _json_convertible(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    def _save_eval_results(self, results: dict, path: str):
        with open(path, "w") as f:
            json.dump(results, f, default=self._json_convertible, indent=4)

    def _eval(self, recs: Dict[int, List[int]]) -> Dict[str, Dict]:
        """
        Evaluate a recommendation mapping { user_id -> [item_id, ...] }.
        Uses dictionary lookups to avoid large dense allocations.
        Returns per-user metrics and averages.
        """
        k = self.top_k
        per_user_metrics = {}
        all_metrics = {
            "ndcg@k": [],
            "hit_like@k": [],
            "hit_like_listen@k": [],
            "auc": [],
            "dislike_fpr@k": [],
            "novelty@k": []
        }

        for uid, group in self._user_groups.items():
            if uid not in recs:
                continue

            rec_items = recs[uid]
            # Ensure list of ints
            rec_items = [int(x) for x in rec_items][:k]

            # Ground-truth arrays (original item IDs, graded scores)
            gt_items = group["item_idx"].to_numpy()
            gt_adj = group["adjusted_score"].to_numpy()

            # Build fast maps/sets
            gt_map = {int(i): float(s) for i, s in zip(gt_items, gt_adj)}
            gt_set = set(int(i) for i in gt_items)
            topk_set = set(rec_items)

            # ---------- NDCG@k (graded) ----------
            # Relevance for recommended items only
            top_rel = np.array([max(gt_map.get(i, 0.0), 0.0) for i in rec_items], dtype=float)
            dcg = np.sum((2 ** top_rel - 1) / np.log2(np.arange(2, len(top_rel) + 2)))
            ideal = np.sort(np.maximum(gt_adj, 0.0))[::-1][:k]
            idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
            ndcg = float(dcg / (idcg if idcg > 0 else 1.0))
            all_metrics["ndcg@k"].append(ndcg)

            # ---------- Hits ----------
            like_items = set(int(i) for i in gt_items[gt_adj > 1.0])
            pos_items = set(int(i) for i in gt_items[gt_adj > 0.5])
            hit_like = float(len(like_items & topk_set) > 0)
            hit_like_listen = float(len(pos_items & topk_set) > 0)
            all_metrics["hit_like@k"].append(hit_like)
            all_metrics["hit_like_listen@k"].append(hit_like_listen)

            # ---------- AUC (pos vs neg among GT only) ----------
            pos_auc = [int(i) for i in gt_items[gt_adj > 0]]
            neg_auc = [int(i) for i in gt_items[gt_adj < 0]]
            auc_val = 0.0
            if len(pos_auc) > 0 and len(neg_auc) > 0:
                # Pull scores only for GT pos/neg items (no dense reindex)
                user_scores = self._scores_df.loc[uid]
                y_true = np.concatenate([np.ones(len(pos_auc)), np.zeros(len(neg_auc))])
                y_score = np.concatenate([
                    user_scores.reindex(pos_auc, fill_value=0).to_numpy(),
                    user_scores.reindex(neg_auc, fill_value=0).to_numpy()
                ])
                try:
                    auc_val = float(roc_auc_score(y_true, y_score))
                except ValueError:
                    auc_val = 0.0
            all_metrics["auc"].append(auc_val)

            # ---------- Dislike FPR@k ----------
            dislike_items = set(int(i) for i in gt_items[gt_adj < 0])
            dislike_fpr = float(len(dislike_items & topk_set) > 0) if len(dislike_items) else 0.0
            all_metrics["dislike_fpr@k"].append(dislike_fpr)

            # ---------- Novelty@k ----------
            unseen_in_topk = sum(1 for i in rec_items if i not in gt_set)
            novelty = float(unseen_in_topk) / float(k if k > 0 else 1)
            all_metrics["novelty@k"].append(novelty)

            # Store per-user
            per_user_metrics[int(uid)] = {
                "ndcg@k": ndcg,
                "hit_like@k": hit_like,
                "hit_like_listen@k": hit_like_listen,
                "auc": auc_val,
                "dislike_fpr@k": dislike_fpr,
                "novelty@k": novelty,
            }

        # Average metrics
        avg_metrics = {m: float(np.mean(v)) if len(v) else 0.0 for m, v in all_metrics.items()}

        return {
            "per_user": per_user_metrics,
            "avg": avg_metrics
        }

    # -------------------------------
    # Public entry
    # -------------------------------
    def eval(self):
        self._eval_gnn_recs()
        self._eval_baselines()
