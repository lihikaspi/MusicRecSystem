import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Dict, List
import json
from config import Config



class RecEvaluator:
    """
    Evaluator class for ANN recommendations.
    """
    def __init__(self, recs, config: Config):
        self.gnn_recs = recs["gnn"]
        self.content_recs = recs["content"]
        self.popular_recs = recs["popular"]
        self.random_recs = recs["random"]
        self.cf_recs = recs["cf"]

        self.top_k = config.ann.top_k
        self.relevance_scores = config.paths.test_scores_file
        self.eval_dir = config.paths.eval_dir # directory (not a single file)


    def _popular_baseline(self):
        """
        Evaluates the popular recommendation system.
        """
        popular_eval = self._eval(self.popular_recs)
        save_path = self.eval_dir + "/popular_eval_results.json"
        self._save_eval_results(popular_eval, save_path)


    def _random_baseline(self):
        """
        Evaluates the random recommendation system.
        """
        random_eval = self._eval(self.random_recs)
        save_path = self.eval_dir + "/random_eval_results.json"
        self._save_eval_results(random_eval, save_path)


    def _cf_baseline(self):
        """
        Evaluates the collaborative filtering recommendation system.
        """
        cf_eval = self._eval(self.cf_recs)
        save_path = self.eval_dir + "/cf_eval_results.json"
        self._save_eval_results(cf_eval, save_path)


    def _content_baseline(self):
        """
        Evaluates the content-based recommendation system.
        """
        content_eval = self._eval(self.content_recs)
        save_path = self.eval_dir + "/content_eval_results.json"
        self._save_eval_results(content_eval, save_path)


    def _eval_gnn_recs(self):
        """
        Evaluates the ANN recommendations
        """
        gnn_eval = self._eval(self.gnn_recs)
        save_path = self.eval_dir + "/gnn_eval_results.json"
        self._save_eval_results(gnn_eval, save_path)


    def _eval_baselines(self):
        """
        Evaluates the recommendations of different baselines: popular, random, collaborative filtering, and content-based.
        """
        self._popular_baseline()
        self._random_baseline()
        self._cf_baseline()
        self._content_baseline()


    def _save_eval_results(self, results: dict, path: str):
        """
        Save evaluation results (per-user and avg) to a JSON file.

        Args:
            results: dict returned by _eval()
            path: file path to save, e.g., 'eval_results.json'
        """
        # JSON cannot serialize numpy types, convert to native Python types
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o

        with open(path, "w") as f:
            json.dump(results, f, default=convert, indent=4)


    def _eval(self, recs: Dict[int, List[int]]) -> Dict[str, Dict]:
        """
        General evaluation function for ANN or baseline recommendations.

        Args:
            recs: dict mapping user_id -> list of recommended item_ids (top-k)

        Returns:
            dict containing:
                - "per_user": dict[user_id -> metrics]
                - "avg": dict of average metrics over all users
        """
        k = self.top_k
        df = pd.read_parquet(self.relevance_scores)
        df = df.rename(columns={"user_id": "user_idx", "item_id": "item_idx"})

        # Prepare score lookup for ranking / AUC
        scores_df = df.set_index(["user_idx", "item_idx"])["score"]

        per_user_metrics = {}
        all_metrics = {
            "ndcg@k": [],
            "hit_like@k": [],
            "hit_like_listen@k": [],
            "auc": [],
            "dislike_fpr@k": [],
            "novelty@k": []
        }

        for uid, group in df.groupby("user_idx"):
            if uid not in recs:
                continue

            rec_items = recs[uid]
            # Sort recommended items by their precomputed score
            user_scores = scores_df.loc[uid].reindex(rec_items, fill_value=0).to_numpy()
            topk_idx = [x for _, x in sorted(zip(user_scores, rec_items), reverse=True)][:k]
            topk_set = set(topk_idx)

            # Ground-truth
            gt_items = group["item_idx"].values
            gt_adj = group["adjusted_score"].values

            max_item_idx = max(np.max(gt_items), max(topk_idx)) + 1
            relevance = np.zeros(max_item_idx, dtype=float)
            relevance[gt_items] = gt_adj

            # Compute metrics
            user_metrics = {}

            # NDCG@k
            top_rel = relevance[topk_idx]
            dcg = np.sum((2 ** np.maximum(top_rel, 0) - 1) / np.log2(np.arange(2, len(top_rel) + 2)))
            ideal = np.sort(np.maximum(gt_adj, 0))[::-1][:k]
            idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
            ndcg = dcg / (idcg if idcg > 0 else 1.0)
            user_metrics["ndcg@k"] = ndcg
            all_metrics["ndcg@k"].append(ndcg)

            # Hit@k (like)
            like_items = gt_items[gt_adj > 1.0]
            hit_like = float(len(set(like_items) & topk_set) > 0)
            user_metrics["hit_like@k"] = hit_like
            all_metrics["hit_like@k"].append(hit_like)

            # Hit@k (like+listen)
            pos_items = gt_items[gt_adj > 0.5]
            hit_like_listen = float(len(set(pos_items) & topk_set) > 0)
            user_metrics["hit_like_listen@k"] = hit_like_listen
            all_metrics["hit_like_listen@k"].append(hit_like_listen)

            # AUC
            pos_mask = relevance > 0
            neg_mask = relevance < 0
            auc_val = 0.0
            if pos_mask.any() and neg_mask.any():
                user_all_scores = scores_df.loc[uid].reindex(range(len(relevance)), fill_value=0).to_numpy()
                y_true = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
                y_score = np.concatenate([user_all_scores[pos_mask], user_all_scores[neg_mask]])
                auc_val = roc_auc_score(y_true, y_score)
            user_metrics["auc"] = auc_val
            all_metrics["auc"].append(auc_val)

            # Dislike FPR@k
            dislike_items = gt_items[gt_adj < 0]
            dislike_fpr = float(len(set(dislike_items) & topk_set) > 0) if len(dislike_items) else 0.0
            user_metrics["dislike_fpr@k"] = dislike_fpr
            all_metrics["dislike_fpr@k"].append(dislike_fpr)

            # Novelty@k
            unseen_in_topk = sum(1 for i in topk_idx if i not in gt_items)
            novelty = unseen_in_topk / k
            user_metrics["novelty@k"] = novelty
            all_metrics["novelty@k"].append(novelty)

            # Save per-user metrics
            per_user_metrics[uid] = user_metrics

        # Average metrics
        avg_metrics = {m: float(np.mean(v)) if len(v) else 0.0 for m, v in all_metrics.items()}

        return {
            "per_user": per_user_metrics,
            "avg": avg_metrics
        }


    def eval(self):
        """
        Evaluates the ANN recommendations against popular, random, collaborative filtering, and content-based baselines
        and plots the results
        """
        self._eval_gnn_recs()
        self._eval_baselines()
