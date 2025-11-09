import matplotlib as plt
import numpy as np
import seaborn as sns
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
        self.eval_dir = config.paths.eval_dir # directory (not a single file)


    def _popular_baseline(self):
        """
        Evaluates the popular recommendation system.
        """
        pass


    def _random_baseline(self):
        """
        Evaluates the random recommendation system.
        """
        pass


    def _cf_baseline(self):
        """
        Evaluates the collaborative filtering recommendation system.
        """
        pass


    def _content_baseline(self):
        """
        Evaluates the content-based recommendation system.
        """
        pass


    def _plot_eval(self):
        """
        Plots the ANN recommendations against the baselines
        """
        # TODO: each baseline against the ANN recs
        # TODO: all the baselines together against the ANN recs
        pass


    def _eval_recs(self):
        """
        Evaluates the ANN recommendations
        """
        pass


    def _eval_baselines(self):
        """
        Evaluates the recommendations of different baselines: popular, random, collaborative filtering, and content-based.
        """
        self._popular_baseline()
        self._random_baseline()
        self._cf_baseline()
        self._content_baseline()


    def eval(self):
        """
        Evaluates the ANN recommendations against popular, random, collaborative filtering, and content-based baselines
        and plots the results
        """
        self._eval_recs()
        self._eval_baselines()
        self._plot_eval()
