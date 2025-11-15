import torch
import optuna
import json
from torch_geometric.data import HeteroData
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import config


def val_evaluation(model: LightGCN, train_graph: HeteroData):
    val_evaluator = GNNEvaluator(model, train_graph, "val", config)
    val_metrics = val_evaluator.evaluate()

    print(f"trial NDCG@K: {val_metrics['ndcg@k']}")

    return val_metrics['ndcg@k']


def objective(trial):
    # --- CORRECTED RANGES ---
    try:
        # lr: Range shrunk. 0.1 is very high for this type of model.
        # Centered around your default of 0.01.
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)

        # neg_samples_per_pos: Range is good, centered on default of 5.
        neg_samples_per_pos = trial.suggest_int("neg_samples_per_pos", 2, 8)

        # listen_weight: Range extended to 1.0 to test "no extra weight".
        listen_weight = trial.suggest_float("listen_weight", 0.5, 1.0)

        # neutral_neg_weight: Range is good, centered on default of 0.3.
        neutral_neg_weight = trial.suggest_float("neutral_neg_weight", 0.1, 0.5)

        # num_layers: Range shrunk significantly. LightGCN over-smooths.
        # Centered around your default of 3.
        num_layers = trial.suggest_int("num_layers", 2, 5)

        # --- END CORRECTIONS ---

        config.gnn.lr = lr
        config.gnn.neg_samples_per_pos = neg_samples_per_pos
        config.gnn.listen_weight = listen_weight
        config.gnn.neutral_neg_weight = neutral_neg_weight
        config.gnn.num_layers = num_layers

        config.gnn.num_epochs = 15

        config.gnn.batch_size = 16
        config.gnn.accum_steps = 4

        torch.cuda.empty_cache()

        train_graph = torch.load(config.paths.train_graph_file)

        model = LightGCN(train_graph, config)
        trainer = GNNTrainer(model, train_graph, config)

        # Optional: short training for hyperparameter search
        # trainer.num_epochs = 8
        trainer.train(trial=True)

        metric = val_evaluation(model, train_graph)  # returns the rank-based metric you care about

        return metric

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n--- !!! Trial {trial.number} FAILED due to CUDA OOM !!! ---")
        print(f"    Parameters: {trial.params}")
        print(f"    Error: {e}")
        # Clean up memory
        torch.cuda.empty_cache()
        # Tell Optuna to prune this trial
        raise optuna.exceptions.TrialPruned()


def main():
    storage_url = "sqlite:///hp_search.db"
    study_name = "gnn_hp_search"

    study = optuna.create_study(
        storage=storage_url,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    with open(config.paths.best_param, "w") as f:
        json.dump(best_params, f, indent=4)

    print("Best hyperparameters:", best_params)


if __name__ == "__main__":
    main()