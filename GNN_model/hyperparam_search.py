import torch
import optuna
import json
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import config

def test_evaluation(model: LightGCN, train_graph, k_hit: int = 10):
    print("evaluating best model...")
    # Evaluate using the test parquet file
    test_evaluator = GNNEvaluator(model, train_graph, config.gnn.device, config.gnn.eval_event_map)
    test_metrics = test_evaluator.evaluate(config.paths.test_set_file, k=k_hit)

    return test_metrics

def objective(trial):
    lr = trial.suggest_loguniform("lr",1e-3,1e-1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    neg_samples_per_pos = trial.suggest_int("neg_samples_per_pos", 1, 10)
    listen_weight = trial.suggest_float("listen_weight", 0.5, 0.9)
    neutral_neg_weight = trial.suggest_float("neutral_neg_weight", 0.1, 0.5)
    num_layers = trial.suggest_int("num_layers", 1, 10)

    config.gnn.lr = lr
    config.gnn.batch_size = batch_size
    config.gnn.neg_samples_per_pos = neg_samples_per_pos
    config.gnn.listen_weight = listen_weight
    config.gnn.neutral_neg_weight = neutral_neg_weight
    config.gnn.num_layers = num_layers

    torch.cuda.empty_cache()

    train_graph = torch.load(config.paths.train_graph_file)

    model = LightGCN(train_graph, config)
    trainer = GNNTrainer(model, train_graph, config)

    # Optional: short training for hyperparameter search
    trainer.num_epochs = 1
    trainer.train()

    metric = test_evaluation(model, train_graph)  # returns the rank-based metric you care about

    return metric


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    with open(config.paths.best_param, "w") as f:
        json.dump(best_params, f, indent=4)

    print("Best hyperparameters:", best_params)


if __name__ == "__main__":
    main()
