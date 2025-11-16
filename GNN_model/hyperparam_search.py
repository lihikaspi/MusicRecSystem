import torch
import optuna
import json
from torch_geometric.data import HeteroData
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import config


def objective(trial):
    # --- CORRECTED RANGES ---
    try:
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        # neg_samples_per_pos = trial.suggest_int("neg_samples_per_pos", 2, 8)
        # listen_weight = trial.suggest_float("listen_weight", 0.5, 1.0)
        # neutral_neg_weight = trial.suggest_float("neutral_neg_weight", 0.1, 0.5)
        num_layers = trial.suggest_int("num_layers", 2, 5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        metadata_scale = trial.suggest_float("metadata_scale", 1.0, 50.0, log=True)
        audio_scale = trial.suggest_float("audio_scale", 0.1, 2.0)
        # margin = trial.suggest_float("bpr_margin", 0.1, 0.5)

        bs_config = trial.suggest_categorical("bs_config",
                                              ["8_4", "16_2", "16_4", "32_2"])
        bs, accum = map(int, bs_config.split('_'))

        config.gnn.lr = lr
        # config.gnn.neg_samples_per_pos = neg_samples_per_pos
        # config.gnn.listen_weight = listen_weight
        # config.gnn.neutral_neg_weight = neutral_neg_weight
        config.gnn.num_layers = num_layers
        config.gnn.weight_decay = weight_decay
        config.gnn.dropout = dropout
        config.gnn.metadata_scale = metadata_scale
        config.gnn.audio_scale = audio_scale
        # config.gnn.bpr_margin = margin

        config.gnn.batch_size = bs
        config.gnn.accum_steps = accum


        torch.cuda.empty_cache()

        train_graph = torch.load(config.paths.train_graph_file)

        model = LightGCN(train_graph, config)
        trainer = GNNTrainer(model, train_graph, config)

        # Optional: short training for hyperparameter search
        trainer.train(trial=True)

        metric = trainer.best_ndcg

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
    storage_url = f"sqlite:///{config.paths.eval_dir}/hp_search.db"
    study_name = "gnn_hp_search_first_pass"

    study = optuna.create_study(
        storage=storage_url,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50)

    best_params = study.best_params

    with open(config.paths.best_param, "w") as f:
        json.dump(best_params, f, indent=4)

    print("Best hyperparameters:", best_params)


if __name__ == "__main__":
    main()