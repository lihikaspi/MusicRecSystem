import torch
import os
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import config


def check_prev_files():
    needed = [config.paths.audio_embeddings_file, config.paths.train_graph_file, config.paths.test_set_file]
    fail = False
    for file in needed:
        if not os.path.exists(file):
            print("Couldn't find file: {}".format(file))
            fail = True
    if fail:
        raise FileNotFoundError("Needed files are missing, run previous stage to create the needed files!")
    else:
        print("All needed files are present! starting GNN training ... ")


def test_evaluation(model: LightGCN, train_graph, k_hit: int):
    # Evaluate using the test parquet file
    test_evaluator = GNNEvaluator(model, train_graph, config.gnn.device, config.gnn.eval_event_map)
    test_metrics = test_evaluator.evaluate(config.paths.test_set_file, k=k_hit)

    print(f"Test set metrics @K={k_hit}:")
    print(f"  NDCG@{k_hit}: {test_metrics['ndcg@k']:.4f}")
    print(f"  Hit@{k_hit} (like only): {test_metrics['hit_like@k']:.4f}")
    print(f"  Hit@{k_hit} (like+listen): {test_metrics['hit_like_listen@k']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Dislike-FPR@{k_hit}: {test_metrics['dislike_fpr@k']:.4f}")


def save_final_embeddings(model: LightGCN, train_graph, user_embed_path, song_embed_path):
    with torch.no_grad():
        # Get embeddings from the model
        user_emb, item_emb, _ = model(train_graph)

        # Move to CPU
        user_emb = user_emb.cpu()
        item_emb = item_emb.cpu()

        # Get node ID mappings (replace with your actual mappings if available)
        user_ids = torch.arange(user_emb.size(0))  # example: 0..num_users-1
        item_ids = torch.arange(item_emb.size(0))  # example: 0..num_items-1

        # Save user embeddings
        torch.save({
            "user_ids": user_ids,
            "user_emb": user_emb
        }, user_embed_path)

        # Save item/song embeddings
        torch.save({
            "item_ids": item_ids,
            "item_emb": item_emb
        }, song_embed_path)

    print(f"User embeddings saved to {user_embed_path}")
    print(f"Song embeddings saved to {song_embed_path}")


def main():
    print("Loading pre-built train graph...")
    train_graph = torch.load(config.paths.train_graph_file)

    print("Initializing LightGCN model...")
    model = LightGCN(train_graph, config)

    print("Starting training...")
    trainer = GNNTrainer(model, train_graph, config)
    trainer.train()

    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(config.paths.trained_gnn, map_location=config.gnn.device))

    print("Evaluating on test set...")
    test_evaluation(model, train_graph, config.gnn.k_hit)

    print("Saving user and item embeddings...")
    save_final_embeddings(model, train_graph, config.paths.user_embeddings_gnn, config.paths.song_embeddings_gnn)


if __name__ == "__main__":
    check_prev_files()
    main()
