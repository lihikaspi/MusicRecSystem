import torch
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import config


def main():
    # Step 1: Load train graph
    print("Loading pre-built train graph...")
    train_graph = torch.load(config.paths.train_graph_file)

    # Step 2: Initialize the model
    print("Initializing LightGCN model...")
    model = LightGCN(train_graph, config)

    # Step 3: Initialize trainer
    trainer = GNNTrainer(model, train_graph, config)

    # Step 4: Train model
    print("Starting training...")
    trainer.train()

    # Step 5: Load best model & evaluate on test set
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(config.paths.trained_gnn, map_location=config.gnn.device))

    print("Evaluating on test set...")
    test_evaluator = GNNEvaluator(model, train_graph, config.gnn.device, config.gnn.eval_event_map)

    # Evaluate using the test parquet file
    k_hit = config.gnn.k_hit
    test_metrics = test_evaluator.evaluate(config.paths.test_set_file, k=k_hit)

    print(f"Test set metrics @K={k_hit}:")
    print(f"  NDCG@{k_hit}: {test_metrics['ndcg@k']:.4f}")
    print(f"  Hit@{k_hit} (like only): {test_metrics['hit_like@k']:.4f}")
    print(f"  Hit@{k_hit} (like+listen): {test_metrics['hit_like_listen@k']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Dislike-FPR@{k_hit}: {test_metrics['dislike_fpr@k']:.4f}")

    # Step 6: Save final embeddings
    print("Saving user and item embeddings...")
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
        }, config.paths.user_embeddings_gnn)

        # Save item/song embeddings
        torch.save({
            "item_ids": item_ids,
            "item_emb": item_emb
        }, config.paths.song_embeddings_gnn)

    print(f"User embeddings saved to {config.paths.user_embeddings_gnn}")
    print(f"Song embeddings saved to {config.paths.song_embeddings_gnn}")


if __name__ == "__main__":
    main()
