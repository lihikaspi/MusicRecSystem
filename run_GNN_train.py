import torch
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import (
    TRAIN_GRAPH_FILE, VAL_SET_FILE, TEST_SET_FILE, TRAINED_GNN, USER_EMBEDDINGS_GNN,
    SONG_EMBEDDINGS_GNN, DEVICE, BATCH_SIZE, LR, NUM_EPOCHS, LAMBDA_ALIGN, K_HIT, EVAL_EVENT_MAP,
    EVAL_EVERY, NUM_WORKERS, WEIGHT_DECAY
)


def main():
    # Step 1: Load train graph
    print("Loading pre-built train graph...")
    train_graph = torch.load(TRAIN_GRAPH_FILE)

    # Step 2: Initialize the model
    print("Initializing LightGCN model...")
    model = LightGCN(train_graph, lambda_align=LAMBDA_ALIGN)

    # Step 3: Initialize trainer
    trainer = GNNTrainer(
        model=model,
        train_graph=train_graph,
        val_parquet=VAL_SET_FILE,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        lr=LR,
        lambda_align=LAMBDA_ALIGN,
        event_map=EVAL_EVENT_MAP,
        num_workers=NUM_WORKERS,
        weight_decay=WEIGHT_DECAY
    )

    # Step 4: Train model
    print("Starting training...")
    trainer.train(NUM_EPOCHS, TRAINED_GNN, K_HIT, EVAL_EVERY)

    # Step 5: Load best model & evaluate on test set
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(TRAINED_GNN, map_location=DEVICE))

    print("Evaluating on test set...")
    test_evaluator = GNNEvaluator(model, train_graph, DEVICE, EVAL_EVENT_MAP)

    # Evaluate using the test parquet file
    test_metrics = test_evaluator.evaluate(TEST_SET_FILE, k=K_HIT)

    print(f"Test set metrics @K={K_HIT}:")
    print(f"  NDCG@{K_HIT}: {test_metrics['ndcg@k']:.4f}")
    print(f"  Hit@{K_HIT} (like only): {test_metrics['hit_like@k']:.4f}")
    print(f"  Hit@{K_HIT} (like+listen): {test_metrics['hit_like_listen@k']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Dislike-FPR@{K_HIT}: {test_metrics['dislike_fpr@k']:.4f}")

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
        }, USER_EMBEDDINGS_GNN)

        # Save item/song embeddings
        torch.save({
            "item_ids": item_ids,
            "item_emb": item_emb
        }, SONG_EMBEDDINGS_GNN)

    print(f"User embeddings saved to {USER_EMBEDDINGS_GNN}")
    print(f"Song embeddings saved to {SONG_EMBEDDINGS_GNN}")


if __name__ == "__main__":
    main()
