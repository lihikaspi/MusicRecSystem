import torch
import pyarrow.parquet as pq
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN
from config import (
    TRAIN_GRAPH_FILE, VAL_SET_FILE, TEST_SET_FILE, TRAINED_GNN, USER_EMBEDDINGS_GNN,
    SONG_EMBEDDINGS_GNN, DEVICE, BATCH_SIZE, LR, NUM_EPOCHS, LAMBDA_ALIGN, K_HIT
)
from GNN_model.eval_GNN import evaluate_model  # Should implement Hit@K evaluation

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
        lambda_align=LAMBDA_ALIGN
    )

    # Step 4: Train model
    print("Starting training...")
    trainer.train(num_epochs=NUM_EPOCHS, save_path=TRAINED_GNN, k_hit=K_HIT)

    # Step 5: Load best model & evaluate on test set
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(TRAINED_GNN, map_location=DEVICE))
    model.eval()

    print("Loading test interactions...")
    test_df = pq.read_table(TEST_SET_FILE).to_pandas()
    test_interactions = test_df[['user_idx', 'item_idx', 'event_type']].to_numpy()

    print("Evaluating on test set...")
    hit_k = evaluate_model(model, test_interactions, k=K_HIT)
    print(f"Hit@{K_HIT} on test set: {hit_k:.4f}")

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
