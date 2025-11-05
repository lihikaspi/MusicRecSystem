import torch
import os
import numpy as np
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import config


def check_prev_files():
    """
    check for the files created in the previous stage.
    if at least one file is missing raises FileNotFoundError
    """
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
    print("evaluating best model on test set...")
    # Evaluate using the test parquet file
    # TODO: fix to current evaluator
    test_evaluator = GNNEvaluator(model, train_graph, config.gnn.device, config.gnn.eval_event_map)
    test_metrics = test_evaluator.evaluate(config.paths.test_set_file, k=k_hit)

    print(f"Test set metrics @K={k_hit}:")
    print(f"  NDCG@{k_hit}: {test_metrics['ndcg@k']:.4f}")
    print(f"  Hit@{k_hit} (like only): {test_metrics['hit_like@k']:.4f}")
    print(f"  Hit@{k_hit} (like+listen): {test_metrics['hit_like_listen@k']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Dislike-FPR@{k_hit}: {test_metrics['dislike_fpr@k']:.4f}")


def save_final_embeddings(model: LightGCN, user_embed_path, song_embed_path):
    """
    Save embeddings and original IDs as float32 NumPy arrays, ready for FAISS.
    """
    model.eval()
    with torch.no_grad():
        # Full-graph forward
        user_emb, item_emb, _ = model()

        # Convert to NumPy float32
        user_emb_np = user_emb.cpu().numpy().astype(np.float32)
        item_emb_np = item_emb.cpu().numpy().astype(np.float32)

        # Original IDs
        user_ids_np = model.user_original_ids.cpu().numpy()
        item_ids_np = model.item_original_ids.cpu().numpy()

        # Save as .npz (one file per type)
        np.savez(user_embed_path, embeddings=user_emb_np, original_ids=user_ids_np)
        np.savez(song_embed_path, embeddings=item_emb_np, original_ids=item_ids_np)

    print(f"User embeddings saved to {user_embed_path}")
    print(f"Song embeddings saved to {song_embed_path}")


def main():
    torch.cuda.empty_cache()
    train_graph = torch.load(config.paths.train_graph_file)

    print("Initializing the GNN model")
    model = LightGCN(train_graph, config)

    print("Starting training...")
    trainer = GNNTrainer(model, train_graph, config)
    trainer.train()

    model.load_state_dict(torch.load(config.paths.trained_gnn, map_location=config.gnn.device))
    test_evaluation(model, train_graph, config.gnn.k_hit)

    save_final_embeddings(model, config.paths.user_embeddings_gnn, config.paths.song_embeddings_gnn)


if __name__ == "__main__":
    check_prev_files()
    main()
