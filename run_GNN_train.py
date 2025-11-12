import torch
import torch.nn.functional as F
import os
import numpy as np
import gc
import json
from torch_geometric.data import HeteroData
from GNN_model.train_GNN import GNNTrainer
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import config
from GNN_model.diagnostics import diagnose_embedding_scales


def check_prev_files():
    """
    check for the files created in the previous stage.
    if at least one file is missing raises FileNotFoundError
    """
    needed = [config.paths.audio_embeddings_file, config.paths.train_graph_file,
              config.paths.test_scores_file]
    fail = False
    for file in needed:
        if not os.path.exists(file):
            print("Couldn't find file: {}".format(file))
            fail = True
    if fail:
        raise FileNotFoundError("Needed files are missing, run previous stage to create the needed files!")
    else:
        print("All needed files are present! starting GNN training ... ")


def test_evaluation(model: LightGCN, train_graph: HeteroData):
    print("evaluating best model on test set...")
    test_evaluator = GNNEvaluator(model, train_graph, "test", config)
    test_metrics = test_evaluator.evaluate()
    k_hit = config.gnn.k_hit

    print(f"Test set metrics @K={k_hit}:")
    print(f"  NDCG@{k_hit}: {test_metrics['ndcg@k']:.4f}")
    print(f"  Hit@{k_hit} (like only): {test_metrics['hit_like@k']:.4f}")
    print(f"  Hit@{k_hit} (like+listen): {test_metrics['hit_like_listen@k']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Dislike-FPR@{k_hit}: {test_metrics['dislike_fpr@k']:.4f}")
    print(f"  Novelty@{k_hit}: {test_metrics['novelty@k']:.4f}")

    # TODO: fix to be written pretty in the txt file
    with open(config.paths.test_eval, "w") as f:
        json.dump(test_metrics, f, indent=4)


def save_final_embeddings(model: LightGCN, user_embed_path: str, song_embed_path: str):
    """
    Save embeddings with memory-efficient batched computation
    """
    torch.cuda.empty_cache()
    model.eval()

    with torch.no_grad():
        # Move edge data to CPU after computing edge weights
        device = next(model.parameters()).device

        # Compute edge weights on GPU but keep small
        edge_features = model.edge_features.to(device)
        edge_weight = model.edge_mlp(edge_features).cpu()
        del edge_features
        torch.cuda.empty_cache()

        # Move everything to CPU for computation
        edge_index_cpu = model.edge_index
        edge_weight_cpu = edge_weight

        # Compute embeddings on CPU (slower but won't OOM)
        user_nodes = torch.arange(model.num_users)
        item_nodes = torch.arange(model.num_items)

        # Get initial embeddings (move to CPU immediately)
        user_embed = F.normalize(model.user_emb.weight, p=2, dim=-1).cpu()

        # Get item embeddings batch by batch
        batch_size = 10000
        item_embeds = []
        for i in range(0, model.num_items, batch_size):
            end_idx = min(i + batch_size, model.num_items)
            batch_items = torch.arange(i, end_idx)
            item_embed_batch = model._get_item_embeddings(batch_items, torch.device('cpu'))
            item_embeds.append(item_embed_batch)
        item_embed = torch.cat(item_embeds, dim=0)

        # Concatenate on CPU
        x = torch.cat([user_embed, item_embed], dim=0)

        # LightGCN propagation on CPU
        all_emb_sum = x

        for i, conv in enumerate(model.convs):
            print(f"Processing layer {i+1}/{model.num_layers}...")
            x = conv(x, edge_index_cpu, edge_weight=edge_weight_cpu)
            all_emb_sum = all_emb_sum + x

        x = all_emb_sum / (model.num_layers + 1)
        x = F.normalize(x, p=2, dim=-1)

        user_emb = x[:model.num_users]
        item_emb = x[model.num_users:]

        # Convert to NumPy
        user_emb_np = user_emb.numpy().astype(np.float32)
        item_emb_np = item_emb.numpy().astype(np.float32)

        user_ids_np = model.user_original_ids.numpy()
        item_ids_np = model.item_original_ids.numpy()

        np.savez(user_embed_path, embeddings=user_emb_np, original_ids=user_ids_np)
        np.savez(song_embed_path, embeddings=item_emb_np, original_ids=item_ids_np)

    print(f"User embeddings saved to {user_embed_path}")
    print(f"Song embeddings saved to {song_embed_path}")


def main():
    torch.cuda.empty_cache()
    train_graph = torch.load(config.paths.train_graph_file)

    model = LightGCN(train_graph, config)

    # audio_scale, metadata_scale = diagnose_embedding_scales(model)
    # model.audio_scale = audio_scale
    # model.metadata_scale = metadata_scale

    trainer = GNNTrainer(model, train_graph, config)
    trainer.train()

    model = trainer.model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    model.load_state_dict(torch.load(config.paths.trained_gnn, map_location=config.gnn.device))
    test_evaluation(model, train_graph)

    save_final_embeddings(model, config.paths.user_embeddings_gnn, config.paths.song_embeddings_gnn)


if __name__ == "__main__":
    check_prev_files()
    main()
