import torch
import numpy as np
from tqdm import tqdm
from config import DEVICE


def hit_at_k(user_emb, item_emb, interactions, k=10, valid_types=("listen", "like")):
    """
    Computes Hit@K for a set of user-item interactions,
    counting only interactions of specified types.
    """
    user_emb = user_emb.to(DEVICE)
    item_emb = item_emb.to(DEVICE)
    hits = 0
    valid_interactions = [(u, i) for u, i, t in interactions if t in valid_types]

    for u, i_true in valid_interactions:
        scores = torch.matmul(user_emb[u:u + 1], item_emb.T).squeeze()
        top_k = torch.topk(scores, k).indices.cpu().numpy()
        if i_true in top_k:
            hits += 1
    return hits / len(valid_interactions)


def evaluate_model(model, interactions, k=10, valid_types=("listen", "like")):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb, _ = model()
        hr = hit_at_k(user_emb, item_emb, interactions, k, valid_types)
    print(f"Hit@{k}: {hr:.4f}")
    return hr

