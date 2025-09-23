
import argparse, json, math, os, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.utils import to_undirected

from GNN_model.GNN_class_niv import LightGCNWithContent

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_graph(graph_dir, make_undirected=True, normalize_weighted=True):
    graph_dir = Path(graph_dir)
    edge_index = torch.from_numpy(np.load(graph_dir / "edge_index.npy")).long()
    edge_weight = None
    if (graph_dir / "edge_weight.npy").exists():
        edge_weight = torch.from_numpy(np.load(graph_dir / "edge_weight.npy")).float()

    with open(graph_dir / "user2idx.json") as f:
        u2i = {int(k): int(v) for k, v in json.load(f).items()}
    with open(graph_dir / "item2idx.json") as f:
        v2i = {int(k): int(v) for k, v in json.load(f).items()}
    num_users = max(u2i.values()) + 1
    num_items = max(v2i.values()) + 1
    N = int(edge_index.max().item()) + 1
    assert num_users + num_items == N, "node count mismatch"

    if make_undirected:
        if edge_weight is None:
            edge_index = to_undirected(edge_index, num_nodes=N)
        else:
            from torch_geometric.utils import coalesce
            rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
            ei2 = torch.cat([edge_index, rev], dim=1)
            ew2 = torch.cat([edge_weight, edge_weight], dim=0)
            edge_index, edge_weight = coalesce(ei2, ew2, m=N, n=N, reduce="sum")

    if normalize_weighted and edge_weight is not None:
        deg = torch.zeros(N, dtype=torch.float32)
        deg.scatter_add_(0, edge_index[0], edge_weight)
        deg.scatter_add_(0, edge_index[1], edge_weight)
        deg = torch.clamp(deg, min=1e-12)
        d_inv_sqrt = deg.pow(-0.5)
        edge_weight = edge_weight * d_inv_sqrt[edge_index[0]] * d_inv_sqrt[edge_index[1]]

    return edge_index, edge_weight, num_users, num_items, u2i, v2i

def load_item_content(data_dir, v2i, dtype=torch.float32, device="cpu"):
    import pyarrow.parquet as pq
    data_dir = Path(data_dir)
    pf = pq.ParquetFile(data_dir / "embeddings.parquet")
    want_ids = set(v2i.keys())
    dim, num_items = None, max(v2i.values()) + 1
    content = None
    filled = 0
    for rg in range(pf.num_row_groups):
        batch = pf.read_row_group(rg, columns=["item_id", "normalized_embed"])
        ids = batch.column("item_id").to_pylist()
        embs = batch.column("normalized_embed").to_pylist()
        for item_id, vec in zip(ids, embs):
            if item_id in want_ids:
                if dim is None:
                    dim = len(vec)
                    content = np.zeros((num_items, dim), dtype=np.float32)
                content[v2i[item_id]] = np.asarray(vec, dtype=np.float32)
                filled += 1
    if content is None:
        raise RuntimeError("no overlapping items between graph and embeddings.parquet")
    norm = np.linalg.norm(content, axis=1, keepdims=True) + 1e-12
    content = content / norm
    return torch.from_numpy(content).to(dtype=dtype, device=device)

def build_splits_from_parquet(weights_parquet, u2i, v2i, train=0.8, val=0.1, test=0.1):
    import pandas as pd
    df = pd.read_parquet(weights_parquet, columns=["uid", "item_id", "ts_max"])
    df = df[df["uid"].isin(u2i.keys()) & df["item_id"].isin(v2i.keys())].copy()
    df["u"] = df["uid"].map(u2i).astype(np.int64)
    df["i"] = df["item_id"].map(v2i).astype(np.int64)
    df = df.sort_values(["u", "ts_max", "i"]).drop_duplicates(["u", "i"], keep="last")

    train_pos, val_pos, test_pos = {}, {}, {}
    for u, g in df.groupby("u"):
        items = g["i"].tolist()
        n = len(items)
        if n == 0: 
            continue
        n_train = max(1, int(n * train))
        n_val = max(1, int(n * val)) if n - n_train >= 2 else 0
        n_test = max(1, n - n_train - n_val) if n - n_train - n_val >= 1 else 0
        train_pos[u] = items[:n_train]
        if n_val:  val_pos[u]  = items[n_train:n_train+n_val]
        if n_test: test_pos[u] = items[n_train+n_val:n_train+n_val+n_test]
    return train_pos, val_pos, test_pos

def build_random_splits(edge_index, num_users, train=0.9, val=0.05, test=0.05):
    from collections import defaultdict
    user_pos = defaultdict(set)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for u, j in zip(src, dst):
        if u < num_users and j >= num_users:
            user_pos[u].add(j - num_users)
    train_pos, val_pos, test_pos = {}, {}, {}
    rng = random.Random(42)
    for u, items in user_pos.items():
        items = list(items); rng.shuffle(items)
        n = len(items)
        if n == 0: 
            continue
        n_train = max(1, int(n * train))
        n_val = max(0, int(n * val))
        n_test = max(0, n - n_train - n_val)
        train_pos[u] = items[:n_train]
        if n_val:  val_pos[u]  = items[n_train:n_train+n_val]
        if n_test: test_pos[u] = items[n_train+n_val:]
    return train_pos, val_pos, test_pos

class BPRBatcher:
    def __init__(self, num_items, train_pos, batch_size, seed=42):
        self.num_items = num_items
        self.user_items = {u: set(items) for u, items in train_pos.items() if len(items) > 0}
        self.users = np.array(list(self.user_items.keys()), dtype=np.int64)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def sample_batch(self):
        if len(self.users) == 0:
            raise RuntimeError("no users with positives in training set")
        u = self.rng.choice(self.users, size=min(self.batch_size, len(self.users)), replace=False)
        pos = np.empty_like(u); neg = np.empty_like(u)
        for k, uu in enumerate(u):
            pos_item = self.rng.choice(list(self.user_items[int(uu)]))
            while True:
                neg_item = int(self.rng.integers(0, self.num_items))
                if neg_item not in self.user_items[int(uu)]:
                    break
            pos[k] = pos_item; neg[k] = neg_item
        return torch.from_numpy(u), torch.from_numpy(pos), torch.from_numpy(neg)

def recall_ndcg_at_k(user_emb, item_emb, train_pos, eval_pos, K=20, approx_neg=1000, device="cpu"):
    rng = np.random.default_rng(123)
    num_items = item_emb.size(0)
    users = list(eval_pos.keys())
    if not users: 
        return 0.0, 0.0
    recs, ndcgs = [], []
    item_emb = item_emb.to(device); user_emb = user_emb.to(device)
    for u in users:
        gt = set(eval_pos[u])
        cand = set(); banned = set(train_pos.get(u, []))
        while len(cand) < approx_neg:
            x = int(rng.integers(0, num_items))
            if x not in banned and x not in gt:
                cand.add(x)
        cand = list(cand) + list(gt)
        cand_t = torch.tensor(cand, dtype=torch.long, device=device)
        scores = (user_emb[u].unsqueeze(0) * item_emb[cand_t]).sum(-1)
        topk = torch.topk(scores, k=min(K, scores.numel())).indices.cpu().numpy()
        top_items = [cand[i] for i in topk]
        hits = sum((i in gt) for i in top_items)
        recall = hits / max(1, len(gt))
        recs.append(recall)
        dcg = 0.0
        for rank, it in enumerate(top_items, start=1):
            if it in gt: dcg += 1.0 / math.log2(rank + 1.0)
        ideal_hits = min(len(gt), K)
        idcg = sum(1.0 / math.log2(r + 1.0) for r in range(1, ideal_hits + 1))
        ndcgs.append((dcg / idcg) if idcg > 0 else 0.0)
    return float(np.mean(recs)), float(np.mean(ndcgs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", type=Path, required=True)
    ap.add_argument("--data_dir",  type=Path, required=True)
    ap.add_argument("--weights_parquet", type=Path, default=None)

    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lambda_content", type=float, default=0.1)
    ap.add_argument("--freeze_content", action="store_true", default=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--steps_per_epoch", type=int, default=1000)

    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--approx_neg", type=int, default=1000)
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    print(f"[device] {device}")

    edge_index, edge_weight, num_users, num_items, u2i, v2i = load_graph(args.graph_dir, True, True)
    print(f"[graph] users={num_users} items={num_items} edges={edge_index.size(1)}")
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    print("[content] loading item content ...")
    item_content = load_item_content(args.data_dir, v2i, dtype=torch.float32, device=device)
    print(f"[content] shape={tuple(item_content.shape)}")

    if args.weights_parquet is not None and Path(args.weights_parquet).exists():
        print("[split] chronological from interactions_with_weights.parquet")
        train_pos, val_pos, test_pos = build_splits_from_parquet(args.weights_parquet, u2i, v2i, 0.8, 0.1, 0.1)
    else:
        print("[split] random from graph edges")
        train_pos, val_pos, test_pos = build_random_splits(edge_index.detach().cpu(), num_users, 0.9, 0.05, 0.05)

    if not any(len(v) for v in val_pos.values()):
        for u, items in list(train_pos.items())[:1000]:
            if len(items) >= 2:
                val_pos[u] = [items.pop()]
        print("[split] ensured some validation items")

    model = LightGCNWithContent(
        num_users=num_users,
        num_items=num_items,
        item_content_dim=item_content.size(1),
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        lambda_content=args.lambda_content,
        freeze_content=args.freeze_content,
    ).to(device)
    model.register_item_content(item_content)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = BPRBatcher(num_items=num_items, train_pos=train_pos, batch_size=args.batch_size, seed=args.seed)

    best_ndcg, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(range(args.steps_per_epoch), desc=f"epoch {epoch}/{args.epochs}")
        for _ in pbar:
            u, pos, neg = sampler.sample_batch()
            u, pos, neg = u.to(device), pos.to(device), neg.to(device)
            optim.zero_grad(set_to_none=True)
            loss, metrics, _ = model(edge_index, edge_weight, u, pos, neg, item_content=None, sample_items_for_align=None)
            loss.backward(); optim.step()
            losses.append(metrics["loss/total"])
            pbar.set_postfix({k: f"{v:.3f}" for k, v in metrics.items()})
        print(f"\n[epoch {epoch}] train_loss={float(np.mean(losses)):.4f}")

        model.eval()
        with torch.no_grad():
            u_all, i_all = model.get_all_embeddings(edge_index, edge_weight)
            recall, ndcg = recall_ndcg_at_k(u_all, i_all, train_pos, val_pos, K=args.topk, approx_neg=args.approx_neg, device=device)
        print(f"[epoch {epoch}] val Recall@{args.topk}={recall:.4f}  NDCG@{args.topk}={ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = {
                "model": model.state_dict(),
                "num_users": num_users,
                "num_items": num_items,
                "embed_dim": args.embed_dim,
                "layers": args.layers,
                "lambda_content": args.lambda_content,
                "graph_dir": str(args.graph_dir),
                "data_dir": str(args.data_dir),
                "val_ndcg": ndcg,
                "val_recall": recall,
            }
            ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
            torch.save(best_state, ckpt_dir / "lightgcn_content_best.pt")
            print("[ckpt] saved -> checkpoints/lightgcn_content_best.pt")

    print(f"[done] best NDCG@{args.topk} = {best_ndcg:.4f}")

if __name__ == "__main__":
    main()
