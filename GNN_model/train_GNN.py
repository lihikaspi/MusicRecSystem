import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pyarrow.parquet as pq
from torch_geometric.data import HeteroData
from tqdm import tqdm
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator


# -------------------------------
# BPR Dataset
# -------------------------------
class BPRDataset(Dataset):
    """
    Dataset for BPR training.
    Performs negative sampling on the fly.
    """
    def __init__(self, interactions, num_items):
        self.user_item = interactions
        self.num_items = num_items
        self.user2items = {}
        for u, i in interactions:
            self.user2items.setdefault(u, set()).add(i)

    def __len__(self):
        return len(self.user_item)

    def __getitem__(self, idx):
        u, i_pos = self.user_item[idx]
        # Negative sampling: pick an item not seen by user
        while True:
            i_neg = np.random.randint(0, self.num_items)
            if i_neg not in self.user2items[u]:
                break
        return u, i_pos, i_neg

# -------------------------------
# Trainer class
# -------------------------------
class GNNTrainer:
    def __init__(
        self,
        model: LightGCN,
        train_graph: HeteroData,
        val_parquet: str,
        device: str,
        batch_size: int,
        lr: float,
        lambda_align: float,
        event_map: dict
    ):
        """
        Args:
            model: LightGCN instance
            train_graph: HeteroData train graph
            val_parquet: path to validation Parquet file
            device: "cuda" or "cpu"
            batch_size: BPR batch size
            lr: learning rate
            lambda_align: weight for alignment loss
        """
        self.model = model.to(device)
        self.train_graph = train_graph
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_align = lambda_align

        # Load validation interactions
        self.val_interactions = self._load_interactions(val_parquet)
        self.num_items = train_graph['item'].x.shape[0]

        # Prepare train edges from heterograph
        edge_index = train_graph['user', 'interacts', 'item'].edge_index
        user_idx, item_idx = edge_index.cpu().numpy()
        train_edges = np.stack([user_idx, item_idx], axis=1)

        self.dataset = BPRDataset(train_edges, self.num_items)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.evaluator = GNNEvaluator(self.model, self.train_graph, device, event_map)
        self.val_parquet = val_parquet

    @staticmethod
    def _load_interactions(parquet_path: str):
        df = pq.read_table(parquet_path).to_pandas()
        return df[['user_idx', 'item_idx', 'event_type']].to_numpy()

    @staticmethod
    def _bpr_loss(u_emb, pos_emb, neg_emb):
        pos_score = (u_emb * pos_emb).sum(dim=1)
        neg_score = (u_emb * neg_emb).sum(dim=1)
        return torch.nn.functional.softplus(-(pos_score - neg_score)).mean()

    # Training loop
    def train(self, num_epochs: int, save_path: str, k_hit: int):
        self.model.train()
        best_ndcg = 0.0

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0
            progress = tqdm(self.loader, desc=f"Epoch {epoch}", leave=False)

            for batch in progress:
                u_idx, i_pos_idx, i_neg_idx = [torch.tensor(x, device=self.device) for x in batch]
                self.optimizer.zero_grad()

                # Forward pass on the full train graph
                user_emb, item_emb, align_loss = self.model(self.train_graph)

                u_emb = user_emb[u_idx]
                i_pos_emb = item_emb[i_pos_idx]
                i_neg_emb = item_emb[i_neg_idx]

                loss_bpr = self._bpr_loss(u_emb, i_pos_emb, i_neg_emb)
                loss = loss_bpr + self.lambda_align * align_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(u_idx)
                progress.set_postfix({"loss": total_loss / ((progress.n+1)*self.batch_size)})

            avg_loss = total_loss / len(self.dataset)
            print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

            # Validation
            metrics = self.evaluator.evaluate(self.val_parquet, k=k_hit)

            # Logging
            print(
                f"NDCG@{k_hit}={metrics['ndcg@k']:.4f} "
                f"| Hit@{k_hit} (like)={metrics['hit_like@k']:.4f} "
                f"| Hit@{k_hit} (like+listen)={metrics['hit_like_listen@k']:.4f} "
                f"| AUC={metrics['auc']:.4f} "
                f"| Dislike-FPR@{k_hit}={metrics['dislike_fpr@k']:.4f}"
            )

            # ---- Optional: checkpoint based on metric ----
            if metrics['ndcg@k'] > best_ndcg:
                torch.save(self.model.state_dict(), save_path)
                best_ndcg = metrics['ndcg@k']
                print("Saved best model based on NDCG@K")
