import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pyarrow.parquet as pq
from torch_geometric.data import HeteroData
from tqdm import tqdm
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import Config


# BPR Dataset
class BPRDataset(Dataset):
    """
    Dataset for BPR training.
    Performs negative sampling on the fly.
    """
    def __init__(self, interactions, num_items, neg_samples_per_pos):
        self.user_item = interactions
        self.num_items = num_items
        self.neg_samples_per_pos = neg_samples_per_pos

        self.user2items = {}
        for u, i in interactions:
            self.user2items.setdefault(u, set()).add(i)

        # Precompute some negative samples to reduce online computation
        self.precomputed_negatives = {}
        self._precompute_negatives()

    def _precompute_negatives(self):
        """ Precompute negative samples for each user """
        for user, pos_items in self.user2items.items():
            neg_pool = []
            attempts = 0
            while len(neg_pool) < self.neg_samples_per_pos * 100 and attempts < 1000:  # Safety limit
                candidate = np.random.randint(0, self.num_items)
                if candidate not in pos_items:
                    neg_pool.append(candidate)
                attempts += 1
            self.precomputed_negatives[user] = neg_pool

    def __len__(self):
        return len(self.user_item)

    def __getitem__(self, idx):
        u, i_pos = self.user_item[idx]

        # Use precomputed negatives when available
        if u in self.precomputed_negatives and self.precomputed_negatives[u]:
            i_neg = self.precomputed_negatives[u].pop()
        else:
            # Fallback to online sampling
            while True:
                i_neg = np.random.randint(0, self.num_items)
                if i_neg not in self.user2items[u]:
                    break

        return u, i_pos, i_neg


# Trainer class
class GNNTrainer:
    def __init__(self, model: LightGCN, train_graph: HeteroData, config: Config):
        """
        Args:
            model: LightGCN instance
            train_graph: HeteroData train graph
            config: Config object
        """
        self.config = config

        self.device = config.gnn.device
        self.model = model.to(self.device)
        self.train_graph = train_graph.to("cpu")
        self.batch_size = config.gnn.batch_size
        self.lr = config.gnn.lr
        self.lambda_align = config.gnn.lambda_align

        # Load validation interactions
        self.val_parquet = config.paths.val_set_file
        self.val_interactions = self._load_interactions(self.val_parquet)
        self.num_items = train_graph['item'].x.shape[0]

        # Prepare train edges from heterograph
        edge_index = train_graph['user', 'interacts', 'item'].edge_index
        user_idx, item_idx = edge_index.cpu().numpy()
        train_edges = np.stack([user_idx, item_idx], axis=1)

        self.dataset = BPRDataset(train_edges, self.num_items, config.gnn.neg_samples_per_pos)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.gnn.num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=config.gnn.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )

        self.evaluator = GNNEvaluator(self.model, self.train_graph, self.device, config.gnn.eval_event_map)

        # Cache embeddings computation
        self._cached_embeddings = None
        self._cache_valid = False

    @staticmethod
    def _load_interactions(parquet_path: str):
        df = pq.read_table(parquet_path).to_pandas()
        return df[['user_idx', 'item_idx', 'event_type']].to_numpy()

    @staticmethod
    def _bpr_loss(u_emb, pos_emb, neg_emb):
        pos_score = (u_emb * pos_emb).sum(dim=1)
        neg_score = (u_emb * neg_emb).sum(dim=1)
        return torch.nn.functional.softplus(-(pos_score - neg_score)).mean()

    def _get_embeddings(self):
        """ Get embeddings with caching for efficiency during evaluation """
        if not self._cache_valid or self._cached_embeddings is None:
            self.model.eval()
            with torch.no_grad():
                user_emb, item_emb, _ = self.model()
                self._cached_embeddings = (user_emb, item_emb)
                self._cache_valid = True
        return self._cached_embeddings

    # Training loop
    def train(self):
        num_epochs = self.config.gnn.num_epochs
        save_path = self.config.paths.trained_gnn
        k_hit = self.config.gnn.k_hit
        eval_every = self.config.gnn.eval_every

        # Initialize best NDCG
        best_ndcg = 0.0

        # Compile model for PyTorch 2.0+ if available
        # if hasattr(torch, 'compile'):
        #     self.model = torch.compile(self.model)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            self._cache_valid = False  # Invalidate cache when training

            total_loss = 0
            num_batches = 0
            progress = tqdm(self.loader, desc=f"Epoch {epoch}", leave=False)

            for batch in progress:
                u_idx, i_pos_idx, i_neg_idx = [
                    torch.tensor(x)
                    for x in batch
                ]
                self.optimizer.zero_grad()

                # Forward pass on the full train graph
                user_emb_full, item_emb_full, align_loss = self.model(self.train_graph)

                u_emb = user_emb_full[u_idx].to(self.device)
                i_pos_emb = item_emb_full[i_pos_idx].to(self.device)
                i_neg_emb = item_emb_full[i_neg_idx].to(self.device)

                loss_bpr = self._bpr_loss(u_emb, i_pos_emb, i_neg_emb)
                loss = loss_bpr + self.lambda_align * align_loss

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if num_batches % 100 == 0:  # Update progress less frequently
                    progress.set_postfix({"loss": total_loss / num_batches})

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

            # Evaluate less frequently to speed up training
            if epoch % eval_every == 0 or epoch == num_epochs:
                metrics = self.evaluator.evaluate(self.val_parquet, k=k_hit)

                print(
                    f"NDCG@{k_hit}={metrics['ndcg@k']:.4f} "
                    f"| Hit@{k_hit} (like)={metrics['hit_like@k']:.4f} "
                    f"| Hit@{k_hit} (like+listen)={metrics['hit_like_listen@k']:.4f} "
                    f"| AUC={metrics['auc']:.4f} "
                    f"| Dislike-FPR@{k_hit}={metrics['dislike_fpr@k']:.4f}"
                )

                # Update learning rate
                self.scheduler.step(metrics['ndcg@k'])

                # Save best model
                if metrics['ndcg@k'] > best_ndcg:
                    torch.save(self.model.state_dict(), save_path)
                    best_ndcg = metrics['ndcg@k']
                    print(f"Saved best model (NDCG@{k_hit}: {best_ndcg:.4f})")
