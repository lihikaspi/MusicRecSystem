import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import pyarrow.parquet as pq
from torch_geometric.data import HeteroData
from tqdm import tqdm
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import Config


class BPRDataset(Dataset):
    """
    Dataset for BPR training.
    Performs negative sampling on the fly.
    Expects `interactions` as an (N,2) ndarray of (user_idx, item_idx).
    """
    def __init__(self, interactions, num_items, neg_samples_per_pos):
        self.user_item = interactions
        self.num_items = int(num_items)
        self.neg_samples_per_pos = int(neg_samples_per_pos)

        self.user2items = {}
        for u, i in interactions:
            self.user2items.setdefault(int(u), set()).add(int(i))

        # Precompute some negative samples to reduce online computation
        self.precomputed_negatives = {}
        self._precompute_negatives()

    def _precompute_negatives(self):
        """ Precompute negative samples for each user """
        for user, pos_items in self.user2items.items():
            neg_pool = []
            attempts = 0
            # keep a reasonably large pool; safe-guard with attempts limit
            while len(neg_pool) < self.neg_samples_per_pos * 100 and attempts < 1000:
                candidate = np.random.randint(0, self.num_items)
                if candidate not in pos_items:
                    neg_pool.append(candidate)
                attempts += 1
            self.precomputed_negatives[user] = neg_pool

    def __len__(self):
        return len(self.user_item)

    def __getitem__(self, idx):
        u, i_pos = self.user_item[idx]
        u = int(u)
        i_pos = int(i_pos)

        # Use precomputed negatives when available
        if u in self.precomputed_negatives and self.precomputed_negatives[u]:
            i_neg = self.precomputed_negatives[u].pop()
        else:
            # Fallback to online sampling
            while True:
                i_neg = int(np.random.randint(0, self.num_items))
                if i_neg not in self.user2items[u]:
                    break

        return torch.tensor(u, dtype=torch.long), torch.tensor(i_pos, dtype=torch.long), torch.tensor(i_neg, dtype=torch.long)


class GNNTrainer:
    def __init__(self, model: LightGCN, train_graph: HeteroData, config: Config):
        """
        model: LightGCN instance (already constructed)
        train_graph: HeteroData with 'user' and 'item' nodes and ('user','interacts','item') edges
        config: Config object
        """
        self.model = model
        self.train_graph = train_graph  # keep on CPU
        self.config = config
        self.device = config.gnn.device
        self.lambda_align = config.gnn.lambda_align
        self.batch_size = config.gnn.batch_size

        # optimizer & scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.gnn.lr,
            weight_decay=getattr(config.gnn, "weight_decay", 0.0)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )

        # validation / eval settings
        self.val_parquet = config.paths.val_set_file
        self.evaluator = GNNEvaluator(self.model, self.train_graph, self.device, config.gnn.eval_event_map)

        # build BPR dataset from the heterograph edges
        edge_index = train_graph['user', 'interacts', 'item'].edge_index  # tensor[2, E] - global ids: users 0..U-1, items 0..I-1
        user_idx_np, item_idx_np = edge_index.cpu().numpy()
        train_edges = np.stack([user_idx_np.astype(np.int64), item_idx_np.astype(np.int64)], axis=1)

        self.num_items = int(train_graph['item'].num_nodes)

        self.dataset = BPRDataset(train_edges, self.num_items, config.gnn.neg_samples_per_pos)

        # DataLoader for BPR dataset (standard PyTorch DataLoader)
        pin_memory = True if ('cuda' in str(self.device)) else False
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=getattr(config.gnn, "num_workers", 0),
            pin_memory=pin_memory
        )

        # caches
        self._cached_embeddings = None
        self._cache_valid = False

    @staticmethod
    def _bpr_loss(u_emb, pos_emb, neg_emb):
        pos_score = (u_emb * pos_emb).sum(dim=1)
        neg_score = (u_emb * neg_emb).sum(dim=1)
        return F.softplus(-(pos_score - neg_score)).mean()

    def _get_embeddings(self):
        """ Get embeddings with caching for efficiency during evaluation """
        if not self._cache_valid or self._cached_embeddings is None:
            self.model.eval()
            with torch.no_grad():
                user_emb, item_emb, _ = self.model()  # full-graph forward (should be CPU if model on CPU)
                self._cached_embeddings = (user_emb, item_emb)
                self._cache_valid = True
        return self._cached_embeddings

    def train(self):
        num_epochs = self.config.gnn.num_epochs
        save_path = self.config.paths.trained_gnn
        k_hit = self.config.gnn.k_hit
        eval_every = self.config.gnn.eval_every

        best_ndcg = 0.0

        # Precompute item embeddings cache (audio + artist + album) on CPU and detach
        # (model.item_h_cache expected by forward())
        with torch.no_grad():
            self.model.item_h_cache = (
                self.model.item_audio_emb +
                self.model.artist_emb(self.model.artist_ids) +
                self.model.album_emb(self.model.album_ids)
            ).detach()

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            self._cache_valid = False  # invalidate evaluation cache
            total_loss = 0.0
            num_batches = 0

            progress = tqdm(self.loader, desc=f"Epoch {epoch}", leave=False)
            for batch in progress:
                # batch is (u_idx, i_pos_idx, i_neg_idx) as tensors
                u_idx, i_pos_idx, i_neg_idx = [x.to(self.device) for x in batch]

                # Concatenate positive and negative items for forward pass (global ids)
                all_item_idx = torch.cat([i_pos_idx, i_neg_idx], dim=0)

                self.optimizer.zero_grad()

                # Forward: expects user_idx (global) and item_idx (global: pos+neg)
                u_emb_batch, i_emb_all, align_loss = self.model(
                    user_idx=u_idx,
                    item_idx=all_item_idx,
                    return_projections=(self.lambda_align > 0)
                )

                # Split item embeddings into pos and neg (in the same order)
                i_pos_emb = i_emb_all[: len(i_pos_idx)]
                i_neg_emb = i_emb_all[len(i_pos_idx):]

                # Compute BPR loss
                loss_bpr = self._bpr_loss(u_emb_batch, i_pos_emb, i_neg_emb)

                # Total loss with optional alignment
                loss = loss_bpr + self.lambda_align * align_loss

                # Backprop
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                if num_batches % 100 == 0:
                    progress.set_postfix({"loss": total_loss / num_batches})

            avg_loss = total_loss / max(1, num_batches)
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

                # Update learning rate scheduler
                self.scheduler.step(metrics['ndcg@k'])

                # Save best model
                if metrics['ndcg@k'] > best_ndcg:
                    torch.save(self.model.state_dict(), save_path)
                    best_ndcg = metrics['ndcg@k']
                    print(f"Saved best model (NDCG@{k_hit}: {best_ndcg:.4f})")
