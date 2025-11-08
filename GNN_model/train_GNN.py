import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import subgraph
from GNN_model.GNN_class import LightGCN
from config import Config


class BPRDataset(Dataset):
    def __init__(self, train_graph, config: Config):
        edge_index = train_graph['user', 'interacts', 'item'].edge_index
        edge_attr = train_graph['user', 'interacts', 'item'].edge_attr

        user_idx, item_idx = edge_index
        edge_types = edge_attr[:, 0]

        self.user2pos = {}
        self.user2pos_types = {}
        self.user2neg = {}
        self.num_items = int(train_graph['item'].num_nodes)

        self.listen_weight = config.gnn.listen_weight
        self.neutral_neg_weight = config.gnn.neutral_neg_weight
        self.neg_samples_per_pos = config.gnn.neg_samples_per_pos
        self.edge_type_mapping = config.preprocessing.edge_type_mapping

        pos_types = [self.edge_type_mapping[k] for k in ["listen", "like", "undislike"]]
        neg_types = [self.edge_type_mapping[k] for k in ["dislike", "unlike"]]

        for u, i, t in zip(user_idx.tolist(), item_idx.tolist(), edge_types.tolist()):
            t_int = int(round(t))
            if t_int in pos_types:
                self.user2pos.setdefault(u, []).append(i)
                self.user2pos_types.setdefault(u, []).append(t_int)
            elif t_int in neg_types:
                self.user2neg.setdefault(u, []).append(i)

        for u in self.user2neg:
            self.user2neg[u] = list(set(self.user2neg[u]) - set(self.user2pos.get(u, [])))

        self.users = list(self.user2pos.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos_items = self.user2pos[u]
        pos_types = self.user2pos_types[u]
        neg_items = self.user2neg.get(u, [])

        i_pos_idx = np.random.randint(len(pos_items))
        i_pos = pos_items[i_pos_idx]
        edge_type_of_pos = pos_types[i_pos_idx]
        pos_weight = self.listen_weight if edge_type_of_pos == self.edge_type_mapping["listen"] else 1.0

        known_pos_set = set(pos_items)
        all_negs_set = set(neg_items)

        neg_samples = []
        neg_weights = []
        for _ in range(self.neg_samples_per_pos):

            # First, try to get a "hard" negative
            if len(neg_items) > 0:
                i_neg = np.random.choice(neg_items)
                neg_items.remove(i_neg)  # Sample without replacement
                weight = 1.0

            # **NEW LOGIC: If no hard negatives, sample from all items**
            else:
                while True:
                    # Sample a random item from the *entire catalog*
                    i_neg = np.random.randint(self.num_items)

                    # Keep sampling until we find one that is NOT positive
                    if i_neg not in known_pos_set:
                        break  # Found a valid random negative

                weight = self.neutral_neg_weight

            neg_samples.append(i_neg)
            neg_weights.append(weight)

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i_pos, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long),
            torch.tensor(pos_weight, dtype=torch.float),
            torch.tensor(neg_weights, dtype=torch.float),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long),
        )


def collate_bpr(batch):
    u_idx = torch.stack([b[0] for b in batch])
    i_pos_idx = torch.stack([b[1] for b in batch])
    i_neg_idx = torch.stack([b[2] for b in batch])
    pos_weights = torch.stack([b[3] for b in batch])
    neg_weights = torch.stack([b[4] for b in batch])
    all_pos = [b[5] for b in batch]
    all_neg = [b[6] for b in batch]
    return u_idx, i_pos_idx, i_neg_idx, pos_weights, neg_weights, all_pos, all_neg


class GNNTrainer:
    """
    Memory-efficient trainer with ZERO extra parameter storage.
    Uses pure SGD with gradient accumulation and careful scheduling.
    """

    def __init__(self, model: LightGCN, train_graph, config: Config):
        self.device = config.gnn.device
        self.model = model.to(self.device)
        self.train_graph = train_graph
        self.batch_size = config.gnn.batch_size
        self.num_epochs = config.gnn.num_epochs
        self.save_path = config.paths.trained_gnn

        self.dataset = BPRDataset(train_graph, config)
        pin_memory = 'cuda' in str(self.device)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.gnn.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_bpr
        )

        self.edge_index_full = train_graph['user', 'interacts', 'item'].edge_index.cpu()

        # **MEMORY-EFFICIENT: No momentum storage, just scalar tracking**
        self.lr_base = config.gnn.lr
        self.lr_decay = config.gnn.lr_decay  # Slower decay
        self.weight_decay = config.gnn.weight_decay
        self.max_grad_norm = config.gnn.max_grad_norm

        # **Track only scalars for adaptive LR (negligible memory)**
        self.step_count = 0
        self.warmup_steps = 100  # Gradual warmup

        # **Gradient accumulation to simulate larger batches without memory cost**
        self.accum_steps = config.gnn.accum_steps  # Accumulate over 2 batches

    def _get_lr(self, epoch):
        """
        Learning rate schedule without storing state
        """
        # Warmup for first epoch
        if self.step_count < self.warmup_steps:
            return self.lr_base * (self.step_count / self.warmup_steps)

        # Exponential decay after warmup
        return self.lr_base * (self.lr_decay ** epoch)

    def _bpr_loss(self, u_emb, pos_emb, neg_emb, pos_weight, neg_weight):
        """Weighted BPR loss with margin"""
        pos_score = (u_emb * pos_emb).sum(dim=-1, keepdim=True)
        neg_score = (u_emb.unsqueeze(1) * neg_emb).sum(dim=-1)

        margin = 0.1
        diff = pos_score - neg_score + margin
        loss = -torch.log(torch.sigmoid(diff) + 1e-8)

        weighted_loss = loss * pos_weight.unsqueeze(1) * neg_weight
        return weighted_loss.mean()

    def _clip_and_get_norm(self):
        """
        Clip gradients by global norm, return the norm value.
        This is done in-place, no extra memory.
        """
        total_norm = 0.0
        params_with_grad = []

        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                params_with_grad.append(param)

        total_norm = total_norm ** 0.5

        # Clip if needed
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in params_with_grad:
                param.grad.data.mul_(clip_coef)

        return total_norm

    def _update_parameters(self, lr):
        """
        Pure SGD update with per-parameter LR and weight decay.
        NO extra memory usage.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue

                # **Different LR for different parameter types**
                if 'user_emb' in name:
                    param_lr = lr * 1.0  # Normal for user embeddings
                elif 'artist_emb' in name or 'album_emb' in name:
                    param_lr = lr * 2.0  # 2x faster for metadata (they need to catch up)
                elif 'mlp' in name or 'edge' in name:
                    param_lr = lr * 0.5  # Slower for MLP
                else:
                    param_lr = lr

                # **Add weight decay directly to gradient**
                grad = param.grad.data
                if self.weight_decay > 0 and 'emb' in name:
                    grad = grad.add(param.data, alpha=self.weight_decay)

                # **Pure SGD update**
                param.data.add_(grad, alpha=-param_lr)

    def train(self):
        print(f">>> starting training")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.model.to(self.device)

        best_loss = float('inf')
        patience = 0
        max_patience = 5

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()

            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            num_batches = 0

            current_lr = self._get_lr(epoch)

            progress = tqdm(self.loader, desc=f"Epoch {epoch} (LR={current_lr:.6f})", leave=True)

            for batch_idx, batch in enumerate(progress):
                u_idx, i_pos_idx, i_neg_idx, pos_weights, neg_weights, all_pos_list, all_neg_list = [
                    x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                ]

                # Build subgraph
                batch_nodes_set = set()
                for u, pos_items, neg_samples in zip(u_idx.tolist(), all_pos_list, i_neg_idx.tolist()):
                    nodes = [u] + pos_items.tolist() + neg_samples
                    batch_nodes_set.update(nodes)
                batch_nodes = torch.tensor(list(batch_nodes_set), device=self.device)

                batch_nodes_cpu = batch_nodes.cpu()
                edge_index_full_cpu = self.edge_index_full.cpu()
                edge_features_full_cpu = self.model.edge_features.cpu()

                # Pass edge_attr to subgraph()
                edge_index_sub, edge_features_sub = subgraph(
                    batch_nodes_cpu,
                    edge_index_full_cpu,
                    edge_attr=edge_features_full_cpu,  # <-- PASS FEATURES
                    relabel_nodes=True
                )

                # Move subgraph data to device
                edge_index_sub = edge_index_sub.to(self.device)
                edge_features_sub = edge_features_sub.to(self.device)  # <-- NEW

                # Forward pass
                x_sub, user_nodes_sub, item_nodes_sub = self.model.forward_subgraph(
                    batch_nodes,
                    edge_index_sub,
                    edge_features_sub  # <-- PASS TO MODEL
                )
                # Map nodes
                node_map = torch.full((int(batch_nodes.max()) + 1,), -1, device=self.device)
                node_map[batch_nodes] = torch.arange(len(batch_nodes), device=self.device)
                u_emb_sub = x_sub[node_map[u_idx]]
                pos_emb_sub = x_sub[node_map[i_pos_idx]]
                neg_emb_sub = x_sub[node_map[i_neg_idx]]

                # **DIAGNOSTIC: Print every 10 batches**
                # if batch_idx % 10 == 0:
                #     with torch.no_grad():
                #         item_audio = self.model.item_audio_emb[item_nodes_sub].to(self.device)
                #         artist_emb = self.model.artist_emb(self.model.artist_ids[item_nodes_sub].to(self.device))
                #         album_emb = self.model.album_emb(self.model.album_ids[item_nodes_sub].to(self.device))
                #
                #         # **Show scaled norms (what actually goes into the model)**
                #         audio_scaled = (item_audio * self.model.audio_scale).norm(dim=-1).mean()
                #         metadata_scaled = ((artist_emb + album_emb) * self.model.metadata_scale).norm(dim=-1).mean()
                #
                #         print(f"\nBatch {batch_idx} | "
                #               f"audio*scale: {audio_scaled:.4f}, "
                #               f"metadata*scale: {metadata_scaled:.4f}, "
                #               f"ratio: {metadata_scaled / audio_scaled:.2f}")

                # Compute loss
                loss = self._bpr_loss(u_emb_sub, pos_emb_sub, neg_emb_sub, pos_weights, neg_weights)

                # **Scale loss for gradient accumulation**
                loss = loss / self.accum_steps

                # Backward pass
                loss.backward()

                # **Update only every accum_steps batches**
                if (batch_idx + 1) % self.accum_steps == 0:
                    # Clip gradients
                    grad_norm = self._clip_and_get_norm()

                    # Get current LR
                    self.step_count += 1
                    step_lr = self._get_lr(epoch)

                    # Update parameters
                    self._update_parameters(step_lr)

                    # Zero gradients
                    self.model.zero_grad(set_to_none=True)

                    epoch_grad_norm += grad_norm

                # Track metrics (unscaled loss for reporting)
                epoch_loss += loss.item() * self.accum_steps
                num_batches += 1

                progress.set_postfix({
                    'loss': f'{epoch_loss / num_batches:.4f}',
                })

            # Clear any remaining gradients
            self.model.zero_grad(set_to_none=True)

            progress.close()
            avg_loss = epoch_loss / num_batches
            avg_grad = epoch_grad_norm / (num_batches / self.accum_steps)

            print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Avg grad: {avg_grad:.4f}")

            # Save best model
            # if avg_loss < best_loss:
            #     improvement = best_loss - avg_loss
            #     best_loss = avg_loss
            #     patience = 0
            #     torch.save(self.model.state_dict(), self.save_path)
            #     print(f"✓ Best model saved (↓{improvement:.6f})")
            # else:
            #     patience += 1
            #     print(f"No improvement ({patience}/{max_patience})")
            #     if patience >= max_patience:
            #         print(f"Early stopping at epoch {epoch}")
            #         break

        print(f"\n >>> finished training")
        # print(f"Best loss: {best_loss:.6f}")
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")
