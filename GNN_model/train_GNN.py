import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
from torch_geometric.utils import subgraph
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import Config
from collections import defaultdict


class BPRDataset(Dataset):
    """
    Advanced BPR Dataset that samples "hard" negatives (dislikes)
    and "easy" negatives (random unseen).
    """
    def __init__(self, train_graph, config: Config):
        print(">>> Initializing Advanced BPRDataset...")
        edge_index = train_graph['user', 'interacts', 'item'].edge_index
        edge_attr = train_graph['user', 'interacts', 'item'].edge_attr

        user_idx, item_idx = edge_index.cpu()
        edge_types = edge_attr[:, 0].cpu()

        self.user2pos = {}
        self.user2pos_types = {}
        self.user2neg = {}
        self.num_items = int(train_graph['item'].num_nodes)
        self.all_users_pos_sets = defaultdict(set)

        self.listen_weight = config.gnn.listen_weight
        self.neutral_neg_weight = config.gnn.neutral_neg_weight
        self.neg_samples_per_pos = config.gnn.neg_samples_per_pos
        self.edge_type_mapping = config.preprocessing.edge_type_mapping

        pos_types = [self.edge_type_mapping[k] for k in ["listen", "like", "undislike"]]
        neg_types = [self.edge_type_mapping[k] for k in ["dislike", "unlike"]]

        print("    Building user-to-item positive/negative maps...")
        for u, i, t in zip(user_idx.tolist(), item_idx.tolist(), edge_types.tolist()):
            t_int = int(round(t))
            if t_int in pos_types:
                self.user2pos.setdefault(u, []).append(i)
                self.user2pos_types.setdefault(u, []).append(t_int)
                self.all_users_pos_sets[u].add(i)
            elif t_int in neg_types:
                self.user2neg.setdefault(u, []).append(i)

        print("    Cleaning hard negative lists...")
        for u in self.user2neg:
            self.user2neg[u] = list(set(self.user2neg[u]) - self.all_users_pos_sets[u])

        self.users = sorted(list(self.user2pos.keys()))
        print(f"Advanced BPRDataset: Loaded {len(self.users)} users with positive interactions.")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos_items = self.user2pos[u]
        pos_types = self.user2pos_types[u]
        hard_negs_list = self.user2neg.get(u, []).copy()
        known_pos_set = self.all_users_pos_sets[u]

        i_pos_idx = np.random.randint(len(pos_items))
        i_pos = pos_items[i_pos_idx]
        edge_type_of_pos = pos_types[i_pos_idx]
        pos_weight = self.listen_weight if edge_type_of_pos == self.edge_type_mapping["listen"] else 1.0

        neg_samples = []
        neg_weights = []
        for _ in range(self.neg_samples_per_pos):
            if len(hard_negs_list) > 0:
                i_neg_idx = np.random.randint(len(hard_negs_list))
                i_neg = hard_negs_list.pop(i_neg_idx)
                weight = 1.0
            else:
                while True:
                    i_neg = np.random.randint(self.num_items)
                    if i_neg not in known_pos_set:
                        break
                weight = self.neutral_neg_weight
            neg_samples.append(i_neg)
            neg_weights.append(weight)

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i_pos, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long),  # [k]
            torch.tensor(pos_weight, dtype=torch.float),  # scalar
            torch.tensor(neg_weights, dtype=torch.float)  # [k]
        )


def collate_bpr_advanced(batch):
    u_idx = torch.stack([b[0] for b in batch])
    i_pos_idx = torch.stack([b[1] for b in batch])
    i_neg_idx = torch.stack([b[2] for b in batch])  # [batch_size, k]
    pos_weights = torch.stack([b[3] for b in batch])
    neg_weights = torch.stack([b[4] for b in batch])  # [batch_size, k]
    return u_idx, i_pos_idx, i_neg_idx, pos_weights, neg_weights


class GNNTrainer:
    """Trainer for BPR loss (patched: lower dropout, constant LR for sanity)."""
    def __init__(self, model: LightGCN, train_graph, config: Config):
        self.config = config
        self.device = config.gnn.device
        self.model = model.to(self.device)
        self.train_graph = train_graph
        self.batch_size = config.gnn.batch_size
        self.num_epochs = config.gnn.num_epochs
        self.save_path = config.paths.trained_gnn
        self.neg_samples_per_pos = config.gnn.neg_samples_per_pos

        self.dataset = BPRDataset(train_graph, config)
        self.num_users = self.model.num_users
        self.num_items = self.dataset.num_items
        self.total_nodes = self.num_users + self.num_items

        pin_memory = 'cuda' in str(self.device)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.gnn.num_workers,
            collate_fn=collate_bpr_advanced,
            pin_memory=pin_memory,
            persistent_workers=True if config.gnn.num_workers > 0 else False
        )

        self.edge_index_full = self.model.edge_index.cpu()
        self.edge_weight_full = self.model.edge_weight_init.cpu()

        self.lr_base = config.gnn.lr
        self.lr_decay = config.gnn.lr_decay
        self.weight_decay = config.gnn.weight_decay
        self.max_grad_norm = config.gnn.max_grad_norm

        self.step_count = 0
        self.warmup_steps = len(self.loader)
        self.accum_steps = config.gnn.accum_steps

    def _get_lr(self, epoch):
        """
        Patched for sanity: use a constant LR (no warmup/cosine) to verify learning.
        Revert to your original scheduler once metrics climb.
        """
        return self.lr_base

    def _update_parameters(self, lr):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                if 'user_emb' in name:
                    param_lr = lr * 1.0
                elif 'artist_emb' in name or 'album_emb' in name:
                    param_lr = lr * 2.0
                else:
                    param_lr = lr
                grad = param.grad.data
                if self.weight_decay > 0 and 'emb' in name:
                    grad = grad.add(param.data, alpha=self.weight_decay)
                param.data.add_(grad, alpha=-param_lr)

    def train(self, trial=False):
        print(f">>> starting training with ADVANCED BPR ranking loss (patched dropout/LR)")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.model.to(self.device)

        best_ndcg = 0.0
        best_metrics = None
        patience = 0
        max_patience = 10

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            num_batches = 0
            current_lr = self._get_lr(epoch)
            progress = tqdm(self.loader, desc=f"Epoch {epoch} (LR={current_lr:.6f})", leave=True)

            for batch_idx, batch in enumerate(progress):
                u_idx, i_pos_idx, i_neg_idx, pos_weights, neg_weights = batch

                # --- Build Subgraph ---
                batch_nodes_set = set(u_idx.numpy())
                batch_nodes_set.update(i_pos_idx.numpy() + self.num_users)
                batch_nodes_set.update(i_neg_idx.numpy().flatten() + self.num_users)
                batch_nodes = torch.tensor(list(batch_nodes_set), device='cpu')

                edge_index_sub, edge_weight_sub = subgraph(
                    batch_nodes,
                    self.edge_index_full,
                    edge_attr=self.edge_weight_full,
                    relabel_nodes=True,
                    num_nodes=self.total_nodes
                )

                # PATCH: Lower edge dropout to 0.05
                if self.model.training:
                    dropout_rate = 0.05  # was 0.2
                    keep_mask = (torch.rand(edge_index_sub.size(1), device=edge_index_sub.device) > dropout_rate)
                    edge_index_sub = edge_index_sub[:, keep_mask]
                    edge_weight_sub = edge_weight_sub[keep_mask]

                batch_nodes = batch_nodes.to(self.device)
                edge_index_sub = edge_index_sub.to(self.device)
                edge_weight_sub = edge_weight_sub.to(self.device)

                x_sub, _, _ = self.model.forward_subgraph(
                    batch_nodes,
                    edge_index_sub,
                    edge_weight_sub
                )

                node_map = torch.full((int(batch_nodes.max()) + 1,), -1, device=self.device)
                node_map[batch_nodes] = torch.arange(len(batch_nodes), device=self.device)

                u_idx_gpu = u_idx.to(self.device)
                i_pos_idx_gpu = i_pos_idx.to(self.device)
                i_neg_idx_gpu = i_neg_idx.to(self.device)
                pos_weights_gpu = pos_weights.to(self.device)
                neg_weights_gpu = neg_weights.to(self.device)

                u_sub_idx = node_map[u_idx_gpu]
                pos_i_sub_idx = node_map[i_pos_idx_gpu + self.num_users]
                neg_i_sub_idx = node_map[i_neg_idx_gpu.flatten() + self.num_users].view_as(i_neg_idx_gpu)

                u_emb = x_sub[u_sub_idx]
                pos_i_emb = x_sub[pos_i_sub_idx]
                neg_i_emb = x_sub[neg_i_sub_idx]

                # PATCH: Lower embedding dropout to 0.05
                if self.model.training:
                    u_emb = F.dropout(u_emb, p=0.05, training=True)      # was 0.2
                    pos_i_emb = F.dropout(pos_i_emb, p=0.05, training=True)  # was 0.2

                pos_scores = (u_emb * pos_i_emb).sum(dim=-1, keepdim=True)
                neg_scores = (u_emb.unsqueeze(1) * neg_i_emb).sum(dim=-1)
                diff = pos_scores - neg_scores
                loss_per_neg = -F.logsigmoid(diff)

                weighted_loss = loss_per_neg * pos_weights_gpu.unsqueeze(1) * neg_weights_gpu
                loss = weighted_loss.mean()

                if not torch.isfinite(loss):
                    print(f"\n!!! NaN/Inf detected in BPR loss at epoch {epoch}, batch {batch_idx}!!!")
                    raise ValueError("NaN/Inf in BPR loss. Stopping training.")

                loss = loss / self.accum_steps
                loss.backward()

                if (batch_idx + 1) % self.accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    if not np.isfinite(grad_norm.item()):
                        print(f"\n!!! NaN/Inf GRADIENT detected at epoch {epoch}, batch {batch_idx}!!!")
                        self.model.zero_grad(set_to_none=True)
                    else:
                        self.step_count += 1
                        current_step_lr = self._get_lr(epoch)
                        self._update_parameters(current_step_lr)
                        epoch_grad_norm += grad_norm.item()
                    self.model.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * self.accum_steps
                num_batches += 1
                progress.set_postfix({'bpr_loss': f'{epoch_loss / num_batches:.6f}'})

            self.model.zero_grad(set_to_none=True)
            progress.close()
            avg_loss = epoch_loss / num_batches
            avg_grad = epoch_grad_norm / (num_batches / self.accum_steps)
            print(f"Epoch {epoch} | BPR Loss: {avg_loss:.6f} | Avg grad: {avg_grad:.4f}")

            if not trial:
                self.model.eval()
                val_evaluator = GNNEvaluator(self.model, self.train_graph, "val", self.config)
                val_metrics = val_evaluator.evaluate()
                cur_ndcg = val_metrics['ndcg@k']
                print(f"Epoch {epoch} | NDCG@K: {cur_ndcg:.6f}")

                if cur_ndcg > best_ndcg:
                    improvement = cur_ndcg - best_ndcg
                    best_ndcg = cur_ndcg
                    best_metrics = val_metrics
                    patience = 0
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"> Best model saved ({improvement:.6f})")
                else:
                    patience += 1
                    print(f"No improvement ({patience}/{max_patience})")
                    if patience >= max_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        print(f"\n>>> finished training")
        if not trial:
            print(f"Best NDCG@K: {best_ndcg:.6f}")
            with open(self.config.paths.val_eval, "w") as f:
                json.dump(best_metrics, f, indent=4)
            print(f"Model saved to {self.save_path}")
