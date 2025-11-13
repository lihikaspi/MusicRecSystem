import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import json
from torch_geometric.utils import subgraph
from GNN_model.GNN_class import LightGCN
from GNN_model.eval_GNN import GNNEvaluator
from config import Config
from collections import defaultdict


class BPRDataset(Dataset):
    """
    Advanced BPR Dataset that samples "hard" negatives (dislikes)
    and "easy" negatives (random unseen).

    This is based on your ...9b55a... version, adapted for the new graph.
    """

    def __init__(self, train_graph, config: Config):
        print(">>> Initializing Advanced BPRDataset...")
        edge_index = train_graph['user', 'interacts', 'item'].edge_index
        edge_attr = train_graph['user', 'interacts', 'item'].edge_attr

        user_idx, item_idx = edge_index.cpu()
        # edge_attr[:, 0] is the 'edge_type'
        edge_types = edge_attr[:, 0].cpu()

        self.user2pos = {}
        self.user2pos_types = {}
        self.user2neg = {}  # For "hard" negatives (dislikes)
        self.num_items = int(train_graph['item'].num_nodes)
        self.all_users_pos_sets = defaultdict(set)  # For "easy" negative sampling

        self.listen_weight = config.gnn.listen_weight
        self.neutral_neg_weight = config.gnn.neutral_neg_weight
        self.neg_samples_per_pos = config.gnn.neg_samples_per_pos
        self.edge_type_mapping = config.preprocessing.edge_type_mapping

        # Define positive and negative event types from config
        pos_types = [self.edge_type_mapping[k] for k in ["listen", "like", "undislike"]]
        neg_types = [self.edge_type_mapping[k] for k in ["dislike", "unlike"]]

        print("    Building user-to-item positive/negative maps...")
        for u, i, t in zip(user_idx.tolist(), item_idx.tolist(), edge_types.tolist()):
            t_int = int(round(t))
            if t_int in pos_types:
                self.user2pos.setdefault(u, []).append(i)
                self.user2pos_types.setdefault(u, []).append(t_int)
                self.all_users_pos_sets[u].add(i)  # Add to set for fast lookup
            elif t_int in neg_types:
                self.user2neg.setdefault(u, []).append(i)

        # Clean up hard negatives: remove any item that is *also* positive
        print("    Cleaning hard negative lists...")
        for u in self.user2neg:
            self.user2neg[u] = list(set(self.user2neg[u]) - self.all_users_pos_sets[u])

        # self.users is the list of users who have at least one positive item
        self.users = sorted(list(self.user2pos.keys()))
        print(f"Advanced BPRDataset: Loaded {len(self.users)} users with positive interactions.")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos_items = self.user2pos[u]
        pos_types = self.user2pos_types[u]

        # Get a mutable list of hard negatives for this user
        hard_negs_list = self.user2neg.get(u, []).copy()

        # Get the set of all positive items for this user for fast lookup
        known_pos_set = self.all_users_pos_sets[u]

        # --- 1. Sample one positive item ---
        i_pos_idx = np.random.randint(len(pos_items))
        i_pos = pos_items[i_pos_idx]
        edge_type_of_pos = pos_types[i_pos_idx]

        # Assign weight based on positive interaction type
        pos_weight = self.listen_weight if edge_type_of_pos == self.edge_type_mapping["listen"] else 1.0

        # --- 2. Sample K negative items ---
        neg_samples = []
        neg_weights = []
        for _ in range(self.neg_samples_per_pos):

            # First, try to sample a "hard" negative (explicit dislike)
            if len(hard_negs_list) > 0:
                i_neg_idx = np.random.randint(len(hard_negs_list))
                i_neg = hard_negs_list.pop(i_neg_idx)  # Sample without replacement
                weight = 1.0  # Hard negatives get full weight

            # If no hard negatives left, sample an "easy" negative (random unseen)
            else:
                while True:
                    # Sample a random item from the *entire catalog*
                    i_neg = np.random.randint(self.num_items)

                    # Keep sampling until we find one that is NOT positive
                    if i_neg not in known_pos_set:
                        break  # Found a valid random negative

                weight = self.neutral_neg_weight  # Easy negatives get reduced weight

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
    """
    Collates the output of the Advanced BPRDataset.
    """
    u_idx = torch.stack([b[0] for b in batch])
    i_pos_idx = torch.stack([b[1] for b in batch])
    i_neg_idx = torch.stack([b[2] for b in batch])  # [batch_size, k]
    pos_weights = torch.stack([b[3] for b in batch])
    neg_weights = torch.stack([b[4] for b in batch])  # [batch_size, k]

    return u_idx, i_pos_idx, i_neg_idx, pos_weights, neg_weights


class GNNTrainer:
    """
    Trainer for BPR loss, compatible with the *current* GNN_class.py (homogeneous graph, no EdgeWeightMLP).

    Merges logic from ...9b55a... (dataset) and ...125a... (model).
    """

    def __init__(self, model: LightGCN, train_graph, config: Config):
        self.config = config
        self.device = config.gnn.device
        self.model = model.to(self.device)
        self.train_graph = train_graph
        self.batch_size = config.gnn.batch_size
        self.num_epochs = config.gnn.num_epochs
        self.save_path = config.paths.trained_gnn
        self.neg_samples_per_pos = config.gnn.neg_samples_per_pos  # Used by dataset

        # --- Use the Advanced BPRDataset ---
        self.dataset = BPRDataset(train_graph, config)

        # Get node counts from model and dataset
        self.num_users = self.model.num_users
        self.num_items = self.dataset.num_items
        self.total_nodes = self.num_users + self.num_items

        # --- DATALOADER ---
        pin_memory = 'cuda' in str(self.device)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.gnn.num_workers,
            collate_fn=collate_bpr_advanced,  # Use the correct collate fn
            pin_memory=pin_memory,
            persistent_workers=True if config.gnn.num_workers > 0 else False
        )
        # --- END DATALOADER ---

        # Get data from the *model* buffers, which are already homogeneous
        # This is the ...125a... model structure
        self.edge_index_full = self.model.edge_index.cpu()
        self.edge_weight_full = self.model.edge_weight_init.cpu()  # <-- CRITICAL: Use the 1D weights

        self.lr_base = config.gnn.lr
        self.lr_decay = config.gnn.lr_decay
        self.weight_decay = config.gnn.weight_decay
        self.max_grad_norm = config.gnn.max_grad_norm

        self.step_count = 0
        self.warmup_steps = 100
        self.accum_steps = config.gnn.accum_steps

    def _get_lr(self, epoch):
        """
        Learning rate schedule with warmup and decay.
        """
        if self.step_count == 0:
            return self.lr_base * (1 / self.warmup_steps)

        if self.step_count < self.warmup_steps:
            return self.lr_base * (self.step_count / self.warmup_steps)

        return self.lr_base * (self.lr_decay ** (epoch - (self.warmup_steps / len(self.loader))))

    def _update_parameters(self, lr):
        """
        Pure SGD update with per-parameter LR and weight decay.
        """
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
        print(f">>> starting training with ADVANCED BPR ranking loss")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.model.to(self.device)

        best_ndcg = 0.0
        best_metrics = None
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
                # --- 1. Unpack batch ---
                u_idx, i_pos_idx, i_neg_idx, pos_weights, neg_weights = batch

                # --- 2. Build Subgraph ---
                # This logic is from Version 3 (compatible with homogeneous graph)
                batch_nodes_set = set(u_idx.numpy())

                # Add HOMOGENEOUS item indices (idx + num_users)
                batch_nodes_set.update(i_pos_idx.numpy() + self.num_users)
                batch_nodes_set.update(i_neg_idx.numpy().flatten() + self.num_users)

                batch_nodes = torch.tensor(list(batch_nodes_set), device='cpu')

                # --- CRITICAL FIX ---
                # Pass the 1D edge_weight_full, not edge_features
                edge_index_sub, edge_weight_sub = subgraph(
                    batch_nodes,
                    self.edge_index_full,
                    edge_attr=self.edge_weight_full,  # <-- Use 1D weights
                    relabel_nodes=True,
                    num_nodes=self.total_nodes
                )

                # Move subgraph data to device
                batch_nodes = batch_nodes.to(self.device)
                edge_index_sub = edge_index_sub.to(self.device)
                edge_weight_sub = edge_weight_sub.to(self.device)  # <-- This is the 1D weight tensor

                # --- 3. Forward Pass ---
                # Pass the 1D edge_weight_sub to the model
                x_sub, _, _ = self.model.forward_subgraph(
                    batch_nodes,
                    edge_index_sub,
                    edge_weight_sub  # <-- Pass 1D weights
                )

                # --- 4. Map nodes and Calculate BPR Loss ---
                node_map = torch.full((int(batch_nodes.max()) + 1,), -1, device=self.device)
                node_map[batch_nodes] = torch.arange(len(batch_nodes), device=self.device)

                # Move IDs to device
                u_idx_gpu = u_idx.to(self.device)
                i_pos_idx_gpu = i_pos_idx.to(self.device)
                i_neg_idx_gpu = i_neg_idx.to(self.device)  # [batch_size, k]
                pos_weights_gpu = pos_weights.to(self.device)
                neg_weights_gpu = neg_weights.to(self.device)

                # Get subgraph indices
                u_sub_idx = node_map[u_idx_gpu]
                pos_i_sub_idx = node_map[i_pos_idx_gpu + self.num_users]
                neg_i_sub_idx = node_map[i_neg_idx_gpu.flatten() + self.num_users].view_as(i_neg_idx_gpu)

                # Get embeddings
                u_emb = x_sub[u_sub_idx]  # [batch_size, dim]
                pos_i_emb = x_sub[pos_i_sub_idx]  # [batch_size, dim]
                neg_i_emb = x_sub[neg_i_sub_idx]  # [batch_size, k, dim]

                # --- Calculate Weighted BPR Loss ---
                pos_scores = (u_emb * pos_i_emb).sum(dim=-1, keepdim=True)  # [batch_size, 1]
                neg_scores = (u_emb.unsqueeze(1) * neg_i_emb).sum(dim=-1)  # [batch_size, k]

                diff = pos_scores - neg_scores

                loss_per_neg = -F.logsigmoid(diff)  # [batch_size, k]

                # Apply weights
                # pos_weights: [batch_size] -> [batch_size, 1]
                # neg_weights: [batch_size, k]
                weighted_loss = loss_per_neg * pos_weights_gpu.unsqueeze(1) * neg_weights_gpu

                loss = weighted_loss.mean()
                # --- End Loss Calculation ---

                if not torch.isfinite(loss):
                    print(f"\n!!! NaN/Inf detected in BPR loss at epoch {epoch}, batch {batch_idx}!!!")
                    raise ValueError("NaN/Inf in BPR loss. Stopping training.")

                loss = loss / self.accum_steps
                loss.backward()

                # --- 5. Optimizer Step ---
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

                progress.set_postfix({
                    'bpr_loss': f'{epoch_loss / num_batches:.6f}',
                })

            self.model.zero_grad(set_to_none=True)
            progress.close()
            avg_loss = epoch_loss / num_batches
            avg_grad = epoch_grad_norm / (num_batches / self.accum_steps)

            print(f"Epoch {epoch} | BPR Loss: {avg_loss:.6f} | Avg grad: {avg_grad:.4f}")

            # --- 6. Evaluation ---
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
                if not trial:
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