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


class RegressionDataset(Dataset):
    """
    Dataset for regression.
    __getitem__ returns all interactions for a single user.
    """

    def __init__(self, train_graph, config: Config):
        # We need the user-centric view: user -> list of items and scores
        edge_index = train_graph['user', 'interacts', 'item'].edge_index
        edge_weights = train_graph['user', 'interacts', 'item'].edge_weight_init

        user_ids_all = edge_index[0].cpu().numpy()
        item_ids_all = edge_index[1].cpu().numpy()
        scores_all = edge_weights.cpu().numpy()

        self.user_data = defaultdict(lambda: {'items': [], 'scores': []})
        for user_id, item_id, score in zip(user_ids_all, item_ids_all, scores_all):
            self.user_data[user_id]['items'].append(item_id)
            self.user_data[user_id]['scores'].append(score)

        self.users = list(self.user_data.keys())

        # Get node counts from the graph
        self.num_users = train_graph['user'].num_nodes
        self.num_items = train_graph['item'].num_nodes

        print(f"RegressionDataset: Loaded {len(self.users)} users with {len(scores_all)} total interactions.")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        data = self.user_data[user_id]

        # item_ids will be offset by num_users in the collate fn
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(data['items'], dtype=torch.long),
            torch.tensor(data['scores'], dtype=torch.float)
        )


def collate_regression(batch):
    """
    Collates a batch of user-centric data.
    """
    users = [b[0] for b in batch]
    item_lists = [b[1] for b in batch]
    score_lists = [b[2] for b in batch]

    return users, item_lists, score_lists


class GNNTrainer:
    """
    User-centric regression trainer.
    """

    def __init__(self, model: LightGCN, train_graph, config: Config):
        self.config = config
        self.device = config.gnn.device
        self.model = model.to(self.device)
        self.train_graph = train_graph
        self.batch_size = config.gnn.batch_size
        self.num_epochs = config.gnn.num_epochs
        self.save_path = config.paths.trained_gnn

        self.dataset = RegressionDataset(train_graph, config)

        # --- DATALOADER SPEEDUP FIXES ---
        pin_memory = 'cuda' in str(self.device)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.gnn.num_workers,
            collate_fn=collate_regression,
            pin_memory=pin_memory,
            persistent_workers=True if config.gnn.num_workers > 0 else False
        )
        # --- END FIXES ---

        # Get data from the *model* buffers, which are already homogeneous
        self.edge_index_full = self.model.edge_index.cpu()
        self.edge_weight_full = self.model.edge_weight_init.cpu()

        self.num_users = self.model.num_users
        self.num_items = self.model.num_items
        self.total_nodes = self.num_users + self.num_items

        self.lr_base = config.gnn.lr
        self.lr_decay = config.gnn.lr_decay
        self.weight_decay = config.gnn.weight_decay
        self.max_grad_norm = config.gnn.max_grad_norm

        self.step_count = 0
        self.warmup_steps = 100
        self.accum_steps = config.gnn.accum_steps

        self.loss_fn = torch.nn.MSELoss()

    def _get_lr(self, epoch):
        """
        Learning rate schedule with warmup and decay.
        """
        if self.step_count == 0:  # Avoid division by zero on first step
            return self.lr_base * (1 / self.warmup_steps)

        if self.step_count < self.warmup_steps:
            return self.lr_base * (self.step_count / self.warmup_steps)

        # Exponential decay after warmup
        return self.lr_base * (self.lr_decay ** (epoch - (self.warmup_steps / len(self.loader))))

    def _update_parameters(self, lr):
        """
        Pure SGD update with per-parameter LR and weight decay.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue

                # **Different LR for different parameter types**
                if 'user_emb' in name:
                    param_lr = lr * 1.0
                elif 'artist_emb' in name or 'album_emb' in name:
                    param_lr = lr * 2.0
                else:
                    param_lr = lr

                # **Add weight decay directly to gradient**
                grad = param.grad.data
                if self.weight_decay > 0 and 'emb' in name:
                    grad = grad.add(param.data, alpha=self.weight_decay)

                # **Pure SGD update**
                param.data.add_(grad, alpha=-param_lr)

    def train(self, trial=False):
        print(f">>> starting training with REGRESSION loss")
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
                users, item_lists, score_lists = batch

                # --- 1. Build Subgraph ---
                batch_nodes_set = set(users)
                all_items_flat = []
                for items in item_lists:
                    # item_ids from dataset are 0 to num_items-1
                    # offset them to get global homogeneous ID
                    all_items_flat.extend(items.numpy() + self.num_users)

                batch_nodes_set.update(all_items_flat)
                batch_nodes = torch.tensor(list(batch_nodes_set), device='cpu')  # Subgraph on CPU

                edge_index_sub, edge_weight_sub = subgraph(
                    batch_nodes,
                    self.edge_index_full,
                    edge_attr=self.edge_weight_full,
                    relabel_nodes=True,
                    num_nodes=self.total_nodes
                )

                # Move subgraph data to device
                batch_nodes = batch_nodes.to(self.device)
                edge_index_sub = edge_index_sub.to(self.device)
                edge_weight_sub = edge_weight_sub.to(self.device)

                # --- 2. Forward Pass ---
                x_sub, _, _ = self.model.forward_subgraph(
                    batch_nodes,
                    edge_index_sub,
                    edge_weight_sub
                )

                # --- 3. Map nodes and Calculate Loss ---
                # Create a reverse map from global node_id -> subgraph_idx
                node_map = torch.full((int(batch_nodes.max()) + 1,), -1, device=self.device)
                node_map[batch_nodes] = torch.arange(len(batch_nodes), device=self.device)

                # Collect all (user, item, score) pairs from the batch
                pred_scores_list = []
                true_scores_list = []

                for u, items, scores in zip(users, item_lists, score_lists):
                    # Get subgraph indices
                    u_sub_idx = node_map[u]
                    i_sub_idx = node_map[items.to(self.device) + self.num_users]  # offset items

                    # Get embeddings
                    u_emb_sub = x_sub[u_sub_idx]
                    i_emb_sub = x_sub[i_sub_idx]

                    # Calculate dot product
                    pred_scores = (u_emb_sub * i_emb_sub).sum(dim=-1)

                    pred_scores_list.append(pred_scores)
                    true_scores_list.append(scores.to(self.device))

                pred_scores_all = torch.cat(pred_scores_list)
                true_scores_all = torch.cat(true_scores_list)

                # --- DIAGNOSTIC: Check for NaN/Inf before loss ---
                if not torch.isfinite(pred_scores_all).all():
                    print(f"\n!!! NaN/Inf detected in Pred scores at epoch {epoch}, batch {batch_idx}!!!")
                    # D-print norms
                    print(f"Pred scores norm: {pred_scores_all.norm().item()}")
                    print(f"True scores norm: {true_scores_all.norm().item()}")

                    with torch.no_grad():
                        base_user_emb_norm = self.model.user_emb.weight.norm().item()
                        base_item_emb_norm = self.model._get_item_embeddings(torch.arange(5, device=self.device),
                                                                             self.device).norm().item()
                        print(f"Base user emb norm: {base_user_emb_norm}")
                        print(f"Base item emb norm: {base_item_emb_norm}")

                    raise ValueError("NaN/Inf in predicted scores. Stopping training.")
                # --- End Diagnostic ---

                loss = self.loss_fn(pred_scores_all, true_scores_all)

                # Scale loss for gradient accumulation
                loss = loss / self.accum_steps
                loss.backward()

                # --- 4. Optimizer Step ---
                if (batch_idx + 1) % self.accum_steps == 0:
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                    # --- OOM/NaN Fix: Check the grad_norm *float* ---
                    if not np.isfinite(grad_norm.item()):
                        print(f"\n!!! NaN/Inf GRADIENT detected at epoch {epoch}, batch {batch_idx}!!!")
                        print(f"Grad norm was: {grad_norm.item()}")
                        self.model.zero_grad(set_to_none=True)
                        print("Skipping update for this step.")
                    else:
                        # Get current LR
                        self.step_count += 1
                        current_step_lr = self._get_lr(epoch)

                        # Update parameters
                        self._update_parameters(current_step_lr)

                        epoch_grad_norm += grad_norm.item()

                    # Zero gradients
                    self.model.zero_grad(set_to_none=True)

                # Track metrics (unscaled loss for reporting)
                epoch_loss += loss.item() * self.accum_steps
                num_batches += 1

                progress.set_postfix({
                    'mse_loss': f'{epoch_loss / num_batches:.6f}',
                })

            # Clear any remaining gradients
            self.model.zero_grad(set_to_none=True)

            progress.close()
            avg_loss = epoch_loss / num_batches
            avg_grad = epoch_grad_norm / (num_batches / self.accum_steps)

            print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Avg grad: {avg_grad:.4f}")

            if not trial:
                # Save best model
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