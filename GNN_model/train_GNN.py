import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import subgraph
from GNN_model.GNN_class import LightGCN
from config import Config

# ------------------------------
# BPR Dataset
# ------------------------------
class BPRDataset(Dataset):
    def __init__(self, interactions, num_items):
        self.user_item = interactions
        self.num_items = int(num_items)
        self.user2items = {}
        for u, i in interactions:
            self.user2items.setdefault(int(u), set()).add(int(i))

    def __len__(self):
        return len(self.user_item)

    def __getitem__(self, idx):
        u, i_pos = self.user_item[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(i_pos, dtype=torch.long)

def collate_bpr(batch):
    u_idx = torch.stack([b[0] for b in batch])
    i_pos_idx = torch.stack([b[1] for b in batch])
    return u_idx, i_pos_idx

# ------------------------------
# InfoNCE Trainer
# ------------------------------
class GNNTrainer:
    def __init__(self, model: LightGCN, train_graph, config: Config):
        self.device = config.gnn.device
        self.model = model.to(self.device)
        self.train_graph = train_graph
        self.batch_size = config.gnn.batch_size
        self.lr = config.gnn.lr
        self.num_epochs = config.gnn.num_epochs
        self.save_path = config.paths.trained_gnn
        self.tau = config.gnn.tau

        # Build dataset
        edge_index = train_graph['user', 'interacts', 'item'].edge_index
        user_idx_np, item_idx_np = edge_index.cpu().numpy()
        train_edges = np.stack([user_idx_np.astype(np.int64), item_idx_np.astype(np.int64)], axis=1)
        self.num_items = int(train_graph['item'].num_nodes)
        self.num_users = int(train_graph['user'].num_nodes)
        self.dataset = BPRDataset(train_edges, self.num_items)

        pin_memory = 'cuda' in str(self.device)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.gnn.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_bpr
        )

        # Precompute user->items tensor on GPU
        max_items_per_user = max(len(v) for v in self.dataset.user2items.values())
        self.user2items_tensor = torch.full(
            (self.num_users, max_items_per_user),
            -1, dtype=torch.long, device=self.device
        )
        for u, items in self.dataset.user2items.items():
            items_list = list(items)
            self.user2items_tensor[u, :len(items_list)] = torch.tensor(items_list, device=self.device)

        # Full edge index on GPU
        self.edge_index_full = edge_index.to(self.device)

    @staticmethod
    def _infonce_loss(u_emb, i_emb, temperature):
        u_emb = F.normalize(u_emb, dim=1)
        i_emb = F.normalize(i_emb, dim=1)
        logits = torch.matmul(u_emb, i_emb.T) / temperature
        labels = torch.arange(u_emb.size(0), device=u_emb.device)
        return F.cross_entropy(logits, labels)

    def train(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(self.loader, desc=f"Epoch {epoch}", leave=True)

            for batch in progress:
                u_idx, i_pos_idx = [x.to(self.device) for x in batch]

                # -------------------
                # Vectorized batch items
                # -------------------
                batch_items = self.user2items_tensor[u_idx].flatten()
                batch_items = batch_items[batch_items >= 0].unique()

                # -------------------
                # Subgraph nodes (users + items, no offset)
                # -------------------
                batch_nodes = torch.cat([u_idx, batch_items]).unique()

                # -------------------
                # Extract subgraph
                # -------------------
                edge_index_sub, _ = subgraph(batch_nodes, self.edge_index_full, relabel_nodes=True)

                # -------------------
                # Forward pass
                # -------------------
                x_sub = self.model.forward_subgraph(batch_nodes, edge_index_sub)
                if isinstance(x_sub, tuple): x_sub = x_sub[0]
                elif isinstance(x_sub, dict): x_sub = x_sub['x']

                # -------------------
                # Map users/items to local subgraph indices
                # -------------------
                node_map = torch.full((int(batch_nodes.max()) + 1,), -1, device=self.device)
                node_map[batch_nodes] = torch.arange(len(batch_nodes), device=self.device)

                u_indices = node_map[u_idx]
                i_indices = node_map[i_pos_idx]  # no offset here
                u_emb_sub = torch.index_select(x_sub, 0, u_indices)
                i_emb_sub = torch.index_select(x_sub, 0, i_indices)

                # -------------------
                # Compute , NCE loss
                # -------------------
                loss = _infonce_loss(u_emb_sub, i_emb_sub, self.tau)
                loss.backward()

                # -------------------
                # Manual in-place SGD
                # -------------------
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.add_(param.grad, alpha=-self.lr)
                    self.model.zero_grad(set_to_none=True)

                total_loss += loss.item()
                progress.set_postfix({'loss': total_loss / (progress.n + 1)})

            progress.close()

            avg_loss = total_loss / len(self.loader)
            # print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

        # Save model
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Training finished. Model saved to {self.save_path}")
