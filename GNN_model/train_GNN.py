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
# Dataset with weighted positives and multiple negatives
# ------------------------------
class BPRDataset(Dataset):
    def __init__(self, train_graph, config: Config):
        edge_index = train_graph['user', 'interacts', 'item'].edge_index
        edge_attr = train_graph['user', 'interacts', 'item'].edge_attr  # shape [E, F]

        user_idx, item_idx = edge_index
        edge_types = edge_attr[:, 0]  # float values

        self.user2pos = {}
        self.user2pos_types = {}  # store edge type of each positive
        self.user2neg = {}
        self.num_items = int(train_graph['item'].num_nodes)

        self.listen_weight = config.gnn.listen_weight
        self.neutral_neg_weight = config.gnn.neutral_neg_weight
        self.neg_samples_per_pos = config.gnn.neg_samples_per_pos
        self.edge_type_mapping = config.preprocessing.edge_type_mapping

        pos_types = [self.edge_type_mapping[k] for k in ["listen", "like", "undislike"]]
        neg_types = [self.edge_type_mapping[k] for k in ["dislike", "unlike"]]

        # Populate positives and negatives
        for u, i, t in zip(user_idx.tolist(), item_idx.tolist(), edge_types.tolist()):
            t_int = int(round(t))
            if t_int in pos_types:
                self.user2pos.setdefault(u, []).append(i)
                self.user2pos_types.setdefault(u, []).append(t_int)
            elif t_int in neg_types:
                self.user2neg.setdefault(u, []).append(i)

        # Remove positives from true negatives
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

        # Sample one positive
        i_pos_idx = np.random.randint(len(pos_items))
        i_pos = pos_items[i_pos_idx]
        edge_type_of_pos = pos_types[i_pos_idx]

        # Assign positive weight
        pos_weight = self.listen_weight if edge_type_of_pos == self.edge_type_mapping["listen"] else 1.0

        # Sample multiple negatives only from known negatives
        neg_samples = []
        neg_weights = []
        for _ in range(self.neg_samples_per_pos):
            if len(neg_items) > 0:
                i_neg = np.random.choice(neg_items)
                weight = 1.0
            else:
                # fallback: sample from positives to avoid empty
                i_neg = np.random.choice(pos_items)
                weight = self.neutral_neg_weight
            neg_samples.append(i_neg)
            neg_weights.append(weight)

        # Return sampled negatives instead of all possible negatives
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i_pos, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long),
            torch.tensor(pos_weight, dtype=torch.float),
            torch.tensor(neg_weights, dtype=torch.float),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long),  # use sampled negatives only
        )

def collate_bpr(batch):
    u_idx = torch.stack([b[0] for b in batch])
    i_pos_idx = torch.stack([b[1] for b in batch])
    i_neg_idx = torch.stack([b[2] for b in batch])  # shape [batch, neg_samples_per_pos]
    pos_weights = torch.stack([b[3] for b in batch])
    neg_weights = torch.stack([b[4] for b in batch])  # shape [batch, neg_samples_per_pos]
    all_pos = [b[5] for b in batch]
    all_neg = [b[6] for b in batch]
    return u_idx, i_pos_idx, i_neg_idx, pos_weights, neg_weights, all_pos, all_neg


# ------------------------------
# Trainer
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

        self.edge_index_full = train_graph['user', 'interacts', 'item'].edge_index.to(self.device)

    # Weighted BPR loss over multiple negatives
    def _bpr_loss(self, u_emb, pos_emb, neg_emb, pos_weight, neg_weight):
        # u_emb: [B, D], pos_emb: [B, D], neg_emb: [B, K, D]
        pos_score = (u_emb * pos_emb).sum(dim=-1, keepdim=True)  # [B, 1]
        neg_score = (u_emb.unsqueeze(1) * neg_emb).sum(dim=-1)  # [B, K]
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)  # [B, K]
        weighted_loss = loss * pos_weight.unsqueeze(1) * neg_weight  # broadcasting
        return weighted_loss.mean()

    def train(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(self.loader, desc=f"Epoch {epoch}", leave=True)

            for batch in progress:
                u_idx, i_pos_idx, i_neg_idx, pos_weights, neg_weights, all_pos_list, all_neg_list = [
                    x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                ]

                # -------------------
                # Subgraph nodes: batch users + positives + sampled negatives only
                # -------------------
                batch_nodes_set = set()
                for u, pos_items, neg_samples in zip(u_idx.tolist(), all_pos_list, i_neg_idx.tolist()):
                    nodes = [u] + pos_items.tolist() + neg_samples
                    batch_nodes_set.update(nodes)
                batch_nodes = torch.tensor(list(batch_nodes_set), device=self.device)

                # -------------------
                # Extract subgraph
                # -------------------
                edge_index_sub, _ = subgraph(batch_nodes, self.edge_index_full, relabel_nodes=True)

                # -------------------
                # Forward pass
                # -------------------
                x_sub = self.model.forward_subgraph(batch_nodes, edge_index_sub)
                if isinstance(x_sub, tuple):
                    x_sub = x_sub[0]
                elif isinstance(x_sub, dict):
                    x_sub = x_sub['x']

                # -------------------
                # Map nodes to local subgraph indices
                # -------------------
                node_map = torch.full((int(batch_nodes.max()) + 1,), -1, device=self.device)
                node_map[batch_nodes] = torch.arange(len(batch_nodes), device=self.device)

                u_emb_sub = x_sub[node_map[u_idx]]  # [B, D]
                pos_emb_sub = x_sub[node_map[i_pos_idx]]  # [B, D]
                neg_emb_sub = x_sub[node_map[i_neg_idx]]  # [B, K, D]

                with torch.no_grad():
                    pos_score = (u_emb_sub * pos_emb_sub).sum(dim=-1)  # [B]
                    neg_score = (u_emb_sub.unsqueeze(1) * neg_emb_sub).sum(dim=-1)  # [B, K]
                    fraction_pos_greater = (pos_score.unsqueeze(1) > neg_score).float().mean().item()
                    print(f"Batch fraction pos > neg: {fraction_pos_greater:.4f}")

                # -------------------
                # Loss + backprop
                # -------------------
                loss = self._bpr_loss(u_emb_sub, pos_emb_sub, neg_emb_sub, pos_weights, neg_weights)
                loss.backward()

                # -------------------
                # Manual SGD
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

        torch.save(self.model.state_dict(), self.save_path)
        print(f"Training finished. Model saved to {self.save_path}")

