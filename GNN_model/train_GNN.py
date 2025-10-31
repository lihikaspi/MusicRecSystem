import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from GNN_model.GNN_class import LightGCN
from config import Config, PreprocessingConfig

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
        u = int(u)
        i_pos = int(i_pos)

        # fallback negative sampling (full graph) will override in trainer
        while True:
            i_neg = np.random.randint(0, self.num_items)
            if i_neg not in self.user2items[u]:
                break

        return torch.tensor(u, dtype=torch.long), \
               torch.tensor(i_pos, dtype=torch.long), \
               torch.tensor(i_neg, dtype=torch.long)


def collate_bpr(batch):
    u_idx = torch.stack([b[0] for b in batch])
    i_pos_idx = torch.stack([b[1] for b in batch])
    i_neg_idx = torch.stack([b[2] for b in batch])
    return u_idx, i_pos_idx, i_neg_idx


# ------------------------------
# Trainer
# ------------------------------
class GNNTrainer:
    def __init__(self, model: LightGCN, train_graph, config: Config):
        self.device = config.gnn.device
        self.model = model.to(self.device)
        self.train_graph = train_graph
        self.config = config
        self.batch_size = config.gnn.batch_size
        self.lr = config.gnn.lr

        # Build BPR dataset
        edge_index = train_graph['user', 'interacts', 'item'].edge_index
        user_idx_np, item_id_np = edge_index.cpu().numpy()
        train_edges = np.stack([user_idx_np.astype(np.int64), item_id_np.astype(np.int64)], axis=1)
        self.num_items = int(train_graph['item'].num_nodes)
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

    @staticmethod
    def _bpr_loss(u_emb, pos_emb, neg_emb):
        pos_score = (u_emb * pos_emb).sum(dim=1)
        neg_score = (u_emb * neg_emb).sum(dim=1)
        return F.softplus(-(pos_score - neg_score)).mean()

    def train(self):
        num_epochs = self.config.gnn.num_epochs
        edge_index_full = self.train_graph['user', 'interacts', 'item'].edge_index.to(self.device)
        edge_types_full = self.train_graph['user', 'interacts', 'item'].edge_attr[:, 0].to(self.device)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(self.loader, desc=f"Epoch {epoch}", leave=False)

            for batch in progress:
                # -------------------
                # Unpack batch
                # -------------------
                u_idx, i_pos_idx, _ = [x.to(self.device) for x in batch]
                batch_users = u_idx.cpu().numpy()

                # -------------------
                # Build subgraph nodes
                # -------------------
                batch_items_set = set()
                for u in batch_users:
                    batch_items_set.update(self.dataset.user2items[u])
                batch_items = torch.tensor(sorted(batch_items_set), device=self.device)

                # Combine users + items (items offset by num_users)
                batch_nodes = torch.cat([u_idx, batch_items + self.model.num_users])
                node_to_idx = {int(n.item() if isinstance(n, torch.Tensor) else n): i
                               for i, n in enumerate(batch_nodes)}

                # -------------------
                # Filter edges for subgraph
                # -------------------
                mask_user = torch.isin(edge_index_full[0], torch.tensor(batch_users, device=self.device))
                mask_item = torch.isin(edge_index_full[1], batch_items + self.model.num_users)
                mask_sub = mask_user & mask_item

                edge_index_sub = edge_index_full[:, mask_sub]
                edge_types_sub = edge_types_full[mask_sub]

                # -------------------
                # Forward pass on subgraph
                # -------------------
                x_sub = self.model.forward_subgraph(batch_nodes, edge_index_sub)

                # -------------------
                # Map batch users/items to local subgraph indices
                # -------------------
                u_list = u_idx.cpu().numpy().tolist()
                u_indices = torch.tensor([node_to_idx[i] for i in u_list],
                                         dtype=torch.long,
                                         device=self.device)

                # Convert to list of Python ints first
                i_pos_list = (i_pos_idx.cpu().numpy() + self.model.num_users).tolist()

                # Map to local subgraph indices
                i_pos_indices = torch.tensor([node_to_idx[i] for i in i_pos_list],
                                             dtype=torch.long,
                                             device=self.device)

                u_emb_sub = x_sub[u_indices]
                i_pos_sub = x_sub[i_pos_indices]

                # -------------------
                # Negative sampling from items in subgraph (dislikes/unlikes)
                # -------------------
                all_items_local = batch_items.cpu().numpy()
                neg_candidates = []
                for i, u in enumerate(batch_users):
                    pos_set = self.dataset.user2items[u]
                    # Only pick items not in positive set
                    sub_neg = [item for item in all_items_local if item not in pos_set]
                    if len(sub_neg) == 0:
                        # fallback: random from all items
                        sub_neg = np.random.choice(self.num_items, size=1).tolist()
                    neg_candidates.append(np.random.choice(sub_neg))

                i_neg_indices = torch.tensor(
                    [node_to_idx[n + self.model.num_users] for n in neg_candidates],
                    dtype=torch.long,
                    device=self.device
                )
                i_neg_sub = x_sub[i_neg_indices]

                # -------------------
                # Compute BPR loss & manual SGD
                # -------------------
                loss_bpr = self._bpr_loss(u_emb_sub, i_pos_sub, i_neg_sub)
                loss_bpr.backward()

                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param -= self.lr * param.grad
                    self.model.zero_grad(set_to_none=True)

                total_loss += loss_bpr.item()

            avg_loss = total_loss / len(self.loader)
            print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")


