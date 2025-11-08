import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.data import HeteroData
from config import Config


class EdgeWeightMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.gnn.edge_mlp_input_dim, config.gnn.edge_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gnn.edge_mlp_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, edge_features):
        return self.mlp(edge_features).squeeze()


class LightGCN(nn.Module):
    def __init__(self, data: HeteroData, config: Config):
        print('>>> starting GNN init')
        super().__init__()
        self.config = config
        self.num_layers = config.gnn.num_layers
        self.lambda_align = config.gnn.lambda_align
        self.embed_dim = config.gnn.embed_dim

        # Node counts
        self.num_users = data['user'].num_nodes
        self.num_items = data['item'].num_nodes

        # **FIX 1: Initialize embeddings with smaller scale**
        self.user_emb = nn.Embedding(self.num_users, self.embed_dim)
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)

        # Keep on CPU, move to GPU only when needed
        self.register_buffer('item_audio_emb', data['item'].x.cpu())
        self.register_buffer('artist_ids', data['item'].artist_id.cpu())
        self.register_buffer('album_ids', data['item'].album_id.cpu())
        self.register_buffer('user_original_ids', data['user'].uid.cpu())
        self.register_buffer('item_original_ids', data['item'].item_id.cpu())

        num_artists = self.artist_ids.max().item() + 1
        num_albums = self.album_ids.max().item() + 1
        self.artist_emb = nn.Embedding(num_artists, self.embed_dim)
        self.album_emb = nn.Embedding(num_albums, self.embed_dim)

        # **FIX 2: Better initialization for metadata embeddings**
        nn.init.xavier_uniform_(self.artist_emb.weight)
        nn.init.xavier_uniform_(self.album_emb.weight)

        # **FIX 3: Optimal scaling from diagnostic (30% audio, 70% metadata)**
        self.audio_scale = 0.30  # Audio contributes 30%
        self.metadata_scale = 0.44  # Metadata contributes 70%

        # **FIX 4: Optional: Add projection layer for audio (comment out if OOM)**
        # audio_dim = data['item'].x.shape[1]
        # self.audio_proj = nn.Linear(audio_dim, self.embed_dim)
        # nn.init.xavier_uniform_(self.audio_proj.weight)
        self.audio_proj = None  # Set to None to save memory

        # Edge features
        self.register_buffer('edge_index', data['user', 'interacts', 'item'].edge_index.cpu())
        self.register_buffer('edge_attr', data['user', 'interacts', 'item'].edge_attr.cpu())
        self.register_buffer('edge_weight_init', data['user', 'interacts', 'item'].edge_weight_init.cpu())
        edge_features = torch.cat([self.edge_attr, self.edge_weight_init.unsqueeze(1)], dim=1)
        self.register_buffer('edge_features', edge_features.cpu())

        self.edge_mlp = EdgeWeightMLP(config)
        self.convs = nn.ModuleList([LGConv() for _ in range(self.num_layers)])

        print(">>> finished GNN init")

    def _get_item_embeddings(self, item_nodes, device):
        """
        Combine audio + metadata embeddings with fixed scaling
        """
        # Get raw audio (keep frozen)
        item_audio = self.item_audio_emb[item_nodes].to(device)

        # Optional projection (if self.audio_proj is not None)
        if self.audio_proj is not None:
            item_audio = self.audio_proj(item_audio)

        # Get metadata embeddings
        artist_emb = self.artist_emb(self.artist_ids[item_nodes].to(device))
        album_emb = self.album_emb(self.album_ids[item_nodes].to(device))

        # **FIX 6: Scale to balance frozen vs trainable**
        audio_part = item_audio * self.audio_scale
        metadata_part = (artist_emb + album_emb) * self.metadata_scale

        # Combine and normalize
        item_embed = audio_part + metadata_part
        item_embed = F.normalize(item_embed, p=2, dim=-1)

        return item_embed

    def forward(self, return_projections=False):
        """
        Full-graph forward (used for evaluation / saving final embeddings).
        """
        device = next(self.parameters()).device

        # Move edge data to device
        edge_index = self.edge_index.to(device)
        edge_features = self.edge_features.to(device)
        edge_weight = self.edge_mlp(edge_features)

        # Initial embeddings
        user_nodes = torch.arange(self.num_users, device=device)
        item_nodes = torch.arange(self.num_items, device=device)

        user_embed = F.normalize(self.user_emb(user_nodes), p=2, dim=-1)
        item_embed = self._get_item_embeddings(item_nodes, device)

        # Concatenate users + items
        x = torch.cat([user_embed, item_embed], dim=0)

        # LightGCN propagation
        all_emb = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            all_emb.append(x)

        # **FIX 7: Average all layer outputs (standard LightGCN)**
        x = torch.stack(all_emb, dim=0).mean(dim=0)

        # Normalize final embeddings
        x = F.normalize(x, p=2, dim=-1)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]

        align_loss = torch.tensor(0.0, device=device)
        return user_emb, item_emb, align_loss

    def forward_subgraph(self, batch_nodes, edge_index_sub, edge_features_sub):
        """
        Forward pass on a subgraph with proper embedding combination
        """
        device = next(self.parameters()).device
        batch_nodes = batch_nodes.to(device)
        edge_index_sub = edge_index_sub.to(device)
        edge_weight_sub = self.edge_mlp(edge_features_sub)

        # Identify users vs items
        user_mask = batch_nodes < self.num_users
        item_mask = ~user_mask

        user_nodes = batch_nodes[user_mask]
        item_nodes = batch_nodes[item_mask] - self.num_users

        # Get embeddings
        user_embed = F.normalize(self.user_emb(user_nodes), p=2, dim=-1)
        item_embed = self._get_item_embeddings(item_nodes, device)

        # Concatenate in subgraph order
        x_sub = torch.zeros((len(batch_nodes), self.embed_dim), device=device)
        x_sub[user_mask] = user_embed
        x_sub[item_mask] = item_embed

        # **FIX 8: Propagate with layer averaging**
        all_emb = [x_sub]
        for conv in self.convs:
            x_sub = conv(x_sub, edge_index_sub, edge_weight=edge_weight_sub)  # <-- USE WEIGHTS
            all_emb.append(x_sub)

        x_sub = torch.stack(all_emb, dim=0).mean(dim=0)

        # Normalize final embeddings
        x_sub = F.normalize(x_sub, p=2, dim=-1)

        return x_sub, user_nodes, item_nodes