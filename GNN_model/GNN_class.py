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
        super().__init__()
        self.config = config
        self.num_layers = config.gnn.num_layers
        self.lambda_align = config.gnn.lambda_align
        self.embed_dim = config.gnn.embed_dim

        # Node counts
        self.num_users = data['user'].num_nodes
        self.num_items = data['item'].num_nodes

        # Embeddings
        self.user_emb = nn.Embedding(self.num_users, self.embed_dim)

        # Keep on CPU, move to GPU only when needed
        self.register_buffer('item_audio_emb', data['item'].x.cpu())
        self.register_buffer('artist_ids', data['item'].artist_id.cpu())
        self.register_buffer('album_ids', data['item'].album_id.cpu())

        num_artists = self.artist_ids.max().item() + 1
        num_albums = self.album_ids.max().item() + 1
        self.artist_emb = nn.Embedding(num_artists, self.embed_dim)
        self.album_emb = nn.Embedding(num_albums, self.embed_dim)

        self.audio_proj = nn.Linear(self.item_audio_emb.size(1), self.embed_dim, bias=False)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.artist_emb.weight)
        nn.init.xavier_uniform_(self.album_emb.weight)
        nn.init.xavier_uniform_(self.audio_proj.weight)

        # Edge features - keep on CPU
        self.register_buffer('edge_index', data['user', 'interacts', 'item'].edge_index.cpu())
        self.register_buffer('edge_attr', data['user', 'interacts', 'item'].edge_attr.cpu())
        self.register_buffer('edge_weight_init', data['user', 'interacts', 'item'].edge_weight_init.cpu())
        edge_features = torch.cat([self.edge_attr, self.edge_weight_init.unsqueeze(1)], dim=1)
        self.register_buffer('edge_features', edge_features.cpu())

        # Edge MLP - will compute weights on-the-fly instead of precomputing
        self.edge_mlp = EdgeWeightMLP(config)

        # LGConv layers
        self.convs = nn.ModuleList([LGConv() for _ in range(self.num_layers)])

    def _alignment_loss(self, u_emb, i_emb_pos):
        """
        Align user embeddings with their corresponding positive item embeddings.
        u_emb: [batch_size, embed_dim]
        i_emb_pos: [batch_size, embed_dim]  (only positive items)
        """
        if u_emb is None or i_emb_pos is None:
            return torch.tensor(0.0, device=u_emb.device if u_emb is not None else i_emb_pos.device)

        # Cosine similarity along embedding dimension
        cos_sim = F.cosine_similarity(u_emb, i_emb_pos, dim=-1)  # [batch_size]

        # Loss: maximize similarity → minimize (1 - cos_sim)
        return (1 - cos_sim).mean()

    def forward(self, user_idx=None, item_idx=None, pos_item_idx=None, return_projections=False):
        device = next(self.parameters()).device

        # Ensure edge_index and edge_weight are on the same device
        edge_index = self.edge_index.to(device)

        # Compute edge weights using MLP on-the-fly to save memory
        edge_features = self.edge_features.to(device)
        edge_weight = self.edge_mlp(edge_features)

        # For training: use mini-batch subgraph sampling to reduce memory
        if user_idx is not None and item_idx is not None:
            # Get unique nodes in batch
            batch_users = user_idx.unique()
            batch_items = item_idx.unique()

            # Filter edges: only keep edges where BOTH endpoints are in batch
            # More memory-efficient than torch.isin for large graphs
            user_mask = (edge_index[0].unsqueeze(1) == batch_users.unsqueeze(0)).any(dim=1)
            item_mask = (edge_index[1].unsqueeze(1) == (batch_items + self.num_users).unsqueeze(0)).any(dim=1)
            mask = user_mask & item_mask

            edge_index_sub = edge_index[:, mask]
            edge_weight_sub = edge_weight[mask] if edge_weight is not None else None
        else:
            # Full graph (for evaluation) - process all edges
            edge_index_sub = edge_index
            edge_weight_sub = edge_weight

        # Compute embeddings
        user_embed = self.user_emb.weight

        # Compute item embeddings on-the-fly instead of using cache
        item_audio = self.item_audio_emb.to(device)
        artist_ids = self.artist_ids.to(device)
        album_ids = self.album_ids.to(device)
        item_embed = item_audio + self.artist_emb(artist_ids) + self.album_emb(album_ids)
        item_embed = self.audio_proj(item_embed)

        x = torch.cat([user_embed, item_embed], dim=0)

        # LGConv layers
        for conv in self.convs:
            x = conv(x, edge_index_sub, edge_weight=edge_weight_sub)
        x = F.normalize(x, p=2, dim=1)

        # Slice embeddings for batch users and items
        u_emb = x[user_idx] if user_idx is not None else None
        i_emb = x[self.num_users + item_idx] if item_idx is not None else None

        # Alignment loss only on positive items
        if return_projections and u_emb is not None and pos_item_idx is not None:
            pos_emb = x[self.num_users + pos_item_idx]
            align_loss = (1 - F.cosine_similarity(u_emb, pos_emb, dim=-1)).mean()
        else:
            align_loss = torch.tensor(0.0, device=device)

        return u_emb, i_emb, align_loss