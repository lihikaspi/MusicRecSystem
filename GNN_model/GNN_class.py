import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.data import HeteroData
from config import Config

class EdgeWeightMLP(nn.Module):
    """MLP to compute edge weights from numeric edge attributes and initial weight."""
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

        # Node embeddings
        num_users = data['user'].user_id.max().item() + 1
        num_artists = data['item'].artist_id.max().item() + 1
        num_albums = data['item'].album_id.max().item() + 1

        self.user_emb = nn.Embedding(num_users, self.embed_dim)
        self.artist_emb = nn.Embedding(num_artists, self.embed_dim)
        self.album_emb = nn.Embedding(num_albums, self.embed_dim)

        # Fixed audio embeddings
        self.register_buffer('item_audio_emb', data['item'].x)
        self.audio_proj = nn.Linear(data['item'].x.size(1), self.embed_dim, bias=False)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.artist_emb.weight)
        nn.init.xavier_uniform_(self.album_emb.weight)
        nn.init.xavier_uniform_(self.audio_proj.weight)

        # Edge MLP
        self.edge_mlp = EdgeWeightMLP(config)

        # LGConv layers
        self.convs = nn.ModuleList([LGConv() for _ in range(self.num_layers)])

        # Precompute static components
        self.register_buffer('edge_index', data['user', 'interacts', 'item'].edge_index)
        self.register_buffer('edge_attr', data['user', 'interacts', 'item'].edge_attr)
        self.register_buffer('edge_weight_init', data['user', 'interacts', 'item'].edge_weight_init)
        self.register_buffer('artist_ids', data['item'].artist_id)
        self.register_buffer('album_ids', data['item'].album_id)

        # Precompute edge features
        edge_features = torch.cat([self.edge_attr, self.edge_weight_init.unsqueeze(1)], dim=1)
        self.register_buffer('edge_features', edge_features)

        # Adjust edge_index for concatenated tensor (user offset = 0, item offset = num_users)
        adjusted_edge_index = self.edge_index.clone()
        adjusted_edge_index[1] += num_users
        self.register_buffer('adjusted_edge_index', adjusted_edge_index)

        self.num_users = num_users

        print(f"[DEBUG] adjusted_edge_index range: {adjusted_edge_index.min().item()} - {adjusted_edge_index.max().item()}")
        print(f"[DEBUG] total nodes (users + items): {num_users + data['item'].num_nodes}")

    def forward(self, return_projections=False):
        device = self.user_emb.weight.device

        # Node embeddings
        user_h = self.user_emb.weight
        item_h = self.item_audio_emb.to(device) + self.artist_emb(self.artist_ids) + self.album_emb(self.album_ids)

        # Optional projected audio for alignment
        projected_audio = None
        if self.lambda_align > 0 or return_projections:
            projected_audio = self.audio_proj(self.item_audio_emb.to(device))

        # Compute edge weights on the same device
        edge_weight = self.edge_mlp(self.edge_features).to(device)

        # Concatenate for homogeneous LGConv
        x = torch.cat([user_h, item_h], dim=0)

        # Propagation (memory-safe: keep only final layer)
        for conv in self.convs:
            x = conv(x, self.adjusted_edge_index, edge_weight=edge_weight)
        x_final = x

        final_user_h = x_final[:self.num_users]
        final_item_h = x_final[self.num_users:]

        # Optional alignment loss
        align_loss = torch.tensor(0.0, device=device)
        if self.lambda_align > 0 and projected_audio is not None:
            align_loss = F.mse_loss(final_item_h, projected_audio)

        return final_user_h, final_item_h, align_loss
