import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.data import HeteroData
from config import EMBED_DIM, NUM_LAYERS, EDGE_MLP_INPUT_DIM, EDGE_MLP_HIDDEN_DIM

class EdgeWeightMLP(nn.Module):
    """
    MLP to compute edge weights from numeric edge attributes and initial weight.
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(EDGE_MLP_INPUT_DIM, EDGE_MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(EDGE_MLP_HIDDEN_DIM, 1),
            nn.Sigmoid()
        )

    def forward(self, edge_features):
        return self.mlp(edge_features).squeeze()


class LightGCN(nn.Module):
    def __init__(self, data: HeteroData, lambda_align: float = 0.0):
        """
        Args:
            data: HeteroData containing 'user' and 'item' nodes and 'interacts' edges
            lambda_align: weight for optional alignment loss between item embeddings and projected audio
        """
        super().__init__()
        self.num_layers = NUM_LAYERS
        self.lambda_align = lambda_align

        # ---------- Node embeddings ----------
        num_users = data['user'].user_id.max().item() + 1
        num_artists = data['item'].artist_id.max().item() + 1
        num_albums = data['item'].album_id.max().item() + 1

        self.user_emb = nn.Embedding(num_users, EMBED_DIM)
        self.artist_emb = nn.Embedding(num_artists, EMBED_DIM)
        self.album_emb = nn.Embedding(num_albums, EMBED_DIM)

        # Fixed audio embeddings
        self.register_buffer('item_audio_emb', data['item'].x)
        self.audio_proj = nn.Linear(data['item'].x.size(1), EMBED_DIM, bias=False)

        # ---------- Initialize embeddings ----------
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.artist_emb.weight)
        nn.init.xavier_uniform_(self.album_emb.weight)
        nn.init.xavier_uniform_(self.audio_proj.weight)

        # ---------- Edge MLP ----------
        self.edge_mlp = EdgeWeightMLP()

        # ---------- LGConv layers ----------
        self.convs = nn.ModuleList([LGConv() for _ in range(NUM_LAYERS)])

        # ---------- Precompute static components ----------
        self.register_buffer('edge_index', data['user', 'interacts', 'item'].edge_index)
        self.register_buffer('edge_attr', data['user', 'interacts', 'item'].edge_attr)
        self.register_buffer('edge_weight_init', data['user', 'interacts', 'item'].edge_weight_init)
        self.register_buffer('artist_ids', data['item'].artist_id)
        self.register_buffer('album_ids', data['item'].album_id)

        # Precompute edge features and adjusted edge indices
        edge_features = torch.cat([self.edge_attr, self.edge_weight_init.unsqueeze(1)], dim=1)
        self.register_buffer('edge_features', edge_features)

        # Adjust edge_index for concatenated tensor (user offset = 0, item offset = num_users)
        adjusted_edge_index = self.edge_index.clone()
        adjusted_edge_index[1] += num_users
        self.register_buffer('adjusted_edge_index', adjusted_edge_index)

        self.num_users = num_users

    def forward(self, return_projections=False):
        # ---------- Initial embeddings ----------
        user_h = self.user_emb.weight
        item_h = self.item_audio_emb + self.artist_emb(self.artist_ids) + self.album_emb(self.album_ids)

        # Optional projected audio for alignment (only compute if needed)
        projected_audio = None
        if self.lambda_align > 0 or return_projections:
            projected_audio = self.audio_proj(self.item_audio_emb)

        # ---------- Compute edge weights ----------
        edge_weight = self.edge_mlp(self.edge_features)

        # ---------- Concatenate for homogeneous LGConv ----------
        x = torch.cat([user_h, item_h], dim=0)

        # ---------- Propagation ----------
        xs = [x]
        for conv in self.convs:
            x = conv(x, self.adjusted_edge_index, edge_weight=edge_weight)
            xs.append(x)

        # Use learnable combination weights instead of simple mean
        x_final = torch.stack(xs, dim=0).mean(dim=0)  # Can be replaced with learnable weights

        final_user_h = x_final[:self.num_users]
        final_item_h = x_final[self.num_users:]

        # ---------- Optional alignment loss ----------
        align_loss = torch.tensor(0.0, device=x.device)
        if self.lambda_align > 0 and projected_audio is not None:
            align_loss = F.mse_loss(final_item_h, projected_audio)

        return final_user_h, final_item_h, align_loss
