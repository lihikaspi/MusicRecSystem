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
            nn.Linear(EDGE_MLP_HIDDEN_DIM, 1)
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
        # Optional linear projection to GCN space
        self.audio_proj = nn.Linear(data['item'].x.size(1), EMBED_DIM, bias=False)

        # ---------- Initialize embeddings ----------
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.artist_emb.weight, std=0.1)
        nn.init.normal_(self.album_emb.weight, std=0.1)

        # ---------- Edge MLP ----------
        self.edge_mlp = EdgeWeightMLP()

        # ---------- LGConv layers ----------
        self.convs = nn.ModuleList([LGConv() for _ in range(NUM_LAYERS)])

        # ---------- Store graph ----------
        self.edge_index = data['user', 'interacts', 'item'].edge_index
        self.edge_attr = data['user', 'interacts', 'item'].edge_attr
        self.edge_weight_init = data['user', 'interacts', 'item'].edge_weight_init
        self.artist_ids = data['item'].artist_id
        self.album_ids = data['item'].album_id

    def forward(self):
        # ---------- Initial embeddings ----------
        user_h = self.user_emb.weight
        item_h = self.item_audio_emb + self.artist_emb[self.artist_ids] + self.album_emb[self.album_ids]

        # Optional projected audio for alignment
        projected_audio = self.audio_proj(self.item_audio_emb)

        # ---------- Compute edge weights ----------
        edge_features = torch.cat([self.edge_attr, self.edge_weight_init.unsqueeze(1)], dim=1)
        edge_weight = self.edge_mlp(edge_features)

        # ---------- Concatenate for homogeneous LGConv ----------
        x = torch.cat([user_h, item_h], dim=0)
        u_offset = 0
        i_offset = user_h.size(0)

        # Adjust edge_index for concatenated tensor
        edge_index = self.edge_index.clone()
        # user indices are fine, item indices need offset
        edge_index[1] += i_offset

        # ---------- Propagation ----------
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            xs.append(x)
        x_final = torch.stack(xs, dim=0).mean(dim=0)

        final_user_h = x_final[u_offset:i_offset]
        final_item_h = x_final[i_offset:]

        # ---------- Optional alignment loss ----------
        align_loss = F.mse_loss(final_item_h, projected_audio) if self.lambda_align > 0 else torch.tensor(0.0, device=x.device)

        return final_user_h, final_item_h, align_loss
