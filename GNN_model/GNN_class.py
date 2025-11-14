import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_remaining_self_loops
from config import Config


def print_model_info(model):
    """
    Prints parameter counts and estimated memory footprint (float32).
    """
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    bytes_per_param = 4
    total_mb = (total_params * bytes_per_param) / (1024 ** 2)
    total_gb = (total_params * bytes_per_param) / (1024 ** 3)
    print(f"--- Model '{model.__class__.__name__}' Info ---")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Params: {total_params - trainable_params:,}")
    print(f"Estimated Size: {total_mb:.4f} MB | {total_gb:.6f} GB")


class LightGCN(nn.Module):
    """
    Patched version with:
    - Safer edge-weight mapping (keeps weights in [1e-6, 1]).
    - Much weaker self-loops (fill_value=1e-3) to avoid over-dominant identity signal.
    All other behavior is unchanged.
    """
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
        self.num_nodes_total = self.num_users + self.num_items

        self.user_emb = nn.Embedding(self.num_users, self.embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)

        # --- Sanitize Audio Embeddings ---
        raw_audio_emb = data['item'].x.cpu()
        if torch.isnan(raw_audio_emb).any() or torch.isinf(raw_audio_emb).any():
            print(">>> WARNING: NaNs or Infs detected in raw audio embeddings. Replacing with 0.")
            raw_audio_emb = torch.nan_to_num(raw_audio_emb, nan=0.0, posinf=0.0, neginf=0.0)
        self.register_buffer('item_audio_emb', raw_audio_emb)

        self.register_buffer('artist_ids', data['item'].artist_id.cpu())
        self.register_buffer('album_ids', data['item'].album_id.cpu())
        self.register_buffer('user_original_ids', data['user'].uid.cpu())
        self.register_buffer('item_original_ids', data['item'].item_id.cpu())

        num_artists = self.artist_ids.max().item() + 1
        num_albums = self.album_ids.max().item() + 1
        self.artist_emb = nn.Embedding(num_artists, self.embed_dim)
        self.album_emb = nn.Embedding(num_albums, self.embed_dim)
        nn.init.xavier_uniform_(self.artist_emb.weight)
        nn.init.xavier_uniform_(self.album_emb.weight)

        self.audio_scale = config.gnn.audio_scale
        self.metadata_scale = config.gnn.metadata_scale
        self.audio_proj = None  # keep None to save memory

        # --- Build homogeneous graph ---
        edge_index_bip = data['user', 'interacts', 'item'].edge_index.cpu()
        edge_w_init_bip = data['user', 'interacts', 'item'].edge_weight_init.cpu()

        # Sanitize edge weights
        if torch.isnan(edge_w_init_bip).any() or torch.isinf(edge_w_init_bip).any():
            print(">>> WARNING: NaNs/Infs in edge weights. Replacing with 0.")
            edge_w_init_bip = torch.nan_to_num(edge_w_init_bip, nan=0.0, posinf=0.0, neginf=0.0)

        # Forward (user->item) mapped into homogeneous index space
        fwd_edge_index = edge_index_bip.clone()
        fwd_edge_index[1] += self.num_users
        # Backward (item->user)
        bwd_edge_index = torch.stack([fwd_edge_index[1], fwd_edge_index[0]], dim=0)
        # Combine
        edge_index_full = torch.cat([fwd_edge_index, bwd_edge_index], dim=1)
        edge_weight_full = torch.cat([edge_w_init_bip, edge_w_init_bip], dim=0)

        # Map weights: keep positive, bounded; previous code used (w+1)*0.5 + 0.1
        # New mapping: (w+1)*0.5 -> [0,1], then clamp to [1e-6, 1]
        edge_weight_full = (edge_weight_full + 1.0) * 0.5
        edge_weight_full = torch.clamp(edge_weight_full, min=1e-6, max=1.0)

        self.register_buffer('edge_index', edge_index_full)
        self.register_buffer('edge_weight_init', edge_weight_full)

        # Convs (no self-loops added here; added on-the-fly in forward passes)
        self.convs = nn.ModuleList([LGConv(normalize=True) for _ in range(self.num_layers)])
        print('>>> finished GNN init')

    def _get_item_embeddings(self, item_nodes, device):
        item_nodes_cpu = item_nodes.cpu()
        item_audio = self.item_audio_emb[item_nodes_cpu].to(device)
        if self.audio_proj is not None:
            item_audio = self.audio_proj(item_audio)
        artist_ids_batch = self.artist_ids[item_nodes_cpu].to(device)
        album_ids_batch = self.album_ids[item_nodes_cpu].to(device)
        artist_emb = self.artist_emb(artist_ids_batch)
        album_emb = self.album_emb(album_ids_batch)
        audio_part = item_audio * self.audio_scale
        metadata_part = (artist_emb + album_emb) * self.metadata_scale
        item_embed = audio_part + metadata_part
        item_embed = F.normalize(item_embed, p=2, dim=-1, eps=1e-12)
        return item_embed

    def forward(self, return_projections: bool = False):
        device = next(self.parameters()).device
        edge_index = self.edge_index.to(device)
        edge_weight = self.edge_weight_init.to(device)

        # Weaker self-loops for stability without over-dominance
        edge_index_with_loops, edge_weight_with_loops = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1e-3,  # was 1.0
            num_nodes=self.num_nodes_total
        )

        user_nodes = torch.arange(self.num_users, device=device)
        item_nodes = torch.arange(self.num_items, device=device)
        user_embed = F.normalize(self.user_emb(user_nodes), p=2, dim=-1, eps=1e-12)
        item_embed = self._get_item_embeddings(item_nodes, device)
        x = torch.cat([user_embed, item_embed], dim=0)
        all_emb_sum = x

        for conv in self.convs:
            x = conv(x, edge_index_with_loops, edge_weight=edge_weight_with_loops)
            all_emb_sum = all_emb_sum + x
        x = all_emb_sum / (self.num_layers + 1)
        x = F.normalize(x, p=2, dim=-1, eps=1e-12)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]
        align_loss = torch.tensor(0.0, device=device)
        return user_emb, item_emb, align_loss

    def forward_cpu(self):
        torch.cuda.empty_cache()
        edge_weight_cpu = self.edge_weight_init.cpu()
        edge_index_cpu = self.edge_index.cpu()
        edge_index_with_loops, edge_weight_with_loops = add_remaining_self_loops(
            edge_index_cpu, edge_weight_cpu, fill_value=1e-3,  # was 1.0
            num_nodes=self.num_nodes_total
        )
        cpu_device = torch.device('cpu')
        param_device = next(self.parameters()).device

        user_embed = F.normalize(self.user_emb.weight, p=2, dim=-1, eps=1e-12).cpu()
        batch_size = 10000
        item_embeds = []
        for i in range(0, self.num_items, batch_size):
            end_idx = min(i + batch_size, self.num_items)
            batch_items_gpu = torch.arange(i, end_idx, device=param_device)
            item_embed_batch_gpu = self._get_item_embeddings(batch_items_gpu, param_device)
            item_embeds.append(item_embed_batch_gpu.cpu())
        item_embed = torch.cat(item_embeds, dim=0)
        del item_embeds
        torch.cuda.empty_cache()

        x = torch.cat([user_embed, item_embed], dim=0)
        all_emb_sum = x
        for conv in self.convs:
            x = conv(x, edge_index_with_loops, edge_weight=edge_weight_with_loops)
            all_emb_sum = all_emb_sum + x
        x = all_emb_sum / (self.num_layers + 1)
        x = F.normalize(x, p=2, dim=-1, eps=1e-12)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]
        align_loss_placeholder = torch.tensor(0.0, device=cpu_device)
        return user_emb, item_emb, align_loss_placeholder

    def forward_subgraph(self, batch_nodes, edge_index_sub, edge_weight_init_sub):
        device = next(self.parameters()).device
        batch_nodes = batch_nodes.to(device)
        edge_index_sub = edge_index_sub.to(device)
        edge_weight_sub = edge_weight_init_sub.to(device)

        # Weaker self-loops on the subgraph as well
        edge_index_sub_loops, edge_weight_sub_loops = add_remaining_self_loops(
            edge_index_sub, edge_weight_sub, fill_value=1e-3,  # was 1.0
            num_nodes=batch_nodes.size(0)
        )

        user_mask = batch_nodes < self.num_users
        item_mask = ~user_mask
        user_nodes = batch_nodes[user_mask]
        item_nodes = batch_nodes[item_mask] - self.num_users

        user_embed = F.normalize(self.user_emb(user_nodes), p=2, dim=-1, eps=1e-12)
        item_embed = self._get_item_embeddings(item_nodes, device)

        x_sub = torch.zeros((len(batch_nodes), self.embed_dim), device=device)
        x_sub[user_mask] = user_embed
        x_sub[item_mask] = item_embed

        all_emb = [x_sub]
        for conv in self.convs:
            x_sub = conv(x_sub, edge_index_sub_loops, edge_weight=edge_weight_sub_loops)
            all_emb.append(x_sub)
        x_sub = torch.stack(all_emb, dim=0).mean(dim=0)
        x_sub = F.normalize(x_sub, p=2, dim=-1, eps=1e-12)
        return x_sub, user_nodes, item_nodes
