import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.data import HeteroData
from config import Config


def print_model_info(model):
    """
    Prints the total number of parameters, trainable parameters,
    and estimated model size in megabytes (MB) and gigabytes (GB).

    This calculation assumes parameters are stored in float32 format (4 bytes per param).

    Args:
        model (torch.nn.Module): The PyTorch model.
    """

    total_params = 0
    trainable_params = 0

    # Iterate over all parameters in the model
    # self.parameters() is available because model is an nn.Module
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    # --- Calculate Model Size ---
    bytes_per_param = 4  # Assuming float32
    total_bytes = total_params * bytes_per_param
    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)

    # Print the information
    print(f"--- Model '{model.__class__.__name__}' Info (from __init__) ---")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Params: {total_params - trainable_params:,}")
    print("-----------------------------------")
    print(f"Estimated Size (float32):")
    print(f"  {total_mb:.4f} MB")
    print(f"  {total_gb:.6f} GB")
    print("-----------------------------------")


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
        self.audio_scale = config.gnn.audio_scale
        self.metadata_scale = config.gnn.metadata_scale

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

        print_model_info(self)

        print(">>> finished GNN init")

    def _get_item_embeddings(self, item_nodes, device):
        """
        Combine audio + metadata embeddings with fixed scaling.

        It is expected that 'item_nodes' and 'device' are the model's
        main device (e.g., 'cuda:0') and all internal buffers
        (self.artist_ids, etc.) and sub-modules also live on this device.
        """
        # 'item_nodes' is on 'device', so indexing a tensor on 'device' works.
        item_audio = self.item_audio_emb[item_nodes]

        # Optional projection
        if self.audio_proj is not None:
            item_audio = self.audio_proj(item_audio)

        # Get metadata embeddings
        # self.artist_ids and self.album_ids are assumed to be on 'device'
        artist_ids_batch = self.artist_ids[item_nodes]
        album_ids_batch = self.album_ids[item_nodes]

        # self.artist_emb and self.album_emb (nn.Modules) are on 'device'
        artist_emb = self.artist_emb(artist_ids_batch)
        album_emb = self.album_emb(album_ids_batch)

        # **FIX 6: Scale to balance frozen vs trainable**
        audio_part = item_audio * self.audio_scale
        metadata_part = (artist_emb + album_emb) * self.metadata_scale

        # Combine and normalize
        item_embed = audio_part + metadata_part
        item_embed = F.normalize(item_embed, p=2, dim=-1)

        # The returned tensor is on 'device' (e.g., 'cuda:0')
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

        # LightGCN propagation with memory-efficient accumulation

        # 'all_emb_sum' will accumulate the embeddings from all layers.
        # We start it with the initial embeddings (layer 0).
        all_emb_sum = x

        for conv in self.convs:
            # Propagate to get the next layer's embeddings
            x = conv(x, edge_index, edge_weight=edge_weight)

            # Add the new layer's output to the sum.
            # This is done in-place (or creates one temporary tensor)
            # instead of storing all previous tensors.
            all_emb_sum = all_emb_sum + x

        # We have (num_layers + 1) total layers (layer 0 + the GCN layers)
        # Get the mean by dividing the sum.
        x = all_emb_sum / (self.num_layers + 1)

        # Normalize final embeddings
        x = F.normalize(x, p=2, dim=-1)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]

        align_loss = torch.tensor(0.0, device=device)
        return user_emb, item_emb, align_loss

    def forward_cpu(self):
        """
        Performs the full-graph forward pass explicitly on the CPU.

        This method is designed to be memory-efficient and avoid OOM errors
        on the GPU, making it ideal for saving final embeddings.
        It replicates the logic from the standard `forward` method but ensures
        all large tensors and computations (especially propagation) happen on the CPU.
        """
        print("Starting forward pass on CPU...")
        torch.cuda.empty_cache()

        # --- 1. Compute Edge Weight (Hybrid GPU/CPU) ---
        param_device = next(self.parameters()).device

        edge_features_gpu = self.edge_features.to(param_device)
        edge_weight_cpu = self.edge_mlp(edge_features_gpu).cpu()
        del edge_features_gpu
        torch.cuda.empty_cache()
        print("Computed edge weights on GPU and moved to CPU.")

        # --- 2. Get Initial Embeddings (CPU) ---
        cpu_device = torch.device('cpu')

        user_embed = F.normalize(self.user_emb.weight, p=2, dim=-1).cpu()
        print(f"Loaded initial user embeddings to CPU. Shape: {user_embed.shape}")

        # Get initial item embeddings on CPU (batched)
        batch_size = 10000
        item_embeds = []
        print("Loading initial item embeddings (batches on GPU, results on CPU)...")
        for i in range(0, self.num_items, batch_size):
            end_idx = min(i + batch_size, self.num_items)

            # --- FIX: START ---
            # Create the batch indices and move them to the *GPU* (param_device)
            # for the lookup.
            batch_items_gpu = torch.arange(i, end_idx, device=param_device)

            # Call _get_item_embeddings on the GPU. This is fine,
            # as it's a small, batched operation.
            item_embed_batch_gpu = self._get_item_embeddings(batch_items_gpu, param_device)

            # IMPORTANT: Move the small batch *result* to the CPU for accumulation.
            item_embeds.append(item_embed_batch_gpu.cpu())
            # --- FIX: END ---

        # This concatenation now happens on the CPU
        item_embed = torch.cat(item_embeds, dim=0)
        del item_embeds, item_embed_batch_gpu, batch_items_gpu  # Free GPU memory
        torch.cuda.empty_cache()
        print(f"Loaded initial item embeddings to CPU. Shape: {item_embed.shape}")

        # Concatenate users + items on CPU
        x = torch.cat([user_embed, item_embed], dim=0)
        del user_embed, item_embed  # Free memory
        print(f"Concatenated initial embeddings. Shape: {x.shape}")

        # --- 3. LightGCN Propagation (CPU) ---

        edge_index_cpu = self.edge_index.to(cpu_device)

        all_emb_sum = x

        print("Starting GCN propagation on CPU...")
        for i, conv in enumerate(self.convs):
            print(f"  Processing layer {i + 1}/{self.num_layers}...")
            x = conv(x, edge_index_cpu, edge_weight=edge_weight_cpu)
            all_emb_sum = all_emb_sum + x
            print(f"  Finished layer {i + 1}.")

        # --- 4. Finalization (CPU) ---

        x = all_emb_sum / (self.num_layers + 1)
        del all_emb_sum  # Free memory
        print("Averaged layer embeddings.")

        # Normalize final embeddings
        x = F.normalize(x, p=2, dim=-1)
        print("Normalized final embeddings.")

        # Split into final user and item embeddings (still on CPU)
        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]

        print("CPU forward pass complete.")

        # Return a 3-tuple to match the eval function's expectation
        # In a real CPU pass, align_loss is not computed.
        align_loss_placeholder = torch.tensor(0.0, device=cpu_device)

        return user_emb, item_emb, align_loss_placeholder


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