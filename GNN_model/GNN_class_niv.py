# GNN_class_niv.py
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv  # LightGCN convolution


class LightGCN(nn.Module):
    """
    LightGCN over a homogeneous bipartite graph:
      - Nodes 0...(U-1) = users
      - Nodes U...(U+I-1) = items
    Uses weightless LGConv layers; we do layer-wise embedding averaging.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 128,
        num_layers: int = 3,
        init_std: float = 0.1,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Trainable ID embeddings
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        # Init (LightGCN prefers relatively small variance)
        nn.init.normal_(self.user_emb.weight, std=init_std)
        nn.init.normal_(self.item_emb.weight, std=init_std)

        # LGConv has no parameters; multiple layers for higher-order neighbors
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def propagate(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns final (user_emb, item_emb) after LightGCN propagation & layer-avg.
        """
        x0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        xs = [x0]
        x = x0
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            xs.append(x)
        x_final = torch.stack(xs, dim=0).mean(dim=0)

        u_final = x_final[: self.num_users]
        i_final = x_final[self.num_users :]
        return u_final, i_final

    @torch.no_grad()
    def get_all_embeddings(
        self, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        return self.propagate(edge_index, edge_weight)

    def score(self, u_emb: torch.Tensor, i_emb: torch.Tensor) -> torch.Tensor:
        return (u_emb * i_emb).sum(dim=-1)


class LightGCNWithContent(LightGCN):
    """
    LightGCN + content :
      - A linear projection W_c maps fixed item content vectors to the model space.
      - We add an L2 alignment loss between graph-learned item embeddings and W_c(content).
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        item_content_dim: int,
        embed_dim: int = 128,
        num_layers: int = 3,
        init_std: float = 0.1,
        lambda_content: float = 0.1,
        freeze_content: bool = True,
    ):
        super().__init__(num_users, num_items, embed_dim, num_layers, init_std)
        self.lambda_content = lambda_content

        # Linear projection from content to the LightGCN embedding space
        self.content_proj = nn.Linear(item_content_dim, embed_dim, bias=False)

        # Optionally keep a buffer for item content (if you want to register once)
        self._frozen_content: Optional[torch.Tensor] = None
        self._freeze_content = freeze_content

    def register_item_content(self, item_content: torch.Tensor):
        """
        item_content: [num_items, item_content_dim], ideally normalized.
        If freeze_content=True, we store it as a non-trainable buffer.
        """
        if self._freeze_content:
            # Make sure it’s on the same device later
            self._frozen_content = item_content.detach().clone()
        else:
            # Keep None; you’ll pass content each forward() for flexibility
            self._frozen_content = None

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        batch_users: torch.Tensor,
        batch_pos_items: torch.Tensor,
        batch_neg_items: torch.Tensor,
        item_content: Optional[torch.Tensor] = None,
        sample_items_for_align: Optional[torch.Tensor] = None,
    ):
        """
        Performs one forward pass for BPR training + content alignment.

        Args:
          edge_index: [2, E] (homogeneous; items offset by +num_users)
          edge_weight: [E] or None
          batch_users: [B]
          batch_pos_items: [B]
          batch_neg_items: [B]
          item_content: [num_items, item_content_dim] or None (if registered)
          sample_items_for_align: optional [K] subset of item indices to compute alignment on
                                  (saves memory; defaults to all items if None)

        Returns:
          loss_total, dict(metrics), (u_final, i_final) for optional logging
        """
        u_final, i_final = self.propagate(edge_index, edge_weight)

        # ----- BPR loss -----
        u = u_final[batch_users]                       # [B, D]
        i_pos = i_final[batch_pos_items]               # [B, D]
        i_neg = i_final[batch_neg_items]               # [B, D]

        pos_score = (u * i_pos).sum(-1)
        neg_score = (u * i_neg).sum(-1)
        bpr = F.softplus(-(pos_score - neg_score)).mean()

        # L2 regularization on the ID embeddings touched in this batch (classic LightGCN trick)
        reg = (u.pow(2).sum(dim=1) + i_pos.pow(2).sum(dim=1) + i_neg.pow(2).sum(dim=1)).mean() * 1e-4

        # ----- Content alignment -----
        # Prefer frozen buffer if present; else use provided tensor
        content_mat = self._frozen_content if self._frozen_content is not None else item_content
        align_loss = torch.tensor(0.0, device=u.device)

        if content_mat is not None and self.lambda_content > 0:
            if sample_items_for_align is None:
                proj = self.content_proj(content_mat)        # [I, D]
                align_loss = F.mse_loss(i_final, proj)
            else:
                proj = self.content_proj(content_mat[sample_items_for_align])  # [K, D]
                align_loss = F.mse_loss(i_final[sample_items_for_align], proj)

        loss = bpr + reg + self.lambda_content * align_loss

        metrics = {
            "loss/bpr": float(bpr.detach().cpu().item()),
            "loss/reg": float(reg.detach().cpu().item()),
            "loss/align": float(align_loss.detach().cpu().item()),
            "loss/total": float(loss.detach().cpu().item()),
        }
        return loss, metrics, (u_final, i_final)
