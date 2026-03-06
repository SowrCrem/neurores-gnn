"""
Dense Graph-Attention (DenseGAT) for brain graph super-resolution.

Redesigned for stability with 167 samples (DESIGN_CONSTRAINTS.md):

  - GCN-first: 2 GCN blocks provide stable structure-aware encoding (like DenseGCN).
  - 1 GAT block with strong edge_scale (0.5): graph dominates, attention refines.
  - Single linear upsample (like DenseGCN) - no complex MLP.
  - No hr_refine: pure attention on HR with no graph structure overfits (past failure).
  - Conservative decoder init (gain=0.1), softplus, output clamp [0,1] per spec.
  - GELU throughout (avoids ReLU collapse before decoder).

Past failures addressed: attention-only encoder collapsed; hr_refine overfit;
complex upsample and aggressive init caused instability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dense_gcn import DenseGCNBlock


class GraphAttentionBlock(nn.Module):
    """
    Pre-norm block: multi-head attention with additive adjacency bias.
    Strong edge_scale ensures graph structure dominates; attention refines.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
        ffn_mult: int = 2,
        edge_scale: float = 0.5,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

        self.edge_bias_proj = nn.Linear(1, num_heads, bias=False)
        self._edge_scale = edge_scale

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, D = H.shape
        x = self.norm1(H)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        edge_bias = self._edge_scale * self.edge_bias_proj(A.unsqueeze(-1))
        edge_bias = edge_bias.permute(0, 3, 1, 2)
        attn = attn + edge_bias
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        H = H + out
        H = H + self.ffn(self.norm2(H))
        return H


class DenseGATGenerator(nn.Module):
    """
    Dense GAT for LR → HR brain graph super-resolution.

    GCN-first design: 2 GCN blocks + 1 GAT block (graph-heavy attention).
    Single linear upsample, bilinear decoder, softplus + clamp [0,1].

    Forward signature: forward(A_lr, X_lr) -> pred_vec
    """

    def __init__(
        self,
        n_lr: int = 160,
        n_hr: int = 268,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.4,
        ffn_mult: int = 2,
        num_decoder_heads: int = 4,
        hr_refine_layers: int = 0,
        edge_scale: float = 0.5,
    ):
        super().__init__()
        self.n_lr = n_lr
        self.n_hr = n_hr

        gcn_count = max(1, num_layers - 1)
        gat_count = 1 if num_layers >= 2 else 0

        self.input_proj = nn.Sequential(
            nn.Linear(n_lr, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gcn_layers = nn.ModuleList([
            DenseGCNBlock(hidden_dim, dropout=dropout)
            for _ in range(gcn_count)
        ])
        self.gat_layer = GraphAttentionBlock(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            ffn_mult=ffn_mult,
            edge_scale=edge_scale,
        ) if gat_count > 0 else None

        self.encoder_norm = nn.LayerNorm(hidden_dim)

        self.upsample = nn.Linear(n_lr, n_hr)

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.edge_P = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.edge_P, gain=0.1)

        self._triu_idx = None

    @staticmethod
    def _normalise(A: torch.Tensor) -> torch.Tensor:
        N = A.size(-1)
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
        A_hat = A + I
        deg = A_hat.sum(dim=-1).clamp(min=1e-8)
        D_inv_sqrt = torch.diag_embed(deg.pow(-0.5))
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    def _get_triu_indices(self, device: torch.device):
        if self._triu_idx is None or self._triu_idx[0].device != device:
            self._triu_idx = torch.triu_indices(
                self.n_hr, self.n_hr, offset=1, device=device
            )
        return self._triu_idx

    def forward(self, A_lr: torch.Tensor, X_lr: torch.Tensor) -> torch.Tensor:
        S = self._normalise(A_lr)
        H = self.input_proj(X_lr)

        for layer in self.gcn_layers:
            H = layer(S, H)

        if self.gat_layer is not None:
            H = self.gat_layer(H, A_lr)

        H = self.encoder_norm(H)
        H = self.upsample(H.transpose(1, 2)).transpose(1, 2)
        H = self.decoder_norm(H)
        H = F.gelu(H)

        HP = H @ self.edge_P
        A_pred = 0.5 * (HP @ HP.transpose(1, 2) + (HP @ HP.transpose(1, 2)).transpose(1, 2))
        idx = self._get_triu_indices(A_pred.device)
        pred = A_pred[:, idx[0], idx[1]]
        pred = F.softplus(pred)
        return pred.clamp(min=0.0, max=1.0)
