"""
GCN + Cross-Attention for brain graph super-resolution.

Optimized for 167 samples and MAE (DESIGN_CONSTRAINTS.md):
  - GCN encoder (structure-aware, stable)
  - Cross-attention upsampling: HR queries attend to LR node embeddings
  - Full adjacency rows as input (160-dim), not handcrafted 2-dim features
  - Conservative init, GELU, softplus, clamp [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dense_gcn import DenseGCNBlock


class DenseGCNCrossAttnGenerator(nn.Module):
    """
    GCN encoder + cross-attention upsampling for LR → HR.

    - Input: full adjacency rows (B, 160, 160)
    - 3 GCN blocks encode LR nodes
    - Learnable HR queries (268, d) attend to LR embeddings
    - Bilinear decoder, softplus, clamp [0,1]

    Forward: (A_lr, X_lr) -> pred_vec (B, 35778)
    """

    def __init__(
        self,
        n_lr: int = 160,
        n_hr: int = 268,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.n_lr = n_lr
        self.n_hr = n_hr

        self.input_proj = nn.Sequential(
            nn.Linear(n_lr, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gcn_layers = nn.ModuleList([
            DenseGCNBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        self.hr_queries = nn.Parameter(torch.randn(n_hr, hidden_dim) * 0.01)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

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
        H = self.encoder_norm(H)

        B = H.shape[0]
        Q = self.hr_queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=Q, key=H, value=H)
        H_hr = self.attn_norm(attn_out + Q)

        H_hr = self.decoder_norm(H_hr)
        H_hr = F.gelu(H_hr)

        HP = H_hr @ self.edge_P
        A_pred = 0.5 * (HP @ HP.transpose(1, 2) + (HP @ HP.transpose(1, 2)).transpose(1, 2))
        idx = self._get_triu_indices(A_pred.device)
        pred = A_pred[:, idx[0], idx[1]]
        pred = F.softplus(pred)
        return pred.clamp(min=0.0, max=1.0)
