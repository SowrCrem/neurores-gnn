"""
GraphGPS-style: GCN (local) + Linear Attention (global) for brain graph super-resolution.

Combines local message passing (GCN) with global attention over LR nodes.
- Local: 2-3 GCN blocks (same as DenseGCN)
- Global: 1 linear-attention block (O(N) via kernel φ(x)=elu(x)+1)
- Same pipeline: linear upsample 160→268, bilinear decoder + softplus

Reference: Rampášek et al., "Recipe for a General, Powerful, Scalable Graph Transformer" (NeurIPS 2022).
Linear attention: Katharopoulos et al., "Transformers are RNNs" (ICML 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dense_gcn import DenseGCNBlock


def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Kernel feature map φ(x) = elu(x) + 1 for linear attention."""
    return F.elu(x) + 1.0


class LinearAttentionBlock(nn.Module):
    """Linear-complexity self-attention over nodes. Uses φ(Q)(φ(K)^T V) / (φ(Q)(φ(K)^T 1))."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, N, dim) node representations
        Returns:
            H': (B, N, dim) updated
        """
        B, N, D = H.shape
        residual = H

        qkv = self.qkv(H)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply kernel: φ(x) = elu(x) + 1
        q = _elu_plus_one(q)
        k = _elu_plus_one(k)

        # Linear attention: out = φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ (φ(K)^T @ 1))
        # S = K^T @ V: (B, heads, head_dim, head_dim)
        kv = torch.einsum("bhnd,bhnv->bhdv", k, v)
        # Z = sum_j φ(K_j): (B, heads, head_dim)
        k1 = k.sum(dim=2)
        # Q @ (K^T V): (B, heads, N, head_dim) @ (B, heads, head_dim, head_dim) -> (B, heads, N, head_dim)
        qkv_out = torch.einsum("bhnd,bhdv->bhnv", q, kv)
        # Q @ (K^T 1): (B, heads, N, head_dim) @ (B, heads, head_dim, 1) -> (B, heads, N, 1)
        qk1 = (q * k1.unsqueeze(2)).sum(dim=-1, keepdim=True)  # (B, heads, N, 1)
        out = qkv_out / (qk1.clamp(min=1e-6))

        out = out.reshape(B, N, D)
        out = self.proj(out)
        out = self.norm(out)
        out = self.drop(out)
        return out + residual


class DenseGCNGPSGenerator(nn.Module):
    """GraphGPS-style: GCN (local) + Linear Attention (global) for LR → HR brain graph.

    Forward signature: forward(A_lr, X_lr) -> pred_vec
    """

    def __init__(
        self,
        n_lr: int = 160,
        n_hr: int = 268,
        hidden_dim: int = 192,
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
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gcn_layers = nn.ModuleList([
            DenseGCNBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.attn_block = LinearAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)

        self.upsample = nn.Linear(n_lr, n_hr)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.edge_P = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.edge_P, gain=0.1)

        self._triu_idx = None

    def _get_triu_indices(self, device: torch.device):
        if self._triu_idx is None or self._triu_idx[0].device != device:
            self._triu_idx = torch.triu_indices(
                self.n_hr, self.n_hr, offset=1, device=device,
            )
        return self._triu_idx

    @staticmethod
    def _normalise(A: torch.Tensor) -> torch.Tensor:
        """S = D^{-1/2} (A + I) D^{-1/2}"""
        N = A.size(-1)
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
        A_hat = A + I
        deg = A_hat.sum(dim=-1).clamp(min=1e-8)
        D_inv_sqrt = torch.diag_embed(deg.pow(-0.5))
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    def forward(self, A_lr: torch.Tensor, X_lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_lr: (B, n_lr, n_lr) weighted symmetric adjacency
            X_lr: (B, n_lr, n_lr) node features = adjacency rows
        Returns:
            pred: (B, n_hr*(n_hr-1)/2) predicted HR edge-weight vector
        """
        S = self._normalise(A_lr)

        H = self.input_proj(X_lr)

        for layer in self.gcn_layers:
            H = layer(S, H)

        H = self.attn_block(H)

        H = self.upsample(H.transpose(1, 2)).transpose(1, 2)
        H = self.decoder_norm(H)
        H = F.gelu(H)

        HP = H @ self.edge_P
        A_pred = 0.5 * (HP @ HP.transpose(1, 2) + (HP @ HP.transpose(1, 2)).transpose(1, 2))
        idx = self._get_triu_indices(A_pred.device)
        pred = A_pred[:, idx[0], idx[1]]
        pred = F.softplus(pred)
        return pred.clamp(min=0.0, max=1.0)
