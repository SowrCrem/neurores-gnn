"""
Dense Graph-Attention Transformer (DenseGATNet) for brain graph super-resolution.

Improvements over DenseGCN:
  1. Multi-head graph attention encoder (structure-aware, adaptive weighting)
     instead of fixed symmetric-normalised GCN aggregation.
  2. Two-stage node upsample: MLP (160 → 268) with nonlinearity.
  3. Multi-head bilinear edge decoder (K independent W matrices, averaged)
     for richer edge-level expressivity.
  4. Pre-norm Transformer-style blocks for better gradient flow.
  5. Optional cross-attention refinement on HR nodes.

All operations are dense tensor matmuls — no DGL dependency.

Architecture:
    1. Node features = LR adjacency rows (B, 160, 160)
    2. Input projection: Linear(160, d) + LN + GELU
    3. K × GraphAttentionBlock (multi-head self-attention with adjacency bias)
    4. MLP node upsample: 160 → 268
    5. (Optional) HR self-attention refinement layers
    6. Multi-head bilinear edge decoder: mean_k(H @ W_k @ H^T), symmetrised
    7. Extract upper triangle, clamp ≥ 0
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionBlock(nn.Module):
    """
    Pre-norm Transformer block with structure-aware multi-head self-attention.

    Adjacency weights are injected as additive bias to attention logits,
    so the model can attend both by content (learned Q/K) and by graph
    topology (edge weights).
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, ffn_mult: int = 4):
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
        # Scale down adjacency bias so content (Q,K) matters; avoid attention collapse
        self.edge_bias_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, N, dim)  node representations
            A: (B, N, N)    weighted adjacency (used as attention bias)
        Returns:
            H': (B, N, dim) updated representations
        """
        B, N, D = H.shape

        # --- multi-head self-attention with adjacency bias ---
        x = self.norm1(H)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, heads, N, hd)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # adjacency bias: project scalar edge weights to per-head bias (scaled so content matters)
        edge_bias = self.edge_bias_scale * self.edge_bias_proj(A.unsqueeze(-1))  # (B, N, N, heads)
        edge_bias = edge_bias.permute(0, 3, 1, 2)         # (B, heads, N, N)
        attn = attn + edge_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        H = H + out

        # --- FFN ---
        H = H + self.ffn(self.norm2(H))
        return H


class HRRefinementBlock(nn.Module):
    """
    Standard pre-norm self-attention block for refining HR node embeddings
    after upsampling (no adjacency bias since HR graph is unknown).
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        x = self.norm1(H)
        out, _ = self.attn(x, x, x)
        H = H + out
        H = H + self.ffn(self.norm2(H))
        return H


class DenseGATGenerator(nn.Module):
    """
    Dense Graph-Attention Transformer for LR → HR brain graph super-resolution.

    Forward signature matches DenseGCNGenerator: forward(A_lr, X_lr) -> pred_vec
    """

    def __init__(
        self,
        n_lr: int = 160,
        n_hr: int = 268,
        hidden_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.25,
        ffn_mult: int = 4,
        num_decoder_heads: int = 4,
        hr_refine_layers: int = 1,
    ):
        super().__init__()
        self.n_lr = n_lr
        self.n_hr = n_hr
        self.num_decoder_heads = num_decoder_heads

        # --- input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(n_lr, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- LR encoder: graph-attention blocks ---
        self.encoder = nn.ModuleList([
            GraphAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        # --- node upsample: MLP (n_lr → n_hr) over node dimension ---
        self.upsample = nn.Sequential(
            nn.Linear(n_lr, n_hr),
            nn.GELU(),
            nn.Linear(n_hr, n_hr),
        )

        # --- HR refinement ---
        self.hr_refine = nn.ModuleList([
            HRRefinementBlock(hidden_dim, num_heads=num_heads, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(hr_refine_layers)
        ]) if hr_refine_layers > 0 else nn.ModuleList()
        if hr_refine_layers > 0:
            self.hr_norm = nn.LayerNorm(hidden_dim)

        # --- multi-head bilinear edge decoder ---
        self.edge_Ws = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            for _ in range(num_decoder_heads)
        ])
        for W in self.edge_Ws:
            nn.init.xavier_uniform_(W)

        # Shift decoder output into range where softplus has gradient (avoid vanishing when raw < -20)
        self.decoder_bias = nn.Parameter(torch.full((1,), 3.0))

        self._triu_idx = None

    def _get_triu_indices(self, device: torch.device):
        if self._triu_idx is None or self._triu_idx[0].device != device:
            self._triu_idx = torch.triu_indices(self.n_hr, self.n_hr, offset=1, device=device)
        return self._triu_idx

    def forward(self, A_lr: torch.Tensor, X_lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_lr: (B, n_lr, n_lr) weighted symmetric adjacency
            X_lr: (B, n_lr, n_lr) node features = adjacency rows
        Returns:
            pred: (B, n_hr*(n_hr-1)/2) predicted HR edge-weight vector
        """
        H = self.input_proj(X_lr)                        # (B, n_lr, d)

        for block in self.encoder:
            H = block(H, A_lr)                           # (B, n_lr, d)
        H = self.encoder_norm(H)

        # node upsample: (B, n_lr, d) → (B, n_hr, d)
        H = self.upsample(H.transpose(1, 2)).transpose(1, 2)

        for block in self.hr_refine:
            H = block(H)
        if self.hr_refine:
            H = self.hr_norm(H)

        # multi-head bilinear edge decode
        A_sum = torch.zeros(H.shape[0], self.n_hr, self.n_hr, device=H.device, dtype=H.dtype)
        for W in self.edge_Ws:
            HW = H @ W                                    # (B, n_hr, d)
            A_sum = A_sum + HW @ H.transpose(1, 2)        # (B, n_hr, n_hr)
        A_pred = A_sum / self.num_decoder_heads

        A_pred = 0.5 * (A_pred + A_pred.transpose(1, 2))  # symmetrise

        idx = self._get_triu_indices(A_pred.device)
        pred = A_pred[:, idx[0], idx[1]]                   # (B, 35778)
        return F.softplus(pred + self.decoder_bias)
