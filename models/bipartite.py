"""
Track A: Bipartite LR→HR Super-Resolution (weighted connectomes)
===============================================================

LR vector -> LR adjacency -> LR node features (structural + optional Laplacian PE)
-> LR encoder (weighted message passing)
-> HR node init (learned embeddings + optional global conditioning)
-> LR→HR bridge (cross-attention blocks over a complete bipartite graph)
-> HR edge decoder (pairwise symmetric MLP)
-> HR adjacency -> (optional) output in the same vectorized format as dataset.

Designed to be easy to swap parts:
- Positional encodings: change `PositionalEncodingBase` implementation.
- LR encoder: replace `WeightedGCNEncoder` with your own.
- Bridge: replace `CrossAttentionBridge`.
- Decoder: replace `SymmetricMLPEdgeDecoder`.

Assumptions:
- Dataset has already replaced negative and NaN values with 0.
- Input vectors can be either:
  (A) full flatten of an NxN matrix (length = N*N), or
  (B) upper-triangular without diagonal (length = N*(N-1)/2).
  The utilities below handle both; set `vector_format` explicitly if you know it.

Test-time usage:
    model = GraphSRModel(lr_n=160, hr_n=268, ...)
    y_hat_vec = model(x_lr_vec)  # returns HR vector in same format as `out_vector_format`.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.matrix_vectorizer import MatrixVectorizer


# -------------------------
# Type definitions and helper functions
# -------------------------

VectorFormat = Literal["triu_no_diag"]  # Simplified: only support upper-triangle without diagonal


def vec_to_adj(vec: torch.Tensor, n: int) -> torch.Tensor:
    """
    Convert upper-triangle vector (no diagonal) -> symmetric adjacency.
    Pure PyTorch implementation to preserve gradients.
    
    Args:
        vec: (B, E) or (E,) where E = n*(n-1)/2
        n: size of the square adjacency matrix
        
    Returns:
        (B, n, n) or (n, n) symmetric adjacency matrix
    """
    squeeze = (vec.dim() == 1)
    if squeeze:
        vec = vec.unsqueeze(0)
    
    B = vec.shape[0]
    device = vec.device
    dtype = vec.dtype
    
    # Create adjacency matrices using pure PyTorch
    A = torch.zeros((B, n, n), device=device, dtype=dtype)
    
    # Fill upper triangle (excluding diagonal)
    triu_indices = torch.triu_indices(n, n, offset=1, device=device)
    A[:, triu_indices[0], triu_indices[1]] = vec
    
    # Make symmetric by adding the transpose
    A = A + A.transpose(-2, -1)
    
    return A.squeeze(0) if squeeze else A


def adj_to_vec(A: torch.Tensor) -> torch.Tensor:
    """
    Convert symmetric adjacency -> upper-triangle vector (no diagonal).
    Pure PyTorch implementation to preserve gradients.
    
    Args:
        A: (B, n, n) or (n, n) symmetric adjacency matrix
        
    Returns:
        (B, E) or (E,) upper-triangular vector where E = n*(n-1)/2
    """
    squeeze = (A.dim() == 2)
    if squeeze:
        A = A.unsqueeze(0)
    
    B, n, _ = A.shape
    
    # Extract upper triangle (excluding diagonal)
    triu_indices = torch.triu_indices(n, n, offset=1, device=A.device)
    vec = A[:, triu_indices[0], triu_indices[1]]
    
    return vec.squeeze(0) if squeeze else vec


# -------------------------
# Positional encodings (swap these freely)
# -------------------------

class PositionalEncodingBase(nn.Module):
    """
    Base class for node positional encodings.
    Implement `forward(A)` -> (B, N, k) or None.
    """
    def forward(self, A: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError


class NonePosEnc(PositionalEncodingBase):
    def forward(self, A: torch.Tensor) -> Optional[torch.Tensor]:
        return None


class LaplacianPosEnc(PositionalEncodingBase):
    """
    Laplacian eigenvector positional encodings.
    Computes top-k non-trivial eigenvectors of normalized Laplacian per graph.

    Notes:
    - Dense eigendecomposition is O(N^3). For N=160 it's usually OK.
    - We loop over batch; keep batch sizes modest.
    """
    def __init__(self, k: int = 8, normalized: bool = True, eps: float = 1e-8):
        super().__init__()
        self.k = k
        self.normalized = normalized
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        A: (B, N, N) symmetric, nonnegative
        returns: (B, N, k)
        
        Note: eigendecomposition is computed without gradients (for speed),
        but the output PE tensor remains in the computation graph.
        """
        B, N, _ = A.shape
        out = []

        for b in range(B):
            Ab = A[b]
            # Degree / strength
            d = Ab.sum(dim=-1)  # (N,)
            if self.normalized:
                dinv_sqrt = (d + self.eps).pow(-0.5)
                Dinv = torch.diag(dinv_sqrt)
                L = torch.eye(N, device=A.device, dtype=A.dtype) - Dinv @ Ab @ Dinv
            else:
                L = torch.diag(d) - Ab

            # Compute eigendecomposition without tracking gradients (for efficiency)
            with torch.no_grad():
                evals, evecs = torch.linalg.eigh(L)  # (N,), (N,N)
            
            # skip the first eigenvector (often constant); take next k
            k = min(self.k, N - 1)
            pe = evecs[:, 1 : 1 + k]  # (N,k)
            # if k < self.k, pad with zeros for consistent dims
            if k < self.k:
                pe = F.pad(pe, (0, self.k - k), value=0.0)
            out.append(pe)

        return torch.stack(out, dim=0)  # (B,N,k)


# -------------------------
# LR feature builder (swap/extend)
# -------------------------

class LRFeatureBuilder(nn.Module):
    """
    Build LR node features from adjacency only.

    Features included:
    - strength (sum of weights)
    - mean incident weight
    - std incident weight
    - optional Laplacian PE (k dims)

    Returns X: (B, N, d_in)
    """
    def __init__(self, pos_enc: PositionalEncodingBase, eps: float = 1e-8):
        super().__init__()
        self.pos_enc = pos_enc
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        B, N, _ = A.shape
        strength = A.sum(dim=-1)  # (B,N)

        # mean/std over incident weights (including self-edge if present; diagonal is usually 0)
        mean_w = A.mean(dim=-1)  # (B,N)
        var_w = A.var(dim=-1, unbiased=False)
        std_w = torch.sqrt(var_w + self.eps)

        feats = [strength, mean_w, std_w]  # each (B,N)
        X = torch.stack(feats, dim=-1)     # (B,N,3)

        PE = self.pos_enc(A)               # (B,N,k) or None
        if PE is not None:
            X = torch.cat([X, PE], dim=-1)
        return X


# -------------------------
# Weighted LR encoder
# -------------------------

class WeightedGCNLayer(nn.Module):
    """
    Dense weighted GCN-style layer:
        H' = sigma( A_norm @ H @ W )
    where A_norm = D^{-1/2} A D^{-1/2} (strength-based normalization)
    """
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0, eps: float = 1e-8):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.dropout = dropout
        self.eps = eps

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # H: (B,N,d), A: (B,N,N)
        d = A.sum(dim=-1)  # (B,N)
        dinv_sqrt = (d + self.eps).pow(-0.5)  # (B,N)
        A_norm = A * dinv_sqrt.unsqueeze(-1) * dinv_sqrt.unsqueeze(-2)  # (B,N,N)

        H = F.dropout(H, p=self.dropout, training=self.training)
        H = self.lin(H)
        H = torch.matmul(A_norm, H)
        return F.gelu(H)


class WeightedGCNEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(
                WeightedGCNLayer(
                    d_in if i == 0 else d_model,
                    d_model,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = X
        for layer in self.layers:
            H = layer(H, A)
        return self.norm(H)  # (B,N,d_model)


# -------------------------
# HR initialization
# -------------------------

class HRInitializer(nn.Module):
    """
    HR node init:
    - Learned embedding table for HR indices (0..hr_n-1)
    - Optionally add a pooled LR "global" conditioning vector
    """
    def __init__(self, hr_n: int, d_model: int, use_global_cond: bool = True):
        super().__init__()
        self.hr_n = hr_n
        self.emb = nn.Embedding(hr_n, d_model)
        self.use_global_cond = use_global_cond
        if use_global_cond:
            self.global_proj = nn.Linear(d_model, d_model)

    def forward(self, H_lr: torch.Tensor) -> torch.Tensor:
        """
        H_lr: (B, lr_n, d_model)
        returns H_hr0: (B, hr_n, d_model)
        """
        B = H_lr.shape[0]
        idx = torch.arange(self.hr_n, device=H_lr.device)
        base = self.emb(idx).unsqueeze(0).expand(B, -1, -1)  # (B,hr_n,d)

        if not self.use_global_cond:
            return base

        g = H_lr.mean(dim=1)  # (B,d)
        g = self.global_proj(g).unsqueeze(1)  # (B,1,d)
        return base + g


# -------------------------
# LR->HR bridge (cross-attention)
# -------------------------

class CrossAttentionBlock(nn.Module):
    """
    A transformer-style cross-attention block:
      HR queries attend to LR keys/values.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_hr: torch.Tensor, H_lr: torch.Tensor) -> torch.Tensor:
        # H_hr: (B,hr_n,d), H_lr: (B,lr_n,d)
        q = self.ln1(H_hr)
        k = H_lr
        v = H_lr
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        H_hr = H_hr + self.dropout(attn_out)
        H_hr = H_hr + self.dropout(self.ff(self.ln2(H_hr)))
        return H_hr


class CrossAttentionBridge(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, H_hr: torch.Tensor, H_lr: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            H_hr = blk(H_hr, H_lr)
        return H_hr


# -------------------------
# HR edge decoder (swap this easily)
# -------------------------

class SymmetricMLPEdgeDecoder(nn.Module):
    """
    Decode a full symmetric weighted adjacency from HR node embeddings.

    Uses pair features:
        [h_i, h_j, |h_i-h_j|, h_i*h_j]
    and an MLP to output scalar weights.

    Complexity: O(hr_n^2 * d). For hr_n=268 this is usually fine.
    """
    def __init__(self, d_model: int, hidden: int = 256, out_activation: Literal["none", "softplus", "sigmoid"] = "none"):
        super().__init__()
        self.out_activation = out_activation
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: (B, N, d)
        returns A_hat: (B, N, N)
        """
        B, N, d = H.shape
        hi = H.unsqueeze(2).expand(B, N, N, d)
        hj = H.unsqueeze(1).expand(B, N, N, d)
        feats = torch.cat([hi, hj, (hi - hj).abs(), hi * hj], dim=-1)  # (B,N,N,4d)

        w = self.mlp(feats).squeeze(-1)  # (B,N,N)
        # enforce symmetry
        w = 0.5 * (w + w.transpose(-1, -2))
        # zero diagonal
        w = w - torch.diag_embed(torch.diagonal(w, dim1=-2, dim2=-1))

        if self.out_activation == "softplus":
            w = F.softplus(w)
        elif self.out_activation == "sigmoid":
            w = torch.sigmoid(w)

        return w


# -------------------------
# Full model
# -------------------------

@dataclass
class GraphSRConfig:
    lr_n: int = 160
    hr_n: int = 268

    # features / enc
    posenc_k: int = 8
    d_model: int = 128
    lr_enc_layers: int = 3
    dropout: float = 0.1

    # bridge
    bridge_layers: int = 4
    heads: int = 4

    # decoder
    dec_hidden: int = 256
    out_activation: Literal["none", "softplus", "sigmoid"] = "none"


class GraphSRModel(nn.Module):
    def __init__(self, cfg: GraphSRConfig):
        super().__init__()
        self.cfg = cfg

        # PosEnc is modular; swap LaplacianPosEnc with your own
        self.pos_enc = LaplacianPosEnc(k=cfg.posenc_k)

        self.feat_builder = LRFeatureBuilder(pos_enc=self.pos_enc)
        d_in = 3 + cfg.posenc_k

        self.lr_encoder = WeightedGCNEncoder(
            d_in=d_in,
            d_model=cfg.d_model,
            n_layers=cfg.lr_enc_layers,
            dropout=cfg.dropout,
        )

        self.hr_init = HRInitializer(cfg.hr_n, cfg.d_model, use_global_cond=True)

        self.bridge = CrossAttentionBridge(
            d_model=cfg.d_model,
            n_layers=cfg.bridge_layers,
            n_heads=cfg.heads,
            dropout=cfg.dropout,
        )

        self.decoder = SymmetricMLPEdgeDecoder(
            d_model=cfg.d_model,
            hidden=cfg.dec_hidden,
            out_activation=cfg.out_activation,
        )

    def forward(self, x_lr_vec: torch.Tensor) -> torch.Tensor:
        """
        x_lr_vec: (B, L_lr) or (L_lr,) where L_lr = 160*159/2 = 12720
        returns y_hr_vec: (B, L_hr) or (L_hr,) where L_hr = 268*267/2 = 35778
        
        Uses upper-triangular vector format (no diagonal) throughout.
        """
        squeeze_back = (x_lr_vec.dim() == 1)

        # 1) vector -> adjacency (upper-triu to full symmetric)
        A_lr = vec_to_adj(x_lr_vec, n=self.cfg.lr_n)  # (B,lr_n,lr_n) or (lr_n,lr_n)
        
        if A_lr.dim() == 2:
            A_lr = A_lr.unsqueeze(0)

        # 2) compute LR node features from adjacency
        X_lr = self.feat_builder(A_lr)  # (B,lr_n,d_in)

        # 3) encode LR graph with weighted GCN
        H_lr = self.lr_encoder(X_lr, A_lr)  # (B,lr_n,d_model)

        # 4) initialize HR nodes
        H_hr = self.hr_init(H_lr)  # (B,hr_n,d_model)

        # 5) cross-attention bridge (LR -> HR)
        H_hr = self.bridge(H_hr, H_lr)  # (B,hr_n,d_model)

        # 6) decode HR adjacency from node embeddings
        A_hr_hat = self.decoder(H_hr)  # (B,hr_n,hr_n)

        # 7) adjacency -> vector (full symmetric to upper-triu)
        y_hat_vec = adj_to_vec(A_hr_hat)  # (B,L_hr) or (L_hr,)

        return y_hat_vec.squeeze(0) if squeeze_back else y_hat_vec


# -------------------------
# Losses (keep modular)
# -------------------------

class GraphSRLoss(nn.Module):
    """
    Default loss: MSE on vectorized HR edges + optional strength matching.

    If you want topology regularizers later, add them here.
    """
    def __init__(self, hr_n: int, strength_weight: float = 0.0):
        super().__init__()
        self.hr_n = hr_n
        self.strength_weight = strength_weight

    def forward(self, y_hat_vec: torch.Tensor, y_true_vec: torch.Tensor) -> torch.Tensor:
        # Edge-level MSE on vector representation
        mse = F.mse_loss(y_hat_vec, y_true_vec)

        if self.strength_weight <= 0:
            return mse

        # Strength loss computed on full adjacency (more interpretable)
        A_hat = vec_to_adj(y_hat_vec, n=self.hr_n)
        A_true = vec_to_adj(y_true_vec, n=self.hr_n)

        if A_hat.dim() == 2:
            A_hat = A_hat.unsqueeze(0)
            A_true = A_true.unsqueeze(0)

        s_hat = A_hat.sum(dim=-1)   # (B,hr_n)
        s_true = A_true.sum(dim=-1) # (B,hr_n)

        strength_mse = F.mse_loss(s_hat, s_true)
        return mse + self.strength_weight * strength_mse