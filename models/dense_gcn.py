"""
Dense Multi-Layer GCN for brain graph super-resolution.

Unlike the SGC baseline (single linear transform on 2-dim features), this model
uses full adjacency-row node features (160-dim) and applies multiple nonlinear
GCN layers with residual connections and layer normalization.

All operations are dense tensor matmuls — no DGL dependency.

Architecture:
    1. Node features = LR adjacency rows (B, 160, 160)
    2. Input projection: Linear(160, hidden_dim)
    3. Normalise LR adjacency: S = D^{-1/2} (A+I) D^{-1/2}
    4. N x DenseGCNBlock: H' = dropout(ReLU(LN(S @ H @ W))) + H
    5. Learned node upsample: 160 -> 268
    6. Bilinear edge decoder: A_hr = H W_edge H^T, symmetrised
    7. Extract upper triangle, clamp >= 0
"""

import torch
import torch.nn as nn


class DenseGCNBlock(nn.Module):
    """Single dense GCN layer with LayerNorm, residual connection, and dropout."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, S: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: (B, N, N) normalised adjacency
            H: (B, N, dim) node representations
        Returns:
            H': (B, N, dim) updated node representations
        """
        out = S @ H                   # neighbourhood aggregation
        out = self.linear(out)        # learnable transform
        out = self.norm(out)          # stabilise
        out = torch.relu(out)
        out = self.drop(out)
        return out + H                # residual


class DenseGCNGenerator(nn.Module):
    """Dense multi-layer GCN for LR -> HR brain graph super-resolution.

    Forward signature matches SGCBaseline: forward(A_lr, X_lr) -> pred_vec
    """

    def __init__(
        self,
        n_lr: int = 160,
        n_hr: int = 268,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        raw_output: bool = False,
        lap_pe_dim: int = 0,
        pearl_pe_dim: int = 0,
    ):
        super().__init__()
        self.n_lr = n_lr
        self.n_hr = n_hr
        self.raw_output = raw_output
        self.lap_pe_dim = lap_pe_dim
        self.pearl_pe_dim = pearl_pe_dim

        input_dim = n_lr + lap_pe_dim + pearl_pe_dim  # adjacency rows + optional Laplacian PE + PEARL PE
        
        if self.pearl_pe_dim > 0:
            self.pearl_pe = nn.Parameter(torch.empty(1, n_lr, pearl_pe_dim))
            nn.init.normal_(self.pearl_pe, mean=0.0, std=0.02)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gcn_layers = nn.ModuleList([
            DenseGCNBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.upsample = nn.Linear(n_lr, n_hr)

        self.edge_W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.edge_W, gain=0.5)

        self._triu_idx = None

    def _get_triu_indices(self, device):
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

    def forward(self, A_lr: torch.Tensor, X_lr: torch.Tensor, lap_pe: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            A_lr:   (B, n_lr, n_lr) weighted symmetric adjacency
            X_lr:   (B, n_lr, n_lr) node features = adjacency rows
            lap_pe: (B, n_lr, lap_pe_dim) optional Laplacian eigenvector PE; None = disabled
        Returns:
            pred: (B, n_hr*(n_hr-1)/2) predicted HR edge-weight vector
        """
        S = self._normalise(A_lr)

        # Concatenate Laplacian PE to node features before projection
        if lap_pe is not None and self.lap_pe_dim > 0:
            X_lr = torch.cat([X_lr, lap_pe], dim=-1)  # (B, n_lr, n_lr + lap_pe_dim)
            
        # Concatenate PEARL learnable PE
        if self.pearl_pe_dim > 0:
            pearl_feat = self.pearl_pe.expand(A_lr.size(0), -1, -1) # (B, n_lr, pearl_pe_dim)
            X_lr = torch.cat([X_lr, pearl_feat], dim=-1)

        H = self.input_proj(X_lr)                       # (B, n_lr, hidden_dim)

        for layer in self.gcn_layers:
            H = layer(S, H)                             # (B, n_lr, hidden_dim)

        # Node upsample: (B, n_lr, d) -> (B, n_hr, d)
        H = self.upsample(H.transpose(1, 2)).transpose(1, 2)

        # Bilinear edge decode + symmetrise
        HW = H @ self.edge_W                            # (B, n_hr, d)
        A = HW @ H.transpose(1, 2)                      # (B, n_hr, n_hr)
        A = 0.5 * (A + A.transpose(1, 2))

        idx = self._get_triu_indices(A.device)
        pred = A[:, idx[0], idx[1]]                      # (B, 35778)
        if self.raw_output:
            return pred  # residual mode: caller adds y_mean back; can be negative
        return torch.nn.functional.softplus(pred)  # non-negative + gradient flow
