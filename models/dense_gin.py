"""
Dense GIN (Graph Isomorphism Network) for brain graph super-resolution.

Replaces GCN message passing with GIN-style sum aggregation + MLP:
  h' = MLP((1+ε)·h + A @ h) + h

- Weighted aggregation: A @ H preserves edge weights [0,1].
- Same pipeline as DenseGCN: linear upsample, bilinear decoder.
- Conservative decoder init, softplus, clamp [0,1] per spec.

Reference: Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGINBlock(nn.Module):
    """Single GIN layer: (1+ε)*H + A@H → MLP → residual."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, S: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: (B, N, N) symmetrically-normalised adjacency D^{-1/2}(A+I)D^{-1/2}
            H: (B, N, dim) node representations
        Returns:
            H': (B, N, dim) updated node representations
        """
        agg = (1.0 + self.epsilon) * H + S @ H  # degree-normalised aggregation
        out = self.mlp(agg)
        out = self.drop(out)
        return out + H


class DenseGINGenerator(nn.Module):
    """Dense GIN for LR → HR brain graph super-resolution.

    Forward signature: forward(A_lr, X_lr) -> pred_vec
    """

    def __init__(
        self,
        n_lr: int = 160,
        n_hr: int = 268,
        hidden_dim: int = 192,
        num_layers: int = 3,
        dropout: float = 0.35,
        raw_output: bool = False,
    ):
        super().__init__()
        self.n_lr = n_lr
        self.n_hr = n_hr
        self.raw_output = raw_output

        self.input_proj = nn.Sequential(
            nn.Linear(n_lr, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gin_layers = nn.ModuleList([
            DenseGINBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

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
        """S = D^{-1/2} (A + I) D^{-1/2}  (same as DenseGCN normalisation)."""
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
        S = self._normalise(A_lr)   # symmetrically-normalised adjacency
        H = self.input_proj(X_lr)

        for layer in self.gin_layers:
            H = layer(S, H)         # pass S, not raw A

        H = self.upsample(H.transpose(1, 2)).transpose(1, 2)
        H = self.decoder_norm(H)
        H = F.gelu(H)

        HP = H @ self.edge_P
        A_pred = 0.5 * (HP @ HP.transpose(1, 2) + (HP @ HP.transpose(1, 2)).transpose(1, 2))
        idx = self._get_triu_indices(A_pred.device)
        pred = A_pred[:, idx[0], idx[1]]
        if self.raw_output:
            return pred  # residual mode: caller adds y_mean back; can be negative
        return F.softplus(pred).clamp(min=0.0, max=1.0)
