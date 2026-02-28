"""
Dense GraphSAGE for brain graph super-resolution.

Replaces GCN with GraphSAGE-style mean aggregation:
  h' = MLP(mean(h_j for j in N(i) ∪ {i})) + h

- Mean aggregation: row-normalized (A+I) @ H gives mean over self + neighbors.
- Same pipeline as DenseGCN: linear upsample, bilinear decoder.
- Conservative decoder init, softplus, clamp [0,1] per spec.

Reference: Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGraphSAGEBlock(nn.Module):
    """Single GraphSAGE layer: mean aggregation → MLP → residual."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, S_mean: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S_mean: (B, N, N) row-normalized (A+I), each row sums to 1
            H: (B, N, dim) node representations
        Returns:
            H': (B, N, dim) updated node representations
        """
        agg = S_mean @ H  # mean over self + neighbors
        out = self.linear(agg)
        out = self.norm(out)
        out = torch.relu(out)
        out = self.drop(out)
        return out + H


class DenseGraphSAGEGenerator(nn.Module):
    """Dense GraphSAGE for LR → HR brain graph super-resolution.

    Forward signature: forward(A_lr, X_lr) -> pred_vec
    """

    def __init__(
        self,
        n_lr: int = 160,
        n_hr: int = 268,
        hidden_dim: int = 192,
        num_layers: int = 3,
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

        self.sage_layers = nn.ModuleList([
            DenseGraphSAGEBlock(hidden_dim, dropout=dropout)
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
    def _mean_normalise(A: torch.Tensor) -> torch.Tensor:
        """Row-normalize (A+I) so each row sums to 1 → mean aggregation."""
        A_hat = A + torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0)
        row_sum = A_hat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return A_hat / row_sum

    def forward(self, A_lr: torch.Tensor, X_lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_lr: (B, n_lr, n_lr) weighted symmetric adjacency
            X_lr: (B, n_lr, n_lr) node features = adjacency rows
        Returns:
            pred: (B, n_hr*(n_hr-1)/2) predicted HR edge-weight vector
        """
        S_mean = self._mean_normalise(A_lr)

        H = self.input_proj(X_lr)

        for layer in self.sage_layers:
            H = layer(S_mean, H)

        H = self.upsample(H.transpose(1, 2)).transpose(1, 2)
        H = self.decoder_norm(H)
        H = F.gelu(H)

        HP = H @ self.edge_P
        A_pred = 0.5 * (HP @ HP.transpose(1, 2) + (HP @ HP.transpose(1, 2)).transpose(1, 2))
        idx = self._get_triu_indices(A_pred.device)
        pred = A_pred[:, idx[0], idx[1]]
        pred = F.softplus(pred)
        return pred.clamp(min=0.0, max=1.0)
