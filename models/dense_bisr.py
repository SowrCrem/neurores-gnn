"""
Bi-SR (Bipartite Super-Resolution) for brain graph super-resolution.

Replaces linear upsample with bipartite message passing: LR nodes (160) and HR nodes (268)
are connected via a bipartite graph; each HR node aggregates from all LR nodes.

- LR encoder: Reuses DenseGCNBlock from dense_gcn.
- Bipartite GNN: Dedicated BipartiteGNNBlock with GELU (avoids collapse before decoder).
- Decoder: Bilinear H @ P @ P^T @ H^T + softplus (like DenseGAT).

Reference: Singh & Rekik, "Rethinking Graph Super-resolution: Dual Frameworks
for Topological Fidelity" (2025), arXiv:2511.08853.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dense_gcn import DenseGCNBlock


class BipartiteGNNBlock(nn.Module):
    """GNN block for bipartite graph. Uses GELU (not ReLU) to avoid collapse before decoder."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, S_b: torch.Tensor, H_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S_b: (B, n_lr+n_hr, n_lr+n_hr) normalised bipartite adjacency
            H_b: (B, n_lr+n_hr, dim) concatenated [H_lr; H_hr]
        Returns:
            H_b': (B, n_lr+n_hr, dim) updated
        """
        out = S_b @ H_b
        out = self.linear(out)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.drop(out)
        return out + H_b


class DenseBiSRGenerator(nn.Module):
    """Bi-SR: Bipartite super-resolution for LR -> HR brain graph.

    Forward signature: forward(A_lr, X_lr) -> pred_vec (B, 35778)
    """

    def __init__(
        self,
        n_lr: int = 160,
        n_hr: int = 268,
        hidden_dim: int = 192,
        num_layers: int = 3,
        bipartite_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_lr = n_lr
        self.n_hr = n_hr
        self.n_bipartite = n_lr + n_hr

        self.input_proj = nn.Sequential(
            nn.Linear(n_lr, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # GELU avoids ReLU collapse; matches bipartite block
            nn.Dropout(dropout),
        )

        self.gcn_layers = nn.ModuleList([
            DenseGCNBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Fixed small random init for HR nodes to break symmetry (0.1 scale avoids collapse)
        hr_init = 0.1 * torch.rand(n_hr, hidden_dim)
        self.register_buffer("hr_init", hr_init)

        self.bipartite_layers = nn.ModuleList([
            BipartiteGNNBlock(hidden_dim, dropout=dropout)
            for _ in range(bipartite_layers)
        ])

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.edge_P = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.edge_P, gain=0.1)  # conservative init to avoid collapse


        self._triu_idx = None
        self._S_b = None
        self._S_b_device = None

    def _get_triu_indices(self, device: torch.device):
        if self._triu_idx is None or self._triu_idx[0].device != device:
            self._triu_idx = torch.triu_indices(
                self.n_hr, self.n_hr, offset=1, device=device,
            )
        return self._triu_idx

    def _get_bipartite_adjacency(self, device: torch.device, dtype: torch.dtype, batch_size: int) -> torch.Tensor:
        """Build normalised bipartite adjacency S_b. Cached per device."""
        if self._S_b is None or self._S_b_device != device:
            n_lr, n_hr = self.n_lr, self.n_hr
            # A_b = [[0, 1], [1, 0]] block structure
            A_b = torch.zeros(self.n_bipartite, self.n_bipartite, device=device, dtype=dtype)
            A_b[:n_lr, n_lr:] = 1.0
            A_b[n_lr:, :n_lr] = 1.0
            deg = A_b.sum(dim=-1).clamp(min=1e-8)
            D_inv_sqrt = torch.diag(deg.pow(-0.5))
            self._S_b = D_inv_sqrt @ A_b @ D_inv_sqrt
            self._S_b_device = device
        S = self._S_b.unsqueeze(0).expand(batch_size, -1, -1)
        return S

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
        B = A_lr.size(0)
        S_lr = self._normalise(A_lr)

        H_lr = self.input_proj(X_lr)
        for layer in self.gcn_layers:
            H_lr = layer(S_lr, H_lr)

        H_hr_init = self.hr_init.unsqueeze(0).expand(B, -1, -1)
        H_b = torch.cat([H_lr, H_hr_init], dim=1)

        S_b = self._get_bipartite_adjacency(A_lr.device, A_lr.dtype, B)
        for layer in self.bipartite_layers:
            H_b = layer(S_b, H_b)

        H_hr = H_b[:, self.n_lr:, :]
        H_hr = self.decoder_norm(H_hr)
        H_hr = F.gelu(H_hr)

        HP = H_hr @ self.edge_P
        A_pred = 0.5 * (HP @ HP.transpose(1, 2) + (HP @ HP.transpose(1, 2)).transpose(1, 2))
        idx = self._get_triu_indices(A_pred.device)
        pred = A_pred[:, idx[0], idx[1]]
        pred = F.softplus(pred)
        # Spec: data range [0,1]; post-process to avoid negatives and cap at 1
        return pred.clamp(min=0.0, max=1.0)
