"""
Simple Graph Convolution (SGC) baseline for brain graph super-resolution.

Reference: Wu et al., "Simplifying Graph Convolutional Networks", ICML 2019.
Adapted from DGL Tutorial 2:
    https://github.com/basiralab/DGL/tree/main/Tutorials/Tutorial-2

SGC removes all nonlinearities between GCN layers, collapsing K layers of
message passing into a single pre-computable matrix power:

    H = S^K X          (no learnable weights — just neighbourhood averaging)
    Y = H Theta         (single linear transformation)

where S = D^{-1/2} (A + I) D^{-1/2}.

This model uses the same dense-tensor interface as gcn-encoder-ca-decoder
so it can be trained with the same data pipeline.
"""

import torch
import torch.nn as nn


class SGCBaseline(nn.Module):
    """Naive SGC baseline for LR → HR brain graph super-resolution.

    Architecture:
        1. Normalise LR adjacency: S = D^{-1/2} (A+I) D^{-1/2}
        2. K-hop propagation (no learnable weights): H = S^K @ X
        3. Single linear projection: H' = H @ Theta
        4. Learned node upsample: 160 → 268
        5. Bilinear edge decoder: A_hr = H' W H'^T, symmetrised
        6. Extract upper triangle, clamp >= 0
    """

    def __init__(
        self,
        n_lr: int = 160,
        n_hr: int = 268,
        d_model: int = 64,
        K: int = 2,
        in_node_feat_dim: int = 2,
    ):
        super().__init__()
        self.n_lr = n_lr
        self.n_hr = n_hr
        self.K = K

        self.linear = nn.Linear(in_node_feat_dim, d_model)
        self.upsample = nn.Linear(n_lr, n_hr)
        self.W = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.W)

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

    def forward(self, A_lr: torch.Tensor, X_lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_lr: (B, n_lr, n_lr) weighted symmetric adjacency
            X_lr: (B, n_lr, in_node_feat_dim) node features
        Returns:
            pred: (B, n_hr*(n_hr-1)/2) predicted HR edge-weight vector
        """
        S = self._normalise(A_lr)

        # K-hop propagation — no learnable weights
        H = X_lr
        for _ in range(self.K):
            H = S @ H                              # (B, n_lr, feat_dim)

        # Single linear transformation (the only trainable layer in propagation path)
        H = self.linear(H)                          # (B, n_lr, d_model)

        # Node upsample: (B, n_lr, d) → (B, n_hr, d)
        H = self.upsample(H.transpose(1, 2)).transpose(1, 2)

        # Bilinear edge decode + symmetrise
        HW = H @ self.W                             # (B, n_hr, d)
        A = HW @ H.transpose(1, 2)                  # (B, n_hr, n_hr)
        A = 0.5 * (A + A.transpose(1, 2))

        idx = self._get_triu_indices(A.device)
        pred = A[:, idx[0], idx[1]]                  # (B, 35778)
        return torch.clamp(pred, min=0.0)
