"""
Variational Graph Auto-Encoder (VGAE) baseline for brain graph super-resolution.

Simple architecture: 2-layer GCN encoder, deterministic latent, inner-product decoder.
"""

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """Simple GCN layer with symmetric degree normalization."""
    
    def __init__(self, in_feats, out_feats, activation=None, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, A, X):
        """
        Args:
            A: (B, n, n) adjacency matrix
            X: (B, n, in_feats) node features
        Returns:
            H: (B, n, out_feats) node features
        """
        # Symmetric degree normalization: D^{-1/2} A D^{-1/2}
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, n, 1)
        D_inv_sqrt = 1.0 / torch.sqrt(deg)  # (B, n, 1)
        
        # D^{-1/2} A D^{-1/2} X
        A_norm = A * D_inv_sqrt * D_inv_sqrt.transpose(1, 2)  # (B, n, n)
        H = A_norm @ X  # (B, n, in_feats)
        
        # Linear transformation
        H = self.linear(H)  # (B, n, out_feats)
        
        if self.activation is not None:
            H = self.activation(H)
        
        H = self.dropout(H)
        return H


class VGAEBaseline(nn.Module):
    """Simple VGAE baseline for LR → HR brain graph super-resolution.
    
    Architecture:
        1. 2-layer GCN encoder: X → H1 → Z (latent embedding, no KL divergence)
        2. Learned node upsample: 160 → 268
        3. Inner-product decoder: A_hr = Z Z^T, symmetrised
        4. Extract upper triangle, clamp >= 0
    """
    
    def __init__(
        self,
        n_lr: int = 160,
        n_hr: int = 268,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        in_node_feat_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_lr = n_lr
        self.n_hr = n_hr
        self.latent_dim = latent_dim
        
        # 2-layer GCN encoder
        self.gcn1 = GCNLayer(in_node_feat_dim, hidden_dim, activation=nn.ReLU(), dropout=dropout)
        self.gcn2 = GCNLayer(hidden_dim, latent_dim, activation=None, dropout=dropout)
        
        # Node upsample layer: 160 → 268
        self.upsample = nn.Linear(n_lr, n_hr)
        
        # Upper-triangle indices for extraction
        self._triu_idx = None
    
    def _get_triu_indices(self, device):
        if self._triu_idx is None or self._triu_idx[0].device != device:
            self._triu_idx = torch.triu_indices(
                self.n_hr, self.n_hr, offset=1, device=device,
            )
        return self._triu_idx
    
    def forward(self, A_lr: torch.Tensor, X_lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_lr: (B, n_lr, n_lr) LR adjacency matrix
            X_lr: (B, n_lr, in_node_feat_dim) LR node features
        Returns:
            pred: (B, n_hr*(n_hr-1)/2) predicted HR edge-weight vector
        """
        # 2-layer GCN encoder
        H1 = self.gcn1(A_lr, X_lr)  # (B, n_lr, hidden_dim)
        Z = self.gcn2(A_lr, H1)      # (B, n_lr, latent_dim)
        
        # Node upsample: (B, n_lr, latent_dim) → (B, n_hr, latent_dim)
        Z_up = self.upsample(Z.transpose(1, 2)).transpose(1, 2)  # (B, n_hr, latent_dim)
        
        # Inner-product decoder: A_hr = Z_up @ Z_up^T
        A = Z_up @ Z_up.transpose(1, 2)  # (B, n_hr, n_hr)
        
        # Symmetrise
        A = 0.5 * (A + A.transpose(1, 2))
        
        # Extract upper triangle (no diagonal)
        idx = self._get_triu_indices(A.device)
        pred = A[:, idx[0], idx[1]]  # (B, n_hr*(n_hr-1)/2)
        
        # Clamp to non-negative
        return torch.clamp(pred, min=0.0)
