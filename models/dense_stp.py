import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dense_gcn import DenseGCNBlock

class EdgeMLPDecoder(nn.Module):
    """
    Decodes HR edge probabilities using an Edge-MLP over node features.
    This replaces the Bilinear bottleneck and directly operates on the edge-space representations.
    Ensures symmetry by operating on permutation-invariant combinations of node pairs (e.g. Hadamard or Sum/Diff).
    """
    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        # E_ij = (u * v) || (u + v)
        self.mlp = nn.Sequential(
            nn.Linear(in_features * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, H: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, N, D) HR node embeddings
            idx: (2, E) Upper-triangular indices, where E = 35778
        Returns:
            edges: (B, E) predicted edge weights
        """
        B, N, D = H.shape
        u = H[:, idx[0], :]  # (B, E, D)
        v = H[:, idx[1], :]  # (B, E, D)
        
        # Symmetric edge representation
        e_uv = torch.cat([u * v, u + v], dim=-1) # (B, E, D * 2)
        
        edge_preds = self.mlp(e_uv).squeeze(-1) # (B, E)
        return edge_preds

class DenseSTPGenerator(nn.Module):
    """
    STP-GSR inspired Edge-Dual network for brain graph super-resolution.
    Uses GCN over LR space, node upsample, and then an Edge-Space Dual MLP Decoder
    to circumvent the bilinear rank bottleneck.
    """
    def __init__(
        self,
        n_lr: int = 160,
        n_hr: int = 268,
        hidden_dim: int = 192,
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

        input_dim = n_lr + lap_pe_dim + pearl_pe_dim
        
        if self.pearl_pe_dim > 0:
            self.pearl_pe = nn.Parameter(torch.empty(1, n_lr, pearl_pe_dim))
            nn.init.normal_(self.pearl_pe, mean=0.0, std=0.02)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gcn_layers = nn.ModuleList([
            DenseGCNBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.upsample = nn.Linear(n_lr, n_hr)
        
        # Edge space decoder
        self.edge_decoder = EdgeMLPDecoder(in_features=hidden_dim, hidden_dim=64, dropout=dropout)

        self._triu_idx = None

    def _get_triu_indices(self, device):
        if self._triu_idx is None or self._triu_idx[0].device != device:
            self._triu_idx = torch.triu_indices(
                self.n_hr, self.n_hr, offset=1, device=device,
            )
        return self._triu_idx

    @staticmethod
    def _normalise(A: torch.Tensor) -> torch.Tensor:
        N = A.size(-1)
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
        A_hat = A + I
        deg = A_hat.sum(dim=-1).clamp(min=1e-8)
        D_inv_sqrt = torch.diag_embed(deg.pow(-0.5))
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    def forward(self, A_lr: torch.Tensor, X_lr: torch.Tensor, lap_pe: torch.Tensor | None = None) -> torch.Tensor:
        S = self._normalise(A_lr)

        if lap_pe is not None and self.lap_pe_dim > 0:
            X_lr = torch.cat([X_lr, lap_pe], dim=-1)
            
        if self.pearl_pe_dim > 0:
            pearl_feat = self.pearl_pe.expand(A_lr.size(0), -1, -1)
            X_lr = torch.cat([X_lr, pearl_feat], dim=-1)

        H = self.input_proj(X_lr)

        for layer in self.gcn_layers:
            H = layer(S, H)

        # Node upsample: (B, n_lr, d) -> (B, n_hr, d)
        H = self.upsample(H.transpose(1, 2)).transpose(1, 2)
        H = F.gelu(H) # important for nonlinear mapped reps

        idx = self._get_triu_indices(A_lr.device)
        pred = self.edge_decoder(H, idx) # (B, 35778)
        
        if self.raw_output:
            return pred
        return F.softplus(pred)
