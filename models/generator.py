"""
GNN generator: maps Low-Resolution (LR) brain connectivity graphs
to High-Resolution (HR) connectivity graphs.

Architecture:
    LR DGL graph (160 nodes, features = adj rows)
        -> Input projection (160-dim -> hidden_dim)
        -> GraphConvBlock x N  (message passing on LR topology)
        -> Node upsample       (learned linear: 160 -> 268 over node dim)
        -> Bilinear edge decode (h W h^T -> 268x268, symmetrised)
        -> Extract upper triangle, clamp >= 0
    Output: (B, 35778) predicted HR edge-weight vector
"""

import torch
import torch.nn as nn
import dgl

from models.layers import GraphConvBlock


class BrainGNNGenerator(nn.Module):
    def __init__(
        self,
        lr_nodes: int = 160,
        hr_nodes: int = 268,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lr_nodes = lr_nodes
        self.hr_nodes = hr_nodes
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(lr_nodes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gnn_layers = nn.ModuleList([
            GraphConvBlock(hidden_dim, hidden_dim, residual=True, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Learned mapping from lr_nodes -> hr_nodes applied per feature channel
        self.upsample = nn.Linear(lr_nodes, hr_nodes)

        # Bilinear decoder weight
        self.edge_W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.edge_W)

        self._triu_idx = None

    def _get_triu_indices(self, device):
        if self._triu_idx is None or self._triu_idx[0].device != device:
            self._triu_idx = torch.triu_indices(
                self.hr_nodes, self.hr_nodes, offset=1, device=device
            )
        return self._triu_idx

    def forward(self, lr_graph: dgl.DGLGraph) -> torch.Tensor:
        feat = lr_graph.ndata['feat']           # (B*N_lr, N_lr)

        h = self.input_proj(feat)               # (B*N_lr, d)

        for layer in self.gnn_layers:
            h = layer(lr_graph, h)              # (B*N_lr, d)

        B = lr_graph.batch_size
        h = h.view(B, self.lr_nodes, self.hidden_dim)

        # Upsample nodes: (B, 160, d) -> (B, 268, d)
        h = self.upsample(h.transpose(1, 2)).transpose(1, 2)

        # Bilinear edge prediction + symmetrise
        hw = h @ self.edge_W                    # (B, 268, d)
        A = hw @ h.transpose(1, 2)              # (B, 268, 268)
        A = 0.5 * (A + A.transpose(1, 2))

        idx = self._get_triu_indices(A.device)
        pred = A[:, idx[0], idx[1]]             # (B, 35778)
        return torch.clamp(pred, min=0.0)
