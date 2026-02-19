"""
Custom DGL layers and graph convolution blocks used by the generator.
"""

import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class GraphConvBlock(nn.Module):
    """Single graph convolution block with optional residual connection."""

    def __init__(self, in_feats: int, out_feats: int, activation=nn.ReLU(), residual: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.residual = residual
        if residual and in_feats != out_feats:
            self.res_proj = nn.Linear(in_feats, out_feats, bias=False)
        else:
            self.res_proj = None

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
