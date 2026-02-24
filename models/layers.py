"""
Custom DGL graph convolution blocks used by the generator.
"""

import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class GraphConvBlock(nn.Module):
    """GCN-style convolution with edge weights, residual connection, and dropout.

    Implements symmetric-normalised message passing:
        h' = Dropout(act(LN(D^{-1/2} A_w D^{-1/2} H W) + residual))
    where A_w is the weighted adjacency stored in ``graph.edata['w']``.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        activation=nn.ReLU(),
        residual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(out_feats)
        self.drop = nn.Dropout(dropout)
        self.residual = residual
        if residual and in_feats != out_feats:
            self.res_proj = nn.Linear(in_feats, out_feats, bias=False)
        else:
            self.res_proj = None

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            identity = feat

            # Symmetric degree normalisation  D^{-1/2}
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).unsqueeze(1)

            feat = feat * norm

            graph.ndata['h'] = feat
            if 'w' in graph.edata:
                w = graph.edata['w']
                if w.dim() == 1:
                    w = w.unsqueeze(-1)
                graph.edata['_w'] = w
                graph.update_all(fn.u_mul_e('h', '_w', 'm'), fn.sum('m', 'agg'))
            else:
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'agg'))

            feat = graph.ndata['agg']
            feat = feat * norm

            feat = self.linear(feat)
            feat = self.layer_norm(feat)

            if self.residual:
                if self.res_proj is not None:
                    identity = self.res_proj(identity)
                feat = feat + identity

            feat = self.activation(feat)
            feat = self.drop(feat)
            return feat
