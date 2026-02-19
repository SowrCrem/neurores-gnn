"""
GNN generator: maps Low-Resolution (LR) brain connectivity graphs
to High-Resolution (HR) connectivity graphs.
"""

import torch
import torch.nn as nn
import dgl


class BrainGNNGenerator(nn.Module):
    def __init__(self, lr_nodes: int, hr_nodes: int, hidden_dim: int):
        """
        Args:
            lr_nodes:   Number of ROIs in the LR graph.
            hr_nodes:   Number of ROIs in the target HR graph.
            hidden_dim: Width of hidden graph convolution layers.
        """
        super().__init__()
        self.lr_nodes = lr_nodes
        self.hr_nodes = hr_nodes
        self.hidden_dim = hidden_dim

    def forward(self, lr_graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Args:
            lr_graph: Input LR brain graph with node features.

        Returns:
            Predicted HR adjacency matrix of shape (hr_nodes, hr_nodes).
        """
        raise NotImplementedError
