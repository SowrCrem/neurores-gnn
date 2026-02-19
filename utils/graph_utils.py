"""
Graph and adjacency matrix utility functions.

Covers: normalization, thresholding, and conversion between
numpy adjacency matrices and DGL graph objects.
"""

import numpy as np
import torch
import dgl


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Applies symmetric degree normalization: D^{-1/2} A D^{-1/2}.

    Args:
        adj: Adjacency matrix of shape (N, N).

    Returns:
        Normalized adjacency matrix of shape (N, N).
    """
    raise NotImplementedError


def adj_to_dgl_graph(adj: np.ndarray, threshold: float = 0.0) -> dgl.DGLGraph:
    """
    Converts a weighted adjacency matrix to a DGL graph.

    Edges are added for all entries with |weight| > threshold.
    Edge weights are stored as the 'w' feature.

    Args:
        adj:       Adjacency matrix of shape (N, N).
        threshold: Minimum absolute weight to include an edge.

    Returns:
        DGL graph with N nodes and node features set to the row of adj.
    """
    raise NotImplementedError


def dgl_graph_to_adj(graph: dgl.DGLGraph, num_nodes: int) -> np.ndarray:
    """
    Reconstructs a dense adjacency matrix from a DGL graph's edge weights.

    Args:
        graph:     DGL graph with 'w' edge feature.
        num_nodes: Total number of nodes (used for output shape).

    Returns:
        Dense adjacency matrix of shape (num_nodes, num_nodes).
    """
    raise NotImplementedError
