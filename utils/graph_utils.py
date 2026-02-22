"""
Graph and adjacency matrix utility functions.

Covers:
  - Data pre-processing (NaN/negative cleaning as per competition spec)
  - Symmetric degree normalisation
  - Conversion between numpy adjacency matrices and DGL graph objects
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def preprocess_matrix(adj: np.ndarray) -> np.ndarray:
    """
    Cleans a single adjacency matrix per competition spec:
      1. Replace NaN values with 0.
      2. Replace negative values with 0.

    Args:
        adj: Adjacency matrix of shape (N, N).

    Returns:
        Cleaned adjacency matrix of the same shape.
    """
    adj = np.nan_to_num(adj, nan=0.0)
    adj = np.clip(adj, a_min=0.0, a_max=None)
    return adj


def preprocess_data(X: np.ndarray) -> np.ndarray:
    """
    Applies preprocess_matrix row-wise to a vectorised data matrix.

    Args:
        X: Data matrix of shape (N_subjects, n_features) in vectorised form.

    Returns:
        Cleaned data matrix of the same shape.
    """
    X = np.nan_to_num(X, nan=0.0)
    X = np.clip(X, a_min=0.0, a_max=None)
    return X


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Applies symmetric degree normalisation: D^{-1/2} A D^{-1/2}.

    Zero-degree nodes are handled safely (degree set to 1 to avoid division
    by zero, resulting in a zero row/column after multiplication).

    Args:
        adj: Adjacency matrix of shape (N, N).

    Returns:
        Normalised adjacency matrix of shape (N, N).
    """
    degree = adj.sum(axis=1)
    # Avoid division by zero for isolated nodes
    degree[degree == 0] = 1.0
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    return d_inv_sqrt @ adj @ d_inv_sqrt


# ---------------------------------------------------------------------------
# DGL conversions
# ---------------------------------------------------------------------------

def adj_to_dgl_graph(adj: np.ndarray, threshold: float = 0.0):
    """
    Converts a weighted adjacency matrix to a DGL graph.

    Edges are added for all entries with weight > threshold.
    Node features ('feat') are set to the rows of the adjacency matrix so
    each node carries a summary of its connectivity profile.
    Edge weights are stored as the 'w' feature.

    Args:
        adj:       Pre-processed adjacency matrix of shape (N, N).
        threshold: Minimum weight to include an edge (default 0 keeps all
                   non-zero edges).

    Returns:
        DGL graph with N nodes.
    """
    import dgl, torch
    src, dst = np.where(adj > threshold)
    weights = adj[src, dst].astype(np.float32)

    g = dgl.graph((src, dst))
    g.edata['w'] = torch.tensor(weights)
    g.ndata['feat'] = torch.tensor(adj, dtype=torch.float32)
    return g


def dgl_graph_to_adj(graph, num_nodes: int) -> np.ndarray:
    """
    Reconstructs a dense adjacency matrix from a DGL graph's 'w' edge feature.

    Args:
        graph:     DGL graph with 'w' edge feature.
        num_nodes: Total number of nodes (used for output shape).

    Returns:
        Dense adjacency matrix of shape (num_nodes, num_nodes).
    """
    import torch
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    src, dst = graph.edges()
    weights = graph.edata['w'].numpy()
    adj[src.numpy(), dst.numpy()] = weights
    return adj
