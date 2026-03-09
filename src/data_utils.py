"""Shared tensor helpers for vectorized LR/HR brain graph data."""

from __future__ import annotations

import numpy as np
import torch

from utils.matrix_vectorizer import MatrixVectorizer


def vec_to_adj(vec: torch.Tensor, n: int) -> torch.Tensor:
    """Convert upper-triangle vectors (no diagonal) to symmetric adjacency tensors."""
    batch_size = vec.shape[0]
    device = vec.device
    dtype = vec.dtype

    vec_np = vec.detach().cpu().numpy()
    mats = [MatrixVectorizer.anti_vectorize(vec_np[i], n, include_diagonal=False) for i in range(batch_size)]
    return torch.from_numpy(np.stack(mats)).to(device=device, dtype=dtype)


def adj_to_vec(adj: torch.Tensor) -> torch.Tensor:
    """Convert symmetric adjacency tensors to upper-triangle vectors (no diagonal)."""
    batch_size = adj.shape[0]
    device = adj.device
    dtype = adj.dtype

    adj_np = adj.detach().cpu().numpy()
    vecs = [MatrixVectorizer.vectorize(adj_np[i], include_diagonal=False) for i in range(batch_size)]
    return torch.from_numpy(np.stack(vecs)).to(device=device, dtype=dtype)


def lr_node_features(adj_lr: torch.Tensor) -> torch.Tensor:
    """Build minimal LR node features: constant 1 and weighted degree."""
    deg = adj_lr.sum(dim=-1, keepdim=True)
    ones = torch.ones_like(deg)
    return torch.cat([ones, deg], dim=-1)


def to_tensor(x: np.ndarray, device: str | torch.device) -> torch.Tensor:
    """Move a numpy array to a float tensor on the target device."""
    return torch.from_numpy(x).float().to(device)
