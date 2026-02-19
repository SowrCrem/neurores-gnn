"""
Evaluation metrics for the Brain Graph Super-Resolution Challenge.

Primary metric: Mean Columnwise MAE between predicted and ground-truth
HR adjacency matrices.
"""

import numpy as np
import torch


def mean_columnwise_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the mean of per-column MAE values across the HR adjacency matrix.

    Args:
        y_pred: Predicted HR matrix, shape (N, hr_nodes, hr_nodes).
        y_true: Ground-truth HR matrix, shape (N, hr_nodes, hr_nodes).

    Returns:
        Scalar mean columnwise MAE.
    """
    raise NotImplementedError


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Element-wise MAE over the full prediction array."""
    return float(np.mean(np.abs(y_pred - y_true)))
