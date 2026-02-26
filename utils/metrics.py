"""
Evaluation metrics for the Brain Graph Super-Resolution Challenge.

Primary metric:
    Mean Absolute Error (MAE) computed over vectorized (flattened) adjacency
    matrices, optionally excluding the diagonal.

This module also computes several secondary metrics:
    - Pearson Correlation Coefficient (PCC) over vectorized matrices
    - Jensen–Shannon Distance (JSD) between normalized edge-weight distributions
    - Avg. node-wise MAE for multiple graph-derived measures:
        * PageRank Centrality (PC)
        * Eigenvector Centrality (EC)
        * Betweenness Centrality (BC)
    Additional geometric/topological measures (spec §II.A.a):
        * Node Strength (weighted degree)
        * Weighted Clustering Coefficient

Notes:
    - Inputs are assumed to be non-negative (if not, we clamp negatives to 0).
    - JSD requires probability distributions; we normalize vectors to sum to 1.
    - Eigenvector centrality may fail to converge; we use a robust fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from utils.matrix_vectorizer import MatrixVectorizer


# -----------------------------------------------------------------------------
# Metric ordering (controls plotting order and returned dictionary order)
# -----------------------------------------------------------------------------
METRIC_ORDER = [
    "MAE", "PCC", "JSD",
    "MAE (PC)", "MAE (EC)", "MAE (BC)",
    "MAE (Strength)", "MAE (Clustering)",
]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _to_nonneg(x: np.ndarray) -> np.ndarray:
    """
    Clamp an array to be non-negative by setting all negative values to 0.

    Parameters
    ----------
    x : np.ndarray
        Input array (any shape). Typically adjacency matrices of shape:
        (num_samples, num_nodes, num_nodes).

    Returns
    -------
    np.ndarray
        Copy of `x` with negative values replaced by 0.
    """
    x = np.asarray(x).copy()
    x[x < 0] = 0
    return x


def _mask_diagonal(mat: np.ndarray) -> np.ndarray:
    """
    Zero out the diagonal of a square matrix.

    This is useful when self-connections are not meaningful (common in
    adjacency/functional connectivity matrices), and you want to exclude them
    from evaluation.

    Parameters
    ----------
    mat : np.ndarray
        Square matrix of shape (N, N).

    Returns
    -------
    np.ndarray
        Copy of `mat` with diagonal elements set to 0.
    """
    mat = mat.copy()
    np.fill_diagonal(mat, 0.0)
    return mat


def _normalize_to_prob(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a non-negative vector into a probability distribution.

    Jensen–Shannon distance (JSD) requires inputs that represent probability
    distributions (non-negative and sum to 1). This function ensures that.

    Edge case:
        If the vector sums to ~0 (all zeros), we return a uniform distribution
        to avoid division by zero and undefined JSD behavior.

    Parameters
    ----------
    v : np.ndarray
        Input vector of edge weights (1D).
    eps : float, optional
        Small constant to detect near-zero sums.

    Returns
    -------
    np.ndarray
        Probability distribution vector of same shape as `v`.
    """
    v = np.maximum(v.astype(float), 0.0)
    s = v.sum()
    if s <= eps:
        # All-zero (or extremely small) vector -> return uniform distribution
        return np.full_like(v, 1.0 / v.size, dtype=float)
    return v / s


def _safe_ec(G: nx.Graph) -> dict:
    """
    Compute eigenvector centrality robustly, handling disconnected graphs.

    Attempts: (1) numpy method, (2) power iteration, (3) EC on largest
    connected component with 0 for nodes outside it.
    """
    try:
        return nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        pass
    try:
        return nx.eigenvector_centrality(G, weight="weight", max_iter=2000, tol=1e-8)
    except Exception:
        pass
    result = {n: 0.0 for n in G.nodes()}
    largest_cc = max(nx.connected_components(G), key=len)
    sub = G.subgraph(largest_cc).copy()
    try:
        ec = nx.eigenvector_centrality_numpy(sub, weight="weight")
    except Exception:
        try:
            ec = nx.eigenvector_centrality(sub, weight="weight", max_iter=2000, tol=1e-8)
        except Exception:
            return result
    result.update(ec)
    return result


def _avg_node_mae(d_pred: dict, d_gt: dict) -> float:
    """
    Compute MAE between two node-score dictionaries (node -> value).

    This is used for metrics such as centralities where each node has a score.
    We compute the MAE across nodes for one sample.

    Parameters
    ----------
    d_pred : dict
        Predicted node scores (node -> value).
    d_gt : dict
        Ground-truth node scores (node -> value).

    Returns
    -------
    float
        Mean Absolute Error across node scores.
    """
    nodes = sorted(d_gt.keys())
    p = np.array([d_pred[n] for n in nodes], dtype=float)
    g = np.array([d_gt[n] for n in nodes], dtype=float)
    return mean_absolute_error(g, p)


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------
def evaluate_fold(
    pred_mats: np.ndarray,
    gt_mats: np.ndarray,
    ignore_diagonal: bool = True,
    verbose: bool = True,
    cache_path: str | Path | None = None,
) -> dict:
    """
    Evaluate a fold (or any batch) of predicted vs ground-truth adjacency matrices.

    The function computes:
        Matrix-level metrics (over concatenated vectorized matrices):
            - MAE
            - PCC
            - JSD (after converting vectors to probability distributions)
        Graph-level metrics (average across samples of node-level MAE):
            - MAE (PC): PageRank centrality
            - MAE (EC): Eigenvector centrality
            - MAE (BC): Betweenness centrality
            - MAE (Strength): node strength / weighted degree
            - MAE (Clustering): weighted clustering coefficient

    Parameters
    ----------
    pred_mats : np.ndarray
        Predicted adjacency matrices of shape (S, N, N) where:
            S = number of samples in the fold
            N = number of nodes/ROIs
        Values are expected to be non-negative; negatives are clamped to 0.
    gt_mats : np.ndarray
        Ground-truth adjacency matrices of shape (S, N, N).
        Values are expected to be non-negative; negatives are clamped to 0.
    ignore_diagonal : bool, optional
        If True, diagonal values are set to 0 before vectorization and graph
        construction (ignoring self-loops).
    verbose : bool, optional
        If True, print each metric value (simple logging).

    Returns
    -------
    dict
        Dictionary of computed metrics in the order defined by `METRIC_ORDER`.
        Keys:
            "MAE", "PCC", "JSD",
            "MAE (PC)", "MAE (EC)", "MAE (BC)",
            "MAE (Strength)", "MAE (Clustering)"

    Notes
    -----
    - PCC is set to NaN if either vector has ~zero variance.
    - JSD is computed on probability distributions derived from concatenated
      vectorized adjacency values.
    """
    # Ensure non-negative values (challenge data is typically pre-processed,
    # but predictions might contain negatives).
    pred_mats = _to_nonneg(pred_mats)
    gt_mats = _to_nonneg(gt_mats)

    # -------------------------------------------------------------------------
    # 1) Matrix-level metrics computed on concatenated, vectorized edges
    # -------------------------------------------------------------------------
    pred_vecs, gt_vecs = [], []
    for i in range(pred_mats.shape[0]):
        p = _mask_diagonal(pred_mats[i]) if ignore_diagonal else pred_mats[i]
        g = _mask_diagonal(gt_mats[i]) if ignore_diagonal else gt_mats[i]
        pred_vecs.append(MatrixVectorizer.vectorize(p))
        gt_vecs.append(MatrixVectorizer.vectorize(g))

    pred_1d = np.concatenate(pred_vecs)
    gt_1d = np.concatenate(gt_vecs)

    # MAE over all edges across all samples
    mae = mean_absolute_error(gt_1d, pred_1d)

    # PCC over all edges across all samples (guard for constant vectors)
    if np.std(pred_1d) < 1e-12 or np.std(gt_1d) < 1e-12:
        pcc = np.nan
    else:
        pcc = pearsonr(pred_1d, gt_1d)[0]

    # Jensen–Shannon distance over normalized distributions
    jsd = jensenshannon(_normalize_to_prob(pred_1d), _normalize_to_prob(gt_1d))

    # -------------------------------------------------------------------------
    # 2) Graph-level metrics: compute node-measures per sample, then average MAE
    #    Supports incremental caching so the run can resume after interruption.
    # -------------------------------------------------------------------------
    GRAPH_METRIC_KEYS = ["MAE (PC)", "MAE (EC)", "MAE (BC)", "MAE (Strength)", "MAE (Clustering)"]

    cached_samples: list[dict] = []
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as _cf:
                cached_samples = json.load(_cf)
            if verbose:
                print(f"  Resuming graph metrics from cache ({len(cached_samples)}/{pred_mats.shape[0]} samples done)")

    n_samples = pred_mats.shape[0]
    start_idx = len(cached_samples)

    for i in tqdm(range(start_idx, n_samples), desc="Graph metrics", disable=not verbose,
                  initial=start_idx, total=n_samples):
        p = _mask_diagonal(pred_mats[i]) if ignore_diagonal else pred_mats[i]
        g = _mask_diagonal(gt_mats[i]) if ignore_diagonal else gt_mats[i]

        Gp = nx.from_numpy_array(p, edge_attr="weight")
        Gg = nx.from_numpy_array(g, edge_attr="weight")

        pred_pc = nx.pagerank(Gp, weight="weight")
        gt_pc = nx.pagerank(Gg, weight="weight")

        pred_ec = _safe_ec(Gp)
        gt_ec = _safe_ec(Gg)

        pred_bc = nx.betweenness_centrality(Gp, weight="weight")
        gt_bc = nx.betweenness_centrality(Gg, weight="weight")

        pred_strength = dict(Gp.degree(weight="weight"))
        gt_strength = dict(Gg.degree(weight="weight"))

        pred_clust = nx.clustering(Gp, weight="weight")
        gt_clust = nx.clustering(Gg, weight="weight")

        cached_samples.append({
            "MAE (PC)": float(_avg_node_mae(pred_pc, gt_pc)),
            "MAE (EC)": float(_avg_node_mae(pred_ec, gt_ec)),
            "MAE (BC)": float(_avg_node_mae(pred_bc, gt_bc)),
            "MAE (Strength)": float(_avg_node_mae(pred_strength, gt_strength)),
            "MAE (Clustering)": float(_avg_node_mae(pred_clust, gt_clust)),
        })

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as _cf:
                json.dump(cached_samples, _cf)

    maes = {k: [s[k] for s in cached_samples] for k in GRAPH_METRIC_KEYS}

    # Average graph-level MAEs across samples
    out = {
        "MAE": float(mae),
        "PCC": float(pcc) if np.isfinite(pcc) else np.nan,
        "JSD": float(jsd),
        **{k: float(np.mean(v)) for k, v in maes.items()},
    }

    # Enforce ordering/presence (useful for consistent plotting)
    ordered_out = {k: out[k] for k in METRIC_ORDER}

    # -------------------------------------------------------------------------
    # Simple metric logging (prints)
    # -------------------------------------------------------------------------
    if verbose:
        print("Evaluation metrics:")
        for k in METRIC_ORDER:
            v = ordered_out[k]
            # Keep NaN readable
            if isinstance(v, float) and np.isnan(v):
                print(f"  {k:16s}: NaN")
            else:
                print(f"  {k:16s}: {v:.6f}" if isinstance(v, (float, np.floating)) else f"  {k:16s}: {v}")

    return ordered_out