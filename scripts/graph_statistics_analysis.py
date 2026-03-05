#!/usr/bin/env python3
"""
Statistical analysis of LR vs HR brain connectivity graphs.

Derives from src/dataset.py, utils/matrix_vectorizer.py, utils/graph_utils.py
and the training CSVs. No speculation — all quantities computed from data.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.linalg import eigsh

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.matrix_vectorizer import MatrixVectorizer
from utils.graph_utils import preprocess_data

LR_NODES = 160
HR_NODES = 268
LR_FEATURES = LR_NODES * (LR_NODES - 1) // 2   # 12720
HR_FEATURES = HR_NODES * (HR_NODES - 1) // 2   # 35778


def load_graphs(data_dir: str = "data"):
    """Load LR and HR adjacency matrices using the same pipeline as dataset.py."""
    vectorizer = MatrixVectorizer()

    lr_df = pd.read_csv(os.path.join(data_dir, "lr_train.csv"), header=0)
    hr_df = pd.read_csv(os.path.join(data_dir, "hr_train.csv"), header=0)

    lr_data = preprocess_data(lr_df.values.astype(np.float32))
    hr_data = preprocess_data(hr_df.values.astype(np.float32))

    lr_adjs = [vectorizer.anti_vectorize(lr_data[i], LR_NODES) for i in range(len(lr_data))]
    hr_adjs = [vectorizer.anti_vectorize(hr_data[i], HR_NODES) for i in range(len(hr_data))]

    return np.array(lr_adjs), np.array(hr_adjs), lr_data, hr_data


def compute_edge_stats(adjs: np.ndarray, name: str) -> dict:
    """Compute mean, variance, skewness over edge weights (upper triangle)."""
    n_samples, n, _ = adjs.shape
    triu_idx = np.triu_indices(n, k=1)
    edges = np.array([adj[triu_idx] for adj in adjs])  # (N_samples, n_edges)

    flat = edges.flatten()
    nonzero = flat[flat > 0]

    return {
        "name": name,
        "n_samples": n_samples,
        "n_edges": edges.shape[1],
        "mean_all": float(np.mean(flat)),
        "var_all": float(np.var(flat)),
        "skew_all": float(stats.skew(flat)) if len(flat) > 0 else np.nan,
        "mean_nonzero": float(np.mean(nonzero)) if len(nonzero) > 0 else np.nan,
        "var_nonzero": float(np.var(nonzero)) if len(nonzero) > 0 else np.nan,
        "skew_nonzero": float(stats.skew(nonzero)) if len(nonzero) > 2 else np.nan,
    }


def compute_sparsity(adjs: np.ndarray) -> dict:
    """Sparsity = fraction of zero edges (upper triangle)."""
    n_samples, n, _ = adjs.shape
    triu_idx = np.triu_indices(n, k=1)
    n_possible = n * (n - 1) // 2

    sparsities = []
    for adj in adjs:
        edges = adj[triu_idx]
        sparsities.append(np.sum(edges == 0) / n_possible)

    return {
        "mean_sparsity": float(np.mean(sparsities)),
        "std_sparsity": float(np.std(sparsities)),
        "min_sparsity": float(np.min(sparsities)),
        "max_sparsity": float(np.max(sparsities)),
    }


def compute_node_strength_dist(adjs: np.ndarray) -> dict:
    """Node strength = row sum of adjacency. Distribution over samples and nodes."""
    n_samples, n, _ = adjs.shape
    strengths = np.array([adj.sum(axis=1) for adj in adjs])  # (N_samples, n_nodes)

    flat = strengths.flatten()
    return {
        "mean_strength": float(np.mean(flat)),
        "var_strength": float(np.var(flat)),
        "skew_strength": float(stats.skew(flat)) if len(flat) > 2 else np.nan,
        "mean_per_sample": [float(np.mean(s)) for s in strengths],
        "std_per_sample": [float(np.std(s)) for s in strengths],
    }


def compute_spectral_decay(adjs: np.ndarray, n_eigs: int = 50) -> dict:
    """Eigenvalue decay: largest n_eigs eigenvalues of symmetric adjacency."""
    n_samples = adjs.shape[0]
    all_eigs = []

    for i in range(min(n_samples, 50)):  # Subsample if many
        adj = adjs[i]
        try:
            eigs, _ = np.linalg.eigh(adj)
            eigs = np.sort(eigs)[::-1]  # Descending
            all_eigs.append(eigs[:n_eigs])
        except np.linalg.LinAlgError:
            continue

    all_eigs = np.array(all_eigs)
    mean_eigs = np.mean(all_eigs, axis=0)
    std_eigs = np.std(all_eigs, axis=0)
    mean_abs = np.mean(np.abs(all_eigs), axis=0)

    return {
        "mean_eigenvalues": mean_eigs.tolist(),
        "std_eigenvalues": std_eigs.tolist(),
        "eig_decay_ratio_10": float(mean_eigs[9] / mean_eigs[0]) if mean_eigs[0] > 1e-10 else np.nan,
        "eig_decay_ratio_50": float(mean_eigs[-1] / mean_eigs[0]) if mean_eigs[0] > 1e-10 else np.nan,
        "eig_decay_abs_10": float(mean_abs[9] / mean_abs[0]) if mean_abs[0] > 1e-10 else np.nan,
        "eig_decay_abs_50": float(mean_abs[-1] / mean_abs[0]) if mean_abs[0] > 1e-10 else np.nan,
    }


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Effective rank = exp(entropy of normalized eigenvalue distribution)."""
    eigs = np.abs(eigenvalues)
    eigs = eigs[eigs > 1e-12]
    if len(eigs) == 0:
        return 0.0
    p = eigs / eigs.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)


def compute_low_rank_metrics(adjs: np.ndarray, n_eigs: int = 268) -> dict:
    """Quantify low-rank: effective rank, eigenvalue decay, Frobenius norm concentration."""
    n_samples = min(adjs.shape[0], 30)
    n = adjs.shape[1]
    n_eigs = min(n_eigs, n)

    eff_ranks = []
    frob_concentrations = []  # sum of top-k eig^2 / total frob^2

    for i in range(n_samples):
        adj = adjs[i]
        try:
            eigs, _ = np.linalg.eigh(adj)
            eigs = np.sort(np.abs(eigs))[::-1]
            eff_ranks.append(effective_rank(eigs))
            frob_sq = np.sum(eigs ** 2)
            if frob_sq > 1e-12:
                for k in [10, 50, 100]:
                    if k <= len(eigs):
                        conc = np.sum(eigs[:k] ** 2) / frob_sq
                        frob_concentrations.append((k, conc))
        except np.linalg.LinAlgError:
            continue

    # Aggregate Frobenius concentration by k
    k10 = [c for k, c in frob_concentrations if k == 10]
    k50 = [c for k, c in frob_concentrations if k == 50]
    k100 = [c for k, c in frob_concentrations if k == 100]

    return {
        "effective_rank_mean": float(np.mean(eff_ranks)),
        "effective_rank_std": float(np.std(eff_ranks)),
        "frob_concentration_k10_mean": float(np.mean(k10)) if k10 else np.nan,
        "frob_concentration_k50_mean": float(np.mean(k50)) if k50 else np.nan,
        "frob_concentration_k100_mean": float(np.mean(k100)) if k100 else np.nan,
    }


def lr_hr_mapping_smoothness(lr_data: np.ndarray, hr_data: np.ndarray, n_neighbors: int = 5) -> dict:
    """
    Evaluate LR→HR mapping: if smooth, nearby LR points should map to nearby HR points.
    Use k-NN in LR space, compute mean HR distance within neighborhood vs outside.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    n_samples = lr_data.shape[0]
    if n_samples < n_neighbors + 5:
        return {"error": "Insufficient samples for k-NN analysis"}

    # Standardize
    scaler_lr = StandardScaler()
    lr_scaled = scaler_lr.fit_transform(lr_data)
    hr_scaled = StandardScaler().fit_transform(hr_data)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(lr_scaled)
    distances_lr, indices = nn.kneighbors(lr_scaled)

    # For each point: mean HR distance to k nearest neighbors vs mean HR distance to rest
    in_dist = []
    out_dist = []

    for i in range(n_samples):
        neighbors = indices[i, 1:]  # Exclude self
        hr_i = hr_scaled[i]
        hr_neighbors = hr_scaled[neighbors]
        hr_rest = np.delete(hr_scaled, np.append([i], neighbors), axis=0)

        d_in = np.mean(np.linalg.norm(hr_neighbors - hr_i, axis=1))
        d_out = np.mean(np.linalg.norm(hr_rest - hr_i, axis=1)) if len(hr_rest) > 0 else np.nan

        in_dist.append(d_in)
        out_dist.append(d_out)

    in_dist = np.array(in_dist)
    out_arr = np.array(out_dist)
    out_dist = out_arr[~np.isnan(out_arr)]

    ratio = np.mean(in_dist) / np.mean(out_dist) if np.mean(out_dist) > 1e-10 else np.nan

    return {
        "mean_hr_dist_within_knn": float(np.mean(in_dist)),
        "mean_hr_dist_outside_knn": float(np.mean(out_dist)),
        "smoothness_ratio": float(ratio),  # <1 suggests smooth mapping
        "n_neighbors": n_neighbors,
    }


def cv_stability_analysis(n_samples: int = 167, n_splits: int = 3, n_bootstrap: int = 500) -> dict:
    """
    Assess 3-fold CV stability: bootstrap fold assignment variance.
    With 167 samples, 3-fold gives ~111 train, ~56 val per fold.
    """
    from sklearn.model_selection import KFold

    rng = np.random.RandomState(42)
    fold_sizes = []
    val_overlap = []

    for seed in range(n_bootstrap):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        indices = np.arange(n_samples)
        folds = list(kf.split(indices))
        train_sizes = [len(f[0]) for f in folds]
        val_sizes = [len(f[1]) for f in folds]
        fold_sizes.append((train_sizes, val_sizes))

        # Overlap between val sets
        v1, v2 = set(folds[0][1]), set(folds[1][1])
        overlap = len(v1 & v2) / len(v1) if len(v1) > 0 else 0
        val_overlap.append(overlap)

    train_sizes = [f[0][0] for f in fold_sizes]
    val_sizes = [f[1][0] for f in fold_sizes]

    # With fixed seed=42, single split
    kf_fixed = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds_fixed = list(kf_fixed.split(np.arange(n_samples)))
    val_sizes_fixed = [len(f[1]) for f in folds_fixed]

    return {
        "n_samples": n_samples,
        "n_splits": n_splits,
        "val_size_per_fold_fixed": val_sizes_fixed,
        "train_size_per_fold_fixed": [len(f[0]) for f in folds_fixed],
        "val_size_mean": float(np.mean(val_sizes)),
        "val_size_std": float(np.std(val_sizes)),
        "val_overlap_0_1_mean": float(np.mean(val_overlap)),
        "effective_n_val": n_samples // n_splits,
        "cv_variance_approx": "var(metric)/3 for mean across folds",
    }


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    print("Loading data...")
    lr_adjs, hr_adjs, lr_data, hr_data = load_graphs(data_dir)
    print(f"Loaded {len(lr_adjs)} LR graphs (160x160), {len(hr_adjs)} HR graphs (268x268)")

    results = {}

    # 1. Edge statistics
    print("\n--- Edge statistics ---")
    lr_edge = compute_edge_stats(lr_adjs, "LR")
    hr_edge = compute_edge_stats(hr_adjs, "HR")
    results["lr_edge"] = lr_edge
    results["hr_edge"] = hr_edge

    # 2. Sparsity
    print("\n--- Sparsity ---")
    lr_sparse = compute_sparsity(lr_adjs)
    hr_sparse = compute_sparsity(hr_adjs)
    results["lr_sparsity"] = lr_sparse
    results["hr_sparsity"] = hr_sparse

    # 3. Node strength
    print("\n--- Node strength distribution ---")
    lr_strength = compute_node_strength_dist(lr_adjs)
    hr_strength = compute_node_strength_dist(hr_adjs)
    results["lr_strength"] = {k: v for k, v in lr_strength.items() if k not in ["mean_per_sample", "std_per_sample"]}
    results["hr_strength"] = {k: v for k, v in hr_strength.items() if k not in ["mean_per_sample", "std_per_sample"]}

    # 4. Spectral eigenvalue decay
    print("\n--- Spectral eigenvalue decay ---")
    lr_spectral = compute_spectral_decay(lr_adjs, n_eigs=50)
    hr_spectral = compute_spectral_decay(hr_adjs, n_eigs=50)
    results["lr_spectral"] = lr_spectral
    results["hr_spectral"] = hr_spectral

    # 5. Low-rank (HR)
    print("\n--- HR low-rank metrics ---")
    hr_lowrank = compute_low_rank_metrics(hr_adjs, n_eigs=268)
    lr_lowrank = compute_low_rank_metrics(lr_adjs, n_eigs=160)
    results["hr_lowrank"] = hr_lowrank
    results["lr_lowrank"] = lr_lowrank

    # 6. LR→HR mapping smoothness
    print("\n--- LR→HR mapping smoothness ---")
    smoothness = lr_hr_mapping_smoothness(lr_data, hr_data, n_neighbors=5)
    results["lr_hr_smoothness"] = smoothness

    # 7. 3-fold CV stability
    print("\n--- 3-fold CV stability ---")
    cv_stab = cv_stability_analysis(n_samples=167, n_splits=3)
    results["cv_stability"] = cv_stab

    # Write report
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "GRAPH_STATISTICS_ANALYSIS.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_report(results, out_path)
    print(f"\nReport written to {out_path}")

    return results


def write_report(results: dict, path: str):
    """Write markdown report."""
    lines = [
        "# Graph Statistics Analysis: LR vs HR",
        "",
        "Derived from `src/dataset.py`, `utils/matrix_vectorizer.py`, `utils/graph_utils.py`",
        "and training CSVs. No speculation — all quantities computed from data.",
        "",
        "## 1. Mean, Variance, Skewness (Edge Weights)",
        "",
        "### LR (160×160, 12720 edges per sample)",
        "",
        f"- **Mean (all edges)**: {results['lr_edge']['mean_all']:.6f}",
        f"- **Variance (all edges)**: {results['lr_edge']['var_all']:.6f}",
        f"- **Skewness (all edges)**: {results['lr_edge']['skew_all']:.4f}",
        f"- **Mean (nonzero only)**: {results['lr_edge']['mean_nonzero']:.6f}",
        f"- **Variance (nonzero)**: {results['lr_edge']['var_nonzero']:.6f}",
        f"- **Skewness (nonzero)**: {results['lr_edge']['skew_nonzero']:.4f}",
        "",
        "### HR (268×268, 35778 edges per sample)",
        "",
        f"- **Mean (all edges)**: {results['hr_edge']['mean_all']:.6f}",
        f"- **Variance (all edges)**: {results['hr_edge']['var_all']:.6f}",
        f"- **Skewness (all edges)**: {results['hr_edge']['skew_all']:.4f}",
        f"- **Mean (nonzero only)**: {results['hr_edge']['mean_nonzero']:.6f}",
        f"- **Variance (nonzero)**: {results['hr_edge']['var_nonzero']:.6f}",
        f"- **Skewness (nonzero)**: {results['hr_edge']['skew_nonzero']:.4f}",
        "",
        "## 2. Sparsity",
        "",
        "### LR",
        f"- **Mean sparsity**: {results['lr_sparsity']['mean_sparsity']:.4f}",
        f"- **Std sparsity**: {results['lr_sparsity']['std_sparsity']:.4f}",
        "",
        "### HR",
        f"- **Mean sparsity**: {results['hr_sparsity']['mean_sparsity']:.4f}",
        f"- **Std sparsity**: {results['hr_sparsity']['std_sparsity']:.4f}",
        "",
        "## 3. Node Strength Distribution",
        "",
        "### LR",
        f"- **Mean strength**: {results['lr_strength']['mean_strength']:.4f}",
        f"- **Variance strength**: {results['lr_strength']['var_strength']:.4f}",
        f"- **Skewness strength**: {results['lr_strength']['skew_strength']:.4f}",
        "",
        "### HR",
        f"- **Mean strength**: {results['hr_strength']['mean_strength']:.4f}",
        f"- **Variance strength**: {results['hr_strength']['var_strength']:.4f}",
        f"- **Skewness strength**: {results['hr_strength']['skew_strength']:.4f}",
        "",
        "## 4. Spectral Eigenvalue Decay",
        "",
        "Ratios use signed eigenvalues (λ₅₀/λ₁ can be negative when spectrum crosses zero); ",
        "|λ|-based decay avoids sign flip.",
        "",
        "### LR",
        f"- **λ₁₀/λ₁ (signed)**: {results['lr_spectral']['eig_decay_ratio_10']:.4f}",
        f"- **λ₅₀/λ₁ (signed)**: {results['lr_spectral']['eig_decay_ratio_50']:.4f}",
        f"- **|λ₁₀|/|λ₁| (magnitude decay)**: {results['lr_spectral']['eig_decay_abs_10']:.4f}",
        f"- **|λ₅₀|/|λ₁| (magnitude decay)**: {results['lr_spectral']['eig_decay_abs_50']:.4f}",
        "",
        "### HR",
        f"- **λ₁₀/λ₁ (signed)**: {results['hr_spectral']['eig_decay_ratio_10']:.4f}",
        f"- **λ₅₀/λ₁ (signed)**: {results['hr_spectral']['eig_decay_ratio_50']:.4f}",
        f"- **|λ₁₀|/|λ₁| (magnitude decay)**: {results['hr_spectral']['eig_decay_abs_10']:.4f}",
        f"- **|λ₅₀|/|λ₁| (magnitude decay)**: {results['hr_spectral']['eig_decay_abs_50']:.4f}",
        "",
        "## 5. Low-Rank Assessment (HR)",
        "",
        "Effective rank = exp(entropy of normalized eigenvalue distribution).",
        "Frobenius concentration = fraction of ||A||²_F in top-k eigenvalues.",
        "",
        "### HR",
        f"- **Effective rank (mean)**: {results['hr_lowrank']['effective_rank_mean']:.2f}",
        f"- **Effective rank (std)**: {results['hr_lowrank']['effective_rank_std']:.2f}",
        f"- **Frob concentration k=10**: {results['hr_lowrank']['frob_concentration_k10_mean']:.4f}",
        f"- **Frob concentration k=50**: {results['hr_lowrank']['frob_concentration_k50_mean']:.4f}",
        f"- **Frob concentration k=100**: {results['hr_lowrank']['frob_concentration_k100_mean']:.4f}",
        "",
        "### LR (for comparison)",
        f"- **Effective rank (mean)**: {results['lr_lowrank']['effective_rank_mean']:.2f}",
        f"- **Frob concentration k=10**: {results['lr_lowrank']['frob_concentration_k10_mean']:.4f}",
        "",
        "## 6. LR→HR Mapping: Smooth vs Structurally Nonlinear",
        "",
        "k-NN smoothness: if mapping is smooth, nearby LR points map to nearby HR points.",
        "Ratio = mean HR distance within k-NN / mean HR distance outside. Ratio < 1 ⇒ smooth.",
        "",
        f"- **Mean HR dist within 5-NN**: {results['lr_hr_smoothness']['mean_hr_dist_within_knn']:.4f}",
        f"- **Mean HR dist outside 5-NN**: {results['lr_hr_smoothness']['mean_hr_dist_outside_knn']:.4f}",
        f"- **Smoothness ratio**: {results['lr_hr_smoothness']['smoothness_ratio']:.4f}",
        "",
        "## 7. 3-Fold CV Statistical Stability (n=167)",
        "",
        f"- **Val size per fold (fixed seed=42)**: {results['cv_stability']['val_size_per_fold_fixed']}",
        f"- **Train size per fold**: {results['cv_stability']['train_size_per_fold_fixed']}",
        f"- **Val overlap (fold 0 vs 1, over 500 bootstrap seeds)**: {results['cv_stability']['val_overlap_0_1_mean']:.4f}",
        f"- **Effective n_val per fold**: ~{results['cv_stability']['effective_n_val']}",
        "",
        "**Assessment**: With 167 samples, 3-fold CV yields ~56 validation samples per fold. ",
        "The mean across 3 folds has variance ≈ σ²_fold/3. The fold-level std is estimated from ",
        "only 3 values, so the reported 'mean ± std' has high variance in the std term. ",
        "SE(mean) ≈ σ_metric/√(n_val×3) ≈ σ_metric/√168; thus the 3-fold mean is nearly as stable ",
        "as a single validation on 168 samples. The main limitation is estimating σ from 3 folds.",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
