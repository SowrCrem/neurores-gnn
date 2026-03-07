"""
Plotting utilities for Brain Graph Super-Resolution Challenge evaluation.

This module visualizes fold-level evaluation metrics as bar charts in a layout
matching the challenge reference:
    - Fold 1
    - Fold 2
    - Fold 3
    - Average across folds (with error bars)

If metric scales differ dramatically (e.g., some metrics near 0 while others near 1),
the bars for smaller metrics can become visually “obscured”. To address this,
`plot_folds` can automatically split the plot into:
    1) Matrix-level metrics (MAE, PCC, JSD)
    2) Graph-level metrics (centrality/topology MAEs)

Inputs:
    fold_results: list of dicts returned by `evaluate_fold` where each dict maps
    metric name -> value.

Dependencies:
    - METRIC_ORDER is imported from utils.metrics to ensure a consistent ordering
      between evaluation and plots.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import METRIC_ORDER

# Typical mean strength for 268-node brain graphs (for normalising legacy/cached MAE(Strength))
DEFAULT_MEAN_STRENGTH = 25.0


def normalize_strength_in_fold_metrics(
    fold_results: list[dict],
    mean_strength: float = DEFAULT_MEAN_STRENGTH,
    threshold: float = 2.0,
) -> list[dict]:
    """
    Normalise MAE (Strength) in pre-computed fold metrics for plotting.

    Use when fold_metrics come from cached cv_summary.json (computed before
    utils.metrics normalised MAE(Strength)). Values > threshold are divided
    by mean_strength so all 8 metrics fit on one graph.

    Parameters
    ----------
    fold_results : list[dict]
        List of fold metric dicts (e.g. from cv_summary['folds']).
    mean_strength : float
        Typical mean node strength (default 25 for 268-node brain graphs).
    threshold : float
        Only normalise if MAE(Strength) > threshold (already normalised if below).

    Returns
    -------
    list[dict]
        Copy of fold_results with MAE (Strength) normalised.
    """
    out = []
    for fr in fold_results:
        copy = dict(fr)
        s = copy.get("MAE (Strength)")
        if s is not None and isinstance(s, (int, float)) and s > threshold:
            copy["MAE (Strength)"] = s / mean_strength
        out.append(copy)
    return out


def summarize_folds(fold_results: list[dict]) -> tuple[dict, dict]:
    """
    Compute mean and standard deviation of each metric across folds.

    This is used to generate the "Avg. Across Folds" subplot where bars represent
    the average value and error bars represent the standard deviation.

    Parameters
    ----------
    fold_results : list[dict]
        List of fold metric dictionaries.
        Each dictionary should contain the keys in METRIC_ORDER (e.g. "MAE", "PCC", ...).

        Example:
            fold_results = [
                {"MAE": 0.5, "PCC": 0.7, ...},
                {"MAE": 0.6, "PCC": 0.75, ...},
                ...
            ]

    Returns
    -------
    means : dict
        Dictionary mapping metric name -> mean value across folds.
    stds : dict
        Dictionary mapping metric name -> standard deviation across folds.

    Notes
    -----
    - Uses np.nanmean / np.nanstd so that NaNs (e.g., PCC in degenerate cases)
      do not break aggregation.
    - Uses ddof=1 for sample standard deviation if there is more than one fold.
    """
    means, stds = {}, {}
    for k in METRIC_ORDER:
        vals = np.array([fr[k] for fr in fold_results], dtype=float)
        means[k] = float(np.nanmean(vals))
        stds[k] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return means, stds


def plot_folds(fold_results: list[dict], split_if_obscured: bool = True, verbose: bool = True) -> None:
    """
    Plot evaluation metrics for up to 3 folds + average across folds.

    Creates a 2x2 grid:
        [ Fold 1 ] [ Fold 2 ]
        [ Fold 3 ] [ Avg Across Folds (mean ± std) ]

    All 8 metrics are shown in one figure. MAE (Strength) is normalised in
    utils.metrics. When metric range > 20x, a log scale is used so small
    metrics (e.g. MAE(PC)) remain visible.

    Parameters
    ----------
    fold_results : list[dict]
        List of fold metric dictionaries returned by `evaluate_fold`.
    verbose : bool, optional
        If True, prints basic diagnostics.
    """
    means, stds = summarize_folds(fold_results)

    # Collect all metric values across all folds to check scale disparity
    all_vals = np.array([fr[k] for fr in fold_results for k in METRIC_ORDER], dtype=float)
    finite = all_vals[np.isfinite(all_vals)]
    ratio = (finite.max() / max(finite.min(), 1e-12)) if finite.size else 1.0

    if verbose:
        print(f"Plotting {len(fold_results)} fold(s).")
        print(f"Scale ratio (max/min over finite values): {ratio:.3f}")

    # With MAE (Strength) normalised, all 8 metrics fit on one graph; use log scale
    # when ratio > 20 so small metrics (e.g. MAE(PC)) remain visible.
    # Skip log if any value is non-positive (e.g. negative PCC).
    use_log = ratio > 20 and finite.min() > 1e-10
    if verbose and use_log:
        print("Using log scale for y-axis (metric range > 20x).")
    _plot_grid(fold_results, means, stds, METRIC_ORDER, suptitle=None, use_log_scale=use_log)


def _plot_grid(
    fold_results: list[dict],
    means: dict,
    stds: dict,
    metric_list: list[str],
    suptitle: str | None,
    use_log_scale: bool = False,
) -> None:
    """
    Internal helper to create the 2x2 grid plot for a given subset of metrics.

    Parameters
    ----------
    fold_results : list[dict]
        List of fold metric dictionaries.
    means : dict
        Metric means across folds (from summarize_folds).
    stds : dict
        Metric standard deviations across folds (from summarize_folds).
    metric_list : list[str]
        Subset of metrics to plot (order is used for x-axis labels).
    suptitle : str or None
        Optional figure-level title. If None, no title is set.

    Returns
    -------
    None
        Displays the figure.

    Plot layout
    -----------
    - First three panels: bar charts for Fold 1..3 (or fewer if fewer folds)
    - Bottom-right panel: mean bars with std error bars
    """
    x = np.arange(len(metric_list))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    colours = [
        "#4C72B0",  # blue
        "#DD8452",  # orange
        "#55A868",  # green
        "#C44E52",  # red
        "#8172B3",  # purple
        "#937860",  # brown
        "#DA8BC3",  # pink
    ]

    # Plot up to the first 3 folds in the first 3 panels
    for i in range(min(3, len(fold_results))):
        vals = [max(fold_results[i][k], 1e-10) if use_log_scale else fold_results[i][k] for k in metric_list]
        axs[i].bar(x, vals, color=colours[:len(metric_list)])
        axs[i].set_title(f"Fold {i+1}")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(metric_list, rotation=45, ha="right")
        if use_log_scale:
            axs[i].set_yscale("log")
            axs[i].set_ylim(bottom=1e-6)

    # Plot average across folds with error bars (std)
    mean_vals = [means[k] for k in metric_list]
    err_vals = [stds[k] for k in metric_list]
    # For log scale, ensure mean - err > 0 and use a floor for bar heights
    if use_log_scale:
        mean_vals = [max(m, 1e-10) for m in mean_vals]
        err_vals = [min(e, m * 0.99) for m, e in zip(mean_vals, err_vals)]  # avoid negative
    axs[3].bar(x, mean_vals, yerr=err_vals, capsize=5, color=colours[:len(metric_list)])
    axs[3].set_title("Avg. Across Folds")
    axs[3].set_xticks(x)
    axs[3].set_xticklabels(metric_list, rotation=45, ha="right")
    if use_log_scale:
        axs[3].set_yscale("log")
        axs[3].set_ylim(bottom=1e-6)

    if suptitle:
        fig.suptitle(suptitle)

    plt.tight_layout()
    plt.show()


# Metric groups for grouped plotting (Spec §II.A)
EDGE_PREDICTION_METRICS = ["MAE", "PCC", "JSD"]
CENTRALITY_METRICS = ["MAE (PC)", "MAE (EC)", "MAE (BC)"]
STRUCTURAL_TOPOLOGY_METRICS = ["MAE (Strength)", "MAE (Clustering)"]
METRIC_GROUPS = [
    ("Edge prediction", EDGE_PREDICTION_METRICS),
    ("Centrality preservation", CENTRALITY_METRICS),
    ("Structural topology", STRUCTURAL_TOPOLOGY_METRICS),
]


def plot_folds_grouped(
    fold_results: list[dict],
    normalize_strength: bool = True,
    mean_strength: float = DEFAULT_MEAN_STRENGTH,
    verbose: bool = True,
) -> None:
    """
    Plot evaluation metrics grouped by category: Fold 1, 2, 3 + Avg per group.

    Groups:
        - Edge prediction: MAE, PCC, JSD
        - Centrality preservation: MAE (PC), MAE (EC), MAE (BC)
        - Structural topology: MAE (Strength), MAE (Clustering)

    Creates one figure per group, each with 2x2 layout (Fold 1, 2, 3, Avg).
    """
    if normalize_strength:
        fold_results = normalize_strength_in_fold_metrics(
            fold_results, mean_strength=mean_strength
        )
    means, stds = summarize_folds(fold_results)

    for group_name, metric_list in METRIC_GROUPS:
        if verbose:
            print(f"Plotting {group_name}: {metric_list}")
        use_log = False
        all_vals = np.array([fr[k] for fr in fold_results for k in metric_list], dtype=float)
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size:
            ratio = finite.max() / max(finite.min(), 1e-12)
            use_log = ratio > 20 and finite.min() > 1e-10
        _plot_grid(
            fold_results, means, stds, metric_list,
            suptitle=group_name, use_log_scale=use_log
        )