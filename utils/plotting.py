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

    If scales differ too much, the plot can be split into two separate figures:
        - Matrix-level metrics: MAE, PCC, JSD
        - Graph-level metrics: centrality/topology MAE metrics

    Parameters
    ----------
    fold_results : list[dict]
        List of fold metric dictionaries returned by `evaluate_fold`.
        The first 3 folds are plotted individually; if more than 3 folds are provided,
        only the first 3 are shown in the per-fold panels, but the average/std is
        computed across all folds.
    split_if_obscured : bool, optional
        If True, automatically split metrics into two figures when the ratio
        between the largest and smallest finite metric is too large.
        This improves readability when some bars are tiny compared to others.
    verbose : bool, optional
        If True, prints basic diagnostics:
            - number of folds
            - split decision and ratio

    Returns
    -------
    None
        Displays plots using matplotlib.

    Notes
    -----
    - The "obscured" decision uses a simple heuristic:
        ratio = max(metric_values) / min(metric_values)
      and splits if ratio > 20.
    """
    means, stds = summarize_folds(fold_results)

    # Collect all metric values across all folds to check scale disparity
    all_vals = np.array([fr[k] for fr in fold_results for k in METRIC_ORDER], dtype=float)
    finite = all_vals[np.isfinite(all_vals)]
    ratio = (finite.max() / max(finite.min(), 1e-12)) if finite.size else 1.0

    if verbose:
        print(f"Plotting {len(fold_results)} fold(s).")
        print(f"Scale ratio (max/min over finite values): {ratio:.3f}")

    if split_if_obscured and ratio > 20:
        matrix_metrics = ["MAE", "PCC", "JSD"]
        graph_metrics = [k for k in METRIC_ORDER if k not in matrix_metrics]

        if verbose:
            print("Scale disparity detected — splitting into two figures:")
            print(f"  Matrix-level metrics: {matrix_metrics}")
            print(f"  Graph-level metrics : {graph_metrics}")

        _plot_grid(fold_results, means, stds, matrix_metrics, suptitle="Matrix-level Metrics")
        _plot_grid(fold_results, means, stds, graph_metrics, suptitle="Graph-level Metrics")
        return

    _plot_grid(fold_results, means, stds, METRIC_ORDER, suptitle=None)


def _plot_grid(
    fold_results: list[dict],
    means: dict,
    stds: dict,
    metric_list: list[str],
    suptitle: str | None
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
        vals = [fold_results[i][k] for k in metric_list]
        axs[i].bar(x, vals, color=colours[:len(metric_list)])
        axs[i].set_title(f"Fold {i+1}")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(metric_list, rotation=45, ha="right")

    # Plot average across folds with error bars (std)
    mean_vals = [means[k] for k in metric_list]
    err_vals = [stds[k] for k in metric_list]
    axs[3].bar(x, mean_vals, yerr=err_vals, capsize=5, color=colours[:len(metric_list)])
    axs[3].set_title("Avg. Across Folds")
    axs[3].set_xticks(x)
    axs[3].set_xticklabels(metric_list, rotation=45, ha="right")

    if suptitle:
        fig.suptitle(suptitle)

    plt.tight_layout()
    plt.show()