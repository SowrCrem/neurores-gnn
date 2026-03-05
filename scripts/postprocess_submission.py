#!/usr/bin/env python3
"""
Post-process Kaggle submissions: shrinkage toward training mean, geometric mean ensemble.

Usage:
  # Shrinkage (eps=0.05): copy v3r_submission.csv -> v3r_shrinkage_005.csv
  python scripts/postprocess_submission.py shrinkage --input submission/v3r_submission.csv \\
    --output submission/v3r_shrinkage_005.csv --eps 0.05

  # Geometric mean ensemble of seed 42, 43, 44
  python scripts/postprocess_submission.py geom-ensemble \\
    --inputs submission/v3r_submission.csv submission/v3r_seed43.csv submission/v3r_seed44.csv \\
    --output submission/v3r_geom_ensemble.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HR_FEATURES = 268 * 267 // 2  # 35778


def load_submission(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load submission CSV; return (ids, preds)."""
    df = pd.read_csv(path, header=0)
    ids = df["ID"].values.astype(np.int64)
    preds = df["Predicted"].values.astype(np.float64)
    return ids, preds


def save_submission(path: str, ids: np.ndarray, preds: np.ndarray) -> None:
    """Save submission CSV."""
    preds = np.clip(preds, 0.0, 1.0)
    df = pd.DataFrame({"ID": ids, "Predicted": preds})
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(preds):,} rows)")


def run_shrinkage(args: argparse.Namespace) -> None:
    """Apply shrinkage: preds = (1-eps)*preds + eps*y_mean."""
    ids, preds = load_submission(args.input)
    n_total = len(preds)
    n_subjects = n_total // HR_FEATURES
    assert n_subjects * HR_FEATURES == n_total, f"Expected {HR_FEATURES}*N, got {n_total}"

    # Load HR training mean
    data_dir = getattr(args, "data_dir", "data")
    hr_path = os.path.join(data_dir, "hr_train.csv")
    if not os.path.exists(hr_path):
        raise FileNotFoundError(f"hr_train.csv not found at {hr_path}. Set --data-dir if needed.")
    hr_df = pd.read_csv(hr_path, header=0)
    y_mean = hr_df.values.astype(np.float64).mean(axis=0)
    assert len(y_mean) == HR_FEATURES

    # Reshape preds to (n_subjects, HR_FEATURES)
    preds = preds.reshape(n_subjects, HR_FEATURES)
    eps = args.eps
    preds = (1 - eps) * preds + eps * y_mean[np.newaxis, :]
    preds = preds.reshape(-1)

    save_submission(args.output, ids, preds)
    print(f"Shrinkage eps={eps} applied. Source: {args.input}")


def run_geom_ensemble(args: argparse.Namespace) -> None:
    """Geometric mean of multiple submission predictions."""
    all_preds = []
    for p in args.inputs:
        _, preds = load_submission(p)
        all_preds.append(preds)
    preds_stack = np.stack(all_preds, axis=0)
    # Geometric mean: exp(mean(log(p + eps)))
    eps = 1e-10
    geom = np.exp(np.mean(np.log(preds_stack + eps), axis=0))
    ids, _ = load_submission(args.inputs[0])
    save_submission(args.output, ids, geom)
    print(f"Geometric mean of {len(args.inputs)} submissions -> {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process Kaggle submissions")
    sub = parser.add_subparsers(dest="mode", required=True)

    shrink = sub.add_parser("shrinkage", help="Shrink predictions toward training mean")
    shrink.add_argument("--input", required=True, help="Input submission CSV")
    shrink.add_argument("--output", required=True, help="Output submission CSV")
    shrink.add_argument("--eps", type=float, default=0.05, help="Shrinkage toward mean (default 0.05)")
    shrink.add_argument("--data-dir", default="data", help="Data directory for hr_train.csv")

    geom = sub.add_parser("geom-ensemble", help="Geometric mean of multiple submissions")
    geom.add_argument("--inputs", nargs="+", required=True, help="Input submission CSVs")
    geom.add_argument("--output", required=True, help="Output submission CSV")

    args = parser.parse_args()
    if args.mode == "shrinkage":
        run_shrinkage(args)
    else:
        run_geom_ensemble(args)


if __name__ == "__main__":
    main()
