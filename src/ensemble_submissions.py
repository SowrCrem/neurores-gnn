"""
Ensemble multiple submission CSV files by averaging predictions.

Usage:
    python -m src.ensemble_submissions submission/a.csv submission/b.csv -o submission/ensemble.csv
    python -m src.ensemble_submissions submission/*.csv -o submission/ensemble.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average predictions from multiple submission CSVs into one ensemble."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Paths to submission CSV files (ID,Predicted format)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("submission/ensemble.csv"),
        help="Output path (default: submission/ensemble.csv)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional comma-separated weights for weighted average (e.g. 0.6,0.4). Default: equal weights.",
    )
    args = parser.parse_args()

    dfs = []
    for p in args.inputs:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
        df = pd.read_csv(p)
        if "ID" not in df.columns or "Predicted" not in df.columns:
            raise ValueError(f"{p}: expected columns ID, Predicted")
        dfs.append(df)

    if len(dfs) < 2:
        raise ValueError("Need at least 2 input files to ensemble")

    # Check IDs match
    ids_ref = dfs[0]["ID"].values
    for i, df in enumerate(dfs[1:], start=1):
        if not np.array_equal(df["ID"].values, ids_ref):
            raise ValueError(f"Input {i+1}: ID column does not match first file")

    preds = np.stack([d["Predicted"].values.astype(np.float64) for d in dfs], axis=0)

    if args.weights:
        w = np.array([float(x.strip()) for x in args.weights.split(",")])
        if len(w) != len(dfs):
            raise ValueError(f"--weights has {len(w)} values but {len(dfs)} inputs")
        w = w / w.sum()
        ensemble = np.average(preds, axis=0, weights=w)
        print(f"Weighted average: weights={w}")
    else:
        ensemble = np.mean(preds, axis=0)

    ensemble = np.clip(ensemble, 0.0, None)

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": ids_ref, "Predicted": ensemble}).to_csv(out, index=False)
    print(f"Saved: {out} (ensemble of {len(dfs)} submissions)")


if __name__ == "__main__":
    main()
