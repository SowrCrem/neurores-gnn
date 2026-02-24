# main.py
"""
Minimal entry point.

Uses:
- config.py for all hyperparameters
- train_cv.py for 3-fold training + test inference
- data/*.csv as provided datasets

Run from repo root:
    python gcn-encoder-ca-decoder/main.py
"""

from pathlib import Path
import numpy as np
import torch

from config import ModelConfig, TrainConfig
from train_cv import train_cv_and_predict_test


def read_csv(path: str | Path) -> np.ndarray:
    """
    Load feature CSV where first row is column indices (0,1,2,...).
    """
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def write_submission(pred: np.ndarray, out_path: str | Path):
    """
    Writes raw prediction matrix (112, 35778) with no header.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, pred, delimiter=",")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    repo_root = get_repo_root()

    lr_train_path = repo_root / "data" / "lr_train.csv"
    hr_train_path = repo_root / "data" / "hr_train.csv"
    lr_test_path  = repo_root / "data" / "lr_test.csv"
    submission_path = repo_root / "submission.csv"

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    X_lr_train = read_csv(lr_train_path)
    Y_hr_train = read_csv(hr_train_path)
    X_lr_test  = read_csv(lr_test_path)

    # ------------------------------------------------------------------
    # Configs (single source of truth)
    # ------------------------------------------------------------------
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Resolve device once
    device = train_cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model_kwargs = dict(
        n_lr=model_cfg.n_lr,
        n_hr=model_cfg.n_hr,
        d_model=model_cfg.d_model,
        gcn_layers=model_cfg.gcn_layers,
        attn_heads=model_cfg.attn_heads,
        dropout=model_cfg.dropout,
    )

    train_kwargs = dict(
        random_state=train_cfg.random_state,
        folds=train_cfg.folds,
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        device=device,
    )

    # ------------------------------------------------------------------
    # Train + predict (3-fold CV)
    # ------------------------------------------------------------------
    pred_test = train_cv_and_predict_test(
        X_lr_train=X_lr_train,
        Y_hr_train=Y_hr_train,
        X_lr_test=X_lr_test,
        model_kwargs=model_kwargs,
        train_kwargs=train_kwargs,
    )

    # ------------------------------------------------------------------
    # Save submission
    # ------------------------------------------------------------------
    write_submission(pred_test, submission_path)


if __name__ == "__main__":
    main()