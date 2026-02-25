"""
DenseGCN training/evaluation script for brain graph super-resolution.

This script centralizes the experiment flow previously duplicated in notebooks:
  1) 3-fold cross-validation with full 8-metric evaluation
  2) Resource tracking (total wall-clock time + peak RAM RSS)
  3) Full-data retraining on train_LR/train_HR
  4) test_LR inference and Kaggle-format submission export

Usage examples:
    # 3-fold CV with metrics + resource logs
    python -m src.train_dense_gcn cv

    # Full retraining + submission generation
    python -m src.train_dense_gcn full
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from models.dense_gcn import DenseGCNGenerator
from utils.matrix_vectorizer import MatrixVectorizer
from utils.metrics import METRIC_ORDER, evaluate_fold
from utils.plotting import summarize_folds

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------------------
# Defaults (match dense_gcn_v2 notebook baseline)
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "data"
DEFAULT_OUT_DIR = "artifacts/dense_gcn_v2"
DEFAULT_SUBMISSION_PATH = "submission/dense_gcn_v2_full_retrain_submission.csv"

N_LR = 160
N_HR = 268
HR_FEATURES = N_HR * (N_HR - 1) // 2  # 35778


@dataclass
class TrainConfig:
    n_lr: int = N_LR
    n_hr: int = N_HR
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.5
    epochs: int = 400
    patience: int = 30
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_folds: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DenseGCN runner (3F-CV logs + full retraining submission)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
        p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--epochs", type=int, default=400)
        p.add_argument("--patience", type=int, default=30)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--lr", type=float, default=5e-4)
        p.add_argument("--weight-decay", type=float, default=1e-4)
        p.add_argument("--hidden-dim", type=int, default=128)
        p.add_argument("--num-layers", type=int, default=3)
        p.add_argument("--dropout", type=float, default=0.5)
        p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    cv = sub.add_parser("cv", help="Run 3-fold CV and log metrics/resources.")
    add_shared(cv)
    cv.add_argument("--num-folds", type=int, default=3)

    full = sub.add_parser("full", help="Retrain on full train set and predict test set.")
    add_shared(full)
    full.add_argument("--submission-path", type=str, default=DEFAULT_SUBMISSION_PATH)
    full.add_argument("--checkpoint-name", type=str, default="full_model.pt")

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    return arr if arr.ndim > 1 else arr[None, :]


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = load_csv(data_dir / "lr_train.csv")
    y_train = load_csv(data_dir / "hr_train.csv")
    x_test = load_csv(data_dir / "lr_test.csv")
    return x_train, y_train, x_test


def current_ram_gb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / (1024 ** 3)


def vec_to_adj(vec: torch.Tensor, n: int, vectorizer: MatrixVectorizer) -> torch.Tensor:
    """
    Convert upper-triangular vectors (B, E) to adjacency tensors (B, n, n).
    """
    vec_np = vec.detach().cpu().numpy()
    mats = [vectorizer.anti_vectorize(v, n, include_diagonal=False) for v in vec_np]
    return torch.from_numpy(np.stack(mats)).to(device=vec.device, dtype=vec.dtype)


def build_model(cfg: TrainConfig, device: torch.device) -> DenseGCNGenerator:
    return DenseGCNGenerator(
        n_lr=cfg.n_lr,
        n_hr=cfg.n_hr,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)


def train_with_validation(
    cfg: TrainConfig,
    device: torch.device,
    vectorizer: MatrixVectorizer,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    fold_id: int,
) -> tuple[DenseGCNGenerator, float, int]:
    xtr = torch.from_numpy(x_tr).float().to(device)
    ytr = torch.from_numpy(y_tr).float().to(device)
    xva = torch.from_numpy(x_va).float().to(device)
    yva = torch.from_numpy(y_va).float().to(device)

    tr_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(xva, yva), batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x_vec, y_vec in tr_loader:
            a = vec_to_adj(x_vec, cfg.n_lr, vectorizer)
            pred = model(a, a)
            loss = loss_fn(pred, y_vec)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_vec, y_vec in va_loader:
                a = vec_to_adj(x_vec, cfg.n_lr, vectorizer)
                val_losses.append(loss_fn(model(a, a), y_vec).item())

        val_loss = float(np.mean(val_losses))
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            marker = " *" if improved else ""
            print(f"  Fold {fold_id} | Epoch {epoch:3d}/{cfg.epochs} | Val: {val_loss:.6f}{marker}")

        if stale_epochs >= cfg.patience:
            print(
                f"  Fold {fold_id} | Early stopping at epoch {epoch} "
                f"(no improvement for {cfg.patience} epochs)"
            )
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold_id}: no best checkpoint was captured.")

    model.load_state_dict(best_state)
    print(f"  Fold {fold_id} best val loss: {best_val:.6f} at epoch {best_epoch}")
    return model, best_val, best_epoch


def train_full(
    cfg: TrainConfig,
    device: torch.device,
    vectorizer: MatrixVectorizer,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
) -> DenseGCNGenerator:
    xtr = torch.from_numpy(x_tr).float().to(device)
    ytr = torch.from_numpy(y_tr).float().to(device)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.batch_size, shuffle=True)

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for x_vec, y_vec in loader:
            a = vec_to_adj(x_vec, cfg.n_lr, vectorizer)
            pred = model(a, a)
            loss = loss_fn(pred, y_vec)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs:
            print(f"  Full-train | Epoch {epoch:3d}/{cfg.epochs} | Loss: {np.mean(train_losses):.6f}")

    return model


def predict_vectors(
    model: DenseGCNGenerator,
    x_np: np.ndarray,
    device: torch.device,
    vectorizer: MatrixVectorizer,
    batch_size: int,
    n_lr: int,
) -> np.ndarray:
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, len(x_np), batch_size):
            x_batch = torch.from_numpy(x_np[start:start + batch_size]).float().to(device)
            a = vec_to_adj(x_batch, n_lr, vectorizer)
            pred = model(a, a)
            outputs.append(pred.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def vectors_to_matrices(vecs: np.ndarray, n: int, vectorizer: MatrixVectorizer) -> np.ndarray:
    mats = [vectorizer.anti_vectorize(v, n, include_diagonal=False) for v in vecs]
    return np.stack(mats)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_cv(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_folds=args.num_folds,
    )

    seed_everything(cfg.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, _ = load_data(Path(args.data_dir))
    assert x_train.shape[0] == y_train.shape[0], "Mismatched train sample count."

    print(f"Device: {device}")
    print(f"Train LR: {x_train.shape} | Train HR: {y_train.shape}")
    print(f"Config: {asdict(cfg)}")

    vectorizer = MatrixVectorizer()
    kf = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)

    fold_records = []
    fold_metric_dicts = []
    fold_seconds = []
    peak_ram = current_ram_gb()
    cv_start = time.perf_counter()

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(x_train), start=1):
        print("\n" + "=" * 60)
        print(f"Fold {fold_id}: train={len(tr_idx)}, val={len(va_idx)}")
        print("=" * 60)

        fold_t0 = time.perf_counter()
        model, best_val, best_epoch = train_with_validation(
            cfg,
            device,
            vectorizer,
            x_train[tr_idx],
            y_train[tr_idx],
            x_train[va_idx],
            y_train[va_idx],
            fold_id,
        )
        fold_elapsed = time.perf_counter() - fold_t0
        fold_seconds.append(fold_elapsed)

        # Save fold checkpoint
        fold_ckpt = ckpt_dir / f"fold_{fold_id}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "fold": fold_id,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "config": asdict(cfg),
            },
            fold_ckpt,
        )

        print(f"  Computing metrics for fold {fold_id} ...")
        pred_vecs = predict_vectors(
            model, x_train[va_idx], device, vectorizer, cfg.batch_size, cfg.n_lr
        )
        gt_vecs = y_train[va_idx]

        pred_mats = vectors_to_matrices(pred_vecs, cfg.n_hr, vectorizer)
        gt_mats = vectors_to_matrices(gt_vecs, cfg.n_hr, vectorizer)
        metrics = evaluate_fold(pred_mats, gt_mats, verbose=True)

        fold_metric_dicts.append(metrics)
        fold_records.append(
            {
                "fold": fold_id,
                "train_size": int(len(tr_idx)),
                "val_size": int(len(va_idx)),
                "best_val_loss": float(best_val),
                "best_epoch": int(best_epoch),
                "fold_seconds": float(fold_elapsed),
                "checkpoint": str(fold_ckpt),
                "metrics": metrics,
            }
        )

        now_ram = current_ram_gb()
        if np.isfinite(now_ram):
            peak_ram = np.nanmax([peak_ram, now_ram])

    total_seconds = float(time.perf_counter() - cv_start)
    mean_metrics, std_metrics = summarize_folds(fold_metric_dicts)

    cv_summary = {
        "task": "dense_gcn_v2_3fold_cv",
        "device": str(device),
        "config": asdict(cfg),
        "folds": fold_records,
        "metric_order": METRIC_ORDER,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }
    resource_summary = {
        "task": "dense_gcn_v2_3fold_cv",
        "device": str(device),
        "total_cv_seconds": total_seconds,
        "total_cv_minutes": total_seconds / 60.0,
        "fold_seconds": fold_seconds,
        "peak_ram_gb_rss": None if not np.isfinite(peak_ram) else float(peak_ram),
        "psutil_available": psutil is not None,
    }

    write_json(out_dir / "cv_summary.json", cv_summary)
    write_json(out_dir / "resource_summary.json", resource_summary)

    print("\n" + "=" * 60)
    print("CV completed.")
    print(f"Saved: {out_dir / 'cv_summary.json'}")
    print(f"Saved: {out_dir / 'resource_summary.json'}")
    print(f"Total CV time: {total_seconds:.1f}s ({total_seconds/60.0:.2f} min)")
    if np.isfinite(peak_ram):
        print(f"Peak RAM RSS: {peak_ram:.3f} GB")
    else:
        print("Peak RAM RSS: N/A (install psutil to enable RAM tracking)")


def run_full(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    seed_everything(cfg.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    submission_path = Path(args.submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test = load_data(data_dir)
    print(f"Device: {device}")
    print(f"Train LR: {x_train.shape} | Train HR: {y_train.shape} | Test LR: {x_test.shape}")
    print(f"Config: {asdict(cfg)}")

    vectorizer = MatrixVectorizer()
    peak_ram = current_ram_gb()
    train_start = time.perf_counter()
    full_model = train_full(cfg, device, vectorizer, x_train, y_train)
    train_seconds = float(time.perf_counter() - train_start)

    checkpoint_path = out_dir / "checkpoints" / args.checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": full_model.state_dict(), "config": asdict(cfg)}, checkpoint_path)

    preds = predict_vectors(full_model, x_test, device, vectorizer, cfg.batch_size, cfg.n_lr)
    preds = np.clip(preds, a_min=0.0, a_max=None)
    assert preds.shape[1] == HR_FEATURES, f"Expected {HR_FEATURES} HR features, got {preds.shape[1]}"

    n_subjects, n_features = preds.shape
    ids = np.arange(1, n_subjects * n_features + 1)
    submission = np.column_stack([ids, preds.reshape(-1)])

    # Save CSV without pandas to keep dependencies minimal.
    np.savetxt(
        submission_path,
        submission,
        delimiter=",",
        header="ID,Predicted",
        comments="",
        fmt=["%d", "%.10f"],
    )

    now_ram = current_ram_gb()
    if np.isfinite(now_ram):
        peak_ram = np.nanmax([peak_ram, now_ram])

    full_summary = {
        "task": "dense_gcn_v2_full_retrain",
        "device": str(device),
        "config": asdict(cfg),
        "train_seconds": train_seconds,
        "train_minutes": train_seconds / 60.0,
        "peak_ram_gb_rss": None if not np.isfinite(peak_ram) else float(peak_ram),
        "psutil_available": psutil is not None,
        "checkpoint_path": str(checkpoint_path),
        "submission_path": str(submission_path),
        "submission_rows": int(n_subjects * n_features),
    }
    write_json(out_dir / "full_retrain_summary.json", full_summary)

    print("\n" + "=" * 60)
    print("Full retraining completed.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Submission: {submission_path} ({n_subjects*n_features:,} rows)")
    print(f"Train time: {train_seconds:.1f}s ({train_seconds/60.0:.2f} min)")
    if np.isfinite(peak_ram):
        print(f"Peak RAM RSS: {peak_ram:.3f} GB")
    else:
        print("Peak RAM RSS: N/A (install psutil to enable RAM tracking)")
    print(f"Saved: {out_dir / 'full_retrain_summary.json'}")


def main() -> None:
    args = parse_args()
    if args.mode == "cv":
        run_cv(args)
    elif args.mode == "full":
        run_full(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
