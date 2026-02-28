"""
DenseGCN training/evaluation script for brain graph super-resolution.

  1) 3-fold cross-validation with full 8-metric evaluation
  2) Hyperparameter tuning (Optuna Bayesian optimization on 3-fold CV)
  3) Full-data retraining on train_LR/train_HR (default 50 epochs)
  4) test_LR inference and Kaggle-format submission export

Usage:
    .venv/bin/python -m src.train_dense_gcn cv --preset v4 --fresh
    .venv/bin/python -m src.train_dense_gcn tune --preset v4 --out-dir artifacts/dense_gat_v4_tune
    .venv/bin/python -m src.train_dense_gcn full --preset v4 --max-epochs 50 --submission-path submission/dense_gat_v4_submission.csv

See docs/HYPERPARAMETER_TUNING.md for full tuning commands and workflow.
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
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models.dense_bisr import DenseBiSRGenerator
from models.dense_gin import DenseGINGenerator
from models.dense_gcn import DenseGCNGenerator
from models.dense_gcn_ca import DenseGCNCrossAttnGenerator
from models.dense_gcn_gps import DenseGCNGPSGenerator
from models.dense_graphsage import DenseGraphSAGEGenerator
from models.dense_gat import DenseGATGenerator
from utils.matrix_vectorizer import MatrixVectorizer
from utils.metrics import METRIC_ORDER, evaluate_fold
from utils.plotting import summarize_folds

try:
    import psutil
except ImportError:
    psutil = None

try:
    import optuna
except ImportError:
    optuna = None


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
    model_name: str = "dense_gcn"
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
    loss_name: str = "mse"
    huber_beta: float = 1.0
    l1_weight: float = 0.7
    seed: int = 42
    num_folds: int = 3
    # GAT-specific (ignored for dense_gcn)
    num_heads: int = 4
    ffn_mult: int = 4
    num_decoder_heads: int = 4
    hr_refine_layers: int = 1
    edge_scale: float = 0.2
    bipartite_layers: int = 1  # Bi-SR only
    max_epochs_full: int = 50  # cap for full retrain (CV guides best; 50 ≈ convergence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DenseGCN runner (3F-CV logs + full retraining submission)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
        p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
        p.add_argument("--preset", type=str, default="v2", choices=["v2", "v3", "v4", "v5", "bisr", "bisr_v2", "gcn_ca", "gin", "gps", "graphsage"])
        p.add_argument("--model", type=str, default="dense_gcn", choices=["dense_gcn", "dense_gat", "dense_bisr", "dense_gcn_ca", "dense_gin", "dense_gcn_gps", "dense_graphsage"])
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--epochs", type=int, default=400)
        p.add_argument("--patience", type=int, default=30)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--lr", type=float, default=5e-4)
        p.add_argument("--weight-decay", type=float, default=1e-4)
        p.add_argument("--hidden-dim", type=int, default=128)
        p.add_argument("--num-layers", type=int, default=3)
        p.add_argument("--dropout", type=float, default=0.5)
        p.add_argument(
            "--loss",
            type=str,
            default="mse",
            choices=["mse", "smoothl1", "l1", "hybrid"],
            help="Training loss: hybrid = l1_weight*L1 + (1-l1_weight)*MSE",
        )
        p.add_argument("--huber-beta", type=float, default=1.0)
        p.add_argument("--l1-weight", type=float, default=0.7)
        p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
        # GAT-specific
        p.add_argument("--num-heads", type=int, default=4)
        p.add_argument("--ffn-mult", type=int, default=4)
        p.add_argument("--num-decoder-heads", type=int, default=4)
        p.add_argument("--hr-refine-layers", type=int, default=1)
        p.add_argument("--edge-scale", type=float, default=0.2, help="GAT: adjacency bias strength (0.1=weak, 0.5=strong)")
        p.add_argument("--bipartite-layers", type=int, default=1, help="Bi-SR: number of bipartite GNN layers (1-2)")

    cv = sub.add_parser("cv", help="Run 3-fold CV and log metrics/resources.")
    add_shared(cv)
    cv.add_argument("--num-folds", type=int, default=3)
    cv.add_argument(
        "--fresh",
        action="store_true",
        help="Clear CV progress, fold checkpoints, and metrics cache; run all folds from scratch.",
    )
    cv.add_argument(
        "--skip-graph-metrics",
        action="store_true",
        help="Skip expensive graph metrics (PC, EC, BC, etc.); only compute MAE, PCC, JSD. Use for quick sanity checks.",
    )
    cv.add_argument(
        "--graph-metrics-subsample",
        type=int,
        default=None,
        metavar="N",
        help="Compute graph metrics on N random val samples (e.g. 5) instead of all. Speeds up evaluation (~30 min -> ~3 min per fold).",
    )

    full = sub.add_parser("full", help="Retrain on full train set and predict test set.")
    add_shared(full)
    full.add_argument("--submission-path", type=str, default=DEFAULT_SUBMISSION_PATH)
    full.add_argument("--checkpoint-name", type=str, default="full_model.pt")
    full.add_argument("--max-epochs", type=int, default=400, help="Max epochs for full retrain (default 400)")
    full.add_argument("--val-ratio", type=float, default=0.15, help="Hold out this fraction for validation and early stopping (0 = no val, train on all)")
    full.add_argument(
        "--ensemble-seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for full-retrain ensemble (e.g. 42,43,44). Each model trained on all 167; predictions averaged.",
    )

    tune = sub.add_parser("tune", help="Hyperparameter tuning: Optuna Bayesian optimization on 3-fold CV.")
    add_shared(tune)
    tune.add_argument("--num-folds", type=int, default=3)
    tune.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials (default 20)")
    tune.add_argument("--out-config", type=str, default=None, help="Path to save best config JSON for full retrain")
    tune.add_argument("--fresh", action="store_true", help="Clear Optuna study and start from scratch")
    tune.add_argument("--full-retrain", action="store_true", help="After tuning, run full retrain with best config and save submission")
    tune.add_argument("--submission-path", type=str, default=None, help="Submission CSV path (used with --full-retrain; default: out-dir/submission.csv)")

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


def build_model(cfg: TrainConfig, device: torch.device) -> nn.Module:
    if cfg.model_name == "dense_gat":
        return DenseGATGenerator(
            n_lr=cfg.n_lr,
            n_hr=cfg.n_hr,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            ffn_mult=cfg.ffn_mult,
            num_decoder_heads=cfg.num_decoder_heads,
            hr_refine_layers=cfg.hr_refine_layers,
            edge_scale=cfg.edge_scale,
        ).to(device)
    if cfg.model_name == "dense_bisr":
        return DenseBiSRGenerator(
            n_lr=cfg.n_lr,
            n_hr=cfg.n_hr,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            bipartite_layers=cfg.bipartite_layers,
            dropout=cfg.dropout,
        ).to(device)
    if cfg.model_name == "dense_gcn_ca":
        return DenseGCNCrossAttnGenerator(
            n_lr=cfg.n_lr,
            n_hr=cfg.n_hr,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        ).to(device)
    if cfg.model_name == "dense_gin":
        return DenseGINGenerator(
            n_lr=cfg.n_lr,
            n_hr=cfg.n_hr,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(device)
    if cfg.model_name == "dense_gcn_gps":
        return DenseGCNGPSGenerator(
            n_lr=cfg.n_lr,
            n_hr=cfg.n_hr,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        ).to(device)
    if cfg.model_name == "dense_graphsage":
        return DenseGraphSAGEGenerator(
            n_lr=cfg.n_lr,
            n_hr=cfg.n_hr,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(device)
    return DenseGCNGenerator(
        n_lr=cfg.n_lr,
        n_hr=cfg.n_hr,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """
    Fill defaults according to an experiment preset.

    v2: existing dense_gcn_v2 baseline.
    v3: balanced capacity + MAE-aligned objective.
    v4: DenseGATNet — graph-attention encoder + multi-head bilinear decoder.
    """
    preset_map = {
        "v2": {
            "model": "dense_gcn",
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.5,
            "lr": 5e-4,
            "patience": 30,
            "loss": "mse",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
        },
        "v3": {
            "model": "dense_gcn",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 45,
            "loss": "smoothl1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
        },
        "v4": {
            "model": "dense_gat",
            "hidden_dim": 192,
            "num_layers": 4,
            "dropout": 0.2,   # more reg than 0.1
            "lr": 8e-4,
            "patience": 50,
            "loss": "smoothl1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 0,  # drop HR refine (no structure; overfits)
            "edge_scale": 0.3,     # stronger graph bias (was 0.1)
        },
        "v5": {
            "model": "dense_gat",
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.4,
            "lr": 5e-4,
            "patience": 45,
            "loss": "smoothl1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 2,
            "num_decoder_heads": 4,
            "hr_refine_layers": 0,
            "edge_scale": 0.7,
        },
        "gcn_ca": {
            "model": "dense_gcn_ca",
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 5e-4,
            "patience": 45,
            "loss": "smoothl1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 0,
            "edge_scale": 0.2,
        },
        "bisr": {
            "model": "dense_bisr",
            "hidden_dim": 192,
            "num_layers": 3,
            "bipartite_layers": 1,
            "dropout": 0.3,
            "lr": 8e-4,
            "patience": 50,
            "loss": "smoothl1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
        },
        "bisr_v2": {
            "model": "dense_bisr",
            "hidden_dim": 128,
            "num_layers": 3,
            "bipartite_layers": 1,
            "dropout": 0.4,
            "lr": 5e-4,
            "patience": 40,
            "loss": "smoothl1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
        },
        "gin": {
            "model": "dense_gin",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 45,
            "loss": "smoothl1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
        },
        "gps": {
            "model": "dense_gcn_gps",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 45,
            "loss": "smoothl1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
        },
        "graphsage": {
            "model": "dense_graphsage",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 45,
            "loss": "smoothl1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
        },
    }
    chosen = preset_map[args.preset]
    defaults = parse_args_defaults()
    for k, v in chosen.items():
        if getattr(args, k) == defaults[k]:
            setattr(args, k, v)
    return args


def parse_args_defaults() -> dict:
    """
    Centralized parser defaults for deciding if user explicitly overrode a flag.
    """
    return {
        "model": "dense_gcn",
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.5,
        "lr": 5e-4,
        "patience": 30,
        "loss": "mse",
        "l1_weight": 0.7,
        "huber_beta": 1.0,
        "num_heads": 4,
        "ffn_mult": 4,
        "num_decoder_heads": 4,
        "hr_refine_layers": 1,
        "edge_scale": 0.2,
        "bipartite_layers": 1,
    }


def build_loss(cfg: TrainConfig) -> nn.Module:
    if cfg.loss_name == "mse":
        return nn.MSELoss()
    if cfg.loss_name == "l1":
        return nn.L1Loss()
    if cfg.loss_name == "smoothl1":
        return nn.SmoothL1Loss(beta=cfg.huber_beta)
    if cfg.loss_name == "hybrid":
        # Wrapped below with explicit closure in training loops.
        return nn.Identity()
    raise ValueError(f"Unsupported loss: {cfg.loss_name}")


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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )
    base_loss_fn = build_loss(cfg)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses_epoch = []
        for x_vec, y_vec in tr_loader:
            a = vec_to_adj(x_vec, cfg.n_lr, vectorizer)
            pred = model(a, a)
            if cfg.loss_name == "hybrid":
                loss = cfg.l1_weight * nn.functional.l1_loss(pred, y_vec) + \
                    (1.0 - cfg.l1_weight) * nn.functional.mse_loss(pred, y_vec)
            else:
                loss = base_loss_fn(pred, y_vec)
            train_losses_epoch.append(loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.model_name in ("dense_gat", "dense_bisr", "dense_gcn_ca", "dense_gin", "dense_gcn_gps", "dense_graphsage"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_vec, y_vec in va_loader:
                a = vec_to_adj(x_vec, cfg.n_lr, vectorizer)
                pred = model(a, a)
                if cfg.loss_name == "hybrid":
                    val_loss = cfg.l1_weight * nn.functional.l1_loss(pred, y_vec) + \
                        (1.0 - cfg.l1_weight) * nn.functional.mse_loss(pred, y_vec)
                else:
                    val_loss = base_loss_fn(pred, y_vec)
                val_losses.append(val_loss.item())

        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
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
            mean_tr = np.mean(train_losses_epoch)
            if cfg.model_name in ("dense_gat", "dense_bisr", "dense_gcn_ca", "dense_gin", "dense_gcn_gps", "dense_graphsage"):
                print(f"  Fold {fold_id} | Epoch {epoch:3d}/{cfg.epochs} | Train: {mean_tr:.6f} | Val: {val_loss:.6f}{marker}")
            else:
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
    max_epochs: int | None = None,
    x_va: np.ndarray | None = None,
    y_va: np.ndarray | None = None,
    patience: int | None = None,
) -> DenseGCNGenerator:
    """Full retrain. If x_va/y_va provided, use early stopping; else run for max_epochs."""
    epochs = max_epochs if max_epochs is not None else cfg.max_epochs_full
    xtr = torch.from_numpy(x_tr).float().to(device)
    ytr = torch.from_numpy(y_tr).float().to(device)
    tr_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.batch_size, shuffle=True)

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    base_loss_fn = build_loss(cfg)
    if cfg.model_name in ("dense_gat", "dense_bisr", "dense_gcn_ca", "dense_gin", "dense_gcn_gps", "dense_graphsage"):
        grad_clip = lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    else:
        grad_clip = lambda: None

    use_early_stop = x_va is not None and y_va is not None and patience is not None
    if use_early_stop:
        xva = torch.from_numpy(x_va).float().to(device)
        yva = torch.from_numpy(y_va).float().to(device)
        va_loader = DataLoader(TensorDataset(xva, yva), batch_size=cfg.batch_size, shuffle=False)
    best_val = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x_vec, y_vec in tr_loader:
            a = vec_to_adj(x_vec, cfg.n_lr, vectorizer)
            pred = model(a, a)
            if cfg.loss_name == "hybrid":
                loss = cfg.l1_weight * nn.functional.l1_loss(pred, y_vec) + \
                    (1.0 - cfg.l1_weight) * nn.functional.mse_loss(pred, y_vec)
            else:
                loss = base_loss_fn(pred, y_vec)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip()
            optimizer.step()
            train_losses.append(loss.item())

        if use_early_stop:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_vec, y_vec in va_loader:
                    a = vec_to_adj(x_vec, cfg.n_lr, vectorizer)
                    pred = model(a, a)
                    if cfg.loss_name == "hybrid":
                        v = cfg.l1_weight * nn.functional.l1_loss(pred, y_vec) + (1.0 - cfg.l1_weight) * nn.functional.mse_loss(pred, y_vec)
                    else:
                        v = base_loss_fn(pred, y_vec)
                    val_losses.append(v.item())
            val_loss = float(np.mean(val_losses))
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Full-train | Epoch {epoch:3d}/{epochs} | Train: {np.mean(train_losses):.6f} | Val: {val_loss:.6f}")
            if stale_epochs >= patience:
                print(f"  Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                if best_state is not None:
                    model.load_state_dict(best_state)
                return model
        else:
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                print(f"  Full-train | Epoch {epoch:3d}/{epochs} | Loss: {np.mean(train_losses):.6f}")

    if use_early_stop and best_state is not None:
        model.load_state_dict(best_state)
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
    args = apply_preset(args)
    cfg = TrainConfig(
        model_name=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        huber_beta=args.huber_beta,
        l1_weight=args.l1_weight,
        seed=args.seed,
        num_folds=args.num_folds,
        num_heads=args.num_heads,
        ffn_mult=args.ffn_mult,
        num_decoder_heads=args.num_decoder_heads,
        hr_refine_layers=args.hr_refine_layers,
        edge_scale=getattr(args, "edge_scale", 0.2),
        bipartite_layers=getattr(args, "bipartite_layers", 1),
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

    progress_path = out_dir / "cv_progress.json"

    # --- --fresh: clear CV progress, checkpoints, and metrics cache ---
    if getattr(args, "fresh", False):
        if progress_path.exists():
            progress_path.unlink()
        for p in ckpt_dir.glob("fold_*.pt"):
            p.unlink()
        for p in out_dir.glob("fold_*_metrics_cache.json"):
            p.unlink()
        print("Fresh run: cleared CV progress, checkpoints, and metrics cache.")

    # --- Fold-level resume: load progress from previous interrupted run ---
    fold_records: list[dict] = []
    fold_metric_dicts: list[dict] = []
    fold_seconds: list[float] = []
    completed_folds: set[int] = set()

    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as _pf:
            prior = json.load(_pf)
        fold_records = prior.get("fold_records", [])
        fold_metric_dicts = [r["metrics"] for r in fold_records]
        fold_seconds = [r["fold_seconds"] for r in fold_records]
        completed_folds = {r["fold"] for r in fold_records}
        print(f"Resuming CV: {len(completed_folds)} fold(s) already done {sorted(completed_folds)}")

    peak_ram = current_ram_gb()
    cv_start = time.perf_counter()

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(x_train), start=1):
        if fold_id in completed_folds:
            print(f"\nFold {fold_id}: already completed, skipping.")
            continue

        print("\n" + "=" * 60)
        print(f"Fold {fold_id}: train={len(tr_idx)}, val={len(va_idx)}")
        print("=" * 60)

        # Clear metrics cache for this fold so we recompute for this run's predictions
        # (cache is keyed only by fold + sample count; reuse would show stale graph metrics)
        metric_cache = out_dir / f"fold_{fold_id}_metrics_cache.json"
        if metric_cache.exists():
            metric_cache.unlink()

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
        subsample = getattr(args, "graph_metrics_subsample", None)
        metric_cache = None if (getattr(args, "skip_graph_metrics", False) or subsample is not None) else (out_dir / f"fold_{fold_id}_metrics_cache.json")
        metrics = evaluate_fold(
            pred_mats, gt_mats, verbose=True, cache_path=metric_cache,
            skip_graph_metrics=getattr(args, "skip_graph_metrics", False),
            max_samples=subsample,
        )

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

        # Save fold-level progress for resume
        write_json(progress_path, {
            "fold_records": fold_records,
            "completed_folds": sorted(r["fold"] for r in fold_records),
        })

    total_seconds = float(time.perf_counter() - cv_start)
    mean_metrics, std_metrics = summarize_folds(fold_metric_dicts)

    cv_summary = {
        "task": f"dense_gcn_{args.preset}_3fold_cv",
        "device": str(device),
        "config": asdict(cfg),
        "folds": fold_records,
        "metric_order": METRIC_ORDER,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }
    resource_summary = {
        "task": f"dense_gcn_{args.preset}_3fold_cv",
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
    args = apply_preset(args)
    ensemble_seeds_str = getattr(args, "ensemble_seeds", None)
    seeds = [int(s.strip()) for s in ensemble_seeds_str.split(",")] if ensemble_seeds_str else [args.seed]

    cfg = TrainConfig(
        model_name=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        huber_beta=args.huber_beta,
        l1_weight=args.l1_weight,
        seed=seeds[0],
        num_folds=getattr(args, "num_folds", 3),
        num_heads=args.num_heads,
        ffn_mult=args.ffn_mult,
        num_decoder_heads=args.num_decoder_heads,
        hr_refine_layers=args.hr_refine_layers,
        edge_scale=getattr(args, "edge_scale", 0.2),
        bipartite_layers=getattr(args, "bipartite_layers", 1),
    )

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
    if len(seeds) > 1:
        print(f"Ensemble mode: {len(seeds)} seeds {seeds}")

    vectorizer = MatrixVectorizer()
    peak_ram = current_ram_gb()
    train_start = time.perf_counter()
    max_epochs = getattr(args, "max_epochs", 400)
    val_ratio = getattr(args, "val_ratio", 0.15)
    patience = getattr(args, "patience", 50)

    preds_list = []
    total_train_seconds = 0.0

    for i, seed in enumerate(seeds):
        cfg.seed = seed
        seed_everything(seed)
        if len(seeds) > 1:
            print(f"\n--- Ensemble seed {i+1}/{len(seeds)}: {seed} ---")

        x_tr, y_tr = x_train, y_train
        x_va, y_va = None, None
        if val_ratio > 0 and val_ratio < 1:
            x_tr, x_va, y_tr, y_va = train_test_split(
                x_train, y_train, test_size=val_ratio, random_state=seed, shuffle=True
            )
            if i == 0:
                print(f"Full retrain: train {x_tr.shape[0]}, val {x_va.shape[0]} (early stopping patience={patience})")

        fold_train_start = time.perf_counter()
        full_model = train_full(
            cfg, device, vectorizer, x_tr, y_tr,
            max_epochs=max_epochs, x_va=x_va, y_va=y_va, patience=patience if (x_va is not None) else None,
        )
        total_train_seconds += time.perf_counter() - fold_train_start

        checkpoint_path = out_dir / "checkpoints" / (f"full_model_seed{seed}.pt" if len(seeds) > 1 else args.checkpoint_name)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": full_model.state_dict(), "config": asdict(cfg)}, checkpoint_path)

        preds = predict_vectors(full_model, x_test, device, vectorizer, cfg.batch_size, cfg.n_lr)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        preds_list.append(preds)

    preds = np.mean(preds_list, axis=0)
    preds = np.clip(preds, a_min=0.0, a_max=1.0)
    assert preds.shape[1] == HR_FEATURES, f"Expected {HR_FEATURES} HR features, got {preds.shape[1]}"

    n_subjects, n_features = preds.shape
    ids = np.arange(1, n_subjects * n_features + 1)
    submission = np.column_stack([ids, preds.reshape(-1)])

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
        "task": f"dense_gcn_{args.preset}_full_retrain",
        "device": str(device),
        "config": asdict(cfg),
        "ensemble_seeds": seeds if len(seeds) > 1 else None,
        "train_seconds": total_train_seconds,
        "train_minutes": total_train_seconds / 60.0,
        "peak_ram_gb_rss": None if not np.isfinite(peak_ram) else float(peak_ram),
        "psutil_available": psutil is not None,
        "submission_path": str(submission_path),
        "submission_rows": int(n_subjects * n_features),
    }
    write_json(out_dir / "full_retrain_summary.json", full_summary)

    print("\n" + "=" * 60)
    print("Full retraining completed.")
    if len(seeds) > 1:
        print(f"Ensemble: {len(seeds)} models (seeds {seeds})")
    print(f"Submission: {submission_path} ({n_subjects*n_features:,} rows)")
    print(f"Train time: {total_train_seconds:.1f}s ({total_train_seconds/60.0:.2f} min)")
    if np.isfinite(peak_ram):
        print(f"Peak RAM RSS: {peak_ram:.3f} GB")
    else:
        print("Peak RAM RSS: N/A (install psutil to enable RAM tracking)")
    print(f"Saved: {out_dir / 'full_retrain_summary.json'}")


def run_tune(args: argparse.Namespace) -> None:
    """Optuna Bayesian optimization over DenseGAT hyperparams on 3-fold CV. Resumable via SQLite."""
    if optuna is None:
        raise ImportError("Optuna is required for tune mode. Install with: pip install optuna")

    args = apply_preset(args)
    if args.model != "dense_gat":
        print("Tune mode is for dense_gat. Use --preset v4 or --model dense_gat.")
        args.model = "dense_gat"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_path = out_dir / "optuna_study.db"
    storage_url = f"sqlite:///{storage_path}"
    n_trials = getattr(args, "n_trials", 20)
    fresh = getattr(args, "fresh", False)

    seed_everything(args.seed)
    device = get_device(args.device)
    data_dir = Path(args.data_dir)
    x_train, y_train, _ = load_data(data_dir)
    vectorizer = MatrixVectorizer()
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    def objective(trial: "optuna.Trial") -> float:
        edge_scale = trial.suggest_float("edge_scale", 0.1, 0.5)
        hr_refine_layers = trial.suggest_int("hr_refine_layers", 0, 1)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        lr = trial.suggest_float("lr", 5e-4, 1.5e-3, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 192])
        num_layers = trial.suggest_int("num_layers", 3, 4)

        cfg = TrainConfig(
            model_name="dense_gat",
            n_lr=N_LR,
            n_hr=N_HR,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=lr,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            huber_beta=args.huber_beta,
            l1_weight=args.l1_weight,
            seed=args.seed,
            num_folds=args.num_folds,
            num_heads=args.num_heads,
            ffn_mult=args.ffn_mult,
            num_decoder_heads=4,
            hr_refine_layers=hr_refine_layers,
            edge_scale=edge_scale,
        )
        fold_vals = []
        for fold_id, (tr_idx, va_idx) in enumerate(kf.split(x_train), start=1):
            _, best_val, _ = train_with_validation(
                cfg, device, vectorizer,
                x_train[tr_idx], y_train[tr_idx],
                x_train[va_idx], y_train[va_idx],
                fold_id,
            )
            fold_vals.append(best_val)
        mean_val = float(np.mean(fold_vals))
        trial.set_user_attr("fold_vals", fold_vals)
        return mean_val

    if fresh and storage_path.exists():
        storage_path.unlink()
        print("Fresh run: cleared Optuna study.")

    study = optuna.create_study(
        direction="minimize",
        storage=storage_url,
        load_if_exists=not fresh,
        study_name="dense_gat_tune",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_mean_val = study.best_value
    best_config = {
        "model_name": "dense_gat",
        "n_lr": N_LR,
        "n_hr": N_HR,
        "hidden_dim": best_params["hidden_dim"],
        "num_layers": best_params["num_layers"],
        "dropout": best_params["dropout"],
        "learning_rate": best_params["lr"],
        "edge_scale": best_params["edge_scale"],
        "hr_refine_layers": best_params["hr_refine_layers"],
    }
    all_results = [
        {"params": t.params, "value": t.value}
        for t in study.trials
        if t.value is not None
    ]

    out_config_path = getattr(args, "out_config", None)
    if out_config_path is None:
        out_config_path = out_dir / "best_tune_config.json"
    else:
        out_config_path = Path(out_config_path)

    write_json(out_config_path, {
        "best_mean_val_loss": best_mean_val,
        "config": best_config,
        "best_params": best_params,
        "all_results": all_results,
    })
    print(f"\nBest config saved to {out_config_path}")
    print(f"Best mean val loss: {best_mean_val:.6f}")
    print(f"Best params: {best_params}")

    if getattr(args, "full_retrain", False):
        submission_path = getattr(args, "submission_path", None) or str(out_dir / "submission.csv")
        submission_path = Path(submission_path)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        val_ratio = 0.15
        patience_full = 50
        max_epochs_full = 400
        print(f"\n--full-retrain: running full retrain with best config (val_ratio={val_ratio}, patience={patience_full}, max_epochs={max_epochs_full}) -> {submission_path}")
        cfg = TrainConfig(
            model_name="dense_gat",
            n_lr=N_LR,
            n_hr=N_HR,
            hidden_dim=best_params["hidden_dim"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=best_params["lr"],
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            huber_beta=args.huber_beta,
            l1_weight=args.l1_weight,
            seed=args.seed,
            num_folds=args.num_folds,
            num_heads=args.num_heads,
            ffn_mult=args.ffn_mult,
            num_decoder_heads=4,
            hr_refine_layers=best_params["hr_refine_layers"],
            edge_scale=best_params["edge_scale"],
        )
        x_train, y_train, x_test = load_data(data_dir)
        x_tr, x_va, y_tr, y_va = train_test_split(
            x_train, y_train, test_size=val_ratio, random_state=cfg.seed, shuffle=True
        )
        full_model = train_full(
            cfg, device, vectorizer, x_tr, y_tr,
            max_epochs=max_epochs_full, x_va=x_va, y_va=y_va, patience=patience_full,
        )
        ckpt_path = out_dir / "checkpoints" / "full_model_best_tune.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": full_model.state_dict(), "config": asdict(cfg)}, ckpt_path)
        preds = predict_vectors(full_model, x_test, device, vectorizer, cfg.batch_size, cfg.n_lr)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        assert preds.shape[1] == HR_FEATURES, f"Expected {HR_FEATURES} HR features, got {preds.shape[1]}"
        n_subjects, n_features = preds.shape
        ids = np.arange(1, n_subjects * n_features + 1)
        submission = np.column_stack([ids, preds.reshape(-1)])
        np.savetxt(submission_path, submission, delimiter=",", header="ID,Predicted", comments="", fmt=["%d", "%.10f"])
        print(f"Full retrain done. Submission: {submission_path}")
    else:
        print(f"Run full retrain with:")
        print(f"  .venv/bin/python -m src.train_dense_gcn full --preset v4 --max-epochs 400 --val-ratio 0.15 --patience 50 \\")
        print(f"    --edge-scale {best_params['edge_scale']} --hr-refine-layers {best_params['hr_refine_layers']} \\")
        print(f"    --dropout {best_params['dropout']} --lr {best_params['lr']} \\")
        print(f"    --hidden-dim {best_params['hidden_dim']} --num-layers {best_params['num_layers']} \\")
        print(f"    --submission-path submission/dense_gat_v4_submission.csv")
        print(f"Or re-run tune with --full-retrain to do it automatically.")


def main() -> None:
    args = parse_args()
    if args.mode == "cv":
        run_cv(args)
    elif args.mode == "full":
        run_full(args)
    elif args.mode == "tune":
        run_tune(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
