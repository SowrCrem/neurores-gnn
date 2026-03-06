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

For hyperparameter tuning, use the tune subcommand with --preset.
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
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models.dense_bisr import DenseBiSRGenerator
from models.dense_gin import DenseGINGenerator
from models.dense_gcn import DenseGCNGenerator
from models.dense_gcn_ca import DenseGCNCrossAttnGenerator
from models.dense_gcn_gps import DenseGCNGPSGenerator
from models.dense_gcn_lrs import DenseGCNLRSGenerator
from models.dense_graphsage import DenseGraphSAGEGenerator
from models.dense_gat import DenseGATGenerator
from models.dense_stp import DenseSTPGenerator
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
    use_residual: bool = False   # subtract per-edge HR mean from targets; add back at inference
    mixup_alpha: float = 0.0     # Graph Mixup alpha: 0=disabled, 0.2=recommended
    subject_scale: bool = False  # per-subject LR std normalisation (divide LR & HR by subject's LR std)
    calibrate: bool = False      # post-hoc linear calibration of val predictions (slope+intercept per fold)
    lap_pe_dim: int = 0          # Laplacian PE: number of eigenvectors appended to node features (0=disabled)
    pearl_pe_dim: int = 0        # PEARL positional encoding dimension (0=disabled)
    lr_schedule: str = "plateau" # LR schedule: 'plateau' (ReduceLROnPlateau), 'cosine' (CosineAnnealingLR), 'none'
    lr_min: float = 1e-6         # minimum LR for cosine / plateau schedules
    curriculum_phase_epochs: int = 0   # Phase 1: train on heavy edges only; 0=disabled
    curriculum_heavy_percentile: float = 50.0  # Percentile for "heavy" edges (top X% by mean weight)
    spectral_alignment_weight: float = 0.0   # Auxiliary loss: L_total = L_base + weight * L_spec; 0=disabled
    spectral_alignment_k: int = 5      # Top-k eigenvalues for spectral loss (5–20)
    use_edge_bias: bool = True         # Learnable per-edge bias in decoder (v3r_eb, v3r_eb_ffnn); False = legacy v3r
    # Stronger data augmentation (0=disabled)
    edge_dropout: float = 0.0         # LR edge dropout prob (0.05–0.15); forces robustness to missing edges
    gaussian_noise_std: float = 0.0   # Add N(0, std) to LR adjacency (0.01–0.05); clamp to [0,1]
    mixup_prob: float = 0.5           # Probability of applying mixup when alpha > 0 (default 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DenseGCN runner (3F-CV logs + full retraining submission)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
        p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
        p.add_argument("--preset", type=str, default="v2", choices=["v2", "v3", "v3r", "v3r_eb", "v3r_eb_ffnn", "v3r_eb_ffnn_aug", "v3r_eb_ffnn_spec", "v3sn", "v3r_pe", "v3r_lrs", "v3r_cos", "v4", "v5", "bisr", "bisr_v2", "bisr_v3r", "gcn_ca", "gin", "gin_n", "gin_v3r", "gps", "gps_v3r", "graphsage", "sage_v3r", "stp", "stp_pe"])
        p.add_argument("--model", type=str, default="dense_gcn", choices=["dense_gcn", "dense_gat", "dense_bisr", "dense_gcn_ca", "dense_gin", "dense_gcn_gps", "dense_gcn_lrs", "dense_graphsage", "dense_stp"])
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
        p.add_argument("--use-residual", action="store_true", default=False, help="Subtract per-edge HR mean from targets; add back at inference (residual learning)")
        p.add_argument("--mixup-alpha", type=float, default=0.0, help="Graph Mixup alpha; 0=disabled, 0.2=recommended")
        p.add_argument("--subject-scale", action="store_true", default=False, help="Per-subject LR-std normalisation of LR & HR edge weights before training")
        p.add_argument("--calibrate", action="store_true", default=False, help="Post-hoc linear calibration of predictions using val fold (slope+intercept)")
        p.add_argument("--lap-pe-dim", type=int, default=0, help="Laplacian PE: number of eigenvectors appended as node features (0=disabled, 4=recommended)")
        p.add_argument("--pearl-pe-dim", type=int, default=0, help="PEARL positional encoding: number of learnable features (0=disabled, 32-128=recommended)")
        p.add_argument("--lr-schedule", type=str, default="plateau", choices=["plateau", "cosine", "none"],
                       help="LR schedule: 'plateau' (ReduceLROnPlateau, default), 'cosine' (CosineAnnealingLR), 'none'")
        p.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR for cosine/plateau schedule (default 1e-6)")
        p.add_argument("--curriculum-phase-epochs", type=int, default=0,
                       help="Curriculum: phase 1 trains on heavy edges only for this many epochs; 0=disabled")
        p.add_argument("--curriculum-heavy-percentile", type=float, default=50.0,
                       help="Curriculum: percentile for heavy edges (top X%% by mean weight; default 50)")
        p.add_argument("--spectral-alignment-weight", type=float, default=0.0,
                       help="Spectral alignment auxiliary loss weight; 0=disabled, 0.01=recommended")
        p.add_argument("--spectral-alignment-k", type=int, default=5,
                       help="Number of top eigenvalues for spectral loss (default 5)")
        p.add_argument("--use-edge-bias", action="store_true", default=False,
                       help="Learnable per-edge bias in decoder (v3r_eb, v3r_eb_ffnn)")
        p.add_argument("--no-edge-bias", action="store_true", default=False,
                       help="Disable per-edge bias (legacy v3r)")
        p.add_argument("--edge-dropout", type=float, default=0.0,
                       help="LR edge dropout prob (0=disabled, 0.05-0.15); forces robustness to missing edges")
        p.add_argument("--gaussian-noise-std", type=float, default=0.0,
                       help="Add N(0,std) to LR adjacency (0=disabled, 0.01-0.05); clamp to [0,1]")
        p.add_argument("--mixup-prob", type=float, default=0.5,
                       help="Probability of applying mixup when alpha>0 (default 0.5)")

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
    full.add_argument(
        "--shrinkage-eps",
        type=float,
        default=0.05,
        help="Inference shrinkage toward training mean: pred=(1-eps)*pred+eps*mean. 0=disabled. Try 0.03-0.10; stop when MAE worsens.",
    )
    full.add_argument("--bias-correct", action="store_true", default=False,
                      help="Post-hoc bias: subtract mean(pred-gt) on val from test preds. Requires val-ratio>0.")

    tune = sub.add_parser("tune", help="Hyperparameter tuning: Optuna Bayesian optimization on 3-fold CV.")
    add_shared(tune)
    tune.add_argument("--num-folds", type=int, default=3)
    tune.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials (default 20)")
    tune.add_argument("--out-config", type=str, default=None, help="Path to save best config JSON for full retrain")
    tune.add_argument("--fresh", action="store_true", help="Clear Optuna study and start from scratch")
    tune.add_argument("--full-retrain", action="store_true", help="After tuning, run full retrain with best config and save submission")
    tune.add_argument("--submission-path", type=str, default=None, help="Submission CSV path (used with --full-retrain; default: out-dir/submission.csv)")
    tune.add_argument("--shrinkage-eps", type=float, default=0.05, help="Inference shrinkage (used with --full-retrain). 0=disabled.")

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


def spectral_alignment_loss(
    pred_vec: torch.Tensor,
    gt_vec: torch.Tensor,
    n: int,
    k: int,
    vectorizer: MatrixVectorizer,
    device: torch.device,
) -> torch.Tensor:
    """
    L_spec = || λ(L_pred) - λ(L_gt) ||₂² for top-k eigenvalues of normalized Laplacian.

    L = I - D^{-1/2} A D^{-1/2}. Takes top-k largest eigenvalues (last k from eigvalsh).
    """
    pred_mats = vec_to_adj(pred_vec, n, vectorizer)  # (B, n, n)
    gt_mats = vec_to_adj(gt_vec, n, vectorizer)  # (B, n, n)
    pred_mats = pred_mats.clamp(min=0.0)
    gt_mats = gt_mats.clamp(min=0.0)

    eps = 1e-6
    D_pred = pred_mats.sum(dim=-1) + eps  # (B, n)
    D_gt = gt_mats.sum(dim=-1) + eps  # (B, n)
    D_pred_inv_sqrt = D_pred ** (-0.5)
    D_gt_inv_sqrt = D_gt ** (-0.5)
    # L = I - D^{-1/2} A D^{-1/2}
    L_pred = pred_mats * D_pred_inv_sqrt.unsqueeze(-1) * D_pred_inv_sqrt.unsqueeze(-2)
    L_pred = torch.eye(n, device=device, dtype=pred_mats.dtype).unsqueeze(0) - L_pred
    L_gt = gt_mats * D_gt_inv_sqrt.unsqueeze(-1) * D_gt_inv_sqrt.unsqueeze(-2)
    L_gt = torch.eye(n, device=device, dtype=gt_mats.dtype).unsqueeze(0) - L_gt

    eig_pred = torch.linalg.eigvalsh(L_pred)  # (B, n), ascending
    eig_gt = torch.linalg.eigvalsh(L_gt)  # (B, n), ascending
    eig_pred_k = eig_pred[:, -k:]
    eig_gt_k = eig_gt[:, -k:]
    return ((eig_pred_k - eig_gt_k) ** 2).mean()


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
            raw_output=cfg.use_residual,
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
            raw_output=cfg.use_residual,
        ).to(device)
    if cfg.model_name == "dense_gcn_gps":
        return DenseGCNGPSGenerator(
            n_lr=cfg.n_lr,
            n_hr=cfg.n_hr,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            raw_output=cfg.use_residual,
        ).to(device)
    if cfg.model_name == "dense_gcn_lrs":
        return DenseGCNLRSGenerator(
            n_lr=cfg.n_lr,
            n_hr=cfg.n_hr,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            raw_output=cfg.use_residual,
            lap_pe_dim=cfg.lap_pe_dim,
        ).to(device)
    if cfg.model_name == "dense_graphsage":
        return DenseGraphSAGEGenerator(
            n_lr=cfg.n_lr,
            n_hr=cfg.n_hr,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(device)
    if cfg.model_name == "dense_stp":
        return DenseSTPGenerator(
            n_lr=cfg.n_lr,
            n_hr=cfg.n_hr,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            raw_output=cfg.use_residual,
            lap_pe_dim=cfg.lap_pe_dim,
            pearl_pe_dim=cfg.pearl_pe_dim,
        ).to(device)
    return DenseGCNGenerator(
        n_lr=cfg.n_lr,
        n_hr=cfg.n_hr,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        raw_output=cfg.use_residual,
        lap_pe_dim=cfg.lap_pe_dim,
        pearl_pe_dim=cfg.pearl_pe_dim,
        ffn_mult=cfg.ffn_mult,
        use_edge_bias=cfg.use_edge_bias,
    ).to(device)


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """
    Fill defaults according to an experiment preset.

    v2: existing dense_gcn_v2 baseline.
    v3: balanced capacity + MAE-aligned objective.
    v4: DenseGATNet - graph-attention encoder + multi-head bilinear decoder.
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
            "huber_beta": 0.05,   # was 1.0 - now L1 region kicks in at |err|>0.05 (appropriate for [0,1] edges)
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
        # bisr_v3r = Bi-SR + v3r training (L1, residual, mixup) - transfers from GIN v3r success
        "bisr_v3r": {
            "model": "dense_bisr",
            "hidden_dim": 192,
            "num_layers": 3,
            "bipartite_layers": 1,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
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
        # v3r = legacy baseline (no edge bias, no FFN) - reproduces v3r_shrinkage_008 (0.132)
        "v3r": {
            "model": "dense_gcn",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 0,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
            "use_edge_bias": False,
        },
        # v3r_eb = v3r + learnable per-edge bias - Dhruv "v3 + per edge bias" (0.127)
        "v3r_eb": {
            "model": "dense_gcn",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 0,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
            "use_edge_bias": True,
        },
        # v3r_eb_ffnn = v3r + edge bias + FFN in GCN blocks - Dhruv "v3 + edge bias + ffnn" (0.127)
        "v3r_eb_ffnn": {
            "model": "dense_gcn",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
            "use_edge_bias": True,
        },
        # v3r_eb_ffnn_aug = v3r_eb_ffnn + stronger augmentation (edge dropout, Gaussian noise, higher mixup prob)
        "v3r_eb_ffnn_aug": {
            "model": "dense_gcn",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.3,
            "mixup_prob": 0.8,
            "use_edge_bias": True,
            "edge_dropout": 0.08,
            "gaussian_noise_std": 0.02,
        },
        # v3r_eb_ffnn_spec = v3r_eb_ffnn + spectral alignment loss (k=10; v3r got 0.1315 with spectral)
        "v3r_eb_ffnn_spec": {
            "model": "dense_gcn",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
            "use_edge_bias": True,
            "spectral_alignment_weight": 0.01,
            "spectral_alignment_k": 10,
        },
        # v3sn = v3r + per-subject scale normalisation + post-hoc calibration
        "v3sn": {
            "model": "dense_gcn",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 0,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
            "use_edge_bias": False,
            "subject_scale": True,
            "calibrate": True,
        },
        # gin_n = GIN with normalised adjacency + subject scale normalisation
        "gin_n": {
            "model": "dense_gin",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 45,
            "loss": "smoothl1",
            "l1_weight": 0.7,
            "huber_beta": 0.05,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "subject_scale": True,
            "calibrate": True,
        },
        # v3r_pe = v3r + Laplacian eigenvector positional encoding (k=4)
        "v3r_pe": {
            "model": "dense_gcn",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 0,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
            "use_edge_bias": False,
            "lap_pe_dim": 4,
        },
        # gin_v3r = GIN (normalised adjacency) + v3r training improvements
        "gin_v3r": {
            "model": "dense_gin",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
        },
        # gps_v3r = GraphGPS (GCN + linear attention) + v3r training improvements
        "gps_v3r": {
            "model": "dense_gcn_gps",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
        },
        # sage_v3r = GraphSAGE (mean aggregation) + v3r training improvements
        "sage_v3r": {
            "model": "dense_graphsage",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
        },
        # v3r_lrs = DenseGCN + Low-Rank + Sparse decoder + v3r training improvements
        "v3r_lrs": {
            "model": "dense_gcn_lrs",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
        },
        # v3r_cos = v3r + cosine annealing LR schedule (lr: 8e-4 → 1e-5 over T epochs)
        "v3r_cos": {
            "model": "dense_gcn",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 0,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
            "use_edge_bias": False,
            "lr_min": 1e-5,
        },
        "stp": {
            "model": "dense_stp",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
        },
        "stp_pe": {
            "model": "dense_stp",
            "hidden_dim": 192,
            "num_layers": 3,
            "dropout": 0.35,
            "lr": 8e-4,
            "patience": 60,
            "loss": "l1",
            "l1_weight": 0.7,
            "huber_beta": 1.0,
            "num_heads": 4,
            "ffn_mult": 4,
            "num_decoder_heads": 4,
            "hr_refine_layers": 1,
            "edge_scale": 0.2,
            "use_residual": True,
            "mixup_alpha": 0.2,
            "pearl_pe_dim": 128,
        },
    }
    chosen = preset_map[args.preset]
    defaults = parse_args_defaults()
    for k, v in chosen.items():
        if getattr(args, k, None) == defaults.get(k):
            setattr(args, k, v)
    # --no-edge-bias overrides preset
    if getattr(args, "no_edge_bias", False):
        args.use_edge_bias = False
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
        "use_residual": False,
        "mixup_alpha": 0.0,
        "subject_scale": False,
        "calibrate": False,
        "lap_pe_dim": 0,
        "pearl_pe_dim": 0,
        "lr_schedule": "plateau",
        "lr_min": 1e-6,
        "use_edge_bias": False,
        "spectral_alignment_weight": 0.0,
        "spectral_alignment_k": 5,
        "edge_dropout": 0.0,
        "gaussian_noise_std": 0.0,
        "mixup_prob": 0.5,
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


# ---------------------------------------------------------------------------
# Per-subject scale normalisation helpers
# ---------------------------------------------------------------------------

def compute_subject_scales(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute per-subject scale = std of each subject's edge-weight vector.

    Args:
        x: (N, E) array of vectorised adjacency matrices, one row per subject.
        eps: Floor to prevent division by zero for near-constant subjects.
    Returns:
        scales: (N, 1) float32 array, one scale per subject.
    """
    scales = x.std(axis=1, keepdims=True).astype(np.float32)
    return np.maximum(scales, eps)


def apply_subject_scale(x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Divide each subject's edge-weight vector by its own scale.

    Args:
        x:      (N, E) edge-weight matrix.
        scales: (N, 1) per-subject scales from compute_subject_scales().
    Returns:
        x_norm: (N, E) normalised array (same dtype as x).
    """
    return (x / scales).astype(x.dtype)


# ---------------------------------------------------------------------------
# Post-hoc calibration helpers
# ---------------------------------------------------------------------------

def fit_calibration(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Fit a linear calibration y_cal = slope * y_pred + intercept via OLS.

    Flattens both arrays and solves the 1-D least-squares problem.  Used to
    remove systematic bias introduced by softplus / residual learning offsets.

    Args:
        pred: (N, E) predicted edge weights.
        true: (N, E) ground-truth edge weights.
    Returns:
        (slope, intercept): calibration coefficients.
    """
    p = pred.reshape(-1).astype(np.float64)
    t = true.reshape(-1).astype(np.float64)
    # OLS: [slope, intercept] = (X^T X)^{-1} X^T t  with X = [p, 1]
    A = np.stack([p, np.ones_like(p)], axis=1)
    result, *_ = np.linalg.lstsq(A, t, rcond=None)
    slope, intercept = float(result[0]), float(result[1])
    return slope, intercept


def apply_calibration(
    pred: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """Apply linear calibration and clamp to [0, inf).

    Args:
        pred:      (N, E) raw predictions.
        slope:     calibration slope.
        intercept: calibration intercept.
    Returns:
        pred_cal: (N, E) calibrated predictions, clipped to >= 0.
    """
    pred_cal = slope * pred + intercept
    return np.clip(pred_cal, a_min=0.0, a_max=None).astype(pred.dtype)


def compute_bias_correction(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Compute per-edge bias: mean(pred - gt). Shape (E,)."""
    return (pred - true).mean(axis=0).astype(np.float32)




def compute_lap_pe(
    x_vecs: np.ndarray,
    n_lr: int,
    k: int,
    vectorizer: MatrixVectorizer,
) -> np.ndarray:
    """Compute k-dim Laplacian eigenvector PE for each subject.

    Uses the k eigenvectors of the normalised Laplacian L = I - D^{-1/2}AD^{-1/2}
    corresponding to the *smallest non-trivial* eigenvalues (i.e. skip the 0-eigenvalue).
    Sign ambiguity is fixed: each eigenvector is flipped so its first non-zero element
    is positive (canonical sign convention from Dwivedi et al., 2022).

    Args:
        x_vecs:     (N, E) upper-triangle adjacency vectors, one row per subject.
        n_lr:       number of LR nodes (160).
        k:          number of eigenvectors to use as PE.
        vectorizer: MatrixVectorizer used elsewhere for adjacency conversion.
    Returns:
        (N, n_lr * k) float32 array of flattened per-subject PE matrices.
    """
    N = len(x_vecs)
    pe_all = np.zeros((N, n_lr, k), dtype=np.float32)
    I = np.eye(n_lr, dtype=np.float64)
    for i, x_vec in enumerate(x_vecs):
        A = vectorizer.anti_vectorize(x_vec, n_lr, include_diagonal=False).astype(np.float64)
        A = (A + A.T) / 2.0  # enforce symmetry
        # Row-normalise: D^{-1/2} A D^{-1/2}
        deg = A.sum(axis=1).clip(min=1e-8)
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        S = (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]  # normalised adjacency
        L = I - S  # normalised Laplacian
        # Compute smallest k+1 eigenpairs; eigvecs sorted ascending by eigenvalue
        eigvals, eigvecs = np.linalg.eigh(L)  # (n_lr,), (n_lr, n_lr) sorted ascending
        # Skip trivial eigenvector (eigenvalue ≈ 0); take columns 1..k
        pe = eigvecs[:, 1:k + 1].astype(np.float32)  # (n_lr, k)
        # Pad if not enough eigenvectors (shouldn't happen for n_lr=160, k=4)
        if pe.shape[1] < k:
            pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
        # Sign fix: flip each eigenvector so its first large-magnitude element is positive
        for j in range(k):
            col = pe[:, j]
            nz = np.where(np.abs(col) > 1e-6)[0]
            if len(nz) > 0 and col[nz[0]] < 0:
                pe[:, j] = -col
        pe_all[i] = pe
    return pe_all.reshape(N, n_lr * k)  # (N, n_lr * k)


def mixup_batch(
    x_vec: torch.Tensor, y_vec: torch.Tensor, alpha: float, prob: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph Mixup: λ ~ Beta(alpha, alpha), applied with given probability.

    Convex combinations of symmetric connectivity matrices are valid brain
    graphs (symmetric, non-negative, [0,1]-bounded under clamp), so mixup
    in adjacency space is domain-valid.
    """
    if alpha <= 0.0 or np.random.random() > prob:
        return x_vec, y_vec
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x_vec.size(0), device=x_vec.device)
    x_mix = lam * x_vec + (1.0 - lam) * x_vec[idx]
    y_mix = lam * y_vec + (1.0 - lam) * y_vec[idx]
    return x_mix, y_mix


def augment_lr_batch(
    x_vec: torch.Tensor,
    n_lr_e: int,
    edge_dropout: float,
    gaussian_noise_std: float,
) -> torch.Tensor:
    """Apply edge dropout and Gaussian noise to LR adjacency (first n_lr_e elements)."""
    if edge_dropout <= 0.0 and gaussian_noise_std <= 0.0:
        return x_vec
    out = x_vec.clone()
    adj = out[:, :n_lr_e]
    if edge_dropout > 0.0:
        mask = torch.rand_like(adj, device=adj.device) > edge_dropout
        adj = adj * mask
    if gaussian_noise_std > 0.0:
        noise = torch.randn_like(adj, device=adj.device) * gaussian_noise_std
        adj = (adj + noise).clamp(min=0.0, max=1.0)
    out[:, :n_lr_e] = adj
    return out


def train_with_validation(
    cfg: TrainConfig,
    device: torch.device,
    vectorizer: MatrixVectorizer,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    fold_id: int,
    y_mean: np.ndarray | None = None,
) -> tuple[DenseGCNGenerator, float, int]:
    n_lr_e = cfg.n_lr * (cfg.n_lr - 1) // 2  # length of adjacency upper-triangle vector
    xtr = torch.from_numpy(x_tr).float().to(device)
    ytr = torch.from_numpy(y_tr).float().to(device)
    xva = torch.from_numpy(x_va).float().to(device)
    yva = torch.from_numpy(y_va).float().to(device)

    # Residual learning: subtract per-edge mean so the model predicts deviations
    if cfg.use_residual and y_mean is not None:
        y_mean_t = torch.from_numpy(y_mean).float().to(device)
        ytr = ytr - y_mean_t.unsqueeze(0)
        yva = yva - y_mean_t.unsqueeze(0)

    tr_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(xva, yva), batch_size=cfg.batch_size, shuffle=False)

    # Curriculum: phase 1 = heavy edges only (top X% by mean weight)
    edge_mask_t: torch.Tensor | None = None
    if cfg.curriculum_phase_epochs > 0:
        y_mean_edges = y_tr.mean(axis=0)
        thresh = np.percentile(y_mean_edges, cfg.curriculum_heavy_percentile)
        edge_mask = (y_mean_edges >= thresh)
        edge_mask_t = torch.from_numpy(edge_mask).to(device)
        n_heavy = int(edge_mask.sum())
        print(f"  Curriculum: phase 1 (epochs 1–{cfg.curriculum_phase_epochs}) on {n_heavy}/{len(edge_mask)} heavy edges (≥p{cfg.curriculum_heavy_percentile:.0f})")

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min
        )
    elif cfg.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15, min_lr=cfg.lr_min
        )
    else:
        scheduler = None
    base_loss_fn = build_loss(cfg)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        use_curriculum = edge_mask_t is not None and epoch <= cfg.curriculum_phase_epochs
        train_losses_epoch = []
        for x_vec, y_vec in tr_loader:
            # Graph Mixup augmentation
            x_vec, y_vec = mixup_batch(x_vec, y_vec, cfg.mixup_alpha, cfg.mixup_prob)
            # Edge dropout + Gaussian noise on LR adjacency
            x_vec = augment_lr_batch(
                x_vec, n_lr_e,
                cfg.edge_dropout, cfg.gaussian_noise_std,
            )
            if cfg.lap_pe_dim > 0:
                adj_vec = x_vec[:, :n_lr_e]
                pe_vec  = x_vec[:, n_lr_e:].reshape(x_vec.size(0), cfg.n_lr, cfg.lap_pe_dim)
                a = vec_to_adj(adj_vec, cfg.n_lr, vectorizer)
                pred = model(a, a, pe_vec)
            else:
                a = vec_to_adj(x_vec, cfg.n_lr, vectorizer)
                pred = model(a, a)
            if use_curriculum:
                pred_m = pred[:, edge_mask_t]
                y_m = y_vec[:, edge_mask_t]
                if cfg.loss_name == "hybrid":
                    loss = cfg.l1_weight * nn.functional.l1_loss(pred_m, y_m) + \
                        (1.0 - cfg.l1_weight) * nn.functional.mse_loss(pred_m, y_m)
                else:
                    loss = base_loss_fn(pred_m, y_m)
            else:
                if cfg.loss_name == "hybrid":
                    loss = cfg.l1_weight * nn.functional.l1_loss(pred, y_vec) + \
                        (1.0 - cfg.l1_weight) * nn.functional.mse_loss(pred, y_vec)
                else:
                    loss = base_loss_fn(pred, y_vec)
            if cfg.spectral_alignment_weight > 0:
                l_spec = spectral_alignment_loss(
                    pred, y_vec, cfg.n_hr, cfg.spectral_alignment_k,
                    vectorizer, device,
                )
                loss = loss + cfg.spectral_alignment_weight * l_spec
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
                if cfg.lap_pe_dim > 0:
                    adj_vec = x_vec[:, :n_lr_e]
                    pe_vec  = x_vec[:, n_lr_e:].reshape(x_vec.size(0), cfg.n_lr, cfg.lap_pe_dim)
                    a = vec_to_adj(adj_vec, cfg.n_lr, vectorizer)
                    pred = model(a, a, pe_vec)
                else:
                    a = vec_to_adj(x_vec, cfg.n_lr, vectorizer)
                    pred = model(a, a)
                if cfg.loss_name == "hybrid":
                    val_loss = cfg.l1_weight * nn.functional.l1_loss(pred, y_vec) + \
                        (1.0 - cfg.l1_weight) * nn.functional.mse_loss(pred, y_vec)
                else:
                    val_loss = base_loss_fn(pred, y_vec)
                if cfg.spectral_alignment_weight > 0:
                    l_spec = spectral_alignment_loss(
                        pred, y_vec, cfg.n_hr, cfg.spectral_alignment_k,
                        vectorizer, device,
                    )
                    val_loss = val_loss + cfg.spectral_alignment_weight * l_spec
                val_losses.append(val_loss.item())

        val_loss = float(np.mean(val_losses))
        if scheduler is not None:
            if cfg.lr_schedule == "cosine":
                scheduler.step()
            else:
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
    y_mean: np.ndarray | None = None,
) -> DenseGCNGenerator:
    """Full retrain. If x_va/y_va provided, use early stopping; else run for max_epochs."""
    epochs = max_epochs if max_epochs is not None else cfg.max_epochs_full
    xtr = torch.from_numpy(x_tr).float().to(device)
    ytr = torch.from_numpy(y_tr).float().to(device)

    # Residual learning: subtract per-edge mean so the model predicts deviations
    if cfg.use_residual and y_mean is not None:
        y_mean_t = torch.from_numpy(y_mean).float().to(device)
        ytr = ytr - y_mean_t.unsqueeze(0)

    tr_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.batch_size, shuffle=True)

    # Curriculum: phase 1 = heavy edges only
    edge_mask_t: torch.Tensor | None = None
    if cfg.curriculum_phase_epochs > 0:
        y_mean_edges = y_tr.mean(axis=0)
        thresh = np.percentile(y_mean_edges, cfg.curriculum_heavy_percentile)
        edge_mask = (y_mean_edges >= thresh)
        edge_mask_t = torch.from_numpy(edge_mask).to(device)
        n_heavy = int(edge_mask.sum())
        print(f"  Curriculum: phase 1 (epochs 1–{cfg.curriculum_phase_epochs}) on {n_heavy}/{len(edge_mask)} heavy edges")

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    if cfg.lr_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=cfg.lr_min
        )
    else:
        lr_scheduler = None
    base_loss_fn = build_loss(cfg)
    if cfg.model_name in ("dense_gat", "dense_bisr", "dense_gcn_ca", "dense_gin", "dense_gcn_gps", "dense_graphsage"):
        grad_clip = lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    else:
        grad_clip = lambda: None

    use_early_stop = x_va is not None and y_va is not None and patience is not None
    if use_early_stop:
        xva = torch.from_numpy(x_va).float().to(device)
        yva = torch.from_numpy(y_va).float().to(device)
        # Residual learning: val targets also centred on the same mean
        if cfg.use_residual and y_mean is not None:
            y_mean_t_va = torch.from_numpy(y_mean).float().to(device)
            yva = yva - y_mean_t_va.unsqueeze(0)
        va_loader = DataLoader(TensorDataset(xva, yva), batch_size=cfg.batch_size, shuffle=False)
    best_val = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        use_curriculum = edge_mask_t is not None and epoch <= cfg.curriculum_phase_epochs
        train_losses = []
        for x_vec, y_vec in tr_loader:
            # Graph Mixup augmentation
            x_vec, y_vec = mixup_batch(x_vec, y_vec, cfg.mixup_alpha, cfg.mixup_prob)
            # Edge dropout + Gaussian noise on LR adjacency
            n_lr_e_full = cfg.n_lr * (cfg.n_lr - 1) // 2
            x_vec = augment_lr_batch(
                x_vec, n_lr_e_full,
                cfg.edge_dropout, cfg.gaussian_noise_std,
            )
            if cfg.lap_pe_dim > 0:
                adj_vec = x_vec[:, :n_lr_e_full]
                pe_vec  = x_vec[:, n_lr_e_full:].reshape(x_vec.size(0), cfg.n_lr, cfg.lap_pe_dim)
                a = vec_to_adj(adj_vec, cfg.n_lr, vectorizer)
                pred = model(a, a, pe_vec)
            else:
                a = vec_to_adj(x_vec, cfg.n_lr, vectorizer)
                pred = model(a, a)
            if use_curriculum:
                pred_m, y_m = pred[:, edge_mask_t], y_vec[:, edge_mask_t]
                if cfg.loss_name == "hybrid":
                    loss = cfg.l1_weight * nn.functional.l1_loss(pred_m, y_m) + \
                        (1.0 - cfg.l1_weight) * nn.functional.mse_loss(pred_m, y_m)
                else:
                    loss = base_loss_fn(pred_m, y_m)
            else:
                if cfg.loss_name == "hybrid":
                    loss = cfg.l1_weight * nn.functional.l1_loss(pred, y_vec) + \
                        (1.0 - cfg.l1_weight) * nn.functional.mse_loss(pred, y_vec)
                else:
                    loss = base_loss_fn(pred, y_vec)
            if cfg.spectral_alignment_weight > 0:
                l_spec = spectral_alignment_loss(
                    pred, y_vec, cfg.n_hr, cfg.spectral_alignment_k,
                    vectorizer, device,
                )
                loss = loss + cfg.spectral_alignment_weight * l_spec
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip()
            optimizer.step()
            train_losses.append(loss.item())

        # Step LR scheduler once per epoch (cosine annealing)
        if lr_scheduler is not None:
            lr_scheduler.step()

        if use_early_stop:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_vec, y_vec in va_loader:
                    n_lr_e_full = cfg.n_lr * (cfg.n_lr - 1) // 2
                    if cfg.lap_pe_dim > 0:
                        adj_vec = x_vec[:, :n_lr_e_full]
                        pe_vec  = x_vec[:, n_lr_e_full:].reshape(x_vec.size(0), cfg.n_lr, cfg.lap_pe_dim)
                        a = vec_to_adj(adj_vec, cfg.n_lr, vectorizer)
                        pred = model(a, a, pe_vec)
                    else:
                        a = vec_to_adj(x_vec, cfg.n_lr, vectorizer)
                        pred = model(a, a)
                    if cfg.loss_name == "hybrid":
                        v = cfg.l1_weight * nn.functional.l1_loss(pred, y_vec) + (1.0 - cfg.l1_weight) * nn.functional.mse_loss(pred, y_vec)
                    else:
                        v = base_loss_fn(pred, y_vec)
                    if cfg.spectral_alignment_weight > 0:
                        l_spec = spectral_alignment_loss(
                            pred, y_vec, cfg.n_hr, cfg.spectral_alignment_k,
                            vectorizer, device,
                        )
                        v = v + cfg.spectral_alignment_weight * l_spec
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
    y_mean: np.ndarray | None = None,
    lap_pe_dim: int = 0,
) -> np.ndarray:
    """Run inference for all subjects. If y_mean is given, add it back (residual mode).

    When lap_pe_dim > 0, x_np is expected to be (N, n_lr_e + n_lr*lap_pe_dim) - the
    adjacency vector concatenated with the flattened Laplacian PE.
    """
    n_lr_e = n_lr * (n_lr - 1) // 2
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, len(x_np), batch_size):
            x_batch = torch.from_numpy(x_np[start:start + batch_size]).float().to(device)
            if lap_pe_dim > 0:
                adj_batch = x_batch[:, :n_lr_e]
                pe_batch  = x_batch[:, n_lr_e:].reshape(x_batch.size(0), n_lr, lap_pe_dim)
                a = vec_to_adj(adj_batch, n_lr, vectorizer)
                pred = model(a, a, pe_batch)
            else:
                a = vec_to_adj(x_batch, n_lr, vectorizer)
                pred = model(a, a)
            outputs.append(pred.cpu().numpy())

    result = np.concatenate(outputs, axis=0)
    if y_mean is not None:
        result = result + y_mean[np.newaxis, :]  # add per-edge mean back for residual mode
    return result


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
        use_residual=getattr(args, "use_residual", False),
        mixup_alpha=getattr(args, "mixup_alpha", 0.0),
        subject_scale=getattr(args, "subject_scale", False),
        calibrate=getattr(args, "calibrate", False),
        lap_pe_dim=getattr(args, "lap_pe_dim", 0),
        pearl_pe_dim=getattr(args, "pearl_pe_dim", 0),
        lr_schedule=getattr(args, "lr_schedule", "plateau"),
        lr_min=getattr(args, "lr_min", 1e-6),
        curriculum_phase_epochs=getattr(args, "curriculum_phase_epochs", 0),
        curriculum_heavy_percentile=getattr(args, "curriculum_heavy_percentile", 50.0),
        spectral_alignment_weight=getattr(args, "spectral_alignment_weight", 0.0),
        spectral_alignment_k=getattr(args, "spectral_alignment_k", 5),
        use_edge_bias=getattr(args, "use_edge_bias", False) and not getattr(args, "no_edge_bias", False),
        edge_dropout=getattr(args, "edge_dropout", 0.0),
        gaussian_noise_std=getattr(args, "gaussian_noise_std", 0.0),
        mixup_prob=getattr(args, "mixup_prob", 0.5),
    )

    seed_everything(cfg.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test = load_data(Path(args.data_dir))
    assert x_train.shape[0] == y_train.shape[0], "Mismatched train sample count."

    print(f"Device: {device}")
    print(f"Train LR: {x_train.shape} | Train HR: {y_train.shape}")
    print(f"Config: {asdict(cfg)}")
    if cfg.spectral_alignment_weight > 0:
        print(f"  Spectral alignment: weight={cfg.spectral_alignment_weight}, k={cfg.spectral_alignment_k}")
    if cfg.edge_dropout > 0 or cfg.gaussian_noise_std > 0:
        print(f"  Augmentation: edge_dropout={cfg.edge_dropout}, gaussian_noise_std={cfg.gaussian_noise_std}, mixup_prob={cfg.mixup_prob}")

    # --- Per-subject scale normalisation ---
    subj_scales_lr: np.ndarray | None = None
    if cfg.subject_scale:
        subj_scales_lr = compute_subject_scales(x_train)  # (N, 1), based on LR std
        x_train = apply_subject_scale(x_train, subj_scales_lr)
        y_train = apply_subject_scale(y_train, subj_scales_lr)  # same scale for HR
        print(f"  Subject-scale normalisation enabled. LR scale mean={subj_scales_lr.mean():.4f}, "
              f"min={subj_scales_lr.min():.4f}, max={subj_scales_lr.max():.4f}")

    vectorizer = MatrixVectorizer()

    # --- Laplacian PE: precompute once for all training subjects (before splitting folds) ---
    if cfg.lap_pe_dim > 0:
        print(f"  Precomputing Laplacian PE (k={cfg.lap_pe_dim}) for {len(x_train)} subjects ...")
        lap_pe_train = compute_lap_pe(x_train, cfg.n_lr, cfg.lap_pe_dim, vectorizer)
        x_train = np.concatenate([x_train, lap_pe_train], axis=1)  # (N, n_lr_e + n_lr*k)
        print(f"  x_train shape with LapPE: {x_train.shape}")

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
        # Compute per-fold HR mean for residual learning (training split only - no leakage)
        y_mean_fold = y_train[tr_idx].mean(axis=0) if cfg.use_residual else None
        model, best_val, best_epoch = train_with_validation(
            cfg,
            device,
            vectorizer,
            x_train[tr_idx],
            y_train[tr_idx],
            x_train[va_idx],
            y_train[va_idx],
            fold_id,
            y_mean=y_mean_fold,
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
            model, x_train[va_idx], device, vectorizer, cfg.batch_size, cfg.n_lr,
            y_mean=y_mean_fold, lap_pe_dim=cfg.lap_pe_dim,
        )
        gt_vecs = y_train[va_idx]

        # --- Un-scale predictions and GT back to original edge-weight units ---
        cal_slope: float | None = None
        cal_intercept: float | None = None
        if cfg.subject_scale and subj_scales_lr is not None:
            va_scales = subj_scales_lr[va_idx]  # (val_N, 1)
            pred_vecs = pred_vecs * va_scales
            gt_vecs   = gt_vecs  * va_scales

        # --- Post-hoc linear calibration (fit on val fold, apply to val fold) ---
        if cfg.calibrate:
            cal_slope, cal_intercept = fit_calibration(pred_vecs, gt_vecs)
            pred_vecs = apply_calibration(pred_vecs, cal_slope, cal_intercept)
            print(f"  Calibration | slope={cal_slope:.6f}, intercept={cal_intercept:.6f}")

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
                **({"cal_slope": cal_slope, "cal_intercept": cal_intercept} if cal_slope is not None else {}),
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

        # Save predictions_fold_{fold_num}.csv (Spec 3.1.1) - test predictions from this fold's model
        x_test_in = x_test
        if cfg.lap_pe_dim > 0:
            lap_pe_test = compute_lap_pe(x_test, cfg.n_lr, cfg.lap_pe_dim, vectorizer)
            x_test_in = np.concatenate([x_test, lap_pe_test], axis=1)
        test_pred = predict_vectors(
            model, x_test_in, device, vectorizer, cfg.batch_size, cfg.n_lr,
            y_mean=y_mean_fold, lap_pe_dim=cfg.lap_pe_dim,
        )
        test_pred = np.clip(test_pred, a_min=0.0, a_max=None)
        pred_path = out_dir / f"predictions_fold_{fold_id}.csv"
        n_subj, n_feat = test_pred.shape
        ids = np.arange(1, n_subj * n_feat + 1)
        pd.DataFrame({"ID": ids, "Predicted": test_pred.flatten()}).to_csv(pred_path, index=False)
        print(f"  Saved {pred_path}")

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

    # Ensemble of 3 folds -> submission.csv (Spec 3.1)
    pred_files = [out_dir / f"predictions_fold_{i}.csv" for i in (1, 2, 3)]
    if all(p.exists() for p in pred_files):
        dfs = [pd.read_csv(p) for p in pred_files]
        ensemble = np.clip(np.mean([d["Predicted"].values for d in dfs], axis=0), 0.0, None)
        sub_path = out_dir / "submission.csv"
        pd.DataFrame({"ID": dfs[0]["ID"], "Predicted": ensemble}).to_csv(sub_path, index=False)
        print(f"Saved: {sub_path} (ensemble of 3 folds)")

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
        use_residual=getattr(args, "use_residual", False),
        mixup_alpha=getattr(args, "mixup_alpha", 0.0),
        subject_scale=getattr(args, "subject_scale", False),
        calibrate=getattr(args, "calibrate", False),
        lap_pe_dim=getattr(args, "lap_pe_dim", 0),
        pearl_pe_dim=getattr(args, "pearl_pe_dim", 0),
        lr_schedule=getattr(args, "lr_schedule", "plateau"),
        lr_min=getattr(args, "lr_min", 1e-6),
        curriculum_phase_epochs=getattr(args, "curriculum_phase_epochs", 0),
        curriculum_heavy_percentile=getattr(args, "curriculum_heavy_percentile", 50.0),
        spectral_alignment_weight=getattr(args, "spectral_alignment_weight", 0.0),
        spectral_alignment_k=getattr(args, "spectral_alignment_k", 5),
        use_edge_bias=getattr(args, "use_edge_bias", False) and not getattr(args, "no_edge_bias", False),
        edge_dropout=getattr(args, "edge_dropout", 0.0),
        gaussian_noise_std=getattr(args, "gaussian_noise_std", 0.0),
        mixup_prob=getattr(args, "mixup_prob", 0.5),
    )

    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    submission_path = Path(args.submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test = load_data(data_dir)
    y_mean_pop = y_train.mean(axis=0).astype(np.float32)  # for inference shrinkage (before subject_scale)
    shrinkage_eps = getattr(args, "shrinkage_eps", 0.0)
    if shrinkage_eps > 0:
        print(f"Shrinkage: eps={shrinkage_eps} (pred = (1-eps)*pred + eps*train_mean)")
    print(f"Device: {device}")
    print(f"Train LR: {x_train.shape} | Train HR: {y_train.shape} | Test LR: {x_test.shape}")
    print(f"Config: {asdict(cfg)}")
    if len(seeds) > 1:
        print(f"Ensemble mode: {len(seeds)} seeds {seeds}")

    # --- Per-subject scale normalisation (full-retrain mode) ---
    subj_scales_train: np.ndarray | None = None
    subj_scales_test:  np.ndarray | None = None
    if cfg.subject_scale:
        subj_scales_train = compute_subject_scales(x_train)
        subj_scales_test  = compute_subject_scales(x_test)
        x_train = apply_subject_scale(x_train, subj_scales_train)
        y_train = apply_subject_scale(y_train, subj_scales_train)  # same LR scale for HR
        x_test  = apply_subject_scale(x_test,  subj_scales_test)
        print(f"  Subject-scale enabled. Train scale: mean={subj_scales_train.mean():.4f}, "
              f"Test scale: mean={subj_scales_test.mean():.4f}")

    vectorizer = MatrixVectorizer()

    # --- Laplacian PE: precompute once for train and test ---
    if cfg.lap_pe_dim > 0:
        print(f"  Precomputing Laplacian PE (k={cfg.lap_pe_dim}) for {len(x_train)} train + {len(x_test)} test subjects ...")
        lap_pe_train = compute_lap_pe(x_train, cfg.n_lr, cfg.lap_pe_dim, vectorizer)
        lap_pe_test  = compute_lap_pe(x_test,  cfg.n_lr, cfg.lap_pe_dim, vectorizer)
        x_train = np.concatenate([x_train, lap_pe_train], axis=1)
        x_test  = np.concatenate([x_test,  lap_pe_test],  axis=1)
        print(f"  x_train/x_test shape with LapPE: {x_train.shape} / {x_test.shape}")
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

        # Compute HR mean for residual learning (from training split only, not val)
        y_mean_full = y_tr.mean(axis=0) if cfg.use_residual else None
        if y_mean_full is not None and i == 0:
            print(f"  Residual mode: subtracting per-edge HR mean (mean={y_mean_full.mean():.4f})")
        if (cfg.edge_dropout > 0 or cfg.gaussian_noise_std > 0) and i == 0:
            print(f"  Augmentation: edge_dropout={cfg.edge_dropout}, gaussian_noise_std={cfg.gaussian_noise_std}, mixup_prob={cfg.mixup_prob}")

        fold_train_start = time.perf_counter()
        full_model = train_full(
            cfg, device, vectorizer, x_tr, y_tr,
            max_epochs=max_epochs, x_va=x_va, y_va=y_va,
            patience=patience if (x_va is not None) else None,
            y_mean=y_mean_full,
        )
        total_train_seconds += time.perf_counter() - fold_train_start

        checkpoint_path = out_dir / "checkpoints" / (f"full_model_seed{seed}.pt" if len(seeds) > 1 else args.checkpoint_name)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": full_model.state_dict(), "config": asdict(cfg)}, checkpoint_path)

        preds = predict_vectors(full_model, x_test, device, vectorizer, cfg.batch_size, cfg.n_lr,
                                y_mean=y_mean_full, lap_pe_dim=cfg.lap_pe_dim)
        preds = np.clip(preds, a_min=0.0, a_max=None)

        # Post-hoc bias correction: subtract mean(pred-gt) on val from test preds (same space)
        if getattr(args, "bias_correct", False) and x_va is not None and y_va is not None:
            val_preds = predict_vectors(full_model, x_va, device, vectorizer, cfg.batch_size, cfg.n_lr,
                                       y_mean=y_mean_full, lap_pe_dim=cfg.lap_pe_dim)
            bias = compute_bias_correction(val_preds, y_va)
            preds = preds - bias[np.newaxis, :]
            preds = np.clip(preds, a_min=0.0, a_max=None)
            print(f"  Bias correction | mean(bias)={bias.mean():.6f}, std(bias)={bias.std():.6f}")

        # Un-scale test predictions back to original edge-weight units
        if cfg.subject_scale and subj_scales_test is not None:
            preds = preds * subj_scales_test

        # Post-hoc calibration: fit on val holdout, apply to test
        if cfg.calibrate and x_va is not None and y_va is not None:
            # Predict on val holdout (already scaled) to fit calibration
            val_preds = predict_vectors(full_model, x_va, device, vectorizer, cfg.batch_size, cfg.n_lr,
                                       y_mean=y_mean_full, lap_pe_dim=cfg.lap_pe_dim)
            val_gt = y_va
            if cfg.subject_scale and subj_scales_train is not None:
                # reconstruct val subject indices (train_test_split breaks index continuity)
                # val_gt is already in scaled space; un-scale for calibration in original units
                val_preds_cal = val_preds  # will un-scale below
                # We need the scales for the val subjects; since train_test_split shuffles,
                # we work in the scaled space (consistent model input/output basis)
                # Calibration in scaled space, un-scale after applying
                cal_slope_full, cal_intercept_full = fit_calibration(val_preds_cal, val_gt)
                cal_scales = subj_scales_test  # test scales for un-scaling after calibration
                # Apply calibration in scaled space then un-scale
                preds_scaled = preds / subj_scales_test  # re-scale test preds temporarily
                preds_scaled = apply_calibration(preds_scaled, cal_slope_full, cal_intercept_full)
                preds = preds_scaled * subj_scales_test   # un-scale again
            else:
                cal_slope_full, cal_intercept_full = fit_calibration(val_preds, val_gt)
                preds = apply_calibration(preds, cal_slope_full, cal_intercept_full)
            print(f"  Full-retrain calibration | slope={cal_slope_full:.6f}, intercept={cal_intercept_full:.6f}")

        preds_list.append(preds)

    preds = np.mean(preds_list, axis=0)
    if shrinkage_eps > 0:
        preds = (1 - shrinkage_eps) * preds + shrinkage_eps * y_mean_pop[np.newaxis, :]
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
    """Optuna Bayesian optimization on 3-fold CV. Supports v3r (DenseGCN) and v4 (DenseGAT). Resumable via SQLite."""
    if optuna is None:
        raise ImportError("Optuna is required for tune mode. Install with: pip install optuna")

    args = apply_preset(args)
    tune_v3r = args.preset in ("v3r", "v3r_eb", "v3r_eb_ffnn", "v3r_eb_ffnn_aug", "v3r_eb_ffnn_spec") or (args.model == "dense_gcn" and args.preset in ("v3r", "v3r_eb", "v3r_eb_ffnn", "v3r_eb_ffnn_aug", "v3r_eb_ffnn_spec", "v3"))
    if not tune_v3r and args.model != "dense_gat":
        print("Tune mode: use --preset v3r for DenseGCN or --preset v4 for DenseGAT.")
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

    def objective_gat(trial: "optuna.Trial") -> float:
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
            pearl_pe_dim=getattr(args, "pearl_pe_dim", 0),
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

    def objective_v3r(trial: "optuna.Trial") -> float:
        lr = trial.suggest_float("lr", 5e-4, 1.5e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.25, 0.45)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 192, 256])
        num_layers = trial.suggest_int("num_layers", 3, 4)
        mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.3)
        cfg = TrainConfig(
            model_name="dense_gcn",
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
            loss_name="l1",
            huber_beta=1.0,
            l1_weight=0.7,
            seed=args.seed,
            num_folds=args.num_folds,
            use_residual=True,
            mixup_alpha=mixup_alpha,
            num_heads=4,
            ffn_mult=4,
            num_decoder_heads=4,
            hr_refine_layers=1,
            edge_scale=0.2,
            lap_pe_dim=0,
            pearl_pe_dim=0,
            use_edge_bias=True,
        )
        fold_vals = []
        for fold_id, (tr_idx, va_idx) in enumerate(kf.split(x_train), start=1):
            y_mean_fold = y_train[tr_idx].mean(axis=0)
            _, best_val, _ = train_with_validation(
                cfg, device, vectorizer,
                x_train[tr_idx], y_train[tr_idx],
                x_train[va_idx], y_train[va_idx],
                fold_id,
                y_mean=y_mean_fold,
            )
            fold_vals.append(best_val)
        mean_val = float(np.mean(fold_vals))
        trial.set_user_attr("fold_vals", fold_vals)
        return mean_val

    objective = objective_v3r if tune_v3r else objective_gat
    study_name = "v3r_tune" if tune_v3r else "dense_gat_tune"

    if fresh and storage_path.exists():
        storage_path.unlink()
        print("Fresh run: cleared Optuna study.")

    study = optuna.create_study(
        direction="minimize",
        storage=storage_url,
        load_if_exists=not fresh,
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_mean_val = study.best_value
    if tune_v3r:
        best_config = {
            "model_name": "dense_gcn",
            "n_lr": N_LR,
            "n_hr": N_HR,
            "hidden_dim": best_params["hidden_dim"],
            "num_layers": best_params["num_layers"],
            "dropout": best_params["dropout"],
            "learning_rate": best_params["lr"],
            "use_residual": True,
            "mixup_alpha": best_params["mixup_alpha"],
        }
    else:
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
        max_epochs_full = 600 if tune_v3r else 400
        print(f"\n--full-retrain: running full retrain with best config (val_ratio={val_ratio}, patience={patience_full}, max_epochs={max_epochs_full}) -> {submission_path}")
        if tune_v3r:
            cfg = TrainConfig(
                model_name="dense_gcn",
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
                loss_name="l1",
                huber_beta=1.0,
                l1_weight=0.7,
                seed=args.seed,
                num_folds=args.num_folds,
                use_residual=True,
                mixup_alpha=best_params["mixup_alpha"],
                num_heads=4,
                ffn_mult=4,
                num_decoder_heads=4,
                hr_refine_layers=1,
                edge_scale=0.2,
                lap_pe_dim=0,
                pearl_pe_dim=0,
                use_edge_bias=True,
            )
        else:
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
                pearl_pe_dim=getattr(args, "pearl_pe_dim", 0),
            )
        x_train, y_train, x_test = load_data(data_dir)
        y_mean_pop = y_train.mean(axis=0).astype(np.float32)
        shrinkage_eps = getattr(args, "shrinkage_eps", 0.05)
        x_tr, x_va, y_tr, y_va = train_test_split(
            x_train, y_train, test_size=val_ratio, random_state=cfg.seed, shuffle=True
        )
        y_mean_full = y_tr.mean(axis=0) if cfg.use_residual else None
        full_model = train_full(
            cfg, device, vectorizer, x_tr, y_tr,
            max_epochs=max_epochs_full, x_va=x_va, y_va=y_va, patience=patience_full,
            y_mean=y_mean_full,
        )
        ckpt_path = out_dir / "checkpoints" / "full_model_best_tune.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": full_model.state_dict(), "config": asdict(cfg)}, ckpt_path)
        preds = predict_vectors(full_model, x_test, device, vectorizer, cfg.batch_size, cfg.n_lr, y_mean=y_mean_full)
        if shrinkage_eps > 0:
            preds = (1 - shrinkage_eps) * preds + shrinkage_eps * y_mean_pop[np.newaxis, :]
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
