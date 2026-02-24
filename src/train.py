"""
Training loop for the LR -> HR brain graph super-resolution model.

3-fold cross-validation with per-fold checkpointing and full metric
evaluation at the end of each fold.

Usage:
    python -m src.train --config configs/base_model.yaml
"""

import argparse
import os
import json
import time

import numpy as np
import yaml
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from utils.dgl_compat import patch as _patch_dgl; _patch_dgl()
from dgl.dataloading import GraphDataLoader

from src.dataset import load_train_dataset
from models.generator import BrainGNNGenerator
from utils.matrix_vectorizer import MatrixVectorizer
from utils.metrics import evaluate_fold as compute_metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, checkpoint_dir: str, tag: str = "latest"):
    """Atomically save a training state dict."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{tag}.pt")
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(config, device):
    return BrainGNNGenerator(
        lr_nodes=config["data"]["lr_nodes"],
        hr_nodes=config["data"]["hr_nodes"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"].get("num_layers", 4),
        dropout=config["model"].get("dropout", 0.1),
    ).to(device)


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def train_fold(config, dataset, train_idx, val_idx, fold_id, device):
    """Train a single fold and return the best model + val loss."""
    print(f"\n{'='*60}")
    print(f"Fold {fold_id}")
    print(f"  Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")
    print(f"{'='*60}")

    bs = config["training"]["batch_size"]
    train_loader = GraphDataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=bs, shuffle=True, drop_last=False,
    )
    val_loader = GraphDataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=bs, shuffle=False, drop_last=False,
    )

    model = _build_model(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_epochs = config["training"]["epochs"]
    sched_name = config["training"].get("scheduler", "none").lower()
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        scheduler = None

    criterion = nn.L1Loss()
    log_interval = config["logging"].get("log_interval", 10)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(total_epochs):
        t0 = time.time()

        # --- Train --------------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        for graphs, hr_targets in train_loader:
            graphs, hr_targets = graphs.to(device), hr_targets.to(device)
            optimizer.zero_grad()
            preds = model(graphs)
            loss = criterion(preds, hr_targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_sum / len(train_loader)

        # --- Validate -----------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for graphs, hr_targets in val_loader:
                graphs, hr_targets = graphs.to(device), hr_targets.to(device)
                preds = model(graphs)
                val_loss_sum += criterion(preds, hr_targets).item()

        val_loss = val_loss_sum / len(val_loader)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            marker = " *" if improved else ""
            elapsed = time.time() - t0
            print(f"  Epoch [{epoch+1:>4}/{total_epochs}]  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"({elapsed:.1f}s){marker}")

    model.load_state_dict(best_state)
    print(f"  Best val MAE: {best_val_loss:.6f}")

    save_checkpoint(
        {"model": best_state, "fold": fold_id, "val_loss": best_val_loss, "config": config},
        config["logging"]["checkpoint_dir"],
        tag=f"fold_{fold_id}",
    )

    return model, best_val_loss


# ---------------------------------------------------------------------------
# Full-metric evaluation on a fold's validation split
# ---------------------------------------------------------------------------

def evaluate_on_val(model, dataset, val_idx, config, device):
    """Run model on the validation split and compute all 8 metrics."""
    loader = GraphDataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=config["training"]["batch_size"],
        shuffle=False, drop_last=False,
    )

    vectorizer = MatrixVectorizer()
    hr_nodes = config["data"]["hr_nodes"]

    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for graphs, hr_targets in loader:
            graphs = graphs.to(device)
            all_preds.append(model(graphs).cpu().numpy())
            all_targets.append(hr_targets.numpy())

    pred_vecs = np.concatenate(all_preds, axis=0)
    gt_vecs = np.concatenate(all_targets, axis=0)

    pred_mats = np.stack([vectorizer.anti_vectorize(v, hr_nodes) for v in pred_vecs])
    gt_mats = np.stack([vectorizer.anti_vectorize(v, hr_nodes) for v in gt_vecs])

    return compute_metrics(pred_mats, gt_mats, verbose=True)


# ---------------------------------------------------------------------------
# 3-fold cross-validation
# ---------------------------------------------------------------------------

def train_cv(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = load_train_dataset(
        data_dir=config["data"].get("data_dir", "data"),
        threshold=0.0,
    )

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    indices = np.arange(len(dataset))

    fold_results = []

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        model, val_loss = train_fold(
            config, dataset, train_idx, val_idx, fold_id, device,
        )
        print(f"\n  Computing full metrics for fold {fold_id} ...")
        metrics = evaluate_on_val(model, dataset, val_idx, config, device)
        fold_results.append({
            "fold": fold_id,
            "val_mae": val_loss,
            "metrics": metrics,
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: val MAE = {r['val_mae']:.6f}")
    mean_mae = np.mean([r["val_mae"] for r in fold_results])
    print(f"  Mean val MAE: {mean_mae:.6f}")

    ckpt_dir = config["logging"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    summary_path = os.path.join(ckpt_dir, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(fold_results, f, indent=2, default=str)
    print(f"\nCV summary saved to {summary_path}")

    return fold_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train brain graph super-resolution model (3-fold CV)",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    train_cv(config)


if __name__ == "__main__":
    main()
