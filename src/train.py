"""
Training loop for the LR -> HR brain graph super-resolution model.

Checkpoint-safe: every N epochs the full training state is written to
  checkpoints/latest.pt   <- overwritten each save (safe resume point)
  checkpoints/best.pt     <- only updated when val loss improves

On restart the script automatically detects and loads the latest checkpoint,
so training resumes from exactly the epoch it was interrupted on.

Usage:
    python src/train.py --config configs/base_model.yaml
    python src/train.py --config configs/base_model.yaml --resume          # explicit flag (auto-detected anyway)
    python src/train.py --config configs/base_model.yaml --resume --checkpoint checkpoints/epoch_050.pt
"""

import argparse
import os
import json
import time
import yaml
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader

from src.dataset import load_train_dataset


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, checkpoint_dir: str, tag: str = "latest"):
    """
    Atomically saves a training state dict.

    Writes to <tag>.tmp first then renames to avoid a corrupt file if the
    process is killed mid-write (safe after hibernation / power loss).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{tag}.pt")
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)          # atomic on POSIX


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler=None, device: torch.device = torch.device("cpu")) -> dict:
    """
    Loads a checkpoint into model/optimizer/scheduler in-place.

    Returns the metadata dict (epoch, best_loss, history).
    """
    print(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return {
        "start_epoch": ckpt["epoch"] + 1,
        "best_loss":   ckpt.get("best_loss", float("inf")),
        "history":     ckpt.get("history", []),
    }


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Returns the path of the most recent checkpoint or None if none exist."""
    latest = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest):
        return latest
    return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict, resume_path: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Data ----------------------------------------------------------------
    dataset = load_train_dataset(threshold=0.0)
    loader = GraphDataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    # -- Model ---------------------------------------------------------------
    # from models.generator import BrainGNNGenerator
    # model = BrainGNNGenerator(**config["model"]).to(device)
    raise NotImplementedError(
        "Instantiate your model in train() before running. "
        "Uncomment the BrainGNNGenerator lines above."
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # -- Scheduler -----------------------------------------------------------
    sched_name = config["training"].get("scheduler", "none").lower()
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"]
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        scheduler = None

    # -- Resume --------------------------------------------------------------
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    start_epoch = 0
    best_loss = float("inf")
    history = []                        # list of {"epoch", "loss", "lr"} dicts

    # Auto-detect latest checkpoint if no explicit path given
    if resume_path is None:
        resume_path = find_latest_checkpoint(checkpoint_dir)

    if resume_path is not None:
        meta = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch = meta["start_epoch"]
        best_loss   = meta["best_loss"]
        history     = meta["history"]
        print(f"Resuming at epoch {start_epoch}  (best loss so far: {best_loss:.6f})")
    else:
        print("Starting fresh training run.")

    # -- Loop ----------------------------------------------------------------
    criterion = nn.L1Loss()             # MAE — matches competition metric
    total_epochs = config["training"]["epochs"]
    log_interval = config["logging"].get("log_interval", 10)
    save_interval = config["logging"].get("save_interval", 10)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for graphs, hr_targets in loader:
            graphs = graphs.to(device)
            hr_targets = hr_targets.to(device)

            optimizer.zero_grad()
            preds = model(graphs)               # shape: (B, 35778)
            loss = criterion(preds, hr_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / len(loader)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "loss": avg_loss, "lr": current_lr})

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"Epoch [{epoch+1:>4}/{total_epochs}]  "
                  f"loss={avg_loss:.6f}  lr={current_lr:.2e}  "
                  f"({elapsed:.1f}s)")

        # -- Checkpointing ---------------------------------------------------
        state = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict() if scheduler else None,
            "best_loss":  best_loss,
            "history":    history,
            "config":     config,
        }

        if (epoch + 1) % save_interval == 0:
            save_checkpoint(state, checkpoint_dir, tag="latest")
            # Also keep a named snapshot every save_interval epochs
            save_checkpoint(state, checkpoint_dir, tag=f"epoch_{epoch+1:04d}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            state["best_loss"] = best_loss
            save_checkpoint(state, checkpoint_dir, tag="best")
            print(f"  → New best: {best_loss:.6f}  (saved best.pt)")

    # Final save
    save_checkpoint(state, checkpoint_dir, tag="latest")
    print(f"\nTraining complete. Best MAE: {best_loss:.6f}")
    _save_history(history, checkpoint_dir)


def _save_history(history: list, checkpoint_dir: str):
    path = os.path.join(checkpoint_dir, "history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Loss history written to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train brain graph super-resolution model")
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from latest checkpoint (auto-detected if omitted)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint path to resume from")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    resume_path = args.checkpoint if args.checkpoint else None
    train(config, resume_path=resume_path)


if __name__ == "__main__":
    main()
