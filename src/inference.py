"""
Inference script: generates HR brain graph predictions from the test set
and writes a submission CSV to submission/.

Submission format (per competition sample_submission.csv):
  ID,Predicted
  1,<value>
  2,<value>
  ...

IDs are assigned sequentially by flattening all subject predictions:
  ID = subject_idx * HR_FEATURES + feature_idx + 1
  Total rows = 112 subjects x 35778 features = 4,007,136

Usage:
    python -m src.inference --config configs/base_model.yaml --checkpoint checkpoints/fold_1.pt
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
import torch

from utils.dgl_compat import patch as _patch_dgl; _patch_dgl()
import dgl
from dgl.dataloading import GraphDataLoader

from src.dataset import load_test_dataset
from models.generator import BrainGNNGenerator

HR_FEATURES = 268 * 267 // 2  # 35778


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and produce submission file")
    parser.add_argument("--config",     type=str, required=True,  help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True,  help="Path to model checkpoint")
    parser.add_argument("--output",     type=str, default="submission/submission.csv")
    return parser.parse_args()


def run_inference(model: torch.nn.Module, dataset, device: torch.device,
                  batch_size: int = 16) -> np.ndarray:
    """Return predictions stacked into a (N_subjects, HR_FEATURES) array."""
    model.eval()
    loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            graph = batch.to(device) if isinstance(batch, dgl.DGLGraph) else batch[0].to(device)
            pred = model(graph)
            all_preds.append(pred.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def predictions_to_submission(preds: np.ndarray, output_path: str):
    """Convert (N_subjects, HR_FEATURES) predictions to competition CSV."""
    preds = np.clip(preds, a_min=0.0, a_max=None)

    n_subjects, n_features = preds.shape
    ids = np.arange(1, n_subjects * n_features + 1)
    flat_preds = preds.reshape(-1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    submission = pd.DataFrame({"ID": ids, "Predicted": flat_preds})
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}  ({len(submission):,} rows)")


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = load_test_dataset(
        data_dir=config["data"].get("data_dir", "data"),
    )

    model = BrainGNNGenerator(
        lr_nodes=config["data"]["lr_nodes"],
        hr_nodes=config["data"]["hr_nodes"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"].get("num_layers", 4),
        dropout=config["model"].get("dropout", 0.1),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    preds = run_inference(
        model, test_dataset, device,
        batch_size=config["training"].get("batch_size", 16),
    )
    predictions_to_submission(preds, args.output)


if __name__ == "__main__":
    main()
