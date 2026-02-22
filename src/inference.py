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
  Total rows = 112 subjects × 35778 features = 4,007,136

Usage:
    python src/inference.py --config configs/base_model.yaml --checkpoint <path>
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
import dgl
from dgl.dataloading import GraphDataLoader

from src.dataset import load_test_dataset
from utils.graph_utils import preprocess_data

HR_FEATURES = 268 * 267 // 2   # 35778


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and produce submission file")
    parser.add_argument("--config",     type=str, required=True,  help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True,  help="Path to model checkpoint")
    parser.add_argument("--output",     type=str, default="submission/submission.csv")
    parser.add_argument("--data_dir",   type=str, default="data")
    return parser.parse_args()


def run_inference(model: torch.nn.Module, dataset, device: torch.device) -> np.ndarray:
    """
    Runs the model over the test dataset and returns all predictions stacked
    into a (N_subjects, HR_FEATURES) numpy array.
    """
    model.eval()
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            graph = batch.to(device) if isinstance(batch, dgl.DGLGraph) else batch[0].to(device)
            pred = model(graph)   # expected shape: (1, 35778) or (35778,)
            all_preds.append(pred.cpu().numpy().reshape(-1))

    return np.stack(all_preds, axis=0)   # (N_subjects, 35778)


def predictions_to_submission(preds: np.ndarray, output_path: str):
    """
    Converts a (N_subjects, HR_FEATURES) prediction array to the competition
    submission CSV format.

    Args:
        preds:       Array of shape (N_subjects, 35778).
        output_path: Path to write the output CSV.
    """
    # Clip negatives as per competition preprocessing requirement
    preds = np.clip(preds, a_min=0.0, a_max=None)

    n_subjects, n_features = preds.shape
    ids = np.arange(1, n_subjects * n_features + 1)
    flat_preds = preds.reshape(-1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission = pd.DataFrame({"ID": ids, "Predicted": flat_preds})
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}  ({len(submission):,} rows)")


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    test_dataset = load_test_dataset(data_dir=args.data_dir)

    # Load model
    # from models.generator import BrainGNNGenerator
    # model = BrainGNNGenerator(**config["model"]).to(device)
    # model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    raise NotImplementedError("Instantiate your model here before running inference.")

    preds = run_inference(model, test_dataset, device)
    predictions_to_submission(preds, args.output)


if __name__ == "__main__":
    main()
