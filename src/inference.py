"""
Inference script: generates HR brain graph predictions from the test set
and writes submission files to submission/.

Usage:
    python src/inference.py --config configs/base_model.yaml --checkpoint <path>
"""

import argparse
import yaml
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and produce submission file")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="submission/predictions.npy")
    return parser.parse_args()


def run_inference(config: dict, checkpoint_path: str, output_path: str):
    pass


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    run_inference(config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
