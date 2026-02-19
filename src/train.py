"""
Training loop for the LR -> HR brain graph super-resolution model.

Usage:
    python src/train.py --config configs/base_model.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train brain graph super-resolution model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def train(config: dict):
    pass


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
