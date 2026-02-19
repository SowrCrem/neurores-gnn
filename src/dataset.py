"""
DGL Dataset class for loading brain connectivity graphs.

Each sample is a pair of:
  - Low-Resolution (LR) adjacency matrix  -> input graph
  - High-Resolution (HR) adjacency matrix -> target graph
"""

import dgl
import torch
import numpy as np
from dgl.data import DGLDataset


class BrainGraphDataset(DGLDataset):
    def __init__(self, lr_path: str, hr_path: str | None = None, name: str = "brain_graph"):
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.lr_graphs = []
        self.hr_graphs = []
        super().__init__(name=name)

    def process(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.lr_graphs)
