"""
DGL Dataset class for loading brain connectivity graphs.

Data layout (vectorised form, competition spec):
  - lr_train.csv : (167, 12720)  -- LR training subjects
  - hr_train.csv : (167, 35778)  -- HR training subjects (targets)
  - lr_test.csv  : (112, 12720)  -- LR test subjects (no HR labels)

Pipeline per sample:
  1. Load vectorised row from CSV.
  2. Preprocess: replace NaN/negatives with 0.
  3. Anti-vectorize: 12720 -> 160x160 adjacency matrix (LR)
                     35778 -> 268x268 adjacency matrix (HR)
  4. Build a DGL graph from the LR adjacency matrix.
  5. Return (lr_graph, hr_vector) for training or (lr_graph,) for inference.
"""

import os
import pandas as pd
import numpy as np
import torch

from utils.dgl_compat import patch as _patch_dgl; _patch_dgl()
import dgl
from dgl.data import DGLDataset

from utils.matrix_vectorizer import MatrixVectorizer
from utils.graph_utils import preprocess_data, adj_to_dgl_graph

LR_NODES = 160
HR_NODES = 268
LR_FEATURES = LR_NODES * (LR_NODES - 1) // 2   # 12720
HR_FEATURES = HR_NODES * (HR_NODES - 1) // 2   # 35778


class BrainGraphDataset(DGLDataset):
    """
    Dataset for the DGL 2026 Brain Graph Super-Resolution Challenge.

    Args:
        lr_path:   Path to the LR CSV file (vectorised, shape N x 12720).
        hr_path:   Path to the HR CSV file (vectorised, shape N x 35778).
                   Pass None for the test set (no labels available).
        threshold: Edge weight threshold when building DGL graphs (default 0).
        name:      Dataset name passed to DGLDataset.
    """

    def __init__(self, lr_path: str, hr_path: str | None = None,
                 threshold: float = 0.0, name: str = "brain_graph"):
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.threshold = threshold
        self.lr_graphs: list[dgl.DGLGraph] = []
        self.hr_vectors: list[torch.Tensor] = []   # shape (35778,) per sample
        super().__init__(name=name)

    # ------------------------------------------------------------------
    # DGLDataset interface
    # ------------------------------------------------------------------

    def process(self):
        vectorizer = MatrixVectorizer()

        # Load and preprocess LR data
        lr_df = pd.read_csv(self.lr_path, header=0)
        lr_data = preprocess_data(lr_df.values.astype(np.float32))   # (N, 12720)

        # Load and preprocess HR data (training only)
        if self.hr_path is not None:
            hr_df = pd.read_csv(self.hr_path, header=0)
            hr_data = preprocess_data(hr_df.values.astype(np.float32))  # (N, 35778)
        else:
            hr_data = None

        for i in range(len(lr_data)):
            # Anti-vectorize LR row -> 160x160 adjacency matrix
            lr_adj = vectorizer.anti_vectorize(lr_data[i], LR_NODES)

            # Build DGL graph from LR adjacency matrix
            g = adj_to_dgl_graph(lr_adj, threshold=self.threshold)
            self.lr_graphs.append(g)

            # Store HR vector as-is (vectorised form for loss computation)
            if hr_data is not None:
                self.hr_vectors.append(torch.tensor(hr_data[i], dtype=torch.float32))

    def has_cache(self) -> bool:
        return False

    def __getitem__(self, idx: int):
        if self.hr_vectors:
            return self.lr_graphs[idx], self.hr_vectors[idx]
        return self.lr_graphs[idx]

    def __len__(self) -> int:
        return len(self.lr_graphs)


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_train_dataset(data_dir: str = "data", threshold: float = 0.0) -> BrainGraphDataset:
    return BrainGraphDataset(
        lr_path=os.path.join(data_dir, "lr_train.csv"),
        hr_path=os.path.join(data_dir, "hr_train.csv"),
        threshold=threshold,
        name="brain_graph_train",
    )


def load_test_dataset(data_dir: str = "data", threshold: float = 0.0) -> BrainGraphDataset:
    return BrainGraphDataset(
        lr_path=os.path.join(data_dir, "lr_test.csv"),
        hr_path=None,
        threshold=threshold,
        name="brain_graph_test",
    )
