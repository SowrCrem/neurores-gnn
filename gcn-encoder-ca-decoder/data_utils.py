# data_utils.py
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.matrix_vectorizer import MatrixVectorizer

def vec_to_adj(vec: torch.Tensor, n: int) -> torch.Tensor:
    """
    Convert upper-triangle vector (no diagonal) -> symmetric adjacency.

    vec: (B, E) where E = n*(n-1)/2
    returns: (B, n, n)
    """
    B = vec.shape[0]
    device = vec.device
    dtype = vec.dtype
    
    # Convert to numpy, reconstruct matrix, convert back to torch
    vec_np = vec.cpu().numpy()
    A_list = [MatrixVectorizer.anti_vectorize(vec_np[i], n, include_diagonal=False) for i in range(B)]
    A = torch.from_numpy(np.stack(A_list)).to(dtype).to(device)
    return A

def adj_to_vec(A: torch.Tensor) -> torch.Tensor:
    """
    Convert symmetric adjacency -> upper-triangle vector (no diagonal).
    A: (B, n, n)
    returns: (B, n*(n-1)/2)
    """
    B = A.shape[0]
    device = A.device
    dtype = A.dtype
    
    # Convert to numpy, vectorize, convert back to torch
    A_np = A.cpu().numpy()
    vec_list = [MatrixVectorizer.vectorize(A_np[i], include_diagonal=False) for i in range(B)]
    vec = torch.from_numpy(np.stack(vec_list)).to(dtype).to(device)
    return vec

def lr_node_features(A_lr: torch.Tensor) -> torch.Tensor:
    """
    Minimal node features from adjacency:
      - constant 1
      - weighted degree/strength (sum of weights)
    A_lr: (B, n_lr, n_lr)
    returns: (B, n_lr, 2)
    """
    deg = A_lr.sum(dim=-1, keepdim=True)  # (B, n, 1)
    ones = torch.ones_like(deg)
    return torch.cat([ones, deg], dim=-1)

def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(x).float().to(device)