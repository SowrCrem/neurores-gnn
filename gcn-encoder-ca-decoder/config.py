# config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_lr: int = 160
    n_hr: int = 268
    d_model: int = 128
    gcn_layers: int = 3
    attn_heads: int = 4
    dropout: float = 0.1

@dataclass
class TrainConfig:
    random_state: int = 42
    folds: int = 3
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda"  # falls back to cpu if unavailable