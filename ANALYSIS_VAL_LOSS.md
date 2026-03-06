# Analysis: Validation Loss Not Moving (DenseGAT v4)

## Observed Symptoms

From the terminal output:
- **Val loss identical** from epoch 1 to 50: `0.055850`, `0.060625`, `0.057941` (exact, not rounding)
- **Best val always at epoch 1** - model never improves on validation after first epoch
- **Train loss decreases**: 0.11 → ~0.06, so the model *is* learning on training data
- **PCC = NaN** - Pearson correlation undefined, indicating **zero or near-zero variance** in predictions (constant outputs)

## Root Causes

### 1. **Learning rate too high (5e-3)**
v4 preset uses `lr=5e-3` (10× typical 5e-4). This causes:
- Overshoot in the first few steps
- Model quickly lands in a basin where it outputs near-constant values
- Gradients may be large enough to destabilize; validation predictions collapse

### 2. **ReLU before decoder → output collapse**
```python
H = F.relu(H)  # zeros all negative values
HW = H @ self.edge_W
A_pred = HW @ H.transpose(1, 2)
```
- If encoder produces many negative values (common with LayerNorm + attention), ReLU zeros them
- Bilinear `H @ W @ H^T` with sparse/near-zero H → near-zero A_pred
- Model outputs collapse to constants → val loss constant, PCC=NaN

### 3. **dropout=0**
- Intended for train/eval parity, but removes regularization
- Model overfits to training set; for "unfamiliar" validation inputs, may default to constant output
- Train loss drops (memorization) while val stays flat

### 4. **No output scale / residual**
- Decoder has no inductive bias; must learn full mapping from scratch
- With unstable training, easy to collapse to mean prediction

## DenseGCN vs DenseGAT

| Aspect | DenseGCN | DenseGAT |
|--------|----------|----------|
| Encoder | GCN (S @ H @ W) | GAT (attention + edge bias) |
| Pre-decoder | ReLU in blocks only | Explicit `F.relu(H)` before decoder |
| Upsample | Single Linear(160→268) | Linear→GELU→Linear |
| Dropout | 0.3–0.5 | 0 (v4) |

DenseGCN uses ReLU in GCN blocks, so H is non-negative, but the path is more constrained. DenseGAT’s attention can produce larger-magnitude activations; ReLU then zeros many, causing collapse.

## Fixes Applied

1. **DenseGAT (critical)**: Replace `clamp(pred, 0)` with `softplus(pred)` - clamp cuts gradients when pre-clamp values are negative; model collapsed to all-zero outputs after 1 epoch. Softplus ensures non-negative output with full gradient flow.
2. **DenseGAT**: Parameterize decoder as `H @ P @ P^T @ H^T` (P instead of W) for better conditioning.
3. **DenseGAT**: GELU before decoder (smooth activation).
4. **Training**: Lower v4 LR from 5e-3 → 8e-4, add ReduceLROnPlateau scheduler.
5. **Training**: Restore small dropout (0.1) for regularization.
6. **DenseGCN**: Same softplus fix + smaller edge_W init for consistency.
