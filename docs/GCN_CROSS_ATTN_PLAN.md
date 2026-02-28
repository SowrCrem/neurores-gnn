# GCN + Cross-Attention Implementation Plan

> Elegant implementation optimized for 167 samples, MAE on hidden Kaggle test set, and [DESIGN_CONSTRAINTS.md](DESIGN_CONSTRAINTS.md).

---

## Design Principles

| Constraint | Application |
|------------|-------------|
| **Data scarcity (167)** | Small capacity (hidden=128), dropout 0.35, conservative init |
| **Graph structure** | GCN encoder (structure-aware); cross-attention for upsampling only |
| **GELU before decoder** | Avoid ReLU collapse |
| **Conservative decoder init** | Xavier gain 0.1 |
| **Output** | Softplus + clamp [0,1] per spec |
| **Loss** | SmoothL1 (MAE-aligned) |
| **Gradient clipping** | max_norm=5 for stability |
| **Full adjacency rows** | Use X_lr = A_lr (160-dim per node), not 2-dim handcrafted features |

---

## Architecture

```
Input: A_lr (B, 160, 160), X_lr = A_lr (adjacency rows as features)

1. Input projection: Linear(160, hidden) → LayerNorm → GELU → Dropout
2. GCN encoder: 3 × DenseGCNBlock(S, H)  [reuse from dense_gcn.py]
3. Cross-attention upsampling:
   - Q: learnable HR queries (268, hidden), expanded to (B, 268, hidden)
   - K, V: H_lr (B, 160, hidden)
   - HR = LayerNorm(CrossAttn(Q, K, V) + Q)   [residual]
4. Decoder: H_hr → LayerNorm → GELU → Bilinear H@P@P^T@H^T
5. Output: softplus(pred) → clamp [0, 1]
```

**Key differences from existing gcn-encoder-ca-decoder:**
- Full adjacency rows (160-dim) instead of 2-dim (degree, constant)
- DenseGCNBlock (residual, LayerNorm) instead of DenseGCNLayer
- GELU throughout
- Conservative decoder init (gain 0.1)
- Softplus + clamp [0,1]
- Integrated into train_dense_gcn pipeline

---

## Implementation Steps

### 1. Create `models/dense_gcn_ca.py`

```python
# New model: DenseGCNCrossAttnGenerator
# - Reuse DenseGCNBlock from dense_gcn
# - Input: Linear(160, hidden), LayerNorm, GELU, Dropout
# - 3 GCN blocks
# - HR queries: nn.Parameter(torch.randn(268, hidden) * 0.01)
# - nn.MultiheadAttention(embed_dim=hidden, num_heads=4, dropout=0.35, batch_first=True)
# - Residual + LayerNorm after cross-attn
# - Decoder: LayerNorm, GELU, edge_P (Xavier gain 0.1), bilinear, softplus, clamp
# - forward(A_lr, X_lr) -> pred_vec  [X_lr = A_lr in our pipeline]
```

### 2. Register in `src/train_dense_gcn.py`

- Add `"dense_gcn_ca"` to model choices
- Add `build_model` branch for `dense_gcn_ca`
- Add preset `gcn_ca` with: hidden=128, num_layers=3, dropout=0.35, lr=5e-4, loss=smoothl1, patience=45
- Add grad clipping for `dense_gcn_ca` (like Bi-SR)
- Add `--out-dir artifacts/dense_gcn_ca` when using preset

### 3. TrainConfig / preset params

| Param | Value | Rationale |
|-------|-------|-----------|
| hidden_dim | 128 | Small; 167 samples |
| num_layers | 3 | Match DenseGCN |
| dropout | 0.35 | Strong reg |
| lr | 5e-4 | Conservative |
| loss | smoothl1 | MAE-aligned |
| patience | 45 | Allow convergence |
| num_heads | 4 | Cross-attention |

### 4. Validation

- Quick CV with `--skip-graph-metrics` (5 epochs) to check no collapse
- If val MAE < 0.30, run full CV
- Full retrain → Kaggle submission

---

## File Changes Summary

| File | Change |
|------|--------|
| `models/dense_gcn_ca.py` | New file |
| `src/train_dense_gcn.py` | Add model, preset, build_model, grad_clip |
| `docs/GCN_CROSS_ATTN_PLAN.md` | This plan |

---

## Usage

```bash
# Quick sanity check
.venv/bin/python -m src.train_dense_gcn cv --model dense_gcn_ca --preset gcn_ca \
  --epochs 10 --patience 3 --skip-graph-metrics --out-dir artifacts/dense_gcn_ca_quick --fresh

# Full CV
.venv/bin/python -m src.train_dense_gcn cv --model dense_gcn_ca --preset gcn_ca \
  --out-dir artifacts/dense_gcn_ca --fresh

# Full retrain
.venv/bin/python -m src.train_dense_gcn full --model dense_gcn_ca --preset gcn_ca \
  --submission-path submission/dense_gcn_ca_submission.csv
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Attention collapse (like GAT) | Cross-attention only for upsampling; GCN does heavy lifting |
| Overfitting | hidden=128, dropout=0.35, conservative init |
| Instability | Grad clipping, GELU, smoothplus |
| Wrong input format | Use X_lr = A_lr (same as DenseGCN) |

---

## Success Criteria

- Val MAE < 0.26 (beat SGC)
- No collapse (PCC non-NaN, val loss decreases)
- Kaggle MAE target: < 0.22 (approach DenseGCN)
