# DenseGAT Redesign (v5)

> Redesign based on past failures, DenseGCN success, and [DESIGN_CONSTRAINTS.md](DESIGN_CONSTRAINTS.md).

---

## Past Failures (v4, ~0.47 MAE)

| Issue | Cause |
|-------|-------|
| Output collapse | Attention-dominated encoder; ReLU/GELU before decoder zeros activations |
| Overfitting | hr_refine = pure attention on HR nodes with no graph structure |
| Instability | Complex upsample (Linear→GELU→Linear); aggressive decoder init |
| High LR | 8e-4 or 5e-3 caused overshoot and collapse |

---

## Design Principles (from DESIGN_CONSTRAINTS)

- **GCN is simpler and more stable** with 167 samples
- **GAT with adjacency bias** — avoid pure attention; graph must dominate
- **Prefer GELU** over ReLU before decoder (avoids zeroing)
- **Conservative decoder init** (gain 0.1)
- **Output clamp [0, 1]** per spec
- **Parameter budget** well below 10k–20k effective; dropout 0.2–0.4

---

## v5 Architecture

```
Input (B, 160, 160) → Linear→GELU→Dropout
  → 2× DenseGCNBlock(S, H)     [GCN-first: stable]
  → 1× GraphAttentionBlock(H, A)  [edge_scale=0.7: graph dominates]
  → LayerNorm
  → Linear(160→268)             [single upsample, like DenseGCN]
  → LayerNorm → GELU
  → Bilinear H@P@P^T@H^T       [gain=0.1]
  → softplus → clamp [0,1]
```

**Key changes vs v4:**
- 2 GCN + 1 GAT (was 4 GAT)
- Single linear upsample (was Linear→GELU→Linear)
- No hr_refine
- Decoder init gain 0.1 (was 0.5)
- hidden=128, dropout=0.4, edge_scale=0.7
- ffn_mult=2 (smaller FFN in GAT block)

---

## Usage

```bash
# Quick CV (skip graph metrics)
.venv/bin/python -m src.train_dense_gcn cv --model dense_gat --preset v5 \
  --skip-graph-metrics --out-dir artifacts/dense_gat_v5 --fresh

# Full retrain
.venv/bin/python -m src.train_dense_gcn full --model dense_gat --preset v5 \
  --submission-path submission/dense_gat_v5_submission.csv
```

---

## Expected Behavior

- **Val loss:** Should decrease (no collapse)
- **Val MAE:** May still trail DenseGCN (0.17) — attention adds sensitivity
- **Kaggle:** Target < 0.26 (beat SGC); stretch < 0.20 (approach DenseGCN)
