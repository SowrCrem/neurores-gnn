# Bi-SR Implementation

> Bi-SR (Bipartite Super-Resolution) for brain graph super-resolution. See [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md) for architecture details.

## Architecture Overview

Bi-SR replaces the linear upsample (160→268) in DenseGCN with bipartite message passing:

1. **LR encoder:** GCN blocks on LR graph (reuses `DenseGCNBlock`)
2. **Bipartite upsampling:** Bipartite graph connects LR (160) and HR (268) nodes; each HR node aggregates from all LR nodes via `BipartiteGNNBlock` (GELU)
3. **Decoder:** Bilinear `H @ P @ P^T @ H^T` + softplus

## Quick Validation (Before Full Tuning)

### Kaggle Benchmarks

| Model | Kaggle MAE |
|-------|------------|
| DenseGCN v3 (full retrain) | **0.176** |
| DenseGCN v1 / v2 / v3 ensemble | 0.195–0.197 |
| SGC v2 | 0.261 |
| DenseGAT v4 | 0.476 (collapsed) |

**Bi-SR must beat:** SGC v2 (0.261)  
**Bi-SR should beat:** DenseGCN v2 (0.197)  
**Target:** DenseGCN v3 (0.176)

### Phase 1: Sanity Check (~5–10 min)

```bash
.venv/bin/python -m src.train_dense_gcn cv --model dense_bisr --preset bisr \
  --epochs 5 --patience 2 --out-dir artifacts/dense_bisr_quick --fresh
```

**Pass:** Loss decreases; no PCC=NaN; no constant outputs.

### Phase 2: CV + Full Retrain + Submission (~45–60 min)

```bash
# Step 1: CV
.venv/bin/python -m src.train_dense_gcn cv --model dense_bisr --preset bisr \
  --epochs 80 --patience 20 --out-dir artifacts/dense_bisr_prelim --fresh

# Step 2–3: Full retrain + submission
.venv/bin/python -m src.train_dense_gcn full --model dense_bisr --preset bisr \
  --max-epochs 80 --val-ratio 0.15 --patience 20 \
  --submission-path submission/dense_bisr_prelim_submission.csv
```

**Step 4:** Upload `submission/dense_bisr_prelim_submission.csv` to Kaggle; note the public MAE.

### Phase 2: Decision (After Kaggle Score)

| Kaggle MAE | Action |
|------------|--------|
| < 0.197 | Proceed to full tune |
| 0.197 – 0.261 | Competitive; tune to reach v3 level |
| > 0.261 | Fix architecture before tuning |
| > 0.4 | Likely collapse; debug first |

---

## CLI Usage

### 3-Fold CV

```bash
.venv/bin/python -m src.train_dense_gcn cv --model dense_bisr --preset bisr \
  --out-dir artifacts/dense_bisr --fresh
```

### Full Retrain + Submission

```bash
.venv/bin/python -m src.train_dense_gcn full --model dense_bisr --preset bisr \
  --submission-path submission/dense_bisr_submission.csv
```

### Hyperparameter Tuning (Optuna)

Tuning support for Bi-SR is implemented when the Kaggle score justifies it (see plan Step 8).

```bash
.venv/bin/python -m src.train_dense_gcn tune --model dense_bisr --preset bisr \
  --out-dir artifacts/dense_bisr_tune \
  --out-config artifacts/dense_bisr_tune/best_config.json
```

---

## Preset `bisr`

- `hidden_dim=192`, `num_layers=3`, `bipartite_layers=1`
- `dropout=0.3`, `lr=8e-4`, `loss=smoothl1`
- `patience=50`

---

## Outputs

- `artifacts/dense_bisr_prelim/cv_summary.json` — 3-fold CV metrics
- `artifacts/dense_bisr_prelim/resource_summary.json` — runtime, RAM
- `artifacts/dense_bisr_prelim/full_retrain_summary.json` — full retrain summary
- `submission/dense_bisr_prelim_submission.csv` — Kaggle submission (ID, Predicted)
