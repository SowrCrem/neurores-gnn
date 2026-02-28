# GIN Implementation

> Dense GIN (Graph Isomorphism Network) for brain graph super-resolution. See [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md) for architecture context.

## Architecture Overview

GIN replaces GCN message passing with sum aggregation + MLP:

1. **LR encoder:** GIN blocks — `h' = MLP((1+ε)·h + A @ h) + h` (weighted aggregation)
2. **Upsample:** Linear 160→268 (same as DenseGCN)
3. **Decoder:** Bilinear `H @ P @ P^T @ H^T` + softplus + clamp [0,1]

Reference: Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019).

## CLI Usage

### 3-Fold CV

```bash
.venv/bin/python -m src.train_dense_gcn cv --model dense_gin --preset gin \
  --out-dir artifacts/dense_gin --fresh
```

### Full Retrain + Submission

```bash
.venv/bin/python -m src.train_dense_gcn full --model dense_gin --preset gin \
  --submission-path submission/dense_gin_submission.csv
```

## Preset `gin`

- `hidden_dim=192`, `num_layers=3`
- `dropout=0.35`, `lr=8e-4`, `loss=smoothl1`
- `patience=45`

## Outputs

- `artifacts/dense_gin/cv_summary.json` — 3-fold CV metrics
- `artifacts/dense_gin/resource_summary.json` — runtime, RAM
- `artifacts/dense_gin/full_retrain_summary.json` — full retrain summary
- `submission/dense_gin_submission.csv` — Kaggle submission (ID, Predicted)
