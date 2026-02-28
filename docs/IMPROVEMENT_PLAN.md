# Improvement Plan

> Focused, actionable plan based on Kaggle results. Highest ROI: DenseGCN v3 full-retrain ensemble.

---

## Prior Results Summary

| Model | Kaggle MAE | Tuning? | Full Retrain? |
|-------|------------|---------|---------------|
| **DenseGCN v3** | **0.176** | No | Yes |
| DenseGCN v3 ensemble | 0.196 | No | No (3-fold CV) |
| DenseGCN v1/v2 | 0.195–0.197 | No | No |
| SGC v2 | 0.261 | No | No |
| DenseGAT / Bi-SR | ~0.475 | No | Yes |

**Insight:** No submissions used Optuna tuning. Headroom exists without new architectures.

---

## Highest ROI: DenseGCN v3 Full-Retrain Ensemble

**Why:** Best model (0.176) has never been ensembled with full retrain. Prior v3 ensemble (0.196) used 3-fold CV models (each on 2/3 data). Spec allows full retrain + ensemble of models each trained on all 167.

**Command:**
```bash
.venv/bin/python -m src.train_dense_gcn full --preset v3 \
  --ensemble-seeds 42,43,44 \
  --submission-path submission/dense_gcn_v3_ensemble_full_submission.csv
```

**Expected:** 0.17–0.20 (may beat 0.176 via prediction diversity).

---

## Fallback: Manual DenseGCN v3 Variants

```bash
# Stronger regularization
.venv/bin/python -m src.train_dense_gcn full --preset v3 --dropout 0.4 \
  --submission-path submission/dense_gcn_v3_d04_submission.csv

# Larger capacity + reg
.venv/bin/python -m src.train_dense_gcn full --preset v3 --hidden-dim 256 --dropout 0.4 \
  --submission-path submission/dense_gcn_v3_h256_d04_submission.csv
```

---

## GCN + Cross-Attention (gcn_ca)

New model: GCN encoder + cross-attention upsampling. See [docs/GCN_CROSS_ATTN_PLAN.md](docs/GCN_CROSS_ATTN_PLAN.md).

```bash
.venv/bin/python -m src.train_dense_gcn full --model dense_gcn_ca --preset gcn_ca \
  --submission-path submission/dense_gcn_ca_submission.csv
```

Quick CV (15 epochs): val MAE ~0.44, PCC 0.05–0.16. No collapse; full retrain needed to assess Kaggle.

---

## DenseGAT v5 (Redesigned)

DenseGAT v4 collapsed (~0.47). v5 redesign: GCN-first (2 GCN + 1 GAT), single linear upsample, no hr_refine, conservative init. See [docs/DENSEGAT_REDESIGN.md](docs/DENSEGAT_REDESIGN.md).

```bash
.venv/bin/python -m src.train_dense_gcn full --model dense_gat --preset v5 \
  --submission-path submission/dense_gat_v5_submission.csv
```

---

## Next Steps (Lower Priority)

| Action | Effort | Expected MAE |
|--------|--------|---------------|
| DenseGCN Optuna tune | Code + 4–6h | 0.16–0.18 |
| GCN + Cross-Attention | Code + 2–4h | 0.18–0.22 or 0.47 |

**Recommendation:** Start with full-retrain ensemble. If it beats 0.176, submit. If not, try 1–2 manual variants. Defer Optuna and GCN+Cross-Attn until after.
