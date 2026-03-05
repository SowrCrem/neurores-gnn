# Kaggle Submissions Ground Truth

> Canonical record of all submissions to the DGL 2026 Brain Graph Super-Resolution Challenge.  
> Primary metric: **MAE** (lower is better).  
> Last updated: 2025-03-04.

---

## Leaderboard Context

| Rank | Team | MAE |
|------|------|-----|
| 2 | Dawg Moggers | **0.126118** |
| 3 | Los Pollos Hermanos | 0.129439 |
| 4 | Adjacency Dwags | 0.129464 |
| 5 | [Deleted] | 0.129664 |
| 6 | Hot Dawgs | 0.130471 |
| 7 | **Big Dawgs (us)** | **0.132152** (v3r + 8% shrinkage) |

**Path to 0.126:** [PATH_TO_0_126.md](PATH_TO_0_126.md)

---

## 1. Master Table (All Submissions)

Sorted by Kaggle MAE ascending. `—` = not recorded.

| Rank | File | Kaggle MAE | Model | Preset | Epochs | Seed | Key Hyperparams | Val Split? | Local MAE (CV) |
|------|------|------------|-------|--------|--------|------|-----------------|------------|----------------|
| 1 | v3r_shrinkage_008.csv | **0.132152** | DenseGCN | v3r | 600 | 42 | L1 + residual + mixup + 8% shrinkage | Yes | — |
| 2 | v3r_shrinkage_005.csv | **0.132301** | DenseGCN | v3r | 600 | 42 | L1 + residual + mixup + 5% shrinkage | Yes | — |
| 3 | v3r_submission.csv | **0.132567** | DenseGCN | v3r | 600 | 42 | L1 loss, residual, mixup α=0.2 | Yes | 0.142 ± 0.005 |
| 4 | v3r_nomixup.csv | 0.132685 | DenseGCN | v3r | 600 | 42 | L1, residual, no mixup | Yes | — |
| 5 | v3r_ensemble_val.csv | 0.132789 | DenseGCN | v3r | — | — | Ensemble w/ val set | Yes | — |
| 6 | v3r_geom_ensemble.csv | 0.133376 | DenseGCN | v3r | — | 42,43,44 | Geometric mean of 3 seeds | Yes | — |
| 7 | v3r_godzilla.csv | 0.133625 | DenseGCN | v3r | 600 | 42 | hidden=256, 4 layers, cosine LR | Yes | — |
| 8 | v3r_ensemble.csv | 0.133705 | DenseGCN | v3r | — | — | Ensemble (multi-seed) | Yes | — |
| 9 | v3sn_nc_submission.csv | 0.133862 | DenseGCN | v3sn | — | — | Scale norm, NO calibration | Yes | — |
| 10 | v3r_cos_full.csv | 0.133998 | DenseGCN | v3r_cos | 600 | 42 | Cosine annealing LR | Yes | — |
| 11 | v3r_seed43.csv | 0.134561 | DenseGCN | v3r | 600 | 43 | Same as v3r, seed 43 | Yes | — |
| 12 | v3r_seed44.csv | 0.134730 | DenseGCN | v3r | 600 | 44 | Same as v3r, seed 44 | Yes | — |
| 13 | v3r_submission_800.csv | 0.135935 | DenseGCN | v3r | 800 | 42 | Same as v3r, 800 eps | Yes | — |
| 14 | v3r_pe_submission.csv | 0.136743 | DenseGCN | v3r_pe | 600 | 42 | v3r + LapPE k=4 | Yes | — |
| 15 | v3r_curriculum_bias.csv | 0.137122 | DenseGCN | v3r | 600 | 42 | Curriculum (100ep heavy) + bias correction | Yes | — |
| 16 | v3r_lrs_100ep.csv | 0.137181 | DenseGCN | v3r_lrs | 100 | 42 | Low-rank/sparse decoder | Yes | 0.139 ± 0.004 |
| 17 | gin_v3r_175ep_full.csv | 0.138266 | DenseGIN | gin_v3r | 175 | 42 | No val split | No | — |
| 18 | gin_v3r_submission.csv | 0.138503 | DenseGIN | gin_v3r | 200 | 42 | GIN + v3r training | Yes | 0.138 ± 0.004 |
| 19 | v3sn_submission.csv | 0.139125 | DenseGCN | v3sn | — | — | Scale norm + post-hoc calibration | Yes | 0.141 ± 0.005 |
| 20 | gin_v3r_novalsplit.csv | 0.139537 | DenseGIN | gin_v3r | 600 | 42 | No val split (full 167 train) | No | — |
| 21 | v3r_lrs_submission.csv | 0.140277 | DenseGCN | v3r_lrs | 300 | 42 | Low-rank/sparse decoder, no val split | No | — |
| — | dense_gcn_v3_full_retrain | 0.176083 | DenseGCN | v3 | — | 42 | SmoothL1, no residual/mixup | Yes | 0.242 ± 0.003 |
| — | dense_gcn_v3_ensemble_full | 0.176744 | DenseGCN | v3 | — | 42,43,44 | 3-seed ensemble | Yes | — |
| — | dense_gcn_v3_d04 | 0.178061 | DenseGCN | v3 | — | 42 | dropout=0.4 | Yes | — |
| — | dense_gcn_v3_h256_d04 | 0.178131 | DenseGCN | v3 | — | 42 | h=256, dropout=0.4 | Yes | — |
| — | dense_gcn_submission_1 | 0.195672 | DenseGCN | v1 | — | — | Legacy v1 | No | — |
| — | dense_gcn_v3_ensemble | 0.196269 | DenseGCN | v3 | — | — | 3-fold CV ensemble (no full retrain) | No | — |
| — | dense_gcn_v2 | 0.197395 | DenseGCN | v2 | — | — | MSE loss | No | — |
| — | sgc_submission_1 | 0.261464 | SGC | — | — | — | K-hop, 2-dim features | No | — |
| — | sgc_submission | 0.375375 | SGC | — | — | — | SGC v1 | No | — |
| — | dense_gat_v4_lr5e4 | 0.474895 | DenseGAT | v4 | — | — | lr=5e-4 | Yes | — |
| — | dense_gat_v4 | 0.475942 | DenseGAT | v4 | — | — | — | Yes | 0.440 ± 0.002 |
| — | dense_gat_v5 | 0.479093 | DenseGAT | v5 | — | — | GCN-first redesign | Yes | — |
| — | dense_bisr_v2 | 0.476535 | Dense Bi-SR | bisr_v2 | — | — | — | Yes | 0.441 ± 0.002 |
| — | dense_bisr_prelim | 0.476706 | Dense Bi-SR | bisr | — | — | — | Yes | — |
| — | dense_gcn_ca | 0.476486 | GCN+CrossAttn | gcn_ca | — | — | — | Yes | 0.440 ± 0.002 |
| — | dense_gcn_gps | 0.479213 | GraphGPS | gps | — | — | — | Yes | 0.442 ± 0.002 |
| — | dense_gin | 0.481457 | Dense GIN | gin | — | — | Unnormalized adj | Yes | 0.458 ± 0.002 |

---

## 2. Preset Reference (from `train_dense_gcn.py`)

| Preset | Model | Hidden | Layers | Dropout | LR | Loss | Residual | Mixup | Other |
|--------|-------|--------|--------|---------|-----|------|----------|-------|-------|
| v3r | dense_gcn | 192 | 3 | 0.35 | 8e-4 | L1 | ✓ | α=0.2 | — |
| v3r_pe | dense_gcn | 192 | 3 | 0.35 | 8e-4 | L1 | ✓ | α=0.2 | LapPE k=4 |
| v3sn | dense_gcn | 192 | 3 | 0.35 | 8e-4 | L1 | ✓ | α=0.2 | subject_scale, calibrate |
| v3r_lrs | dense_gcn_lrs | 192 | 3 | 0.35 | 8e-4 | L1 | ✓ | α=0.2 | Low-rank + sparse decoder |
| v3r_cos | dense_gcn | 192 | 3 | 0.35 | 8e-4 | L1 | ✓ | α=0.2 | Cosine LR schedule |
| gin_v3r | dense_gin | 192 | 3 | 0.35 | 8e-4 | L1 | ✓ | α=0.2 | Normalized adj |
| v3 | dense_gcn | 192 | 3 | 0.35 | 8e-4 | SmoothL1 | ✗ | ✗ | — |
| v2 | dense_gcn | 128 | 3 | 0.5 | 5e-4 | MSE | ✗ | ✗ | — |

---

## 3. Key Findings (v3r Era)

| Finding | Implication |
|---------|-------------|
| **Shrinkage 8% is now best** | 0.132152 (8%) vs 0.132301 (5%) vs 0.1326 (raw). Try 0.10 next. |
| **Curriculum + bias hurt** | v3r_curriculum_bias 0.1371 vs v3r 0.1326 — ~3.4% worse. Skip. |
| **LapPE hurts v3r** | v3r_pe 0.1367 vs v3r 0.1326 — ~3% worse. Skip LapPE on DenseGCN. |
| **Geometric ensemble hurts** | v3r_geom_ensemble 0.1334 vs v3r 0.1326 — do not use. |
| **v3r beats v3 by ~0.04 MAE** | Residual learning + L1 + mixup critical (0.133 vs 0.176) |
| **Ensemble ≈ single** | v3r_ensemble 0.1337 vs v3r 0.1326 — single seed competitive |
| **DenseGIN v3r reaches 0.138** | GIN with v3r training breaks ~0.48 barrier |
| **v3sn calibration hurts** | v3sn_nc (no cal) 0.1339 vs v3sn (cal) 0.1391 |
| **Low-rank decoder (v3r_lrs)** | 0.137–0.14 — trails standard bilinear |

---

## 4. Reproduction Commands

```bash
# Best single (v3r, 600 epochs)
.venv/bin/python -m src.train_dense_gcn full --preset v3r --max-epochs 600 --submission-path submission/v3r_submission.csv

# v3r no mixup
.venv/bin/python -m src.train_dense_gcn full --preset v3r --mixup-alpha 0 --max-epochs 600 --submission-path submission/v3r_nomixup.csv

# v3r godzilla (hidden=256, 4 layers, cosine)
.venv/bin/python -m src.train_dense_gcn full --preset v3r --hidden-dim 256 --num-layers 4 --lr-schedule cosine --max-epochs 600 --submission-path submission/v3r_godzilla.csv

# v3r cosine LR
.venv/bin/python -m src.train_dense_gcn full --preset v3r_cos --max-epochs 600 --submission-path submission/v3r_cos_full.csv

# DenseGIN v3r
.venv/bin/python -m src.train_dense_gcn full --preset gin_v3r --model dense_gin --max-epochs 200 --submission-path submission/gin_v3r_submission.csv

# v3r low-rank/sparse decoder
.venv/bin/python -m src.train_dense_gcn full --preset v3r_lrs --model dense_gcn_lrs --max-epochs 300 --val-ratio 0 --submission-path submission/v3r_lrs_submission.csv
```

---

## 5. Local MAE Source (from artifacts/)

Local MAE = 3-fold CV mean ± std from `artifacts/<out_dir>/cv_summary.json` → `mean_metrics["MAE"]` ± `std_metrics["MAE"]`.

| Artifact dir | Maps to |
|--------------|---------|
| v3r_cv_full | v3r |
| v3r_pe_cv | v3r_pe |
| v3r_lrs_cv | v3r_lrs |
| gin_v3r_cv | gin_v3r |
| v3sn_cv | v3sn |
| dense_gcn_v3 | v3 |
| dense_gat_v4 | dense_gat_v4 |
| dense_gin_quick | dense_gin (3 ep quick) |
| dense_gcn_gps_quick | gps |
| dense_gcn_ca_quick | gcn_ca |
| dense_bisr_v2_quick | bisr_v2 |

---

## 6. Changelog

| Date | Change |
|------|--------|
| 2025-03-02 | Initial ground truth: 16 v3r-era + 17 legacy submissions |
| 2025-03-02 | Local MAE from artifacts/*/cv_summary.json |
| 2025-03-03 | Post-process: v3r_shrinkage_005.csv (0.1323 ✓), v3r_geom_ensemble.csv (0.1334 ✗) |
| 2025-03-04 | v3r_pe_submission.csv (0.1367 ✗ — LapPE hurt vs v3r) |
| 2025-03-04 | v3r_shrinkage_008.csv (0.1322 ✓ new best), v3r_curriculum_bias.csv (0.1371 ✗) |
