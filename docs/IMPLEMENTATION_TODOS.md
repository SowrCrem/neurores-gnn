# Implementation TODOs

> Planned work to close the gap from 0.1322 → 0.126.  
> See [PATH_TO_0_126.md](PATH_TO_0_126.md) for prioritisation.

---

## Done (2025-03-03)

- [x] **Output shrinkage** — 8% best (0.1322). Try 0.10; stop when MAE worsens.
- [x] **Geometric mean ensemble** — `scripts/postprocess_submission.py geom-ensemble` → `v3r_geom_ensemble.csv` (seeds 42, 43, 44)
- [x] **v3r_pe** — 0.1367 ✗ (LapPE hurt vs v3r 0.1326)
- [x] **dense_stp** — Submitted; await Kaggle result.
- [x] **Curriculum training** — `--curriculum-phase-epochs N --curriculum-heavy-percentile 50`. Phase 1 trains on heavy edges only.
- [x] **Post-hoc bias correction** — Tried with curriculum: 0.1371 ✗. Skip.

---

## Pending

### 1. stp_pe (DenseSTP + PEARL)

**When:** After dense_stp results. If dense_stp beats 0.1326, run stp_pe next.

**Command:**
```bash
.venv/bin/python -m src.train_dense_gcn full --model dense_stp --preset stp_pe --max-epochs 600 \
  --submission-path submission/dense_stp_pe_submission.csv
```

**Effort:** 0 (preset exists). **Expected:** +0–2% over stp if PE helps.

---

### 2. Optuna v3r tuning

**Status:** Implemented. `tune --preset v3r` runs Bayesian optimization over lr, dropout, hidden_dim, num_layers, mixup_alpha.

**Command:**
```bash
# Tune only (30 trials, ~4–6h)
.venv/bin/python -m src.train_dense_gcn tune --preset v3r \
  --out-dir artifacts/v3r_tune \
  --out-config artifacts/v3r_tune/best_config.json \
  --n-trials 30

# Tune + full retrain with best config and save submission
.venv/bin/python -m src.train_dense_gcn tune --preset v3r \
  --out-dir artifacts/v3r_tune \
  --out-config artifacts/v3r_tune/best_config.json \
  --n-trials 30 --full-retrain \
  --submission-path submission/v3r_optuna_submission.csv
```

**Search space:** lr (5e-4–1.5e-3), dropout (0.25–0.45), hidden_dim (128/192/256), num_layers (3–4), mixup_alpha (0.1–0.3).

**Expected:** 0–2% MAE reduction. **Time:** ~4–6h for 30 trials.

---

### 3. Full STP-GSR (dual-graph)

**What:** Edges-as-nodes; 35,778-node dual graph. arXiv:2411.02525.

**Effort:** High (new architecture, sparse ops). **Expected:** Likely path to 0.126.

**Defer** until stp_pe results. Curriculum+bias tried (0.1371 ✗).

---

---

## CV dry-run (few epochs)

Quick sanity check for curriculum + bias (or any preset) without full training:

```bash
# Dry run: 5 epochs, skip graph metrics, curriculum phase 1 for 3 epochs
.venv/bin/python -m src.train_dense_gcn cv --preset v3r --epochs 5 --patience 3 \
  --curriculum-phase-epochs 3 --curriculum-heavy-percentile 50 \
  --skip-graph-metrics --fresh \
  --out-dir artifacts/cv_dry_curriculum
```

~~Curriculum + bias~~ — Tried: 0.1371 ✗. Skip.

---

## Quick reference: postprocess commands

```bash
# Shrinkage (eps=0.08 best; or 0.05, 0.10)
.venv/bin/python scripts/postprocess_submission.py shrinkage \
  --input submission/v3r_submission.csv \
  --output submission/v3r_shrinkage_008.csv --eps 0.08

# Geometric mean ensemble (seeds 42, 43, 44)
.venv/bin/python scripts/postprocess_submission.py geom-ensemble \
  --inputs submission/v3r_submission.csv submission/v3r_seed43.csv submission/v3r_seed44.csv \
  --output submission/v3r_geom_ensemble.csv
```

---

## Generated files (this session)

| File | Description |
|------|-------------|
| `submission/v3r_shrinkage_008.csv` | v3r + 8% shrinkage (best) |
| `submission/v3r_geom_ensemble.csv` | Geometric mean of v3r (seed 42), v3r_seed43, v3r_seed44 |
