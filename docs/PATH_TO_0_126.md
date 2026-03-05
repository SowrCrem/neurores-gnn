# Path to 0.126: Closing the Gap from 0.1322

> **Current best:** 0.132152 (v3r + 8% shrinkage). **Target:** 0.126 (leaderboard top). **Gap:** ~0.0060 MAE.

---

## Leaderboard Context (as of 2025-03-02)

| Rank | Team | MAE |
|------|------|-----|
| 1 | ideal.csv | 0.000 (placeholder) |
| 2 | Dawg Moggers | **0.126118** |
| 3 | Los Pollos Hermanos | 0.129439 |
| 4 | Adjacency Dwags | 0.129464 |
| 5 | [Deleted] | 0.129664 |
| 6 | Hot Dawgs | 0.130471 |
| 7 | **Big Dawgs (us)** | **0.132152** |

---

## Prioritised Action Plan

### Tier 1: Quick Wins (1–2h each)

| Action | Command | Expected Δ |
|--------|---------|------------|
| **Output shrinkage** | 8% best (0.1322). Try 0.10; stop when MAE worsens. | 0–2% ✓ |
| **v3r + LapPE** | Tried: 0.1367 (worse than v3r). ✗ Skip. | — |
| **Geometric mean ensemble** | Tried: 0.1334 (worse than single). ✗ Skip. | — |
| **Post-hoc bias correction** | Subtract `mean(pred - gt)` on val from test preds | 0–2% |

### Tier 2: Medium Effort (4–8h)

| Action | Notes | Expected Δ |
|--------|-------|------------|
| **DenseSTP (Edge-MLP decoder)** | `--model dense_stp --preset stp` — escapes bilinear rank bottleneck | 2–5% (untried) |
| **DenseSTP + PEARL** | `--preset stp_pe` — learnable PE | +0–2% over stp |
| **Curriculum training** | Tried with bias: 0.1371 ✗. Skip. | — |
| **Optuna on v3r** | Tune LR, dropout, hidden_dim, mixup_alpha | 0–2% |

### Tier 3: Paradigm Shift (High Effort)

| Action | Notes | Expected Δ |
|--------|-------|------------|
| **Full STP-GSR / dual-graph** | Edges-as-nodes; 35,778-node dual graph. arXiv:2411.02525 | Likely path to 0.126 |
| **DEFEND** | Edge representation framework. arXiv:2411.06019 | Same paradigm |

---

## What We've Already Tried (and outcome)

| Tried | Result |
|-------|--------|
| v3r + 8% shrinkage | **0.1322** ✓ best |
| v3r + 5% shrinkage | 0.1323 |
| v3r (L1 + residual + mixup) | 0.1326 |
| v3r no mixup | 0.1327 |
| v3r godzilla (h=256, 4 layers, cosine) | 0.1336 |
| v3r geometric ensemble (3 seeds) | 0.1334 ✗ (worse than single) |
| v3r arithmetic ensemble | 0.1328 (≈ single) |
| v3r curriculum + bias | 0.1371 ✗ (worse than v3r) |
| v3r_pe (LapPE k=4) | 0.1367 ✗ (worse than v3r) |
| v3r_lrs (low-rank + sparse decoder) | 0.137–0.14 ✗ |
| v3sn (scale norm + calibration) | 0.139 ✗ |
| v3sn_nc (scale norm, no cal) | 0.1339 |
| DenseGIN v3r | 0.138–0.14 |

---

## Recommended Order

1. ~~**Output shrinkage**~~ — Done. 0.1322 (8% best). Try eps=0.10.
2. ~~**v3r_pe**~~ — 0.1367 ✗ (LapPE hurt). Skip.
3. ~~**Curriculum + bias**~~ — 0.1371 ✗. Skip.
4. **DenseSTP** — Edge-MLP decoder; await Kaggle result.
5. **stp_pe** — If dense_stp beats 0.1322, run stp_pe next.
6. **Shrinkage 0.10** — Try stronger shrinkage.
7. **Optuna v3r** — Tune lr, dropout, mixup_alpha.
8. **Full STP-GSR** — If still short of 0.126.

---

## Reproduction Commands

```bash
# 1. Output shrinkage — in run_full_retrain(), after preds = run_inference(...) and before np.clip:
#    eps = 0.05
#    preds = (1 - eps) * preds + eps * y_mean_full[np.newaxis, :]

# 2. v3r + LapPE
.venv/bin/python -m src.train_dense_gcn full --preset v3r_pe --max-epochs 600 \
  --submission-path submission/v3r_pe_submission.csv

# 3. DenseSTP (Edge-MLP decoder)
.venv/bin/python -m src.train_dense_gcn full --model dense_stp --preset stp --max-epochs 600 \
  --submission-path submission/dense_stp_submission.csv

# 4. DenseSTP + PEARL
.venv/bin/python -m src.train_dense_gcn full --model dense_stp --preset stp_pe --max-epochs 600 \
  --submission-path submission/dense_stp_pe_submission.csv
```

---

## References

- [STP-GSR](https://arxiv.org/abs/2411.02525) — Strongly topology-preserving GNNs; dual-graph paradigm.
- [RESEARCH_ANALYSIS_REPORT.md](RESEARCH_ANALYSIS_REPORT.md) — Full improvement table.
- [PAPER_ANALYSIS.md](PAPER_ANALYSIS.md) — STP-GSR, DEFEND, PEARL.
- [KAGGLE_SUBMISSIONS_GROUND_TRUTH.md](KAGGLE_SUBMISSIONS_GROUND_TRUTH.md) — Submission history.
