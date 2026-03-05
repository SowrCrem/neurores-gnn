# Improvement Plan

> **Current best:** 0.1322 (v3r + 8% shrinkage). **Target:** 0.126.  
> **See [PATH_TO_0_126.md](PATH_TO_0_126.md) for the full action plan.**

---

## Prior Results Summary

| Model                    | Kaggle MAE | Notes                          |
| ------------------------ | ---------- | ------------------------------ |
| **v3r + shrinkage (8%)** | **0.1322** | New best. Try 10% next.       |
| v3r + shrinkage (5%)     | 0.1323     | —                             |
| DenseGCN v3r             | 0.1326     | L1 + residual + mixup, 600 eps |
| v3r curriculum + bias    | 0.1371     | ✗ Worse than v3r               |
| v3r_pe (LapPE)           | 0.1367     | ✗ Worse than v3r               |
| v3r geom ensemble        | 0.1334     | ✗ Worse than single            |
| DenseGCN v3r ensemble    | 0.1328     | ≈ single                       |
| DenseGIN v3r          | 0.138–0.14 | GIN + v3r training             |
| DenseGCN v3           | 0.176      | No residual/mixup              |
| SGC v2                | 0.261      | —                              |
| DenseGAT / Bi-SR      | ~0.475     | Collapse                       |


---

## Immediate Next Steps (from PATH_TO_0_126)

1. ~~**Output shrinkage**~~ — Done (0.1322 @ 8%). Try eps=0.10.
2. ~~**v3r_pe**~~ — 0.1367 ✗ (LapPE hurt).
3. ~~**Curriculum + bias**~~ — 0.1371 ✗.
4. **DenseSTP** — Await Kaggle result.
4. **stp_pe** — If dense_stp beats 0.1323.
5. **Curriculum / Optuna** — If architecture changes don't close gap.

---

## Reproduction

```bash
# Best single
.venv/bin/python -m src.train_dense_gcn full --preset v3r --max-epochs 600 \
  --submission-path submission/v3r_submission.csv

# DenseSTP (untried)
.venv/bin/python -m src.train_dense_gcn full --model dense_stp --preset stp --max-epochs 600 \
  --submission-path submission/dense_stp_submission.csv
```

