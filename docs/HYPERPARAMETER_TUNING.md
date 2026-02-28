# Hyperparameter Tuning (DenseGAT v4)

> **Keep in sync with:** `src/train_dense_gcn.py` tune mode. Update this doc when CLI args or workflow change.

Bayesian optimization with **Optuna** over DenseGAT hyperparameters using 3-fold cross-validation. **Resumable** — Optuna stores trials in SQLite; re-run the same command to continue after interruption.

**Requires:** `pip install optuna`

---

## Quick Start

```bash
# Install Optuna (if not already)
.venv/bin/pip install optuna

# Start tuning (resumes automatically if interrupted)
.venv/bin/python -m src.train_dense_gcn tune --preset v4 \
  --out-dir artifacts/dense_gat_v4_tune \
  --out-config artifacts/dense_gat_v4_tune/best_config.json

# Optional: run full retrain automatically after tuning (no manual second step)
.venv/bin/python -m src.train_dense_gcn tune --preset v4 \
  --out-dir artifacts/dense_gat_v4_tune --full-retrain \
  --submission-path submission/dense_gat_v4_submission.csv

# Or manually: full retrain with best config (after tuning completes)
.venv/bin/python -m src.train_dense_gcn full --preset v4 --max-epochs 400 --val-ratio 0.15 --patience 50 \
  --edge-scale <best> --hr-refine-layers <best> --dropout <best> --lr <best> \
  --hidden-dim <best> --num-layers <best> \
  --submission-path submission/dense_gat_v4_submission.csv
```

---

## Commands

### 1. Hyperparameter tuning (Optuna, resumable)


| Command          | Purpose                                                                                                                                                    |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Start / resume   | `.venv/bin/python -m src.train_dense_gcn tune --preset v4 --out-dir artifacts/dense_gat_v4_tune --out-config artifacts/dense_gat_v4_tune/best_config.json` |
| With full retrain | Add `--full-retrain` (and optional `--submission-path PATH`) to run full retrain with best config after tuning                                            |
| With more trials | Add `--n-trials 30` (default 20)                                                                                                                           |
| Fresh start      | Add `--fresh` to clear study and start from scratch                                                                                                        |


**Rough time:** ~4–6 hours for 20 trials (each trial = 3 folds × ~10–15 min)

**Checkpointing:** Optuna stores trials in `artifacts/dense_gat_v4_tune/optuna_study.db`. Re-run the same command to resume.

---

### 2. 3-fold CV (resumable)


| Command   | Purpose                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------- |
| Run CV    | `.venv/bin/python -m src.train_dense_gcn cv --preset v4 --out-dir artifacts/dense_gat_v4 --fresh` |
| Resume CV | Omit `--fresh`; progress in `cv_progress.json`                                                    |


**Rough time:** ~1 hour (3 folds × ~20 min)

---

### 3. Full retrain (early stopping by default)

Full retrain holds out a validation set (`--val-ratio`, default 0.15) and stops when validation loss does not improve for `--patience` epochs (default 50), up to `--max-epochs` (default 400). Use `--val-ratio 0` to train on all data for a fixed number of epochs (no early stopping).

| Command      | Purpose                                                                                                                                 |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| Full retrain | `.venv/bin/python -m src.train_dense_gcn full --preset v4 --submission-path submission/dense_gat_v4_submission.csv`                     |
| No early stop| Add `--val-ratio 0 --max-epochs 50` to train on all data for 50 epochs                                                                   |

**Rough time:** Depends on early stopping; often ~15–30 min.

**Note:** Full retrain is not checkpointed; if interrupted, re-run from scratch.

---

## Search Space (Optuna)


| Param            | Range                     |
| ---------------- | ------------------------- |
| edge_scale       | 0.1 – 0.5                 |
| hr_refine_layers | 0, 1                      |
| dropout          | 0.1 – 0.4                 |
| lr               | 5e-4 – 1.5e-3 (log scale) |
| hidden_dim       | 128, 192                  |
| num_layers       | 3, 4                      |


Optuna uses Bayesian optimization to suggest configs; typically finds good configs in fewer trials than grid search.

---

## Outputs


| Path                                           | Description                          |
| ---------------------------------------------- | ------------------------------------ |
| `artifacts/dense_gat_v4_tune/optuna_study.db`  | Optuna SQLite storage (resume state) |
| `artifacts/dense_gat_v4_tune/best_config.json` | Best config + all trial results      |


---

## Workflow

1. Run `tune` (optionally with `--full-retrain`) → wait for completion or interrupt (resume later with same command).
2. If you used `--full-retrain`, submission CSV is already written; otherwise read `best_config.json` and run `full` with those overrides.
3. Submit the submission CSV to Kaggle.

