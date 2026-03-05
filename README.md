# NeuroRes-GNN
### DGL 2026: Brain Graph Super-Resolution Challenge

Generative GNN framework to predict High-Resolution (HR) brain connectivity graphs (268×268) from Low-Resolution (LR) inputs (160×160).

**Deadline:** Kaggle → March 6 · Scientia → March 9

---

## 📂 Project Structure
<!-- Cursor Note: Use this map to navigate the codebase and initialize new modules. -->

```
neurores-gnn/
│
├── gcn-encoder-ca-decoder/      # ✅ Baseline: GCN Encoder + Cross-Attention Decoder
│   ├── main.py                  #   Entry point (3-fold CV + test prediction)
│   ├── model.py                 #   LR2HRGenerator architecture
│   ├── data_utils.py            #   Tensor operations & vectorization
│   ├── train_cv.py              #   3-fold cross-validation loop
│   ├── config.py                #   Model & training configs
│   └── README.md                #   Baseline documentation
│
├── data/                        # ⛔ git-ignored — download via Kaggle CLI
│   ├── lr_train.csv             #   (167, 12720)  LR training subjects
│   ├── hr_train.csv             #   (167, 35778)  HR training subjects
│   └── lr_test.csv              #   (112, 12720)  LR test subjects (no labels)
│
├── models/
│   ├── generator.py             # ✅ BrainGNNGenerator (GNN body → upsample → bilinear decoder)
│   └── layers.py                # ✅ GraphConvBlock (GCN + edge weights + residual + LN)
│
├── src/
│   ├── dataset.py               # ✅ BrainGraphDataset — loads, preprocesses, anti-vectorizes
│   ├── train.py                 # ✅ 3-fold CV training with per-fold checkpointing + metrics
│   └── inference.py             # ✅ Batched inference + Kaggle submission CSV
│
├── utils/
│   ├── matrix_vectorizer.py     # ✅ MatrixVectorizer — column-wise vectorize / anti-vectorize
│   ├── graph_utils.py           # ✅ Preprocessing, degree normalisation, adj ↔ DGL
│   ├── metrics.py               # ✅ All 8 metrics (6 required + 2 additional for spec §II.A.a)
│   ├── plotting.py              # ✅ Plotting of metrics into bar graphs
│   └── dgl_compat.py            # ✅ DGL graphbolt compatibility shim
│
├── notebooks/
│   ├── devec_check.ipynb        # ✅ Vectorization validation
│   ├── dense_gcn_analysis.ipynb # ✅ Canonical analysis notebook (reads script artifacts)
│   ├── dense_gcn_v1.ipynb       # 📌 Legacy experiment log (training moved to script)
│   ├── dense_gcn_v2.ipynb       # 📌 Legacy experiment log (training moved to script)
│   ├── modular_pipeline.ipynb   # 🔲 (in-progress) Modular ML pipeline template
│   └── main.ipynb               # 🔲 TODO: submission deliverable
│
├── configs/
│   └── base_model.yaml          # ✅ Hyperparameter config (lr, hidden_dim, epochs, etc.)
│
├── submission/                  # ⛔ git-ignored — CSV outputs
├── checkpoints/                 # ⛔ git-ignored — model checkpoints
├── requirements.txt             # ✅ Dependencies
└── .venv/                       # ⛔ git-ignored — virtual environment
```

**Legend:** ✅ done · 🔲 todo · ⛔ git-ignored

---

## TODO: Manual cleanup (files not needed for Kaggle submission)

The following files/directories are **not required** to produce a submission CSV. They can be removed or archived to slim the repo before sharing or packaging.

| Path | Why not needed |
|------|----------------|
| `docs/` | Planning, analysis, and path docs (PATH_TO_0_126, RESEARCH_ANALYSIS_REPORT, etc.). Used for development only. |
| `notebooks/` | Exploratory notebooks (dense_gcn_analysis, devec_check, sgc_baseline_v2, etc.). Canonical workflow is `src/train_dense_gcn.py`. |
| `scripts/` | Auxiliary scripts (`postprocess_submission.py`, `graph_statistics_analysis.py`). Not used by the main training/inference pipeline. |
| `gcn-encoder-ca-decoder/` | Separate baseline implementation. Not imported by `train_dense_gcn.py`. |
| `configs/` | `base_model.yaml` is used by legacy `src/train.py`, not by `train_dense_gcn.py`. |
| `src/train.py` | Legacy training script. Superseded by `src/train_dense_gcn.py`. |
| `src/dataset.py` | Used by legacy `train.py`. `train_dense_gcn` loads CSV directly. |
| `src/inference.py` | Standalone inference module. `train_dense_gcn` has its own predict logic. |
| `models/generator.py`, `models/layers.py` | Legacy BrainGNN (DGL-based). Not used by dense GCN/GAT/Bi-SR pipeline. |
| `models/sgc_baseline.py` | SGC baseline. Not used by `train_dense_gcn` (different entry point). |
| `models/vgae.py` | VGAE baseline. Not used by main pipeline. |
| `models/bipartite.py` | Standalone bipartite impl. Bi-SR uses `models/dense_bisr.py` instead. |
| `utils/dgl_compat.py` | DGL compatibility shim. Not used by `train_dense_gcn`. |
| `utils/graph_utils.py` | DGL graph helpers. Not used by dense pipeline. |
| `utils/example.py` | Example script. Not part of submission flow. |

**Minimal submission set:** `src/train_dense_gcn.py`, `models/dense_*.py` (dense_gcn, dense_gat, dense_bisr, dense_gin, dense_gcn_ca, dense_gcn_gps, dense_gcn_lrs, dense_graphsage, dense_stp), `utils/matrix_vectorizer.py`, `utils/metrics.py`, `utils/plotting.py`, `requirements.txt`.

---

## 🗺️ Implementation Order

### Phase 1 ✅ — Metrics & infrastructure
- [x] **`utils/metrics.py`** — all 8 metrics (6 required + 2 additional: Node Strength, Clustering Coefficient)
- [x] **`utils/plotting.py`** — fold-wise bar chart visualisation
- [x] **`utils/matrix_vectorizer.py`** — column-wise vectorize / anti-vectorize
- [x] **`utils/graph_utils.py`** — preprocessing, degree normalisation, adj ↔ DGL
- [x] **`src/dataset.py`** — `BrainGraphDataset` (load, preprocess, anti-vectorize)
- [x] **`configs/base_model.yaml`** — hyperparameter config
- [x] **`requirements.txt`** — all dependencies including `networkx`

### Phase 2 ✅ — Reference model (SOTA comparison candidate)
- [x] **`gcn-encoder-ca-decoder/`** — GCN Encoder + Cross-Attention Decoder (coded, not yet trained/evaluated)
- [x] **`notebooks/modular_pipeline.ipynb`** — modular pipeline template (DataModule, Preprocessing, Model, Training, CV, Submission)

### Phase 3 ✅ — Proposed model architecture
- [x] **`models/layers.py`** — `GraphConvBlock` (GCN message passing + edge weights + residual + LayerNorm + dropout)
- [x] **`models/generator.py`** — `BrainGNNGenerator` (GNN body → learned node upsample → bilinear edge decoder)
- [x] **`src/train.py`** — 3-fold CV with per-fold checkpointing + full 8-metric evaluation
- [x] **`src/inference.py`** — batched inference with checkpoint loading + submission CSV

### Phase 4 🔲 — Comparison models  `(spec §3.2B — 6 pts)`
The spec requires benchmarking against **two** other methods using the same 3-fold CV split:
- [x] **SGC baseline** — `models/sgc_baseline.py` — Simple Graph Convolution ([Tutorial 2](https://github.com/basiralab/DGL/tree/main/Tutorials/Tutorial-2), Wu et al. 2019)
- [x] **SOTA comparison** — one of 3 published brain graph super-resolution methods identified in the report (the `gcn-encoder-ca-decoder/` may qualify, or pick a paper)
- [x] Train both with 3-fold CV and evaluate on all 8 metrics for the comparison table

### Phase 5 🔲 — Submission notebook  `(spec §3.1 — 20 pts)`
- [ ] **`notebooks/main.ipynb`** — single-run notebook that generates **all** outputs:
  - Reproducibility setup (`reproducibility.py` seed config at top)
  - Receives train/test data + model params as function parameters
  - 3-fold CV training of proposed model (`KFold(n_splits=3, shuffle=True, random_state=42)`)
  - `predictions_fold_{0,1,2}.csv` — Kaggle format (ID + Predicted columns, 4,007,136 rows each)
  - Bar plots of all 8 metrics per fold with std error bars (see spec Fig. 5)
  - Retrain on full 167 training set → predict test_HR → `submission/submission.csv` for Kaggle
  - Explanatory comments throughout

### Phase 6 🔲 — Kaggle submission  `(spec §2 — 5 pts)`
- [ ] Team registered on Kaggle competition
- [ ] Upload `submission.csv` (ID + Predicted, 4,007,136 rows) before **March 6**
- [ ] Post-process: all predictions clipped to `[0, ∞)` (no negatives)
- [ ] Verify CSV format matches `sample_submission.csv` exactly

### Phase 7 🔲 — Report  `(spec §3.2 — 75 pts, deadline March 9)`
IEEE Conference Paper template, max 5 pages + optional pipeline figure page.

**Methodology & Novelty (40 pts):**
- [ ] Problem description & motivation (5)
- [ ] State-of-the-art: identify 3 GNN-based brain graph SR papers in Table I (3)
- [ ] Main figure: learning pipeline diagram showing train + test flow (4)
- [ ] Brief overview of proposed GNN architecture (5)
- [ ] Two novel contributions vs. existing SOTA in Table II (10)
- [ ] Mathematical properties: permutation invariance (5), equivariance (5), expressiveness (3)

**Experimental Setup & Evaluation (27 pts):**
- [ ] Results: 3-fold CV plots (8 measures), training time + RAM usage, Kaggle score (9)
- [ ] Comparison: proposed model vs. SGC baseline vs. SOTA method, quantitative (6)
- [ ] Scalability analysis (7)
- [ ] Reproducibility analysis (5)

**Discussion & Reflections (8 pts):**
- [ ] Two strengths + two weaknesses in Table IV (4)
- [ ] Two key improvements / future work directions (4)

---

## 🚀 Getting Started

1. **Create and activate the virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Authenticate with Kaggle**:

    Go to [kaggle.com](https://www.kaggle.com) → **Settings → API** and copy your API key. Then paste it into:
    ```bash
    mkdir -p ~/.kaggle
    nano ~/.kaggle/access_token
    ```
    Paste your API key as plain text and save (`Ctrl+O`, `Enter`, `Ctrl+X`).

4. **Download competition data**:
    ```bash
    kaggle competitions download -c dgl-2026-brain-graph-super-resolution-challenge
    rm dgl-2026-brain-graph-super-resolution-challenge.zip
    ```

5. **Validate data pipeline** (optional sanity check):
    Open `notebooks/devec_check.ipynb` with the `neurores-gnn` kernel and run all cells.

6. **Run baseline implementation** (GCN Encoder + Cross-Attention Decoder):
    ```bash
    python gcn-encoder-ca-decoder/main.py
    ```
    ✅ Outputs: 3-fold CV training logs + `submission.csv` with ensemble predictions

7. **Train** (development):
    ```bash
    python src/train.py --config configs/base_model.yaml
    # resumes automatically from checkpoints/latest.pt after hibernation
    ```

8. **DenseGCN v2 (script-first, no notebook redefinition)**:
    ```bash
    # 3-fold CV + all 8 metrics + total runtime + peak RAM logs
    python -m src.train_dense_gcn cv

    # Retrain on full train_LR/train_HR (167) + predict test_LR + save submission
    python -m src.train_dense_gcn full
    ```
    For the balanced **v3 preset** (higher capacity + MAE-aligned loss):
    ```bash
    python -m src.train_dense_gcn cv --preset v3
    python -m src.train_dense_gcn full --preset v3 --submission-path submission/dense_gcn_v3_full_retrain_submission.csv
    ```
    Outputs are written to the chosen `--out-dir`, for example:
    - `artifacts/dense_gcn_v3/cv_summary.json`
    - `artifacts/dense_gcn_v3/resource_summary.json`
    - `artifacts/dense_gcn_v3/full_retrain_summary.json`
    - `submission/dense_gcn_v3_full_retrain_submission.csv`

    For **DenseGAT v4** with Optuna hyperparameter tuning, see [docs/HYPERPARAMETER_TUNING.md](docs/HYPERPARAMETER_TUNING.md).

9. **Analyse and report from artifacts**:
    Open `notebooks/dense_gcn_analysis.ipynb`, set `ARTIFACT_DIR`, and run all cells to generate:
    - 8-metric fold plots,
    - mean/std tables,
    - runtime + RAM usage table,
    - report-ready text snippets for section II-A.

### Current DenseGCN status (quick reference)
- `v1` submission (`dense_gcn_v1_submission.csv`) remains the strongest non-full-retrain baseline.
- `v3` CV-ensemble (`dense_gcn_v3_ensemble_submission.csv`) improves over `v2` but may still trail `v1`.
- `v3` full retrain (`dense_gcn_v3_full_retrain_submission.csv`) can substantially improve leaderboard MAE.
- For report compliance, use: **3F-CV for model selection** -> **full retrain on all 167** -> **predict test_LR**.

10. **Final submission**:
    Open `notebooks/main.ipynb` and run all cells — produces bar plots and `submission/predictions_fold_{0,1,2}.csv`.

### Legacy notebooks
- `notebooks/dense_gcn_v1.ipynb` and `notebooks/dense_gcn_v2.ipynb` are kept for experiment history.
- Canonical workflow is now: `src/train_dense_gcn.py` (training/inference) + `notebooks/dense_gcn_analysis.ipynb` (visualization/reporting).

---

## 🔧 Baseline: GCN Encoder + Cross-Attention Decoder

Located in `gcn-encoder-ca-decoder/` — a self-contained reference implementation.

**Architecture:**
- **Encoder**: Dense GCN layers extract node embeddings from LR graphs (160 nodes)
- **Decoder**: Cross-attention from HR query embeddings (268 nodes) to LR embeddings
- **Output**: Predicted HR adjacency (268×268) via bilinear scoring

**Training**: 3-fold CV with MSE loss

**Hyperparameters** (see `gcn-encoder-ca-decoder/config.py`):
- `d_model=128`, `gcn_layers=3`, `attn_heads=4`, `dropout=0.1`
- `epochs=60`, `batch_size=16`, `lr=1e-3`, `weight_decay=1e-5`

**Status:** Code complete but not yet trained — no logs or outputs produced.

See [gcn-encoder-ca-decoder/README.md](gcn-encoder-ca-decoder/README.md) for detailed documentation.

---

## 📊 Baselines & Model Architectures (3-Fold CV on Kaggle Dataset)

**Dataset:** DGL 2026 Brain Graph Super-Resolution — 167 train (LR 160×160 → HR 268×268), 112 test.

| Model | Architecture | MAE ↓ | PCC ↑ | JSD ↓ | Status |
|-------|---------------|-------|-------|-------|--------|
| **SGC Baseline** | K-hop propagation (no learnable GCN), linear proj, bilinear decoder | 0.38* | 0.03* | — | High fold variance (Fold 2 collapsed) |
| **DenseGCN v1** | 3× GCN blocks, 160-dim features, bilinear decoder | 0.242 | -0.014 | 0.59 | ✅ Best CV MAE (legacy) |
| **DenseGCN v2** | Same, lower LR (5e-4), MSE loss | 0.239 | -0.024 | 0.59 | ✅ Baseline |
| **DenseGCN v3** | 3× GCN, hidden=192, SmoothL1, dropout=0.35 | 0.242 ± 0.003 | -0.022 ± 0.007 | 0.593 | ✅ Balanced preset |
| **DenseGAT v4** | 4× GAT blocks, HR refine, bilinear decoder | 0.260 | NaN | 0.356 | ⚠️ Val loss stuck (fixed in code) |
| **DenseGAT v4 (fixed)** | Softplus decoder, P@P^T, lr=8e-4 | — | — | — | 🔄 Re-run after fix |
| **GCN + Cross-Attn** | Dense GCN encoder, HR queries, cross-attention decoder | — | — | — | 🔲 Not trained |
| **BrainGNN** | DGL GraphConvBlock, bilinear decoder | — | — | — | 🔲 Reference (src/train.py) |

*SGC: Fold 1 MAE 0.234, Fold 2 MAE 0.642 (collapse), Fold 3 MAE 0.254.

**Kaggle leaderboard:** Fill after submission. Primary metric = MAE on test set.

### Architecture summary (for exploration)

- **SGC:** No learnable message passing — `H = S^K X`, single linear. Fast but limited.
- **DenseGCN:** Full adjacency rows as features, GCN blocks (S @ H @ W + ReLU + residual), learned upsample 160→268, bilinear `H W H^T`.
- **DenseGAT:** GAT blocks (attention + adjacency bias), GELU, upsample, HR refine (self-attn), bilinear `H P P^T H^T` + softplus.
- **GCN+Cross-Attn:** GCN encoder, learnable HR query embeddings, cross-attention to LR nodes, bilinear decoder.

### Suggested next directions

1. **Train GCN+Cross-Attn** — already implemented; HR queries may capture structure better than learned upsample.
2. **DenseGAT v4 (fixed)** — run full CV after softplus decoder fix; val loss improved in quick test.
3. **Hybrid GCN+GAT** — GCN for local structure, GAT for long-range; or GAT encoder with GCN decoder.
4. **Decoder variants** — multi-head bilinear, graph attention over HR edges, or MLP over edge features.
5. **SGC stability** — Fold 2 collapse suggests sensitivity; try different K, init, or regularization.

---

## 📊 Evaluation Metrics

Eight measures (6 required + 2 additional per spec §II.A.a), computed on vectorised predictions:

| Measure | Description |
|---|---|
| MAE | Mean Absolute Error across all predicted edge weights |
| PCC | Pearson Correlation Coefficient |
| JSD | Jensen-Shannon Distance |
| BC MAE | Mean Absolute Error of Betweenness Centrality per subject |
| EC MAE | Mean Absolute Error of Eigenvector Centrality per subject |
| PC MAE | Mean Absolute Error of PageRank Centrality per subject |
| Strength MAE | **Additional** — Node Strength (weighted degree) per subject |
| CC MAE | **Additional** — Weighted Clustering Coefficient per subject |

Reference implementation: [`evaluation_measures.py`](https://github.com/basiralab/DGL/blob/main/Project/evaluation_measures.py)

---

## 🏗️ Architecture Overview

```
LR adjacency (160×160)
        │
        ▼
[anti-vectorize → DGL graph]    node features = rows of LR adj (160-dim each)
        │
        ▼
[Input projection]              Linear(160 → d) + LayerNorm + ReLU
        │
        ▼
[GNN body]                      GraphConvBlock × N  →  node embeddings (160, d)
        │
        ▼
[Node upsample]                 Learned linear (160 → 268) over node dimension
        │
        ▼
[Bilinear edge decoder]         h W hᵀ → (268×268), symmetrised
        │
        ▼
[Post-process]                  clamp ≥ 0, extract upper triangle
        │
        ▼
HR vector (35778,)              predicted edge weights
```
