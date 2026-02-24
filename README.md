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
│   ├── generator.py             # ✅ BrainGNNGenerator stub (implement body)
│   └── layers.py                # ✅ GraphConvBlock stub (implement body)
│
├── src/
│   ├── dataset.py               # ✅ BrainGraphDataset — loads, preprocesses, anti-vectorizes
│   ├── train.py                 # ✅ Checkpoint-safe loop skeleton (TODO: 3-fold CV + model body)
│   └── inference.py             # ✅ Submission formatter (TODO: wire up model)
│
├── utils/
│   ├── matrix_vectorizer.py     # ✅ MatrixVectorizer — column-wise vectorize / anti-vectorize
│   ├── graph_utils.py           # ✅ Preprocessing, degree normalisation, adj ↔ DGL
│   └── metrics.py               # ✅ Calculation of MAE, PCC, JSD, BC, EC, PC + 2 custom measures
│   └── plotting.py              # ✅ Plotting of metrics into bar graphs
│
├── notebooks/
│   ├── devec_check.ipynb        # ✅ Vectorization validation
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

## 🗺️ Implementation Order

### Phase 1 — Metrics & infrastructure  `(unblocks everything)`
- [ ] **`utils/metrics.py`** — implement all 6 required measures (MAE, PCC, JSD, BC MAE, EC MAE, PC MAE) + 2 additional topological measures of choice
- [ ] **`requirements.txt`** — add `networkx` (needed for centrality measures)
- [ ] **`src/train.py`** — replace skeleton loop with 3-fold CV (`KFold(n_splits=3, shuffle=True, random_state=42)`), add seed setup from `reproducibility.py`

### Phase 2 — Model architecture  `(the main work)`
- [ ] **`models/layers.py`** — implement `GraphConvBlock` (GCN/GAT conv + residual + dropout)
- [ ] **`models/generator.py`** — implement `BrainGNNGenerator`: GNN body → learned 160→268 node upsample → pairwise edge prediction head → post-process (clip ≥ 0)

### Phase 3 — Baseline for comparison  `(required by spec)`
- [ ] **`models/sgc_baseline.py`** — Simple Graph Convolution baseline (spec requires benchmarking against it)

### Phase 4 — Submission notebook  `(final deliverable)`
- [ ] **`notebooks/main.ipynb`** — single-run notebook:
  - Seed setup
  - 3-fold CV training loop calling `src/train.py`
  - Evaluation on all 8 metrics per fold
  - Bar plots with std error bars across folds
  - Save `submission/predictions_fold_{0,1,2}.csv`
  - Retrain on full dataset → save `submission/submission.csv` for Kaggle

### Phase 5 — Polish
- [ ] Post-processing: ensure all outputs clipped to `[0, 1]`
- [ ] Verify submission CSV format matches `sample_submission.csv` exactly
- [ ] Final Kaggle submission before March 6

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

8. **Final submission**:
    Open `notebooks/main.ipynb` and run all cells — produces bar plots and `submission/predictions_fold_{0,1,2}.csv`.

---

## 🔧 Baseline: GCN Encoder + Cross-Attention Decoder

Located in `gcn-encoder-ca-decoder/` — a self-contained reference implementation.

**Architecture:**
- **Encoder**: Dense GCN layers extract node embeddings from LR graphs (160 nodes)
- **Decoder**: Cross-attention from HR query embeddings (268 nodes) to LR embeddings
- **Output**: Predicted HR adjacency (268×268) via bilinear scoring

**Training**: 3-fold CV with MSE loss

**Hyperparameters** (see `gcn-encoder-ca-decoder/config.py`):
- `d_model=64`, `gcn_layers=2`, `attn_heads=4`
- `epochs=50`, `batch_size=8`, `lr=1e-3`

See [gcn-encoder-ca-decoder/README.md](gcn-encoder-ca-decoder/README.md) for detailed documentation.

---

## 📊 Evaluation Metrics

Six measures required by the spec (computed on vectorised, flattened predictions):

| Measure | Description |
|---|---|
| MAE | Mean Absolute Error across all predicted edge weights |
| PCC | Pearson Correlation Coefficient |
| JSD | Jensen-Shannon Distance |
| BC MAE | Mean Absolute Error of Betweenness Centrality per subject |
| EC MAE | Mean Absolute Error of Eigenvector Centrality per subject |
| PC MAE | Mean Absolute Error of PageRank Centrality per subject |
| + 2 | Additional topological/geometric measures (TBD) |

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
[GNN body]                      GraphConvBlock × N  →  node embeddings (160, d)
        │
        ▼
[Node upsample]                 Linear(160 → 268)   →  HR node embeddings (268, d)
        │
        ▼
[Edge prediction head]          for each HR pair (i,j): MLP(h_i ‖ h_j) → w_ij
        │
        ▼
[Post-process]                  clip to [0, 1]
        │
        ▼
HR vector (35778,)              re-vectorize upper triangle
```
