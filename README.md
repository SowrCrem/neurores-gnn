# DGL 2026 Brain Graph Super-Resolution Challenge

## Contributors

- **Team:** Big Dawgs
- **Members:** Saurajit Seth, Dhruv Himatsingka, Vivian Lopez, Shivam Subudhi

## Problem Description

Graph super-resolution aims to recover a high-resolution (HR) graph from a low-resolution (LR) version of the same underlying structure. In the context of brain connectivity, this corresponds to predicting a detailed brain network (with more regions and finer connections) from a coarser one. Formally, the task is to learn a mapping from a low-resolution adjacency matrix to its corresponding high-resolution adjacency matrix.

This problem is important because acquiring high-resolution brain data (e.g. fMRI with fine parcellations) is expensive, time-consuming and often noisy, whereas low-resolution data is more readily available. A successful super-resolution model enables researchers to infer detailed brain connectivity without requiring costly data acquisition, improving accessibility and scalability of neuroscience studies.

Applications include disease diagnosis (e.g. identifying biomarkers for neurological disorders), brain network analysis, personalized medicine and enhancing downstream tasks such as brain graph classification or clustering. More broadly, graph super-resolution is relevant in any domain where relational structures are observed at multiple scales.

However, the problem is challenging for several reasons. First, the mapping from LR to HR graphs is highly ill-posed, as multiple HR graphs may correspond to the same LR graph. Second, brain graphs are structured, noisy and subject-specific, making generalization difficult. Third, preserving topological properties (e.g. connectivity patterns, centrality measures) while increasing resolution is non-trivial. Finally, the mismatch in dimensionality between LR and HR graphs introduces additional complexity in model design and training.

## NeuroRes-GNN: Methodology

**NeuroRes-GNN** uses the **v3r_eb_ffnn** configuration as the best-performing model to date. Key building blocks:

1. **Dense GCN encoder**: Multi-layer graph convolution with residual connections, LayerNorm and an optional FFN (feed-forward network) sublayer in each block.
2. **Learnable per-edge bias**: Decoder includes a learnable bias term per HR edge, improving MAE over the baseline.
3. **Residual learning + Mixup**: Per-edge HR mean subtraction during training; Graph Mixup (α=0.2) for regularization.
4. **Bilinear edge decoder**: Predicted HR adjacency via H W H^T with edge bias, symmetrised and clamped to [0, ∞).

The model is trained **inductively** with 3-fold cross-validation. Evaluation uses 8 measures: MAE, PCC, JSD and average MAE of PageRank, Eigenvector, Betweenness centralities, plus node strength and clustering coefficient.

*(Figure to be added later.)*

## Repository Structure

What is kept in this repo and why:

| Path | Purpose |
|------|---------|
| `notebooks/main.ipynb` | **Spec deliverable** - 3-fold CV, bar plots, `predictions_fold_{1,2,3}.csv`, `submission.csv`. Run this for the full pipeline. |
| `src/train_dense_gcn.py` | Main training script - 3-fold CV, full retrain, presets (v3r_eb_ffnn, etc.). Invoked by `main.ipynb` for the best config. |
| `models/` | Model definitions - DenseGCN, SGC baseline, VGAE baseline, Bi-SR, GIN, GAT, etc. |
| `utils/` | Shared utilities - `matrix_vectorizer`, `metrics` (8 measures), `plotting` (bar charts). |
| `gcn-encoder-ca-decoder/` | Data helpers - `data_utils.py` (`vec_to_adj`, `lr_node_features`, `to_tensor`) used by `main.ipynb`. |
| `reproducibility.py` | Reproducibility config (Spec Note 2) - seed, CUDA/cudnn settings. |
| `requirements.txt` | Dependencies for reproducible runs. |

**Not in repo** (gitignored; retain locally if needed):

- `docs/` - Internal planning, research notes, Kaggle analysis. Not required for submission.
- `scripts/` - Auxiliary tools (postprocess, graph statistics). Not part of the core pipeline.
- `data/`, `submission/`, `artifacts/`, `checkpoints/` - Data and outputs; download data via Kaggle CLI.

## Used External Libraries

```bash
pip install -r requirements.txt
```

Libraries: **torch**, **dgl**, **numpy**, **scipy**, **pandas**, **scikit-learn**, **kaggle**, **networkx**, **pyyaml**, **tqdm**, **psutil**, **optuna**, **matplotlib**.

## Results

<!-- TODO: Insert bar plots. Run `notebooks/main.ipynb` to generate bar plots and compare NeuroRes-GNN (v3r_eb_ffnn) with SGC and VGAE baselines. -->

## References

- Wu et al., "Simplifying Graph Convolutional Networks", ICML 2019 (SGC baseline)
- Kipf & Welling, "Variational Graph Auto-Encoders", NeurIPS 2016 Workshop (VGAE baseline)
- DGL Tutorial 2: https://github.com/basiralab/DGL/tree/main/Tutorials/Tutorial-2
- basiralab/DGL Project: https://github.com/basiralab/DGL/blob/main/Project/
- *(Add brain graph super-resolution papers used for the best model.)*
