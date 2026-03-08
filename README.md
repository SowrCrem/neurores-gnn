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

**NeuroRes-GNN** uses `v3r_eb_ffnn_aug_light` as the best-performing model in this repository. In this preset name, `v3r` is the residual-learning DenseGCN baseline, `eb` adds a learnable per-edge decoder bias, `ffnn` adds feed-forward sublayers inside each GCN block, and `aug_light` denotes lighter augmentation (5% edge dropout, 0.01 Gaussian noise, 60% mixup probability). The mixup component is a sample-wise mixup-style interpolation on vectorized graph inputs and targets, rather than the graphon-based G-Mixup method of Han et al. (2022). The submission variant applies 3% inference shrinkage toward the training mean.

**Leaderboard result:** Our public-best submission (`v3r_eb_ffnn_aug_light_shrink03`) achieved a Kaggle public MAE of 0.127094, which initially placed the team 6th out of 24 on the public leaderboard. Our best private-leaderboard submission was `v3r_eb_ffnn_aug_light_learnshrink`, which finished 4th place with a final MAE of 0.148109, within 0.74% of the first-place score (0.147022).

### Architecture (`v3r_eb_ffnn_aug_light`)

The model maps a low-resolution (LR) brain graph (160 nodes) to a high-resolution (HR) graph (268 nodes). All operations are dense tensor matmuls - no sparse graph library required.

| Stage | Description |
|-------|-------------|
| **1. Node features** | Each LR node is represented by its adjacency row: `X_lr ∈ ℝ^{160×160}`. The full LR adjacency is used as input. |
| **2. Normalisation** | Symmetric normalisation: `S = D^{-1/2}(A + I)D^{-1/2}` so message passing is scale-invariant. |
| **3. Input projection** | `Linear(160 → 192)` + LayerNorm + ReLU + Dropout. |
| **4. Dense GCN blocks (×3)** | Each block has two sublayers with residual connections: (a) **Message passing**: `H' = S @ H`, linear transform, LayerNorm, ReLU, dropout, then `H = H + H'`; (b) **FFN sublayer**: `FFN(H) = Linear(192→768) → ReLU → Dropout → Linear(768→192)`, then `H = H + FFN(LayerNorm(H))`. |
| **5. Node upsample** | `Linear(160 → 268)` maps LR node embeddings to HR node space. |
| **6. Bilinear edge decoder** | `A_hr = H @ W_edge @ H^T`, symmetrised. Extract upper triangle → 35,778 HR edge weights. |
| **7. Per-edge bias** | Learnable bias `b_e` per HR edge: `pred_e = bilinear_e + b_e`. Captures systematic per-edge offsets (e.g. hub vs sparse edges) that the bilinear form cannot express. |

### Training

| Component | Details |
|-----------|---------|
| **Residual learning** | Targets are centred: `y_target = y_hr - y_mean` (per-edge mean over training set). The model predicts deviations; at inference, `pred_final = pred + y_mean`. |
| **Loss** | L1 (MAE-aligned). |
| **Regularisation** | Mixup-style augmentation (α=0.2, prob=0.6) via direct convex interpolation of vectorized LR graphs and HR targets; 5% edge dropout, Gaussian noise std=0.01; dropout 0.35. |
| **Optimiser** | AdamW, lr=8e-4, patience=60. |
| **Evaluation** | 3-fold cross-validation, inductively. Eight measures: MAE, PCC, JSD, and average MAE of PageRank, Eigenvector, Betweenness centralities, node strength, and clustering coefficient. |

Final predictions are clamped to `[0, ∞)` before submission.

## Repository Structure

What is kept in this repo and why:

| Path | Purpose |
|------|---------|
| `notebooks/main.ipynb` | 3-fold CV, bar plots, `predictions_fold_{1,2,3}.csv`, `submission.csv`. Run this for the full pipeline. |
| `src/train_dense_gcn.py` | Main training script - 3-fold CV, full retrain, presets (v3r_eb_ffnn_aug_light, etc.). Invoked by `main.ipynb` for the best config. |
| `models/` | Model definitions - DenseGCN, SGC baseline, VGAE baseline, Bi-SR, GIN, GAT, etc. |
| `utils/` | Shared utilities - `matrix_vectorizer`, `metrics` (8 measures), `plotting` (bar charts). |
| `gcn-encoder-ca-decoder/` | Data helpers - `data_utils.py` (`vec_to_adj`, `lr_node_features`, `to_tensor`) used by `main.ipynb`. |
| `reproducibility.py` | Reproducibility config (Spec Note 2) - seed, CUDA/cudnn settings. |
| `requirements.txt` | Dependencies for reproducible runs. |

## Used External Libraries

```bash
pip install -r requirements.txt
```

Libraries: **torch**, **dgl**, **numpy**, **scipy**, **pandas**, **scikit-learn**, **kaggle**, **networkx**, **pyyaml**, **tqdm**, **psutil**, **optuna**, **matplotlib**.

## Results

`notebooks/main.ipynb` is the submission notebook. In one run, it:

- loads `lr_train.csv`, `hr_train.csv`, and `lr_test.csv`
- runs 3-fold cross-validation with `shuffle=True` and the anchored `random_seed`
- writes `predictions_fold_1.csv`, `predictions_fold_2.csv`, `predictions_fold_3.csv`, and `submission.csv`
- generates grouped bar plots for all 8 evaluation measures
- optionally compares NeuroRes-GNN against the SGC and VGAE baselines

For the repository submission, the latest generated spec outputs are tracked under `submission/`.

**Reproduce best submission (v3r_eb_ffnn_aug_light_shrink03):**

```bash
# Full pipeline (spec deliverable): run notebooks/main.ipynb for 3-fold CV, bar plots, submission.csv

# Or run training directly for a Kaggle submission
.venv/bin/python -m src.train_dense_gcn full --preset v3r_eb_ffnn_aug_light --max-epochs 600 --val-ratio 0.15 --patience 60 --shrinkage-eps 0.03 --submission-path submission/v3r_eb_ffnn_aug_light_shrink03.csv
```

## References

- Wu et al., "Simplifying Graph Convolutional Networks," *ICML*, 2019.
- Isallari and Rekik, "Brain graph super-resolution using adversarial graph neural network with application to functional brain connectivity," *Medical Image Analysis*, 2021.
- Mhiri, Ben Khalifa, Mahjoub, and Rekik, "Brain graph super-resolution for boosting neurological disorder diagnosis using unsupervised multi-topology connectional brain template learning," *Medical Image Analysis*, 2020.
- Gao and Ji, "Graph U-Nets," *ICML*, 2019.
- Kipf and Welling, "Variational Graph Auto-Encoders," *NeurIPS Workshop*, 2016.
- Velickovic et al., "Graph Attention Networks," *ICLR*, 2018.
- Han, Jiang, Liu, and Hu, "G-Mixup: Graph Data Augmentation for Graph Classification," *ICML*, 2022. This paper informed our discussion of graph-data augmentation, but our implementation uses a simpler sample-wise mixup-style interpolation rather than graphon-based G-Mixup.
- DGL Tutorial 2: https://github.com/basiralab/DGL/tree/main/Tutorials/Tutorial-2
- basiralab/DGL Project: https://github.com/basiralab/DGL/blob/main/Project/
