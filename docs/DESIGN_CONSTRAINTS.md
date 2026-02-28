# Design Constraints for Brain Graph Super-Resolution

> Reference document for implementing any brain graph super-resolution model. Use these constraints when designing or experimenting with GNN architectures and building blocks.

**Context:** DGL 2026 Brain Graph Super-Resolution Challenge — 167 training samples, LR 160×160 → HR 268×268 (35,778 edges).

**Source of truth:** Official spec `70105_3_spec.pdf` (DGL Project Spring 2026: Brain Graph Super-Resolution Challenge).

---

## 0. Explicit Spec Constraints (from 70105_3_spec.pdf)

These are **mandatory** requirements from the official specification. Non-compliance will cause submission errors or grading penalties.

### 0.1 Problem & Data

| Constraint | Spec requirement |
|------------|------------------|
| **Setting** | Inductive GNN training only. Predict HR from LR in an inductive setting. |
| **LR shape** | 160×160 matrix → 12,720 vectorized features (upper triangular, column-wise). |
| **HR shape** | 268×268 matrix → 35,778 vectorized features. |
| **Data split** | 167 train, 56 public test, 56 private test. |
| **Vectorization** | Use MatrixVectorizer (vertical, column-by-column). GitHub: basiralab/DGL → Project/MatrixVectorizer.py. |

### 0.2 Pre-processing

| Constraint | Spec requirement |
|------------|------------------|
| **Negative values** | Replace with 0. |
| **NaN values** | Replace with 0. |
| **Data range** | Pre-processed data is [0, 1]; no negative values. |
| **Output post-processing** | Post-process model outputs so they do not include negative values. |

### 0.3 Training Paradigm

| Constraint | Spec requirement |
|------------|------------------|
| **Inductive vs transductive** | Model must be trained **inductively**, NOT transductively. |
| **3-fold CV** | Use `sklearn.model_selection.KFold` with `shuffle=True` and `random_state` from reproducibility.py. |
| **Full retrain for Kaggle** | Retrain on full 167 train samples before predicting test_HR from test_LR. |

### 0.4 Submission Format (Kaggle)

| Constraint | Spec requirement |
|------------|------------------|
| **Shape** | 112 × 35,778 predictions → flattened to 4,007,136 entries. |
| **Method** | Use `numpy.flatten()` (or equivalent). |
| **CSV columns** | Exactly two: `ID` and `Predicted`. |
| **ID column** | Integers 1 to 4,007,136. |
| **Predicted column** | Real-valued predictions in same order as flattened array. |
| **Compliance** | No extra columns, no excess rows, no non-integer IDs, no non-real Predicted values. |

### 0.5 Competition Rules

| Constraint | Spec requirement |
|------------|------------------|
| **Language** | Python only. |
| **Libraries** | Allowed: scikit-learn, PyTorch, TensorFlow, PyG, etc. |
| **Pre-trained models** | **Not allowed.** Must develop, code, train, and test your own model. No fetching/inferring from Hugging Face etc. |
| **Team size** | 3–4 students. Individual submissions not allowed. |

### 0.6 Code Deliverables (Scientia zip)

| Constraint | Spec requirement |
|------------|------------------|
| **3-fold CV** | Code must run 3-fold CV on 167 train samples. |
| **Output files** | `predictions_fold_1.csv`, `predictions_fold_2.csv`, `predictions_fold_3.csv` (same format as Kaggle). |
| **Evaluation measures** | MAE, PCC, JSD, MAE of PageRank Centrality, Eigenvector Centrality, Betweenness Centrality + **2 additional** geometric/topological measures (8 total). |
| **Bar plots** | Compare predicted vs ground-truth HR across all 8 measures. |
| **Exclusions** | No trained weights, no dataset in zip. Include `requirements.txt`. |
| **Reproducibility** | Use config from basiralab/DGL → Project/reproducibility.py. |

### 0.7 Evaluation Metric

| Constraint | Spec requirement |
|------------|------------------|
| **Primary metric** | Mean Absolute Error (MAE). |

---

## 1. Data Scarcity (167 Samples)

| Constraint | Implication |
|------------|-------------|
| **Parameter budget** | Keep total parameters well below ~10k–20k. With 167 samples, large models (e.g. heavy GATs, deep attention) overfit quickly. |
| **Regularization** | Use dropout (0.2–0.4), weight decay, and early stopping. Avoid dropout=0. |
| **Fold stability** | High fold variance (e.g. SGC Fold 2 collapse) suggests sensitivity to splits. Prefer architectures that generalize across folds. |
| **Data augmentation** | Consider symmetry-preserving augmentations (e.g. node permutation) to increase effective sample size. |

---

## 2. Output Dimensionality (35,778 Edges)

| Constraint | Implication |
|------------|-------------|
| **Decoder design** | Bilinear decoders (H @ W @ H^T) are parameter-efficient and enforce symmetry. Avoid fully connected decoders that predict each edge independently. |
| **Non-negativity** | Use softplus or similar for outputs; avoid ReLU/clamp before the decoder, which can cause collapse and PCC=NaN. |
| **Gradient flow** | Ensure gradients reach the decoder; avoid activations that zero out many values (e.g. ReLU before bilinear). |

---

## 3. Structural and Domain Constraints

| Constraint | Implication |
|------------|-------------|
| **Symmetry** | HR adjacency must be symmetric. Use symmetric decoders (e.g. bilinear) or symmetrization. |
| **Permutation equivariance** | LR→HR mapping should be equivariant to node ordering. Use message-passing GNNs, not arbitrary MLPs over node indices. |
| **Graph structure** | LR graph structure matters. Use adjacency in message passing (GCN) or as bias (GAT edge_scale). Avoid pure attention without graph bias. |
| **Node correspondence** | LR 160 and HR 268 are fixed atlases; node ordering is meaningful. Avoid architectures that assume arbitrary permutation. |

---

## 4. Training Stability

| Constraint | Implication |
|------------|-------------|
| **Learning rate** | Use moderate LR (≈5e-4–1e-3). High LR (e.g. 5e-3) can cause collapse and constant outputs. |
| **Activation choice** | Prefer GELU over ReLU before decoders; ReLU can zero many activations and lead to collapse. |
| **Initialization** | Use conservative init for decoder weights (e.g. Xavier gain=0.5) to avoid large initial outputs. |
| **Gradient clipping** | Use grad clipping (e.g. max_norm=5) to avoid instability. |

---

## 5. Model Capacity vs. Expressiveness

| Constraint | Implication |
|------------|-------------|
| **Depth** | 3–4 layers is a reasonable range. More layers add parameters and overfitting risk without clear benefit. |
| **Hidden dimension** | 128–256 is a good range. 192 works for DenseGAT; larger dims need stronger regularization. |
| **Attention vs. GCN** | GCN is simpler and more stable with small data. GAT adds expressiveness but more parameters and sensitivity; use with care. |
| **HR refinement** | Extra self-attention on HR nodes (e.g. hr_refine_layers) can overfit; consider 0 or 1 layer. |

---

## 6. Evaluation and Optimization

| Constraint | Implication |
|------------|-------------|
| **Primary metric** | Optimize MAE; PCC and JSD are secondary. |
| **Loss choice** | SmoothL1/MAE-aligned losses can help; MSE can overemphasize outliers. |
| **3-fold CV** | Use 3-fold CV for model selection; report mean ± std across folds. |
| **Full retrain** | Final model: retrain on all 167 samples after hyperparameter selection. |

---

## 7. Building-Block Guidelines

| Building block | Recommendation |
|----------------|-----------------|
| **Message passing** | Prefer GCN or GAT with adjacency bias; avoid pure attention without graph structure. |
| **Residual connections** | Use them to stabilize deep stacks and gradient flow. |
| **LayerNorm** | Use pre-norm for stability. |
| **Upsample (160→268)** | Linear or Linear→GELU→Linear; avoid very deep MLPs. |
| **Decoder** | Bilinear H @ P @ P^T @ H^T with softplus; avoid clamp. |
| **Dropout** | 0.2–0.4 in encoder; small or no dropout in decoder. |

---

## 8. Spectral Positional Encodings (Optional)

**None of the baseline models** (DenseGCN, DenseGAT, SGC, GCN+Cross-Attn) use spectral positional encodings. Among the recommended SOTA models, **only GraphGPS** explicitly supports Laplacian-based spectral encodings.

- **When to consider:** For GraphGPS-style or transformer-heavy architectures where structural awareness matters. Laplacian eigenvectors can improve long-range dependency modeling.
- **Caveats:** Eigendecomposition adds compute; with 167 samples, heavy use may overfit. Use sparingly (e.g. top-k eigenvectors) and with strong regularization.
- **Alternatives:** Bi-SR uses fixed random features for HR node initialization; DEFEND/STP-GSR use dual-graph formulations without explicit positional encoding.
