# Kaggle Submission Analysis: Brain Graph Super-Resolution

> Thorough analysis of all 17 submissions to the DGL 2026 Brain Graph Super-Resolution Challenge.  
> Primary metric: **MAE** (lower is better).

---

## 1. Executive Summary


| Finding                                   | Implication                                                                                                 |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Bimodal performance gap**               | DenseGCN family: ~0.17–0.20 MAE. All alternatives (GAT, GIN, Bi-SR, CA, GPS): ~0.47–0.48 MAE.               |
| **Full retrain is critical**              | v3 full retrain: 0.176 vs v3 ensemble (no full retrain): 0.196 — **~11% relative improvement**.             |
| **Architecture matters more than tuning** | DenseGCN v3 variants (dropout, hidden dim) stay in 0.176–0.178. Alternative architectures plateau at ~0.48. |
| **Best single model**                     | `dense_gcn_v3_full_retrain_submission.csv` — **0.176083** MAE.                                              |
| **Best ensemble**                         | `dense_gcn_v3_ensemble_full_submission.csv` — **0.176744** MAE (3 seeds, full retrain).                     |


---

## 2. Results by Tier

### Tier 1: Strong Performers (MAE 0.17–0.20)


| Submission                 | MAE          | Notes                                    |
| -------------------------- | ------------ | ---------------------------------------- |
| dense_gcn_v3_full_retrain  | **0.176083** | Single seed, full retrain on 167 samples |
| dense_gcn_v3_ensemble_full | 0.176744     | 3-seed ensemble, full retrain            |
| dense_gcn_v3_d04           | 0.178061     | v3 + dropout=0.4                         |
| dense_gcn_v3_h256_d04      | 0.178131     | v3 + h_dim=256, dropout=0.4              |
| dense_gcn_submission_1     | 0.195672     | DenseGCN v1 (legacy)                     |
| dense_gcn_v3_ensemble      | 0.196269     | 3-fold CV ensemble (no full retrain)     |
| dense_gcn_v2               | 0.197395     | DenseGCN v2                              |


**Common thread:** All use **DenseGCN** encoder + bilinear decoder. Only DenseGCN achieves this tier.

### Tier 2: Moderate (MAE 0.26)


| Submission       | MAE      | Notes                          |
| ---------------- | -------- | ------------------------------ |
| sgc_submission_1 | 0.261464 | SGC v2 (K-hop, 2-dim features) |
| sgc_submission   | 0.375375 | SGC v1                         |


**Gap:** SGC uses 2-dim node features vs DenseGCN’s 160-dim adjacency rows. SGC has no learnable message passing.

### Tier 3: Weak Performers (MAE ~0.47–0.48)


| Submission         | MAE      | Notes                            |
| ------------------ | -------- | -------------------------------- |
| dense_gat_v4_lr5e4 | 0.474895 | DenseGAT v4, collapse fix        |
| dense_gat_v4       | 0.475942 | Dense GAT (full retrain)         |
| dense_gat_v5       | 0.479093 | DenseGAT v5 (GCN-first redesign) |
| dense_bisr_v2      | 0.476535 | Dense Bi-SR v2                   |
| dense_bisr_prelim  | 0.476706 | Dense Bi-SR preliminary          |
| dense_gcn_ca       | 0.476486 | DenseGCN + cross attention       |
| dense_gcn_gps      | 0.479213 | GraphGPS: GCN + linear attention |
| dense_gin          | 0.481457 | Dense GIN                        |


**Common thread:** All use **non-GCN** encoders or non-standard upsampling. All plateau near ~0.48 regardless of full retrain.

---

## 3. Key Insights

### 3.1 Why DenseGCN Dominates

1. **Stable inductive bias**
  - Normalized adjacency `S = D^{-1/2}(A+I)D^{-1/2}` gives well-scaled aggregation.
  - ReLU in blocks keeps activations non-negative before bilinear decoder.
  - Simple linear upsample 160→268 is easy to learn with 167 samples.
2. **Parameter efficiency**
  - v3: 192 hidden dim, 3 layers, dropout 0.35.
  - Bilinear decoder `H @ W @ H^T` is O(d²) vs O(n²) for dense edge prediction.
3. **Training stability**
  - SmoothL1 loss, moderate LR (8e-4), patience 45.
  - No attention collapse, no output collapse (softplus at decoder).

### 3.2 Why Alternatives Fail (~0.48)


| Architecture              | Likely Cause                                                                                                                                   |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **DenseGAT**              | Attention + ReLU before decoder caused collapse (ANALYSIS_VAL_LOSS.md). v5 GCN-first still ~0.48 — attention block may overfit or destabilize. |
| **DenseGIN**              | Uses **unnormalized** adjacency `A @ H`; for weighted graphs this can distort scale. Sum aggregation differs from GCN’s normalized mean.       |
| **Dense Bi-SR**           | Bipartite upsampling adds complexity; 167 samples may be insufficient for the extra structure.                                                 |
| **GCN + Cross-Attention** | 268×d HR queries add parameters; cross-attention may overfit. Quick CV showed val MAE ~0.44.                                                   |
| **GraphGPS**              | Linear attention + GCN hybrid; more parameters, possible overfitting.                                                                          |


**Hypothesis:** The task favors a **simple, structure-aware encoder** (GCN) with a **bilinear decoder**. Extra expressiveness (attention, bipartite, cross-attention) increases overfitting risk with 167 samples.

### 3.3 Full Retrain vs 3-Fold CV


| Setup                            | Example          | MAE       |
| -------------------------------- | ---------------- | --------- |
| Full retrain (167 samples)       | v3 full retrain  | **0.176** |
| 3-fold CV ensemble (each on 2/3) | v3 ensemble      | 0.196     |
| Full retrain + ensemble          | v3 ensemble full | **0.176** |


**Takeaway:** Full retrain gives ~0.02 MAE improvement. Ensemble of full-retrain models does not beat single full retrain here (0.176 vs 0.176).

### 3.4 Regularization and Capacity


| Variant                      | MAE   | Interpretation                            |
| ---------------------------- | ----- | ----------------------------------------- |
| v3 (baseline)                | 0.176 | Best                                      |
| v3 + dropout 0.4             | 0.178 | Slightly worse — may underfit             |
| v3 + h_dim 256 + dropout 0.4 | 0.178 | Same — larger capacity + reg doesn’t help |
| v3 ensemble full             | 0.176 | Matches single                            |


**Takeaway:** v3 preset is near-optimal. Stronger regularization or larger capacity does not improve.

---

## 4. Recommendations

### Immediate (High ROI)

1. **Stick with DenseGCN v3** — Best architecture for this task and data size.
2. **Hyperparameter tuning** — Optuna on v3 (LR, dropout, hidden_dim, num_layers) has not been tried; may yield 0.16–0.17.
3. **Ensemble diversity** — Try different seeds, different dropout, or different loss (SmoothL1 vs MSE) for ensemble members.

### Medium-Term (Architecture Experiments)

1. **DenseGIN with normalized adjacency** — GIN currently uses unnormalized `A`; try `S` (GCN-style) to test if normalization is the issue.
2. **Bi-SR with stronger regularization** — Reduce bipartite layers to 1, increase dropout, smaller hidden dim.
3. **GCN+CA with fewer HR queries** — Reduce query dim (e.g. d=64) and add more dropout.

### Defer

- GraphGPS, DEFEND, STP-GSR — Higher complexity, higher overfitting risk with 167 samples.
- DenseGAT — Multiple redesigns (v4, v5) still ~0.48; architecture may be ill-suited.

---

## 5. Data Summary Table


| Rank | Submission                 | MAE      | Model            | Full Retrain? |
| ---- | -------------------------- | -------- | ---------------- | ------------- |
| 1    | dense_gcn_v3_full_retrain  | 0.176083 | DenseGCN v3      | Yes           |
| 2    | dense_gcn_v3_ensemble_full | 0.176744 | DenseGCN v3      | Yes           |
| 3    | dense_gcn_v3_d04           | 0.178061 | DenseGCN v3      | Yes           |
| 4    | dense_gcn_v3_h256_d04      | 0.178131 | DenseGCN v3      | Yes           |
| 5    | dense_gcn_submission_1     | 0.195672 | DenseGCN v1      | No            |
| 6    | dense_gcn_v3_ensemble      | 0.196269 | DenseGCN v3      | No            |
| 7    | dense_gcn_v2               | 0.197395 | DenseGCN v2      | No            |
| 8    | sgc_submission_1           | 0.261464 | SGC v2           | No            |
| 9    | sgc_submission             | 0.375375 | SGC v1           | No            |
| 10   | dense_gat_v4_lr5e4         | 0.474895 | DenseGAT v4      | Yes           |
| 11   | dense_gat_v4               | 0.475942 | DenseGAT v4      | Yes           |
| 12   | dense_gat_v5               | 0.479093 | DenseGAT v5      | Yes           |
| 13   | dense_bisr_v2              | 0.476535 | Dense Bi-SR v2   | Yes           |
| 14   | dense_bisr_prelim          | 0.476706 | Dense Bi-SR      | Yes           |
| 15   | dense_gcn_ca               | 0.476486 | GCN + Cross-Attn | Yes           |
| 16   | dense_gcn_gps              | 0.479213 | GraphGPS         | Yes           |
| 17   | dense_gin                  | 0.481457 | Dense GIN        | Yes           |


---

## 6. Conclusion

The DenseGCN v3 architecture is the clear winner for this challenge. Alternative architectures (GAT, GIN, Bi-SR, cross-attention, GraphGPS) all plateau at ~0.48 MAE, suggesting either:

1. **Task–architecture mismatch** — The brain graph super-resolution task may be well-suited to simple GCN + bilinear decoder.
2. **Data scarcity** — 167 samples may be insufficient for more expressive models.
3. **Training instability** — Attention-based and bipartite models may need different hyperparameters or training procedures.

**Bottom line:** Focus optimization efforts on DenseGCN v3 (tuning, ensembles, data augmentation) rather than new architectures until the ~0.48 barrier is understood.