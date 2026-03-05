# Graph Statistics Analysis: LR vs HR

Derived from `src/dataset.py`, `utils/matrix_vectorizer.py`, `utils/graph_utils.py`
and training CSVs. No speculation — all quantities computed from data.

## 1. Mean, Variance, Skewness (Edge Weights)

### LR (160×160, 12720 edges per sample)

- **Mean (all edges)**: 0.198116
- **Variance (all edges)**: 0.040386
- **Skewness (all edges)**: 0.9108
- **Mean (nonzero only)**: 0.273788
- **Variance (nonzero)**: 0.035094
- **Skewness (nonzero)**: 0.7280

### HR (268×268, 35778 edges per sample)

- **Mean (all edges)**: 0.259923
- **Variance (all edges)**: 0.049778
- **Skewness (all edges)**: 0.5598
- **Mean (nonzero only)**: 0.327019
- **Variance (nonzero)**: 0.040685
- **Skewness (nonzero)**: 0.4707

## 2. Sparsity

### LR
- **Mean sparsity**: 0.2764
- **Std sparsity**: 0.0753

### HR
- **Mean sparsity**: 0.2052
- **Std sparsity**: 0.0632

## 3. Node Strength Distribution

### LR
- **Mean strength**: 31.5005
- **Variance strength**: 127.4253
- **Skewness strength**: 0.9208

### HR
- **Mean strength**: 69.3995
- **Variance strength**: 729.6225
- **Skewness strength**: 0.3018

## 4. Spectral Eigenvalue Decay

Ratios use signed eigenvalues (λ₅₀/λ₁ can be negative when spectrum crosses zero); 
|λ|-based decay avoids sign flip.

### LR
- **λ₁₀/λ₁ (signed)**: 0.0817
- **λ₅₀/λ₁ (signed)**: -0.0143
- **|λ₁₀|/|λ₁| (magnitude decay)**: 0.0817
- **|λ₅₀|/|λ₁| (magnitude decay)**: 0.0143

### HR
- **λ₁₀/λ₁ (signed)**: 0.0610
- **λ₅₀/λ₁ (signed)**: -0.0040
- **|λ₁₀|/|λ₁| (magnitude decay)**: 0.0610
- **|λ₅₀|/|λ₁| (magnitude decay)**: 0.0040

## 5. Low-Rank Assessment (HR)

Effective rank = exp(entropy of normalized eigenvalue distribution).
Frobenius concentration = fraction of ||A||²_F in top-k eigenvalues.

### HR
- **Effective rank (mean)**: 102.39
- **Effective rank (std)**: 10.06
- **Frob concentration k=10**: 0.9561
- **Frob concentration k=50**: 0.9768
- **Frob concentration k=100**: 0.9856

### LR (for comparison)
- **Effective rank (mean)**: 77.04
- **Frob concentration k=10**: 0.9211

## 6. LR→HR Mapping: Smooth vs Structurally Nonlinear

k-NN smoothness: if mapping is smooth, nearby LR points map to nearby HR points.
Ratio = mean HR distance within k-NN / mean HR distance outside. Ratio < 1 ⇒ smooth.

- **Mean HR dist within 5-NN**: 235.1609
- **Mean HR dist outside 5-NN**: 266.4070
- **Smoothness ratio**: 0.8827

## 7. 3-Fold CV Statistical Stability (n=167)

- **Val size per fold (fixed seed=42)**: [56, 56, 55]
- **Train size per fold**: [111, 111, 112]
- **Val overlap (fold 0 vs 1, over 500 bootstrap seeds)**: 0.0000
- **Effective n_val per fold**: ~55

**Assessment**: With 167 samples, 3-fold CV yields ~56 validation samples per fold. 
The mean across 3 folds has variance ≈ σ²_fold/3. The fold-level std is estimated from 
only 3 values, so the reported 'mean ± std' has high variance in the std term. 
SE(mean) ≈ σ_metric/√(n_val×3) ≈ σ_metric/√168; thus the 3-fold mean is nearly as stable 
as a single validation on 168 samples. The main limitation is estimating σ from 3 folds.
