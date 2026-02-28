# Model Architectures for Brain Graph Super-Resolution

> Recommended models for the DGL 2026 Brain Graph Super-Resolution Challenge. See [DESIGN_CONSTRAINTS.md](DESIGN_CONSTRAINTS.md) for implementation constraints.

---

## Overview

| Model | Rank | Expected Gain | Effort | Spectral PE |
|-------|------|---------------|--------|-------------|
| Bi-SR | 1 | High | Medium | No |
| GCN + Cross-Attention | 2 | Moderate | Low | No |
| GraphGPS-style | 3 | Low–Moderate | Medium | Yes (optional) |
| STP-GSR | 4 | Moderate–High | High | No |
| DEFEND | 5 | Low–Moderate | High | No |
| Lightweight Edge Decoder | 6 | Low | Medium–High | No |

---

## 1. Bi-SR (Bipartite Super-Resolution)

**Source:** Singh & Rekik, "Rethinking Graph Super-resolution: Dual Frameworks for Topological Fidelity" (2025). [arXiv:2511.08853](https://arxiv.org/abs/2511.08853)

### Idea

Replace matrix-transpose upsampling with a bipartite graph connecting LR (160) and HR (268) nodes. Each HR node aggregates information from all LR nodes via message passing.

### Why It Fits Constraints

| Constraint | How Bi-SR Helps |
|------------|-----------------|
| Structure-aware upsampling | Bipartite MP uses graph structure instead of simple transpose |
| Permutation invariance | HR predictions invariant to LR node ordering |
| Parameter efficiency | Reuses existing GNN blocks; only upsampling changes |
| Small data | GNN-agnostic; can use shallow GCN (2–3 layers) |
| Topology | Better alignment with brain topology than transpose-based methods |

### Implementation Sketch

1. Build bipartite graph: LR nodes ↔ HR nodes (fully connected across sides)
2. Initialize HR nodes with fixed random features U(0,1) to break symmetry
3. Run 1–2 GNN layers on the bipartite graph
4. Take HR node embeddings → bilinear decoder → softplus

### Effort

Medium. Swap the current linear upsample in DenseGCN for a bipartite upsampling module.

---

## 2. GCN + Cross-Attention

**Location:** `gcn-encoder-ca-decoder/` (already implemented)

### Idea

GCN encoder on LR; learnable HR query embeddings (268 × d); cross-attention from HR queries to LR nodes; bilinear decoder.

### Why It Fits Constraints

| Constraint | How It Helps |
|------------|--------------|
| HR structure | HR queries can encode atlas-specific structure |
| Parameter control | 268×d queries; keep d modest (64–128) |
| Stability | GCN is stable; cross-attention is standard |
| Implementation | Already coded; needs training and tuning |

### Tuning Notes

- Use moderate LR (5e-4–1e-3), dropout 0.2–0.3
- Replace clamp with softplus in decoder
- Add ReduceLROnPlateau scheduler

### Effort

Low. Train and tune.

---

## 3. GraphGPS-Style (Local + Global)

**Source:** Rampášek et al., "Recipe for a General, Powerful, Scalable Graph Transformer" (NeurIPS 2022)

### Idea

Combine local message passing (GCN) with global attention. Local part provides graph bias; global part captures long-range dependencies. Use linear-complexity attention (e.g. Performer, Linear Transformer) to avoid O(N²).

### Why It Fits Constraints

| Constraint | How It Helps |
|------------|--------------|
| Over-smoothing | Global attention mitigates over-smoothing in deep GCN |
| Expressiveness | Can model non-local dependencies in brain graphs |
| Small graphs | 160 LR nodes is manageable for linear attention |
| Parameter control | Use small hidden dim and few layers |

### Spectral Positional Encoding

GraphGPS supports Laplacian-based spectral encodings (eigenvectors/eigenvalues). Optional for this architecture; use top-k eigenvectors to limit overfitting with 167 samples.

### Implementation Sketch

1. 2–3 GCN blocks (local)
2. 1 linear-attention block (global) over LR nodes
3. Learned upsample 160→268
4. Bilinear decoder + softplus

### Effort

Medium. Add linear-attention module.

---

## 4. STP-GSR (Strongly Topology-Preserving GNN)

**Source:** Singh & Rekik, "Strongly Topology-preserving GNNs for Brain Graph Super-resolution" (2024). [arXiv:2411.02525](https://arxiv.org/abs/2411.02525) | [GitHub: basiralab/STP-GSR](https://github.com/basiralab/STP-GSR)

### Idea

Primal-dual formulation: map LR edges to dual nodes, run GNN on dual graph, map dual nodes back to HR edges. Direct edge-space learning.

### Why It Fits Constraints

| Constraint | How STP-GSR Helps |
|------------|-------------------|
| Topology | Designed for topology preservation; strong on 7 topological measures |
| Efficiency | Uses shallow GNNs; dual graph is sparse |
| Brain graphs | Evaluated on brain connectomes |
| GNN-agnostic | Can plug in GCN or GAT |

### Implementation Sketch

1. **Target Edge Initializer:** LR graph → GNN → node embeddings → scalar edge features for HR
2. **Primal2Dual:** Flatten HR edge features to dual node features
3. **Dual Graph Learner:** GNN on dual graph
4. **Dual2Primal:** Map dual node outputs back to HR adjacency

### Caveat

LR→HR mapping is more complex (LR node features → LR edge features → dual nodes → HR edges). Requires careful adaptation to 160→268 setup.

### Effort

High. Two-stage pipeline and dual-graph machinery.

---

## 5. DEFEND (Dual Edge Feature Learning and Detection)

**Source:** Same paper as Bi-SR. [GitHub: basiralab/DEFEND](https://github.com/basiralab/DEFEND)

### Idea

Map HR edges to nodes of a dual graph. Run GNN on the dual graph so node-level computations correspond to edge-level learning. Dual graph has ~35,778 nodes and ~97% sparsity.

### Why It Fits Constraints

| Constraint | How DEFEND Helps |
|------------|------------------|
| Edge-level learning | Directly models edges instead of inferring from node dot products |
| Topology | Captures higher-order structure (cliques, hubs) better than node-based models |
| Scalability | Dual graph is sparse; message passing is O(edges) |
| Evaluation | Strong on topological metrics (clustering, centrality, etc.) |

### Implementation Sketch

1. From LR: obtain HR node features (via Bi-SR or current upsample)
2. Initialize edge features: `E_ij = x_i · x_j`
3. Build dual graph: each HR edge → dual node; connect dual nodes if edges share a vertex
4. Run 1–2 GNN layers on dual graph
5. Map dual node outputs back to HR edges → min-max normalize → softplus

### Caveat

35,778 dual nodes is large for 167 samples. Use shallow GNN (1–2 layers) and strong regularization.

### Effort

High. New dual-graph construction and message passing; memory for 35k nodes.

---

## 6. Lightweight Edge Decoder (MLP over Edge Pairs)

### Idea

Keep GCN encoder and upsample. Instead of bilinear `H @ W @ H^T`, predict edges from concatenated node pairs: `e_ij = MLP([h_i || h_j])` with shared MLP. Use sampling or factorized design for parameter efficiency.

### Why It Fits Constraints

| Constraint | How It Helps |
|------------|--------------|
| Expressiveness | MLP can capture non-bilinear edge functions |
| Parameter control | Shared MLP over all edges; no d² bilinear matrix |
| Symmetry | Use `MLP([h_i || h_j])` with shared weights; enforce `e_ij = e_ji` |

### Caveat

35,778 edges is too many to process individually. Use edge sampling (mini-batch of edges per forward) or factorized design.

### Effort

Medium–High. Requires batching/sampling strategy.

---

## Recommended Implementation Order

Prioritized by expected improvement vs. risk given 167 samples and design constraints:

1. **Bi-SR** — Best balance of impact vs. risk. Targets main weakness (linear upsample ignores structure). Low overfitting risk; reuses existing GCN.
2. **GCN + Cross-Attention** — Quick win; already implemented. HR queries encode atlas structure. Validate and tune (LR, dropout, softplus decoder).
3. **GraphGPS-style** — If Bi-SR and GCN+Cross-Attn plateau. Addresses over-smoothing and long-range deps. Medium overfitting risk.
4. **STP-GSR** — For SOTA comparison in report. High ceiling but complex; use very shallow GNNs.
5. **DEFEND** — Powerful edge-level learning but 35k dual nodes + 167 samples = high overfitting risk. Try only if Bi-SR helps and topology metrics need more.
6. **Lightweight Edge Decoder** — Skip for now. Sampling complexity; uncertain benefit over bilinear.
