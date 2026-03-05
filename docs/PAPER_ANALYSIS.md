# Research Paper Analysis: Brain Graph Super-Resolution

> **Current best:** 0.1326. **Target:** 0.126. **Action plan:** [PATH_TO_0_126.md](PATH_TO_0_126.md)

Analysis of 2024–2026 papers for predicting HR (268×268) from LR (160×160) brain graphs, 167 samples.

## Highly Relevant Papers (Adopted / Influential)

### 1. Strongly Topology-preserving GNNs for Brain Graph Super-resolution (STP-GSR)
- **Year:** 2024 (MICCAI)
- **Authors:** Pragya Singh, Islem Rekik
- **ArXiv:** [2411.02525](https://arxiv.org/abs/2411.02525)
- **Problem Statement:** Existing GNNs for graph super-resolution perform representation learning in the *node space*. When predicting edges (connections), standard models rely on inferring edge weights from node representations (e.g., using a bilinear decoder `H @ W @ H^T`). This approach fails to adequately capture higher-order topological structures like cliques and hubs, which are critical in brain networks.
- **Proposed Solution:** STP-GSR introduces a **Dual-Graph (or Line-Graph) paradigm**. It maps the edge space of the LR graph to the *node space* of an HR dual graph. This means the GNN aggregations happen directly on the topological edges (which are now treated as nodes), enforcing mathematical topological consistency.
- **Relevance:** **Extremely High**. Our current framework (DenseGCN) is hitting a hard performance floor (~0.134 MAE) exactly because of this bilinear decoder rank bottleneck. STP-GSR provides the exact mathematical paradigm shift needed for our problem constraints to break this wall.

### 2. Rethinking Graph Super-Resolution: Dual Frameworks for Topological Fidelity
- **Year:** 2024 / 2025 (ICLR 2025 Submission)
- **Authors:** Pragya Singh, Islem Rekik
- **ArXiv:** [2411.06019](https://arxiv.org/abs/2411.06019) (approximate based on search) / OpenReview
- **Problem Statement:** Expanding on the STP-GSR concepts, this paper addresses the lack of permutation invariance and the disregard of underlying graph structures in matrix-based node super-resolution methods.
- **Proposed Solution:** Introduces two frameworks:
  - **Bi-SR (Bipartite Super-Resolution):** Creates a bipartite graph connecting LR and HR nodes, enabling structure-aware node super-resolution.
  - **DEFEND (Dual-graph Edge Representation for Fidelity):** Learns edge representations by mapping HR edges to nodes of a dual graph (similar to STP-GSR).
- **Relevance:** **Extremely High**. DEFEND directly aligns with the STP-GSR paradigm, confirming that edge-as-node representation is the current state-of-the-art for preserving fidelity in brain connectomes. Bi-SR also offers an alternative (Bipartite graphs) if dual-graphs prove computationally too heavy.

### 3. Learning Efficient Positional Encodings with Graph Neural Networks (PEARL)
- **Year:** 2025 (ICLR 2025)
- **ArXiv / OpenReview:** [AWg2tkbydO](https://openreview.net/forum?id=AWg2tkbydO) (OpenReview)
- **Problem Statement:** Positional encodings (PEs) are required for GNNs to understand structural location, but Laplacian eigenvectors (the standard) are computationally expensive and unstable.
- **Proposed Solution:** PEARL demonstrates that message-passing GNNs initialized with random node inputs or standard basis vectors can efficiently generate expressive learnable PEs that match the power of full eigenvector-based methods at linear complexity.
- **Relevance:** **High**. Our data uses fixed brain atlases (the nodes mean the same physical brain regions across all subjects). Breaking node permutation symmetry by giving the model a "sense of place" (e.g., "this node is the visual cortex") without the high cost of Laplacian eigendecomposition is highly valuable for our topological metrics (PCC).

---

## Papers Evaluated but Excluded from Immediate Architecture

### 4. Bridging Distance and Spectral Positional Encodings via Anchor-Based Diffusion Geometry Approximation
- **Year:** 2026
- **ArXiv:** [2601.04517](https://arxiv.org/abs/2601.04517)
- **Main Concept:** Uses shortest-path distances from nodes to a small set of "anchors" to approximate spectral diffusion geometry.
- **Why it was excluded:** Brain connectivity matrices are densely connected (mild sparsity of 20-27%). In such dense graphs, almost all nodes are within 1 or 2 hops of each other. Shortest-path distance metrics lose their discriminative power in dense networks compared to continuous, learnable embeddings.

### 5. Mesh-based Super-resolution of Detonation Flows / Fluid Flows with Multiscale Graph Transformers / GNNs
- **Year:** 2024 / 2025
- **ArXiv:** [2511.12041](https://arxiv.org/abs/2511.12041), [2409.07769](https://arxiv.org/abs/2409.07769)
- **Main Concept:** Uses multiscale graph transformers and GNNs with unpooling layers to super-resolve 2D/3D physical spatial meshes in computational fluid dynamics.
- **Why it was excluded:** 
  1. **Domain Mismatch:** Physical spatial meshes have explicit 2D/3D coordinate geometry (up/down/left/right), whereas brain graphs are non-Euclidean connectivity manifolds. 
  2. **Data Scarcity:** These architectures rely on heavy Transformer backbones which require massive datasets. Applying Transformers to our dataset of only 167 training samples leads to immediate catastrophic overfitting.

### 6. Cross-Modal Super-Resolution Graph Neural Network (Cross-SRGANs)
- **Year:** 2025
- **Main Concept:** Uses Generative Adversarial Networks (GANs) and GNNs to super-resolve neuroimaging data by fusing multiple modalities (MRI, PET, CT).
- **Why it was excluded:** We only have a single modality (structural/functional connectivity matrices) mapped from LR to HR. We do not have cross-modal data (like matching PET/CT scans) to leverage for fusion. Furthermore, GANs are notoriously unstable and hard to train on very small datasets (167 samples) compared to direct regression models.

### 7. Brain Graph Super-Resolution Using Adversarial Graph Neural Network (AGSR-Net)
- **Year:** 2021
- **Main Concept:** Foundational paper using Graph U-Nets and adversarial training for brain graph SR.
- **Why it was excluded:** While foundational, it operates purely in the node-space. The 2024/2025 literature (STP-GSR, DEFEND) has explicitly proven that node-space methods fall short in preserving higher-order topologies compared to Dual-Graph methods. We are skipping older node-space architectures to implement the latest Edge-space SOTA.
