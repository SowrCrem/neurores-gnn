# NeuroRes-GNN 🧠
### DGL 2026: Brain Graph Super-Resolution Challenge

This repository contains the generative Graph Neural Network (GNN) framework designed to predict High-Resolution (HR) brain connectivity graphs from Low-Resolution (LR) inputs.

---

## 📂 Project Structure
> **Cursor Note:** Use this map to navigate the codebase and initialize new modules.

* **`data/`**: (Ignored by Git) Contains `train/`, `public_test/`, and `private_test/` adjacency matrices.
* **`models/`**: 
    * `generator.py`: GNN architecture for LR to HR mapping.
    * `layers.py`: Custom DGL layers or graph convolution blocks.
* **`src/`**: 
    * `dataset.py`: DGL Dataset class for loading brain graphs.
    * `train.py`: Training loop with MAE loss tracking.
    * `inference.py`: Logic for generating submissions from the test set.
* **`utils/`**: 
    * `metrics.py`: Custom implementation of Mean Columnwise MAE.
    * `graph_utils.py`: Normalization and adjacency processing.
* **`configs/`**: YAML files for hyperparameters (learning rate, hidden dims).
* **`submission/`**: Stores the final `.csv` or `.npy` files for Kaggle.

---

## 🚀 Getting Started

1.  **Create and activate the virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Authenticate with Kaggle**:

    Go to [kaggle.com](https://www.kaggle.com) → **Settings → API** and copy your API key. Then paste it into:
    ```bash
    mkdir -p ~/.kaggle
    nano ~/.kaggle/access_token
    ```
    Paste your API key as plain text and save (`Ctrl+O`, `Enter`, `Ctrl+X`).

4.  **Download competition data**:
    ```bash
    kaggle competitions download -c dgl-2026-brain-graph-super-resolution-challenge
    unzip dgl-2026-brain-graph-super-resolution-challenge.zip -d data/
    rm dgl-2026-brain-graph-super-resolution-challenge.zip
    ```

5.  **Training**:
    ```bash
    python src/train.py --config configs/base_model.yaml
    ```

---

## 📊 Evaluation Metric
The primary metric is **Mean Absolute Error (MAE)** between the predicted HR connectivity matrices and the ground truth.

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$
