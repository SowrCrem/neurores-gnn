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

    Download your API token from [kaggle.com](https://www.kaggle.com) → **Settings → API → Create New Token**. This saves a `kaggle.json` file. On WSL, your Windows Downloads folder is at `/mnt/c/Users/<your-username>/Downloads/`:
    ```bash
    mkdir -p ~/.kaggle
    cp /mnt/c/Users/<your-username>/Downloads/kaggle.json ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    ```

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
