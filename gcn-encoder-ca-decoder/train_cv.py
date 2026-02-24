# train_cv.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from data_utils import vec_to_adj, lr_node_features, to_tensor
from model import LR2HRGenerator


def _train_one_fold(
    X_lr_tr: np.ndarray,
    Y_hr_tr: np.ndarray,
    X_lr_va: np.ndarray,
    Y_hr_va: np.ndarray,
    model_kwargs: dict,
    train_kwargs: dict,
    fold_id: int,
):
    device = train_kwargs["device"]
    epochs = train_kwargs["epochs"]
    batch_size = train_kwargs["batch_size"]
    lr = train_kwargs["lr"]
    weight_decay = train_kwargs["weight_decay"]

    print(f"\n===== Fold {fold_id} =====")

    # Tensors
    Xtr = to_tensor(X_lr_tr, device)
    Ytr = to_tensor(Y_hr_tr, device)
    Xva = to_tensor(X_lr_va, device)
    Yva = to_tensor(Y_hr_va, device)

    tr_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, Yva), batch_size=batch_size, shuffle=False)

    model = LR2HRGenerator(**model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):

        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_losses = []

        for x_lr_vec, y_hr_vec in tr_loader:
            A_lr = vec_to_adj(x_lr_vec, model_kwargs["n_lr"])
            X_lr_nodes = lr_node_features(A_lr)

            pred = model(A_lr, X_lr_nodes)
            loss = loss_fn(pred, y_hr_vec)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x_lr_vec, y_hr_vec in va_loader:
                A_lr = vec_to_adj(x_lr_vec, model_kwargs["n_lr"])
                X_lr_nodes = lr_node_features(A_lr)
                pred = model(A_lr, X_lr_nodes)
                val_losses.append(loss_fn(pred, y_hr_vec).item())

        val_loss = float(np.mean(val_losses))

        # -------------------------
        # Best model tracking
        # -------------------------
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # -------------------------
        # Console logging
        # -------------------------
        status = " *" if improved else ""
        print(
            f"Fold {fold_id} | Epoch {epoch:03d}/{epochs} "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f}{status}"
        )

    model.load_state_dict(best_state)
    print(f"Best Val Loss (Fold {fold_id}): {best_val:.6f}")

    return model


def predict(model, X_lr: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """Predict HR vectors for a given LR vector matrix."""
    print("Generating test predictions...")

    model.eval()
    Xt = to_tensor(X_lr, device)
    loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for (x_lr_vec,) in loader:
            A_lr = vec_to_adj(x_lr_vec, model.n_lr)
            X_lr_nodes = lr_node_features(A_lr)
            pred = model(A_lr, X_lr_nodes)
            preds.append(pred.detach().cpu().numpy())

    return np.vstack(preds)


def train_cv_and_predict_test(
    X_lr_train: np.ndarray,
    Y_hr_train: np.ndarray,
    X_lr_test: np.ndarray,
    model_kwargs: dict,
    train_kwargs: dict,
) -> np.ndarray:
    """
    3-fold CV training on (X_lr_train, Y_hr_train), fold-average predictions on X_lr_test.
    """
    print("\nStarting 3-Fold Cross Validation Training...")

    kf = KFold(
        n_splits=train_kwargs["folds"],
        shuffle=True,
        random_state=train_kwargs["random_state"],
    )

    test_preds = []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_lr_train), start=1):
        model = _train_one_fold(
            X_lr_train[tr_idx], Y_hr_train[tr_idx],
            X_lr_train[va_idx], Y_hr_train[va_idx],
            model_kwargs=model_kwargs,
            train_kwargs=train_kwargs,
            fold_id=fold_id
        )

        p = predict(
            model,
            X_lr_test,
            batch_size=train_kwargs["batch_size"],
            device=train_kwargs["device"]
        )
        test_preds.append(p)

    print("\nCross-validation complete. Averaging fold predictions...")
    return np.mean(test_preds, axis=0)