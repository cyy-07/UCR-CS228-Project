"""
src/training.py  —  Shared training & evaluation utilities.

All experiment scripts import `train()`, `evaluate()`, `persistence_reference()`
from here so the training loop stays consistent across workstreams.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
#  Loss-function factory
# ──────────────────────────────────────────────────────────────

def get_loss_fn(name: str = "mse", pos_weight=None) -> nn.Module:
    """
    Build a loss module by name.
        mse          → nn.MSELoss()         (default; matches RMSE)
        l1           → nn.L1Loss()          (matches MAE evaluation)
        huber        → nn.HuberLoss()       (robust to outliers)
        smooth       → nn.SmoothL1Loss()    (alias of Huber w/ beta=1)
        bce          → nn.BCEWithLogitsLoss()             (binary classification)
        bce_balanced → BCEWithLogitsLoss(pos_weight=...)  (class-imbalance aware;
                       pass pos_weight = (1 - p) / p where p = positive rate)
    """
    name = name.lower()
    if name == "mse":     return nn.MSELoss()
    if name == "l1":      return nn.L1Loss()
    if name == "huber":   return nn.HuberLoss(delta=1.0)
    if name == "smooth":  return nn.SmoothL1Loss(beta=1.0)
    if name == "bce":     return nn.BCEWithLogitsLoss()
    if name == "bce_balanced":
        if pos_weight is None:
            raise ValueError("bce_balanced requires pos_weight (float).")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)]))
    raise ValueError(f"Unknown loss: {name}")


# ──────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          label: str = "", epochs: int = 50, lr: float = 1e-3,
          patience: int = 10, loss_name: str = "mse",
          weight_decay: float = 1e-4, grad_clip: float = 1.0,
          log_every: int = 5, seed: int = 42,
          loss_fn: nn.Module = None) -> nn.Module:
    """Adam + ReduceLROnPlateau + early stopping. Returns best model.

    Pass `loss_fn=<nn.Module>` to bypass `loss_name` and use a pre-built loss
    (needed for class-imbalanced BCE with a runtime-computed pos_weight)."""
    if not list(model.parameters()):       # PersistenceBaseline has no params
        return model

    torch.manual_seed(seed); np.random.seed(seed)

    model.to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    if loss_fn is None:
        loss_fn = get_loss_fn(loss_name)
    loss_fn = loss_fn.to(DEVICE)

    best_val, best_state, wait = float("inf"), None, 0
    t0 = time.time()

    def _mae(pred, y):
        # Quantile models output [B, H, Q]; compare median against y.
        if pred.dim() == 3:
            mid = pred.size(-1) // 2
            return (pred[..., mid] - y).abs().mean().item()
        return (pred - y).abs().mean().item()

    for ep in range(1, epochs + 1):
        # ── training pass ──
        model.train()
        train_loss, train_mae, n_train = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x)
            batch_loss = loss_fn(pred, y)
            batch_loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            train_loss += batch_loss.item() * len(x)
            train_mae  += _mae(pred, y) * len(x)
            n_train    += len(x)
        train_loss /= n_train
        train_mae  /= n_train

        # ── validation pass ──
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_loss += loss_fn(pred, y).item() * len(x)
                val_mae  += _mae(pred, y) * len(x)
        val_loss /= len(val_loader.dataset)
        val_mae  /= len(val_loader.dataset)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val, best_state, wait = val_loss, {k: v.cpu().clone()
                for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break

        if ep % log_every == 0 or ep == 1:
            print(f"    ep{ep:3d}  train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"train_mae_n={train_mae:.4f}  val_mae_n={val_mae:.4f}  "
                  f"[{label} {time.time()-t0:.0f}s]")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ──────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader, scaler,
             corrupt_fn=None, log_target: bool = False) -> dict:
    """
    Return MAE & RMSE in original Wh units (inverse-transformed).
    If `log_target` was used during training, exponentiate predictions first.
    """
    model.eval(); model.to(DEVICE)
    preds, trues = [], []
    use_median = hasattr(model, "predict_median")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            if corrupt_fn is not None:
                x = corrupt_fn(x)
            p = model.predict_median(x) if use_median else model(x)
            # Defensive: if model still returns 3D (quantile), take median.
            if p.dim() == 3:
                p = p[..., p.size(-1) // 2]
            preds.append(p.cpu())
            trues.append(y)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    sc, mn = scaler.scale_[0], scaler.mean_[0]
    preds = preds * sc + mn
    trues = trues * sc + mn

    if log_target:                                # invert log1p
        preds = np.expm1(preds)
        trues = np.expm1(trues)

    mae  = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    return {"MAE": mae, "RMSE": rmse}


# ──────────────────────────────────────────────────────────────
#  Collect per-window predictions in original Wh units
# ──────────────────────────────────────────────────────────────

def collect_predictions(model: nn.Module, loader: DataLoader, scaler,
                        log_target: bool = False):
    """Return (preds_wh, trues_wh), each [N, H] in original units, in the
    loader's (un-shuffled) order. Used by per-day / per-segment analyses."""
    model.eval(); model.to(DEVICE)
    preds, trues = [], []
    use_median = hasattr(model, "predict_median")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            p = model.predict_median(x) if use_median else model(x)
            if p.dim() == 3:
                p = p[..., p.size(-1) // 2]
            preds.append(p.cpu()); trues.append(y)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    sc, mn = scaler.scale_[0], scaler.mean_[0]
    preds = preds * sc + mn
    trues = trues * sc + mn
    if log_target:
        preds = np.expm1(preds); trues = np.expm1(trues)
    return preds, trues


# ──────────────────────────────────────────────────────────────
#  Persistence reference loss (for training diagnostics)
# ──────────────────────────────────────────────────────────────

def persistence_reference(loader: DataLoader, scaler, name: str = "",
                          log_target: bool = False) -> dict:
    """
    Compute Persistence MSE/MAE on a loader in normalized AND Wh space.
    Trained models should beat this on the train set.
    """
    pred_len = next(iter(loader))[1].shape[1]
    mse_n, mae_n, n = 0.0, 0.0, 0
    preds_w, trues_w = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = x[:, -1, 0:1].expand(-1, pred_len)
            mse_n += ((pred - y) ** 2).mean().item() * len(x)
            mae_n += (pred - y).abs().mean().item() * len(x)
            n     += len(x)
            preds_w.append(pred.numpy()); trues_w.append(y.numpy())
    mse_n /= n; mae_n /= n
    pw = np.concatenate(preds_w) * scaler.scale_[0] + scaler.mean_[0]
    tw = np.concatenate(trues_w) * scaler.scale_[0] + scaler.mean_[0]
    if log_target:
        pw = np.expm1(pw); tw = np.expm1(tw)
    mae_wh  = float(np.mean(np.abs(pw - tw)))
    rmse_wh = float(np.sqrt(np.mean((pw - tw) ** 2)))
    print(f"  [Persistence ref on {name}]  "
          f"mse_n={mse_n:.5f}  mae_n={mae_n:.4f}  "
          f"MAE_Wh={mae_wh:.3f}  RMSE_Wh={rmse_wh:.3f}")
    return {"mse_n": mse_n, "mae_n": mae_n, "MAE": mae_wh, "RMSE": rmse_wh}


# ──────────────────────────────────────────────────────────────
#  CSV save
# ──────────────────────────────────────────────────────────────

def save_csv(rows: list, path: str):
    """Robust to heterogeneous row dicts: uses the UNION of all keys as
    fieldnames so a row with extra fields can't crash the write."""
    import os, csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames, seen = [], set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k); seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  → saved {path}")


def print_table(rows: list, keys: list):
    header = "  ".join(f"{k:>14}" for k in keys)
    print("\n" + header)
    print("─" * len(header))
    for r in rows:
        print("  ".join(f"{str(r.get(k,''))[:14]:>14}" for k in keys))
