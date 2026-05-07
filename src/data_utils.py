"""
src/data_utils.py  —  Data loading, preprocessing, windowing, corruption.

TENSOR SHAPE CONVENTION (enforced throughout this project):
    x : [B, L, C]   B=batch,  L=seq_len,  C=num_features
    y : [B, H]       B=batch,  H=pred_len  (target = Appliances only)

All C input features include historical Appliances values at index 0,
so the model can exploit its own past values.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

@dataclass
class Config:
    # paths
    data_path : str   = "data/energydata_complete.csv"
    target_col: str   = "Appliances"

    # windowing
    seq_len   : int   = 96     # 16 h look-back  (10-min ticks)
    pred_len  : int   = 24     # 4 h forecast

    # chronological split  (test = 1 - train - val)
    train_ratio: float = 0.70
    val_ratio  : float = 0.10

    # loader
    batch_size : int   = 256
    seed       : int   = 42

    # feature mode
    # "all"   → keep every column incl. rv1/rv2 as input features
    # "no_rv" → drop rv1, rv2 from input features
    feature_mode: str  = "no_rv"

    # data scarcity  (fraction of training windows to keep)
    scarcity_ratio: float = 1.0

    # corruption (applied at inference time by caller)
    noise_sigma : float = 0.1   # Gaussian std on normalised scale
    mask_ratio  : float = 0.20  # fraction of non-target cols zeroed


# ─────────────────────────────────────────────
#  Feature column selection
# ─────────────────────────────────────────────

def _feature_cols(df: pd.DataFrame, cfg: Config) -> list:
    """Return ordered list of input-feature column names (target always first)."""
    drop = {cfg.target_col}
    if cfg.feature_mode == "no_rv":
        drop |= {"rv1", "rv2"}
    elif cfg.feature_mode != "all":
        raise ValueError(f"feature_mode must be 'all' or 'no_rv', got '{cfg.feature_mode}'")
    return [c for c in df.columns if c not in drop]


# ─────────────────────────────────────────────
#  Load → split → scale
# ─────────────────────────────────────────────

def load_and_prepare(cfg: Config):
    """
    Load CSV, split chronologically, fit StandardScaler on train only.

    Returns
    -------
    train_arr, val_arr, test_arr : np.ndarray  shape (N, C)
        C = 1 (target) + len(feature_cols)
    scaler  : fitted StandardScaler
    n_feat  : int  — C (used to build models)
    """
    df = pd.read_csv(cfg.data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    feat_cols = _feature_cols(df, cfg)
    # target at column 0, then all selected features
    all_cols  = [cfg.target_col] + feat_cols

    data = df[all_cols].values.astype(np.float32)  # (N, C)
    n    = len(data)
    n_tr = int(n * cfg.train_ratio)
    n_vl = int(n * cfg.val_ratio)

    train = data[:n_tr]
    val   = data[n_tr : n_tr + n_vl]
    test  = data[n_tr + n_vl :]

    scaler = StandardScaler().fit(train)
    return (scaler.transform(train),
            scaler.transform(val),
            scaler.transform(test),
            scaler,
            data.shape[1])          # n_feat = C


# ─────────────────────────────────────────────
#  Sliding windows
# ─────────────────────────────────────────────

def make_windows(arr: np.ndarray, seq_len: int, pred_len: int):
    """
    Sliding-window sample generation.

    Target column = index 0 (Appliances).

    Returns
    -------
    X : float32  (S, seq_len, C)   — x: [B, L, C]
    y : float32  (S, pred_len)     — y: [B, H]
    """
    X, y = [], []
    for i in range(len(arr) - seq_len - pred_len + 1):
        X.append(arr[i : i + seq_len])
        y.append(arr[i + seq_len : i + seq_len + pred_len, 0])  # target=col 0
    return np.array(X, np.float32), np.array(y, np.float32)


# ─────────────────────────────────────────────
#  Dataset & DataLoader
# ─────────────────────────────────────────────

class TSDataset(Dataset):
    """
    Windowed time-series dataset.
    x : [L, C]  → batched [B, L, C]
    y : [H]     → batched [B, H]
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def get_loaders(cfg: Config):
    """
    Full pipeline → (train_loader, val_loader, test_loader, scaler, n_feat).

    Val and test sets are NEVER subsampled — only training windows are.
    """
    tr, vl, te, scaler, n_feat = load_and_prepare(cfg)

    Xtr, ytr = make_windows(tr, cfg.seq_len, cfg.pred_len)
    Xvl, yvl = make_windows(vl, cfg.seq_len, cfg.pred_len)
    Xte, yte = make_windows(te, cfg.seq_len, cfg.pred_len)

    # data scarcity: keep first k% of training windows (chronological)
    if cfg.scarcity_ratio < 1.0:
        k = max(1, int(len(Xtr) * cfg.scarcity_ratio))
        Xtr, ytr = Xtr[:k], ytr[:k]

    make = lambda X, y, shuf: DataLoader(
        TSDataset(X, y), batch_size=cfg.batch_size,
        shuffle=shuf, drop_last=False)

    return (make(Xtr, ytr, True),
            make(Xvl, yvl, False),
            make(Xte, yte, False),
            scaler, n_feat)


# ─────────────────────────────────────────────
#  Corruption helpers  (called at inference time)
# ─────────────────────────────────────────────

def add_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Additive Gaussian noise on normalised input.  x: [B,L,C]"""
    return x + torch.randn_like(x) * sigma


def random_mask(x: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Zero-out `ratio` fraction of non-target feature columns per sample.
    Column 0 (Appliances history) is never masked.
    x: [B, L, C]
    """
    B, L, C = x.shape
    x = x.clone()
    n_mask = max(1, int(ratio * (C - 1)))
    for i in range(B):
        cols = torch.randperm(C - 1)[:n_mask] + 1   # skip col 0
        x[i, :, cols] = 0.0
    return x


# ─────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    print(f"feature_mode = {cfg.feature_mode}")

    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    print(f"train={len(tr_l.dataset):,}  val={len(vl_l.dataset):,}  "
          f"test={len(te_l.dataset):,}  n_feat={n_feat}")

    x, y = next(iter(tr_l))
    print(f"x {tuple(x.shape)}  y {tuple(y.shape)}")
    assert x.shape == (cfg.batch_size, cfg.seq_len, n_feat)
    assert y.shape == (cfg.batch_size, cfg.pred_len)

    # corruption
    assert add_noise(x, 0.1).shape == x.shape
    assert random_mask(x, 0.2).shape == x.shape

    # scarcity
    cfg2 = Config(scarcity_ratio=0.1)
    tr2, *_ = get_loaders(cfg2)
    print(f"scarcity 10% → train={len(tr2.dataset):,}")

    # feature mode comparison
    cfg_all = Config(feature_mode="all")
    *_, n_all = get_loaders(cfg_all)
    print(f"no_rv={n_feat} feats  all={n_all} feats  diff={n_all-n_feat} (rv1+rv2)")

    print("\n✓ data_utils OK")
