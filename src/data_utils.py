"""
src/data_utils.py — Data loading, preprocessing, windowing, corruption.

TENSOR SHAPE CONVENTION:
    x : [B, L, C]   (B=batch, L=seq_len, C=num_features)
    y : [B, H]      (B=batch, H=pred_len, target = Appliances only)

All C input features include historical Appliances at index 0.
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
    data_path : str   = "data/energydata_complete.csv"
    target_col: str   = "Appliances"

    # windowing
    seq_len   : int   = 96     # 16 h look-back  (10-min ticks)
    pred_len  : int   = 24     # 4 h forecast

    # temporal resampling
    #   None  → native 10-min cadence (default; original short-horizon task)
    #   "1h"  → hourly aggregation     (long-horizon weekly task)
    #           Appliances/lights summed (= energy per period); all other
    #           covariates averaged. After resampling, 1 step = 1 hour, so
    #           seq_len=168/pred_len=168 means "past 1 week → next 1 week".
    resample  : str | None = None

    # chronological split  (test = 1 - train - val)
    train_ratio: float = 0.70
    val_ratio  : float = 0.10

    # loader
    batch_size : int   = 256
    seed       : int   = 42

    # feature subset selection
    #   "all"           → every column incl. rv1/rv2
    #   "no_rv"         → drop rv1, rv2  (default; matches literature)
    #   "target_only"   → only the target's own history
    #   "target_lights" → target + lights
    #   "target_indoor" → target + T1..T9 + RH_1..RH_9
    #   "target_outdoor"→ target + outdoor weather
    #   "target_time"   → target + time encodings only
    feature_mode: str  = "no_rv"

    # add time-of-day / weekday encodings (Workstream A)
    add_time_features: bool = False

    # log1p-transform the Appliances target (Workstream A)
    log_target: bool = False

    # data scarcity (fraction of training windows kept; chronological)
    scarcity_ratio: float = 1.0

    # corruption hyperparameters (applied at inference time by caller)
    noise_sigma : float = 0.1
    mask_ratio  : float = 0.20


# ─────────────────────────────────────────────
#  Weekly-task config preset
# ─────────────────────────────────────────────

#  Settings that turn ANY workstream into its long-horizon WEEKLY version:
#  aggregate to hourly, look back 1 week (168 h), forecast 1 week (168 h).
#  val_ratio is raised to 0.15 because hourly resampling leaves only ~3,288
#  rows — too few for 168+168-step windows to fit a 10% val slice.
WEEKLY_PRESET = dict(resample="1h", seq_len=168, pred_len=168,
                     train_ratio=0.70, val_ratio=0.15)


def make_config(weekly: bool = False, **kwargs) -> "Config":
    """Build a Config; if `weekly`, inject the WEEKLY_PRESET first so every
    experiment can switch to the long-horizon task with a single flag.
    Caller kwargs win, so e.g. exp_horizon can still override pred_len."""
    if weekly:
        base = dict(WEEKLY_PRESET)
        base.update(kwargs)
        return Config(**base)
    return Config(**kwargs)


# ─────────────────────────────────────────────
#  Feature column groups
# ─────────────────────────────────────────────

INDOOR  = [f"T{i}" for i in range(1, 10)] + [f"RH_{i}" for i in range(1, 10)]
OUTDOOR = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
RV      = ["rv1", "rv2"]
LIGHTS  = ["lights"]
TIME    = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
           "is_weekend", "nsm"]


def _select_feature_cols(df: pd.DataFrame, cfg: Config) -> list:
    """Return the ordered list of input-feature columns (target excluded)."""
    mode = cfg.feature_mode
    if   mode == "all":
        cols = [c for c in df.columns if c != cfg.target_col]
    elif mode == "no_rv":
        cols = [c for c in df.columns if c != cfg.target_col and c not in RV]
    elif mode == "target_only":
        cols = []
    elif mode == "target_lights":
        cols = [c for c in LIGHTS if c in df.columns]
    elif mode == "target_indoor":
        cols = [c for c in INDOOR if c in df.columns]
    elif mode == "target_outdoor":
        cols = [c for c in OUTDOOR if c in df.columns]
    elif mode == "target_time":
        cols = []   # time features added separately if add_time_features
    else:
        raise ValueError(f"Unknown feature_mode '{mode}'")
    return cols


# ─────────────────────────────────────────────
#  Time-of-day / weekday feature engineering
# ─────────────────────────────────────────────

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inject deterministic time encodings as columns (uses df.index = datetime)."""
    idx = df.index
    hour    = idx.hour + idx.minute / 60.0          # 0..24
    weekday = idx.weekday                            # 0..6
    df = df.copy()
    df["hour_sin"]    = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"]    = np.cos(2 * np.pi * hour / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    df["is_weekend"]  = (weekday >= 5).astype(np.float32)
    # NSM = number of seconds from midnight (popular in literature)
    df["nsm"] = (idx.hour * 3600 + idx.minute * 60).astype(np.float32)
    return df


# ─────────────────────────────────────────────
#  Temporal resampling (10-min → coarser cadence)
# ─────────────────────────────────────────────

def _resample(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Aggregate the 10-min series to a coarser cadence (cfg.resample, e.g. "1H").

    Energy-like columns (Appliances, lights) are SUMMED → total energy per
    period. Every other covariate (temperatures, humidities, weather, rv1/rv2)
    is AVERAGED. This makes one step a full hour, so seq_len/pred_len are now
    counted in hours instead of 10-min ticks.
    """
    sum_cols = [c for c in ("Appliances", "lights") if c in df.columns]
    mean_cols = [c for c in df.columns if c not in sum_cols]
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "mean" for c in mean_cols})
    out = df.resample(cfg.resample).agg(agg)
    return out.dropna()


# ─────────────────────────────────────────────
#  Load → split → scale
# ─────────────────────────────────────────────

def load_and_prepare(cfg: Config):
    """
    Load CSV, optionally add time features and log-transform target,
    chronologically split, fit StandardScaler on TRAIN ONLY.
    Returns (train_arr, val_arr, test_arr, scaler, n_feat).
    """
    df = pd.read_csv(cfg.data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    if cfg.resample is not None:
        df = _resample(df, cfg)

    if cfg.add_time_features or cfg.feature_mode == "target_time":
        df = _add_time_features(df)

    # log1p transform of target (kept in same column for downstream uniformity)
    if cfg.log_target:
        df[cfg.target_col] = np.log1p(df[cfg.target_col].clip(lower=0))

    feat_cols = _select_feature_cols(df, cfg)
    if cfg.add_time_features or cfg.feature_mode == "target_time":
        feat_cols = list(feat_cols) + [c for c in TIME if c in df.columns
                                       and c not in feat_cols]

    all_cols = [cfg.target_col] + feat_cols
    data = df[all_cols].values.astype(np.float32)
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
            data.shape[1])


# ─────────────────────────────────────────────
#  Sliding windows
# ─────────────────────────────────────────────

def make_windows(arr: np.ndarray, seq_len: int, pred_len: int):
    X, y = [], []
    for i in range(len(arr) - seq_len - pred_len + 1):
        X.append(arr[i : i + seq_len])
        y.append(arr[i + seq_len : i + seq_len + pred_len, 0])
    return np.array(X, np.float32), np.array(y, np.float32)


# ─────────────────────────────────────────────
#  Dataset & DataLoader
# ─────────────────────────────────────────────

class TSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def get_loaders(cfg: Config):
    """Returns (train_loader, val_loader, test_loader, scaler, n_feat)."""
    tr, vl, te, scaler, n_feat = load_and_prepare(cfg)
    Xtr, ytr = make_windows(tr, cfg.seq_len, cfg.pred_len)
    Xvl, yvl = make_windows(vl, cfg.seq_len, cfg.pred_len)
    Xte, yte = make_windows(te, cfg.seq_len, cfg.pred_len)

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
#  Corruption helpers (called at inference time)
# ─────────────────────────────────────────────

def add_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def random_mask(x: torch.Tensor, ratio: float) -> torch.Tensor:
    B, L, C = x.shape
    x = x.clone()
    n_mask = max(1, int(ratio * (C - 1)))
    for i in range(B):
        cols = torch.randperm(C - 1)[:n_mask] + 1
        x[i, :, cols] = 0.0
    return x


# ─────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    for fmode in ["target_only", "target_lights", "target_indoor",
                  "target_outdoor", "target_time", "no_rv", "all"]:
        cfg = Config(feature_mode=fmode, add_time_features=(fmode != "no_rv"))
        tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
        x, y = next(iter(tr_l))
        print(f"  feature_mode={fmode:<16} n_feat={n_feat:2d}  "
              f"x{tuple(x.shape)} y{tuple(y.shape)}")

    # log-target sanity
    cfg = Config(log_target=True)
    tr_l, *_ = get_loaders(cfg)
    x, y = next(iter(tr_l))
    print(f"\n  log_target: y range = [{y.min():.3f}, {y.max():.3f}]  "
          f"(should look log-compressed)")

    print("\n✓ data_utils OK")
