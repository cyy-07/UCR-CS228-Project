"""
src/run_experiments.py  —  Training, evaluation, and all experiments.

Usage
-----
    python src/run_experiments.py              # run all experiments
    python src/run_experiments.py --exp baseline
    python src/run_experiments.py --exp scarcity
    python src/run_experiments.py --exp corruption
    python src/run_experiments.py --exp feature

Experiments
-----------
  baseline    → 5 models, full data, no_rv features
  scarcity    → 4 models × {100,50,20,10}% training data
  corruption  → 4 models × {clean, noise-0.1, noise-0.3, mask-20%, mask-40%}
  feature     → 5 models × {no_rv, all}   (Feature Irrelevance study)

Results saved to  results/  as CSV files + one summary PNG.
"""

import os, sys, time, argparse, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import Config, get_loaders, add_noise, random_mask
from models import build_model

# ──────────────────────────────────────────────────────────────
#  Hyper-parameters  (edit here or via Config)
# ──────────────────────────────────────────────────────────────
EPOCHS    = 50
LR        = 1e-3
PATIENCE  = 10
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS   = "results"
os.makedirs(RESULTS, exist_ok=True)  # ensure dir exists before any log redirect

ALL_MODELS = ["persistence", "mlp", "lstm", "dlinear", "patchtst"]
DEEP_MODELS = ["mlp", "lstm", "dlinear", "patchtst"]   # persistence needs no training


# ──────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          label: str = "") -> nn.Module:
    """Adam + ReduceLROnPlateau + early stopping. Returns best model."""
    if not list(model.parameters()):   # PersistenceBaseline
        return model

    model.to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    best_val, best_state, wait = float("inf"), None, 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        # train
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += loss_fn(model(x), y).item() * len(x)
        val_loss /= len(val_loader.dataset)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val, best_state, wait = val_loss, {k: v.cpu().clone()
                for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

        if ep % 10 == 0 or ep == 1:
            print(f"    ep{ep:3d}  val_mse={val_loss:.5f}  "
                  f"[{label}  {time.time()-t0:.0f}s]")

    model.load_state_dict(best_state)
    return model


# ──────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader,
             scaler, corrupt_fn=None) -> dict:
    """
    Run inference (optionally with corruption), return MAE & RMSE
    in original Wh units (inverse-transformed).
    """
    model.eval()
    model.to(DEVICE)
    preds, trues = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            if corrupt_fn is not None:
                x = corrupt_fn(x)
            preds.append(model(x).cpu())
            trues.append(y)

    preds = torch.cat(preds).numpy()   # (N, H)
    trues = torch.cat(trues).numpy()

    # inverse-transform target only (col 0)
    sc, mn = scaler.scale_[0], scaler.mean_[0]
    preds = preds * sc + mn
    trues = trues * sc + mn

    mae  = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    return {"MAE": mae, "RMSE": rmse}


# ──────────────────────────────────────────────────────────────
#  Save / print utilities
# ──────────────────────────────────────────────────────────────

def save_csv(rows: list, filename: str):
    path = os.path.join(RESULTS, filename)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  → saved {path}")


def print_table(rows: list, keys: list):
    header = "  ".join(f"{k:>16}" for k in keys)
    print("\n" + header)
    print("─" * len(header))
    for r in rows:
        print("  ".join(f"{str(r.get(k,''))[:16]:>16}" for k in keys))


# ──────────────────────────────────────────────────────────────
#  Experiment 1 — Baseline
# ──────────────────────────────────────────────────────────────

def exp_baseline():
    print("\n" + "="*55)
    print("EXP 1 — Full-data baseline  (feature_mode=no_rv)")
    print("="*55)
    cfg = Config()
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    rows = []
    for name in ALL_MODELS:
        print(f"\n  [{name}]")
        model = build_model(name, cfg.seq_len, cfg.pred_len, n_feat)
        t0 = time.time()
        model = train(model, tr_l, vl_l, label=name)
        m = evaluate(model, te_l, scaler)
        npar = sum(p.numel() for p in model.parameters())
        rows.append({"model": name, **m, "params": npar,
                     "train_s": round(time.time()-t0,1)})
        print(f"    MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
              f"params={npar:,}  time={rows[-1]['train_s']}s")
    save_csv(rows, "baseline.csv")
    print_table(rows, ["model","MAE","RMSE","params","train_s"])
    return rows


# ──────────────────────────────────────────────────────────────
#  Experiment 2 — Data scarcity
# ──────────────────────────────────────────────────────────────

def exp_scarcity(ratios=(1.0, 0.5, 0.2, 0.1)):
    print("\n" + "="*55)
    print("EXP 2 — Data scarcity")
    print("="*55)
    rows = []
    # load fixed test/val once at ratio=1.0
    cfg0 = Config()
    _, _, te_l, scaler, n_feat = get_loaders(cfg0)

    for ratio in ratios:
        cfg = Config(scarcity_ratio=ratio)
        tr_l, vl_l, *_ = get_loaders(cfg)
        print(f"\n  ratio={ratio:.0%}  train_win={len(tr_l.dataset):,}")
        for name in DEEP_MODELS:
            model = build_model(name, cfg.seq_len, cfg.pred_len, n_feat)
            model = train(model, tr_l, vl_l, label=f"{name}@{ratio:.0%}")
            m = evaluate(model, te_l, scaler)
            rows.append({"model": name, "ratio": ratio, **m})
            print(f"    {name}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}")

    # persistence reference (constant regardless of ratio)
    pm = build_model("persistence", cfg0.seq_len, cfg0.pred_len, n_feat)
    m0 = evaluate(pm, te_l, scaler)
    for ratio in ratios:
        rows.append({"model": "persistence", "ratio": ratio, **m0})

    save_csv(rows, "scarcity.csv")
    print_table(rows, ["model","ratio","MAE","RMSE"])
    return rows


# ──────────────────────────────────────────────────────────────
#  Experiment 3 — Input corruption
# ──────────────────────────────────────────────────────────────

def exp_corruption():
    print("\n" + "="*55)
    print("EXP 3 — Input corruption  (models trained on clean data)")
    print("="*55)
    cfg = Config()
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)

    # train once on clean data
    trained = {}
    for name in DEEP_MODELS:
        print(f"\n  training {name}...")
        m = build_model(name, cfg.seq_len, cfg.pred_len, n_feat)
        trained[name] = train(m, tr_l, vl_l, label=name)
    trained["persistence"] = build_model(
        "persistence", cfg.seq_len, cfg.pred_len, n_feat)

    conditions = {
        "clean":       None,
        "noise_0.1":   lambda x: add_noise(x, 0.1),
        "noise_0.3":   lambda x: add_noise(x, 0.3),
        "mask_20pct":  lambda x: random_mask(x, 0.2),
        "mask_40pct":  lambda x: random_mask(x, 0.4),
    }

    rows = []
    for cname, fn in conditions.items():
        for name, model in trained.items():
            m = evaluate(model, te_l, scaler, corrupt_fn=fn)
            rows.append({"model": name, "condition": cname, **m})
            print(f"  {name:<14} {cname:<14} MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}")

    save_csv(rows, "corruption.csv")
    print_table(rows, ["model","condition","MAE","RMSE"])
    return rows


# ──────────────────────────────────────────────────────────────
#  Experiment 4 — Feature irrelevance  (rv1/rv2 study)
# ──────────────────────────────────────────────────────────────

def exp_feature():
    print("\n" + "="*55)
    print("EXP 4 — Feature irrelevance  (no_rv vs all)")
    print("="*55)
    rows = []
    for fmode in ["no_rv", "all"]:
        cfg = Config(feature_mode=fmode)
        tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
        print(f"\n  feature_mode={fmode}  n_feat={n_feat}")
        for name in ALL_MODELS:
            model = build_model(name, cfg.seq_len, cfg.pred_len, n_feat)
            model = train(model, tr_l, vl_l, label=f"{name}[{fmode}]")
            m = evaluate(model, te_l, scaler)
            rows.append({"model": name, "feature_mode": fmode, **m})
            print(f"    {name}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}")

    save_csv(rows, "feature.csv")
    print_table(rows, ["model","feature_mode","MAE","RMSE"])
    return rows


# ──────────────────────────────────────────────────────────────
#  Plot (optional, one summary figure)
# ──────────────────────────────────────────────────────────────

def plot_scarcity(csv_path="results/scarcity.csv"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, grp in df.groupby("model"):
            grp = grp.sort_values("ratio")
            ax.plot(grp["ratio"] * 100, grp["MAE"],
                    marker="o", label=name.upper())
        ax.set_xlabel("Training data fraction (%)")
        ax.set_ylabel("MAE (Wh)")
        ax.set_title("MAE vs Training Data Fraction")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        out = "results/scarcity_mae.png"
        fig.savefig(out, dpi=120); plt.close()
        print(f"  → saved {out}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all",
        choices=["all","baseline","scarcity","corruption","feature"])
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    torch.manual_seed(42); np.random.seed(42)

    if args.exp in ("all", "baseline"):   exp_baseline()
    if args.exp in ("all", "scarcity"):   exp_scarcity(); plot_scarcity()
    if args.exp in ("all", "corruption"): exp_corruption()
    if args.exp in ("all", "feature"):    exp_feature()

    print("\nDone. Results in ./results/")
