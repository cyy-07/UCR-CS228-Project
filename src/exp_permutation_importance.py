"""
src/exp_permutation_importance.py — Hard analysis: permutation feature importance.

Why this exists
---------------
The existing `feature_mode` ablation in `exp_features.py` answers
"what happens if I REMOVE this feature group from training?"

That is informative but coarse (groups of features) and confounded with
retraining stochasticity (each ablation trains a different model).

Permutation importance answers the complementary question:
"For a SINGLE trained model, how much does each individual channel
 contribute to its actual predictions?"
We do this by destroying the predictive value of one channel at a time
(without retraining) and measuring the drop in test MAE.

Method
------
  1. Train TC-DPMixer once on the canonical setup.
  2. Collect baseline forecasts on the un-shuffled test set.
  3. For each input channel c ∈ {0..C-1}:
       a. Load the entire test tensor X ∈ ℝ^[N, L, C] into memory.
       b. Construct X_perm by replacing column c with a random
          permutation across the N windows
          (i.e. X_perm[:, :, c] = X[π, :, c], π a uniform permutation).
          This breaks the JOINT signal between channel c and the
          target while preserving the marginal distribution of c.
       c. Re-run inference and compute test MAE.
       d. Importance[c] = MAE_perm[c] − MAE_baseline.
  4. Sort and plot.

For TC-DPMixer's input layout (no_rv + add_time_features) the 32 channels
break down as:
  [0]   Appliances history  (TARGET — should be the most important!)
  [1]   lights
  [2-9] T1..T9        indoor temperatures
  [10-18] RH_1..RH_9  indoor humidities
  [19]   (Press_mm_hg is at 21; sequencing depends on dataframe order)
  ... (see data_utils for exact order)
  [20]  T_out               outdoor temperature
  [21-25] other outdoor weather variables
  [26-31] time features (hour/weekday sin/cos, is_weekend, nsm)

Outputs
-------
  results/permutation_importance.csv                     channel × MAE drop
  results/figures/fig_permutation_importance.{png,pdf}   sorted bar chart

Run
---
    python src/exp_permutation_importance.py
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders, INDOOR, OUTDOOR, LIGHTS, TIME
from models import build_model
from training import train, DEVICE


RESULTS = "results"
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────
#  Recover channel names in the same order data_utils builds them.
# ────────────────────────────────────────────────────────────────

def reconstruct_channel_names(cfg) -> list:
    """Re-derive the column-name list exactly as load_and_prepare() does, so
    we can label every permutation result with a human-readable name."""
    df = pd.read_csv(cfg.data_path)
    cols_all = list(df.columns)
    # data_utils._select_feature_cols for 'no_rv' simply drops rv1/rv2 and
    # keeps every OTHER column ordered as they appear in the CSV.
    rv = ["rv1", "rv2"]
    target = cfg.target_col
    feat_cols = [c for c in cols_all if c != target and c not in rv
                 and c != "date"]
    if cfg.add_time_features:
        feat_cols = list(feat_cols) + [t for t in TIME if t not in feat_cols]
    return [target] + feat_cols


# ────────────────────────────────────────────────────────────────
#  Collect test tensor + truths from the un-shuffled loader.
# ────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_test_tensors(loader):
    Xs, Ys = [], []
    for x, y in loader:
        Xs.append(x); Ys.append(y)
    return torch.cat(Xs), torch.cat(Ys)


@torch.no_grad()
def forward_in_batches(model, X, batch_size=256):
    model.eval(); model.to(DEVICE)
    out = []
    for i in range(0, X.size(0), batch_size):
        chunk = X[i:i + batch_size].to(DEVICE)
        p = model(chunk)
        if p.dim() == 3:
            p = p[..., p.size(-1) // 2]
        out.append(p.cpu())
    return torch.cat(out)


def inverse_to_wh(arr: np.ndarray, scaler, log_target: bool) -> np.ndarray:
    sc, mn = float(scaler.scale_[0]), float(scaler.mean_[0])
    out = arr * sc + mn
    return np.expm1(out) if log_target else out


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("HARD ANALYSIS — Permutation feature importance (TC-DPMixer)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    cfg = make_config(weekly=False, feature_mode="no_rv",
                      add_time_features=True, log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    print(f"  n_feat={n_feat}   seq_len={cfg.seq_len}   pred_len={cfg.pred_len}")

    # Channel name labels.
    names = reconstruct_channel_names(cfg)
    if len(names) != n_feat:
        print(f"  WARNING: derived {len(names)} names but n_feat={n_feat}; "
              "labelling will be approximate.")
        # Pad/truncate to align.
        names = (names + [f"ch{k}" for k in range(n_feat)])[:n_feat]

    # Train TC-DPMixer once.
    print(f"\n  [tcdpmixer] training…")
    t0 = time.time()
    model = build_model("tcdpmixer", cfg.seq_len, cfg.pred_len, n_feat,
                        use_residual=True, time_feat_dim=6)
    model = train(model, tr_l, vl_l, label="tcdpmixer", loss_name="l1")
    print(f"    trained in {time.time() - t0:.1f}s")

    # Test tensor + ground truth in Wh.
    print("  collecting test tensor…")
    X_te, Y_te = collect_test_tensors(te_l)
    print(f"    X_te.shape = {tuple(X_te.shape)}")
    truth_wh = inverse_to_wh(Y_te.numpy(), scaler, cfg.log_target)

    # Baseline MAE.
    base_n = forward_in_batches(model, X_te).numpy()
    base_wh = inverse_to_wh(base_n, scaler, cfg.log_target)
    base_mae = float(np.abs(base_wh - truth_wh).mean())
    print(f"  baseline test MAE = {base_mae:.3f} Wh")

    # Permute each channel and recompute.
    rng = np.random.default_rng(42)
    N = X_te.size(0)
    rows = []
    print("\n  ── per-channel permutation ──")
    for c in range(n_feat):
        perm = rng.permutation(N)
        Xp = X_te.clone()
        Xp[:, :, c] = X_te[perm, :, c]
        pred_n = forward_in_batches(model, Xp).numpy()
        pred_wh = inverse_to_wh(pred_n, scaler, cfg.log_target)
        mae = float(np.abs(pred_wh - truth_wh).mean())
        delta = mae - base_mae
        rows.append({"channel": c, "name": names[c],
                     "MAE_perm": round(mae, 3),
                     "MAE_baseline": round(base_mae, 3),
                     "delta_MAE":  round(delta, 3),
                     "pct_increase": round(100 * delta / base_mae, 2)})
        print(f"    ch{c:2d} {names[c]:14s}  "
              f"perm MAE = {mae:7.3f}   Δ = {delta:+6.3f}")

    rows.sort(key=lambda r: -r["delta_MAE"])
    from training import save_csv
    save_csv(rows, os.path.join(RESULTS, "permutation_importance.csv"))

    _plot(rows, base_mae)
    print("\nDone.")


def _plot(rows, base_mae):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    INK, AMBER = "#1F2A37", "#D99A2B"
    rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "savefig.bbox": "tight", "figure.facecolor": "white",
                     "axes.edgecolor": "#AEB8C2"})

    # Show top-K by absolute delta.
    K = min(20, len(rows))
    top = rows[:K]
    names = [f"{r['name']}  (ch{r['channel']})" for r in top]
    vals  = [r["delta_MAE"] for r in top]

    # Color: amber if delta > 0 (informative channel), gray if ≈ 0 or negative.
    colors = [AMBER if v > 0.1 else "#94A3B0" for v in vals]

    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.34 * K)))
    y_pos = np.arange(K)
    ax.barh(y_pos, vals, color=colors, edgecolor="#AEB8C2", lw=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#D7DEE5", lw=1.0, ls="--")
    ax.set_xlabel("Δ test MAE (Wh) when this channel is permuted   "
                  "[higher = more important]")
    ax.set_title(f"Permutation importance · TC-DPMixer\n"
                 f"baseline test MAE = {base_mae:.2f} Wh  ·  "
                 f"top {K} channels",
                 fontsize=12, weight="bold", color=INK)
    ax.grid(True, color="#E3E8ED", lw=0.7, axis="x"); ax.set_axisbelow(True)
    fig.tight_layout()
    png = os.path.join(FIG_DIR, "fig_permutation_importance.png")
    fig.savefig(png); fig.savefig(png.replace(".png", ".pdf"))
    plt.close()
    print(f"  -> saved {png}  (+ .pdf)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true",
                    help="(ignored — scope-locked to short task)")
    args = ap.parse_args()
    if args.weekly:
        print("Warning: --weekly is currently a no-op (scope-locked).")
    torch.manual_seed(42); np.random.seed(42)
    main()
