"""
src/exp_worst_day.py — Hard analysis: case study of the worst/best day.

Why this exists
---------------
Aggregate metrics (mean MAE over ~3800 test windows) hide everything that
matters in practice. To turn descriptive statistics into a falsifiable
narrative we:

  1. Compute per-window MAE for TC-DPMixer / DLinear / iTransformer on the
     test set (~16 days of contiguous 10-minute records).
  2. Group windows by the calendar day in which their forecast START falls.
  3. Identify the **worst** and **best** day (highest / lowest per-day mean
     MAE for our flagship TC-DPMixer).
  4. Visualise three windows from each (morning / noon / evening), overlay
     all three models' predictions on the ground truth, and pair with the
     hourly gate schedule g(t) that TC-DPMixer used that day.

This converts the report from "we report mean MAE = X" to "we report mean
MAE = X, and here is the one day where the gate (correctly / incorrectly)
flipped its branch trust because of [event]."

Outputs
-------
  results/worst_day.csv                                    per-day MAE per model
  results/figures/fig_worst_day.{png,pdf}                  the case-study figure
  results/log_worst_day.txt                                (when nohup'd)

Run
---
    python src/exp_worst_day.py
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders
from models import build_model
from training import train, collect_predictions, DEVICE


RESULTS = "results"
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

ARCHS = [
    ("dlinear",      {}),
    ("itransformer", {"d_model": 128, "n_layers": 2}),
    ("tcdpmixer",    {"time_feat_dim": 6}),
]


# ────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────

def recover_test_timestamps(cfg) -> pd.DatetimeIndex:
    """Re-load the raw CSV (no scaling) to recover the datetime index, then
    re-derive the test split boundary and the per-window forecast-start
    timestamps. Matches the slicing logic in `load_and_prepare`."""
    df = pd.read_csv(cfg.data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    if cfg.resample is not None:
        df = df.resample(cfg.resample).mean().dropna()
    n    = len(df)
    n_tr = int(n * cfg.train_ratio)
    n_vl = int(n * cfg.val_ratio)
    test_idx = df.index[n_tr + n_vl:]
    # Window i has forecast-start at test_idx[i + seq_len]; valid windows are
    # those with target end <= len(test_idx) - 1.
    n_win = len(test_idx) - cfg.seq_len - cfg.pred_len + 1
    starts = test_idx[cfg.seq_len : cfg.seq_len + n_win]
    return starts


@torch.no_grad()
def collect_tc_gates_per_window(tcdp, te_l) -> np.ndarray:
    """Return [N, 4] gate values (one row per test window)."""
    tcdp.eval(); tcdp.to(DEVICE)
    gs = []
    for x, _ in te_l:
        x = x.to(DEVICE)
        gs.append(tcdp._compute_gates(x).cpu())
    return torch.cat(gs).numpy()


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("HARD ANALYSIS — Worst-day / Best-day case study")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    cfg = make_config(weekly=False, feature_mode="no_rv",
                      add_time_features=True, log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    print(f"  n_feat={n_feat}   seq_len={cfg.seq_len}   pred_len={cfg.pred_len}")

    # Train one model per architecture.
    trained = {}
    preds = {}
    for name, kw in ARCHS:
        print(f"\n  [{name}] training…")
        t0 = time.time()
        m = build_model(name, cfg.seq_len, cfg.pred_len, n_feat,
                        use_residual=True, **kw)
        m = train(m, tr_l, vl_l, label=name, loss_name="l1")
        print(f"    trained in {time.time() - t0:.1f}s")
        trained[name] = m
        p, t = collect_predictions(m, te_l, scaler, log_target=cfg.log_target)
        preds[name] = p
        if "truth" not in preds:
            preds["truth"] = t

    truth = preds["truth"]
    N, H = truth.shape
    print(f"\n  Collected predictions: N={N} windows, H={H} steps each.")

    # Recover timestamps for each window.
    starts = recover_test_timestamps(cfg)
    assert len(starts) == N, (
        f"timestamp count {len(starts)} ≠ window count {N}; "
        f"check data_utils.make_windows logic.")
    days = pd.to_datetime(starts).date

    # Per-window MAE per model.
    mae = {name: np.abs(preds[name] - truth).mean(axis=1)
           for name in trained}

    # Per-day MAE table.
    df = pd.DataFrame({"day": days, **mae})
    daily = df.groupby("day").agg(["mean", "count"])
    daily.columns = [f"{m}_{stat}" for (m, stat) in daily.columns]
    # Reduce to a clean per-day mean.
    per_day = pd.DataFrame({m: df.groupby("day")[m].mean() for m in trained})
    per_day["n_windows"] = df.groupby("day")["dlinear"].count()
    per_day.to_csv(os.path.join(RESULTS, "worst_day.csv"))
    print(f"  -> per-day MAE saved to results/worst_day.csv  "
          f"({len(per_day)} days)")

    # Identify worst & best day for our flagship model (tcdpmixer).
    days_with_data = per_day[per_day["n_windows"] >= cfg.pred_len]
    flagship = "tcdpmixer"
    worst_day = days_with_data[flagship].idxmax()
    best_day  = days_with_data[flagship].idxmin()
    print(f"\n  Worst day for {flagship}:  {worst_day}  "
          f"(MAE = {per_day.loc[worst_day, flagship]:.2f} Wh)")
    print(f"  Best  day for {flagship}:  {best_day}   "
          f"(MAE = {per_day.loc[best_day,  flagship]:.2f} Wh)")
    print(f"  Per-model MAE on worst day:")
    for m in trained:
        print(f"    {m:13s} {per_day.loc[worst_day, m]:7.2f} Wh")
    print(f"  Per-model MAE on best day:")
    for m in trained:
        print(f"    {m:13s} {per_day.loc[best_day, m]:7.2f} Wh")

    # TC-DPMixer per-window gates (for g(t) panel).
    gates = collect_tc_gates_per_window(trained["tcdpmixer"], te_l)
    assert gates.shape == (N, 4), gates.shape

    _plot(worst_day, best_day, starts, truth, preds, gates, cfg)
    print("\nDone.")


def _pick_windows_for_day(starts: pd.DatetimeIndex, target_day) -> dict:
    """Return dict {hour_label: window_index} for ~3 windows that day
    (morning ~08, noon ~12, evening ~18)."""
    starts = pd.to_datetime(starts)
    in_day = np.where(starts.date == target_day)[0]
    if len(in_day) == 0:
        return {}
    out = {}
    for label, target_hour in [("morning 08:00", 8),
                               ("noon 12:00",   12),
                               ("evening 18:00", 18)]:
        # Find window whose start hour is closest to target_hour.
        in_day_hours = starts[in_day].hour
        best = in_day[int(np.argmin(np.abs(in_day_hours - target_hour)))]
        out[label] = best
    return out


def _plot(worst_day, best_day, starts, truth, preds, gates, cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    INK, AMBER, STEEL, GRAY = "#1F2A37", "#D99A2B", "#3F6CA8", "#7F8C9A"
    rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "savefig.bbox": "tight", "figure.facecolor": "white",
                     "axes.edgecolor": "#AEB8C2"})
    color_for = {"dlinear": GRAY, "itransformer": STEEL, "tcdpmixer": AMBER}

    # Plan: 2 columns (worst / best day) × 4 rows
    #   row 0: forecast at morning window
    #   row 1: forecast at noon window
    #   row 2: forecast at evening window
    #   row 3: TC-DPMixer 4-gate values across all windows that day
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 11.5))
    H = truth.shape[1]
    minutes_ahead = np.arange(1, H + 1) * 10   # 10-min cadence

    for col, (label, day) in enumerate([("Worst day", worst_day),
                                        ("Best day",  best_day)]):
        windows = _pick_windows_for_day(starts, day)
        # Top 3 panels: forecast comparisons at the 3 hand-picked windows.
        for row, (hour_label, wi) in enumerate(list(windows.items())[:3]):
            ax = axes[row, col]
            ax.plot(minutes_ahead, truth[wi], color="black", lw=2.2,
                    label="truth", zorder=5)
            for name in ["dlinear", "itransformer", "tcdpmixer"]:
                ax.plot(minutes_ahead, preds[name][wi],
                        color=color_for[name], lw=1.8,
                        marker="o", ms=3, label=name,
                        zorder=3 if name == "tcdpmixer" else 2)
            ax.set_title(f"{label}: {day}  ·  forecast start {hour_label}",
                         fontsize=11, weight="bold", color=INK)
            ax.set_xlabel("minutes ahead")
            ax.set_ylabel("Appliances (Wh)")
            ax.grid(True, color="#E3E8ED", lw=0.7); ax.set_axisbelow(True)
            if row == 0:
                ax.legend(fontsize=8, loc="upper right")
        # Bottom panel: TC-DPMixer 4 gate values across all windows in this day.
        ax = axes[3, col]
        in_day = np.where(pd.to_datetime(starts).date == day)[0]
        if len(in_day) > 0:
            start_hours = pd.to_datetime(starts[in_day]).hour + \
                          pd.to_datetime(starts[in_day]).minute / 60.0
            names = ["persistence", "decomp", "channel", "time"]
            for k, gname in enumerate(names):
                ax.plot(start_hours, gates[in_day, k], marker="o", ms=3,
                        lw=1.6, label=gname)
        ax.axhline(1.0, color="#D7DEE5", lw=1.0, ls="--")
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
        ax.set_title(f"TC-DPMixer gate schedule g(t)  on  {day}",
                     fontsize=11, weight="bold", color=INK)
        ax.set_xlabel("hour of day  (forecast start)")
        ax.set_ylabel("g(·) ∈ [0, 2]")
        ax.legend(fontsize=8, ncol=4, loc="upper right")
        ax.grid(True, color="#E3E8ED", lw=0.7); ax.set_axisbelow(True)

    fig.suptitle("Case study: worst vs best day for TC-DPMixer",
                 fontsize=14, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(FIG_DIR, "fig_worst_day.png")
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
