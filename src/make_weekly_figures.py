"""
src/make_weekly_figures.py — Figures for the long-horizon WEEKLY task.

Produces three figures (UCR blue/gold theme), reusing the style from
make_figures.py:

  fig_weekly_compare.png   — test MAE per model, baselines highlighted
  fig_weekly_perday.png    — MAE growth across the 7-day horizon (day 1..7)
  fig_weekly_forecast.png  — one example test week: truth vs Seasonal-Naive
                             vs our TC-SS-DPMixer (the intuitive money plot)

Run
---
    python src/make_weekly_figures.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(__file__))

UCR_BLUE, UCR_GOLD, UCR_NAVY, UCR_GRAY = "#003DA5", "#FFB81C", "#001F4E", "#54585A"
LIGHT_BG = "#F5F5F5"
PALETTE = [UCR_BLUE, UCR_GOLD, "#7A9CC6", UCR_NAVY, "#E07A5F", UCR_GRAY, "#81B29A"]

rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": UCR_NAVY, "axes.linewidth": 1.2, "axes.grid": True,
    "grid.color": "#D0D4DA", "grid.linestyle": "--", "grid.linewidth": 0.6,
    "grid.alpha": 0.7, "xtick.color": UCR_NAVY, "ytick.color": UCR_NAVY,
    "legend.frameon": True, "legend.fontsize": 9, "savefig.dpi": 160,
    "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": LIGHT_BG,
})

ROOT, FIG_DIR = "results", os.path.join("results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
PERIOD = 168


def _footer(fig):
    fig.subplots_adjust(bottom=0.16)
    ax = fig.add_axes([0, 0, 1, 0.012])
    ax.axhspan(0, 0.5, facecolor=UCR_BLUE); ax.axhspan(0.5, 1, facecolor=UCR_GOLD)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    fig.text(0.99, 0.018, "UCR CS228 — Weekly Energy Forecasting",
             ha="right", va="bottom", fontsize=8, color=UCR_NAVY, alpha=0.7)


def _save(fig, name):
    out = os.path.join(FIG_DIR, name)
    _footer(fig); fig.savefig(out); plt.close(fig)
    print(f"  → {out}")


BASELINES = {"persistence", "seasonalnaive"}
NICE = {"persistence": "Persistence", "seasonalnaive": "Seasonal-Naive\n(last week)",
        "mlp": "MLP", "lstm": "LSTM", "dlinear": "DLinear", "patchtst": "PatchTST",
        "itransformer": "iTransformer", "dpmixer": "DPMixer",
        "tcdpmixer": "TC-DPMixer", "ssdpmixer": "SS-DPMixer",
        "tcssdpmixer": "TC-SS-DPMixer\n(ours)"}


def fig_compare():
    df = pd.read_csv(os.path.join(ROOT, "weekly.csv")).sort_values("test_MAE")
    labels = [NICE.get(c, c) for c in df["config"]]
    colors = [UCR_GOLD if c in BASELINES else UCR_BLUE for c in df["config"]]
    # highlight our headline model in navy
    colors = [UCR_NAVY if c == "tcssdpmixer" else col
              for c, col in zip(df["config"], colors)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(df))
    bars = ax.bar(x, df["test_MAE"], color=colors, edgecolor=UCR_NAVY, width=0.7)
    sn = float(df.loc[df["config"] == "seasonalnaive", "test_MAE"].iloc[0])
    ax.axhline(sn, ls="--", color="red", lw=1.3,
               label=f"Seasonal-Naive baseline = {sn:.0f} Wh/h")
    for b, v in zip(bars, df["test_MAE"]):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}",
                ha="center", va="bottom", fontsize=8.5, color=UCR_NAVY)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Test MAE (Wh per hour)")
    ax.set_title("Weekly task: past 1 week → next 1 week (hourly)")
    ax.legend(loc="upper left")
    _save(fig, "fig_weekly_compare.png")


def fig_perday():
    df = pd.read_csv(os.path.join(ROOT, "weekly_perday.csv"))
    days = np.arange(1, 8)
    show = ["seasonalnaive", "persistence", "dlinear", "itransformer", "tcssdpmixer"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, cfg in enumerate(show):
        row = df[df["config"] == cfg]
        if row.empty:
            continue
        ys = [float(row[f"day{d}_MAE"].iloc[0]) for d in days]
        style = "--" if cfg in BASELINES else "-"
        lw = 2.6 if cfg == "tcssdpmixer" else 1.8
        ax.plot(days, ys, style, marker="o", lw=lw,
                color=PALETTE[i % len(PALETTE)],
                label=NICE.get(cfg, cfg).replace("\n", " "))
    ax.set_xlabel("Forecast day ahead"); ax.set_ylabel("MAE (Wh per hour)")
    ax.set_title("Error grows across the 7-day horizon")
    ax.set_xticks(days); ax.legend(loc="upper left")
    _save(fig, "fig_weekly_perday.png")


def fig_example_forecast():
    """Reload data, build Seasonal-Naive (no train) + retrain TC-SS-DPMixer,
    and plot one example test week (truth vs both)."""
    import torch
    from data_utils import Config, get_loaders
    from models import build_model
    from training import train, DEVICE

    cfg = Config(resample="1h", seq_len=PERIOD, pred_len=PERIOD,
                 train_ratio=0.70, val_ratio=0.15, feature_mode="no_rv",
                 add_time_features=True, log_target=True)
    tr, vl, te, scaler, n_feat = get_loaders(cfg)

    def to_wh(arr):
        arr = arr * scaler.scale_[0] + scaler.mean_[0]
        return np.expm1(arr)

    # pick one test window (a non-overlapping start, e.g. index 0)
    Xte = torch.from_numpy(te.dataset.X.numpy())
    yte = te.dataset.y.numpy()
    idx = 0
    truth = to_wh(yte[idx])                                  # [168]

    sn = build_model("seasonalnaive", PERIOD, PERIOD, n_feat, period=PERIOD)
    sn_pred = to_wh(sn(Xte[idx:idx+1]).detach().numpy()[0])

    model = build_model("tcssdpmixer", PERIOD, PERIOD, n_feat, time_feat_dim=6)
    model = train(model, tr, vl, label="tcss(for-fig)", loss_name="l1",
                  epochs=30, patience=6)
    model.eval()
    with torch.no_grad():
        ours = to_wh(model(Xte[idx:idx+1].to(DEVICE)).cpu().numpy()[0])

    hrs = np.arange(PERIOD)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hrs, truth, color=UCR_NAVY, lw=2.0, label="Ground truth")
    ax.plot(hrs, sn_pred, ls="--", color=UCR_GOLD, lw=1.6,
            label="Seasonal-Naive (last week)")
    ax.plot(hrs, ours, color=UCR_BLUE, lw=1.8, label="TC-SS-DPMixer (ours)")
    for d in range(1, 7):
        ax.axvline(d * 24, color="#C0C4CA", lw=0.8, ls=":")
    ax.set_xlabel("Hour into the forecast week (vertical lines = day boundaries)")
    ax.set_ylabel("Appliance energy (Wh per hour)")
    ax.set_title("One forecast week: truth vs baseline vs our model")
    ax.legend(loc="upper right", ncol=3)
    _save(fig, "fig_weekly_forecast.png")


if __name__ == "__main__":
    fig_compare()
    fig_perday()
    fig_example_forecast()
    print("Done.")
