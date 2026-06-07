"""
make_ablation_figures.py — Plot the three ablation studies for the pre.

Reads:
    results/ablation_alpha.csv
    results/ablation_length.csv
    results/ablation_features.csv

Writes:
    results/figures/fig_ablation_alpha.png   (also .pdf)
    results/figures/fig_ablation_length.png  (also .pdf)
    results/figures/fig_ablation_features.png(also .pdf)
    results/figures/fig_ablations.png        (3-panel combined for one slide)

Run:
    python src/make_ablation_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


INK    = "#1F2A37"
SLATE  = "#6B7C8F"
AMBER  = "#D99A2B"
TEAL   = "#2E8B8B"
GRID   = "#D7DDE3"

rcParams.update({
    "font.family":  "DejaVu Sans",
    "savefig.dpi":  220,
    "savefig.bbox": "tight",
})

RESULTS = "results"
FIGDIR  = os.path.join(RESULTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)


# ──────────────────────────────────────────────────────
#  panel drawers — each takes an Axes, draws one ablation
# ──────────────────────────────────────────────────────

def _format(ax, title, ylabel="Test MAE (Wh)"):
    ax.set_title(title, fontsize=11, color=INK, weight="bold", loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_alpha(ax, df):
    """A1 — α sweep, both horizons (split into 2 sub-axes via twinx)."""
    df = df.copy()
    # x-axis ordering: 0, 0.25, 1, vanilla
    order = {"alpha_0.00_persistence": 0,
             "alpha_0.25": 1, "alpha_1.00": 2, "vanilla_mlp": 3}
    labels = ["α=0\n(persist.)", "α=0.25", "α=1\n(anchored)", "vanilla\nMLP"]
    df["xi"] = df["model"].map(order)

    short  = df[df.horizon == "short"].sort_values("xi")
    weekly = df[df.horizon == "weekly"].sort_values("xi")

    # short on left axis (Wh)
    ax.plot(short["xi"], short["test_MAE"], "-o",
            color=TEAL, lw=2, ms=7, label="short (Wh)")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="y", labelcolor=TEAL, labelsize=8.5)
    ax.set_ylabel("short Test MAE (Wh)", color=TEAL, fontsize=9)
    for s in ("top",):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.6)
    ax.set_title("A1 · α sweep — does the anchor help?",
                 fontsize=11, color=INK, weight="bold", loc="left")

    # weekly on right axis (Wh/h)
    ax2 = ax.twinx()
    ax2.plot(weekly["xi"], weekly["test_MAE"], "--s",
             color=AMBER, lw=2, ms=7, label="weekly (Wh/h)")
    ax2.tick_params(axis="y", labelcolor=AMBER, labelsize=8.5)
    ax2.set_ylabel("weekly Test MAE (Wh/h)", color=AMBER, fontsize=9)
    for s in ("top",):
        ax2.spines[s].set_visible(False)

    # combined legend
    ax.plot([], [], "-o", color=TEAL,  label="short  (Wh)")
    ax.plot([], [], "--s", color=AMBER, label="weekly (Wh/h)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def draw_length(ax, df):
    """A2 — L sweep, two lines (short / weekly), shared y."""
    short  = df[df.horizon == "short"].sort_values("seq_len")
    weekly = df[df.horizon == "weekly"].sort_values("seq_len")

    ax.plot(short["seq_len"], short["test_MAE"], "-o",
            color=TEAL, lw=2, ms=7, label="short (Wh)")
    ax.tick_params(axis="y", labelcolor=TEAL, labelsize=8.5)
    ax.set_ylabel("short Test MAE (Wh)", color=TEAL, fontsize=9)
    ax.set_xlabel("input length L (steps)", fontsize=9, color=INK)
    ax.set_title("A2 · input length L — how much history?",
                 fontsize=11, color=INK, weight="bold", loc="left")
    for s in ("top",):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.6)

    ax2 = ax.twinx()
    ax2.plot(weekly["seq_len"], weekly["test_MAE"], "--s",
             color=AMBER, lw=2, ms=7, label="weekly (Wh/h)")
    ax2.tick_params(axis="y", labelcolor=AMBER, labelsize=8.5)
    ax2.set_ylabel("weekly Test MAE (Wh/h)", color=AMBER, fontsize=9)
    for s in ("top",):
        ax2.spines[s].set_visible(False)

    # annotate each point with L value
    for _, r in short.iterrows():
        ax.annotate(f"L={int(r.seq_len)}",
                    (r.seq_len, r.test_MAE),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5, color=TEAL)
    for _, r in weekly.iterrows():
        ax2.annotate(f"L={int(r.seq_len)}",
                     (r.seq_len, r.test_MAE),
                     textcoords="offset points", xytext=(0, -14),
                     ha="center", fontsize=7.5, color=AMBER)


def draw_features(ax, df):
    """A3 — feature-subset bar chart, grouped by horizon."""
    order = ["target_only", "target_time", "all_no_rv"]
    labels = ["target\nonly\n(n=1)",
              "target +\ntime\n(n=7)",
              "all\nno_rv\n(n=32)"]

    short = df[df.horizon == "short"].set_index("subset").reindex(order)
    weekly = df[df.horizon == "weekly"].set_index("subset").reindex(order)

    x = np.arange(len(order))
    w = 0.36
    bs = ax.bar(x - w / 2, short["test_MAE"], width=w,
                color=TEAL, label="short (Wh)", edgecolor="white")
    ax.set_ylabel("short Test MAE (Wh)", color=TEAL, fontsize=9)
    ax.tick_params(axis="y", labelcolor=TEAL, labelsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("A3 · feature subsets — which channels matter?",
                 fontsize=11, color=INK, weight="bold", loc="left")
    for s in ("top",):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.6)

    ax2 = ax.twinx()
    bw = ax2.bar(x + w / 2, weekly["test_MAE"], width=w,
                 color=AMBER, label="weekly (Wh/h)", edgecolor="white")
    ax2.set_ylabel("weekly Test MAE (Wh/h)", color=AMBER, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=AMBER, labelsize=8.5)
    for s in ("top",):
        ax2.spines[s].set_visible(False)

    # annotate values on bars
    for b, v in zip(bs, short["test_MAE"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5, color=TEAL)
    for b, v in zip(bw, weekly["test_MAE"]):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=7.5, color=AMBER)


# ──────────────────────────────────────────────────────
#  driver
# ──────────────────────────────────────────────────────

def main():
    df_a = pd.read_csv(os.path.join(RESULTS, "ablation_alpha.csv"))
    df_l = pd.read_csv(os.path.join(RESULTS, "ablation_length.csv"))
    df_f = pd.read_csv(os.path.join(RESULTS, "ablation_features.csv"))

    # individual figures
    for tag, fn, df in [
        ("alpha",    draw_alpha,    df_a),
        ("length",   draw_length,   df_l),
        ("features", draw_features, df_f),
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        fn(ax, df)
        out = os.path.join(FIGDIR, f"fig_ablation_{tag}")
        fig.savefig(out + ".png"); fig.savefig(out + ".pdf")
        plt.close(fig)
        print(f"  wrote {out}.png")

    # combined 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.8))
    draw_alpha(axes[0],    df_a)
    draw_length(axes[1],   df_l)
    draw_features(axes[2], df_f)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_ablations")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf")
    plt.close(fig)
    print(f"  wrote {out}.png")


if __name__ == "__main__":
    main()
