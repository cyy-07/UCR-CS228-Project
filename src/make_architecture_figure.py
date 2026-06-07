"""
src/make_architecture_figure.py — publication-quality architecture diagram.

Design language (journal-style):
  • a SINGLE accent colour (amber) marks the novel gating path; everything
    else is neutral slate — the reader's eye goes straight to the contribution
  • thin 1.0–1.3 pt strokes, light tint fills, generous whitespace
  • one typeface, a small set of sizes, small-caps section labels
  • the headline is INTERPRETABILITY: a small network reads the time inputs
    and emits per-sample branch weights g(t) that ARE the explanation.

Run
---
    python src/make_architecture_figure.py
Outputs: results/figures/fig_architecture.png  (+ .pdf vector)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import rcParams

# ── refined palette ──────────────────────────────────────────
INK     = "#1F2A37"   # near-black text
BORDER  = "#AEB8C2"   # light neutral box border
FLOW    = "#6B7C8F"   # muted steel arrows
ACCENT  = "#D99A2B"   # the ONE hero colour (gating path)
ACCENT_D = "#B97E16"  # darker amber for gate border/text
HIST_BG = "#EFF3F7"   # very light blue-grey
BRCH_BG = "#E7EDF3"
GATE_BG = "#FBEFD4"   # soft amber tint
OUT_BG  = "#E9F1EA"   # soft green tint
GROUP_BG = "#F7F9FB"  # faint backbone group

rcParams.update({
    "font.family":  "DejaVu Sans",
    "savefig.dpi":  220,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

FIG_DIR = os.path.join("results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def box(ax, x, y, w, h, text, fc, ec=BORDER, fs=10.5, weight="normal",
        tc=INK, lw=1.0, rsize=1.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rsize}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=weight, zorder=3,
                linespacing=1.35)


def node(ax, x, y, r, sym, ec=FLOW, tc=INK, fs=13):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=ec,
                        linewidth=1.3, zorder=4))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs,
            weight="bold", color=tc, zorder=5)


def arrow(ax, p0, p1, color=FLOW, lw=1.3, dashed=False):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=lw,
        color=color, zorder=1, shrinkA=3, shrinkB=3,
        linestyle=(0, (5, 3)) if dashed else "-"))


def line(ax, xs, ys, color=FLOW, lw=1.1):
    ax.plot(xs, ys, color=color, lw=lw, zorder=1, solid_capstyle="round")


def slabel(ax, x, y, t):
    ax.text(x, y, t.upper(), fontsize=8, color=FLOW, weight="bold",
            family="DejaVu Sans")


def main():
    fig, ax = plt.subplots(figsize=(13.8, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ── title (left-aligned, restrained) ────────────────────
    ax.text(5, 94, "Time-conditioned gated forecaster",
            fontsize=15.5, weight="bold", color=INK)
    ax.text(5, 88.5, "the per-sample gate weights g(t) are the explanation",
            fontsize=10.5, style="italic", color=ACCENT_D)

    # ── backbone group background ────────────────────────────
    box(ax, 22, 26, 27, 52, "", GROUP_BG, ec="#E2E8EE", lw=1.0, rsize=2.0)
    slabel(ax, 23.5, 79.0, "forecasting backbone")

    # ── input ────────────────────────────────────────────────
    slabel(ax, 5.2, 79.0, "input  X ∈ ℝ (L×C)")
    box(ax, 5, 62, 14, 13, "Series history\nchannels 0–25", HIST_BG)
    box(ax, 5, 45, 14, 13, "Time features\nhour · weekday", GATE_BG,
        ec=ACCENT, fs=10.5)

    # ── branches ─────────────────────────────────────────────
    box(ax, 24, 64, 21, 11, "Trend branch", BRCH_BG)
    box(ax, 24, 50, 21, 11, "Seasonal branch", BRCH_BG)
    ax.text(34.5, 47.3, "+  K interchangeable branches", ha="center",
            fontsize=8.5, style="italic", color=FLOW)

    # ── gate network (the contribution) ──────────────────────
    box(ax, 24, 30, 21, 12, "Gate network  (MLP)\nt  →  g₁ … g_K  ∈ [0, 2]",
        GATE_BG, ec=ACCENT, fs=10.5, weight="bold", tc=ACCENT_D, lw=1.4)

    # ── combine nodes ────────────────────────────────────────
    node(ax, 52, 69.5, 2.1, "×")
    node(ax, 52, 55.5, 2.1, "×")
    node(ax, 61, 62.5, 2.6, "Σ")
    node(ax, 74, 56, 2.7, "+")
    box(ax, 66, 31, 16, 10, "Persistence anchor", "white", ec=BORDER, fs=9.5)
    box(ax, 84, 49.5, 13, 13, "Forecast\nŷ ∈ ℝ (H)", OUT_BG, ec="#5AA17A",
        fs=10.5, weight="bold")

    # ── arrows: data flow (neutral) ──────────────────────────
    arrow(ax, (19, 68.5), (24, 69.5))
    arrow(ax, (19, 68.5), (24, 55.5))
    arrow(ax, (45, 69.5), (49.9, 69.5))
    arrow(ax, (45, 55.5), (49.9, 55.5))
    arrow(ax, (54.1, 69.5), (58.7, 64.2))
    arrow(ax, (54.1, 55.5), (58.7, 60.8))
    arrow(ax, (63.6, 62.5), (71.5, 57.2))
    arrow(ax, (76.6, 56), (84, 56))

    # ── gating path (the ONE accent) ─────────────────────────
    arrow(ax, (19, 51.5), (24, 39), color=ACCENT, lw=1.6)        # time → gate
    arrow(ax, (45, 39), (50.6, 67.6), color=ACCENT, lw=1.5, dashed=True)
    arrow(ax, (45, 37), (50.6, 53.6), color=ACCENT, lw=1.5, dashed=True)
    ax.text(47.6, 60.0, "g₁", color=ACCENT_D, fontsize=11, weight="bold")
    ax.text(47.6, 44.5, "g₂", color=ACCENT_D, fontsize=11, weight="bold")

    # ── persistence anchor routed cleanly outside everything ─
    line(ax, [5, 2.3], [64.5, 64.5])
    line(ax, [2.3, 2.3], [64.5, 23])
    line(ax, [2.3, 74], [23, 23])
    arrow(ax, (74, 23), (74, 30.7))
    arrow(ax, (74, 41), (74, 53.3))
    ax.text(30, 24.6, "copy last value of channel 0", fontsize=7.6,
            color=FLOW, style="italic")

    # ── one-line interpretability note under a thin rule ─────
    line(ax, [5, 97], [15.5, 15.5], color="#D7DEE5", lw=1.0)
    ax.text(5, 10.8,
            "g(t) depends only on the time inputs and is computed per sample — "
            "plotting it shows which pattern the model trusts at each hour.  "
            "The same gate", fontsize=9.3, color=INK)
    ax.text(5, 6.6,
            "plugs into different branch families (DLinear, DPMixer), so it is "
            "a reusable mechanism rather than a single-model trick.",
            fontsize=9.3, color=INK)

    out = os.path.join(FIG_DIR, "fig_architecture.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"  -> saved {out}  (+ .pdf)")


if __name__ == "__main__":
    main()
