"""
src/make_anchored_mlp_figure.py — visualise the formula

        ŷ = persistence(x) + α · MLP(x)

for an audience that gets lost in the text.  Two parallel pipelines with
concrete shapes annotated at EVERY arrow, then an element-wise add.

Outputs: results/figures/fig_anchored_mlp.png  (+ .pdf)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib import rcParams

INK      = "#1F2A37"
BORDER   = "#AEB8C2"
FLOW     = "#6B7C8F"
ACCENT   = "#D99A2B"      # persistence path
ACCENT_D = "#B97E16"
MUTED_BG = "#EFF3F7"
GREEN_BG = "#E9F1EA"

rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220,
                 "savefig.bbox": "tight", "figure.facecolor": "white"})

FIG_DIR = os.path.join("results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def box(ax, x, y, w, h, text, fc, ec=BORDER, fs=10.0, weight="normal",
        tc=INK, lw=1.0, rsize=1.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rsize}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=weight, zorder=3,
                linespacing=1.35)


def node(ax, x, y, r, sym, ec=FLOW, tc=INK, fs=13, fc="white"):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec,
                        linewidth=1.3, zorder=4))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs,
            weight="bold", color=tc, zorder=5)


def arrow(ax, p0, p1, color=FLOW, lw=1.3):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=lw,
        color=color, zorder=1, shrinkA=3, shrinkB=3))


def vec(ax, x, y, w, h, n_cells, fc, label="", note=""):
    cell_h = h / n_cells
    for i in range(n_cells):
        ax.add_patch(Rectangle(
            (x, y + i * cell_h), w, cell_h, facecolor=fc,
            edgecolor=BORDER, linewidth=0.4, zorder=2))
    ax.add_patch(Rectangle(
        (x, y), w, h, facecolor="none", edgecolor=BORDER, linewidth=0.9,
        zorder=3))
    if label:
        ax.text(x + w / 2, y + h + 1.0, label, ha="center", va="bottom",
                fontsize=9.5, color=INK, weight="bold")
    if note:
        ax.text(x + w / 2, y - 1.2, note, ha="center", va="top",
                fontsize=7.8, color=FLOW, style="italic")


def matrix(ax, x, y, w, h, n_rows, n_cols, highlight=None):
    """Draw a small grid representing X with an optional highlighted cell."""
    cell_w = w / n_cols
    cell_h = h / n_rows
    for r in range(n_rows):
        for c in range(n_cols):
            fc = MUTED_BG
            if highlight is not None and r == highlight[0] and c == highlight[1]:
                fc = ACCENT
            ax.add_patch(Rectangle(
                (x + c * cell_w, y + h - (r + 1) * cell_h),
                cell_w, cell_h, facecolor=fc, edgecolor=BORDER,
                linewidth=0.3, zorder=2))
    ax.add_patch(Rectangle(
        (x, y), w, h, facecolor="none", edgecolor=BORDER, linewidth=1.0,
        zorder=3))


def main():
    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ── title ────────────────────────────────────────────────
    ax.text(5, 94, "Residual-anchored MLP:   ŷ  =  persistence(x)  +  α · MLP(x)",
            fontsize=15.5, weight="bold", color=INK)
    ax.text(5, 89.2,
            "persistence gives a sensible floor;  the MLP only has to learn the small residual;  α controls how much the MLP is trusted",
            fontsize=10.0, style="italic", color=ACCENT_D)

    # ── X matrix (input) ─────────────────────────────────────
    ax.text(13, 79.5, "Input  X", ha="center", fontsize=11.5,
            weight="bold", color=INK)
    matrix(ax, x=6, y=46, w=14, h=30, n_rows=8, n_cols=6, highlight=(7, 0))
    ax.text(13, 44, "shape:  (L=96, C=32)", ha="center", va="top",
            fontsize=9, color=FLOW)
    ax.text(13, 40, "amber cell  =  X[-1, 0]\n(last timestep, channel 0 = Appliances)",
            ha="center", va="top", fontsize=8.5, color=ACCENT_D, style="italic")

    # ── PERSISTENCE path (top, amber) ────────────────────────
    arrow(ax, (20, 49), (27, 68), color=ACCENT, lw=1.6)
    box(ax, 27, 65, 14, 7, "pick one number\nX[-1, 0]   →   a scalar",
        "white", ec=ACCENT, fs=9.5, tc=ACCENT_D, lw=1.2)
    arrow(ax, (41, 68.5), (46, 68.5), color=ACCENT, lw=1.5)
    box(ax, 46, 65, 14, 7, "tile  H = 24  times", "white",
        ec=ACCENT, fs=9.5, tc=ACCENT_D, lw=1.2)
    arrow(ax, (60, 68.5), (65, 68.5), color=ACCENT, lw=1.5)
    vec(ax, x=65, y=58, w=3.3, h=22, n_cells=8, fc=ACCENT,
        label="p ∈ ℝ²⁴", note="24 copies of\nthe same scalar")

    # ── MLP path (bottom, neutral) ───────────────────────────
    arrow(ax, (20, 60), (27, 31))
    box(ax, 27, 27, 14, 8, "flatten\n96 × 32  =  3072 dims",
        "white", fs=9.3)
    arrow(ax, (41, 31), (46, 31))
    box(ax, 46, 24, 14, 14,
        "MLP\n3072 → 256 → 128 → 24\n(~800K parameters)",
        MUTED_BG, fs=9.3)
    arrow(ax, (60, 31), (64, 31))
    vec(ax, x=64, y=20, w=3.3, h=22, n_cells=8, fc=MUTED_BG,
        label="MLP(x)", note="∈ ℝ²⁴")
    arrow(ax, (67.3, 31), (71, 31))
    node(ax, 73.2, 31, 2.2, "×α", fs=10)
    arrow(ax, (75.4, 31), (79, 31))
    vec(ax, x=79, y=20, w=3.3, h=22, n_cells=8, fc=MUTED_BG,
        label="α · MLP(x)", note="learned residual")

    # ── combine: ⊕ ──────────────────────────────────────────
    arrow(ax, (68.3, 68), (89, 52.5), color=ACCENT, lw=1.5)
    arrow(ax, (82.3, 33), (89, 47.5))
    node(ax, 90, 50, 2.8, "+", fs=15)
    arrow(ax, (92.8, 50), (94.5, 50))
    vec(ax, x=94.5, y=39, w=3.3, h=22, n_cells=8, fc=GREEN_BG,
        label="ŷ ∈ ℝ²⁴", note="forecast\n(next 4 h)")

    # ── footer: behaviour summary ────────────────────────────
    ax.plot([5, 99], [13, 13], color="#D7DEE5", lw=1.0, zorder=1)
    ax.text(5, 9.0,
            "α = 0  ⇒  ŷ = p  (pure persistence; MLP disabled).      "
            "α = 1  ⇒  ŷ = p + MLP(x)  (full residual).      "
            "Adam learns the best α for the data.",
            fontsize=9.5, color=INK)
    ax.text(5, 4.2,
            "Why this beats vanilla MLP:  a plain MLP with weight decay drifts "
            "toward predicting the mean (≈ 0 after standardisation), which is "
            "a bad guess for a spiky signal.  With the anchor, the MLP only "
            "has to learn the small residual — a far easier problem.",
            fontsize=9.0, color=FLOW, style="italic")

    out = os.path.join(FIG_DIR, "fig_anchored_mlp.png")
    fig.savefig(out); fig.savefig(out.replace(".png", ".pdf"))
    plt.close()
    print(f"  -> saved {out}  (+ .pdf)")


if __name__ == "__main__":
    main()
