"""
make_relwork_figures.py  —  two small Related-Work figures.

For the presentation's Related Work section (limited time), the user wants
to discuss only TWO prior works:

  (A)  MoLE  (Ni et al., AISTATS 2024)  — inspiration for the gating idea.
       Show ONE head of MoLE, zoomed in.
  (B)  DLinear (Zeng et al., AAAI 2023) — inspiration for the
       decomposition branch in our architecture.
       Simplify the user's RLinear/RMLP/DLinear template to the
       DLinear specialization only.

Both figures use the same visual language as the originals (light-blue
rounded outer boxes, warm-yellow inner learnable blocks, dashed = optional,
curved blue annotation arrows).

Outputs:
    CS228/results/figures/fig_relwork_mole_head.{png,pdf}
    CS228/results/figures/fig_relwork_dlinear.{png,pdf}
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import rcParams

# ── palette  (matches MoLE / DLinear template colours) ──────────
OUTER_BLUE   = "#C8DCEC"
OUTER_EDGE   = "#5A85AC"
INNER_YEL    = "#FFEBA8"
INNER_EDGE   = "#C9A04A"
INK          = "#1F2A37"
ANNOT_BLUE   = "#3B78C4"

rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


# ────────────────────────────────────────────────────────────────
#  Drawing primitives
# ────────────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, color, edge, *, lw=1.4, dashed=False, rsize=2.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rsize}",
        linewidth=lw, edgecolor=edge, facecolor=color,
        linestyle=(0, (5, 3)) if dashed else "-",
        zorder=2,
    )
    ax.add_patch(p)


def stacked_rbox(ax, x, y, w, h, color, edge, n=3, offset=1.3,
                 lw=1.2, dashed=False):
    for k in range(n - 1, -1, -1):
        ox = k * offset
        oy = -k * offset
        rbox(ax, x + ox, y + oy, w, h, color, edge,
             lw=lw, dashed=dashed)


def lbl(ax, x, y, t, *, size=11, weight="normal", color=INK,
        ha="center", va="center", style="normal"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=size, weight=weight,
            color=color, style=style, zorder=4)


def arrow(ax, p0, p1, *, color=INK, lw=1.2, dashed=False, head=10):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=head,
        linewidth=lw, color=color, zorder=5,
        shrinkA=2, shrinkB=2,
        linestyle=(0, (5, 3)) if dashed else "-",
    )
    ax.add_patch(a)


def curved_annot(ax, p_from, p_to, color=ANNOT_BLUE, lw=1.4, head=10):
    """Curved annotation arrow (like the blue arrows in the user's template)."""
    x0, y0 = p_from; x1, y1 = p_to
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        connectionstyle="arc3,rad=-0.2",
        arrowstyle="-|>", mutation_scale=head,
        linewidth=lw, color=color, zorder=5,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def circle(ax, x, y, r, sym, *, ec=OUTER_EDGE, fs=15, lw=1.6):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=ec,
                        linewidth=lw, zorder=3))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs,
            weight="bold", color=INK, zorder=4)


# ────────────────────────────────────────────────────────────────
#  Figure A: ONE zoomed-in MoLE head
# ────────────────────────────────────────────────────────────────

def figure_mole_head():
    fig, ax = plt.subplots(figsize=(7.0, 7.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 110); ax.axis("off")

    # Title strip
    lbl(ax, 50, 106, "MoLE  ·  zoom-in on one head",
        size=13.5, weight="bold")
    lbl(ax, 50, 102, "Ni et al., AISTATS 2024",
        size=9.5, color=OUTER_EDGE, style="italic")

    # Top input
    lbl(ax, 50, 95, "Time series data", size=10.5, weight="bold")
    arrow(ax, (50, 92), (50, 85), lw=1.4)

    # Big outer head box
    rbox(ax, 8, 18, 84, 65, OUTER_BLUE, OUTER_EDGE, lw=1.6, rsize=3.0)
    lbl(ax, 88, 78, "1 head",
        size=11, weight="bold", color=OUTER_EDGE, ha="right")

    # Inner preprocessing block
    rbox(ax, 32, 68, 36, 10, OUTER_BLUE, OUTER_EDGE, lw=1.3, rsize=1.4)
    lbl(ax, 50, 73, "preprocessing", size=11, weight="bold")

    # Two parallel paths: linear (mandatory) | linear (optional, stacked)
    # Mandatory linear (left)
    rbox(ax, 18, 44, 24, 14, INNER_YEL, INNER_EDGE, lw=1.4, rsize=1.4)
    lbl(ax, 30, 51, "linear", size=11.5, weight="bold")

    # Optional linear (right) — stacked dashed boxes
    stacked_rbox(ax, 58, 44, 26, 14, INNER_YEL, INNER_EDGE,
                 n=3, offset=1.4, lw=1.1, dashed=True)
    lbl(ax, 71, 51, "linear\n(optional)", size=10, weight="bold")

    # From preprocessing → both linears (DIRECT, no shared stem; matches MoLE original)
    arrow(ax, (42, 68), (30, 58.5), lw=1.3)
    arrow(ax, (58, 68), (71, 58.5), lw=1.3, dashed=True)

    # + node
    circle(ax, 50, 30, 2.6, "+", fs=18)

    # Both paths converge into +
    arrow(ax, (30, 44), (47.5, 32.0), lw=1.3)
    arrow(ax, (71, 44), (52.5, 32.0), lw=1.3, dashed=True)

    # Output arrow downward
    arrow(ax, (50, 27.4), (50, 12), lw=1.3)
    lbl(ax, 50, 8, "head output", size=10, weight="bold",
        color=OUTER_EDGE)

    # Take-home note at bottom
    lbl(ax, 50, 2,
        "MoLE has K such heads (HOMOGENEOUS), routed by a small\n"
        "time-conditioned gating network not shown here.",
        size=8.5, color=OUTER_EDGE, style="italic")

    out_dir = os.path.join("CS228", "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "fig_relwork_mole_head.png")
    fig.savefig(out)
    try:
        fig.savefig(out.replace(".png", ".pdf"))
    except PermissionError as e:
        print(f"  (PDF skipped — file open elsewhere: {e})")
    plt.close()
    print(f"  -> saved  {out}  (+ .pdf)")


# ────────────────────────────────────────────────────────────────
#  Figure B: DLinear-only specialization of the user's template
# ────────────────────────────────────────────────────────────────

def figure_dlinear():
    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 110); ax.axis("off")

    # Title
    lbl(ax, 50, 106, "DLinear  ·  decomposition + two linear layers",
        size=13.5, weight="bold")
    lbl(ax, 50, 102, "Zeng et al., AAAI 2023",
        size=9.5, color=OUTER_EDGE, style="italic")

    # Top input
    lbl(ax, 30, 95, "Time series data", size=10.5, weight="bold")
    arrow(ax, (30, 92), (30, 84.5), lw=1.4)

    # Preprocessing block
    rbox(ax, 18, 73, 24, 11, OUTER_BLUE, OUTER_EDGE, lw=1.5, rsize=1.8)
    lbl(ax, 30, 78.5, "preprocessing", size=11, weight="bold")

    # Curved annotation: preprocessing = decomposition
    curved_annot(ax, (42, 80), (62, 84))
    lbl(ax, 64, 85, "Time series decomposition  (trend  +  seasonal)",
        size=10, color=ANNOT_BLUE, ha="left")

    # From preprocessing → two parallel "linear" boxes
    arrow(ax, (24, 73), (16, 58), lw=1.2)
    arrow(ax, (36, 73), (44, 58), lw=1.2, dashed=True)

    # Mandatory linear (trend) on the left
    rbox(ax, 4, 44, 22, 14, INNER_YEL, INNER_EDGE, lw=1.4, rsize=1.4)
    lbl(ax, 15, 51, "linear", size=11.5, weight="bold")
    lbl(ax, 15, 47, "(trend)", size=9, color=INK, style="italic")

    # Optional linear (seasonal) on the right — single dashed (not stacked)
    rbox(ax, 32, 44, 22, 14, INNER_YEL, INNER_EDGE, lw=1.4,
         rsize=1.4, dashed=True)
    lbl(ax, 43, 51, "linear", size=11.5, weight="bold")
    lbl(ax, 43, 47, "(seasonal)", size=9, color=INK, style="italic")

    # Curved annotation to the right of the two linears
    curved_annot(ax, (55, 51), (66, 51))
    lbl(ax, 68, 52,
        "2 linear layers:  one for trend,  one for seasonal",
        size=10, color=ANNOT_BLUE, ha="left")

    # Both linears → +
    arrow(ax, (15, 44), (24, 32.5), lw=1.2)
    arrow(ax, (43, 44), (33, 32.5), lw=1.2, dashed=True)
    circle(ax, 29, 30, 2.6, "+", fs=18)

    # Postprocessing — annotate as "None for DLinear"
    arrow(ax, (29, 27.4), (29, 19), lw=1.3)
    rbox(ax, 17, 9, 24, 10, OUTER_BLUE, OUTER_EDGE, lw=1.3,
         rsize=1.4, dashed=True)
    lbl(ax, 29, 16, "postprocessing", size=10.5, weight="bold")
    lbl(ax, 29, 12, "(optional)", size=8.5, style="italic")

    curved_annot(ax, (41, 14), (60, 14))
    lbl(ax, 62, 14, "None  (DLinear has no inverse normalisation)",
        size=10, color=ANNOT_BLUE, ha="left")

    # Final output arrow
    arrow(ax, (29, 8.5), (29, 2.5), lw=1.3)

    out_dir = os.path.join("CS228", "results", "figures")
    out = os.path.join(out_dir, "fig_relwork_dlinear.png")
    fig.savefig(out)
    try:
        fig.savefig(out.replace(".png", ".pdf"))
    except PermissionError as e:
        print(f"  (PDF skipped — file open elsewhere: {e})")
    plt.close()
    print(f"  -> saved  {out}  (+ .pdf)")


def main():
    figure_mole_head()
    figure_dlinear()


if __name__ == "__main__":
    main()
