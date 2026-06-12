"""
make_architecture_mole_style.py — TC-DPMixer architecture in MoLE-paper visual style.

Light-blue rounded outer boxes, yellow inner learnable blocks, ×/+ nodes,
mixing-layer on the right feeding the gating signals.

    MoLE      : K stacked homogeneous heads (all DLinear)
    TC-DPMixer: 4 heterogeneous branches
                (Persistence / Decomp / Channel-mix / Time-mix)

Output:
    results/figures/fig_architecture_mole_style.png  (+ .pdf vector)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib import rcParams

# ── MoLE-style palette ───────────────────────────────────────────
OUTER_BLUE   = "#C8DCEC"   # light blue, branch enclosure
OUTER_EDGE   = "#5A85AC"   # darker blue stroke
INNER_YEL    = "#FFEBA8"   # warm yellow, learnable inner blocks
INNER_EDGE   = "#C9A04A"   # darker yellow stroke
NO_PARAM_BG  = "#EFEFEF"   # grey for "no parameters" block
NO_PARAM_EDGE = "#9CA3AF"
INK          = "#1F2A37"
ARROW_BLACK  = "#1F2A37"
GATE_BG      = "#FFD580"   # amber gate
GATE_EDGE    = "#D99A2B"
WHITE        = "#FFFFFF"

rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


# ── primitives ──────────────────────────────────────────────────

def rbox(ax, x, y, w, h, color, edge, *, lw=1.3, dashed=False, rsize=1.8):
    """Rounded rectangle (MoLE-style outer / inner boxes)."""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rsize}",
        linewidth=lw, edgecolor=edge, facecolor=color,
        linestyle=(0, (5, 3)) if dashed else "-",
        zorder=2,
    )
    ax.add_patch(p)


def stacked_rbox(ax, x, y, w, h, color, edge, n=3, offset=1.2,
                 lw=1.0, dashed=False):
    """Stacked shadow effect (for blocks shown as multiple copies)."""
    for k in range(n - 1, -1, -1):
        ox = k * offset
        oy = -k * offset
        rbox(ax, x + ox, y + oy, w, h, color, edge, lw=lw, dashed=dashed)


def label(ax, x, y, text, *, size=10, weight="normal", color=INK,
          ha="center", va="center", style="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, weight=weight,
            color=color, style=style, zorder=4)


def arrow(ax, p0, p1, *, color=ARROW_BLACK, lw=1.2, dashed=False,
          head=10):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=head,
        linewidth=lw, color=color, zorder=1,
        shrinkA=2, shrinkB=2,
        linestyle=(0, (5, 3)) if dashed else "-",
    )
    ax.add_patch(a)


def line(ax, xs, ys, color=ARROW_BLACK, lw=1.2, dashed=False):
    ax.plot(xs, ys, color=color, lw=lw, zorder=1,
            linestyle=(0, (5, 3)) if dashed else "-")


def circle_node(ax, x, y, r, sym, *, ec=OUTER_EDGE, fs=15, lw=1.5):
    ax.add_patch(Circle((x, y), r, facecolor=WHITE, edgecolor=ec,
                        linewidth=lw, zorder=3))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs,
            weight="bold", color=INK, zorder=4)


# ── one branch ──────────────────────────────────────────────────

def draw_branch(ax, x, y, w, h, *, title, subtitle, inner_blocks,
                show_stack=False):
    """
    Draw one branch as a MoLE-style head:
        outer light-blue rounded box (optionally stacked behind),
        inner learnable yellow box(es) showing the branch's computation.

    inner_blocks: list of (text, kind) where kind ∈ {"learn", "no_param"}.
    """
    # Outer enclosure (stacked if branch has internal multiplicity).
    if show_stack:
        stacked_rbox(ax, x, y, w, h, OUTER_BLUE, OUTER_EDGE, n=3, offset=1.4)
    else:
        rbox(ax, x, y, w, h, OUTER_BLUE, OUTER_EDGE, lw=1.4)

    # Title bar at top of branch.
    label(ax, x + w / 2, y + h - 2.6, title,
          size=10.5, weight="bold", color=INK)
    label(ax, x + w / 2, y + h - 5.0, subtitle,
          size=8.0, color=OUTER_EDGE, style="italic")

    # Stack inner blocks vertically, centered within available space.
    n = len(inner_blocks)
    block_h = 4.5
    gap = 1.4
    total = n * block_h + (n - 1) * gap
    avail_top = y + h - 7.5     # below title
    avail_bot = y + 2.0
    avail = avail_top - avail_bot
    pad = max(0, (avail - total) / 2)
    cur_y = avail_top - pad - block_h
    for i, (text, kind) in enumerate(inner_blocks):
        bx, bw = x + 1.6, w - 3.2
        if kind == "learn":
            rbox(ax, bx, cur_y, bw, block_h,
                 INNER_YEL, INNER_EDGE, lw=1.1, rsize=1.0)
        else:
            rbox(ax, bx, cur_y, bw, block_h,
                 NO_PARAM_BG, NO_PARAM_EDGE, lw=1.0, rsize=1.0)
        label(ax, bx + bw / 2, cur_y + block_h / 2, text,
              size=8.5, color=INK)
        # vertical chevron between blocks (down arrow)
        if i < n - 1:
            arrow(ax, (x + w / 2, cur_y),
                  (x + w / 2, cur_y - gap), lw=0.9, head=8)
        cur_y -= (block_h + gap)


def main():
    fig, ax = plt.subplots(figsize=(11, 11.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 120); ax.axis("off")

    # ── title strip (optional, top) ─────────────────────────────
    label(ax, 50, 117.5, "TC-DPMixer  ·  heterogeneous-branch routing",
          size=13.5, weight="bold", color=INK)
    label(ax, 50, 114.0,
          "K = 4 distinct inductive biases, gated by time features "
          "(contrast: MoLE routes K homogeneous DLinear heads)",
          size=9.5, color=OUTER_EDGE, style="italic")

    # ── top inputs ──────────────────────────────────────────────
    # Input labels
    label(ax, 25, 108.0, r"Time series  X  $\in \mathbb{R}^{96 \times 32}$",
          size=10.5, weight="bold")
    label(ax, 84, 108.0, "Time features  (ch 26–31)",
          size=10.5, weight="bold", color=GATE_EDGE)

    # X feeds all 4 branches; time features feed gate only.
    branch_xs   = [4, 24, 44, 64]
    branch_w    = 18
    branch_y    = 60
    branch_h    = 40
    branch_tops = branch_y + branch_h

    # arrows from "X" to each branch top
    for cx in [c + branch_w / 2 for c in branch_xs]:
        # fork
        line(ax, [25, cx], [106, 104], color=ARROW_BLACK, lw=1.0)
        arrow(ax, (cx, 104), (cx, branch_tops + 0.5), lw=1.1)

    # arrow from time features (right side, sole arrow) down to gate
    arrow(ax, (92, 106), (92, 54), color=GATE_EDGE, lw=1.3)

    # ── 4 heterogeneous branches ────────────────────────────────
    branches = [
        dict(
            title="Persistence",
            subtitle="(after NLinear; no params)",
            blocks=[
                ("select  X[ −1 , 0 ]", "no_param"),
                ("tile  H  steps",      "no_param"),
            ],
            stack=False,
        ),
        dict(
            title="Decomposition",
            subtitle="(after DLinear)",
            blocks=[
                ("moving avg → trend / season", "no_param"),
                ("Linear  →  H",                "learn"),
                ("Linear  →  H",                "learn"),
            ],
            stack=False,
        ),
        dict(
            title="Channel mixer",
            subtitle="(after TSMixer)",
            blocks=[
                ("LayerNorm",                "learn"),
                ("MLP-Mixer × 2",            "learn"),
                ("Linear  →  H",             "learn"),
            ],
            stack=True,        # stacked = "× 2 mixer blocks"
        ),
        dict(
            title="Time mixer",
            subtitle="(after TSMixer)",
            blocks=[
                ("take channel 0",   "no_param"),
                ("MLP-Mixer × 2",    "learn"),
                ("Linear  →  H",     "learn"),
            ],
            stack=True,
        ),
    ]
    for x0, spec in zip(branch_xs, branches):
        draw_branch(
            ax, x0, branch_y, branch_w, branch_h,
            title=spec["title"], subtitle=spec["subtitle"],
            inner_blocks=spec["blocks"], show_stack=spec["stack"],
        )

    # ── per-branch output arrow → × node ────────────────────────
    x_centers = [c + branch_w / 2 for c in branch_xs]
    x_nodes_y = 53
    for cx in x_centers:
        arrow(ax, (cx, branch_y - 0.3), (cx, x_nodes_y + 2.0), lw=1.2)
    for i, cx in enumerate(x_centers):
        circle_node(ax, cx, x_nodes_y, 2.2, "×", fs=14)

    # ── gate network on the right (clearly clear of branches) ──
    gate_x, gate_y, gate_w, gate_h = 85, 38, 14, 16
    rbox(ax, gate_x, gate_y, gate_w, gate_h,
         GATE_BG, GATE_EDGE, lw=1.5)
    label(ax, gate_x + gate_w / 2, gate_y + gate_h - 3.0,
          "Gate network", size=10, weight="bold", color=INK)
    label(ax, gate_x + gate_w / 2, gate_y + gate_h - 6.0,
          "(small MLP)", size=8.5, color=INK, style="italic")
    label(ax, gate_x + gate_w / 2, gate_y + 4.0,
          "g₁ … g₄ ∈ [0, 2]", size=8.5, weight="bold", color=INK)

    # Gate's 4 output arrows fan out LEFTWARD to each × node (dashed amber).
    g_out_x = gate_x                       # left edge of gate
    g_out_y = gate_y + gate_h / 2
    for cx, lab in zip(x_centers, ["g₁", "g₂", "g₃", "g₄"]):
        arrow(ax, (g_out_x, g_out_y), (cx + 2.4, x_nodes_y),
              color=GATE_EDGE, lw=1.2, dashed=True)
        # gate label placed near middle of each arrow
        mx = (g_out_x + cx) / 2
        my = (g_out_y + x_nodes_y) / 2
        label(ax, mx, my + 1.6, lab,
              size=9.5, weight="bold", color=GATE_EDGE)

    # ── sum + output ────────────────────────────────────────────
    sum_x, sum_y = 50, 33
    for cx in x_centers:
        # vertical from × down to sum row
        line(ax, [cx, cx], [x_nodes_y - 2.2, sum_y + 2.4],
             color=ARROW_BLACK, lw=1.1)
        # horizontal into sum (only if not central)
        if abs(cx - sum_x) > 1:
            line(ax, [cx, sum_x], [sum_y + 2.4, sum_y + 2.4],
                 color=ARROW_BLACK, lw=1.1)
    arrow(ax, (sum_x, sum_y + 2.4), (sum_x, sum_y + 0.2),
          lw=1.1)
    circle_node(ax, sum_x, sum_y, 2.7, "+", fs=18)

    # arrow to output
    arrow(ax, (sum_x, sum_y - 2.7), (sum_x, 20), lw=1.3)
    rbox(ax, sum_x - 14, 11, 28, 9, "#E0F0E4", "#5AA17A", lw=1.4)
    label(ax, sum_x, 17, r"$\hat{y} \in \mathbb{R}^{H}$",
          size=11, weight="bold")
    label(ax, sum_x, 13.5,
          "forecast  (H = 24 short / 168 weekly)",
          size=8.5, color=OUTER_EDGE)

    # ── legend ──────────────────────────────────────────────────
    leg_x, leg_y = 4, 4
    rbox(ax, leg_x, leg_y - 0.8, 24, 4, INNER_YEL, INNER_EDGE,
         lw=1.0, rsize=1.0)
    label(ax, leg_x + 12, leg_y + 1.1,
          "yellow  =  learnable parameters",
          size=8.5)
    rbox(ax, leg_x + 26, leg_y - 0.8, 22, 4, NO_PARAM_BG,
         NO_PARAM_EDGE, lw=1.0, rsize=1.0)
    label(ax, leg_x + 37, leg_y + 1.1,
          "grey  =  no parameters",
          size=8.5)
    rbox(ax, leg_x + 50, leg_y - 0.8, 22, 4, GATE_BG, GATE_EDGE,
         lw=1.0, rsize=1.0)
    label(ax, leg_x + 61, leg_y + 1.1,
          "amber  =  routing signal",
          size=8.5)

    # ── output to file ──────────────────────────────────────────
    out_dir = os.path.join("CS228", "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "fig_architecture_mole_style.png")
    fig.savefig(out_png)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close()
    print(f"  -> saved  {out_png}  (+ .pdf vector)")


if __name__ == "__main__":
    main()
