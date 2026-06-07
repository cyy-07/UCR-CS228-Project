"""
make_architecture_teaching.py — teaching-version architecture diagram for S5.

Goal (different from make_architecture_figure.py):
  Every branch is tagged with the prior-art it came from
  (NLinear / DLinear / TSMixer / MoLE), so the audience instantly
  sees what's borrowed vs. what we designed (the amber gate path).

Output:
  results/figures/fig_architecture_teaching.png  (+ .pdf)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams

INK      = "#1F2A37"
BORDER   = "#AEB8C2"
FLOW     = "#6B7C8F"
ACCENT   = "#D99A2B"
ACCENT_D = "#B97E16"
HIST_BG  = "#EFF3F7"
BRCH_BG  = "#E7EDF3"
GATE_BG  = "#FBEFD4"
OUT_BG   = "#E9F1EA"
TAG_BG   = "#F4E9D2"   # small prior-art chip
OURS_BG  = "#FCE3B8"   # our-contribution chip

rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

def box(ax, x, y, w, h, fc, ec, lw=1.1, r=0.04):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.005,rounding_size={r}",
        fc=fc, ec=ec, lw=lw))

def arrow(ax, x0, y0, x1, y1, color=FLOW, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style,
        mutation_scale=10, color=color, lw=lw))

def chip(ax, x, y, txt, fc=TAG_BG, ec=ACCENT_D, fs=7.0, w=1.30):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, 0.26, boxstyle="round,pad=0.01,rounding_size=0.05",
        fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + 0.13, txt, ha="center", va="center",
            fontsize=fs, color=ACCENT_D, weight="bold")

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 6.6); ax.axis("off")

    # ── title ──
    ax.text(5.6, 6.30, "TC-DPMixer", ha="center", va="center",
            fontsize=15, weight="bold", color=INK)
    ax.text(5.6, 5.95,
            "borrowed parts in slate · our contribution (the gate) in amber",
            ha="center", va="center", fontsize=9, color=FLOW, style="italic")

    # ── input ──
    box(ax, 0.20, 2.55, 1.55, 1.10, HIST_BG, BORDER)
    ax.text(0.97, 3.30, "Input  X", ha="center", fontsize=10.5,
            weight="bold", color=INK)
    ax.text(0.97, 3.00, "(L, 32)", ha="center", fontsize=9, color=INK)
    ax.text(0.97, 2.78, "L=96 short / 168 weekly", ha="center",
            fontsize=7.5, color=FLOW, style="italic")

    # ── 4 branches column ──
    bx = 3.20
    bw = 3.05
    bh = 0.82
    ys = [4.55, 3.55, 2.55, 1.55]
    titles = [
        ("Branch 1 · Persistence anchor",  "x[-1, 0]  ·  repeat H times"),
        ("Branch 2 · Trend + Seasonal",    "moving-avg decomp, target only"),
        ("Branch 3 · Channel-Mixer",       "MLP-mix across 32 channels @ last step"),
        ("Branch 4 · Time-Mixer",          "MLP-mix across L steps, target only"),
    ]
    tags = ["from NLinear", "from DLinear", "from TSMixer", "from TSMixer"]

    for y, (t, sub), tag in zip(ys, titles, tags):
        box(ax, bx, y, bw, bh, BRCH_BG, BORDER)
        ax.text(bx + 0.12, y + bh - 0.22, t, fontsize=9.2,
                weight="bold", color=INK)
        ax.text(bx + 0.12, y + 0.20, sub, fontsize=7.8, color=INK)
        # chip sits OUTSIDE the box on the right shoulder
        chip(ax, bx + bw + 0.06, y + bh / 2 - 0.13, tag)

        # arrow input → branch
        arrow(ax, 1.78, 3.10, bx, y + bh / 2)
        # arrow branch → sum (start AFTER chip)
        arrow(ax, bx + bw + 1.42, y + bh / 2, 9.05, 3.10)

    # ── gate net (amber, our contribution) ──
    gx, gy, gw, gh = 3.20, 0.30, 3.05, 1.00
    box(ax, gx, gy, gw, gh, GATE_BG, ACCENT_D, lw=1.5)
    ax.text(gx + gw / 2, gy + gh - 0.22,
            "Gate network  g(t)", ha="center",
            fontsize=10, weight="bold", color=ACCENT_D)
    ax.text(gx + gw / 2, gy + 0.48,
            "time feats of  x[-1, -6:]   →   MLP (6→16→4)", ha="center",
            fontsize=8.2, color=INK)
    ax.text(gx + gw / 2, gy + 0.20,
            "→  sigmoid · 2   ⇒   (g₁, g₂, g₃, g₄) ∈ [0, 2]", ha="center",
            fontsize=8.2, color=INK)
    # chip placed BELOW the gate box so it doesn't overlap text
    chip(ax, gx + gw + 0.06, gy + gh / 2 - 0.13,
         "MoLE-style", fc=OURS_BG)

    # arrows gate → weighted-sum node (amber)
    arrow(ax, gx + gw + 1.42, gy + gh / 2, 9.05, 2.95,
          color=ACCENT, lw=1.3, style="-|>")

    # ── weighted-sum node ──
    sx, sy = 9.05, 2.80
    box(ax, sx, sy, 0.85, 0.65, OUT_BG, BORDER, lw=1.2)
    ax.text(sx + 0.42, sy + 0.33,
            "Σ gᵢ·Bᵢ", ha="center", va="center",
            fontsize=10.5, weight="bold", color=INK)

    # ── output ──
    ox, oy = 10.10, 2.55
    box(ax, ox, oy, 1.00, 1.10, OUT_BG, BORDER)
    ax.text(ox + 0.50, oy + 0.75, "ŷ", ha="center",
            fontsize=14, weight="bold", color=INK)
    ax.text(ox + 0.50, oy + 0.45, "(H,)", ha="center",
            fontsize=9, color=INK)
    ax.text(ox + 0.50, oy + 0.20, "H=24 / 168",
            ha="center", fontsize=7.2, color=FLOW, style="italic")
    arrow(ax, sx + 0.85, sy + 0.33, ox, sy + 0.33, lw=1.3)

    # ── footer legend ──
    ax.text(0.20, 0.10,
            "Borrowed: NLinear · DLinear · TSMixer · MoLE-style routing      |      "
            "Ours: heterogeneous branches + forecast-step conditioning + 24-h "
            "interpretable schedule",
            fontsize=7.6, color=FLOW, style="italic")

    out_png = os.path.join(out_dir, "fig_architecture_teaching.png")
    out_pdf = os.path.join(out_dir, "fig_architecture_teaching.pdf")
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
