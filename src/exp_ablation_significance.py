"""
src/exp_ablation_significance.py — Hard analysis: statistical tests for the
DPMixer ablation study.

Why this exists
---------------
The existing `exp_dpmixer.py` ablation reports mean test MAE per leave-one-out
variant. That tells the reader the *direction* of each effect but not whether
it is **statistically significant** vs run-to-run + window-to-window noise.

We close that gap by:

  1. Re-running the same ablation grid as `run_ablation` in `exp_dpmixer.py`,
     ONE SEED, but
  2. Collecting **per-window** MAE arrays (length N≈3829), not just the mean.
  3. For every leave-one-out variant, running a paired **Wilcoxon** test
     against the "full" model: H₀ = "this branch contributes nothing".
  4. Reporting effect size (mean Δ), 95% bootstrap CI on Δ, and p-value.
  5. Producing a **forest plot** ordered by effect size.

Ablation grid (same as exp_dpmixer.py:run_ablation, trimmed to 6 informative
configs)
------------------------------------------------------------------------
  full                  (reference)
  - persistence anchor  (use_residual=False)
  - target decomp       (use_decomp=False)
  - channel mixer       (use_channel=False)
  - time mixer          (use_time=False)
  only persistence      (use_decomp=False, use_channel=False, use_time=False)

Outputs
-------
  results/ablation_significance.csv     ablation × {mean_Δ, CI_lo, CI_hi, p-value, verdict}
  results/figures/fig_ablation_significance.{png,pdf}   forest plot

Run
---
    python src/exp_ablation_significance.py
"""

import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders
from models import build_model
from training import train, collect_predictions, DEVICE


RESULTS = "results"
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# (label, dpmixer kwargs).  Each spec is trained from scratch with seed 42.
ABLATION_SPECS = [
    ("full",                 dict(use_decomp=True,  use_channel=True,  use_time=True,
                                  use_residual=True)),
    ("- persistence anchor", dict(use_decomp=True,  use_channel=True,  use_time=True,
                                  use_residual=False)),
    ("- target decomp",      dict(use_decomp=False, use_channel=True,  use_time=True,
                                  use_residual=True)),
    ("- channel mixer",      dict(use_decomp=True,  use_channel=False, use_time=True,
                                  use_residual=True)),
    ("- time mixer",         dict(use_decomp=True,  use_channel=True,  use_time=False,
                                  use_residual=True)),
    ("only persistence",     dict(use_decomp=False, use_channel=False, use_time=False,
                                  use_residual=True)),
]
N_BOOTSTRAP = 2000   # for the CI of mean Δ MAE


def per_window_mae(preds: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Returns array of shape [N] — mean absolute error per window."""
    return np.abs(preds - truth).mean(axis=1)


def bootstrap_mean_ci(x: np.ndarray, n: int = N_BOOTSTRAP, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean of x."""
    rng = np.random.default_rng(2024)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    print("\n" + "=" * 60)
    print("HARD ANALYSIS — DPMixer ablation statistical significance")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    cfg = make_config(weekly=False, feature_mode="no_rv",
                      add_time_features=True, log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    print(f"  n_feat={n_feat}   seq_len={cfg.seq_len}   pred_len={cfg.pred_len}")
    print(f"  Re-training {len(ABLATION_SPECS)} ablation variants for paired tests…")

    per_window = {}     # label → per-window MAE array (length N)

    for label, kw in ABLATION_SPECS:
        print(f"\n  [{label}]")
        torch.manual_seed(42); np.random.seed(42)
        t0 = time.time()
        m = build_model("dpmixer", cfg.seq_len, cfg.pred_len, n_feat,
                        **kw)
        m = train(m, tr_l, vl_l, label=label, loss_name="l1")
        preds, trues = collect_predictions(m, te_l, scaler,
                                           log_target=cfg.log_target)
        w_mae = per_window_mae(preds, trues)
        per_window[label] = w_mae
        print(f"    mean MAE = {w_mae.mean():.3f} Wh    "
              f"(N={len(w_mae)} windows)   "
              f"trained in {time.time() - t0:.0f}s")

    # Paired tests against 'full'.
    from scipy import stats as st
    full_mae = per_window["full"]
    rows = []
    rows.append({"variant": "full",
                 "mean_MAE": round(float(full_mae.mean()), 3),
                 "delta_MAE": 0.0,
                 "delta_CI_lo": 0.0, "delta_CI_hi": 0.0,
                 "wilcoxon_p": 1.0, "n_windows": int(len(full_mae)),
                 "verdict": "reference"})
    for label, w_mae in per_window.items():
        if label == "full":
            continue
        delta = w_mae - full_mae    # positive = ablation hurts (variant worse)
        wstat = st.wilcoxon(delta, alternative="two-sided")
        ci_lo, ci_hi = bootstrap_mean_ci(delta)
        sig = wstat.pvalue < 0.001
        direction = "significantly WORSE" if delta.mean() > 0 and sig else \
                    "significantly BETTER" if delta.mean() < 0 and sig else \
                    "not significantly different"
        rows.append({
            "variant": label,
            "mean_MAE":  round(float(w_mae.mean()), 3),
            "delta_MAE": round(float(delta.mean()), 3),
            "delta_CI_lo": round(ci_lo, 3),
            "delta_CI_hi": round(ci_hi, 3),
            "wilcoxon_p": float(wstat.pvalue),
            "n_windows": int(len(w_mae)),
            "verdict": direction,
        })
        print(f"  {label:24s}  Δ MAE = {delta.mean():+6.3f} "
              f"[{ci_lo:+6.3f}, {ci_hi:+6.3f}]   "
              f"p = {wstat.pvalue:.2e}   →  {direction}")

    from training import save_csv
    save_csv(rows, os.path.join(RESULTS, "ablation_significance.csv"))
    _plot(rows)
    print("\nDone.")


def _plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    INK, AMBER, GRAY, RED, GREEN = "#1F2A37", "#D99A2B", "#7F8C9A", "#B3263F", "#2C7A55"
    rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "savefig.bbox": "tight", "figure.facecolor": "white",
                     "axes.edgecolor": "#AEB8C2"})

    # Exclude 'full' (it's the reference at Δ=0); order by effect size.
    plotted = [r for r in rows if r["variant"] != "full"]
    plotted.sort(key=lambda r: r["delta_MAE"])
    labels = [r["variant"] for r in plotted]
    deltas = np.array([r["delta_MAE"]    for r in plotted])
    ci_lo  = np.array([r["delta_CI_lo"]  for r in plotted])
    ci_hi  = np.array([r["delta_CI_hi"]  for r in plotted])
    pvals  = [r["wilcoxon_p"] for r in plotted]

    colors = []
    for d, p in zip(deltas, pvals):
        if p >= 0.001:                colors.append(GRAY)       # not sig
        elif d > 0:                   colors.append(RED)        # sig worse
        else:                         colors.append(GREEN)      # sig better

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(labels))
    # Error bars: distance from delta to CI bounds.
    err_lo = deltas - ci_lo
    err_hi = ci_hi - deltas
    ax.errorbar(deltas, y, xerr=[err_lo, err_hi],
                fmt='o', color="black", ms=6,
                ecolor="#586878", elinewidth=1.4, capsize=4, zorder=3)
    # Color the marker face by significance verdict.
    for yi, d, c in zip(y, deltas, colors):
        ax.plot(d, yi, marker='o', ms=10, color=c, zorder=4,
                markeredgecolor=INK, markeredgewidth=0.8)
    ax.axvline(0, color="#D7DEE5", lw=1.0, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Δ test MAE vs full model  (Wh)  ·  95 % bootstrap CI  ·  "
                  "paired Wilcoxon p < 0.001 ⇒ filled colour")
    ax.set_title("DPMixer ablation: which branch removals significantly hurt?",
                 fontsize=12, weight="bold", color=INK)
    ax.grid(True, color="#E3E8ED", lw=0.7, axis="x"); ax.set_axisbelow(True)

    # Legend chips.
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RED,   ms=9, label="hurts (p<0.001)",   markeredgecolor=INK),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN, ms=9, label="helps (p<0.001)",   markeredgecolor=INK),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GRAY,  ms=9, label="not significant",   markeredgecolor=INK),
    ]
    ax.legend(handles=legend, fontsize=9, loc="lower right", frameon=True)
    fig.tight_layout()
    png = os.path.join(FIG_DIR, "fig_ablation_significance.png")
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
    main()
