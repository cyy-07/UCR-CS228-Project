"""
src/exp_gate_viz.py — visualise the learned gate policy g(t).

This is the interpretability money-shot. We train the time-conditioned gated
DLinear, then PROBE its gate network directly: hold everything else fixed and
sweep the hour-of-day through the time inputs. Because the gate depends ONLY
on the time features, g(t) is a clean, readable function — "at this hour the
model leans on the trend branch; at that hour it leans on the seasonal one."

Why probe synthetically instead of averaging over the test set?
    The gate is g(time_features) by construction, so feeding canonical
    (hour, weekday) encodings recovers the EXACT learned policy with no
    confounding from the other channels. It is the model's decision rule made
    visible.

Output: gate_policy.csv (+ _weekly) ; fig_gate_policy.png (+ .pdf)

Run
---
    python src/exp_gate_viz.py
    python src/exp_gate_viz.py --weekly
"""

import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders, TIME
from models import build_model
from training import train, DEVICE

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

TIME_FEAT_DIM = 6   # hour_sin, hour_cos, weekday_sin, weekday_cos, is_weekend, nsm


def _raw_time_features(hour: float, weekday: int) -> np.ndarray:
    """Build the 6 raw time features for a given (hour, weekday), in the same
    order as data_utils.TIME."""
    return np.array([
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * weekday / 7.0),
        np.cos(2 * np.pi * weekday / 7.0),
        1.0 if weekday >= 5 else 0.0,
        hour * 3600.0,                       # nsm (minute=0)
    ], dtype=np.float32)


def _gate_for(model, scaler, hour, weekday):
    """Return [g_trend, g_seasonal] the model would use at (hour, weekday)."""
    raw = _raw_time_features(hour, weekday)
    mean = scaler.mean_[-TIME_FEAT_DIM:]
    scale = scaler.scale_[-TIME_FEAT_DIM:]
    z = (raw - mean) / scale                 # standardise exactly as training
    tf = torch.tensor(z, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        g = torch.sigmoid(model.gate_net(tf)) * 2.0     # [1, 2] ∈ [0, 2]
    return g.squeeze(0).cpu().numpy()


def main(weekly=False):
    tag = " (WEEKLY)" if weekly else ""
    print("\n" + "=" * 60)
    print("INTERPRETABILITY — learned gate policy g(t)" + tag)
    print("=" * 60)

    cfg = make_config(weekly, feature_mode="no_rv",
                      add_time_features=True, log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    assert [c for c in TIME] , "time features must be enabled"

    print(f"  Device: {DEVICE}   n_feat={n_feat}")
    model = build_model("tcdlinear", cfg.seq_len, cfg.pred_len, n_feat,
                        use_residual=True, use_tc_gating=True,
                        time_feat_dim=TIME_FEAT_DIM)
    model = train(model, tr_l, vl_l, label="tcdlinear", loss_name="l1")
    model.eval()

    hours = np.arange(0, 24, 0.5)
    rows = []
    curves = {}   # (day_label, weekday) -> (g_trend[], g_seasonal[])
    for day_label, wd in [("weekday(Wed)", 2), ("weekend(Sun)", 6)]:
        gt, gs = [], []
        for h in hours:
            g = _gate_for(model, scaler, h, wd)
            gt.append(float(g[0])); gs.append(float(g[1]))
            rows.append({"day": day_label, "hour": round(float(h), 2),
                         "g_trend": round(g[0], 4),
                         "g_seasonal": round(g[1], 4)})
        curves[(day_label, wd)] = (np.array(gt), np.array(gs))

    # plain-English takeaway from the weekday curve
    gt_w, gs_w = curves[("weekday(Wed)", 2)]
    span_t = float(gt_w.max() - gt_w.min())
    span_s = float(gs_w.max() - gs_w.min())
    peak_trend_h = float(hours[int(np.argmax(gt_w))])
    peak_seas_h  = float(hours[int(np.argmax(gs_w))])
    print(f"\n  Gate swing over the day:  trend {span_t:.2f}, "
          f"seasonal {span_s:.2f}  (0 = constant/uninformative)")
    print(f"  Trusts TREND most around  {peak_trend_h:04.1f}h;  "
          f"SEASONAL most around {peak_seas_h:04.1f}h.")
    if max(span_t, span_s) < 0.05:
        print("  TAKEAWAY: gate is nearly flat — it learned a near-static mix.")
    else:
        print("  TAKEAWAY: the gate visibly re-weights branches across the "
              "day — a readable, time-driven decision rule.")

    from training import save_csv
    sfx = "_weekly" if weekly else ""
    save_csv(rows, os.path.join(RESULTS, f"gate_policy{sfx}.csv"))
    _plot(hours, curves, weekly)
    print("\nDone.")


def _plot(hours, curves, weekly):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    INK, STEEL, AMBER = "#1F2A37", "#3F6CA8", "#D99A2B"
    rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "savefig.bbox": "tight", "figure.facecolor": "white",
                     "axes.edgecolor": "#AEB8C2"})
    keys = list(curves.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(11.5, 4.3), sharey=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, (label, wd) in zip(axes, keys):
        gt, gs = curves[(label, wd)]
        ax.plot(hours, gt, color=STEEL, lw=2.4, label="trend branch  g₁")
        ax.plot(hours, gs, color=AMBER, lw=2.4, label="seasonal branch  g₂")
        ax.axhline(1.0, color="#C2C9D1", lw=1.0, ls="--", zorder=0)
        ax.set_title(label, fontsize=12, weight="bold", color=INK)
        ax.set_xlabel("hour of day")
        ax.set_xlim(0, 23.5); ax.set_xticks(range(0, 24, 4))
        ax.grid(True, color="#E3E8ED", lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("gate weight  g ∈ [0, 2]")
    axes[0].legend(loc="upper right", fontsize=9, frameon=True)
    fig.suptitle("Learned gate policy g(t): which branch the model trusts, "
                 "by time of day" + (" — weekly" if weekly else ""),
                 fontsize=13, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = os.path.join(RESULTS, "figures",
                       f"fig_gate_policy{'_weekly' if weekly else ''}.png")
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png); fig.savefig(png.replace(".png", ".pdf"))
    plt.close()
    print(f"  -> saved {png}  (+ .pdf)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    main(weekly=args.weekly)
