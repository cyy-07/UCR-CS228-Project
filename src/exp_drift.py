"""
src/exp_drift.py — Evidence D: robustness under distribution shift.

Research question (plain English):
    A single test-set MAE assumes "the future looks like the average of the
    test window." Real deployments drift: weather turns, occupancy changes,
    seasons move. A model with a slightly worse average MAE but that stays
    STABLE as conditions change can be the safer choice. We split the
    (chronological, un-shuffled) test set into K equal time segments and
    report each model's MAE per segment. The "degradation ratio" =
    MAE(last segment) / MAE(first segment) measures how much a model rots as
    it forecasts further from its training distribution. Lower = more robust.

Output: drift.csv (+ _weekly) ; fig_drift.png (MAE vs test-time segment).

Run
---
    python src/exp_drift.py
    python src/exp_drift.py --weekly
"""

import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import Config, make_config, get_loaders
from models import build_model
from training import (train, collect_predictions, persistence_reference,
                      save_csv, print_table, DEVICE)

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

K_SEGMENTS = 5

CONTENDERS = [
    ("dlinear",       {}),
    ("itransformer",  {"d_model": 128, "n_layers": 2}),
    ("patchtst",      {"d_model": 64,  "n_layers": 2}),
    ("dpmixer",       {"d_channel": 64, "d_time": 128, "use_gating": True}),
    ("tcssdpmixer",   {"time_feat_dim": 6}),
]


def _segment_mae(preds, trues, k):
    """preds/trues: [N, H] in chronological order. Return list of K MAEs over
    contiguous chunks of windows (per-element MAE within each chunk)."""
    n = len(preds)
    bounds = np.linspace(0, n, k + 1, dtype=int)
    out = []
    for i in range(k):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            out.append(float("nan")); continue
        out.append(float(np.mean(np.abs(preds[a:b] - trues[a:b]))))
    return out


def main(weekly=False):
    tag = " (WEEKLY: hourly, 1wk→1wk)" if weekly else ""
    print("\n" + "=" * 60)
    print("EVIDENCE D — Robustness under distribution shift" + tag)
    print("=" * 60)
    print(f"Device: {DEVICE}   K={K_SEGMENTS} chronological test segments")

    cfg = make_config(weekly,
                      feature_mode="no_rv",
                      add_time_features=True,
                      log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    print(f"\n  Setup: {cfg.feature_mode} + time + log,  n_feat={n_feat}")
    persistence_reference(te_l, scaler, "test", log_target=cfg.log_target)

    rows = []
    seg_keys = [f"seg{i+1}_MAE" for i in range(K_SEGMENTS)]
    for name, kwargs in CONTENDERS:
        print(f"\n  [{name}]")
        torch.manual_seed(42); np.random.seed(42)
        model = build_model(name, cfg.seq_len, cfg.pred_len, n_feat,
                            use_residual=True, **kwargs)
        model = train(model, tr_l, vl_l, label=name, loss_name="l1")
        preds, trues = collect_predictions(model, te_l, scaler,
                                           log_target=cfg.log_target)
        segs = _segment_mae(preds, trues, K_SEGMENTS)
        degr = segs[-1] / segs[0] if segs[0] else float("nan")
        row = {"config": name}
        row.update({k: round(v, 3) for k, v in zip(seg_keys, segs)})
        row["overall_MAE"] = round(float(np.mean(np.abs(preds - trues))), 3)
        row["degradation"] = round(degr, 3)
        rows.append(row)
        print("    segs=[" + ", ".join(f"{v:.2f}" for v in segs) + "]"
              f"  degradation(last/first)={degr:.3f}")

    learned = [r for r in rows]
    most_robust = min(learned, key=lambda r: r["degradation"])
    ours = next((r for r in rows if r["config"] == "tcssdpmixer"), None)
    print("\n" + "-" * 60)
    print(f"  Most robust (lowest degradation): {most_robust['config']} "
          f"({most_robust['degradation']:.3f})")
    if ours is not None:
        rank = sorted(rows, key=lambda r: r["degradation"]).index(ours) + 1
        print(f"  TAKEAWAY: tcssdpmixer degradation={ours['degradation']:.3f}, "
              f"ranked #{rank}/{len(rows)} for stability across the test span.")
    print("-" * 60)

    sfx = "_weekly" if weekly else ""
    save_csv(rows, os.path.join(RESULTS, f"drift{sfx}.csv"))
    print_table(rows, ["config"] + seg_keys + ["degradation"])
    _plot(rows, seg_keys, weekly)
    print("\nDone.")


def _plot(rows, seg_keys, weekly):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    unit = "Wh/h" if weekly else "Wh"
    x = np.arange(1, len(seg_keys) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in rows:
        ys = [r[k] for k in seg_keys]
        ours = r["config"] == "tcssdpmixer"
        ax.plot(x, ys, marker="o",
                lw=3 if ours else 1.5,
                color="#c0392b" if ours else None,
                zorder=3 if ours else 2,
                label=r["config"])
    ax.set_xticks(x)
    ax.set_xlabel("Test-time segment (early → late, chronological)")
    ax.set_ylabel(f"MAE ({unit}) — lower is better")
    ax.set_title("Does accuracy degrade over the test span?"
                 + (" (weekly)" if weekly else ""))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = f"fig_drift{'_weekly' if weekly else ''}.png"
    fig.savefig(os.path.join(RESULTS, png), dpi=120)
    plt.close()
    print(f"  → saved {os.path.join(RESULTS, png)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true",
                    help="run the hourly 1-week→1-week long-horizon version")
    args = ap.parse_args()
    main(weekly=args.weekly)
