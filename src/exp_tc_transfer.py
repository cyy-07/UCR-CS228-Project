"""
src/exp_tc_transfer.py — Evidence C: is TC gating a reusable MECHANISM?

Research question (plain English):
    Our only genuinely-novel ingredient is time-conditioned (TC) gating: a
    tiny network reads the time-of-day / weekday features and decides, per
    sample, how much to trust each branch of the model. On DPMixer this buys
    ~0.4 Wh. The skeptic's reply: "that's a one-model trick."

    To answer that, we test the SAME mechanism on a SECOND, unrelated model
    family (a 2-branch DLinear) and on BOTH tasks (short 4-h and weekly
    1-week). Within each (family × task) cell we flip ONE switch:
        static learned gate   →   time-conditioned gate
    keeping everything else identical. If TC helps in most cells, the
    contribution is a transferable mechanism, not a lucky architecture.

Grid (4 cells per task × 2 tasks = 8 comparisons):
    DPMixer family : dpmixer(static gating)      vs  tcdpmixer(TC gating)
    DLinear family : tcdlinear(use_tc_gating=F)  vs  tcdlinear(use_tc_gating=T)

NOTE: this script runs BOTH tasks in one invocation — the cross-task
comparison IS the experiment — so it is NOT part of the per-task --weekly
suite. The --weekly flag is accepted for harness compatibility but ignored.

Output: tc_transfer.csv ; fig_tc_transfer.png (grouped bars, static vs TC).

Run
---
    python src/exp_tc_transfer.py
"""

import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import Config, make_config, get_loaders
from models import build_model
from training import train, evaluate, persistence_reference, save_csv, DEVICE

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

# Each pair: (family, static_spec, tc_spec)  where *_spec = (model_name, kwargs)
PAIRS = [
    ("DPMixer",
     ("dpmixer",  {"d_channel": 64, "d_time": 128, "use_gating": True}),
     ("tcdpmixer", {"time_feat_dim": 6})),
    ("DLinear",
     ("tcdlinear", {"use_tc_gating": False, "time_feat_dim": 6}),
     ("tcdlinear", {"use_tc_gating": True,  "time_feat_dim": 6})),
]


def _fit_eval(name, kwargs, cfg, tr_l, vl_l, te_l, scaler):
    torch.manual_seed(42); np.random.seed(42)
    model = build_model(name, cfg.seq_len, cfg.pred_len, cfg.n_feat,
                        use_residual=True, **kwargs)
    model = train(model, tr_l, vl_l, label=name, loss_name="l1")
    m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
    return m["MAE"]


def _run_task(weekly):
    cfg = make_config(weekly,
                      feature_mode="no_rv",
                      add_time_features=True,
                      log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    cfg.n_feat = n_feat
    task = "weekly" if weekly else "short"
    print(f"\n  --- task={task}  n_feat={n_feat} ---")
    persistence_reference(te_l, scaler, task, log_target=cfg.log_target)

    rows = []
    for family, static_spec, tc_spec in PAIRS:
        s_mae = _fit_eval(*static_spec, cfg, tr_l, vl_l, te_l, scaler)
        t_mae = _fit_eval(*tc_spec,     cfg, tr_l, vl_l, te_l, scaler)
        delta = s_mae - t_mae            # positive ⇒ TC is better
        helped = delta > 0
        rows.append({"task": task, "family": family,
                     "static_MAE": round(s_mae, 3),
                     "TC_MAE": round(t_mae, 3),
                     "delta_Wh": round(delta, 3),
                     "TC_helps": helped})
        arrow = "TC better" if helped else "TC worse"
        print(f"    {family:8s}  static={s_mae:7.3f}  TC={t_mae:7.3f}  "
              f"Δ={delta:+.3f}  → {arrow}")
    return rows


def main(weekly=False):
    print("\n" + "=" * 60)
    print("EVIDENCE C — Is time-conditioned gating a reusable mechanism?")
    print("=" * 60)
    print(f"Device: {DEVICE}  (runs BOTH short + weekly tasks)")

    rows = _run_task(weekly=False) + _run_task(weekly=True)

    n_cells = len(rows)
    n_win   = sum(r["TC_helps"] for r in rows)
    print("\n" + "-" * 60)
    print(f"  TC gating helped in {n_win}/{n_cells} (family × task) cells.")
    if n_win == n_cells:
        verdict = "TRANSFERS — TC helps every family on every task."
    elif n_win >= n_cells - 1:
        verdict = "MOSTLY TRANSFERS — helps in all but one cell."
    elif n_win > n_cells // 2:
        verdict = "PARTIALLY transfers — helps more often than not."
    else:
        verdict = "DOES NOT transfer — gains were model-specific."
    print(f"  TAKEAWAY: {verdict}")
    print("-" * 60)
    rows.append({"task": "VERDICT", "family": f"{n_win}/{n_cells}",
                 "delta_Wh": verdict})

    save_csv(rows, os.path.join(RESULTS, "tc_transfer.csv"))
    _plot([r for r in rows if r["task"] != "VERDICT"])
    print("\nDone.")


def _plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    labels = [f"{r['family']}\n({r['task']})" for r in rows]
    static = [r["static_MAE"] for r in rows]
    tc     = [r["TC_MAE"]     for r in rows]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, static, w, label="static gate", color="#7f8c8d")
    ax.bar(x + w/2, tc,     w, label="TC gate",     color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Test MAE — lower is better")
    ax.set_title("Static vs time-conditioned gate, two families × two tasks")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png = "fig_tc_transfer.png"
    fig.savefig(os.path.join(RESULTS, png), dpi=120)
    plt.close()
    print(f"  → saved {os.path.join(RESULTS, png)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true",
                    help="(ignored — this script always runs both tasks)")
    args = ap.parse_args()
    main(weekly=args.weekly)
