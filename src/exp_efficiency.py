"""
src/exp_efficiency.py — Evidence B: accuracy/efficiency Pareto frontier.

Research question (plain English):
    "Best MAE" is only half the story. A model that shaves 0.6 Wh off the
    error but is 6× larger and 4× slower is not obviously "better" for anyone
    who has to deploy it. We measure, for every model, three costs alongside
    accuracy:
        • params      — model size
        • train_s     — wall-clock to fit
        • latency_ms  — inference time per sample (warm-up + timed loop)
    and plot the accuracy-vs-cost trade-off. A model is "Pareto-optimal" if no
    other model is both more accurate AND cheaper. The claim we can defend is
    not "we are #1 on MAE" but "we sit on the Pareto frontier" — i.e. nobody
    beats us on accuracy without paying more in size/latency.

Output: efficiency.csv (+ _weekly) ; fig_efficiency.png (MAE vs params,
        MAE vs latency, Pareto frontier highlighted).

Run
---
    python src/exp_efficiency.py
    python src/exp_efficiency.py --weekly
"""

import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import Config, make_config, get_loaders
from models import build_model
from training import (train, evaluate, persistence_reference,
                      save_csv, print_table, DEVICE)

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

CONTENDERS = [
    ("dlinear",       {}),
    ("itransformer",  {"d_model": 128, "n_layers": 2}),
    ("patchtst",      {"d_model": 64,  "n_layers": 2}),
    ("dpmixer",       {"d_channel": 64, "d_time": 128, "use_gating": True}),
    ("tcdpmixer",     {"time_feat_dim": 6}),
    ("tcssdpmixer",   {"time_feat_dim": 6}),
]


def _measure_latency(model, te_l, n_warmup=5, n_timed=30):
    """Return mean inference latency in ms PER SAMPLE."""
    model.eval()
    x, _ = next(iter(te_l))
    x = x.to(DEVICE)
    bs = x.size(0)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_timed):
            _ = model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
    return (dt / (n_timed * bs)) * 1e3   # ms / sample


def _pareto_mask(maes, costs):
    """Lower-is-better on both axes. Point i is on the frontier if no other
    point j is <= on both and < on at least one."""
    n = len(maes)
    on = [True] * n
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            if (maes[j] <= maes[i] and costs[j] <= costs[i] and
                    (maes[j] < maes[i] or costs[j] < costs[i])):
                on[i] = False
                break
    return on


def main(weekly=False):
    tag = " (WEEKLY: hourly, 1wk→1wk)" if weekly else ""
    print("\n" + "=" * 60)
    print("EVIDENCE B — Accuracy / efficiency Pareto frontier" + tag)
    print("=" * 60)
    print(f"Device: {DEVICE}")

    cfg = make_config(weekly,
                      feature_mode="no_rv",
                      add_time_features=True,
                      log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    print(f"\n  Setup: {cfg.feature_mode} + time + log,  n_feat={n_feat}")
    persistence_reference(te_l, scaler, "test", log_target=cfg.log_target)

    rows = []
    for name, kwargs in CONTENDERS:
        print(f"\n  [{name}]")
        torch.manual_seed(42); np.random.seed(42)
        model = build_model(name, cfg.seq_len, cfg.pred_len, n_feat,
                            use_residual=True, **kwargs)
        t0 = time.time()
        model = train(model, tr_l, vl_l, label=name, loss_name="l1")
        train_s = time.time() - t0
        m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
        npar = sum(p.numel() for p in model.parameters())
        lat = _measure_latency(model, te_l)
        rows.append({"config": name,
                     "test_MAE": round(m["MAE"], 3),
                     "test_RMSE": round(m["RMSE"], 3),
                     "params": npar,
                     "train_s": round(train_s, 1),
                     "latency_ms": round(lat, 4)})
        print(f"    MAE={m['MAE']:.3f}  params={npar:,}  "
              f"train={train_s:.1f}s  latency={lat:.4f} ms/sample")

    maes  = [r["test_MAE"]   for r in rows]
    pars  = [r["params"]     for r in rows]
    lats  = [r["latency_ms"] for r in rows]
    on_p  = _pareto_mask(maes, pars)
    on_l  = _pareto_mask(maes, lats)
    for r, p, l in zip(rows, on_p, on_l):
        r["pareto_params"]  = bool(p)
        r["pareto_latency"] = bool(l)

    ours = next((r for r in rows if r["config"] == "tcssdpmixer"), None)
    if ours is not None:
        flags = []
        if ours["pareto_params"]:  flags.append("size")
        if ours["pareto_latency"]: flags.append("latency")
        msg = (f"on the Pareto frontier for {', '.join(flags)}"
               if flags else "NOT on the frontier — a smaller+better model exists")
        print(f"\n  TAKEAWAY: tcssdpmixer is {msg}.")

    sfx = "_weekly" if weekly else ""
    save_csv(rows, os.path.join(RESULTS, f"efficiency{sfx}.csv"))
    print_table(rows, ["config", "test_MAE", "params",
                       "train_s", "latency_ms"])
    _plot(rows, weekly)
    print("\nDone.")


def _plot(rows, weekly):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    unit = "Wh/h" if weekly else "Wh"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, xlabel, mask_key in [
            (axes[0], "params",     "Params (log scale)",      "pareto_params"),
            (axes[1], "latency_ms", "Latency ms/sample (log)", "pareto_latency")]:
        for r in rows:
            ours = r["config"] == "tcssdpmixer"
            front = r[mask_key]
            ax.scatter(r[key], r["test_MAE"],
                       s=160 if ours else 90,
                       marker="*" if ours else ("o" if front else "x"),
                       color="#c0392b" if ours else ("#27ae60" if front else "#7f8c8d"),
                       zorder=3 if ours else 2)
            ax.annotate(r["config"], (r[key], r["test_MAE"]),
                        fontsize=8, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Test MAE ({unit}) — lower better")
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy vs cost — green=Pareto-optimal, red star=ours"
                 + (" (weekly)" if weekly else ""))
    fig.tight_layout()
    png = f"fig_efficiency{'_weekly' if weekly else ''}.png"
    fig.savefig(os.path.join(RESULTS, png), dpi=120)
    plt.close()
    print(f"  → saved {os.path.join(RESULTS, png)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true",
                    help="run the hourly 1-week→1-week long-horizon version")
    args = ap.parse_args()
    main(weekly=args.weekly)
