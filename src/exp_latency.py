"""
src/exp_latency.py  —  Reviewer R4: defensible inference-latency claims.

What it does
------------
The old `exp_efficiency.py` measures latency with the test-loader's batch
size (256) and a 5-warm-up / 30-timed loop, then divides by sample count.
That number can't support "57× faster per window" in the abstract, because:
  • per-sample latency on bs=256 hides serial overhead,
  • 30 timed runs has noisy std,
  • we never recorded GPU / CUDA / PyTorch / eager-vs-compiled.

This script measures the right thing:
  • two regimes: batch=1 (single-window deploy) and batch=32 (mini-batch),
  • 100 warm-up forwards followed by 1000 timed forwards,
  • full hardware/software metadata dumped to results/latency_meta.json,
  • per-model rows: mean / std / median / p99 of latency_ms_per_window.

Run
---
    python src/exp_latency.py
    python src/exp_latency.py --weekly
"""

import os, sys, time, json, platform, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders
from models import build_model
from training import train, DEVICE

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

CONTENDERS = [
    ("dlinear",      {}),
    ("itransformer", {"d_model": 128, "n_layers": 2}),
    ("patchtst",     {"d_model": 64,  "n_layers": 2}),
    ("dpmixer",      {"d_channel": 64, "d_time": 128, "use_gating": True}),
    ("tcdpmixer",    {"time_feat_dim": 6}),
    ("tcssdpmixer",  {"time_feat_dim": 6}),
]

BATCH_SIZES = [1, 32]
N_WARMUP    = 100
N_TIMED     = 1000


def _hw_meta():
    meta = {
        "python":         platform.python_version(),
        "platform":       platform.platform(),
        "torch":          torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "execution_mode": "eager",
        "n_warmup":       N_WARMUP,
        "n_timed":        N_TIMED,
    }
    if torch.cuda.is_available():
        meta["cuda_version"] = torch.version.cuda
        meta["gpu_name"]     = torch.cuda.get_device_name(0)
        meta["gpu_count"]    = torch.cuda.device_count()
    return meta


def _measure(model, x, bs):
    """Return ms-per-window vector of length N_TIMED."""
    x_b = x[:bs].contiguous().to(DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(x_b)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times = np.empty(N_TIMED, dtype=np.float64)
        for k in range(N_TIMED):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x_b)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times[k] = (time.perf_counter() - t0) * 1e3 / bs   # ms/window
    return times


def main(weekly=False):
    tag = " (WEEKLY)" if weekly else " (SHORT)"
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK" + tag)
    print("=" * 60)
    meta = _hw_meta()
    for k, v in meta.items():
        print(f"  {k:18s} = {v}")
    print(f"  batch_sizes        = {BATCH_SIZES}")

    cfg = make_config(weekly, feature_mode="no_rv",
                      add_time_features=True, log_target=True)
    tr_l, vl_l, te_l, _, _ = get_loaders(cfg)
    x, _ = next(iter(te_l))   # [B, L, C]

    rows = []
    for name, kwargs in CONTENDERS:
        torch.manual_seed(0); np.random.seed(0)
        model = build_model(name, cfg.seq_len, cfg.pred_len, cfg.n_feat,
                            use_residual=True, **kwargs).to(DEVICE)
        # short warm-up train just to ensure params are realistic
        model = train(model, tr_l, vl_l, label=f"{name}-warm",
                      loss_name="l1", seed=0, epochs=2)
        npar = sum(p.numel() for p in model.parameters())

        for bs in BATCH_SIZES:
            if x.size(0) < bs:
                print(f"  ! batch {bs} > test set; skipping")
                continue
            t = _measure(model, x, bs)
            row = {
                "model": name, "batch": bs, "params": npar,
                "mean_ms":   float(t.mean()),
                "std_ms":    float(t.std(ddof=1)),
                "median_ms": float(np.median(t)),
                "p99_ms":    float(np.percentile(t, 99)),
            }
            rows.append(row)
            print(f"  {name:14s} bs={bs:3d}  mean={row['mean_ms']:.4f} ± "
                  f"{row['std_ms']:.4f}  med={row['median_ms']:.4f}  "
                  f"p99={row['p99_ms']:.4f}  (params={npar})")

    out_csv  = os.path.join(RESULTS,
                            f"latency{'_weekly' if weekly else ''}.csv")
    out_meta = os.path.join(RESULTS,
                            f"latency{'_weekly' if weekly else ''}_meta.json")
    from training import save_csv
    save_csv(rows, out_csv)
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  → {out_csv}")
    print(f"  → {out_meta}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weekly", action="store_true")
    a = p.parse_args()
    main(weekly=a.weekly)
