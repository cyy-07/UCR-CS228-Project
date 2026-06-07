"""
bench_lstm_latency.py — measure LSTM inference latency for the backup slide.

We never benched LSTM in `exp_efficiency.py` (it was only in the weekly
head-to-head as a regression baseline, not in the efficiency Pareto run).
This one-shot script fills that gap so we can put a real number next to
TC-DPMixer on the LSTM-vs-Transformer-vs-Mixer comparison slide.

Run on the GPU server (will use CUDA if available):

    python CS228/src/bench_lstm_latency.py

Output: prints LSTM latency in ms/sample for both short and weekly tasks,
alongside our flagship TC-DPMixer for direct comparison.
"""

import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders
from models import build_model
from training import DEVICE


def bench(model, loader, n_warmup=5, n_timed=50):
    """Return mean inference latency in ms PER SAMPLE."""
    model.eval()
    model.to(DEVICE)
    x, _ = next(iter(loader))
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


def main():
    print(f"Device: {DEVICE}")
    print(f"{'='*64}")
    print(f"{'Task':<10}  {'Model':<14}  {'L':<5} {'H':<5}  {'Latency (ms/sample)':>22}")
    print(f"{'='*64}")

    for weekly_flag, task_label in [(False, "short"), (True, "weekly")]:
        cfg = make_config(weekly_flag,
                          feature_mode="no_rv",
                          add_time_features=True,
                          log_target=True)
        tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)

        # ── LSTM ──
        torch.manual_seed(42)
        lstm = build_model("lstm", cfg.seq_len, cfg.pred_len, n_feat,
                           use_residual=True)
        lstm_lat = bench(lstm, te_l)
        n_lstm = sum(p.numel() for p in lstm.parameters())
        print(f"{task_label:<10}  {'LSTM':<14}  {cfg.seq_len:<5} {cfg.pred_len:<5}  "
              f"{lstm_lat:>18.4f}    ({n_lstm:,} params)")

        # ── TC-DPMixer (for direct comparison, same env / same batch shape) ──
        torch.manual_seed(42)
        tcdp = build_model("tcdpmixer", cfg.seq_len, cfg.pred_len, n_feat,
                           use_residual=True, time_feat_dim=6)
        tcdp_lat = bench(tcdp, te_l)
        n_tcdp = sum(p.numel() for p in tcdp.parameters())
        print(f"{task_label:<10}  {'TC-DPMixer':<14}  {cfg.seq_len:<5} {cfg.pred_len:<5}  "
              f"{tcdp_lat:>18.4f}    ({n_tcdp:,} params)")

        ratio = lstm_lat / tcdp_lat if tcdp_lat > 0 else float('inf')
        print(f"{task_label:<10}  → LSTM is {ratio:5.1f}× slower than TC-DPMixer on this task")
        print()


if __name__ == "__main__":
    main()
