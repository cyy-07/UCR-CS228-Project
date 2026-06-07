"""
src/exp_dpmixer.py — Workstream G: DPMixer family (proposed architectures)

This script runs the following sections in one go:

  1. Branch ablation     — remove each of the 4 branches one-by-one
  2. Capacity sweep      — DPMixer tiny / small / medium / large
  3. Static gating       — gated vs ungated DPMixer
  4. Time-conditioned    — TC-DPMixer (Idea A; MoLE-style routing extended
                           to heterogeneous additive branches)
  5. Spike-aware         — SS-DPMixer (Idea C; NILM-inspired auxiliary
                           heads), and TC-SS-DPMixer = A + C combined
  6. Head-to-head        — best DPMixer variant vs DLinear/iTransformer/PatchTST

Honest novelty positioning (see literature review in lit/lit_dpmixer.md):
  • Branches 1–4 are all derivative of NLinear / DLinear / TSMixer.
  • TC-DPMixer is closely related to MoLE (Ni et al., AISTATS 2024); our
    angle is heterogeneous additive branches + forecast-step conditioning.
  • SS-DPMixer borrows NILM s2p (Zhang AAAI 2018) classify+magnitude heads,
    applied to forecasting rather than disaggregation — a fresh use case.
  • TC-SS-DPMixer = combined A + C is the defensible novel contribution.

Run
---
    python src/exp_dpmixer.py
    nohup python -u src/exp_dpmixer.py > results/log_dpmixer.txt 2>&1 &
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


# ──────────────────────────────────────────────────────────────
#  Section 1 — Branch ablation
#  Turn off branches one-by-one to verify each contributes
# ──────────────────────────────────────────────────────────────

BRANCH_ABLATION = [
    # label,                use_decomp, use_channel, use_time, use_residual
    ("full",                True,  True,  True,  True),
    ("- persistence anchor",True,  True,  True,  False),
    ("- target decomp",     False, True,  True,  True),
    ("- channel mixer",     True,  False, True,  True),
    ("- time mixer",        True,  True,  False, True),
    ("only persistence",    False, False, False, True),
    ("only decomp",         True,  False, False, False),
    ("only channel",        False, True,  False, False),
    ("only time",           False, False, True,  False),
]


def run_branch_ablation(cfg, tr_l, vl_l, te_l, scaler, n_feat,
                        loss_name="l1"):
    print("\n" + "=" * 60)
    print("SECTION 1 — DPMixer branch ablation")
    print("=" * 60)
    rows = []
    for label, dec, ch, tm, res in BRANCH_ABLATION:
        print(f"\n  [{label}]  decomp={dec} channel={ch} time={tm} pers={res}")
        t0 = time.time()
        model = build_model("dpmixer", cfg.seq_len, cfg.pred_len, n_feat,
                            use_residual=res, use_decomp=dec,
                            use_channel=ch, use_time=tm)
        model = train(model, tr_l, vl_l, label=f"dpmixer[{label}]",
                      loss_name=loss_name)
        m = evaluate(model, te_l, scaler,
                     log_target=cfg.log_target)
        npar = sum(p.numel() for p in model.parameters())
        row = {"section": "ablation", "config": label,
               "test_MAE": round(m["MAE"], 3),
               "test_RMSE": round(m["RMSE"], 3),
               "params": npar,
               "train_s": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"    test_MAE={m['MAE']:.3f}  test_RMSE={m['RMSE']:.3f}  "
              f"params={npar:,}")
    return rows


# ──────────────────────────────────────────────────────────────
#  Section 2 — Capacity sweep
# ──────────────────────────────────────────────────────────────

CAPACITY_SWEEP = [
    # label,         d_channel, d_time
    ("tiny",    32,  64),
    ("small",   64, 128),
    ("medium", 128, 256),
    ("large",  256, 512),
]


def run_capacity_sweep(cfg, tr_l, vl_l, te_l, scaler, n_feat,
                       loss_name="l1"):
    print("\n" + "=" * 60)
    print("SECTION 2 — DPMixer capacity sweep")
    print("=" * 60)
    rows = []
    for label, dc, dt in CAPACITY_SWEEP:
        print(f"\n  [dpmixer-{label}]  d_channel={dc}  d_time={dt}")
        t0 = time.time()
        model = build_model("dpmixer", cfg.seq_len, cfg.pred_len, n_feat,
                            d_channel=dc, d_time=dt, use_residual=True)
        model = train(model, tr_l, vl_l, label=f"dpmixer-{label}",
                      loss_name=loss_name)
        m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
        npar = sum(p.numel() for p in model.parameters())
        row = {"section": "capacity", "config": f"dpmixer-{label}",
               "test_MAE": round(m["MAE"], 3),
               "test_RMSE": round(m["RMSE"], 3),
               "params": npar,
               "train_s": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"    test_MAE={m['MAE']:.3f}  test_RMSE={m['RMSE']:.3f}  "
              f"params={npar:,}")
    return rows


# ──────────────────────────────────────────────────────────────
#  Section 3 — Head-to-head vs strongest prior model
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
#  Section 4b — Time-Conditioned gating  (Idea A)
# ──────────────────────────────────────────────────────────────

def run_tc_variant(cfg, tr_l, vl_l, te_l, scaler, n_feat, loss_name="l1"):
    """
    Time-conditioned gating ablation:
      • static-gated DPMixer  (4 scalars learned globally)
      • TC-DPMixer            (4 gates per sample, conditioned on time feats)
    """
    print("\n" + "=" * 60)
    print("SECTION 4b — Time-Conditioned gating (TC-DPMixer, Idea A)")
    print("=" * 60)
    print("  NOTE: requires add_time_features=True in Config.")
    rows = []
    contenders = [
        ("dpmixer-static-gate", "dpmixer",
            {"use_gating": True, "use_residual": True}),
        ("dpmixer-tc-gate",    "tcdpmixer",
            {"time_feat_dim": 6, "use_residual": True}),
    ]
    for label, name, kwargs in contenders:
        print(f"\n  [{label}]")
        t0 = time.time()
        model = build_model(name, cfg.seq_len, cfg.pred_len, n_feat, **kwargs)
        model = train(model, tr_l, vl_l, label=label, loss_name=loss_name)
        m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
        npar = sum(p.numel() for p in model.parameters())
        row = {"section": "tc_gating", "config": label,
               "test_MAE":  round(m["MAE"], 3),
               "test_RMSE": round(m["RMSE"], 3),
               "params":    npar,
               "train_s":   round(time.time() - t0, 1)}
        rows.append(row)
        print(f"    test_MAE={m['MAE']:.3f}  test_RMSE={m['RMSE']:.3f}  "
              f"params={npar:,}")
        # Empirical 24-h schedule: collect gates per sample on the val set,
        # then group by the hour-of-day inferred from hour_sin / hour_cos.
        if hasattr(model, "_compute_gates"):
            model.eval()
            hours_all, gates_all = [], []
            with torch.no_grad():
                for x, _ in vl_l:
                    x = x.to(DEVICE)
                    g = model._compute_gates(x).cpu().numpy()    # [B, 4]
                    # hour_sin, hour_cos are at indices -6, -5 (TIME order
                    # in data_utils: hour_sin, hour_cos, weekday_sin, ...)
                    hs = x[:, -1, -6].cpu().numpy()
                    hc = x[:, -1, -5].cpu().numpy()
                    hr = (np.arctan2(hs, hc) * 24 / (2 * np.pi)) % 24
                    hours_all.append(hr); gates_all.append(g)
            hours_all = np.concatenate(hours_all)                  # [N]
            gates_all = np.concatenate(gates_all)                  # [N, 4]
            # bucket by integer hour
            sched = np.zeros((24, 4))
            for h in range(24):
                mask = (hours_all >= h) & (hours_all < h + 1)
                if mask.sum() > 0:
                    sched[h] = gates_all[mask].mean(axis=0)
            print("    empirical 24h gate schedule (val set) — selected hours:")
            for h in [3, 7, 12, 18, 22]:
                p_, d_, c_, t_ = sched[h]
                print(f"      h={h:02d}  pers={p_:.2f}  decomp={d_:.2f}  "
                      f"chan={c_:.2f}  time={t_:.2f}")
            sfx = "_weekly" if cfg.resample is not None else ""
            np.save(os.path.join(RESULTS,
                                 f"tcdpmixer_schedule{sfx}.npy"), sched)
    return rows


# ──────────────────────────────────────────────────────────────
#  Section 4c — Spike-aware two-stream  (Idea C, NILM-inspired)
# ──────────────────────────────────────────────────────────────

def run_spike_variants(cfg, tr_l, vl_l, te_l, scaler, n_feat, loss_name="l1"):
    """
    Spike-aware variants:
      • SS-DPMixer    (base + spike streams, no time conditioning)
      • TC-SS-DPMixer (combined Idea A + C — our headline contribution)
    """
    print("\n" + "=" * 60)
    print("SECTION 4c — Spike-aware variants (Idea C + A+C combo)")
    print("=" * 60)
    rows = []
    contenders = [
        ("ss-dpmixer",     "ssdpmixer",   {}),
        ("tc-ss-dpmixer",  "tcssdpmixer", {"time_feat_dim": 6}),
    ]
    for label, name, kwargs in contenders:
        print(f"\n  [{label}]")
        t0 = time.time()
        model = build_model(name, cfg.seq_len, cfg.pred_len, n_feat, **kwargs)
        model = train(model, tr_l, vl_l, label=label, loss_name=loss_name)
        m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
        npar = sum(p.numel() for p in model.parameters())
        row = {"section": "spike", "config": label,
               "test_MAE":  round(m["MAE"], 3),
               "test_RMSE": round(m["RMSE"], 3),
               "params":    npar,
               "train_s":   round(time.time() - t0, 1)}
        rows.append(row)
        print(f"    test_MAE={m['MAE']:.3f}  test_RMSE={m['RMSE']:.3f}  "
              f"params={npar:,}")
    return rows


# ──────────────────────────────────────────────────────────────
def run_head_to_head(cfg, tr_l, vl_l, te_l, scaler, n_feat, loss_name="l1"):
    print("\n" + "=" * 60)
    print("SECTION 3 — DPMixer vs prior best, same training recipe")
    print("=" * 60)
    rows = []
    contenders = [
        ("dlinear",       {}),
        ("itransformer",  {"d_model": 128, "n_layers": 2}),
        ("patchtst",      {"d_model": 64,  "n_layers": 2}),
        ("dpmixer",       {"d_channel": 64, "d_time": 128, "use_gating": True}),
        ("tcdpmixer",     {"time_feat_dim": 6}),
        ("ssdpmixer",     {}),
        ("tcssdpmixer",   {"time_feat_dim": 6}),
    ]
    for name, kwargs in contenders:
        print(f"\n  [{name}]")
        t0 = time.time()
        model = build_model(name, cfg.seq_len, cfg.pred_len, n_feat,
                            use_residual=True, **kwargs)
        model = train(model, tr_l, vl_l, label=name, loss_name=loss_name)
        m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
        npar = sum(p.numel() for p in model.parameters())
        row = {"section": "head2head", "config": name,
               "test_MAE": round(m["MAE"], 3),
               "test_RMSE": round(m["RMSE"], 3),
               "params": npar,
               "train_s": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"    test_MAE={m['MAE']:.3f}  test_RMSE={m['RMSE']:.3f}  "
              f"params={npar:,}")
        # Report DPMixer's learned branch weights — interpretability bonus.
        # Keep gates OUT of the CSV row (heterogeneous keys break DictWriter);
        # dump to a separate JSON sidecar instead.
        if hasattr(model, "report_gates"):
            g = model.report_gates()
            if g:
                print(f"    learned gates: " +
                      "  ".join(f"{k}={v:+.3f}" for k, v in g.items()))
                import json
                with open(os.path.join(RESULTS,
                                       f"gates_{name}.json"), "w") as fh:
                    json.dump(g, fh, indent=2)
    return rows


# ──────────────────────────────────────────────────────────────
#  Section 4 — gating on/off ablation
# ──────────────────────────────────────────────────────────────

def run_gating_ablation(cfg, tr_l, vl_l, te_l, scaler, n_feat,
                        loss_name="l1"):
    print("\n" + "=" * 60)
    print("SECTION 4 — Learned-gating ablation (our contribution)")
    print("=" * 60)
    rows = []
    for use_g in (False, True):
        tag = "gated" if use_g else "ungated (sum)"
        print(f"\n  [{tag}]")
        t0 = time.time()
        model = build_model("dpmixer", cfg.seq_len, cfg.pred_len, n_feat,
                            use_residual=True, use_gating=use_g)
        model = train(model, tr_l, vl_l, label=f"dpmixer[{tag}]",
                      loss_name=loss_name)
        m = evaluate(model, te_l, scaler, log_target=cfg.log_target)
        npar = sum(p.numel() for p in model.parameters())
        row = {"section": "gating", "config": tag,
               "test_MAE": round(m["MAE"], 3),
               "test_RMSE": round(m["RMSE"], 3),
               "params": npar,
               "train_s": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"    test_MAE={m['MAE']:.3f}  test_RMSE={m['RMSE']:.3f}  "
              f"params={npar:,}")
        if use_g and hasattr(model, "report_gates"):
            g = model.report_gates()
            print(f"    learned gates: " +
                  "  ".join(f"{k}={v:+.3f}" for k, v in g.items()))
    return rows


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main(weekly=False):
    tag = " (WEEKLY: hourly, 1wk→1wk)" if weekly else ""
    print("\n" + "=" * 60)
    print("WORKSTREAM G — DPMixer (proposed architecture)" + tag)
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Best feature configuration from Workstream A:
    # no_rv + time features + log target, L1 loss
    cfg = make_config(weekly,
                      feature_mode="no_rv",
                      add_time_features=True,
                      log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    print(f"\n  Setup: {cfg.feature_mode} + time + log,  n_feat={n_feat}")
    persistence_reference(te_l, scaler, "test", log_target=cfg.log_target)

    rows = []
    rows += run_branch_ablation(cfg, tr_l, vl_l, te_l, scaler, n_feat,
                                loss_name="l1")
    rows += run_capacity_sweep(cfg, tr_l, vl_l, te_l, scaler, n_feat,
                               loss_name="l1")
    rows += run_gating_ablation(cfg, tr_l, vl_l, te_l, scaler, n_feat,
                                loss_name="l1")
    rows += run_tc_variant(cfg, tr_l, vl_l, te_l, scaler, n_feat,
                           loss_name="l1")
    rows += run_spike_variants(cfg, tr_l, vl_l, te_l, scaler, n_feat,
                               loss_name="l1")
    rows += run_head_to_head(cfg, tr_l, vl_l, te_l, scaler, n_feat,
                             loss_name="l1")

    out = "dpmixer_weekly.csv" if weekly else "dpmixer.csv"
    save_csv(rows, os.path.join(RESULTS, out))
    print_table(rows, ["section","config","test_MAE","test_RMSE","params"])
    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true",
                    help="run the hourly 1-week→1-week long-horizon version")
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    main(weekly=args.weekly)
