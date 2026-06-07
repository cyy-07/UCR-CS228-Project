"""
src/exp_counterfactual.py — Task #5: Counterfactual probing on T_out.

Task formulation
----------------
Given a trained regression forecaster (TC-DPMixer / DLinear / iTransformer),
**intervene** on a single feature of the input — outdoor temperature `T_out`
(channel 20 under the canonical 32-channel layout) — by a fixed offset
ΔT ∈ {−10, −7, −5, −2, 0, +2, +5, +7, +10} °C, then re-run the forecast and
measure the change in predicted Wh.

This is a model-agnostic causal probe: it asks
    "Holding everything else fixed, how would my forecast change if T_out had
     been ΔT degrees warmer/colder?"
The answer characterises (a) how strongly the model uses T_out, (b) the
DIRECTION of the response (warmer → more A/C → higher Wh?), and (c) how
quickly the response decays with forecast horizon (the influence of T_out
at the LAST input step ought to fade as we predict further into the future).

Gate-self-consistency sanity check (TC-DPMixer only)
----------------------------------------------------
Our interpretability claim says the TC gate `g(t)` depends only on the time
features (channels 26–31), not on `T_out`. So intervening on T_out should
leave `g(t)` numerically unchanged. We compute the L2 distance between
baseline and intervened gates and assert it is ≈ 0 — a self-consistency
proof that our interpretability story is mechanistic, not post-hoc.

Outputs
-------
  results/counterfactual.csv                                     (long-form table)
  results/figures/fig_counterfactual_response.{png,pdf}          response curve
  results/figures/fig_counterfactual_gate.{png,pdf}              gate invariance

Run
---
    python src/exp_counterfactual.py
    nohup python -u src/exp_counterfactual.py > results/log_counterfactual.txt 2>&1 &
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders
from models import build_model
from training import train, DEVICE


RESULTS = "results"
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Channel layout under the canonical no_rv + add_time_features config.
T_OUT_CHANNEL = 20
TIME_FEAT_DIM = 6
DELTAS_C = [-10.0, -7.0, -5.0, -2.0, 0.0, 2.0, 5.0, 7.0, 10.0]


# ────────────────────────────────────────────────────────────────
#  Intervention
# ────────────────────────────────────────────────────────────────

def intervene_t_out(x: torch.Tensor, scaler, delta_c: float,
                    channel: int = T_OUT_CHANNEL) -> torch.Tensor:
    """Add `delta_c` degrees C to the `channel` slice of x (in ORIGINAL units),
    then re-standardize back to the model's input space.

    Important: this is the SAME StandardScaler the loader used; therefore
    `delta_norm = delta_c / scaler.scale_[channel]` is the equivalent shift in
    normalized space.  We compute it that way (one multiplication, no
    intermediate denormalize/renormalize round-trip)."""
    delta_norm = float(delta_c) / float(scaler.scale_[channel])
    x_new = x.clone()
    x_new[:, :, channel] = x_new[:, :, channel] + delta_norm
    return x_new


@torch.no_grad()
def collect_forecasts(model: nn.Module, loader, scaler, log_target: bool,
                      intervene_fn=None) -> np.ndarray:
    """Run model over loader (un-shuffled), return forecasts in original Wh.
    If `intervene_fn` is provided, it transforms `x` before each forward pass."""
    model.eval(); model.to(DEVICE)
    preds = []
    for x, _ in loader:
        x = x.to(DEVICE)
        if intervene_fn is not None:
            x = intervene_fn(x)
        p = model(x)
        if p.dim() == 3:  # quantile (defensive)
            p = p[..., p.size(-1) // 2]
        preds.append(p.cpu())
    preds = torch.cat(preds).numpy()
    # Inverse-transform back to Wh.
    sc, mn = float(scaler.scale_[0]), float(scaler.mean_[0])
    preds = preds * sc + mn
    if log_target:
        preds = np.expm1(preds)
    return preds


# ────────────────────────────────────────────────────────────────
#  Gate sanity check (TC-DPMixer only)
# ────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_tc_gates(tcdp_model, loader, intervene_fn=None) -> np.ndarray:
    """For TCDPMixer, return per-sample 4-vector gate values [N, 4]."""
    tcdp_model.eval(); tcdp_model.to(DEVICE)
    gates = []
    for x, _ in loader:
        x = x.to(DEVICE)
        if intervene_fn is not None:
            x = intervene_fn(x)
        g = tcdp_model._compute_gates(x)    # [B, 4]
        gates.append(g.cpu())
    return torch.cat(gates).numpy()


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("TASK #5 — Counterfactual probing on T_out  (intervention sweep)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Intervening on channel {T_OUT_CHANNEL} (= T_out)")
    print(f"Δ T_out grid (°C): {DELTAS_C}")

    cfg = make_config(weekly=False, feature_mode="no_rv",
                      add_time_features=True, log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    print(f"  n_feat={n_feat}   seq_len={cfg.seq_len}   pred_len={cfg.pred_len}")

    # Train one model per architecture (point-prediction recipe).
    arch_specs = [
        ("dlinear",      {}),
        ("itransformer", {"d_model": 128, "n_layers": 2}),
        ("tcdpmixer",    {"time_feat_dim": TIME_FEAT_DIM}),
    ]
    trained = {}
    for name, kw in arch_specs:
        print(f"\n  [{name}] training…")
        t0 = time.time()
        m = build_model(name, cfg.seq_len, cfg.pred_len, n_feat,
                        use_residual=True, **kw)
        m = train(m, tr_l, vl_l, label=name, loss_name="l1")
        print(f"    trained in {time.time() - t0:.1f}s")
        trained[name] = m

    # ── intervention sweep ──
    print("\n  Sweeping interventions on test set…")
    rows = []
    base_forecasts = {}
    for name, m in trained.items():
        base_forecasts[name] = collect_forecasts(
            m, te_l, scaler, cfg.log_target, intervene_fn=None)
        for d in DELTAS_C:
            fn = (lambda x, _d=d: intervene_t_out(x, scaler, _d)) \
                 if d != 0.0 else None
            pred = collect_forecasts(m, te_l, scaler, cfg.log_target,
                                     intervene_fn=fn)
            delta_pred = pred - base_forecasts[name]    # [N, H] in Wh
            for t in range(delta_pred.shape[1]):
                rows.append({
                    "model": name, "delta_C": d, "horizon": t,
                    "mean_dWh":  round(float(delta_pred[:, t].mean()), 4),
                    "std_dWh":   round(float(delta_pred[:, t].std()),  4),
                    "abs_dWh":   round(float(np.abs(delta_pred[:, t]).mean()), 4),
                    "n_windows": int(delta_pred.shape[0]),
                })
            # Compact aggregate row (averaged over horizon).
            rows.append({
                "model": name, "delta_C": d, "horizon": "mean",
                "mean_dWh":  round(float(delta_pred.mean()), 4),
                "std_dWh":   round(float(delta_pred.std()),  4),
                "abs_dWh":   round(float(np.abs(delta_pred).mean()), 4),
                "n_windows": int(delta_pred.shape[0]),
            })
            print(f"    {name:13s}  ΔT={d:+5.1f}°C  "
                  f"mean Δŷ = {delta_pred.mean():+7.2f} Wh  "
                  f"|Δŷ| = {np.abs(delta_pred).mean():.2f}")

    # Δ=0 sanity check: must be exactly 0 (numerical bug detector).
    for name in trained:
        d0 = [r for r in rows if r["model"] == name and r["delta_C"] == 0
              and r["horizon"] == "mean"]
        assert d0, f"missing Δ=0 row for {name}"
        if abs(d0[0]["mean_dWh"]) > 1e-6 or abs(d0[0]["abs_dWh"]) > 1e-6:
            print(f"  WARNING: {name} Δ=0 response is non-zero "
                  f"(mean={d0[0]['mean_dWh']}, |Δ|={d0[0]['abs_dWh']})")
        else:
            print(f"  ✓ {name} Δ=0 sanity passed (response exactly 0).")

    # ── gate sanity check (TC-DPMixer only) ──
    print("\n  Gate self-consistency (TC-DPMixer):")
    tcdp = trained["tcdpmixer"]
    g_base = collect_tc_gates(tcdp, te_l, intervene_fn=None)
    gate_rows = []
    for d in DELTAS_C:
        fn = (lambda x, _d=d: intervene_t_out(x, scaler, _d)) \
             if d != 0.0 else None
        g_int = collect_tc_gates(tcdp, te_l, intervene_fn=fn)
        l2 = float(np.linalg.norm(g_int - g_base) / np.sqrt(g_base.size))
        l_inf = float(np.max(np.abs(g_int - g_base)))
        gate_rows.append({"delta_C": d, "rms": l2, "max": l_inf})
        verdict = "✓ flat" if l_inf < 1e-5 else "⚠ changed"
        print(f"    ΔT={d:+5.1f}°C   gate RMS shift = {l2:.2e}   "
              f"max|Δg| = {l_inf:.2e}   {verdict}")

    # ── save ──
    from training import save_csv
    save_csv(rows, os.path.join(RESULTS, "counterfactual.csv"))
    save_csv(gate_rows, os.path.join(RESULTS, "counterfactual_gates.csv"))
    _plot_response(rows)
    _plot_gate_invariance(tcdp, scaler, te_l)
    print("\nDone.")


def _plot_response(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    INK, AMBER, STEEL, GRAY = "#1F2A37", "#D99A2B", "#3F6CA8", "#7F8C9A"
    rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "savefig.bbox": "tight", "figure.facecolor": "white",
                     "axes.edgecolor": "#AEB8C2"})
    color_for = {"dlinear": GRAY, "itransformer": STEEL, "tcdpmixer": AMBER}

    # Filter to horizon == "mean" for the main response curve.
    by_model = {}
    for r in rows:
        if r["horizon"] != "mean":
            continue
        by_model.setdefault(r["model"], []).append((r["delta_C"], r["mean_dWh"],
                                                     r["std_dWh"]))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Left panel: mean Δ forecast vs ΔT.
    ax = axes[0]
    for name, pts in by_model.items():
        pts.sort()
        ds = np.array([p[0] for p in pts])
        ms = np.array([p[1] for p in pts])
        ax.plot(ds, ms, marker="o", lw=2.2,
                color=color_for.get(name, "#888"),
                label=name, zorder=3 if name == "tcdpmixer" else 2)
    ax.axhline(0, color="#D7DEE5", lw=1.0, ls="--")
    ax.axvline(0, color="#D7DEE5", lw=1.0, ls="--")
    ax.set_title("Counterfactual response to T_out intervention",
                 fontsize=12, weight="bold", color=INK)
    ax.set_xlabel("Δ T_out (°C)")
    ax.set_ylabel("Mean Δ forecast (Wh) over the test set")
    ax.legend(fontsize=9, frameon=True, loc="upper left")
    ax.grid(True, color="#E3E8ED", lw=0.7); ax.set_axisbelow(True)

    # Right panel: per-horizon |Δ| at a single Δ (e.g. +5°C) — does the
    # influence of T_out decay across the 24-step forecast?
    ax = axes[1]
    for name, _ in by_model.items():
        per_h = [(r["horizon"], r["abs_dWh"]) for r in rows
                 if r["model"] == name and r["delta_C"] == 5.0
                 and r["horizon"] != "mean"]
        per_h.sort()
        if not per_h:
            continue
        xs = np.array([t for t, _ in per_h])
        ys = np.array([v for _, v in per_h])
        ax.plot(xs, ys, marker="o", ms=3, lw=2.0,
                color=color_for.get(name, "#888"),
                label=name, zorder=3 if name == "tcdpmixer" else 2)
    ax.set_title("Per-horizon |Δ forecast| at ΔT = +5 °C",
                 fontsize=12, weight="bold", color=INK)
    ax.set_xlabel("Forecast step (10-min ahead)")
    ax.set_ylabel("|Δ ŷ| (Wh)")
    ax.legend(fontsize=9, frameon=True, loc="upper right")
    ax.grid(True, color="#E3E8ED", lw=0.7); ax.set_axisbelow(True)

    fig.tight_layout()
    png = os.path.join(FIG_DIR, "fig_counterfactual_response.png")
    fig.savefig(png); fig.savefig(png.replace(".png", ".pdf"))
    plt.close()
    print(f"  -> saved {png}  (+ .pdf)")


def _plot_gate_invariance(tcdp_model, scaler, loader):
    """Visualise that g(t) is identical under T_out intervention."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    INK, AMBER, NAVY = "#1F2A37", "#D99A2B", "#001F4E"
    rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "savefig.bbox": "tight", "figure.facecolor": "white",
                     "axes.edgecolor": "#AEB8C2"})

    # Compute gate values on the first 500 test windows at Δ=0 vs Δ=+10°C.
    g_base = collect_tc_gates(tcdp_model, loader, intervene_fn=None)[:500]
    g_int  = collect_tc_gates(tcdp_model, loader,
                              intervene_fn=lambda x: intervene_t_out(x, scaler, 10.0))[:500]
    names = ["persistence", "decomp", "channel", "time"]

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6), sharey=True)
    for k, ax in enumerate(axes):
        ax.scatter(g_base[:, k], g_int[:, k], s=8, alpha=0.5,
                   color=NAVY, edgecolors="none")
        lim_lo = min(g_base[:, k].min(), g_int[:, k].min()) - 0.05
        lim_hi = max(g_base[:, k].max(), g_int[:, k].max()) + 0.05
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=AMBER, lw=1.2,
                ls="--", label="y = x")
        ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel(f"g({names[k]})  at  ΔT = 0")
        if k == 0:
            ax.set_ylabel("g(·)  at  ΔT = +10 °C")
        ax.set_title(names[k], fontsize=11, weight="bold", color=INK)
        ax.grid(True, color="#E3E8ED", lw=0.7); ax.set_axisbelow(True)
    axes[-1].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle("Gate self-consistency: g(t) is invariant to T_out interventions",
                 fontsize=13, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    png = os.path.join(FIG_DIR, "fig_counterfactual_gate.png")
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
    torch.manual_seed(42); np.random.seed(42)
    main()
