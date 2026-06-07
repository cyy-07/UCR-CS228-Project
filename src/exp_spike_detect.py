"""
src/exp_spike_detect.py — Task #4: per-step Spike Detection (classification).

Task formulation
----------------
Given X = [B, L, C] (the same canonical 96 × 32 input used by the regression
experiments), predict Y_spike ∈ {0, 1}^[B, H] where

    Y_spike[b, t] = 1   iff   Appliances_wh[future_step_t]  >  τ

τ is chosen from the **training-set** percentile of Appliances (p90 and p95
reported separately; both are robustness checks of the same finding).

Why this experiment matters
---------------------------
Operationally, "will the next 4 h contain a spike?" is often more actionable
than the exact Wh number — it feeds demand-response, peak-shaving, and NILM
event-detection pipelines. Re-casting the same data as binary classification
also tests whether the TC-DPMixer architecture (designed for regression)
carries useful inductive biases for a *discriminative* task.

Methods compared (all share the canonical no_rv + time + log_target setup)
-------------------------------------------------------------------------
  chance         degenerate; predicts the prior positive rate p
  last-value     pred[t] = 1{X[-1, 0] > τ}  for every horizon step t
  last-window    pred[t] = 1{X[L-H+t, 0] > τ}  (recent H steps thresholded)
  logreg-flat    nn.Linear(L*C, H) trained with BCE  (linear-on-flat baseline)
  tcspike        TCSpikeClassifier — TC-gated decomp+channel+time mixer,
                 trained with class-balanced BCE.  Our model.

Metrics (per horizon step + macro-averaged)
-------------------------------------------
  AUC-ROC, AUC-PR (more honest under class imbalance), F1 at best threshold,
  Brier score (calibration).  Per-horizon AUC gives a "skill decay vs forecast
  step" curve — a publishable insight.

Outputs
-------
  results/spike_detect.csv                            (per model × τ × horizon)
  results/figures/fig_spike_auc.{png,pdf}             AUC vs horizon
  results/figures/fig_spike_calibration.{png,pdf}     calibration curves
  results/log_spike.txt  (when invoked via nohup)

Run
---
    python src/exp_spike_detect.py
    nohup python -u src/exp_spike_detect.py > results/log_spike.txt 2>&1 &
"""

import os, sys, time, argparse
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import make_config, get_loaders
from models import build_model
from training import train, get_loss_fn, DEVICE


RESULTS = "results"
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────
#  Helpers: invert scaler / log1p and build spike labels
# ────────────────────────────────────────────────────────────────

def y_norm_to_wh(y_n: torch.Tensor, scaler, log_target: bool) -> torch.Tensor:
    """Invert StandardScaler (channel 0) and optional log1p to recover Wh."""
    sc = float(scaler.scale_[0]); mn = float(scaler.mean_[0])
    y_un = y_n * sc + mn
    return torch.expm1(y_un) if log_target else y_un


class SpikeDataset(Dataset):
    """Wraps an existing regression Dataset and converts y to binary spike."""
    def __init__(self, base: Dataset, scaler, log_target: bool, threshold_wh: float):
        self.base, self.sc, self.mn = base, float(scaler.scale_[0]), float(scaler.mean_[0])
        self.log_target, self.thr = log_target, float(threshold_wh)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y_n = self.base[i]
        y_un = y_n * self.sc + self.mn
        y_wh = torch.expm1(y_un) if self.log_target else y_un
        return x, (y_wh > self.thr).float()


def wrap_loader(orig: DataLoader, scaler, log_target: bool,
                threshold_wh: float, shuffle: bool) -> DataLoader:
    ds = SpikeDataset(orig.dataset, scaler, log_target, threshold_wh)
    return DataLoader(ds, batch_size=orig.batch_size, shuffle=shuffle)


# ────────────────────────────────────────────────────────────────
#  Models: a tiny linear baseline (logreg-flat) for fair comparison
# ────────────────────────────────────────────────────────────────

class LogregFlat(nn.Module):
    """Flatten X → Linear → H logits. Trained with BCE. Linear-baseline ceiling."""
    def __init__(self, seq_len: int, pred_len: int, n_feat: int):
        super().__init__()
        self.fc = nn.Linear(seq_len * n_feat, pred_len)

    def forward(self, x):
        return self.fc(x.flatten(1))

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


# ────────────────────────────────────────────────────────────────
#  Score producers for non-learned baselines
# ────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_labels_and_inputs(loader: DataLoader, scaler, log_target: bool,
                              threshold_wh: float):
    """Iterate the test loader once; return (X_all, y_spike_all) as tensors."""
    Xs, Ys = [], []
    for x, y_n in loader:
        Xs.append(x)
        y_wh = y_norm_to_wh(y_n, scaler, log_target)
        Ys.append((y_wh > threshold_wh).float())
    return torch.cat(Xs), torch.cat(Ys)


def score_chance(X_all: torch.Tensor, prior_p: float) -> np.ndarray:
    """Predicted probability = prior, the same for every (sample, horizon)."""
    n, H = X_all.size(0), X_all.size(1)   # H comes from y later; resize below
    raise NotImplementedError  # supplied directly by caller; see main()


def score_last_value(X_all: torch.Tensor, scaler, log_target: bool,
                     threshold_wh: float, pred_len: int) -> np.ndarray:
    """For every horizon step: pred = 1{X[-1, 0] > τ} (probability ∈ {0,1})."""
    # Channel 0 last timestep, in original Wh.
    last_n = X_all[:, -1, 0]
    last_wh = y_norm_to_wh(last_n, scaler, log_target)
    flag = (last_wh > threshold_wh).float().unsqueeze(1).expand(-1, pred_len)
    return flag.numpy()


def score_last_window(X_all: torch.Tensor, scaler, log_target: bool,
                      threshold_wh: float, pred_len: int) -> np.ndarray:
    """pred[t] = 1{X[L-H+t, 0] > τ} — extend the last H input steps verbatim."""
    L = X_all.size(1)
    seg_n = X_all[:, L - pred_len : L, 0]
    seg_wh = y_norm_to_wh(seg_n, scaler, log_target)
    return (seg_wh > threshold_wh).float().numpy()


@torch.no_grad()
def score_model(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """Run a trained classifier over the loader; return [N, H] probabilities."""
    model.eval(); model.to(DEVICE)
    probs = []
    for x, _ in loader:
        x = x.to(DEVICE)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(x)
        else:
            p = torch.sigmoid(model(x))
        probs.append(p.cpu())
    return torch.cat(probs).numpy()


# ────────────────────────────────────────────────────────────────
#  Metrics
# ────────────────────────────────────────────────────────────────

def compute_metrics_per_horizon(y_true: np.ndarray, y_score: np.ndarray):
    """Returns dict of arrays of shape [H]: auc_roc, auc_pr, f1_best, brier."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss, f1_score)
    H = y_true.shape[1]
    out = {k: np.zeros(H) for k in ("auc_roc", "auc_pr", "f1_best", "brier")}
    out["base_rate"] = y_true.mean(axis=0)
    for t in range(H):
        yt = y_true[:, t]; ys = y_score[:, t]
        # AUC needs both classes present.
        if yt.min() == yt.max():
            out["auc_roc"][t] = np.nan
            out["auc_pr"][t]  = np.nan
        else:
            out["auc_roc"][t] = roc_auc_score(yt, ys)
            out["auc_pr"][t]  = average_precision_score(yt, ys)
        # Best-threshold F1 over a small grid.
        ths = np.linspace(0.05, 0.95, 19)
        f1s = [f1_score(yt, (ys > th).astype(int), zero_division=0) for th in ths]
        out["f1_best"][t] = float(max(f1s))
        out["brier"][t]   = float(brier_score_loss(yt, np.clip(ys, 0, 1)))
    return out


def calibration_curve(y_true: np.ndarray, y_score: np.ndarray, n_bins=10):
    """Pool over all samples × horizons. Returns (bin_centers, observed_freq)."""
    y_true_f  = y_true.flatten()
    y_score_f = np.clip(y_score.flatten(), 0.0, 1.0)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_score_f, edges) - 1, 0, n_bins - 1)
    centers, observed = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 10:
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        observed.append(float(y_true_f[mask].mean()))
    return np.array(centers), np.array(observed)


# ────────────────────────────────────────────────────────────────
#  Main experiment
# ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("TASK #4 — Spike Detection (per-step binary classification)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Canonical setup (matches regression experiments so numbers are
    # directly comparable to the rest of the paper).
    cfg = make_config(weekly=False, feature_mode="no_rv",
                      add_time_features=True, log_target=True)
    tr_l, vl_l, te_l, scaler, n_feat = get_loaders(cfg)
    print(f"  n_feat={n_feat}   pred_len={cfg.pred_len}   seq_len={cfg.seq_len}")

    # ── threshold computation: p90 / p95 of training Appliances in Wh ──
    # Iterate the (un-shuffled) train loader once.
    train_targets_n = torch.cat([y for _, y in tr_l]).flatten()
    train_targets_wh = y_norm_to_wh(train_targets_n, scaler, cfg.log_target)
    tau = {"p90": float(np.percentile(train_targets_wh.numpy(), 90)),
           "p95": float(np.percentile(train_targets_wh.numpy(), 95))}
    print(f"  Thresholds (Wh): p90={tau['p90']:.1f}   p95={tau['p95']:.1f}")
    print(f"  Train marginal positive rate at p95: "
          f"{(train_targets_wh > tau['p95']).float().mean():.3f}")

    rows = []
    calib_data = {}     # (model, τ) → (bin_centers, observed)

    for tau_name, tau_wh in tau.items():
        print(f"\n  ─── Threshold τ = {tau_name} = {tau_wh:.1f} Wh ───")

        # Wrap loaders so labels are binary spike indicators.
        sp_tr = wrap_loader(tr_l, scaler, cfg.log_target, tau_wh, shuffle=True)
        sp_vl = wrap_loader(vl_l, scaler, cfg.log_target, tau_wh, shuffle=False)
        sp_te = wrap_loader(te_l, scaler, cfg.log_target, tau_wh, shuffle=False)

        # Collect ground truth + raw inputs from the un-shuffled test loader
        # for the analytic baselines.
        X_te, Y_te = collect_labels_and_inputs(te_l, scaler, cfg.log_target, tau_wh)
        N, H = Y_te.shape
        prior_p = float(Y_te.mean())
        print(f"  test positive rate = {prior_p:.3f}   (N={N}, H={H})")

        # Compute pos_weight for class-balanced BCE from the TRAIN set.
        train_y = torch.cat([y for _, y in sp_tr]).flatten()
        train_p = float(train_y.mean())
        pos_weight = (1 - train_p) / max(train_p, 1e-6)
        print(f"  train positive rate = {train_p:.3f}   pos_weight = {pos_weight:.2f}")

        # ── 1. chance ─────────────────────────────────────────────────
        scores = {}
        scores["chance"] = np.full_like(Y_te.numpy(), prior_p)

        # ── 2. last-value ─────────────────────────────────────────────
        scores["last-value"] = score_last_value(
            X_te, scaler, cfg.log_target, tau_wh, H)

        # ── 3. last-window ────────────────────────────────────────────
        scores["last-window"] = score_last_window(
            X_te, scaler, cfg.log_target, tau_wh, H)

        # ── 4. logreg-flat (linear-on-flat, BCE) ──────────────────────
        print("  [logreg-flat] training …")
        t0 = time.time()
        lr_model = LogregFlat(cfg.seq_len, cfg.pred_len, n_feat)
        lr_model = train(lr_model, sp_tr, sp_vl, label=f"logreg-{tau_name}",
                         loss_fn=get_loss_fn("bce_balanced",
                                             pos_weight=pos_weight))
        scores["logreg-flat"] = score_model(lr_model, sp_te)
        print(f"    trained in {time.time() - t0:.1f}s")

        # ── 5. tcspike (our model) ────────────────────────────────────
        print("  [tcspike] training …")
        t0 = time.time()
        clf = build_model("tcspike", cfg.seq_len, cfg.pred_len, n_feat,
                          time_feat_dim=6)
        clf = train(clf, sp_tr, sp_vl, label=f"tcspike-{tau_name}",
                    loss_fn=get_loss_fn("bce_balanced",
                                        pos_weight=pos_weight))
        scores["tcspike"] = score_model(clf, sp_te)
        n_params = sum(p.numel() for p in clf.parameters())
        print(f"    trained in {time.time() - t0:.1f}s ({n_params:,} params)")

        # ── compute & store metrics ───────────────────────────────────
        Y_np = Y_te.numpy()
        for name, scr in scores.items():
            m = compute_metrics_per_horizon(Y_np, scr)
            for t in range(H):
                rows.append({"model": name, "threshold": tau_name,
                             "horizon": t,
                             "AUC_ROC": round(float(m["auc_roc"][t]), 4),
                             "AUC_PR":  round(float(m["auc_pr"][t]),  4),
                             "F1_best": round(float(m["f1_best"][t]), 4),
                             "Brier":   round(float(m["brier"][t]),   4),
                             "base_rate": round(float(m["base_rate"][t]), 4)})
            # Macro row.
            rows.append({"model": name, "threshold": tau_name,
                         "horizon": "macro",
                         "AUC_ROC": round(float(np.nanmean(m["auc_roc"])), 4),
                         "AUC_PR":  round(float(np.nanmean(m["auc_pr"])),  4),
                         "F1_best": round(float(np.nanmean(m["f1_best"])), 4),
                         "Brier":   round(float(np.nanmean(m["brier"])),   4),
                         "base_rate": round(float(np.nanmean(m["base_rate"])), 4)})
            # Calibration over all (sample × horizon) pairs.
            bc, obs = calibration_curve(Y_np, scr, n_bins=10)
            calib_data[(name, tau_name)] = (bc, obs)
            print(f"  {name:13s} macro AUC={np.nanmean(m['auc_roc']):.3f}  "
                  f"PR={np.nanmean(m['auc_pr']):.3f}  "
                  f"F1={np.nanmean(m['f1_best']):.3f}  "
                  f"Brier={np.nanmean(m['brier']):.3f}")

    # ── save ──
    from training import save_csv
    save_csv(rows, os.path.join(RESULTS, "spike_detect.csv"))
    _plot_auc(rows)
    _plot_calibration(calib_data)
    print("\nDone.")


def _plot_auc(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    INK, AMBER, STEEL = "#1F2A37", "#D99A2B", "#3F6CA8"
    rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "savefig.bbox": "tight", "figure.facecolor": "white",
                     "axes.edgecolor": "#AEB8C2"})
    thresholds = sorted({r["threshold"] for r in rows})
    fig, axes = plt.subplots(1, len(thresholds), figsize=(11.5, 4.4), sharey=True)
    if len(thresholds) == 1:
        axes = [axes]

    model_colors = {"chance": "#BFC6CE", "last-value": "#94A3B0",
                    "last-window": "#6B7C8F", "logreg-flat": STEEL,
                    "tcspike": AMBER}

    for ax, tau in zip(axes, thresholds):
        for model in ["chance", "last-value", "last-window",
                      "logreg-flat", "tcspike"]:
            xs, ys = [], []
            for r in rows:
                if r["model"] != model or r["threshold"] != tau:
                    continue
                if r["horizon"] == "macro":
                    continue
                xs.append(r["horizon"]); ys.append(r["AUC_ROC"])
            order = np.argsort(xs)
            xs = np.array(xs)[order]; ys = np.array(ys)[order]
            ax.plot(xs, ys, marker="o", ms=3,
                    lw=2.4 if model == "tcspike" else 1.5,
                    color=model_colors[model],
                    label=model, zorder=3 if model == "tcspike" else 2)
        ax.axhline(0.5, color="#D7DEE5", lw=1.0, ls="--")
        ax.set_title(f"τ = {tau}  (training percentile)",
                     fontsize=12, weight="bold", color=INK)
        ax.set_xlabel("Forecast step (10-min ticks ahead)")
        ax.grid(True, color="#E3E8ED", lw=0.7); ax.set_axisbelow(True)
    axes[0].set_ylabel("AUC-ROC  (higher better)")
    axes[-1].legend(loc="lower left", fontsize=8, frameon=True)
    fig.suptitle("Per-horizon spike-detection AUC",
                 fontsize=13, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = os.path.join(FIG_DIR, "fig_spike_auc.png")
    fig.savefig(png); fig.savefig(png.replace(".png", ".pdf"))
    plt.close()
    print(f"  -> saved {png}  (+ .pdf)")


def _plot_calibration(calib_data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception as e:
        print(f"  (skip plot: {e})")
        return
    INK, AMBER, STEEL = "#1F2A37", "#D99A2B", "#3F6CA8"
    rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "savefig.bbox": "tight", "figure.facecolor": "white",
                     "axes.edgecolor": "#AEB8C2"})
    thresholds = sorted({k[1] for k in calib_data})
    fig, axes = plt.subplots(1, len(thresholds), figsize=(11.0, 4.6))
    if len(thresholds) == 1:
        axes = [axes]
    model_colors = {"chance": "#BFC6CE", "last-value": "#94A3B0",
                    "last-window": "#6B7C8F", "logreg-flat": STEEL,
                    "tcspike": AMBER}
    for ax, tau in zip(axes, thresholds):
        ax.plot([0, 1], [0, 1], color="#D7DEE5", lw=1.0, ls="--",
                label="perfectly calibrated")
        for model in ["chance", "last-value", "last-window",
                      "logreg-flat", "tcspike"]:
            if (model, tau) not in calib_data:
                continue
            bc, obs = calib_data[(model, tau)]
            ax.plot(bc, obs, marker="o", ms=4,
                    lw=2.4 if model == "tcspike" else 1.5,
                    color=model_colors[model],
                    label=model, zorder=3 if model == "tcspike" else 2)
        ax.set_title(f"τ = {tau}", fontsize=12, weight="bold", color=INK)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, color="#E3E8ED", lw=0.7); ax.set_axisbelow(True)
    axes[-1].legend(loc="upper left", fontsize=8, frameon=True)
    fig.suptitle("Reliability diagrams (perfect = diagonal)",
                 fontsize=13, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = os.path.join(FIG_DIR, "fig_spike_calibration.png")
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
