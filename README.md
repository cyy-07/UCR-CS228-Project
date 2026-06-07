# CS228 — Energy Forecasting on the UCI Appliances Dataset

A comparative study of deep learning forecasting models on the UCI Appliances
Energy Prediction dataset (Candanedo et al., 2017), with literature
reproduction, feature engineering, loss-function ablation, architectural
modifications, modern Transformer variants, self-supervised pretraining, and a
forecast-horizon sweep.

---

## Repository Structure

```
CS228/
├── data/energydata_complete.csv
├── src/
│   ├── data_utils.py            # loading, windowing, time features, log-target
│   ├── models.py                # 6 models + Persistence anchor (residual)
│   ├── training.py              # shared train/evaluate/persistence_reference
│   ├── run_experiments.py       # ORIGINAL 4 experiments (milestone reproducible)
│   ├── exp_reproduction.py      # Workstream 0  literature reproduction
│   ├── exp_features.py          # Workstream A  feature engineering
│   ├── exp_losses.py            # Workstream B  loss-function comparison
│   ├── exp_residual.py          # Workstream C  Persistence-anchored ablation
│   ├── exp_transformers.py      # Workstream D  PatchTST vs iTransformer
│   ├── exp_ssl.py               # Workstream E  masked-patch pretraining
│   ├── exp_horizon.py           # Workstream F  forecast-horizon sweep
│   ├── exp_significance.py      # Evidence A  multi-seed error bars
│   ├── exp_efficiency.py        # Evidence B  accuracy/cost Pareto frontier
│   ├── exp_tc_transfer.py       # Evidence C  TC-gating transferability
│   └── exp_drift.py             # Evidence D  distribution-shift robustness
├── results/                     # auto-created; CSVs + plots saved here
└── requirements.txt
```

---

## Quick start (server)

```bash
cd CS228
pip install -r requirements.txt

# smoke tests
python src/data_utils.py
python src/models.py
```

## All 7 workstreams + 4 legacy experiments (parallel-launchable)

Every experiment script is **fully independent** — you can `nohup` them in
parallel on the same machine, or queue them on a cluster.

| ID | Script | Approx. GPU time | What it tests |
|----|--------|------------------|----------------|
| L  | `run_experiments.py --exp baseline` | 5 min | Milestone-reproducible baseline |
| L  | `run_experiments.py --exp scarcity` | 15 min | Data-scarcity robustness |
| L  | `run_experiments.py --exp corruption` | 5 min | Noise / feature masking robustness |
| L  | `run_experiments.py --exp feature` | 10 min | rv1 / rv2 sanity |
| 0  | `exp_reproduction.py` | 2 min (CPU) | Reproduce Candanedo et al. (RF/GBM) |
| A  | `exp_features.py` | 25 min | Which feature subset matters? |
| B  | `exp_losses.py` | 20 min | MSE vs L1 vs Huber as training loss |
| C  | `exp_residual.py` | 15 min | Persistence-anchored architecture |
| D  | `exp_transformers.py` | 25 min | PatchTST vs iTransformer |
| E  | `exp_ssl.py` | 30 min | Masked-patch SSL pretraining |
| F  | `exp_horizon.py` | 20 min | MAE vs forecast horizon (1h–16h) |
| G  | `exp_dpmixer.py` | 50 min | **DPMixer family** — ablation, sweep, TC-gating, spike-aware, head-to-head |
| W  | `exp_weekly.py` | ~15 min | **Weekly task** — hourly aggregation, past 1 week → next 1 week, Seasonal-Naive baseline |
| A* | `exp_significance.py` | 25 min | **Evidence A** — 5-seed error bars; is the gap to iTransformer just noise? |
| B* | `exp_efficiency.py` | 12 min | **Evidence B** — accuracy vs params/latency/train-time Pareto frontier |
| C* | `exp_tc_transfer.py` | 20 min | **Evidence C** — does TC gating help a 2nd family (DLinear) on both tasks? |
| D* | `exp_drift.py` | 12 min | **Evidence D** — per-segment MAE; does accuracy stay stable over the test span? |

### Weekly long-horizon mode (`--weekly`)

Every workstream above (A–G) also accepts a `--weekly` flag that re-runs the
*exact same experiment* on the long-horizon task: aggregate to **hourly**
cadence, look back **1 week** (`L=168`) and forecast **1 week** (`H=168`).
Results are written to a separate `*_weekly.csv` so your short-horizon CSVs are
never overwritten. The horizon sweep (F) switches to a day-1…day-7 sweep.

```bash
python src/exp_features.py     --weekly
python src/exp_dpmixer.py      --weekly
# ...etc. for losses / residual / transformers / ssl / horizon

# or run the whole weekly suite at once:
bash run_weekly_all.sh             # sequential
PARALLEL=1 bash run_weekly_all.sh  # all in background via nohup
```

### Advanced evidence studies (A–D) — strengthen the claim, don't add a model

These four scripts do **not** propose a new architecture. They stress-test the
central claim from four independent angles so it survives a skeptical reader,
and each prints a one-line plain-English `TAKEAWAY:` plus an intuitive figure:

- **A `exp_significance.py`** — re-runs the head-to-head under 5 seeds and
  reports MAE mean ± std. Auto-verdict on whether the sub-1-Wh gap to
  iTransformer is within run-to-run noise (i.e. a statistical tie).
- **B `exp_efficiency.py`** — measures params, train-time and inference
  latency next to MAE, and marks which models are **Pareto-optimal** (nobody
  is both more accurate *and* cheaper). The defensible claim is "on the
  frontier", not "rank #1".
- **C `exp_tc_transfer.py`** — flips static-gate → time-conditioned-gate on a
  **second** model family (2-branch DLinear, `tcdlinear`) across **both**
  tasks. If TC helps in most (family × task) cells, it is a reusable
  *mechanism*, not a one-model trick. (Runs both tasks itself — no `--weekly`.)
- **D `exp_drift.py`** — splits the chronological test set into 5 segments and
  reports MAE per segment + a `degradation = MAE(last)/MAE(first)` ratio.
  Rewards models that stay stable as conditions drift.

```bash
bash run_advanced_all.sh             # sequential
PARALLEL=1 bash run_advanced_all.sh  # all in background via nohup
```

### Recommended nohup launch order

```bash
mkdir -p results

# Phase 1 — quick & high-value (run in parallel)
nohup python -u src/run_experiments.py --exp baseline   > results/log_baseline.txt    2>&1 &
nohup python -u src/exp_reproduction.py                 > results/log_reproduction.txt 2>&1 &
nohup python -u src/exp_residual.py                     > results/log_residual.txt    2>&1 &

# Phase 2 — medium-cost  (start after Phase 1 if single-GPU)
nohup python -u src/exp_features.py                     > results/log_features.txt    2>&1 &
nohup python -u src/exp_losses.py                       > results/log_losses.txt      2>&1 &
nohup python -u src/exp_horizon.py                      > results/log_horizon.txt     2>&1 &
nohup python -u src/exp_transformers.py                 > results/log_transformers.txt 2>&1 &

# Phase 3 — most expensive  (run last)
nohup python -u src/exp_ssl.py                          > results/log_ssl.txt         2>&1 &
nohup python -u src/exp_dpmixer.py                      > results/log_dpmixer.txt     2>&1 &
```

Useful commands:

```bash
# live-tail any log
tail -f results/log_features.txt

# check what's running
ps aux | grep "src/exp_"

# kill all running experiments (be careful)
pkill -f "python -u src/exp_"
```

---

## Output files

Each script writes a CSV to `results/`:

| Script | CSV produced |
|--------|--------------|
| `run_experiments.py --exp baseline` | `baseline.csv` |
| `run_experiments.py --exp scarcity` | `scarcity.csv` + `scarcity_mae.png` |
| `run_experiments.py --exp corruption` | `corruption.csv` |
| `run_experiments.py --exp feature` | `feature.csv` |
| `exp_reproduction.py` | `reproduction.csv` |
| `exp_features.py` | `features.csv` |
| `exp_losses.py` | `losses.csv` |
| `exp_residual.py` | `residual.csv` |
| `exp_transformers.py` | `transformers.csv` |
| `exp_ssl.py` | `ssl.csv` |
| `exp_horizon.py` | `horizon.csv` + `horizon_mae.png` |
| `exp_dpmixer.py` | `dpmixer.csv` |
| `exp_weekly.py` | `weekly.csv` + `weekly_perday.csv` |
| `exp_significance.py` | `significance.csv` + `significance_seeds.csv` + `fig_significance.png` |
| `exp_efficiency.py` | `efficiency.csv` + `fig_efficiency.png` |
| `exp_tc_transfer.py` | `tc_transfer.csv` + `fig_tc_transfer.png` |
| `exp_drift.py` | `drift.csv` + `fig_drift.png` |
| any `--weekly` run | same name with `_weekly` suffix (e.g. `dpmixer_weekly.csv`) |

---

## Key references

- Candanedo et al. (2017). *Data driven prediction models of energy use of
  appliances in a low-energy house.* **Energy and Buildings 140, 81–97.**
  (Original dataset paper. Reports GBM test R²=0.57, RMSE≈66.65 Wh, MAE≈29.6 Wh.)
- Zeng et al. (2023). *Are Transformers Effective for Time-Series Forecasting?*
  AAAI 2023. (DLinear.)
- Nie et al. (2023). *A Time Series is Worth 64 Words.* ICLR 2023. (PatchTST.)
- Liu et al. (2024). *iTransformer: Inverted Transformers are Effective for
  Time Series Forecasting.* ICLR 2024. (iTransformer.)

---

## Notes

- All experiments seed `torch.manual_seed(42)` **at the start of every
  `train()` call** so model-to-model comparisons are reproducible
  (see `training.py`).
- All models support a `use_residual=True` flag that wraps the output as
  `persistence_pred + learned_residual`, providing a strong inductive bias.
- Legacy experiments (`run_experiments.py`) default to `use_residual=False`
  to reproduce milestone-report numbers; new experiments enable it where noted.
- Inverse-transform of predictions uses the training-set scaler only (no
  leakage). When `log_target=True` (Workstream A), `evaluate()` does an
  additional `np.expm1` so MAE/RMSE are reported in Wh.
