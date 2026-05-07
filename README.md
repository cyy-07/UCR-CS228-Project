# CS228 — Energy Prediction Experiments

## Structure
```
CS228/
├── data/energydata_complete.csv
├── src/
│   ├── data_utils.py       # data loading, windowing, corruption
│   ├── models.py           # Persistence, MLP, LSTM, DLinear, PatchTST
│   └── run_experiments.py  # training + all 4 experiments
├── results/                # auto-created, CSVs + plot saved here
└── requirements.txt
```

## Quick start (server)
```bash
cd CS228
pip install -r requirements.txt

python src/data_utils.py          # ~3s  smoke test
python src/models.py              # ~2s  shape check

python src/run_experiments.py --exp baseline    # ~20-40 min CPU / ~5 min GPU
python src/run_experiments.py --exp scarcity    # ~60-90 min CPU
python src/run_experiments.py --exp corruption  # ~30 min CPU
python src/run_experiments.py --exp feature     # ~40 min CPU

python src/run_experiments.py     # all experiments back-to-back
```

## Estimated runtimes (50 epochs, early-stopping)
| Setting       | CPU (8-core) | GPU (T4/V100) |
|---------------|-------------|----------------|
| baseline      | ~20-40 min  | ~3-5 min       |
| scarcity      | ~60-90 min  | ~10-15 min     |
| corruption    | ~20-40 min  | ~3-5 min       |
| feature       | ~30-50 min  | ~5-8 min       |
| **Total**     | **~3-4 h**  | **~25-35 min** |

## Key config (top of run_experiments.py)
```python
EPOCHS   = 50      # reduce to 10 for quick tests
LR       = 1e-3
PATIENCE = 10
```
