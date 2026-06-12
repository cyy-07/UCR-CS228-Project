# Server run — revision-round experiments

> Four new scripts that address reviewer concerns R3–R6.
> All write to `CS228/results/` and respect the existing `data_utils` /
> `models.py` / `training.py` infrastructure — no new dependencies.

## What each script answers

| Script | Reviewer concern | Output |
|---|---|---|
| `src/exp_paired_test.py` | **R3** "4.85 σ" is marketing — give W, p, CI, d | `paired_test_weekly.{csv,json}` |
| `src/exp_latency.py` | **R4** "57× faster" has no hardware spec | `latency_weekly.csv` + `latency_weekly_meta.json` (GPU, CUDA, torch, batch=1 + batch=32, 1000-trial median + p99) |
| `src/exp_block_cv.py` | **R5** external validity from a single split | `block_cv_weekly.csv` (4 rolling-origin folds) |
| `src/exp_minimal.py` | **R6** ablation says backbone is redundant — prove it | `minimal_weekly.csv` (full vs minimal-A vs minimal-B, 5 seeds) |

Each script also takes `--short` to run the same comparison on the 4-h
horizon, so the report can reference numbers on both tasks.

## How to run

```bash
cd CS228
bash run_revisions.sh
```

Total wall-clock estimate on a single L4: paired_test ≈ 25 min weekly +
25 min short; latency ≈ 5 min each; block_cv ≈ 1 h each
(4 folds × 3 models); minimal ≈ 25 min each. Plan for ~4 h.

## When the runs finish

Drop the new numbers into the four placeholders in `report.tex`:

1. `% [BLOCK-CV TBD]` in the Robustness paragraph → use the
   mean ± std per model from `block_cv_weekly.csv` (last 3 "AGGREGATE"
   rows).
2. `% [PAIRED TBD]` in the Robustness paragraph → use
   `paired_test_weekly.json`: copy `wilcoxon_W`, `wilcoxon_p_two_sided`,
   `boot95_lo`, `boot95_hi`, `cohens_d_paired`.
3. Abstract / Sec. 4.3 σ wording → replace with the paired-test numbers
   per `REVISION_NOTES.md` §3.
4. Add a Table 4 (`tab:minimal`) with the 3-row summary from
   `minimal_weekly.csv`, then write the 2-sentence "Does the backbone
   earn its keep?" paragraph.
5. Abstract / wherever "57× faster" appears → reference
   `latency_weekly.csv` + `latency_weekly_meta.json` and add the
   hardware caveat (`single GPU [GPU NAME], batch=1, eager-mode PyTorch
   [VERSION], 100 warm-up + 1000 timed forwards`).

## What was deleted from `report.tex`

Two fabricated claims that had no CSV support were removed from the old
§4.6:

- "TC-DPMixer overtaking iTransformer near H ≈ 96" — `horizon.csv` does
  not include TC-DPMixer or iTransformer rows.
- "TC-DPMixer 6 % vs iTransformer 14 % drift" — `drift.csv` shows
  degradation ratios of 1.617 and 1.636 respectively, which is not what
  was claimed.

The replacement story now leans on `block_cv` (real cross-cut
robustness), `paired_test` (defensible stats), and the
already-honest `permutation_importance.csv`.
