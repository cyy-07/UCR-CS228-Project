#!/usr/bin/env bash
# Bundle of revision-round experiments.
# Run from the CS228 project root.
#   bash run_revisions.sh
# Logs land in results/log_*.txt ; CSV / JSON in results/.

set -e
mkdir -p results

# ── R3: paired Wilcoxon + bootstrap CI + Cohen's d ───────────────
nohup python -u src/exp_paired_test.py        > results/log_paired_weekly.txt 2>&1
nohup python -u src/exp_paired_test.py --short > results/log_paired_short.txt  2>&1

# ── R4: latency benchmark with full hardware metadata ────────────
nohup python -u src/exp_latency.py             > results/log_latency_short.txt  2>&1
nohup python -u src/exp_latency.py --weekly    > results/log_latency_weekly.txt 2>&1

# ── R5: rolling-origin (cross-month) block CV ────────────────────
nohup python -u src/exp_block_cv.py            > results/log_blockcv_weekly.txt 2>&1
nohup python -u src/exp_block_cv.py --short    > results/log_blockcv_short.txt  2>&1

# ── R6: minimal DPMixer — is the backbone earning its keep? ──────
nohup python -u src/exp_minimal.py             > results/log_minimal_weekly.txt 2>&1
nohup python -u src/exp_minimal.py --short     > results/log_minimal_short.txt  2>&1

echo
echo "Done. New artefacts:"
ls -1 results/paired_test*.csv results/paired_test*.json \
      results/latency*.csv     results/latency*_meta.json \
      results/block_cv*.csv    results/minimal*.csv 2>/dev/null
