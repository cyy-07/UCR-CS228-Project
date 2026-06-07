#!/usr/bin/env bash
# run_advanced_all.sh — the four "make-it-solid" evidence studies.
#
# These do NOT propose a new model; they stress-test the claim from four
# independent angles so it survives a skeptical reader:
#   A  significance  — multi-seed error bars: is the gap to iTransformer noise?
#   B  efficiency    — accuracy/size/latency Pareto frontier
#   C  tc_transfer   — does TC gating help a 2nd family on both tasks?
#   D  drift         — does accuracy stay stable across the test span?
#
#   bash run_advanced_all.sh            # sequential (safe on a single GPU)
#   PARALLEL=1 bash run_advanced_all.sh # background nohup, all at once
#
# Both short-horizon and weekly variants are run (tc_transfer runs both tasks
# itself in one process, so it is launched once).
set -e
cd "$(dirname "$0")"
mkdir -p results

if [ "${PARALLEL:-0}" = "1" ]; then
  echo "Launching advanced evidence studies in parallel (nohup)..."
  nohup python -u src/exp_significance.py            > results/log_significance.txt        2>&1 &
  nohup python -u src/exp_significance.py  --weekly  > results/log_significance_weekly.txt 2>&1 &
  nohup python -u src/exp_efficiency.py              > results/log_efficiency.txt          2>&1 &
  nohup python -u src/exp_efficiency.py    --weekly  > results/log_efficiency_weekly.txt   2>&1 &
  nohup python -u src/exp_tc_transfer.py             > results/log_tc_transfer.txt         2>&1 &
  nohup python -u src/exp_drift.py                   > results/log_drift.txt               2>&1 &
  nohup python -u src/exp_drift.py         --weekly  > results/log_drift_weekly.txt        2>&1 &
  echo "Started. Tail any log, e.g.:  tail -f results/log_significance.txt"
  wait
else
  echo "[A/4] significance (short)";  python -u src/exp_significance.py
  echo "[A/4] significance (weekly)"; python -u src/exp_significance.py  --weekly
  echo "[B/4] efficiency (short)";    python -u src/exp_efficiency.py
  echo "[B/4] efficiency (weekly)";   python -u src/exp_efficiency.py    --weekly
  echo "[C/4] tc-transfer (both)";    python -u src/exp_tc_transfer.py
  echo "[D/4] drift (short)";         python -u src/exp_drift.py
  echo "[D/4] drift (weekly)";        python -u src/exp_drift.py         --weekly
fi
echo "Done. Collect results/{significance,efficiency,tc_transfer,drift}*.csv + fig_*.png"
