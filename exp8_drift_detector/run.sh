#!/usr/bin/env bash
# Exp 8 — drift-triggered retraining
# Launches 30 seed×classifier instances on a 60-core server with 5-way GNU parallel
# (matches cluster's 5-worker mental model: 5 RUNNING + others queued).
#
# Wall clock estimate (verified after smoke): ~hours (revise once smoke completes).
#
# Usage:
#  bash run.sh  # full run (10 seeds × 3 clf)
#  bash run.sh --smoke  # smoke check (1 seed × 1 clf, 3 months)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source <server-venv>/bin/activate

SEEDS=(42 123 456 789 1011 2026 3141 4242 5555 6789)
CLASSIFIERS=(lightgbm rf mlp)

if [[ "${1-}" == "--smoke" ]]; then
  echo "[smoke] 1 seed × 1 classifier, 3 months only"
  python exp8_drift_detector/run_seed.py --seed 42 --classifier lightgbm --smoke
  exit 0
fi

mkdir -p results/exp8_drift_detector
JOBLIST="$(mktemp)"
trap 'rm -f "$JOBLIST"' EXIT

for s in "${SEEDS[@]}"; do
  for c in "${CLASSIFIERS[@]}"; do
  echo "python exp8_drift_detector/run_seed.py --seed $s --classifier $c --n-jobs 12 \
  >results/exp8_drift_detector/log_${s}_${c}.txt 2>&1" >> "$JOBLIST"
  done
done

N_JOBS="$(wc -l <"$JOBLIST")"
echo "[exp8] launching $N_JOBS jobs, 5 parallel × n_jobs=12 each"

# GNU parallel: 5 concurrent. If parallel not available, fall back to xargs.
if command -v parallel >/dev/null 2>&1; then
  parallel -j 5 --bar < "$JOBLIST"
else
  xargs -L 1 -P 5 -I CMD bash -c CMD < "$JOBLIST"
fi

echo "[exp8] all done; aggregating"
python exp8_drift_detector/aggregate.py
