#!/usr/bin/env bash
# Exp 9 — active learning (LightGBM only). 10 seeds × 2 conditions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source <server-venv>/bin/activate

SEEDS=(42 123 456 789 1011 2026 3141 4242 5555 6789)

if [[ "${1-}" == "--smoke" ]]; then
  echo "[smoke] 1 seed, 3 months only"
  python exp9_active_learning/run_seed.py --seed 42 --smoke
  exit 0
fi

mkdir -p results/exp9_active_learning
JOBLIST="$(mktemp)"
trap 'rm -f "$JOBLIST"' EXIT

for s in "${SEEDS[@]}"; do
  echo "python exp9_active_learning/run_seed.py --seed $s --n-jobs 12 \
  >results/exp9_active_learning/log_${s}.txt 2>&1" >> "$JOBLIST"
done

echo "[exp9] launching 10 jobs, 5 parallel × n_jobs=12 each"
if command -v parallel >/dev/null 2>&1; then
  parallel -j 5 --bar < "$JOBLIST"
else
  xargs -L 1 -P 5 -I CMD bash -c CMD < "$JOBLIST"
fi

echo "[exp9] all done; aggregating"
python exp9_active_learning/aggregate.py
