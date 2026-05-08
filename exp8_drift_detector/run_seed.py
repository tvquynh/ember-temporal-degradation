"""Exp 8 — drift-triggered retraining.

Compares 4 conditions on BODMAS 2019-08 (initial train) + 13 monthly streams
(2019-09 to 2020-09):

  (a) static  — never retrain
  (b) fixed_1pct  — retrain every month with 1% of previous month
  (c) adwin  — retrain only when ADWIN flags drift (1% sample)
  (d) ddm  — retrain only when DDM flags drift (1% sample)

Per-condition output: per-month F1/MCC/FPR/FNR + total_retrains + total_labels_used.

Addresses Reviewer 1 concerns B4 + B6 (modern pipeline must include drift detector;
trigger-based retraining vs fixed schedule).

Usage:
  python run_seed.py --seed 42 --classifier lightgbm
  python run_seed.py --seed 42 --classifier rf --smoke  # smoke-test on 3 months only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from river.drift import ADWIN
from river.drift.binary import DDM

# Allow `import common.*` when run from repo root or paper folder.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.classifiers import make_classifier, predict_proba_malware  # noqa: E402
from common.data_loader import (  # noqa: E402
  DatasetSplit,
  bodmas_monthly_iter,
  load_corpus,
)
from common.metrics import basic_metrics  # noqa: E402
from common.sampling import random_one_percent  # noqa: E402

CONDITIONS = ("static", "fixed_1pct", "adwin", "ddm")


@dataclass
class MonthRecord:
  seed: int
  classifier: str
  condition: str
  month: str
  f1: float
  mcc: float
  fpr: float
  fnr: float
  auc: float
  n_test: int
  retrained_this_month: bool
  cumulative_retrains: int
  cumulative_labels_used: int


def _train(classifier: str, seed: int, train: DatasetSplit, n_jobs: int):
  clf = make_classifier(classifier, seed=seed, n_jobs=n_jobs)
  clf.fit(train.X, train.y)
  return clf


def _eval(clf, test: DatasetSplit) -> tuple[dict[str, float], np.ndarray]:
  """Predict + return basic metrics + correctness stream for drift detectors."""
  y_pred = clf.predict(test.X).astype(np.int8)
  y_proba = predict_proba_malware(clf, test.X)
  metrics = basic_metrics(test.y, y_pred, y_proba)
  correct = (y_pred == test.y).astype(np.int8)
  return metrics, correct


def run_seed(seed: int, classifier: str, n_jobs: int, smoke: bool, out_dir: Path) -> None:
  t_start = time.time()
  corpus = load_corpus()
  months = list(bodmas_monthly_iter(corpus))
  if not months:
  raise RuntimeError("No BODMAS monthly slices found")

  # Identify the training month (the chronologically first month with samples).
  train_month_name, train_split = months[0]
  eval_pairs = months[1:]
  if smoke:
  eval_pairs = eval_pairs[:3]
  print(f"[seed={seed} clf={classifier}] train month {train_month_name} "
  f"(n={len(train_split):,}) -> {len(eval_pairs)} eval months")

  records: list[MonthRecord] = []

  for condition in CONDITIONS:
  # Initialize per-condition state.
  # All conditions start from the same initial train set (month 0).
  accumulated_train = train_split
  clf = _train(classifier, seed, accumulated_train, n_jobs)
  labels_used = len(accumulated_train)
  retrains = 0
  # Drift detectors are condition-specific.
  adwin = ADWIN() if condition == "adwin" else None
  ddm = DDM() if condition == "ddm" else None

  for month_idx, (month_name, month_split) in enumerate(eval_pairs, start=1):
  metrics, correct_stream = _eval(clf, month_split)
  # Decide whether to retrain at month-end.
  retrain_now = False
  if condition == "fixed_1pct":
  retrain_now = True
  elif condition == "adwin":
  # ADWIN expects values to monitor; we feed correctness (1=ok, 0=err).
  triggered = False
  for v in correct_stream.tolist():
  adwin.update(int(v))
  if adwin.drift_detected:
  triggered = True
  # Reset by rebuilding to clear internal state for next window.
  adwin = ADWIN()
  retrain_now = triggered
  elif condition == "ddm":
  # river DDM expects 1=incorrect, 0=correct; we send error indicator.
  triggered = False
  for v in (1 - correct_stream).tolist():
  ddm.update(int(v))
  if ddm.drift_detected:
  triggered = True
  ddm = DDM()
  retrain_now = triggered
  # else: static — never retrain.

  if retrain_now:
  fresh_sample = random_one_percent(month_split, seed, month_idx)
  accumulated_df = pd.concat(
  [accumulated_train.df, fresh_sample.df], ignore_index=True
  )
  accumulated_train = DatasetSplit(name="accumulated", df=accumulated_df)
  clf = _train(classifier, seed, accumulated_train, n_jobs)
  labels_used += len(fresh_sample)
  retrains += 1

  records.append(MonthRecord(
  seed=seed,
  classifier=classifier,
  condition=condition,
  month=month_name,
  f1=metrics["f1"],
  mcc=metrics["mcc"],
  fpr=metrics["fpr"],
  fnr=metrics["fnr"],
  auc=metrics.get("auc", float("nan")),
  n_test=len(month_split),
  retrained_this_month=retrain_now,
  cumulative_retrains=retrains,
  cumulative_labels_used=labels_used,
  ))

  print(f"  [{condition}] retrains={retrains} labels_used={labels_used}")

  out_dir.mkdir(parents=True, exist_ok=True)
  suffix = "_smoke" if smoke else ""
  out_path = out_dir / f"seed_{seed:04d}_{classifier}{suffix}.csv"
  pd.DataFrame([asdict(r) for r in records]).to_csv(out_path, index=False)

  meta = {
  "seed": seed,
  "classifier": classifier,
  "smoke": smoke,
  "n_eval_months": len(eval_pairs),
  "wall_time_sec": round(time.time() - t_start, 1),
  }
  (out_dir / f"seed_{seed:04d}_{classifier}{suffix}.json").write_text(json.dumps(meta, indent=2))
  print(f"[seed={seed} clf={classifier}] DONE in {meta['wall_time_sec']}s -> {out_path}")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--seed", type=int, required=True)
  p.add_argument("--classifier", choices=("lightgbm", "rf", "mlp"), required=True)
  p.add_argument("--n-jobs", type=int, default=12)
  p.add_argument("--smoke", action="store_true")
  p.add_argument(
  "--out-dir",
  type=Path,
  default=Path("results") / "exp8_drift_detector",
  )
  return p.parse_args()


if __name__ == "__main__":
  args = parse_args()
  run_seed(args.seed, args.classifier, args.n_jobs, args.smoke, args.out_dir)
