"""Exp 9 — active learning: uncertainty sampling vs random sampling.

Compares 2 conditions on BODMAS (LightGBM only):
  (a) random_1pct  — paper's existing Exp 4 baseline (1% random / month)
  (b) uncertainty_1pct  — pick the 1% with predicted P(malware) closest to 0.5

Both retrain every month with their selected 1% sample, so the only difference is
*which* 1%. Addresses Reviewer 1 concern B7 (Pendlebury 2024 active learning).

Note: paper's existing Random Forest / MLP results stand; this experiment isolates
the effect of selection strategy on the most-used classifier (LightGBM).

Usage:
  python run_seed.py --seed 42
  python run_seed.py --seed 42 --smoke
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.classifiers import make_classifier, predict_proba_malware  # noqa: E402
from common.data_loader import (  # noqa: E402
  DatasetSplit,
  bodmas_monthly_iter,
  load_corpus,
)
from common.metrics import basic_metrics  # noqa: E402
from common.sampling import random_one_percent, uncertainty_top_k  # noqa: E402

CLASSIFIER = "lightgbm"
CONDITIONS = ("random_1pct", "uncertainty_1pct")


@dataclass
class MonthRecord:
  seed: int
  condition: str
  month: str
  f1: float
  mcc: float
  fpr: float
  fnr: float
  auc: float
  n_test: int
  cumulative_labels_used: int


def _train(seed: int, train: DatasetSplit, n_jobs: int):
  clf = make_classifier(CLASSIFIER, seed=seed, n_jobs=n_jobs)
  clf.fit(train.X, train.y)
  return clf


def run_seed(seed: int, n_jobs: int, smoke: bool, out_dir: Path) -> None:
  t_start = time.time()
  corpus = load_corpus()
  months = list(bodmas_monthly_iter(corpus))
  train_month_name, train_split = months[0]
  eval_pairs = months[1:]
  if smoke:
  eval_pairs = eval_pairs[:3]

  print(f"[seed={seed}] train month {train_month_name} (n={len(train_split):,}) "
  f"-> {len(eval_pairs)} eval months")

  records: list[MonthRecord] = []
  for condition in CONDITIONS:
  accumulated_train = train_split
  clf = _train(seed, accumulated_train, n_jobs)
  labels_used = len(accumulated_train)

  for month_idx, (month_name, month_split) in enumerate(eval_pairs, start=1):
  y_pred = clf.predict(month_split.X).astype(np.int8)
  y_proba = predict_proba_malware(clf, month_split.X)
  metrics = basic_metrics(month_split.y, y_pred, y_proba)
  records.append(MonthRecord(
  seed=seed,
  condition=condition,
  month=month_name,
  f1=metrics["f1"],
  mcc=metrics["mcc"],
  fpr=metrics["fpr"],
  fnr=metrics["fnr"],
  auc=metrics.get("auc", float("nan")),
  n_test=len(month_split),
  cumulative_labels_used=labels_used,
  ))

  # Build the 1% sample for next-month retrain.
  if condition == "random_1pct":
  fresh = random_one_percent(month_split, seed, month_idx)
  else:  # uncertainty_1pct
  fresh = uncertainty_top_k(month_split, y_proba, seed, month_idx)

  accumulated_df = pd.concat(
  [accumulated_train.df, fresh.df], ignore_index=True
  )
  accumulated_train = DatasetSplit(name="accumulated", df=accumulated_df)
  clf = _train(seed, accumulated_train, n_jobs)
  labels_used += len(fresh)

  print(f"  [{condition}] labels_used={labels_used}")

  out_dir.mkdir(parents=True, exist_ok=True)
  suffix = "_smoke" if smoke else ""
  out_path = out_dir / f"seed_{seed:04d}{suffix}.csv"
  pd.DataFrame([asdict(r) for r in records]).to_csv(out_path, index=False)

  meta = {
  "seed": seed,
  "smoke": smoke,
  "n_eval_months": len(eval_pairs),
  "wall_time_sec": round(time.time() - t_start, 1),
  }
  (out_dir / f"seed_{seed:04d}{suffix}.json").write_text(json.dumps(meta, indent=2))
  print(f"[seed={seed}] DONE in {meta['wall_time_sec']}s -> {out_path}")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--seed", type=int, required=True)
  p.add_argument("--n-jobs", type=int, default=12)
  p.add_argument("--smoke", action="store_true")
  p.add_argument(
  "--out-dir",
  type=Path,
  default=Path("results") / "exp9_active_learning",
  )
  return p.parse_args()


if __name__ == "__main__":
  args = parse_args()
  run_seed(args.seed, args.n_jobs, args.smoke, args.out_dir)
