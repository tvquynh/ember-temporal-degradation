"""Aggregate Exp 8 per-seed CSVs into manuscript-ready summary.

Outputs:
  results/exp8_drift_detector/summary_per_condition.csv
  columns: classifier, condition, mean_aut, std_aut, ci_low, ci_high,
  mean_retrains, mean_labels_used, mean_f1_overall

Run with --validate to print sanity-check report (no NaN, expected file count).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.metrics import aut_with_bootstrap  # noqa: E402
from common.seeds import SEEDS  # noqa: E402

CLASSIFIERS = ("lightgbm", "rf", "mlp")
CONDITIONS = ("static", "fixed_1pct", "adwin", "ddm")


def load_all(in_dir: Path) -> pd.DataFrame:
  frames = []
  for f in sorted(in_dir.glob("seed_*.csv")):
  if "_smoke" in f.stem:
  continue
  frames.append(pd.read_csv(f))
  if not frames:
  raise FileNotFoundError(f"No non-smoke seed CSVs in {in_dir}")
  return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
  rows = []
  for clf in CLASSIFIERS:
  for cond in CONDITIONS:
  sub = df[(df.classifier == clf) & (df.condition == cond)]
  if sub.empty:
  continue
  # Build (n_seeds, n_months) matrix of F1.
  pivot = sub.pivot_table(index="seed", columns="month", values="f1").sort_index()
  curves = pivot.to_numpy()
  aut_stats = aut_with_bootstrap(curves)
  # Retrain counts: take last cumulative per seed.
  retrains = sub.sort_values("month").groupby("seed").tail(1)["cumulative_retrains"]
  labels_used = sub.sort_values("month").groupby("seed").tail(1)["cumulative_labels_used"]
  rows.append({
  "classifier": clf,
  "condition": cond,
  "n_seeds": len(curves),
  "n_months": curves.shape[1],
  "mean_aut": aut_stats["mean"],
  "std_aut": aut_stats["std"],
  "ci_low": aut_stats["ci_low"],
  "ci_high": aut_stats["ci_high"],
  "mean_retrains": float(retrains.mean()),
  "mean_labels_used": float(labels_used.mean()),
  "mean_f1_overall": float(sub["f1"].mean()),
  })
  return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> None:
  expected = len(SEEDS) * len(CLASSIFIERS)
  actual = df.groupby(["classifier", "condition"])["seed"].nunique().min()
  if actual < len(SEEDS):
  print(f"[validate] WARN: some (clf, condition) groups have only {actual} seeds (expected {len(SEEDS)})")
  nan_count = int(df[["f1", "mcc", "fpr", "fnr"]].isna().sum().sum())
  if nan_count:
  print(f"[validate] WARN: {nan_count} NaN values across f1/mcc/fpr/fnr")
  bad = df[(df.f1 < 0.0) | (df.f1 > 1.0)]
  if not bad.empty:
  print(f"[validate] WARN: {len(bad)} rows with F1 out of [0,1]")
  print(f"[validate] {len(df)} per-month rows, "
  f"{df['seed'].nunique()} seeds, "
  f"{df['classifier'].nunique()} classifiers, "
  f"{df['condition'].nunique()} conditions")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--in-dir", type=Path, default=Path("results") / "exp8_drift_detector")
  ap.add_argument("--validate", action="store_true")
  args = ap.parse_args()

  df = load_all(args.in_dir)
  if args.validate:
  validate(df)
  summary = aggregate(df)
  out_path = args.in_dir / "summary_per_condition.csv"
  summary.to_csv(out_path, index=False)
  print(f"[exp8] wrote {out_path}")
  # Also dump a JSON for easy paper integration.
  summary.to_json(args.in_dir / "summary_per_condition.json", orient="records", indent=2)
  print(summary.to_string(index=False))


if __name__ == "__main__":
  main()
