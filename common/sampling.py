"""Deterministic 1%-sampling helpers shared by Exp 8 and Exp 9.

Kept independent of lightgbm/river so unit tests can run locally on Windows
(<2 minutes per project rule) without the full server-side stack.
"""

from __future__ import annotations

import numpy as np

from .data_loader import DatasetSplit


def random_one_percent(split: DatasetSplit, seed: int, month_idx: int, frac: float = 0.01) -> DatasetSplit:
  """Uniform-random 1% sample, deterministic given (seed, month_idx)."""
  rng = np.random.default_rng(seed * 1_000_003 + month_idx)
  n = len(split)
  k = max(1, int(round(frac * n)))
  idx = rng.choice(n, size=k, replace=False)
  return DatasetSplit(name=f"{split.name}@rand", df=split.df.iloc[idx].reset_index(drop=True))


def uncertainty_top_k(
  split: DatasetSplit,
  proba: np.ndarray,
  seed: int,
  month_idx: int,
  frac: float = 0.01,
) -> DatasetSplit:
  """Pick the `frac` of samples with predicted P(malware) closest to 0.5.

  Ties broken by tiny RNG noise so different seeds give different samples
  among equally-uncertain points (otherwise sort would be deterministic by
  array order).
  """
  n = len(split)
  k = max(1, int(round(frac * n)))
  distance_from_boundary = np.abs(np.asarray(proba) - 0.5)
  rng = np.random.default_rng(seed * 1_000_011 + month_idx)
  tiebreak = rng.uniform(0.0, 1e-12, size=n)
  order = np.lexsort((tiebreak, distance_from_boundary))
  idx = order[:k]
  return DatasetSplit(name=f"{split.name}@unc", df=split.df.iloc[idx].reset_index(drop=True))
