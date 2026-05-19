"""Determinism + correctness tests for 1% sampling functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.data_loader import FEATURE_COLS, DatasetSplit  # noqa: E402
from common.sampling import (  # noqa: E402
    random_one_percent,
    uncertainty_top_k,
)


def _make_split(n: int, name: str = "test") -> DatasetSplit:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLS))).astype(np.float32),
                      columns=FEATURE_COLS)
    df.insert(0, "sha256", [f"{name}_{i}" for i in range(n)])
    df.insert(1, "timestamp", pd.date_range("2019-08-01", periods=n, freq="h"))
    df.insert(2, "label", rng.integers(0, 2, n).astype(np.int8))
    df.insert(3, "family", ["benign" if l == 0 else "fam_x"
                            for l in df["label"]])
    df.insert(4, "category", "test")
    return DatasetSplit(name=name, df=df)


def test_one_percent_size():
    split = _make_split(1000)
    sub = random_one_percent(split, seed=42, month_idx=1)
    assert len(sub) == 10  # 1% of 1000


def test_one_percent_deterministic():
    split = _make_split(1000)
    a = random_one_percent(split, seed=42, month_idx=1)
    b = random_one_percent(split, seed=42, month_idx=1)
    assert (a.df["sha256"].values == b.df["sha256"].values).all()


def test_one_percent_seed_changes_sample():
    split = _make_split(1000)
    a = random_one_percent(split, seed=42, month_idx=1)
    c = random_one_percent(split, seed=99, month_idx=1)
    assert not (a.df["sha256"].values == c.df["sha256"].values).all()


def test_one_percent_month_changes_sample():
    split = _make_split(1000)
    a = random_one_percent(split, seed=42, month_idx=1)
    b = random_one_percent(split, seed=42, month_idx=2)
    assert not (a.df["sha256"].values == b.df["sha256"].values).all()


def test_uncertainty_picks_boundary():
    split = _make_split(100)
    # Probabilities crafted so first 1 has prob exactly 0.5 (most uncertain).
    proba = np.full(100, 0.95)
    proba[7] = 0.5
    sub = uncertainty_top_k(split, proba, frac=0.01, seed=0, month_idx=1)
    assert len(sub) == 1
    assert sub.df["sha256"].iloc[0] == split.df["sha256"].iloc[7]


def test_uncertainty_top_k_closest():
    split = _make_split(100)
    proba = np.full(100, 0.99)
    # Place the 5 most-uncertain at indices [3, 13, 23, 33, 43].
    most_uncertain_idx = [3, 13, 23, 33, 43]
    for rank, i in enumerate(most_uncertain_idx):
        proba[i] = 0.50 + 0.001 * rank
    sub = uncertainty_top_k(split, proba, frac=0.05, seed=0, month_idx=1)
    selected = set(sub.df["sha256"].tolist())
    expected = set(split.df["sha256"].iloc[most_uncertain_idx].tolist())
    assert selected == expected


def test_random_one_percent_explicit_frac():
    split = _make_split(500)
    a = random_one_percent(split, seed=7, month_idx=2, frac=0.01)
    b = random_one_percent(split, seed=7, month_idx=2, frac=0.01)
    assert len(a) == 5
    assert (a.df["sha256"].values == b.df["sha256"].values).all()
