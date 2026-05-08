"""Unit tests for common.data_loader using small synthetic parquet files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.data_loader import (  # noqa: E402
  FEATURE_COLS,
  NUM_FEATURES,
  DatasetSplit,
  bodmas_monthly_iter,
  load_for_testing,
)


def _make_ember_df(n: int, sha_prefix: str, year: int, with_unlabeled: int = 0) -> pd.DataFrame:
  """Build a tiny EMBER-shape dataframe (`appeared`, `label`, `avclass`, features)."""
  rng = np.random.default_rng(123)
  sha = [f"{sha_prefix}_{i:08d}" for i in range(n)]
  appeared = pd.date_range(f"{year}-01-01", periods=n, freq="D")
  label = rng.integers(0, 2, n).astype(np.int8)
  if with_unlabeled:
  label[:with_unlabeled] = -1
  avclass = ["family_a"] * (n // 2) + ["family_b"] * (n - n // 2)
  feats = rng.standard_normal((n, NUM_FEATURES)).astype(np.float32)
  df = pd.DataFrame(feats, columns=FEATURE_COLS)
  df.insert(0, "sha256", sha)
  df.insert(1, "appeared", appeared)
  df.insert(2, "label", label)
  df.insert(3, "avclass", avclass)
  return df


def _make_bodmas_df(n: int, sha_prefix: str, year_start: str = "2019-08-01") -> pd.DataFrame:
  rng = np.random.default_rng(42)
  sha = [f"{sha_prefix}_{i:08d}" for i in range(n)]
  timestamp = pd.date_range(year_start, periods=n, freq="D")
  label = rng.integers(0, 2, n).astype(np.int8)
  family = ["benign" if l == 0 else f"fam_{i % 5}" for i, l in enumerate(label)]
  category = ["clean" if l == 0 else "malicious" for l in label]
  feats = rng.standard_normal((n, NUM_FEATURES)).astype(np.float32)
  df = pd.DataFrame(feats, columns=FEATURE_COLS)
  df.insert(0, "sha256", sha)
  df.insert(1, "label", label)
  df.insert(2, "timestamp", timestamp)
  df.insert(3, "family", family)
  df.insert(4, "category", category)
  return df


@pytest.fixture
def synth_corpus(tmp_path):
  paths = {
  "ember2017_train": str(tmp_path / "e17_train.parquet"),
  "ember2017_test":  str(tmp_path / "e17_test.parquet"),
  "ember2018_train": str(tmp_path / "e18_train.parquet"),
  "ember2018_test":  str(tmp_path / "e18_test.parquet"),
  "bodmas":  str(tmp_path / "bodmas.parquet"),
  }
  _make_ember_df(20, "e17train", 2017, with_unlabeled=4).to_parquet(paths["ember2017_train"])
  _make_ember_df(10, "e17test",  2017).to_parquet(paths["ember2017_test"])
  _make_ember_df(20, "e18train", 2018, with_unlabeled=4).to_parquet(paths["ember2018_train"])
  _make_ember_df(10, "e18test",  2018).to_parquet(paths["ember2018_test"])
  _make_bodmas_df(20, "bodmas").to_parquet(paths["bodmas"])
  return paths


def test_unlabeled_dropped(synth_corpus):
  corpus = load_for_testing(synth_corpus)
  # Each EMBER train has 4 unlabeled, so train must have 16 rows.
  assert len(corpus["ember2017_train"]) == 16
  assert len(corpus["ember2018_train"]) == 16
  assert (corpus["ember2017_train"].df["label"] != -1).all()


def test_no_cross_dataset_dupes(synth_corpus):
  # Inject 3 duplicate hashes between EMBER 2017 train and BODMAS:
  bodmas = pd.read_parquet(synth_corpus["bodmas"])
  bodmas.loc[:2, "sha256"] = ["e17train_00000004", "e17train_00000005", "e17train_00000006"]
  bodmas.to_parquet(synth_corpus["bodmas"])

  corpus = load_for_testing(synth_corpus)
  seen = set()
  for split in corpus.values():
  s = set(split.df["sha256"])
  overlap = seen & s
  assert not overlap, f"{split.name} has cross-dupes: {overlap}"
  seen |= s
  # BODMAS should have lost the 3 dupes (priority: EMBER train > BODMAS).
  assert len(corpus["bodmas"]) == 17


def test_features_finite(synth_corpus):
  corpus = load_for_testing(synth_corpus)
  for split in corpus.values():
  assert np.isfinite(split.X).all()


def test_dataset_split_X_y_shape(synth_corpus):
  corpus = load_for_testing(synth_corpus)
  split = corpus["bodmas"]
  assert split.X.shape == (len(split), NUM_FEATURES)
  assert split.y.shape == (len(split),)
  assert split.X.dtype == np.float32
  assert split.y.dtype == np.int8


def test_filter_month():
  df = _make_bodmas_df(60, "monthly", year_start="2019-08-01")
  split = DatasetSplit(name="bodmas", df=df)
  aug = split.filter_month("2019-08")
  sep = split.filter_month("2019-09")
  assert len(aug) + len(sep) <= len(split)
  # Aug 2019 has 31 days, freq=D -> first 31 dates fall in Aug.
  assert len(aug) == 31


def test_bodmas_monthly_iter_chronological(synth_corpus):
  corpus = load_for_testing(synth_corpus)
  months = [ym for ym, _ in bodmas_monthly_iter(corpus)]
  assert months == sorted(months)
