"""Data loading + cleaning for IEEE Access resubmission.

Loads EMBER 2017 v2 + EMBER 2018 + BODMAS, applies the same cleaning rules
described in the original manuscript Section 3.1:
  1. Remove unlabeled samples (label == -1) from EMBER train sets
  2. Remove cross-dataset hash duplicates (priority: test > train, older > newer, EMBER > BODMAS)
  3. Unify column schemas across datasets
  4. Verify features have no NaN/Inf

Final corpus per manuscript: 1,684,338 samples (800K EMBER 2017 + 750K EMBER 2018 + 134K BODMAS).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

NUM_FEATURES = 2381
FEATURE_COLS = [f"feature_{i:04d}" for i in range(NUM_FEATURES)]

# Default paths (node11). Override via constructor for local dev / testing.
DEFAULT_PATHS = {
    "ember2017_train": "/home/apps/ember2017_2/ember2017_2_train.parquet",
    "ember2017_test":  "/home/apps/ember2017_2/ember2017_2_test.parquet",
    "ember2018_train": "/home/apps/ember2018/ember2018_train.parquet",
    "ember2018_test":  "/home/apps/ember2018/ember2018_test.parquet",
    "bodmas":          "/home/apps/bodmas/bodmas.parquet",
}


@dataclass(frozen=True)
class DatasetSplit:
    """A unified-schema slice with sha256, timestamp, label, family, features."""
    name: str
    df: pd.DataFrame

    def __post_init__(self) -> None:
        required = {"sha256", "timestamp", "label", "family"} | set(FEATURE_COLS)
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{self.name}: missing columns {sorted(missing)[:5]}...")

    @property
    def X(self) -> np.ndarray:
        return self.df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)

    @property
    def y(self) -> np.ndarray:
        return self.df["label"].to_numpy(dtype=np.int8, copy=False)

    @property
    def family(self) -> np.ndarray:
        return self.df["family"].fillna("__benign__").to_numpy()

    def filter_month(self, year_month: str) -> "DatasetSplit":
        """Return slice for a YYYY-MM string."""
        ts = pd.to_datetime(self.df["timestamp"])
        mask = ts.dt.strftime("%Y-%m") == year_month
        return DatasetSplit(name=f"{self.name}@{year_month}", df=self.df.loc[mask].reset_index(drop=True))

    def __len__(self) -> int:
        return len(self.df)


def _normalize_ember(df: pd.DataFrame) -> pd.DataFrame:
    """Rename EMBER columns to unified schema."""
    out = df.rename(columns={"appeared": "timestamp", "avclass": "family"})
    if "category" not in out.columns:
        out["category"] = pd.Series([pd.NA] * len(out), dtype="object")
    return out


def _normalize_bodmas(df: pd.DataFrame) -> pd.DataFrame:
    """BODMAS already has 'timestamp' + 'family' + 'category'; no rename needed.

    We keep ALL rows here (including ~10K out-of-window outliers timestamped
    before 2019-08) so the corpus size and cross-dataset dedup count match the
    original manuscript's accounting (1,684,338 unique samples; 97 cross-dupes).
    Monthly iteration restricts to the 2019-08 .. 2020-09 analysis window.
    """
    if "category" not in df.columns:
        df = df.assign(category=pd.NA)
    return df


def _drop_unlabeled(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = len(df)
    out = df[df["label"] != -1].reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"[{name}] dropped {dropped:,} unlabeled (label=-1) rows; remaining {len(out):,}")
    return out


def _validate_features(df: pd.DataFrame, name: str) -> None:
    arr = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name}: {n_bad} non-finite feature values")


def _drop_cross_dataset_dupes(
    splits: dict[str, pd.DataFrame],
    priority: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Remove SHA-256 dupes across splits. Earlier in `priority` = kept."""
    seen: set[str] = set()
    kept: dict[str, pd.DataFrame] = {}
    for name in priority:
        df = splits[name]
        before = len(df)
        mask = ~df["sha256"].isin(seen)
        kept[name] = df.loc[mask].reset_index(drop=True)
        new_hashes = df.loc[mask, "sha256"].tolist()
        seen.update(new_hashes)
        dropped = before - len(kept[name])
        if dropped:
            print(f"[dedup] {name}: dropped {dropped:,} cross-dataset dupes; remaining {len(kept[name]):,}")
    return kept


@functools.lru_cache(maxsize=None)
def load_corpus(paths_key: str = "default") -> dict[str, DatasetSplit]:
    """Load and clean the unified 2017-2020 corpus.

    Returns dict with keys: ember2017_train, ember2017_test, ember2018_train,
    ember2018_test, bodmas. Each value is a DatasetSplit with unified schema.

    The cleaning order matches manuscript Section 3.1:
      1. drop unlabeled per-source
      2. priority dedup: test > train within EMBER; older > newer; EMBER > BODMAS
      3. validate finiteness
    """
    paths = DEFAULT_PATHS  # only "default" supported via cache; pass explicit dict for tests
    raw: dict[str, pd.DataFrame] = {}
    for k, p in paths.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"{k}: {p}")
        df = pd.read_parquet(p)
        if "ember" in k:
            df = _normalize_ember(df)
        else:
            df = _normalize_bodmas(df)
        df = _drop_unlabeled(df, k)
        raw[k] = df

    # Priority order: test > train within era, older era > newer, EMBER > BODMAS
    priority = [
        "ember2017_test",
        "ember2017_train",
        "ember2018_test",
        "ember2018_train",
        "bodmas",
    ]
    cleaned = _drop_cross_dataset_dupes(raw, priority)

    out: dict[str, DatasetSplit] = {}
    for name, df in cleaned.items():
        _validate_features(df, name)
        out[name] = DatasetSplit(name=name, df=df)
    return out


def load_for_testing(paths: dict[str, str]) -> dict[str, DatasetSplit]:
    """Bypass-cache loader for unit tests with custom paths."""
    raw = {k: pd.read_parquet(p) for k, p in paths.items()}
    raw = {k: (_normalize_ember(v) if "ember" in k else _normalize_bodmas(v)) for k, v in raw.items()}
    raw = {k: _drop_unlabeled(v, k) for k, v in raw.items()}
    priority = [
        "ember2017_test", "ember2017_train",
        "ember2018_test", "ember2018_train",
        "bodmas",
    ]
    available = [p for p in priority if p in raw]
    cleaned = _drop_cross_dataset_dupes({k: raw[k] for k in available}, available)
    return {k: DatasetSplit(name=k, df=v) for k, v in cleaned.items()}


BODMAS_WINDOW_START = "2019-08-01"
BODMAS_WINDOW_END = "2020-10-01"  # exclusive


def bodmas_monthly_iter(corpus: dict[str, DatasetSplit]) -> Iterable[tuple[str, DatasetSplit]]:
    """Yield (YYYY-MM, monthly_slice) for BODMAS in chronological order.

    Restricts to the 2019-08 .. 2020-09 analysis window (matches manuscript
    Tables 5/6/7). Out-of-window samples (~10K rows timestamped before 2019-08)
    are kept in the corpus for total-count accounting but excluded from
    monthly drift analysis.

    Per Tables 5/6/7, the 14 months are 2019-08 through 2020-09. Exp 8/9 use
    2019-08 as training month and 2019-09..2020-09 (13 months) as eval stream.
    """
    bodmas = corpus["bodmas"]
    ts = pd.to_datetime(bodmas.df["timestamp"])
    tz = ts.dt.tz
    in_window = (ts >= pd.Timestamp(BODMAS_WINDOW_START, tz=tz)) & (
        ts < pd.Timestamp(BODMAS_WINDOW_END, tz=tz)
    )
    windowed = bodmas.df.loc[in_window].reset_index(drop=True)
    ts_w = pd.to_datetime(windowed["timestamp"])
    months = sorted(ts_w.dt.strftime("%Y-%m").unique())
    for ym in months:
        mask = ts_w.dt.strftime("%Y-%m") == ym
        sub = windowed.loc[mask].reset_index(drop=True)
        yield ym, DatasetSplit(name=f"bodmas@{ym}", df=sub)


if __name__ == "__main__":
    corpus = load_corpus()
    for name, split in corpus.items():
        print(f"{name}: {len(split):,} rows")
    total = sum(len(s) for s in corpus.values())
    print(f"TOTAL: {total:,} (manuscript reports 1,684,338)")
