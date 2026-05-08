"""Classifier factory matching the original manuscript Section 3.2.

Three classifiers with the same hyperparameters as published:
  LightGBM: 500 estimators, 64 leaves, is_unbalance=True
  RandomForest: 200 trees, max_depth=30, class_weight='balanced'
  MLP: 2 hidden layers (256, 128), ReLU, with StandardScaler + RandomOverSampler

Determinism: every estimator gets `random_state=seed`. LightGBM additionally sets
`deterministic=True` and `force_row_wise=True` (mirrors P18 fix for XGBoost-style
non-determinism).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ClassifierName = Literal["lightgbm", "rf", "mlp"]


def make_classifier(name: ClassifierName, *, seed: int, n_jobs: int = 12):
  """Build a fresh classifier matching the manuscript spec.

  n_jobs default 12 fits the compute server's 60-core / 5-parallel-process layout.
  """
  if name == "lightgbm":
  return LGBMClassifier(
  n_estimators=500,
  num_leaves=64,
  is_unbalance=True,
  random_state=seed,
  n_jobs=n_jobs,
  deterministic=True,
  force_row_wise=True,
  verbose=-1,
  )
  if name == "rf":
  return RandomForestClassifier(
  n_estimators=200,
  max_depth=30,
  class_weight="balanced",
  random_state=seed,
  n_jobs=n_jobs,
  )
  if name == "mlp":
  return ImbPipeline([
  ("scaler", StandardScaler()),
  ("oversample", RandomOverSampler(random_state=seed)),
  ("mlp", MLPClassifier(
  hidden_layer_sizes=(256, 128),
  activation="relu",
  random_state=seed,
  max_iter=200,
  early_stopping=True,
  validation_fraction=0.1,
  )),
  ])
  raise ValueError(f"Unknown classifier name: {name}")


def predict_proba_malware(clf, X: np.ndarray) -> np.ndarray:
  """Return P(label=1) for samples in X."""
  proba = clf.predict_proba(X)
  classes = getattr(clf, "classes_", None)
  if classes is None and hasattr(clf, "named_steps") and "mlp" in clf.named_steps:
  classes = clf.named_steps["mlp"].classes_
  if classes is None:
  raise RuntimeError("Classifier missing classes_ attribute")
  pos_idx = int(np.where(classes == 1)[0][0])
  return proba[:, pos_idx]
