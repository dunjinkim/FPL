"""
Overlap classifier: distinguish single rods from overlapping clusters.

Two modes:
1. Rule-based (default, no training data needed)
   - solidity < threshold  → overlap
   - area > median * area_multiplier  → overlap

2. ML mode (Random Forest, activated once enough labels are collected)
   - Features: area_px, aspect_ratio, solidity, extent, circularity,
               convexity_defect_count, mean_intensity, std_intensity, perimeter
   - Labels: "single" | "overlap"
   - Model persisted to disk with joblib
"""

from __future__ import annotations

import os
import csv
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

LABEL_SINGLE = "single"
LABEL_OVERLAP = "overlap"
LABEL_UNKNOWN = "unknown"

_FEATURE_COLS = [
    "area_px2",
    "aspect_ratio",
    "solidity",
    "circularity",
    "convexity_defect_count",
    "mean_intensity",
    "std_intensity",
]

MIN_SAMPLES_PER_CLASS = 5  # minimum labels needed before ML training


class OverlapClassifier:
    """
    Manages rule-based + ML-based overlap detection.

    Parameters
    ----------
    model_path    : path to save/load the trained model (.pkl)
    labels_path   : path to the CSV accumulating user labels
    solidity_threshold : rule-based: objects with solidity < this are overlap
    area_multiplier    : rule-based: objects with area > median * this are overlap
    """

    def __init__(
        self,
        model_path: str | Path = "models/overlap_model.pkl",
        labels_path: str | Path = "training_data/labels.csv",
        solidity_threshold: float = 0.75,
        area_multiplier: float = 2.5,
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.solidity_threshold = solidity_threshold
        self.area_multiplier = area_multiplier

        self._pipeline: Pipeline | None = None
        self._cv_accuracy: float | None = None
        self._labels_df: pd.DataFrame = self._load_labels()

        # Try loading a saved model
        if self.model_path.exists() and _SKLEARN_OK:
            try:
                self._pipeline = joblib.load(self.model_path)
            except Exception:
                self._pipeline = None

    # ── Label management ────────────────────────────────────────────────────

    def _load_labels(self) -> pd.DataFrame:
        if self.labels_path.exists():
            try:
                return pd.read_csv(self.labels_path)
            except Exception:
                pass
        return pd.DataFrame(columns=_FEATURE_COLS + ["label"])

    def _save_labels(self) -> None:
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        self._labels_df.to_csv(self.labels_path, index=False)

    def add_label(self, rod_features: dict, label: Literal["single", "overlap"]) -> None:
        """Add a user-provided label for one rod."""
        row = {col: rod_features.get(col, np.nan) for col in _FEATURE_COLS}
        row["label"] = label
        self._labels_df = pd.concat(
            [self._labels_df, pd.DataFrame([row])], ignore_index=True
        )
        self._save_labels()

    def label_counts(self) -> dict[str, int]:
        counts = self._labels_df["label"].value_counts().to_dict()
        return {
            LABEL_SINGLE: counts.get(LABEL_SINGLE, 0),
            LABEL_OVERLAP: counts.get(LABEL_OVERLAP, 0),
        }

    def can_train(self) -> bool:
        c = self.label_counts()
        return (
            _SKLEARN_OK
            and c[LABEL_SINGLE] >= MIN_SAMPLES_PER_CLASS
            and c[LABEL_OVERLAP] >= MIN_SAMPLES_PER_CLASS
        )

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self) -> float:
        """
        Train a Random Forest classifier on accumulated labels.

        Returns
        -------
        Cross-validated accuracy (float, 0-1).
        """
        if not self.can_train():
            raise ValueError("Not enough labelled data to train.")

        df = self._labels_df.dropna(subset=_FEATURE_COLS + ["label"])
        X = df[_FEATURE_COLS].values
        y = df["label"].values

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipeline, X, y, cv=min(5, len(df) // 2))
        self._cv_accuracy = float(scores.mean())

        pipeline.fit(X, y)
        self._pipeline = pipeline

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.model_path)

        return self._cv_accuracy

    # ── Classification ───────────────────────────────────────────────────────

    def _rule_based(self, rod: dict, median_area: float) -> str:
        """Simple rule-based classification."""
        if rod.get("solidity", 1.0) < self.solidity_threshold:
            return LABEL_OVERLAP
        if rod.get("area_px2", 0) > median_area * self.area_multiplier:
            return LABEL_OVERLAP
        return LABEL_SINGLE

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each row in *df* as 'single' or 'overlap'.

        Parameters
        ----------
        df : DataFrame from measurer.measure_rods()

        Returns
        -------
        Same DataFrame with 'overlap_label' column filled.
        """
        df = df.copy()
        median_area = df["area_px2"].median() if not df.empty else 1.0

        if self._pipeline is not None:
            # ML mode
            feature_matrix = df[_FEATURE_COLS].fillna(0).values
            predictions = self._pipeline.predict(feature_matrix)
            df["overlap_label"] = predictions
        else:
            # Rule-based fallback
            df["overlap_label"] = df.apply(
                lambda row: self._rule_based(row.to_dict(), median_area), axis=1
            )

        return df

    @property
    def mode(self) -> str:
        return "ML" if self._pipeline is not None else "rule-based"

    @property
    def accuracy(self) -> float | None:
        return self._cv_accuracy
