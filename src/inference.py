"""Scoring code shared by the API and the batch script.

Both entry points call ChurnScorer, so a customer gets the same answer
whether the request arrives over HTTP or inside a nightly batch file.

Degradation ladder (what happens when things go wrong):

1. Normal:   the trained pipeline scores the row.            served_by = "model"
2. Degraded: the model is missing or crashes. The segment
             rule (contract type x tenure band) answers.     served_by = "fallback"
3. Down:     the metadata file is missing too, so even the
             rule has no numbers. ScoringUnavailable is
             raised and the API turns it into HTTP 503.

The scorer remembers whether the most recent scoring attempt had to use the
fallback because the model raised an error. /health uses this, so a model
that is loaded but crashing shows up as "degraded" and not as "ok". The
next request that the model answers clears the flag again.
"""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_pipeline import clean_features
from src.fallback import segment_scores

logger = logging.getLogger("churn.inference")


class ScoringUnavailable(Exception):
    """Neither the model nor the fallback rule can produce a score."""


class ChurnScorer:
    """Loads the artifacts once and scores raw customer rows."""

    def __init__(self, artifact_dir=config.ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.metadata = None   # thresholds, fallback table, training facts
        self.pipeline = None   # fitted preprocessing + model
        self.load_error = None
        # Error text of the most recent scoring attempt if the model failed
        # and the fallback had to answer. None after a normal model answer.
        self.last_model_error = None
        self._load()

    def _load(self) -> None:
        # Metadata first. Without it there is no threshold and no fallback.
        try:
            with open(self.artifact_dir / config.METADATA_FILE, encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        except Exception as exc:
            self.load_error = f"metadata not loaded: {exc}"
            logger.error(self.load_error)
            return

        try:
            name = self.metadata["production_model"]
            artifact = self.metadata["candidates"][name]["artifact"]
            self.pipeline = joblib.load(self.artifact_dir / artifact)
            logger.info("Loaded model '%s' version %s", name, self.model_version)
        except Exception as exc:
            self.load_error = f"model not loaded: {exc}"
            logger.error(self.load_error)

    # ------------------------------------------------------------ status
    @property
    def model_ready(self) -> bool:
        """The model artifact and the metadata are loaded."""
        return self.pipeline is not None and self.metadata is not None

    @property
    def model_healthy(self) -> bool:
        """The model is loaded and did not fail on the most recent scoring attempt."""
        return self.model_ready and self.last_model_error is None

    @property
    def fallback_ready(self) -> bool:
        return self.metadata is not None

    @property
    def model_version(self) -> str:
        if self.metadata is None:
            return "unavailable"
        return f"{self.metadata['production_model']}-{self.metadata['model_version']}"

    # ----------------------------------------------------------- scoring
    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Score a table of raw customer rows (used by the batch script)."""
        if self.metadata is None:
            raise ScoringUnavailable("Model metadata is not loaded, cannot score.")
        return self._score_features(clean_features(raw))

    def score_one(self, record: dict) -> dict:
        """Score a single customer (used by the API) and explain any issues."""
        if self.metadata is None:
            raise ScoringUnavailable("Model metadata is not loaded, cannot score.")
        features = clean_features(pd.DataFrame([record]))
        row = self._score_features(features).iloc[0]
        return {
            "churn_score": float(row["churn_score"]),
            "churn_prediction": int(row["churn_prediction"]),
            "threshold": float(row["threshold"]),
            "served_by": str(row["served_by"]),
            "warnings": self._warnings_for_row(features.iloc[0]),
        }

    def _score_features(self, features: pd.DataFrame) -> pd.DataFrame:
        # Count data problems before fixing them, so they can be reported.
        issues = self._count_issues(features)
        features = self._clip_to_training_range(features)

        served_by = "model"
        try:
            if self.pipeline is None:
                raise RuntimeError("model artifact is not loaded")
            scores = self.pipeline.predict_proba(features)[:, 1]
            name = self.metadata["production_model"]
            threshold = self.metadata["candidates"][name]["threshold"]
            self.last_model_error = None   # the model answered, so it is healthy again
        except Exception as exc:
            # Graceful degradation: answer with the segment rule instead.
            self.last_model_error = str(exc)   # remembered for /health
            logger.warning("Model scoring failed (%s). Using the fallback rule.", exc)
            try:
                scores = segment_scores(features, self.metadata["fallback"]["segment_table"])
                threshold = self.metadata["fallback"]["threshold"]
                served_by = "fallback"
            except Exception as fallback_exc:
                raise ScoringUnavailable(
                    f"Model and fallback both failed: {fallback_exc}"
                ) from fallback_exc

        result = pd.DataFrame(index=features.index)
        result["churn_score"] = np.round(scores, 6)
        result["churn_prediction"] = (scores >= threshold).astype(int)
        result["threshold"] = threshold
        result["served_by"] = served_by
        return pd.concat([result, issues], axis=1)

    # ------------------------------------------------------ data checks
    def _count_issues(self, features: pd.DataFrame) -> pd.DataFrame:
        """Per-row counts of missing, unknown and out-of-range values."""
        issues = pd.DataFrame(index=features.index)
        issues["n_missing_fields"] = features[config.RAW_FEATURES].isna().sum(axis=1)

        unknown = pd.Series(0, index=features.index)
        for col, known in self.metadata["known_categories"].items():
            values = features[col]
            unknown += (values.notna() & ~values.isin(known)).astype(int)
        issues["n_unknown_categories"] = unknown

        out_of_range = pd.Series(0, index=features.index)
        for col, limits in self.metadata["numeric_ranges"].items():
            values = features[col]
            out_of_range += ((values < limits["min"]) | (values > limits["max"])).astype(int)
        issues["n_clipped_values"] = out_of_range
        return issues

    def _clip_to_training_range(self, features: pd.DataFrame) -> pd.DataFrame:
        """Pull extreme numbers back to the range seen in training.

        The model never saw values outside this range, so I do not let it
        extrapolate. The row is still scored and the issue is reported.
        """
        features = features.copy()
        for col, limits in self.metadata["numeric_ranges"].items():
            features[col] = features[col].clip(lower=limits["min"], upper=limits["max"])
        return features

    def _warnings_for_row(self, row: pd.Series) -> list:
        """Readable warnings for one cleaned row (before clipping)."""
        warnings = []
        missing = [col for col in config.RAW_FEATURES if pd.isna(row[col])]
        if missing:
            warnings.append("missing fields were imputed: " + ", ".join(missing))
        for col, known in self.metadata["known_categories"].items():
            value = row[col]
            if not pd.isna(value) and value not in known:
                warnings.append(f"unknown category for {col}: '{value}' (not seen in training)")
        for col, limits in self.metadata["numeric_ranges"].items():
            value = row[col]
            if not pd.isna(value) and (value < limits["min"] or value > limits["max"]):
                warnings.append(
                    f"{col}={value:g} is outside the training range "
                    f"[{limits['min']:g}, {limits['max']:g}] and was clipped"
                )
        return warnings
