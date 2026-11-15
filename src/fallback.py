"""A simple, readable churn rule based on contract type and tenure.

It has two jobs in this project:

1. Baseline. It shows how far a plain business rule gets, so I can tell
   whether the machine learning model is actually worth having.
2. Fallback. If the model cannot score a request (missing artifact, a
   crash inside the model), the API answers with this rule instead of
   failing. The response says clearly that the fallback was used.

The rule is a lookup table: the churn rate seen in the TRAINING data for
each (contract type, tenure band) group. It only needs two fields.
"""
import numpy as np
import pandas as pd

# Tenure bands in months. Churn is highest in the first year.
TENURE_BANDS = [(0, 12, "0-12"), (13, 24, "13-24"), (25, 48, "25-48"), (49, None, "49+")]

# A group with fewer training rows than this uses the overall churn rate.
MIN_GROUP_SIZE = 30


def tenure_band(tenure) -> str:
    """Return the tenure band label for one tenure value in months."""
    if tenure is None or pd.isna(tenure):
        return "unknown"
    for low, high, label in TENURE_BANDS:
        if tenure >= low and (high is None or tenure <= high):
            return label
    return "unknown"


def _segment_keys(features: pd.DataFrame) -> pd.Series:
    """Build the lookup key 'Contract|tenure band' for every row."""
    bands = features["tenure"].map(tenure_band)
    contracts = features["Contract"].fillna("unknown").astype(str)
    return contracts + "|" + bands


def fit_segment_table(features: pd.DataFrame, target: pd.Series) -> dict:
    """Learn the churn rate of each segment from the training data."""
    keys = _segment_keys(features)
    grouped = pd.DataFrame({"key": keys.values, "y": target.values}).groupby("key")["y"]
    stats = grouped.agg(["mean", "count"])

    global_rate = float(target.mean())
    segments = {}
    for key, row in stats.iterrows():
        if row["count"] >= MIN_GROUP_SIZE:
            segments[key] = {"rate": round(float(row["mean"]), 4), "n": int(row["count"])}
    return {"global_rate": round(global_rate, 4), "segments": segments}


def segment_scores(features: pd.DataFrame, table: dict) -> np.ndarray:
    """Score rows with the lookup table. Unknown groups get the overall rate."""
    rates = {key: value["rate"] for key, value in table["segments"].items()}
    keys = _segment_keys(features)
    return keys.map(rates).fillna(table["global_rate"]).to_numpy(dtype=float)
