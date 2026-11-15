"""Drift monitoring: PSI and the Kolmogorov-Smirnov test.

Run from the repository root (after training):

    python -m src.drift

With no arguments it runs two checks and saves reports/drift_report.json:

1. training data vs test data (the unlabelled copy of the test customers).
   Both come from the same snapshot, so this should come out stable. It
   proves the monitor does not raise false alarms.
2. training data vs a deliberately shifted copy of the test data. This
   should raise an alert. It proves the monitor can see real drift.

To compare any two files:

    python -m src.drift --reference old.csv --current new.csv

How to read PSI (Population Stability Index), the usual rule of thumb:
    below 0.10      stable
    0.10 to 0.25    warning, look into it
    above 0.25      alert, the population has shifted
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src import config
from src.data_pipeline import clean_features
from src.inference import ChurnScorer

PSI_WARNING = 0.10
PSI_ALERT = 0.25
KS_STATISTIC_ALERT = 0.10   # largest gap between the two cumulative curves
KS_P_VALUE = 0.01
SMALL = 1e-4                # stands in for an empty bin, avoids log(0)


def _psi_from_shares(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI = sum over bins of (actual - expected) * ln(actual / expected)."""
    expected = np.clip(expected, SMALL, None)
    actual = np.clip(actual, SMALL, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_numeric(reference, current, bins: int = 10) -> float:
    """PSI for a numeric column. Bin edges are deciles of the reference."""
    reference = pd.Series(reference).dropna().to_numpy(dtype=float)
    current = pd.Series(current).dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    # Open the outer bins so new extreme values still land in a bin.
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    return _psi_from_shares(expected, actual)


def psi_categorical(reference, current) -> float:
    """PSI for a text column. Each category is one bin, missing is a bin too."""
    reference = pd.Series(reference).fillna("(missing)").astype(str)
    current = pd.Series(current).fillna("(missing)").astype(str)
    categories = sorted(set(reference.unique()) | set(current.unique()))
    expected = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    actual = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return _psi_from_shares(expected, actual)


def psi_status(value: float) -> str:
    if value >= PSI_ALERT:
        return "alert"
    if value >= PSI_WARNING:
        return "warning"
    return "stable"


def drift_report(reference_raw: pd.DataFrame, current_raw: pd.DataFrame, scorer=None) -> dict:
    """Compare two raw customer tables feature by feature."""
    reference = clean_features(reference_raw)
    current = clean_features(current_raw)
    rows = []

    for col in config.RAW_NUMERIC:
        psi = psi_numeric(reference[col], current[col])
        ks = ks_2samp(reference[col].dropna(), current[col].dropna())
        ks_flag = bool(ks.statistic >= KS_STATISTIC_ALERT and ks.pvalue < KS_P_VALUE)
        rows.append({
            "feature": col, "type": "numeric", "psi": round(psi, 4), "psi_status": psi_status(psi),
            "ks_statistic": round(float(ks.statistic), 4), "ks_p_value": float(f"{ks.pvalue:.3g}"),
            "ks_flag": ks_flag,
        })

    for col in config.RAW_CATEGORICAL:
        psi = psi_categorical(reference[col], current[col])
        rows.append({"feature": col, "type": "categorical", "psi": round(psi, 4),
                     "psi_status": psi_status(psi)})

    # Drift of the model output itself. It needs no labels, so it is the
    # earliest signal available in production.
    if scorer is not None and scorer.model_ready:
        ref_scores = scorer.score(reference_raw)["churn_score"]
        cur_scores = scorer.score(current_raw)["churn_score"]
        psi = psi_numeric(ref_scores, cur_scores)
        ks = ks_2samp(ref_scores, cur_scores)
        rows.append({
            "feature": "model_score", "type": "score", "psi": round(psi, 4),
            "psi_status": psi_status(psi), "ks_statistic": round(float(ks.statistic), 4),
            "ks_p_value": float(f"{ks.pvalue:.3g}"),
            "ks_flag": bool(ks.statistic >= KS_STATISTIC_ALERT and ks.pvalue < KS_P_VALUE),
        })

    statuses = [row["psi_status"] for row in rows]
    overall = "alert" if "alert" in statuses else "warning" if "warning" in statuses else "stable"
    return {
        "reference_rows": int(len(reference_raw)),
        "current_rows": int(len(current_raw)),
        "overall_status": overall,
        "features_in_alert": [r["feature"] for r in rows if r["psi_status"] == "alert"],
        "features_in_warning": [r["feature"] for r in rows if r["psi_status"] == "warning"],
        "features": rows,
    }


def make_shifted_sample(raw: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Build a drifted copy of a customer table for the alert test.

    The story: a price rise, a campaign that moves long-contract customers
    to monthly plans, a younger customer base, and a new payment method
    that did not exist when the model was trained.
    """
    rng = np.random.default_rng(seed)
    shifted = raw.copy()

    shifted["MonthlyCharges"] = (pd.to_numeric(shifted["MonthlyCharges"], errors="coerce") * 1.25).round(2)
    shifted["tenure"] = (pd.to_numeric(shifted["tenure"], errors="coerce") * 0.6).round().astype(int)

    long_contract = shifted["Contract"].isin(["One year", "Two year"]).to_numpy()
    move = long_contract & (rng.random(len(shifted)) < 0.35)
    shifted.loc[move, "Contract"] = "Month-to-month"

    new_method = rng.random(len(shifted)) < 0.15
    shifted.loc[new_method, "PaymentMethod"] = "UPI AutoPay"
    return shifted


def print_report(title: str, report: dict) -> None:
    print(f"\n{title}")
    print(f"  rows: reference {report['reference_rows']:,}, current {report['current_rows']:,}")
    print(f"  overall status: {report['overall_status'].upper()}")
    print(f"  {'feature':18s} {'PSI':>8s} {'status':>8s} {'KS stat':>8s} {'KS flag':>8s}")
    for row in report["features"]:
        ks_stat = f"{row['ks_statistic']:.4f}" if "ks_statistic" in row else "-"
        ks_flag = str(row["ks_flag"]) if "ks_flag" in row else "-"
        print(f"  {row['feature']:18s} {row['psi']:8.4f} {row['psi_status']:>8s} {ks_stat:>8s} {ks_flag:>8s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PSI and KS drift check between two customer files.")
    parser.add_argument("--reference", help="CSV with the reference (training-time) data")
    parser.add_argument("--current", help="CSV with the current (production) data")
    args = parser.parse_args()

    scorer = ChurnScorer()

    if args.reference and args.current:
        report = drift_report(pd.read_csv(args.reference), pd.read_csv(args.current), scorer)
        print_report(f"Drift check: {args.reference} vs {args.current}", report)
        return

    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    # The test customers without their labels. Drift monitoring never needs
    # labels, and this keeps the labelled test file for evaluate.py only.
    test = pd.read_csv(config.PROCESSED_DIR / "batch_input_unseen.csv")

    stable = drift_report(train, test, scorer)
    print_report("Check 1: training data vs test data (expected: stable)", stable)

    shifted = drift_report(train, make_shifted_sample(test), scorer)
    print_report("Check 2: training data vs deliberately shifted data (expected: alert)", shifted)

    output = {
        "thresholds": {"psi_warning": PSI_WARNING, "psi_alert": PSI_ALERT,
                       "ks_statistic_alert": KS_STATISTIC_ALERT, "ks_p_value": KS_P_VALUE},
        "train_vs_test": stable,
        "train_vs_shifted": shifted,
    }
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORT_DIR / "drift_report.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print("\nSaved reports/drift_report.json")


if __name__ == "__main__":
    main()
