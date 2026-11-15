"""Ingestion, cleaning and the leak-free train / validation / test split.

Run from the repository root:

    python -m src.data_pipeline

Two ideas matter in this file:

1. `clean_features` only uses row-wise rules. It never learns a statistic
   from the data (no mean, no median, no category list). That makes it safe
   to call on any partition, and the API calls the very same function, so
   training and serving cannot drift apart.

2. The split happens here, before any model or imputer is fitted. Everything
   that learns from data (imputer, scaler, encoder, model) lives in the
   scikit-learn pipeline in train.py and is fitted on the training file only.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python data/fetch_data.py' first."
        )
    return pd.read_csv(path)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw customer rows into the feature table the model expects.

    The same function is used for training data, batch files and single
    API requests. Columns that are absent are created as missing values so
    the imputer inside the model pipeline can fill them.
    """
    out = pd.DataFrame(index=df.index)

    # Numeric columns: anything that is not a number becomes missing (NaN).
    # In the raw file TotalCharges is text and has a few blank strings.
    for col in config.RAW_NUMERIC:
        values = df[col] if col in df.columns else np.nan
        out[col] = pd.to_numeric(values, errors="coerce")

    # A customer with tenure 0 has not been billed yet, so a blank
    # TotalCharges for that customer really means 0.
    not_billed_yet = out["TotalCharges"].isna() & (out["tenure"] == 0)
    out.loc[not_billed_yet, "TotalCharges"] = 0.0

    # Categorical columns: read as text, trim spaces, treat empty text as
    # missing. Missing values are stored as NaN so the imputer can see them.
    for col in config.RAW_CATEGORICAL:
        if col in df.columns:
            text = df[col].astype("string").str.strip()
        else:
            text = pd.Series(pd.NA, index=df.index, dtype="string")
        text = text.replace("", pd.NA)
        out[col] = text.astype(object).where(text.notna(), np.nan)

    # SeniorCitizen arrives as 0/1. I turn it into No/Yes so that every
    # categorical column is plain text and is encoded the same way.
    # ("0.0"/"1.0" appear when a CSV column has gaps and is read as float.)
    senior_map = {"0": "No", "1": "Yes", "0.0": "No", "1.0": "Yes"}
    out["SeniorCitizen"] = out["SeniorCitizen"].replace(senior_map)

    # Engineered feature: count of add-on services marked "Yes".
    addon_table = out[config.ADDON_SERVICES]
    out["num_addon_services"] = (addon_table == "Yes").sum(axis=1).astype(float)

    return out[config.MODEL_FEATURES]


def encode_target(series: pd.Series) -> pd.Series:
    """Map Churn Yes/No to 1/0. Churn = 1 is the positive class."""
    cleaned = series.astype(str).str.strip()
    unknown = set(cleaned.unique()) - {"Yes", "No"}
    if unknown:
        raise ValueError(f"Unexpected target labels: {sorted(unknown)}")
    return (cleaned == "Yes").astype(int)


def make_splits(df: pd.DataFrame):
    """Stratified 70 / 15 / 15 split.

    The dataset is one snapshot per customer with no date column, so a
    time-based split is not possible. Stratifying on the target keeps the
    churn rate the same in all three parts.
    """
    y = encode_target(df[config.TARGET_COL])
    holdout_fraction = config.TEST_FRACTION + config.VALIDATION_FRACTION

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_fraction,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
    # Split the 30% holdout in half: validation and test.
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=config.TEST_FRACTION / holdout_fraction,
        stratify=y.loc[holdout_df.index],
        random_state=config.RANDOM_SEED,
    )
    return train_df, val_df, test_df


def load_split(name: str):
    """Load one saved partition and return (ids, clean features, target)."""
    path = config.PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python -m src.data_pipeline' first."
        )
    df = pd.read_csv(path)
    return df[config.ID_COL], clean_features(df), encode_target(df[config.TARGET_COL])


def main() -> None:
    df = load_raw()
    print(f"Raw data: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Basic integrity checks before anything else uses the data.
    if df[config.ID_COL].duplicated().any():
        raise ValueError("Duplicate customer IDs found in the raw data.")
    missing_cols = [c for c in config.RAW_FEATURES + [config.TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw data is missing columns: {missing_cols}")

    blank_total = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
    churn_rate = float(encode_target(df[config.TARGET_COL]).mean())
    print(f"Blank TotalCharges values: {blank_total}")
    print(f"Overall churn rate: {churn_rate:.4f}")

    train_df, val_df, test_df = make_splits(df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "raw_rows": int(df.shape[0]),
        "raw_columns": int(df.shape[1]),
        "blank_total_charges": blank_total,
        "overall_churn_rate": round(churn_rate, 4),
        "random_seed": config.RANDOM_SEED,
        "splits": {},
    }
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
        rate = float(encode_target(part[config.TARGET_COL]).mean())
        summary["splits"][name] = {"rows": int(len(part)), "churn_rate": round(rate, 4)}
        print(f"  {name:5s}: {len(part):5,} rows, churn rate {rate:.4f}")

    # An "unseen" batch file for the batch scoring demo: test rows, no label.
    unseen = test_df.drop(columns=[config.TARGET_COL])
    unseen.to_csv(config.PROCESSED_DIR / "batch_input_unseen.csv", index=False)
    # A small 50-row copy that is committed to the repository, so the batch
    # script and the benchmark can be tried without downloading anything.
    unseen.head(50).to_csv(config.DATA_DIR / "sample_batch_input.csv", index=False)

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORT_DIR / "data_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("Saved splits to data/processed/ and summary to reports/data_summary.json")


if __name__ == "__main__":
    main()
