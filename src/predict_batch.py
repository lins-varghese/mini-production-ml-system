"""Batch scoring from the command line.

Run from the repository root:

    python -m src.predict_batch --input data/sample_batch_input.csv --output reports/batch_predictions.csv

It reads a CSV of customers that the model has not seen, scores all rows
in one vectorised call, logs what happened, and writes one output row per
customer with the customer ID. It uses the same ChurnScorer as the API.

Exit codes: 0 = success, 1 = input problem, 3 = scoring unavailable.
(2 is left alone on purpose: Python's argument parser already exits with 2
when the command line itself is wrong, for example a missing --input.)
"""
import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config
from src.inference import ChurnScorer, ScoringUnavailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("churn.batch")

# Columns that must exist in the file. Their values may still be missing
# in single rows, in which case the pipeline imputes them.
REQUIRED_COLUMNS = ["tenure", "MonthlyCharges", "Contract"]

EXIT_OK = 0
EXIT_INPUT_PROBLEM = 1
EXIT_SCORING_UNAVAILABLE = 3   # not 2: argparse uses 2 for a wrong command line


def run(input_path: Path, output_path: Path) -> int:
    logger.info("Batch scoring started. Input: %s", input_path)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return EXIT_INPUT_PROBLEM
    try:
        raw = pd.read_csv(input_path)
    except Exception as exc:
        logger.error("Could not read the input file as CSV: %s", exc)
        return EXIT_INPUT_PROBLEM
    if raw.empty:
        logger.error("Input file has no rows.")
        return EXIT_INPUT_PROBLEM
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing_columns:
        logger.error("Input file is missing required columns: %s", missing_columns)
        return EXIT_INPUT_PROBLEM
    logger.info("Read %d rows and %d columns.", len(raw), raw.shape[1])

    # Keep a record ID for every output row.
    if config.ID_COL in raw.columns:
        ids = raw[config.ID_COL].astype(str)
    else:
        logger.warning("No %s column. Using row numbers as IDs.", config.ID_COL)
        ids = pd.Series([f"row_{i:07d}" for i in range(1, len(raw) + 1)], index=raw.index)

    scorer = ChurnScorer()
    started = time.perf_counter()
    try:
        scored = scorer.score(raw)
    except ScoringUnavailable as exc:
        logger.error("Scoring unavailable: %s", exc)
        return EXIT_SCORING_UNAVAILABLE
    elapsed = time.perf_counter() - started

    output = pd.DataFrame({config.ID_COL: ids.values})
    output["churn_score"] = scored["churn_score"].values
    output["churn_prediction"] = scored["churn_prediction"].values
    output["threshold"] = scored["threshold"].values
    output["served_by"] = scored["served_by"].values
    output["model_version"] = scorer.model_version
    output["n_missing_fields"] = scored["n_missing_fields"].values
    output["n_unknown_categories"] = scored["n_unknown_categories"].values
    output["n_clipped_values"] = scored["n_clipped_values"].values
    output["scored_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    rows_with_issues = int(
        ((output["n_missing_fields"] + output["n_unknown_categories"] + output["n_clipped_values"]) > 0).sum()
    )
    logger.info("Scored %d rows in %.3f s (%.0f records/sec).",
                len(output), elapsed, len(output) / max(elapsed, 1e-9))
    logger.info("Served by: %s", output["served_by"].value_counts().to_dict())
    logger.info("Predicted churners: %d of %d (%.1f%%).",
                int(output["churn_prediction"].sum()), len(output),
                100 * output["churn_prediction"].mean())
    logger.info("Rows with missing, unknown or clipped values: %d", rows_with_issues)
    logger.info("Output written to %s", output_path)
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a CSV of customers for churn risk.")
    parser.add_argument("--input", required=True, type=Path, help="CSV file with customer rows")
    parser.add_argument("--output", required=True, type=Path, help="Where to write the predictions CSV")
    args = parser.parse_args()
    return run(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
