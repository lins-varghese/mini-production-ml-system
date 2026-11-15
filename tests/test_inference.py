"""Inference integrity: cleaning rules, consistent scores, no leakage."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_pipeline import clean_features
from src.inference import ChurnScorer


# ---------------------------------------------------------- cleaning rules
def test_blank_total_charges_for_new_customer_becomes_zero():
    raw = pd.DataFrame([{"tenure": 0, "MonthlyCharges": 50.0, "TotalCharges": " ",
                         "Contract": "Two year"}])
    assert clean_features(raw)["TotalCharges"].iloc[0] == 0.0


def test_blank_total_charges_for_old_customer_stays_missing():
    raw = pd.DataFrame([{"tenure": 10, "MonthlyCharges": 50.0, "TotalCharges": " ",
                         "Contract": "Two year"}])
    # tenure is not 0, so the amount is truly unknown and left for the imputer.
    assert np.isnan(clean_features(raw)["TotalCharges"].iloc[0])


def test_cleaning_adds_missing_columns_and_counts_addons():
    raw = pd.DataFrame([{"tenure": 3, "MonthlyCharges": 80.0, "Contract": "Month-to-month",
                         "OnlineSecurity": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
                         "SeniorCitizen": 1}])
    features = clean_features(raw)
    assert list(features.columns) == config.MODEL_FEATURES
    assert features["num_addon_services"].iloc[0] == 2
    assert features["SeniorCitizen"].iloc[0] == "Yes"
    assert pd.isna(features["PaymentMethod"].iloc[0])


# ------------------------------------------------------------ real model
def test_score_is_between_zero_and_one(scorer, valid_payload):
    result = scorer.score_one(valid_payload)
    assert 0.0 <= result["churn_score"] <= 1.0
    assert result["served_by"] == "model"


def test_same_input_gives_same_score(scorer, valid_payload):
    first = scorer.score_one(valid_payload)
    second = scorer.score_one(valid_payload)
    assert first["churn_score"] == second["churn_score"]


def test_prediction_follows_threshold(scorer, valid_payload):
    result = scorer.score_one(valid_payload)
    expected = int(result["churn_score"] >= result["threshold"])
    assert result["churn_prediction"] == expected


def test_single_and_batch_paths_agree(scorer, valid_payload):
    single = scorer.score_one(valid_payload)
    batch = scorer.score(pd.DataFrame([valid_payload, valid_payload]))
    assert batch["churn_score"].iloc[0] == pytest.approx(single["churn_score"])
    assert batch["churn_score"].iloc[1] == pytest.approx(single["churn_score"])


def test_risky_customer_scores_higher_than_loyal_customer(scorer, valid_payload):
    # New, month-to-month, electronic check: the classic churn profile.
    risky = dict(valid_payload, tenure=1, Contract="Month-to-month")
    # Six years, two-year contract: the classic loyal profile.
    loyal = dict(valid_payload, tenure=72, Contract="Two year", TotalCharges=2149.2)
    assert scorer.score_one(risky)["churn_score"] > scorer.score_one(loyal)["churn_score"]


def test_missing_optional_fields_are_imputed_with_warning(scorer):
    result = scorer.score_one({"tenure": 5, "MonthlyCharges": 70.0, "Contract": "Month-to-month"})
    assert 0.0 <= result["churn_score"] <= 1.0
    assert any("imputed" in warning for warning in result["warnings"])


def test_unknown_category_is_scored_with_warning(scorer, valid_payload):
    valid_payload["PaymentMethod"] = "Crypto wallet"
    result = scorer.score_one(valid_payload)
    assert result["served_by"] == "model"
    assert any("unknown category" in warning for warning in result["warnings"])


def test_extreme_values_are_clipped_to_training_range(scorer, valid_payload):
    limit = scorer.metadata["numeric_ranges"]["MonthlyCharges"]["max"]
    extreme = scorer.score_one(dict(valid_payload, MonthlyCharges=9999.0))
    at_limit = scorer.score_one(dict(valid_payload, MonthlyCharges=limit))
    # After clipping, an absurd value scores exactly like the training maximum.
    assert extreme["churn_score"] == pytest.approx(at_limit["churn_score"])
    assert any("clipped" in warning for warning in extreme["warnings"])


def test_production_model_can_be_switched_in_metadata(scorer, tmp_path, valid_payload):
    # The metadata file decides which saved pipeline is served. Changing one
    # field must be enough to deploy the logistic regression instead.
    copy_dir = tmp_path / "artifacts"
    shutil.copytree(config.ARTIFACT_DIR, copy_dir)
    metadata_path = copy_dir / config.METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["production_model"] = "logreg"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    switched = ChurnScorer(artifact_dir=copy_dir)
    result = switched.score_one(valid_payload)
    assert switched.model_version.startswith("logreg")
    assert result["served_by"] == "model"
    assert result["threshold"] == metadata["candidates"]["logreg"]["threshold"]


# ------------------------------------------------------------------- API
def test_health_endpoint(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_api_matches_direct_scoring(client, scorer, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    direct = scorer.score_one(valid_payload)
    assert body["churn_score"] == pytest.approx(direct["churn_score"])
    assert body["churn_prediction"] == direct["churn_prediction"]
    assert body["customerID"] == valid_payload["customerID"]
    assert body["model_version"] == scorer.model_version


def test_api_rejects_malformed_payload_with_422(client, valid_payload):
    valid_payload["tenure"] = "twelve"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_request"
    assert body["detail"][0]["field"] == "tenure"


def test_api_rejects_body_that_is_not_json(client):
    response = client.post("/predict", content="not json",
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422


# ----------------------------------------------------- leakage safeguards
def _processed(name):
    path = config.PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip("Processed data not found. Run 'python -m src.data_pipeline' first.")
    return pd.read_csv(path)


def test_splits_share_no_customers():
    train, val, test = (set(_processed(n)[config.ID_COL]) for n in ["train", "val", "test"])
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_imputer_was_fitted_on_training_rows_only(scorer):
    # The medians stored inside the pipeline must equal the medians of the
    # training partition. If validation or test rows had leaked into fit(),
    # these numbers would differ.
    train_features = clean_features(_processed("train"))
    imputer = scorer.pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    expected = train_features[config.MODEL_NUMERIC].median().to_numpy()
    assert np.allclose(imputer.statistics_, expected)
