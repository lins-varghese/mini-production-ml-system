"""The service must degrade gracefully when the model is unavailable."""
import pandas as pd
import pytest

from src.fallback import fit_segment_table, segment_scores, tenure_band
from src.inference import ChurnScorer, ScoringUnavailable


# -------------------------------------------------------- the rule itself
@pytest.mark.parametrize(
    "tenure, expected",
    [(0, "0-12"), (12, "0-12"), (13, "13-24"), (48, "25-48"), (49, "49+"), (None, "unknown"),
     # A tenure with a fraction (possible in a batch file) must not fall between two bands.
     (12.5, "13-24"), (48.5, "49+"),
     # A negative tenure is not a real value and stays unknown.
     (-1, "unknown")],
)
def test_tenure_band(tenure, expected):
    assert tenure_band(tenure) == expected


def test_segment_table_learns_group_rates():
    # 40 new monthly customers, 30 of them churn. 40 loyal two-year customers, none churn.
    features = pd.DataFrame({
        "tenure": [3] * 40 + [60] * 40,
        "Contract": ["Month-to-month"] * 40 + ["Two year"] * 40,
    })
    target = pd.Series([1] * 30 + [0] * 10 + [0] * 40)
    table = fit_segment_table(features, target)

    assert table["segments"]["Month-to-month|0-12"]["rate"] == 0.75
    assert table["segments"]["Two year|49+"]["rate"] == 0.0
    assert table["global_rate"] == 0.375

    scores = segment_scores(features.iloc[[0, 79]], table)
    assert list(scores) == [0.75, 0.0]


def test_unknown_segment_gets_the_overall_rate():
    features = pd.DataFrame({"tenure": [3] * 40, "Contract": ["Month-to-month"] * 40})
    table = fit_segment_table(features, pd.Series([1] * 10 + [0] * 30))
    new_customer = pd.DataFrame({"tenure": [5], "Contract": ["Five year"]})
    assert segment_scores(new_customer, table)[0] == table["global_rate"]


# ------------------------------------------------------- scorer fallback
def test_scorer_uses_fallback_when_model_crashes(scorer, valid_payload, broken_model):
    real_pipeline = scorer.pipeline
    scorer.pipeline = broken_model
    try:
        result = scorer.score_one(valid_payload)
    finally:
        scorer.pipeline = real_pipeline
    assert result["served_by"] == "fallback"
    assert 0.0 <= result["churn_score"] <= 1.0
    assert result["threshold"] == scorer.metadata["fallback"]["threshold"]


def test_scorer_without_any_artifacts_raises_clear_error(tmp_path, valid_payload):
    empty_scorer = ChurnScorer(artifact_dir=tmp_path)   # folder with no files
    assert not empty_scorer.model_ready
    assert not empty_scorer.fallback_ready
    with pytest.raises(ScoringUnavailable):
        empty_scorer.score_one(valid_payload)


# ---------------------------------------------------------- API fallback
def test_api_answers_from_fallback_and_reports_degraded(client, valid_payload):
    api_scorer = client.app.state.scorer
    real_pipeline = api_scorer.pipeline
    api_scorer.pipeline = None   # as if the model file failed to load
    try:
        health = client.get("/health").json()
        response = client.post("/predict", json=valid_payload)
    finally:
        api_scorer.pipeline = real_pipeline

    assert health["status"] == "degraded"
    assert response.status_code == 200
    assert response.json()["served_by"] == "fallback"


def test_health_is_degraded_after_a_model_crash_and_recovers(client, valid_payload, broken_model):
    # The model file is loaded, but the model raises while scoring.
    api_scorer = client.app.state.scorer
    real_pipeline = api_scorer.pipeline
    api_scorer.pipeline = broken_model
    try:
        crashed = client.post("/predict", json=valid_payload)
        health_after_crash = client.get("/health").json()
    finally:
        api_scorer.pipeline = real_pipeline

    assert crashed.status_code == 200
    assert crashed.json()["served_by"] == "fallback"
    assert health_after_crash["status"] == "degraded"
    assert health_after_crash["model_loaded"] is True   # loaded, but not healthy

    # The next request that the model answers clears the degraded state.
    recovered = client.post("/predict", json=valid_payload)
    assert recovered.json()["served_by"] == "model"
    assert client.get("/health").json()["status"] == "ok"


def test_api_returns_503_when_nothing_can_score(client, valid_payload):
    api_scorer = client.app.state.scorer
    real_pipeline, real_metadata = api_scorer.pipeline, api_scorer.metadata
    api_scorer.pipeline, api_scorer.metadata = None, None
    try:
        health = client.get("/health").json()
        response = client.post("/predict", json=valid_payload)
    finally:
        api_scorer.pipeline, api_scorer.metadata = real_pipeline, real_metadata

    assert health["status"] == "down"
    assert response.status_code == 503
    assert response.json()["error"] == "scoring_unavailable"


# -------------------------------------------------------- batch exit codes
def test_batch_script_exit_codes(tmp_path, monkeypatch):
    from src import predict_batch

    # 1 = input problem: the input file does not exist.
    code = predict_batch.run(tmp_path / "no_such_file.csv", tmp_path / "out.csv")
    assert code == predict_batch.EXIT_INPUT_PROBLEM == 1

    # 3 = scoring unavailable: a readable input file, but no artifacts at all.
    input_file = tmp_path / "customers.csv"
    pd.DataFrame({"tenure": [5], "MonthlyCharges": [70.0], "Contract": ["One year"]}).to_csv(input_file, index=False)
    empty_dir = tmp_path / "no_artifacts"
    empty_dir.mkdir()
    monkeypatch.setattr(predict_batch, "ChurnScorer", lambda: ChurnScorer(artifact_dir=empty_dir))
    code = predict_batch.run(input_file, tmp_path / "out.csv")
    assert code == predict_batch.EXIT_SCORING_UNAVAILABLE == 3
    assert not (tmp_path / "out.csv").exists()   # nothing is written when nothing can score
