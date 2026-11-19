"""Send awkward inputs to the API and record what actually happens.

Run from the repository root (after training):

    python -m serving.robustness_check

It uses FastAPI's TestClient, which runs the real app in-process, so no
server has to be started. Every case is sent to /predict and the observed
status code and response are saved to reports/robustness_results.json.

The cases cover: missing values, categories never seen in training,
extreme numbers, malformed payloads, and a broken model (fallback).
"""
import json

from fastapi.testclient import TestClient

from serving.app import app
from serving.schemas import EXAMPLE_CUSTOMER
from src import config


class BrokenModel:
    """Stands in for a model that crashes at scoring time."""

    def predict_proba(self, features):
        raise RuntimeError("simulated model failure")


def customer(**changes) -> dict:
    """The example customer with some fields changed. None removes a field."""
    payload = dict(EXAMPLE_CUSTOMER)
    for key, value in changes.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


def build_cases() -> list:
    minimal = {"tenure": 5, "MonthlyCharges": 70.0, "Contract": "Month-to-month"}
    with_nulls = dict(EXAMPLE_CUSTOMER, TotalCharges=None, PaymentMethod=None, InternetService=None)
    return [
        ("valid payload", "valid", {"json": customer()}),
        ("only the 3 required fields", "missing values", {"json": minimal}),
        ("optional fields sent as null", "missing values", {"json": with_nulls}),
        ("unseen PaymentMethod 'Crypto wallet'", "unseen category",
         {"json": customer(PaymentMethod="Crypto wallet")}),
        ("unseen Contract 'Five year'", "unseen category", {"json": customer(Contract="Five year")}),
        ("MonthlyCharges 9999, tenure 600 and TotalCharges 5000000", "extreme outlier",
         {"json": customer(MonthlyCharges=9999, tenure=600, TotalCharges=5000000)}),
        ("tenure -5", "malformed", {"json": customer(tenure=-5)}),
        ("tenure 'twelve'", "malformed", {"json": customer(tenure="twelve")}),
        ("MonthlyCharges NaN", "malformed",
         {"content": json.dumps(customer(MonthlyCharges=float("nan"))),
          "headers": {"Content-Type": "application/json"}}),
        ("required field Contract left out", "malformed", {"json": customer(Contract=None)}),
        ("unknown extra field 'favourite_colour'", "malformed",
         {"json": dict(customer(), favourite_colour="blue")}),
        ("SeniorCitizen 7", "malformed", {"json": customer(SeniorCitizen=7)}),
        ("empty JSON object", "malformed", {"json": {}}),
        ("body is not JSON", "malformed",
         {"content": "tenure=5;charges=70", "headers": {"Content-Type": "application/json"}}),
    ]


def summarise(response) -> dict:
    """Keep the parts of a response that matter for the report."""
    body = response.json()
    summary = {"status_code": response.status_code}
    if response.status_code == 200:
        summary.update({
            "served_by": body["served_by"],
            "churn_score": round(body["churn_score"], 4),
            "churn_prediction": body["churn_prediction"],
            "warnings": body["warnings"],
        })
    else:
        summary["error"] = body.get("error")
        summary["detail"] = body.get("detail")
    return summary


def main() -> None:
    results = []
    with TestClient(app) as client:
        for name, category, request in build_cases():
            response = client.post("/predict", **request)
            results.append({"case": name, "category": category, **summarise(response)})

        # Failure injection 1: the model crashes, the fallback must answer.
        # /health is asked after the request, because it reports on the most
        # recent scoring attempt.
        scorer = client.app.state.scorer
        real_pipeline = scorer.pipeline
        scorer.pipeline = BrokenModel()
        response = client.post("/predict", json=customer())
        health = client.get("/health").json()
        results.append({"case": "model crashes while scoring", "category": "model failure",
                        "health_status": health["status"], **summarise(response)})

        # Failure injection 2: the model artifact is missing entirely.
        scorer.pipeline = None
        health = client.get("/health").json()
        response = client.post("/predict", json=customer())
        results.append({"case": "model artifact not loaded", "category": "model failure",
                        "health_status": health["status"], **summarise(response)})

        # Failure injection 3: metadata gone too, so even the fallback is blind.
        real_metadata = scorer.metadata
        scorer.metadata = None
        health = client.get("/health").json()
        response = client.post("/predict", json=customer())
        results.append({"case": "model and metadata both unavailable", "category": "model failure",
                        "health_status": health["status"], **summarise(response)})

        # Put the real objects back.
        scorer.pipeline, scorer.metadata = real_pipeline, real_metadata

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORT_DIR / "robustness_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"{'case':58s} {'HTTP':>4s}  outcome")
    for row in results:
        if row["status_code"] == 200:
            outcome = f"scored by {row['served_by']}, score {row['churn_score']:.3f}, {len(row['warnings'])} warning(s)"
        else:
            outcome = f"{row['error']}"
        print(f"{row['case']:58s} {row['status_code']:4d}  {outcome}")
    print("\nSaved reports/robustness_results.json")


if __name__ == "__main__":
    main()
