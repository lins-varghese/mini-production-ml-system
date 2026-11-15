# Mini Production ML System: Customer Churn Prediction

**Author:** Lins Varghese (2025EM1200156)
**Course:** ML Model Engineering, BITS Pilani WILP, Graded Assignment 1 (Option A: Customer Churn)
**Repository:** https://github.com/lins-varghese/mini-production-ml-system

This repository takes one standard ML problem, telecom customer churn, and builds the whole
system around it: a leak-free data pipeline, a baseline and a tuned model, saved artifacts,
a FastAPI scoring service, a batch scoring script, a latency benchmark, robustness checks,
drift monitoring and a fallback for when the model fails.

I kept the code simple on purpose. Every file has one job and can be read top to bottom.

## Results at a glance

All numbers below were produced by the scripts in this repository. The raw outputs are in
`reports/`.

**Model quality on the held-out test set (1,057 customers, 26.6% churn)**

| Model | PR-AUC | ROC-AUC | Recall at precision 0.60 | Threshold | F1-macro | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Dummy prior | 0.266 | 0.500 | 0.000 | 0.50 | 0.423 | 0.000 | 0.000 |
| Segment rule (contract x tenure) | 0.473 | 0.778 | 0.000 | 0.38 | 0.666 | 0.507 | 0.516 |
| Logistic regression (class weighted) | 0.659 | 0.845 | 0.658 | 0.73 | 0.736 | 0.685 | 0.527 |
| **LightGBM (class weighted, tuned), deployed** | **0.671** | 0.840 | **0.662** | 0.66 | **0.737** | 0.617 | 0.609 |

LightGBM is only a little better than logistic regression. I report that honestly. It was
chosen by a fixed rule (higher PR-AUC on the validation set), not by looking at the test set.

**Serving performance (measured on my laptop, see `reports/benchmark_*.json`)**

| Measurement | Run A (quietest) | Run B (laptop busy with other heavy programs) |
|---|---|---|
| API latency, 500 requests, client side | p50 20.7 ms, p95 25.2 ms, p99 29.4 ms | p50 73.1 ms, p95 140.1 ms, p99 179.1 ms |
| API errors during the benchmark | 0 of 500 | 0 of 500 |
| Batch throughput, 100,000 rows | 122,885 records per second | 36,184 records per second |
| Time inside the model itself for one request | about 0.4 ms (most time is pandas row handling) | not measured |

Hardware and settings: Intel Core Ultra 5 125H laptop (18 logical CPUs, 16 GB RAM),
Windows 11, Python 3.12.10, one uvicorn worker, one client sending requests one after
another on localhost. I ran the benchmark more than once and the result depended a lot on
what else the laptop was doing, so I show the quietest and the busiest run. Run A is in
`reports/benchmark_api.json` and `reports/benchmark_batch.json`. Run B is in the two files
ending in `_busy_machine.json`.

**Drift monitor**

| Check | Result |
|---|---|
| Training data vs test data | STABLE (largest PSI 0.018) |
| Training data vs deliberately shifted data | ALERT (tenure PSI 2.20, MonthlyCharges PSI 1.06, PaymentMethod PSI 0.99) |

**Tests:** 52 pytest tests pass.

**Reproduction:** I copied only the source files to a new folder, built a new virtual
environment from `requirements.txt` and ran the steps below in order. Every step worked and
retraining gave result files identical to the committed ones.

## Repository layout

```
mini-production-ml-system/
  README.md
  requirements.txt          pinned dependencies
  Dockerfile                optional, untested (see the Docker section)
  pytest.ini
  data/
    fetch_data.py           downloads the dataset and checks its SHA-256
    sample_batch_input.csv  50 unlabeled customers for quick demos
  artifacts/
    model_lightgbm.joblib   deployed pipeline (preprocessing + LightGBM)
    model_logreg.joblib     baseline pipeline (preprocessing + logistic regression)
    model_metadata.json     thresholds, fallback table, training ranges, versions
  src/
    config.py               paths, column lists, constants
    data_pipeline.py        ingestion, cleaning, leak-free split
    train.py                baselines, LightGBM tuning, threshold choice, export
    evaluate.py             metric functions and the one-time test evaluation
    inference.py            ChurnScorer, shared by the API and the batch script
    fallback.py             segment rule: baseline and serving fallback
    predict_batch.py        batch scoring command line script
    drift.py                PSI and KS drift monitoring
  serving/
    app.py                  FastAPI service with /health and /predict
    schemas.py              Pydantic request and response schemas
    benchmark.py            latency and throughput benchmark
    robustness_check.py     edge-case and failure-injection checks
  tests/                    pytest unit tests
  examples/                 sample JSON payloads for curl
  reports/                  generated metrics, figures and benchmark results
```

## Setup

You need Python 3.12.

Windows (PowerShell):

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux or macOS:

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run every command below from the repository root.

## Run it, step by step

The trained artifacts and my reports are committed, so steps 5 to 7 and step 11 work right
after setup. Steps 1 to 4 rebuild everything from the raw data. Two of the tests check the
split files and are skipped until steps 1 and 2 have been run.

**1. Fetch the data** (downloads about 1 MB and verifies the checksum)

```
python data/fetch_data.py
```

**2. Clean and split** (stratified 70 / 15 / 15)

```
python -m src.data_pipeline
```

**3. Train** (baselines, LightGBM tuning with 5-fold CV, threshold choice, export; under 1 minute)

```
python -m src.train
```

**4. Evaluate on the test set** (the only script that uses the test labels)

```
python -m src.evaluate
```

**5. Start the API**

```
python -m uvicorn serving.app:app --host 127.0.0.1 --port 8711 --no-access-log
```

**6. Call the API** (in a second terminal; on Windows PowerShell type `curl.exe` instead of `curl`)

```
curl http://127.0.0.1:8711/health

curl -X POST http://127.0.0.1:8711/predict -H "Content-Type: application/json" -d "@examples/customer.json"

curl -X POST http://127.0.0.1:8711/predict -H "Content-Type: application/json" -d "@examples/bad_customer.json"
```

A good request returns:

```json
{
  "customerID": "7590-VHVEG",
  "churn_score": 0.86651,
  "churn_prediction": 1,
  "threshold": 0.66,
  "served_by": "model",
  "model_version": "lightgbm-1.0.0",
  "warnings": []
}
```

The bad request returns HTTP 422 with one line per problem:

```json
{
  "error": "invalid_request",
  "detail": [
    {"field": "tenure", "problem": "Input should be a valid integer, unable to parse string as an integer"},
    {"field": "MonthlyCharges", "problem": "Input should be greater than or equal to 0"},
    {"field": "Contract", "problem": "Field required"},
    {"field": "favourite_colour", "problem": "Extra inputs are not permitted"}
  ]
}
```

Interactive API documentation is at http://127.0.0.1:8711/docs while the server runs.
Stop the server with Ctrl+C.

**7. Batch scoring** (reads a CSV, writes predictions with customer IDs, logs what happened)

```
python -m src.predict_batch --input data/sample_batch_input.csv --output reports/batch_predictions_sample.csv
```

**8. Benchmarks**

```
python -m serving.benchmark --mode api --spawn-server
python -m serving.benchmark --mode batch
python -m serving.benchmark --mode stages
```

`--spawn-server` starts the API, sends 20 warm-up requests and 500 timed requests, then
stops the API. Leave the flag out to benchmark a server you started yourself in step 5.

**9. Robustness checks** (missing values, unseen categories, outliers, malformed input, model failure)

```
python -m serving.robustness_check
```

**10. Drift monitoring** (needs steps 1 and 2)

```
python -m src.drift
```

**11. Tests**

```
python -m pytest -q
```

## Design decisions in short

**Serving paradigm.** Online scoring through the API is the main path, for example when a
support agent opens a customer account. A nightly batch run with `predict_batch.py` is the
second path. Both use the same `ChurnScorer` class, so they cannot disagree.

**No leakage.**
- The split is made before anything is fitted. The data has no date column, so a time-based
  split is not possible. If dated snapshots existed, I would train on older months and test
  on the newest month. Here I use a stratified random split so all three parts keep the same
  churn rate.
- `clean_features` uses only row-wise rules and learns nothing from the data.
- Imputers, the scaler and the one-hot encoder live inside the scikit-learn pipeline and are
  fitted on the training partition only. A unit test checks the stored medians against the
  training medians.
- Hyperparameters are tuned with cross-validation inside the training partition. The
  threshold is chosen on the validation partition. The test labels are used once, by
  `src/evaluate.py`. The drift check and the batch demo only use a copy of the test rows
  without the label.

**Class imbalance.** About 26.5% of customers churn. I use class weights in the loss and then
tune the decision threshold. I do not oversample. Because of the class weights the output is
a risk score for ranking customers, not a calibrated probability.

**Threshold.** The default 0.50 gives high recall but many false alarms. I pick the threshold
that gives the best F1-macro on the validation set, which is 0.66 for LightGBM.

**Fallback ladder.**
1. Normal: the model answers, `served_by` is `model`.
2. Model missing or crashing: a segment rule (training churn rate by contract type and tenure
   band) answers, `served_by` is `fallback`, and `/health` reports `degraded`.
3. Metadata missing too: `/predict` returns HTTP 503 with a clear message and `/health`
   reports `down`.

**Input handling.**

| Input problem | What the service does |
|---|---|
| Missing optional field or null | Scores the row, imputes the value, adds a warning |
| Category never seen in training | Scores the row, encoder ignores the value, adds a warning |
| Number outside the training range | Clips it to the training range, scores, adds a warning |
| Wrong type, negative number, NaN, missing required field, unknown field, broken JSON | Rejects with HTTP 422 and a readable reason |

**Monitoring thresholds.**

| Signal | Warning | Alert or action |
|---|---|---|
| PSI per feature and for the model score | 0.10 | 0.25: alert. Two weekly alerts in a row: retrain |
| KS test on numeric features and the score | - | statistic at least 0.10 with p below 0.01 |
| PR-AUC on newly labeled customers | drop of 0.03 | drop of 0.05 from validation: retrain |
| p95 latency | 60 ms | 100 ms for 10 minutes: alert |
| HTTP 5xx rate | 0.5% | 1% over 5 minutes: alert |
| Fallback share of answers | any | 1% over 15 minutes: alert |
| HTTP 422 share and unknown-category warnings | 2% | 5%: check the upstream system |

## Docker (optional, untested)

A `Dockerfile` is included for the API. I could not build it because Docker was not running
on my machine, so treat it as untested. The intended commands are:

```
docker build -t churn-api .
docker run -p 8711:8711 churn-api
```

## Data source

IBM Telco Customer Churn sample dataset, 7,043 customers and 21 columns, from the public
repository `IBM/telco-customer-churn-on-icp4d` (Apache-2.0 license). The download is pinned
to commit `1fd6fd7` and verified with SHA-256
`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.

## Limitations

- The dataset is a single snapshot with no dates, so I cannot test how the model ages over
  time. The drift alert test uses a shift that I created myself.
- Latency was measured with one client on a laptop. It is not a load test.
- The risk score is not calibrated. If the business needs true probabilities, a calibration
  step fitted on the validation set would be the next thing to add.
- Most of the request time is spent building and checking a one-row pandas table, not in the
  model. A plain dictionary path for single requests would be faster, but I kept one shared
  code path for training, batch and API to avoid training-serving skew.
