"""Measure real latency of the API and real throughput of batch scoring.

Run from the repository root:

    python -m serving.benchmark --mode api --spawn-server     # starts its own server, then stops it
    python -m serving.benchmark --mode api                    # uses a server you already started
    python -m serving.benchmark --mode batch                  # in-process batch throughput
    python -m serving.benchmark --mode stages                 # where the time goes for one request

API mode sends requests one after another over one keep-alive connection
and times each one on the client side, so the number includes the HTTP
round trip, validation, preprocessing and the model. The first few
requests are a warm-up and are not counted.

Results are saved as JSON files in reports/ (benchmark_api.json,
benchmark_batch.json, benchmark_stages.json).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import httpx2 as httpx   # HTTP client, same interface as the older httpx
import numpy as np
import pandas as pd

from src import config

DEFAULT_URL = "http://127.0.0.1:8711"
WARMUP_REQUESTS = 20


def load_payloads() -> list:
    """Turn the sample customer file into JSON payloads for /predict."""
    rows = pd.read_csv(config.DATA_DIR / "sample_batch_input.csv")
    payloads = []
    for record in rows.to_dict(orient="records"):
        payload = {}
        for key, value in record.items():
            if isinstance(value, str):
                value = value.strip() or None      # blank text becomes missing
            elif pd.isna(value):
                value = None
            payload[key] = value
        # TotalCharges is text in the raw file. Send it as a number or null.
        if payload.get("TotalCharges") is not None:
            payload["TotalCharges"] = float(payload["TotalCharges"])
        payloads.append(payload)
    return payloads


def start_server(url: str):
    """Start uvicorn as a child process and wait until /health answers."""
    port = url.rsplit(":", 1)[1]
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(config.REPORT_DIR / "benchmark_server.log", "w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.app:app",
         "--host", "127.0.0.1", "--port", port, "--no-access-log"],
        cwd=config.ROOT, stdout=log_file, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return process, log_file
        except httpx.HTTPError:
            time.sleep(0.5)
    process.terminate()
    log_file.close()
    raise RuntimeError("The server did not start within 60 seconds.")


def percentiles(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "p99": round(float(np.percentile(values, 99)), 3),
        "mean": round(float(values.mean()), 3),
        "min": round(float(values.min()), 3),
        "max": round(float(values.max()), 3),
    }


def environment() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    }


def benchmark_api(url: str, n_requests: int) -> dict:
    payloads = load_payloads()
    client_ms, server_ms, errors = [], [], 0

    with httpx.Client(base_url=url, timeout=30) as client:
        health = client.get("/health").json()
        for i in range(WARMUP_REQUESTS):
            client.post("/predict", json=payloads[i % len(payloads)])

        run_started = time.perf_counter()
        for i in range(n_requests):
            payload = payloads[i % len(payloads)]
            started = time.perf_counter()
            response = client.post("/predict", json=payload)
            client_ms.append((time.perf_counter() - started) * 1000)
            if response.status_code != 200:
                errors += 1
            # The server reports its own processing time in a header.
            if "X-Process-Time-Ms" in response.headers:
                server_ms.append(float(response.headers["X-Process-Time-Ms"]))
        total_seconds = time.perf_counter() - run_started

    result = {
        "url": url,
        "requests": n_requests,
        "warmup_requests": WARMUP_REQUESTS,
        "errors": errors,
        "client_latency_ms": percentiles(client_ms),
        "server_latency_ms": percentiles(server_ms) if server_ms else None,
        "sequential_requests_per_second": round(n_requests / total_seconds, 1),
        "health": health,
        "environment": environment(),
        "setup": "one uvicorn worker, one client, sequential requests, keep-alive connection, localhost",
    }
    return result


def benchmark_batch(n_rows: int, repeats: int = 3) -> dict:
    """Time vectorised scoring of n_rows customers, done in-process."""
    from src.inference import ChurnScorer

    source = config.PROCESSED_DIR / "batch_input_unseen.csv"
    if not source.exists():
        source = config.DATA_DIR / "sample_batch_input.csv"
    base = pd.read_csv(source)
    copies = int(np.ceil(n_rows / len(base)))
    frame = pd.concat([base] * copies, ignore_index=True).iloc[:n_rows]

    scorer = ChurnScorer()
    scorer.score(frame.head(1000))   # warm-up
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        scored = scorer.score(frame)
        seconds.append(time.perf_counter() - started)

    median_seconds = float(np.median(seconds))
    return {
        "rows": int(len(frame)),
        "repeats": repeats,
        "seconds_per_run": [round(s, 3) for s in seconds],
        "median_seconds": round(median_seconds, 3),
        "records_per_second": round(len(frame) / median_seconds),
        "served_by": scored["served_by"].value_counts().to_dict(),
        "model_version": scorer.model_version,
        "environment": environment(),
        "setup": f"in-process ChurnScorer.score on {source.name} repeated to {len(frame):,} rows; "
                 "includes cleaning, preprocessing and model",
    }


def profile_stages(repeats: int = 300) -> dict:
    """Time each step of scoring one customer, to see where latency goes."""
    from src.data_pipeline import clean_features
    from src.inference import ChurnScorer

    scorer = ChurnScorer()
    record = load_payloads()[0]
    features = clean_features(pd.DataFrame([record]))
    preprocess = scorer.pipeline.named_steps["preprocess"]
    model = scorer.pipeline.named_steps["model"]
    matrix = preprocess.transform(features)

    stages = {
        "build one-row table and clean it": lambda: clean_features(pd.DataFrame([record])),
        "data checks and clipping": lambda: (scorer._count_issues(features),
                                             scorer._clip_to_training_range(features)),
        "preprocessing (impute, scale, one-hot)": lambda: preprocess.transform(features),
        "model predict_proba": lambda: model.predict_proba(matrix),
        "whole score_one call": lambda: scorer.score_one(record),
    }
    result = {"repeats": repeats, "model_version": scorer.model_version, "stages": {}}
    for name, step in stages.items():
        step()   # warm-up
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            step()
            timings.append((time.perf_counter() - started) * 1000)
        result["stages"][name] = {
            "median_ms": round(float(np.median(timings)), 3),
            "p95_ms": round(float(np.percentile(timings, 95)), 3),
        }
    result["environment"] = environment()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency and throughput benchmark.")
    parser.add_argument("--mode", choices=["api", "batch", "stages"], default="api")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the running API")
    parser.add_argument("--requests", type=int, default=500, help="Number of timed API requests")
    parser.add_argument("--rows", type=int, default=100000, help="Rows to score in batch mode")
    parser.add_argument("--spawn-server", action="store_true",
                        help="Start the API for the benchmark and stop it afterwards")
    args = parser.parse_args()
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "batch":
        result = benchmark_batch(args.rows)
        with open(config.REPORT_DIR / "benchmark_batch.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Batch scoring: {result['rows']:,} rows in {result['median_seconds']:.3f} s "
              f"(median of {result['repeats']} runs) = {result['records_per_second']:,} records/sec")
        print("Saved reports/benchmark_batch.json")
        return

    if args.mode == "stages":
        result = profile_stages()
        with open(config.REPORT_DIR / "benchmark_stages.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Time per step for one customer (median of {result['repeats']} runs)")
        for name, timing in result["stages"].items():
            print(f"  {name:42s} {timing['median_ms']:8.3f} ms   (p95 {timing['p95_ms']:.3f} ms)")
        print("Saved reports/benchmark_stages.json")
        return

    process, log_file = (None, None)
    if args.spawn_server:
        process, log_file = start_server(args.url)
    try:
        result = benchmark_api(args.url, args.requests)
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=15)
            log_file.close()

    with open(config.REPORT_DIR / "benchmark_api.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lat = result["client_latency_ms"]
    print(f"API benchmark: {result['requests']} requests, {result['errors']} errors")
    print(f"  client-side latency ms: p50 {lat['p50']}  p95 {lat['p95']}  p99 {lat['p99']}  "
          f"mean {lat['mean']}  max {lat['max']}")
    if result["server_latency_ms"]:
        srv = result["server_latency_ms"]
        print(f"  server-side latency ms: p50 {srv['p50']}  p95 {srv['p95']}  p99 {srv['p99']}")
    print(f"  sequential throughput: {result['sequential_requests_per_second']} requests/sec")
    print("Saved reports/benchmark_api.json")


if __name__ == "__main__":
    main()
