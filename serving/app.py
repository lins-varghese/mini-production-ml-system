"""FastAPI service for online churn scoring.

Start it from the repository root:

    python -m uvicorn serving.app:app --host 127.0.0.1 --port 8711 --no-access-log

Endpoints:
    GET  /health   is the service up, and is it running on the model or the fallback
    POST /predict  score one customer
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from serving.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src.inference import ChurnScorer, ScoringUnavailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("churn.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts, not on every request.
    app.state.scorer = ChurnScorer()
    yield


app = FastAPI(title="Customer churn scoring service", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_request(request: Request, call_next):
    """Log every request with its status code and latency."""
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    logger.info("%s %s -> %s in %.2f ms", request.method, request.url.path,
                response.status_code, latency_ms)
    return response


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    """Turn Pydantic errors into a short, readable 422 answer."""
    problems = []
    for error in exc.errors():
        # error["loc"] looks like ("body", "tenure"). Drop the "body" part.
        field = ".".join(str(part) for part in error["loc"][1:]) or "body"
        if error["type"] == "json_invalid":
            field = "body"   # the whole body could not be parsed as JSON
        problems.append({"field": field, "problem": error["msg"]})
    logger.warning("Rejected payload: %s", problems)
    return JSONResponse(status_code=422, content={"error": "invalid_request", "detail": problems})


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    scorer = request.app.state.scorer
    if scorer.model_ready:
        status = "ok"
    elif scorer.fallback_ready:
        status = "degraded"   # still answering, but with the fallback rule
    else:
        status = "down"
    return HealthResponse(
        status=status,
        model_loaded=scorer.model_ready,
        fallback_ready=scorer.fallback_ready,
        model_version=scorer.model_version,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request):
    scorer = request.app.state.scorer
    try:
        result = scorer.score_one(customer.model_dump())
    except ScoringUnavailable as exc:
        # Nothing can score right now. Say so clearly instead of guessing.
        logger.error("Scoring unavailable: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": "scoring_unavailable", "detail": str(exc)},
        )

    if result["served_by"] == "fallback" or result["warnings"]:
        logger.warning("customer=%s served_by=%s warnings=%s",
                       customer.customerID, result["served_by"], result["warnings"])

    return PredictionResponse(
        customerID=customer.customerID,
        model_version=scorer.model_version,
        **result,
    )
