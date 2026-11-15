"""Shared test fixtures."""
import pytest
from fastapi.testclient import TestClient

from serving.app import app
from serving.schemas import EXAMPLE_CUSTOMER
from src import config
from src.inference import ChurnScorer


@pytest.fixture
def valid_payload() -> dict:
    """A fresh copy of the example customer for each test."""
    return dict(EXAMPLE_CUSTOMER)


@pytest.fixture(scope="session")
def scorer() -> ChurnScorer:
    """The real trained artifacts. Tests that need them are skipped if absent."""
    if not (config.ARTIFACT_DIR / config.METADATA_FILE).exists():
        pytest.skip("No artifacts found. Run 'python -m src.train' first.")
    return ChurnScorer()


@pytest.fixture
def client(scorer):
    """The real FastAPI app, run in-process."""
    with TestClient(app) as test_client:
        yield test_client


class BrokenModel:
    """A model that always crashes, to test the fallback."""

    def predict_proba(self, features):
        raise RuntimeError("simulated model failure")


@pytest.fixture
def broken_model() -> BrokenModel:
    return BrokenModel()
