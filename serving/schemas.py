"""Request and response schemas for the API (Pydantic).

The request schema is the first line of defence. A payload that breaks
these rules is rejected with HTTP 422 before it gets near the model.

What is strict and what is lenient, and why:

- tenure, MonthlyCharges and Contract are required. They are the strongest
  churn drivers and the fallback rule needs two of them.
- Numbers must be real numbers inside a sanity range (no negatives, no NaN,
  no absurd values). Wrong types are rejected.
- Unknown extra fields are rejected, so a typo in a field name is caught
  instead of being silently ignored.
- Other fields are optional. If one is missing, the model pipeline imputes
  it and the response carries a warning.
- Category text is not restricted to a fixed list. A new plan or payment
  method must not take the service down. The encoder ignores the unknown
  value and the response carries a warning so monitoring can pick it up.
"""
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Reusable type: text that may be left out, but may not be an empty string.
OptionalText = Annotated[Optional[str], Field(min_length=1, max_length=64)]

EXAMPLE_CUSTOMER = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class CustomerFeatures(BaseModel):
    """One customer, in the same shape as a row of the raw dataset."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_CUSTOMER})

    customerID: Optional[str] = Field(default=None, max_length=64)

    # Required fields.
    tenure: int = Field(ge=0, le=1200, description="Months with the company")
    MonthlyCharges: float = Field(ge=0, le=100000, allow_inf_nan=False)
    Contract: str = Field(min_length=1, max_length=64)

    # Optional numeric fields.
    TotalCharges: Optional[float] = Field(default=None, ge=0, le=100000000, allow_inf_nan=False)
    SeniorCitizen: Optional[Literal[0, 1]] = None

    # Optional text fields.
    gender: OptionalText = None
    Partner: OptionalText = None
    Dependents: OptionalText = None
    PhoneService: OptionalText = None
    MultipleLines: OptionalText = None
    InternetService: OptionalText = None
    OnlineSecurity: OptionalText = None
    OnlineBackup: OptionalText = None
    DeviceProtection: OptionalText = None
    TechSupport: OptionalText = None
    StreamingTV: OptionalText = None
    StreamingMovies: OptionalText = None
    PaperlessBilling: OptionalText = None
    PaymentMethod: OptionalText = None


class PredictionResponse(BaseModel):
    """What /predict returns."""

    model_config = ConfigDict(protected_namespaces=())

    customerID: Optional[str] = None
    churn_score: float = Field(description="Risk score between 0 and 1")
    churn_prediction: int = Field(description="1 = likely to churn, 0 = likely to stay")
    threshold: float = Field(description="Score at or above this value means churn")
    served_by: Literal["model", "fallback"]
    model_version: str
    warnings: List[str] = []


class HealthResponse(BaseModel):
    """What /health returns."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded", "down"]
    model_loaded: bool
    fallback_ready: bool
    model_version: str
