"""Request and response schemas for the API (Pydantic).

The request schema is the first line of defence. A payload that breaks
these rules is rejected with HTTP 422 before it gets near the model.

What is strict and what is lenient, and why:

- tenure, MonthlyCharges and Contract are required. The fallback rule needs
  tenure and Contract, and all three are basic account facts that every
  real customer record has. Contract may not be blank: spaces around it are
  removed first, so a value made only of spaces is rejected.
- Numbers must be finite and inside a sanity range (no negatives, no NaN,
  no absurd values). Text that is not a number ("twelve") is rejected, and
  so is a tenure with a fraction (3.5).
- Pydantic runs in its default (lax) mode. For tenure, MonthlyCharges and
  TotalCharges a number sent as text ("12") is converted to a number, and
  true / false are read as 1 / 0. SeniorCitizen takes only 0 or 1 (true /
  false are read as 1 / 0, the text "1" is rejected). I accept the
  conversions, because upstream systems often send numbers as text.
- Unknown extra fields are rejected, so a typo in a field name is caught
  instead of being silently ignored.
- Other fields are optional. If one is missing, the model pipeline imputes
  it and the response carries a warning.
- Category text is not restricted to a fixed list. A new plan or payment
  method must not take the service down. The encoder ignores the unknown
  value and the response carries a warning so monitoring can pick it up.
  The comparison is case-sensitive: "month-to-month" is not the known
  category "Month-to-month" and is treated as unknown.
"""
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Reusable type: text that may be left out, but may not be an empty string.
OptionalText = Annotated[Optional[str], Field(min_length=1, max_length=64)]

# Reusable type: text that must be present. Spaces around it are removed
# before the length check, so "   " counts as empty and is rejected.
RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]

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
    Contract: RequiredText

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
