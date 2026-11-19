"""The request schema must accept good payloads and reject bad ones."""
import pytest
from pydantic import ValidationError

from serving.schemas import CustomerFeatures


def test_valid_payload_is_accepted(valid_payload):
    customer = CustomerFeatures(**valid_payload)
    assert customer.tenure == 1
    assert customer.Contract == "Month-to-month"


def test_only_required_fields_is_accepted():
    customer = CustomerFeatures(tenure=5, MonthlyCharges=70.0, Contract="One year")
    # Optional fields default to None and are imputed later by the pipeline.
    assert customer.PaymentMethod is None
    assert customer.TotalCharges is None


@pytest.mark.parametrize("field", ["tenure", "MonthlyCharges", "Contract"])
def test_missing_required_field_is_rejected(valid_payload, field):
    del valid_payload[field]
    with pytest.raises(ValidationError):
        CustomerFeatures(**valid_payload)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("tenure", -1),                    # negative months
        ("tenure", "twelve"),              # text instead of a number
        ("tenure", 3.5),                   # not a whole number
        ("tenure", 5000),                  # beyond the sanity limit
        ("MonthlyCharges", -10.0),         # negative price
        ("MonthlyCharges", float("nan")),  # not a real number
        ("MonthlyCharges", float("inf")),
        ("TotalCharges", -1.0),
        ("SeniorCitizen", 7),              # only 0 or 1 allowed
        ("Contract", ""),                  # empty text
        ("Contract", "   "),               # only spaces is as good as empty
    ],
)
def test_bad_values_are_rejected(valid_payload, field, bad_value):
    valid_payload[field] = bad_value
    with pytest.raises(ValidationError):
        CustomerFeatures(**valid_payload)


def test_unknown_extra_field_is_rejected(valid_payload):
    valid_payload["favourite_colour"] = "blue"
    with pytest.raises(ValidationError):
        CustomerFeatures(**valid_payload)


def test_new_category_text_is_allowed(valid_payload):
    # A category the model has never seen must not be a schema error.
    valid_payload["PaymentMethod"] = "Crypto wallet"
    assert CustomerFeatures(**valid_payload).PaymentMethod == "Crypto wallet"
