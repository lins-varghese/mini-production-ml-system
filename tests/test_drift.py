"""PSI must be near zero for the same population and large for a shifted one."""
import numpy as np

from src.drift import psi_categorical, psi_numeric, psi_status


def test_psi_is_zero_for_identical_data():
    values = np.random.default_rng(0).normal(50, 10, 5000)
    assert psi_numeric(values, values) < 1e-9


def test_psi_is_small_for_two_samples_of_same_population():
    rng = np.random.default_rng(1)
    reference = rng.normal(50, 10, 5000)
    current = rng.normal(50, 10, 5000)
    assert psi_status(psi_numeric(reference, current)) == "stable"


def test_psi_raises_alert_for_shifted_numeric_data():
    rng = np.random.default_rng(2)
    reference = rng.normal(50, 10, 5000)
    current = rng.normal(65, 10, 5000)   # mean moved by 1.5 standard deviations
    assert psi_status(psi_numeric(reference, current)) == "alert"


def test_psi_categorical_detects_new_category():
    reference = ["card"] * 500 + ["cheque"] * 500
    same_mix = ["card"] * 50 + ["cheque"] * 50
    new_mix = ["card"] * 300 + ["cheque"] * 300 + ["wallet"] * 400
    assert psi_categorical(reference, same_mix) < 1e-9
    assert psi_status(psi_categorical(reference, new_mix)) == "alert"


def test_psi_status_boundaries():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.10) == "warning"
    assert psi_status(0.25) == "alert"
