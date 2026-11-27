import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.mp131.monitoring import calculate_psi, compute_feature_drift, compute_mae_by_group


def test_calculate_psi_identical_distributions():
    data = np.random.RandomState(42).normal(loc=10.0, scale=2.0, size=500)
    psi = calculate_psi(data, data, bins=10)
    assert psi == pytest.approx(0.0, abs=1e-3)


def test_calculate_psi_shifted_distribution():
    rng = np.random.RandomState(42)
    expected = rng.normal(loc=10.0, scale=2.0, size=500)
    actual = rng.normal(loc=25.0, scale=2.0, size=500)
    psi = calculate_psi(expected, actual, bins=10)
    # Significant shift should have PSI well above 0.25
    assert psi > 0.5


def test_calculate_psi_handles_empty_and_nan():
    expected = [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan]
    actual = [1.0, 2.0, 3.0, 4.0, 5.0]
    psi = calculate_psi(expected, actual, bins=5)
    assert psi >= 0.0
    assert calculate_psi([], [1, 2, 3]) == 0.0


def test_compute_feature_drift_identifies_most_drifted():
    rng = np.random.RandomState(42)
    n = 200

    train_df = pd.DataFrame({
        "record_id": [f"TR{i}" for i in range(n)],
        "date": ["2026-01-01"] * n,
        "patients_ahead": rng.randint(0, 50, size=n),
        "doctors_on_duty": rng.randint(1, 6, size=n),
        "arrival_hour": rng.randint(8, 17, size=n),
        "avg_consult_min": rng.uniform(5.0, 15.0, size=n),
        "lab_tests_pending": rng.randint(0, 5, size=n),
        "distance_km": rng.uniform(2.0, 20.0, size=n),
        "department": ["General"] * n,
        "wing": ["Old Wing"] * n,
        "token_called_after_min": [10.0] * n,
        "wait_min": [30.0] * n,
    })

    # create later_df where distance_km has massive drift, others stable
    later_df = train_df.copy()
    later_df["distance_km"] = rng.uniform(60.0, 100.0, size=n)

    psi_by_feat, most_drifted = compute_feature_drift(train_df, later_df, bins=10)

    assert "distance_km" in psi_by_feat
    assert most_drifted == "distance_km"
    assert psi_by_feat["distance_km"] > 0.25


def test_compute_mae_by_group():
    later_df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4"],
        "date": ["2026-07-01"] * 4,
        "patients_ahead": [5, 10, 15, 20],
        "doctors_on_duty": [3, 3, 3, 3],
        "arrival_hour": [10, 11, 12, 13],
        "avg_consult_min": [8.0, 8.0, 8.0, 8.0],
        "lab_tests_pending": [1, 1, 1, 1],
        "distance_km": [5.0, 10.0, 15.0, 20.0],
        "department": ["General", "General", "General", "General"],
        "wing": ["Old Wing", "Old Wing", "New Wing", "New Wing"],
        "wait_min": [20.0, 30.0, 100.0, 110.0],
    })

    # Dummy model predicting constant 25.0
    # Old Wing: actual [20, 30], errors [5, 5] -> MAE = 5.0
    # New Wing: actual [100, 110], errors [75, 85] -> MAE = 80.0
    dummy_model = DummyRegressor(strategy="constant", constant=25.0)
    dummy_model.fit(later_df[["patients_ahead"]], later_df["wait_min"])

    mae_by_group, worst_served = compute_mae_by_group(dummy_model, later_df, group_col="wing")

    assert mae_by_group["Old Wing"] == 5.0
    assert mae_by_group["New Wing"] == 80.0
    assert worst_served == "New Wing"
