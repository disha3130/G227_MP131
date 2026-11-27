"""Unit tests for drift monitoring and PSI calculation."""

import numpy as np
import pandas as pd
import pytest

from src.mp131.drift_monitor import (
    calculate_numeric_feature_drift,
    calculate_psi,
    classify_psi,
)


def test_calculate_psi_identical_distributions():
    """Identical distributions should produce a PSI approximately zero."""
    data = np.random.RandomState(42).normal(loc=20.0, scale=3.0, size=1000)
    psi = calculate_psi(data, data, bins=10)
    assert psi == pytest.approx(0.0, abs=1e-4)
    assert classify_psi(psi) == "Stable"


def test_calculate_psi_shifted_distribution():
    """Shifted distributions must yield significant PSI (> 0.25)."""
    rng = np.random.RandomState(42)
    expected = rng.normal(loc=10.0, scale=2.0, size=1000)
    actual = rng.normal(loc=30.0, scale=2.0, size=1000)
    psi = calculate_psi(expected, actual, bins=10)
    assert psi > 0.25
    assert classify_psi(psi) == "Significant Drift"


def test_calculate_psi_zero_count_bins_no_nan_or_inf():
    """Zero-count bins should be safely smoothed without NaN or Inf."""
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
    # actual is completely shifted away so many bins have zero counts
    actual = np.array([50, 60, 70, 80] * 25)
    psi = calculate_psi(expected, actual, bins=10, epsilon=1e-6)
    assert np.isfinite(psi)
    assert psi > 0


def test_calculate_psi_constant_columns():
    """Constant columns should yield 0 when identical, without division-by-zero crashes."""
    expected = [5.0] * 100
    actual = [5.0] * 100
    psi = calculate_psi(expected, actual, bins=5)
    assert psi == 0.0

    # Different constants should produce positive drift
    actual_different = [10.0] * 100
    psi_diff = calculate_psi(expected, actual_different, bins=5)
    assert np.isfinite(psi_diff)
    assert psi_diff > 0.0


def test_calculate_psi_empty_inputs_raise_error():
    """Empty inputs or arrays with only NaNs must raise ValueError."""
    with pytest.raises(ValueError, match="Expected distribution contains no valid finite values"):
        calculate_psi([], [1.0, 2.0])

    with pytest.raises(ValueError, match="Actual distribution contains no valid finite values"):
        calculate_psi([1.0, 2.0], [])

    with pytest.raises(ValueError):
        calculate_psi([np.nan, np.nan], [1.0, 2.0])


def test_calculate_psi_missing_values_handled_safely():
    """Missing and non-finite values should be filtered out safely."""
    expected = [np.nan, 10.0, 11.0, 12.0, np.nan, 13.0, 14.0]
    actual = [10.0, np.inf, 11.0, 12.0, 13.0, -np.inf, 14.0]
    psi = calculate_psi(expected, actual, bins=5)
    assert np.isfinite(psi)
    assert psi == pytest.approx(0.0, abs=1e-3)


def test_calculate_psi_invalid_parameters_raise_error():
    """Invalid bins or epsilon should raise ValueError."""
    with pytest.raises(ValueError, match="bins must be >= 1"):
        calculate_psi([1.0, 2.0], [1.0, 2.0], bins=0)

    with pytest.raises(ValueError, match="Epsilon must be positive"):
        calculate_psi([1.0, 2.0], [1.0, 2.0], epsilon=-1e-4)


def test_classify_psi_thresholds():
    """Verify standard drift classification boundaries."""
    assert classify_psi(0.05) == "Stable"
    assert classify_psi(0.099) == "Stable"
    assert classify_psi(0.10) == "Moderate Drift"
    assert classify_psi(0.20) == "Moderate Drift"
    assert classify_psi(0.25) == "Moderate Drift"
    assert classify_psi(0.251) == "Significant Drift"
    assert classify_psi(3.55) == "Significant Drift"


def test_calculate_numeric_feature_drift_statistics():
    """Verify statistics calculation and structure for numeric feature drift."""
    rng = np.random.RandomState(42)
    n = 200

    ref_df = pd.DataFrame({
        "patients_ahead": rng.randint(0, 30, size=n),
        "distance_km": rng.uniform(5.0, 25.0, size=n),
        "department": ["General"] * n,
        "token_called_after_min": [15.0] * n,
        "wait_min": [45.0] * n,
    })

    cur_df = ref_df.copy()
    # Apply severe drift to distance_km
    cur_df["distance_km"] = rng.uniform(70.0, 100.0, size=n)

    drift_df = calculate_numeric_feature_drift(
        reference_df=ref_df,
        current_df=cur_df,
        features=["patients_ahead", "distance_km", "token_called_after_min", "department"],
        bins=10,
    )

    # Leakage and categorical columns must be filtered out
    features_present = list(drift_df["feature"])
    assert "patients_ahead" in features_present
    assert "distance_km" in features_present
    assert "token_called_after_min" not in features_present
    assert "department" not in features_present

    # Check calculated statistics
    dist_row = drift_df[drift_df["feature"] == "distance_km"].iloc[0]
    assert dist_row["later_mean"] > dist_row["training_mean"]
    assert dist_row["abs_mean_diff"] > 40.0
    assert dist_row["psi"] > 0.25
    assert dist_row["drift_classification"] == "Significant Drift"

    pat_row = drift_df[drift_df["feature"] == "patients_ahead"].iloc[0]
    assert pat_row["drift_classification"] == "Stable"


def test_calculate_numeric_feature_drift_does_not_mutate_inputs():
    """Verify input DataFrames are not modified during drift calculation."""
    ref_df = pd.DataFrame({"patients_ahead": [5, 10, 15], "distance_km": [2.0, 4.0, 6.0]})
    cur_df = pd.DataFrame({"patients_ahead": [6, 11, 16], "distance_km": [3.0, 5.0, 7.0]})

    ref_copy = ref_df.copy(deep=True)
    cur_copy = cur_df.copy(deep=True)

    _ = calculate_numeric_feature_drift(ref_df, cur_df, features=["patients_ahead", "distance_km"])

    pd.testing.assert_frame_equal(ref_df, ref_copy)
    pd.testing.assert_frame_equal(cur_df, cur_copy)


def test_calculate_numeric_feature_drift_handles_missing_columns():
    """Columns missing from either reference or current DataFrame must be skipped cleanly."""
    ref_df = pd.DataFrame({"patients_ahead": [10, 20]})
    cur_df = pd.DataFrame({"distance_km": [15, 25]})

    drift_df = calculate_numeric_feature_drift(ref_df, cur_df, features=["patients_ahead", "distance_km"])
    assert drift_df.empty
