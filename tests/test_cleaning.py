
import pandas as pd

from src.mp131.cleaning import clean_features, clean_training_data


def sample_data():
    return pd.DataFrame({
        "record_id": [1, 2, 3],
        "date": ["2026-01-01"] * 3,
        "patients_ahead": [5, 10, 8],
        "doctors_on_duty": [-1, 2, 3],
        "arrival_hour": [10, 11, 12],
        "avg_consult_min": [30, 120, 25],
        "lab_tests_pending": [2, 3, 1],
        "distance_km": [5.0, 6.0, 7.0],
        "department": ["General"] * 3,
        "wing": ["Old Wing", "New Wing", "Old Wing"],
        "token_called_after_min": [10, 15, 20],
        "wait_min": [40, 300, 50],
    })


def test_feature_columns():
    df = sample_data()
    X = clean_features(df)

    assert "token_called_after_min" not in X.columns
    assert len(X.columns) == 8


def test_sentinel_value_and_unit_correction():
    df = sample_data()
    X = clean_features(df)

    assert pd.isna(X.iloc[0]["doctors_on_duty"])
    assert X.iloc[1]["avg_consult_min"] == 2.0


def test_target_typo_correction():
    df = sample_data()
    X, y = clean_training_data(df)

    assert len(X) == 3
    assert y.iloc[1] == 30.0
