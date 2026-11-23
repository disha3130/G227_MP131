
import pandas as pd


FEATURE_COLUMNS = [
    "patients_ahead",
    "doctors_on_duty",
    "arrival_hour",
    "avg_consult_min",
    "lab_tests_pending",
    "distance_km",
    "department",
    "wing",
]


def clean_features(df):
    """Clean input features without changing the original dataframe."""
    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates().copy()

    # Remove the post-event leakage column and other non-feature columns
    df = df.drop(
        columns=["record_id", "date", "token_called_after_min", "wait_min"],
        errors="ignore"
    )

    # Correct the unit mix-up in New Wing
    mask = (
        df["wing"].astype(str).str.strip().eq("New Wing")
        & (df["avg_consult_min"] > 30)
    )
    df.loc[mask, "avg_consult_min"] = (
        df.loc[mask, "avg_consult_min"] / 60
    )

    # Replace sentinel value with missing value
    df["doctors_on_duty"] = df["doctors_on_duty"].replace(-1, float("nan"))

    # Keep only the required model features in the correct order
    return df[FEATURE_COLUMNS]


def clean_training_data(df):
    """Clean training data and correct target typos."""
    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates().copy()

    # Correct target typos (values above 200 are recorded 10 times too high)
    mask = df["wait_min"] > 200
    df.loc[mask, "wait_min"] = df.loc[mask, "wait_min"] / 10

    # Remove rows where target is missing
    df = df.dropna(subset=["wait_min"]).copy()

    X = clean_features(df)
    y = df["wait_min"].astype(float)

    return X, y

