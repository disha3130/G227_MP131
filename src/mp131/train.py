
import hashlib
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.abspath("src"))

from mp131.cleaning import clean_training_data, clean_features, FEATURE_COLUMNS


EXPERIMENT_NAME = "MP131-OPD-Wait-Time"
MODEL_NAME = "mp131-wait-time"


def build_pipeline(max_depth):
    numeric_features = [
        "patients_ahead",
        "doctors_on_duty",
        "arrival_hour",
        "avg_consult_min",
        "lab_tests_pending",
        "distance_km",
    ]

    categorical_features = ["department", "wing"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def main():
    os.makedirs("models", exist_ok=True)

    # Read original datasets
    train_df = pd.read_csv("data/raw/train.csv")
    later_df = pd.read_csv("data/raw/later.csv")

    # Clean training and validation data
    X_train, y_train = clean_training_data(train_df)

    X_later = clean_features(later_df)
    y_later = later_df.loc[X_later.index, "wait_min"].astype(float)

    # SHA-256 hash of the original training file
    with open("data/raw/train.csv", "rb") as file:
        data_hash = hashlib.sha256(file.read()).hexdigest()

    # Configure MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    # Try three model configurations
    for depth in [5, 10, None]:
        model = build_pipeline(depth)

        with mlflow.start_run(run_name=f"random-forest-depth-{depth}") as run:
            model.fit(X_train, y_train)

            predictions = model.predict(X_later)
            mae = mean_absolute_error(y_later, predictions)

            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", str(depth))
            mlflow.log_param("data_sha256", data_hash)

            mlflow.log_metric("mae_later", mae)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle",
            )

            results.append({
                "run_id": run.info.run_id,
                "mae": mae,
                "model": model,
                "depth": depth,
            })

            print(f"Depth: {depth}, Later MAE: {mae:.4f}")

    # Select the model with the lowest validation MAE
    best = min(results, key=lambda item: item["mae"])

    # Save the best model locally
    joblib.dump(best["model"], "models/model.joblib")

    # Register the selected model in MLflow
    with mlflow.start_run(run_name="register-best-model"):
        model_info = mlflow.sklearn.log_model(
            sk_model=best["model"],
            name="best_model",
            serialization_format="cloudpickle",
            registered_model_name=MODEL_NAME,
        )

    # Assign champion alias to the registered version
    client = MlflowClient()
    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        str(model_info.registered_model_version),
    )

    print("\nBest model depth:", best["depth"])
    print("Best validation MAE:", round(best["mae"], 4))
    print("Saved model: models/model.joblib")
    print("Registered model:", MODEL_NAME)
    print("Champion version:", model_info.registered_model_version)


if __name__ == "__main__":
    main()
