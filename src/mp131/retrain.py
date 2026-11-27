import hashlib
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.abspath("src"))

from mp131.cleaning import clean_features, clean_training_data
from mp131.train import build_pipeline


DEFAULT_PARAMS = {
    "data": {
        "train_path": "data/raw/train.csv",
        "later_path": "data/raw/later.csv",
        "holdout_path": "data/raw/holdout_features.csv",
    },
    "artifacts": {
        "model_path": "models/model.joblib",
        "predictions_path": "predictions/holdout_predictions.csv",
        "findings_path": "findings.yaml",
    },
    "mlflow": {
        "tracking_uri": "sqlite:///mlflow.db",
        "experiment_name": "MP131-OPD-Wait-Time",
        "registered_model_name": "mp131-wait-time",
        "champion_alias": "champion",
    },
    "retraining": {
        "later_train_fraction": 0.7,
        "promotion_min_improvement": 0.05,
        "max_depth": 10,
    },
}


def load_params(params_path="params.yaml"):
    """Load pipeline parameters from params.yaml using yaml.safe_load."""
    path = Path(params_path)
    if not path.exists():
        return DEFAULT_PARAMS
    import yaml

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    merged = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
    for section, values in loaded.items():
        if isinstance(values, dict) and section in merged:
            merged[section].update(values)
        else:
            merged[section] = values
    return merged


def compute_sha256(file_path):
    """Return the hexadecimal SHA-256 digest of a file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def split_later_by_date(later_df, train_fraction=0.7):
    """Sort later_df chronologically by date and split into train/check slices."""
    later_sorted = later_df.sort_values("date", kind="mergesort").reset_index(drop=True)
    split_idx = int(len(later_sorted) * train_fraction)
    return later_sorted.iloc[:split_idx].copy(), later_sorted.iloc[split_idx:].copy()


def decide_promotion(mae_v1, mae_v2, min_improvement=0.05):
    """Return 'promote' if v2 reduces MAE by at least min_improvement, else 'keep'."""
    if mae_v1 <= 0:
        return "keep"
    relative_improvement = (mae_v1 - mae_v2) / mae_v1
    return "promote" if relative_improvement >= min_improvement else "keep"


def _resolve_local_mlruns_path(uri):
    """Map an mlruns URI (possibly from another machine) to the local mlruns folder."""
    if not uri:
        return None
    normalized = uri.replace("\\", "/")
    marker = "/mlruns/"
    if marker in normalized:
        subpath = normalized.split(marker, 1)[1]
        candidate = Path("mlruns") / subpath
        if candidate.exists():
            return candidate
    return None


def load_registered_model(model_name, version_or_alias, tracking_uri, fallback_path):
    """Load a registered MLflow model by version or @alias with local fallback."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    ref = str(version_or_alias)
    model_uri = f"models:/{model_name}/{ref}"
    if ref.startswith("@"):
        model_uri = f"models:/{model_name}{ref}"

    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        client = MlflowClient()
        if ref.startswith("@"):
            mv = client.get_model_version_by_alias(model_name, ref[1:])
        else:
            mv = client.get_model_version(model_name, ref)

        local_source = _resolve_local_mlruns_path(mv.source)
        if local_source is not None:
            return mlflow.sklearn.load_model(str(local_source))

        run = client.get_run(mv.run_id)
        local_run_art = _resolve_local_mlruns_path(run.info.artifact_uri)
        if local_run_art is not None:
            for candidate in local_run_art.glob("**/model.pkl"):
                return mlflow.sklearn.load_model(str(candidate.parent))

    return joblib.load(fallback_path)


def _get_or_create_writable_experiment(base_name):
    """Return a local writable MLflow experiment, avoiding stale foreign paths."""
    import mlflow

    existing = mlflow.get_experiment_by_name(base_name)
    if existing is not None:
        loc = (existing.artifact_location or "").replace("\\", "/")
        # If the experiment points to this repo's mlruns or a relative path, reuse it.
        repo_mlruns = Path("mlruns").resolve().as_posix()
        if repo_mlruns in loc or not loc.startswith("file:///"):
            mlflow.set_experiment(base_name)
            return base_name

    retrain_exp = f"{base_name}-Retraining"
    if mlflow.get_experiment_by_name(retrain_exp) is None:
        artifact_root = (Path("mlruns") / "retraining").resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(retrain_exp, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(retrain_exp)
    return retrain_exp


def main(params_path="params.yaml"):
    params = load_params(params_path)
    data_cfg = params["data"]
    art_cfg = params["artifacts"]
    mlf_cfg = params["mlflow"]
    ret_cfg = params["retraining"]

    train_df = pd.read_csv(data_cfg["train_path"])
    later_df = pd.read_csv(data_cfg["later_path"])
    data_hash = compute_sha256(data_cfg["train_path"])

    X_train, y_train = clean_training_data(train_df)

    train_fraction = float(ret_cfg["later_train_fraction"])
    min_improvement = float(ret_cfg["promotion_min_improvement"])
    max_depth = ret_cfg["max_depth"]

    later_first, later_last = split_later_by_date(later_df, train_fraction=train_fraction)

    X_later_first = clean_features(later_first)
    y_later_first = later_first.loc[X_later_first.index, "wait_min"].astype(float)

    X_later_last = clean_features(later_last)
    y_later_last = later_last.loc[X_later_last.index, "wait_min"].astype(float)

    X_retrain = pd.concat([X_train, X_later_first], ignore_index=True)
    y_retrain = pd.concat([y_train, y_later_first], ignore_index=True)

    # Load baseline v1 model
    v1_model = load_registered_model(
        mlf_cfg["registered_model_name"],
        "1",
        mlf_cfg["tracking_uri"],
        art_cfg["model_path"],
    )

    # Train v2 model on cleaned train.csv + first 70% of later.csv
    v2_model = build_pipeline(max_depth)
    v2_model.fit(X_retrain, y_retrain)

    # Evaluate both models on the last 30% of later.csv
    mae_v1 = float(mean_absolute_error(y_later_last, v1_model.predict(X_later_last)))
    mae_v2 = float(mean_absolute_error(y_later_last, v2_model.predict(X_later_last)))
    improvement_pct = ((mae_v1 - mae_v2) / mae_v1 * 100.0) if mae_v1 > 0 else 0.0
    decision = decide_promotion(mae_v1, mae_v2, min_improvement=min_improvement)

    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(mlf_cfg["tracking_uri"])
    exp_used = _get_or_create_writable_experiment(mlf_cfg["experiment_name"])

    with mlflow.start_run(run_name="retrain-v2-random-forest") as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", str(max_depth))
        mlflow.log_param("later_train_fraction", train_fraction)
        mlflow.log_param("promotion_min_improvement", min_improvement)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("later_train_rows", len(X_later_first))
        mlflow.log_param("later_check_rows", len(X_later_last))
        mlflow.log_param("decision", decision)
        mlflow.log_param("data_sha256", data_hash)

        mlflow.log_metric("mae_v1_check", mae_v1)
        mlflow.log_metric("mae_v2_check", mae_v2)
        mlflow.log_metric("mae_improvement_pct", improvement_pct)

        model_info = mlflow.sklearn.log_model(
            sk_model=v2_model,
            name="model_v2",
            serialization_format="cloudpickle",
            registered_model_name=mlf_cfg["registered_model_name"],
        )

    client = MlflowClient()
    v2_version = str(model_info.registered_model_version)
    champion_alias = mlf_cfg["champion_alias"]

    if decision == "promote":
        client.set_registered_model_alias(
            mlf_cfg["registered_model_name"],
            champion_alias,
            v2_version,
        )
        Path(art_cfg["model_path"]).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(v2_model, art_cfg["model_path"])

    champion_mv = client.get_model_version_by_alias(
        mlf_cfg["registered_model_name"],
        champion_alias,
    )

    print(f"MLflow experiment used: {exp_used}")
    print(f"v2 run_id: {run.info.run_id}")
    print(f"v2 registered version: {v2_version}")
    print(f"v1 MAE on last 30% ({len(X_later_last)} rows): {mae_v1:.4f}")
    print(f"v2 MAE on last 30% ({len(X_later_last)} rows): {mae_v2:.4f}")
    print(f"Improvement: {improvement_pct:.2f}% (threshold: {min_improvement * 100:.1f}%)")
    print(f"Decision: {decision}")
    print(f"Active @{champion_alias} version: {champion_mv.version}")

    return {
        "mae_v1_on_check": round(mae_v1, 4),
        "mae_v2_on_check": round(mae_v2, 4),
        "improvement_pct": round(improvement_pct, 2),
        "decision": decision,
        "v2_run_id": run.info.run_id,
        "v2_version": int(v2_version),
        "champion_version": int(champion_mv.version),
    }


if __name__ == "__main__":
    main()
