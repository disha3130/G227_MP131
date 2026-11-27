import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.abspath("src"))

from mp131.cleaning import clean_features, clean_training_data
from mp131.retrain import DEFAULT_PARAMS, load_params, load_registered_model

NUMERIC_COLUMNS = [
    "patients_ahead",
    "doctors_on_duty",
    "arrival_hour",
    "avg_consult_min",
    "lab_tests_pending",
    "distance_km",
]


def calculate_psi(expected, actual, bins=10, epsilon=1e-4):
    """
    Calculate the Population Stability Index (PSI) between two distributions.

    Parameters:
    - expected: Baseline/reference dataset (e.g. cleaned train.csv)
    - actual: Current/new dataset (e.g. later.csv)
    - bins: Number of bins to use (default: 10)
    - epsilon: Small smoothing value to handle zero-count bins (default: 1e-4)

    Returns:
    - psi: float, Population Stability Index value
    """
    exp = np.asarray(pd.Series(expected).dropna())
    act = np.asarray(pd.Series(actual).dropna())

    if len(exp) == 0 or len(act) == 0:
        return 0.0

    # Define 10 bin edges based on expected baseline distribution
    bin_edges = np.histogram_bin_edges(exp, bins=bins)

    # Expand outer boundaries to ensure all values from both sets are included
    bin_edges[0] = min(np.min(exp), np.min(act))
    bin_edges[-1] = max(np.max(exp), np.max(act))

    # Calculate frequencies across identical bins
    expected_counts, _ = np.histogram(exp, bins=bin_edges)
    actual_counts, _ = np.histogram(act, bins=bin_edges)

    # Convert counts to proportions
    expected_pct = expected_counts / len(exp)
    actual_pct = actual_counts / len(act)

    # Apply epsilon smoothing to prevent log(0) or division by zero
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    # Compute PSI: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


def compute_feature_drift(train_df, later_df, numeric_columns=None, bins=10):
    """
    Calculate PSI for each numeric feature between cleaned train and later datasets.

    Returns:
    - psi_by_feature: dict mapping feature name to rounded PSI score
    - most_drifted_feature: str, feature name with highest PSI score
    """
    if numeric_columns is None:
        numeric_columns = NUMERIC_COLUMNS

    X_train, _ = clean_training_data(train_df)
    X_later = clean_features(later_df)

    psi_by_feature = {}
    for col in numeric_columns:
        if col in X_train.columns and col in X_later.columns:
            psi_val = calculate_psi(X_train[col], X_later[col], bins=bins)
            psi_by_feature[col] = round(psi_val, 4)

    most_drifted_feature = max(psi_by_feature, key=psi_by_feature.get)
    return psi_by_feature, most_drifted_feature


def compute_mae_by_group(model, later_df, group_col="wing", target_col="wait_min"):
    """
    Compute model MAE on later.csv grouped by wing or another column.

    Returns:
    - mae_by_group: dict mapping group name to rounded MAE
    - worst_served_group: str, group name with the highest MAE
    """
    X_later = clean_features(later_df)
    y_later = later_df.loc[X_later.index, target_col].astype(float)

    predictions = model.predict(X_later)

    eval_df = later_df.loc[X_later.index].copy()
    eval_df["prediction"] = predictions

    groups = eval_df[group_col].dropna().unique()
    mae_by_group = {}

    for grp in groups:
        mask = eval_df[group_col] == grp
        grp_y = y_later[mask]
        grp_pred = eval_df.loc[mask, "prediction"]
        mae = float(mean_absolute_error(grp_y, grp_pred))
        mae_by_group[str(grp)] = round(mae, 4)

    worst_served_group = max(mae_by_group, key=mae_by_group.get)
    return mae_by_group, worst_served_group


def update_findings(psi_by_feature, most_drifted, mae_by_group, worst_served, findings_path="findings.yaml"):
    """Update monitoring fields in findings.yaml while preserving the rest of the file."""
    path = Path(findings_path)
    if not path.exists():
        raise FileNotFoundError(f"{findings_path} not found")

    with path.open("r", encoding="utf-8") as f:
        findings = yaml.safe_load(f) or {}

    if "monitoring" not in findings:
        findings["monitoring"] = {}

    findings["monitoring"]["psi"] = psi_by_feature
    findings["monitoring"]["most_drifted_feature"] = most_drifted
    findings["monitoring"]["mae_by_group"] = mae_by_group
    findings["monitoring"]["worst_served_group"] = worst_served

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)

    print(f"Updated {findings_path} with monitoring metrics.")


def load_baseline_model(baseline_version="2", fallback_path="models/model.joblib"):
    """
    Load baseline v1 model for monitoring comparison.
    Tries:
    1. Direct artifact lookup in mlruns/ matching baseline model
    2. MLflow model registry via load_registered_model
    3. Fallback joblib model file
    """
    candidates = [
        Path("mlruns/1/models/m-6b8bb033889d4a6d876c437d0d710539/artifacts/model.pkl"),
        Path("mlruns/1/models/m-6edc04e1dc714819b0df2be2b2baf2f3/artifacts/model.pkl"),
    ]
    for c in candidates:
        if c.exists():
            return joblib.load(c)

    # Search any model.pkl under mlruns/1
    for p in Path("mlruns/1").glob("**/model.pkl"):
        if p.exists():
            return joblib.load(p)

    try:
        from mp131.retrain import load_registered_model
        return load_registered_model("mp131-wait-time", baseline_version, "sqlite:///mlflow.db", fallback_path)
    except Exception:
        return joblib.load(fallback_path)


def main(params_path="params.yaml"):
    params = load_params(params_path)
    data_cfg = params["data"]
    art_cfg = params["artifacts"]
    mlf_cfg = params["mlflow"]

    print(f"Loading training data from {data_cfg['train_path']}...")
    train_df = pd.read_csv(data_cfg["train_path"])
    print(f"Loading later data from {data_cfg['later_path']}...")
    later_df = pd.read_csv(data_cfg["later_path"])

    print("\n--- 1. Computing Feature Drift (PSI) ---")
    psi_by_feature, most_drifted = compute_feature_drift(train_df, later_df, bins=10)
    for feat, psi_val in psi_by_feature.items():
        drift_level = "Severe Drift (>= 0.25)" if psi_val >= 0.25 else ("Moderate Drift (>= 0.1)" if psi_val >= 0.1 else "Stable (< 0.1)")
        print(f"  {feat:<20}: {psi_val:.4f}  [{drift_level}]")
    print(f"\nMost drifted feature: {most_drifted} (PSI = {psi_by_feature[most_drifted]:.4f})")

    print("\n--- 2. Computing v1 MAE by Wing on later.csv ---")
    baseline_version = str(mlf_cfg.get("baseline_version", "2"))
    v1_model = load_baseline_model(baseline_version, art_cfg.get("model_path", "models/model.joblib"))

    mae_by_group, worst_served = compute_mae_by_group(v1_model, later_df, group_col="wing")
    for grp, mae_val in mae_by_group.items():
        print(f"  {grp:<15}: MAE = {mae_val:.4f}")
    print(f"\nWorst served group: {worst_served} (MAE = {mae_by_group[worst_served]:.4f})")

    findings_path = art_cfg.get("findings_path", "findings.yaml")
    update_findings(psi_by_feature, most_drifted, mae_by_group, worst_served, findings_path=findings_path)

    return {
        "psi": psi_by_feature,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served,
    }


if __name__ == "__main__":
    main()
