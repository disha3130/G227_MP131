"""Performance monitoring and evaluation metrics module for MP131.

Focuses on wing-level MAE calculation, baseline model evaluation on later-period
data, machine-readable report generation, and findings.yaml updates.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.abspath("src"))

from mp131.cleaning import clean_features

logger = logging.getLogger(__name__)


def calculate_mae_by_wing(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
    wings: Union[pd.Series, np.ndarray, List[str]],
) -> Tuple[pd.DataFrame, str]:
    """Calculate Mean Absolute Error (MAE) grouped by hospital wing.

    Parameters
    ----------
    y_true : array-like
        True wait_min target values.
    y_pred : array-like
        Predicted wait_min values from model.
    wings : array-like
        Wing names corresponding to each sample.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        - Structured DataFrame with columns ['wing', 'sample_count', 'mae_minutes'].
        - Name of the worst-served wing (highest MAE).

    Raises
    ------
    ValueError
        If inputs have differing lengths or are empty.
    """
    yt = pd.to_numeric(pd.Series(y_true), errors="coerce")
    yp = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    w = pd.Series(wings).astype(str).str.strip()

    if len(yt) != len(yp) or len(yt) != len(w):
        raise ValueError(
            f"Input lengths mismatch: y_true({len(yt)}), y_pred({len(yp)}), wings({len(w)})"
        )
    if len(yt) == 0:
        raise ValueError("Cannot calculate MAE on empty input arrays.")

    df = pd.DataFrame({"y_true": yt, "y_pred": yp, "wing": w})
    # Remove any rows with NaN in true target or prediction
    valid_df = df.dropna(subset=["y_true", "y_pred"]).copy()
    if valid_df.empty:
        raise ValueError("No valid numeric target/prediction pairs available.")

    rows = []
    # Preserve conventional order if present
    unique_wings = sorted(valid_df["wing"].unique())
    for wing_name in unique_wings:
        wing_subset = valid_df[valid_df["wing"] == wing_name]
        count = len(wing_subset)
        mae = float(mean_absolute_error(wing_subset["y_true"], wing_subset["y_pred"]))
        rows.append({
            "wing": wing_name,
            "sample_count": count,
            "mae_minutes": round(mae, 4),
        })

    result_df = pd.DataFrame(rows)
    worst_served = result_df.loc[result_df["mae_minutes"].idxmax(), "wing"]

    return result_df, worst_served


def load_baseline_model(
    baseline_version: str = "2",
    fallback_path: str = "models/model.joblib",
) -> Any:
    """Load the baseline v1 model for monitoring comparisons.

    Inspects local MLflow artifact directories first, falls back to
    MLflow model registry, and then to the standalone joblib artifact.
    """
    # 1. Direct artifact lookup in local mlruns
    known_artifacts = [
        Path("mlruns/1/models/m-6b8bb033889d4a6d876c437d0d710539/artifacts/model.pkl"),
        Path("mlruns/1/models/m-6edc04e1dc714819b0df2be2b2baf2f3/artifacts/model.pkl"),
    ]
    for candidate in known_artifacts:
        if candidate.exists():
            logger.info("Loaded baseline model from local artifact: %s", candidate)
            return joblib.load(candidate)

    # 2. Search any model.pkl under mlruns/1
    for candidate in Path("mlruns/1").glob("**/model.pkl"):
        if candidate.exists():
            logger.info("Found and loaded baseline model from: %s", candidate)
            return joblib.load(candidate)

    # 3. Fallback to MLflow registry loading if environment permits
    try:
        from mp131.retrain import load_registered_model
        return load_registered_model(
            "mp131-wait-time",
            baseline_version,
            "sqlite:///mlflow.db",
            fallback_path,
        )
    except Exception as exc:
        logger.warning("Could not load from MLflow registry (%s); falling back to %s", exc, fallback_path)

    # 4. Fallback path
    fallback = Path(fallback_path)
    if fallback.exists():
        logger.info("Loaded baseline model from fallback path: %s", fallback)
        return joblib.load(fallback)

    raise FileNotFoundError(
        f"Unable to locate baseline model. Checked mlruns/ and {fallback_path}."
    )


def evaluate_v1_model_on_later(
    model: Any,
    later_df: pd.DataFrame,
    group_col: str = "wing",
    target_col: str = "wait_min",
) -> Tuple[pd.DataFrame, str]:
    """Preprocess later-period data, generate predictions, and calculate MAE by wing."""
    if target_col not in later_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in later DataFrame.")
    if group_col not in later_df.columns:
        raise KeyError(f"Grouping column '{group_col}' not found in later DataFrame.")

    # Clean features using identical preprocessing pipeline
    X_later = clean_features(later_df)
    valid_indices = X_later.index
    y_later = later_df.loc[valid_indices, target_col].astype(float)
    wings = later_df.loc[valid_indices, group_col]

    # Generate predictions
    predictions = model.predict(X_later)

    return calculate_mae_by_wing(y_true=y_later, y_pred=predictions, wings=wings)


def generate_monitoring_report(
    reference_path: str,
    current_path: str,
    model_name: str,
    model_version: str,
    drift_df: pd.DataFrame,
    mae_df: pd.DataFrame,
    most_drifted: str,
    worst_served: str,
    output_path: str = "reports/drift_report.json",
) -> Dict[str, Any]:
    """Generate and write a machine-readable JSON monitoring report."""
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": reference_path,
        "current_dataset": current_path,
        "model": {
            "name": model_name,
            "version": model_version,
        },
        "numerical_feature_drift": {
            "features": drift_df.to_dict(orient="records"),
            "most_drifted_feature": most_drifted,
            "highest_psi": float(drift_df.loc[drift_df["feature"] == most_drifted, "psi"].values[0])
            if not drift_df.empty and most_drifted in drift_df["feature"].values else None,
        },
        "performance_monitoring": {
            "mae_by_wing": mae_df.to_dict(orient="records"),
            "worst_served_group": worst_served,
            "highest_mae_minutes": float(mae_df.loc[mae_df["wing"] == worst_served, "mae_minutes"].values[0])
            if not mae_df.empty and worst_served in mae_df["wing"].values else None,
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def update_findings_yaml(
    psi_by_feature: Dict[str, float],
    most_drifted: str,
    mae_by_group: Dict[str, float],
    worst_served: str,
    findings_path: str = "findings.yaml",
) -> None:
    """Update only the monitoring section of findings.yaml while preserving everything else."""
    path = Path(findings_path)
    if not path.exists():
        raise FileNotFoundError(f"{findings_path} does not exist.")

    with path.open("r", encoding="utf-8") as f:
        findings = yaml.safe_load(f) or {}

    if "monitoring" not in findings:
        findings["monitoring"] = {}

    findings["monitoring"]["psi"] = {k: float(v) for k, v in psi_by_feature.items()}
    findings["monitoring"]["most_drifted_feature"] = str(most_drifted)
    findings["monitoring"]["mae_by_group"] = {k: float(v) for k, v in mae_by_group.items()}
    findings["monitoring"]["worst_served_group"] = str(worst_served)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)
