"""Drift monitoring module for MP131 (OPD Waiting Time Prediction).

Implements Population Stability Index (PSI) calculation and numerical
feature drift monitoring across reference (train) and production (later) data.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("src"))

from mp131.cleaning import clean_features, clean_training_data

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "patients_ahead",
    "doctors_on_duty",
    "arrival_hour",
    "avg_consult_min",
    "lab_tests_pending",
    "distance_km",
]

TARGET_OR_LEAKAGE_COLUMNS = {
    "record_id",
    "date",
    "token_called_after_min",
    "wait_min",
}

CATEGORICAL_COLUMNS = {"department", "wing"}


def classify_psi(psi_value: float) -> str:
    """Classify PSI magnitude according to standard MLOps thresholds.

    - PSI < 0.10: Stable (no significant population change).
    - 0.10 <= PSI <= 0.25: Moderate Drift (distribution shift, monitor closely).
    - PSI > 0.25: Significant Drift (action required / model retraining).
    """
    if np.isnan(psi_value):
        return "Unknown"
    if psi_value < 0.10:
        return "Stable"
    elif psi_value <= 0.25:
        return "Moderate Drift"
    else:
        return "Significant Drift"


def calculate_psi(
    expected: Union[np.ndarray, pd.Series, List[float]],
    actual: Union[np.ndarray, pd.Series, List[float]],
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate the Population Stability Index (PSI) between two distributions.

    Parameters
    ----------
    expected : array-like
        Baseline / reference dataset values (e.g. training data).
    actual : array-like
        Current / production dataset values (e.g. later data).
    bins : int, default 10
        Number of quantile or equal-width intervals to partition the baseline.
    epsilon : float, default 1e-6
        Small smoothing parameter to prevent division-by-zero or log-of-zero.

    Returns
    -------
    float
        The calculated Population Stability Index.

    Raises
    ------
    ValueError
        If inputs are empty, contain no finite values, or bins/epsilon are invalid.
    """
    if bins < 1:
        raise ValueError(f"Number of bins must be >= 1, got {bins}")
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive, got {epsilon}")

    exp_series = pd.to_numeric(pd.Series(expected), errors="coerce")
    act_series = pd.to_numeric(pd.Series(actual), errors="coerce")

    exp_clean = exp_series[np.isfinite(exp_series)].to_numpy(dtype=float)
    act_clean = act_series[np.isfinite(act_series)].to_numpy(dtype=float)

    if len(exp_clean) == 0:
        raise ValueError("Expected distribution contains no valid finite values.")
    if len(act_clean) == 0:
        raise ValueError("Actual distribution contains no valid finite values.")

    # Handle case where both expected and actual are identically constant
    exp_min, exp_max = float(np.min(exp_clean)), float(np.max(exp_clean))
    act_min, act_max = float(np.min(act_clean)), float(np.max(act_clean))

    if exp_min == exp_max and act_min == act_max and exp_min == act_min:
        return 0.0

    # Build reference bin edges
    bin_edges = np.histogram_bin_edges(exp_clean, bins=bins)

    # Expand outermost bounds so all actual values fall within bin boundaries
    bin_edges[0] = min(bin_edges[0], exp_min, act_min)
    bin_edges[-1] = max(bin_edges[-1], exp_max, act_max)

    # Calculate frequencies across identical bin edges
    expected_counts, _ = np.histogram(exp_clean, bins=bin_edges)
    actual_counts, _ = np.histogram(act_clean, bins=bin_edges)

    # Convert counts to proportions
    expected_pct = expected_counts / len(exp_clean)
    actual_pct = actual_counts / len(act_clean)

    # Apply epsilon smoothing for empty bins
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    # Normalize proportions to sum to 1
    expected_pct = expected_pct / np.sum(expected_pct)
    actual_pct = actual_pct / np.sum(actual_pct)

    # PSI = sum((actual% - expected%) * ln(actual% / expected%))
    psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_total = float(np.sum(psi_components))

    return max(0.0, psi_total)


def calculate_numeric_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Calculate drift statistics and PSI for numerical features.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Baseline/training dataset (should be cleaned).
    current_df : pd.DataFrame
        Current/later-period dataset.
    features : list of str, optional
        List of numerical feature column names. Defaults to NUMERIC_FEATURES.
    bins : int, default 10
        Number of bins for PSI calculation.
    epsilon : float, default 1e-6
        Smoothing parameter for empty bins.

    Returns
    -------
    pd.DataFrame
        Structured DataFrame with drift statistics per feature.
    """
    ref_df = reference_df.copy()
    cur_df = current_df.copy()

    if features is None:
        target_candidates = NUMERIC_FEATURES
    else:
        target_candidates = [
            f for f in features
            if f not in TARGET_OR_LEAKAGE_COLUMNS and f not in CATEGORICAL_COLUMNS
        ]

    rows = []
    for col in target_candidates:
        if col not in ref_df.columns:
            logger.warning("Feature '%s' not present in reference DataFrame.", col)
            continue
        if col not in cur_df.columns:
            logger.warning("Feature '%s' not present in current DataFrame.", col)
            continue

        exp_vals = pd.to_numeric(ref_df[col], errors="coerce").dropna()
        act_vals = pd.to_numeric(cur_df[col], errors="coerce").dropna()

        if len(exp_vals) == 0 or len(act_vals) == 0:
            logger.warning("Feature '%s' has no valid finite values to compare.", col)
            continue

        train_mean = float(exp_vals.mean())
        later_mean = float(act_vals.mean())
        train_std = float(exp_vals.std()) if len(exp_vals) > 1 else 0.0
        later_std = float(act_vals.std()) if len(act_vals) > 1 else 0.0

        abs_diff = abs(later_mean - train_mean)
        rel_change = (
            abs_diff / abs(train_mean) if abs(train_mean) > 1e-9 else np.nan
        )

        psi_score = calculate_psi(exp_vals, act_vals, bins=bins, epsilon=epsilon)
        classification = classify_psi(psi_score)

        rows.append({
            "feature": col,
            "training_mean": round(train_mean, 4),
            "later_mean": round(later_mean, 4),
            "training_std": round(train_std, 4),
            "later_std": round(later_std, 4),
            "abs_mean_diff": round(abs_diff, 4),
            "rel_mean_change": round(rel_change, 4) if np.isfinite(rel_change) else None,
            "psi": round(psi_score, 4),
            "drift_classification": classification,
        })

    return pd.DataFrame(rows)


def run_drift_monitoring(
    train_path: str = "data/raw/train.csv",
    later_path: str = "data/raw/later.csv",
    bins: int = 10,
    epsilon: float = 1e-6,
) -> Tuple[pd.DataFrame, str]:
    """Execute numerical drift analysis comparing cleaned train vs later dataset."""
    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path)

    X_train, _ = clean_training_data(train_df)
    X_later = clean_features(later_df)

    drift_df = calculate_numeric_feature_drift(
        reference_df=X_train,
        current_df=X_later,
        features=NUMERIC_FEATURES,
        bins=bins,
        epsilon=epsilon,
    )

    if not drift_df.empty:
        most_drifted = drift_df.loc[drift_df["psi"].idxmax(), "feature"]
    else:
        most_drifted = "None"

    return drift_df, most_drifted


def main() -> None:
    """Entry point for python -m mp131.drift_monitor."""
    from mp131.monitoring_metrics import (
        evaluate_v1_model_on_later,
        generate_monitoring_report,
        load_baseline_model,
        update_findings_yaml,
    )
    from mp131.retrain import load_params

    params = load_params("params.yaml")
    data_cfg = params.get("data", {})
    art_cfg = params.get("artifacts", {})
    mlf_cfg = params.get("mlflow", {})

    train_path = data_cfg.get("train_path", "data/raw/train.csv")
    later_path = data_cfg.get("later_path", "data/raw/later.csv")
    model_path = art_cfg.get("model_path", "models/model.joblib")
    findings_path = art_cfg.get("findings_path", "findings.yaml")

    print(f"Loading reference data from: {train_path}")
    print(f"Loading current data from: {later_path}")

    # Task A & B: PSI and numerical drift
    drift_df, most_drifted = run_drift_monitoring(train_path, later_path)

    print("\n" + "=" * 80)
    print("NUMERICAL FEATURE DRIFT & POPULATION STABILITY INDEX (PSI)")
    print("=" * 80)
    print(drift_df.to_string(index=False))
    print(f"\nMost drifted feature: {most_drifted}")

    # Task C: MAE by Wing
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE MONITORING: MAE BY WING")
    print("=" * 80)
    baseline_version = str(mlf_cfg.get("baseline_version", "2"))
    v1_model = load_baseline_model(
        baseline_version=baseline_version,
        fallback_path=model_path,
    )

    later_df = pd.read_csv(later_path)
    mae_df, worst_served = evaluate_v1_model_on_later(v1_model, later_df, group_col="wing")

    print(mae_df.to_string(index=False))
    print(f"\nWorst-served wing: {worst_served}")

    # Generate Reports
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "drift_report.json"

    psi_map = dict(zip(drift_df["feature"], drift_df["psi"]))
    mae_map = dict(zip(mae_df["wing"], mae_df["mae_minutes"]))

    generate_monitoring_report(
        reference_path=train_path,
        current_path=later_path,
        model_name=mlf_cfg.get("registered_model_name", "mp131-wait-time"),
        model_version=baseline_version,
        drift_df=drift_df,
        mae_df=mae_df,
        most_drifted=most_drifted,
        worst_served=worst_served,
        output_path=str(report_path),
    )

    # Update findings.yaml
    update_findings_yaml(
        psi_by_feature=psi_map,
        most_drifted=most_drifted,
        mae_by_group=mae_map,
        worst_served=worst_served,
        findings_path=findings_path,
    )
    print("\n" + "=" * 80)
    print(f"Artifacts successfully written:")
    print(f"  - Machine-readable report: {report_path}")
    print(f"  - Findings document:      {findings_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
