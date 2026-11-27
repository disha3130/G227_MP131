"""Compatibility facade module for monitoring in MP131.

Re-exports core monitoring functions from drift_monitor and monitoring_metrics.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from mp131.drift_monitor import (
    NUMERIC_FEATURES,
    calculate_numeric_feature_drift,
    calculate_psi,
    classify_psi,
    run_drift_monitoring,
)
from mp131.monitoring_metrics import (
    calculate_mae_by_wing,
    evaluate_v1_model_on_later,
    generate_monitoring_report,
    load_baseline_model,
    update_findings_yaml,
)


def compute_feature_drift(train_df, later_df, numeric_columns=None, bins=10, epsilon=1e-6):
    """Backward-compatible wrapper for calculate_numeric_feature_drift."""
    from mp131.cleaning import clean_features, clean_training_data

    X_train, _ = clean_training_data(train_df)
    X_later = clean_features(later_df)

    drift_df = calculate_numeric_feature_drift(
        reference_df=X_train,
        current_df=X_later,
        features=numeric_columns or NUMERIC_FEATURES,
        bins=bins,
        epsilon=epsilon,
    )
    psi_dict = dict(zip(drift_df["feature"], drift_df["psi"]))
    most_drifted = drift_df.loc[drift_df["psi"].idxmax(), "feature"] if not drift_df.empty else "None"
    return psi_dict, most_drifted


def compute_mae_by_group(model, later_df, group_col="wing", target_col="wait_min"):
    """Backward-compatible wrapper for evaluate_v1_model_on_later."""
    mae_df, worst_served = evaluate_v1_model_on_later(
        model=model,
        later_df=later_df,
        group_col=group_col,
        target_col=target_col,
    )
    mae_dict = dict(zip(mae_df["wing"], mae_df["mae_minutes"]))
    return mae_dict, worst_served


def update_findings(psi_by_feature, most_drifted, mae_by_group, worst_served, findings_path="findings.yaml"):
    """Backward-compatible wrapper for update_findings_yaml."""
    update_findings_yaml(
        psi_by_feature=psi_by_feature,
        most_drifted=most_drifted,
        mae_by_group=mae_by_group,
        worst_served=worst_served,
        findings_path=findings_path,
    )


def main():
    from mp131.drift_monitor import main as drift_main
    drift_main()


if __name__ == "__main__":
    main()
