"""Unit tests for MAE by wing, metric reporting, and findings formatting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.mp131.monitoring_metrics import (
    calculate_mae_by_wing,
    generate_monitoring_report,
    update_findings_yaml,
)


def test_calculate_mae_by_wing_perfect_predictions():
    """Perfect predictions should yield MAE = 0 for all groups."""
    y_true = [20.0, 30.0, 40.0, 50.0]
    y_pred = [20.0, 30.0, 40.0, 50.0]
    wings = ["Old Wing", "Old Wing", "New Wing", "New Wing"]

    mae_df, worst_served = calculate_mae_by_wing(y_true, y_pred, wings)

    assert len(mae_df) == 2
    for mae in mae_df["mae_minutes"]:
        assert mae == pytest.approx(0.0)


def test_calculate_mae_by_wing_constant_errors():
    """Constant offsets should produce known expected MAE values."""
    # Old Wing: error = 5.0
    # New Wing: error = 25.0
    y_true = [10.0, 20.0, 100.0, 110.0]
    y_pred = [15.0, 25.0, 125.0, 135.0]
    wings = ["Old Wing", "Old Wing", "New Wing", "New Wing"]

    mae_df, worst_served = calculate_mae_by_wing(y_true, y_pred, wings)

    old_row = mae_df[mae_df["wing"] == "Old Wing"].iloc[0]
    new_row = mae_df[mae_df["wing"] == "New Wing"].iloc[0]

    assert old_row["mae_minutes"] == pytest.approx(5.0)
    assert new_row["mae_minutes"] == pytest.approx(25.0)
    assert old_row["sample_count"] == 2
    assert new_row["sample_count"] == 2
    assert worst_served == "New Wing"


def test_calculate_mae_by_wing_mismatched_lengths_raise_error():
    """Unequal lengths among inputs must raise ValueError."""
    with pytest.raises(ValueError, match="Input lengths mismatch"):
        calculate_mae_by_wing([10.0, 20.0], [10.0], ["Old Wing", "Old Wing"])


def test_calculate_mae_by_wing_empty_input_raises_error():
    """Empty inputs must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        calculate_mae_by_wing([], [], [])


def test_calculate_mae_by_wing_handles_nan_gracefully():
    """NaN target or prediction values must be dropped safely without failing."""
    y_true = [10.0, np.nan, 20.0, 30.0]
    y_pred = [12.0, 15.0, np.nan, 35.0]
    wings = ["Old Wing", "Old Wing", "New Wing", "New Wing"]

    mae_df, worst_served = calculate_mae_by_wing(y_true, y_pred, wings)

    # Only first (Old Wing, error 2.0) and fourth (New Wing, error 5.0) are valid
    assert len(mae_df) == 2
    assert mae_df.loc[mae_df["wing"] == "Old Wing", "sample_count"].values[0] == 1
    assert mae_df.loc[mae_df["wing"] == "New Wing", "sample_count"].values[0] == 1
    assert worst_served == "New Wing"


def test_generate_monitoring_report_creates_valid_json(tmp_path: Path):
    """Ensure report JSON contains all necessary schema elements."""
    out_file = tmp_path / "drift_report.json"

    drift_df = pd.DataFrame([
        {"feature": "distance_km", "psi": 3.5514, "drift_classification": "Significant Drift"},
        {"feature": "patients_ahead", "psi": 0.0281, "drift_classification": "Stable"},
    ])
    mae_df = pd.DataFrame([
        {"wing": "New Wing", "sample_count": 73, "mae_minutes": 54.5180},
        {"wing": "Old Wing", "sample_count": 108, "mae_minutes": 15.8228},
    ])

    report = generate_monitoring_report(
        reference_path="data/raw/train.csv",
        current_path="data/raw/later.csv",
        model_name="mp131-wait-time",
        model_version="2",
        drift_df=drift_df,
        mae_df=mae_df,
        most_drifted="distance_km",
        worst_served="New Wing",
        output_path=str(out_file),
    )

    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["numerical_feature_drift"]["most_drifted_feature"] == "distance_km"
    assert loaded["performance_monitoring"]["worst_served_group"] == "New Wing"
    assert len(loaded["numerical_feature_drift"]["features"]) == 2
    assert len(loaded["performance_monitoring"]["mae_by_wing"]) == 2


def test_update_findings_yaml_preserves_teammate_fields(tmp_path: Path):
    """Updating monitoring fields in findings.yaml must not erase teammates' work."""
    findings_file = tmp_path / "findings.yaml"
    initial_content = """group_id: "G227"
problem_id: "MP131"
data_issues:
  leaking_column:
    present: true
    column: token_called_after_min
tracking:
  best_run_id: "0503735a4398463e9e9d6b21f4eb0785"
monitoring:
  mae_v1_on_check: 22.2801
  mae_v2_on_check: 8.4905
  decision: "promote"
"""
    findings_file.write_text(initial_content, encoding="utf-8")

    psi_map = {"distance_km": 3.5514, "patients_ahead": 0.0281}
    mae_map = {"Old Wing": 15.8228, "New Wing": 54.5180, "Annexe": 14.4869}

    update_findings_yaml(
        psi_by_feature=psi_map,
        most_drifted="distance_km",
        mae_by_group=mae_map,
        worst_served="New Wing",
        findings_path=str(findings_file),
    )

    yaml = pytest.importorskip("yaml")
    with findings_file.open("r", encoding="utf-8") as f:
        updated = yaml.safe_load(f)

    # Teammate fields preserved
    assert updated["group_id"] == "G227"
    assert updated["data_issues"]["leaking_column"]["column"] == "token_called_after_min"
    assert updated["tracking"]["best_run_id"] == "0503735a4398463e9e9d6b21f4eb0785"
    assert updated["monitoring"]["mae_v1_on_check"] == 22.2801
    assert updated["monitoring"]["decision"] == "promote"

    # Monitoring fields updated
    assert updated["monitoring"]["most_drifted_feature"] == "distance_km"
    assert updated["monitoring"]["worst_served_group"] == "New Wing"
    assert updated["monitoring"]["psi"]["distance_km"] == 3.5514
    assert updated["monitoring"]["mae_by_group"]["New Wing"] == 54.518
