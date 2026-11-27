import pandas as pd

from src.mp131.cleaning import FEATURE_COLUMNS, clean_features
from src.mp131.predict import predict_holdout
from src.mp131.retrain import DEFAULT_PARAMS, build_pipeline, decide_promotion, split_later_by_date


def _synthetic_records(dates, wait_base=60.0):
    rows = []
    for idx, dt in enumerate(dates):
        rows.append({
            "record_id": f"R{100 + idx}",
            "date": dt,
            "patients_ahead": 10 + idx,
            "doctors_on_duty": 3,
            "arrival_hour": 10 + (idx % 6),
            "avg_consult_min": 8.0,
            "lab_tests_pending": 2,
            "distance_km": 12.0 + idx,
            "department": "General",
            "wing": "Old Wing" if idx % 2 == 0 else "Annexe",
            "token_called_after_min": 20.0,
            "wait_min": wait_base + idx * 2.0,
        })
    return pd.DataFrame(rows)


def test_split_later_by_date_sorts_and_splits():
    unsorted_dates = [
        "2026-08-10", "2026-07-01", "2026-08-20", "2026-07-05", "2026-07-15",
        "2026-08-01", "2026-07-20", "2026-08-25", "2026-07-10", "2026-08-15",
    ]
    df = _synthetic_records(unsorted_dates)
    first_70, last_30 = split_later_by_date(df, train_fraction=0.7)

    assert len(first_70) == 7
    assert len(last_30) == 3
    assert list(first_70["date"]) == sorted(first_70["date"])
    assert list(last_30["date"]) == sorted(last_30["date"])
    assert first_70["date"].max() <= last_30["date"].min()


def test_decide_promotion_rule():
    assert decide_promotion(20.0, 19.0, min_improvement=0.05) == "promote"
    assert decide_promotion(20.0, 18.0, min_improvement=0.05) == "promote"
    assert decide_promotion(20.0, 19.1, min_improvement=0.05) == "keep"
    assert decide_promotion(20.0, 21.0, min_improvement=0.05) == "keep"


def test_predict_holdout_format():
    train_df = _synthetic_records([f"2026-06-{10 + i:02d}" for i in range(10)])
    X = clean_features(train_df)
    y = train_df["wait_min"]

    model = build_pipeline(max_depth=3)
    model.fit(X[FEATURE_COLUMNS], y)

    holdout_df = train_df.drop(columns=["token_called_after_min", "wait_min"]).iloc[:4].copy()
    preds = predict_holdout(model, holdout_df)

    assert list(preds.columns) == ["record_id", "prediction"]
    assert len(preds) == 4
    assert list(preds["record_id"]) == list(holdout_df["record_id"])
    assert preds["prediction"].notna().all()


def test_baseline_version_defaults_to_stage2_champion():
    assert str(DEFAULT_PARAMS["mlflow"]["baseline_version"]) == "2"
