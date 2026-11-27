import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath("src"))

from mp131.cleaning import clean_features
from mp131.retrain import load_params, load_registered_model


def predict_holdout(model, holdout_df):
    """Clean holdout features and return DataFrame with record_id and prediction."""
    cleaned_features = clean_features(holdout_df)
    record_ids = holdout_df.loc[cleaned_features.index, "record_id"].reset_index(drop=True)
    predictions = model.predict(cleaned_features)

    return pd.DataFrame({
        "record_id": record_ids,
        "prediction": [round(float(p), 4) for p in predictions],
    })


def main(params_path="params.yaml"):
    params = load_params(params_path)
    data_cfg = params["data"]
    art_cfg = params["artifacts"]
    mlf_cfg = params["mlflow"]

    champion_ref = f"@{mlf_cfg['champion_alias']}"
    champion_model = load_registered_model(
        mlf_cfg["registered_model_name"],
        champion_ref,
        mlf_cfg["tracking_uri"],
        art_cfg["model_path"],
    )

    holdout_df = pd.read_csv(data_cfg["holdout_path"])
    output_df = predict_holdout(champion_model, holdout_df)

    output_path = Path(art_cfg["predictions_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Saved {len(output_df)} holdout predictions to {output_path}")
    print(output_df.head().to_string(index=False))
    return output_df


if __name__ == "__main__":
    main()
