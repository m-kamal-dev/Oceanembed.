"""
Standalone inference wrapper: load a saved model artifact (model.pkl)
and run predictions for arbitrary lat/lon/day-of-year + surface feature
rows. Used for ad-hoc / batch prediction outside the Flask API; the API
itself (api_server.py) has its own request/response shaping around the
same underlying model.
"""
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "model.pkl"

DEFAULT_FEATURE_COLUMNS = ["lat", "lon", "day_of_year", "sst", "ssh", "sss"]
COLUMN_RENAMES = {"latitude": "lat", "longitude": "lon", "time": "day_of_year", "date": "day_of_year"}


def load_artifact(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load a pickled model artifact (dict with at least a 'model' key)."""
    p = Path(path) if path else ARTIFACT
    with open(p, "rb") as f:
        return pickle.load(f)


def predict_from_features(features_df: pd.DataFrame, artifact: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Run the model on a features DataFrame and return predicted temperatures.

    Accepts a few common column name variants (latitude/longitude/date)
    and maps them onto the model's expected feature schema before
    predicting.
    """
    if artifact is None:
        artifact = load_artifact()

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        expected = artifact.get("feature_names", DEFAULT_FEATURE_COLUMNS)
    else:
        model = artifact
        expected = DEFAULT_FEATURE_COLUMNS

    df = features_df.rename(columns=COLUMN_RENAMES)

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Features missing required columns: {missing}")

    predictions = model.predict(df[expected])
    n_outputs = predictions.shape[1] if hasattr(predictions, "shape") else 1
    column_names = ["pred_50m", "pred_100m", "pred_200m", "pred_500m"][:n_outputs]
    return pd.DataFrame(predictions, columns=column_names)


if __name__ == "__main__":
    # Quick smoke test against the bundled test sample.
    sample_path = ROOT / "test_sample.csv"
    if not sample_path.exists():
        raise SystemExit(f"{sample_path} not found — run train_model.py first.")
    sample_df = pd.read_csv(sample_path)
    features = sample_df[DEFAULT_FEATURE_COLUMNS].head(3)
    print(predict_from_features(features))
