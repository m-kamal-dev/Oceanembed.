"""
OceanEmbed model training.

DATA INTEGRITY RULE (see docs/DATA_INTEGRITY.md):
  This script will NOT silently fall back to synthetic/demo data.
  By default it requires the validated real dataset produced by the
  real-data pipeline:

      python scripts/build_dataset.py --use-raw --max-profiles 200
      # -> data/dataset/train_dataset.parquet  (only exists if
      #    scripts/validate_dataset.py passed the hard gate)

  If that file is missing, training stops with instructions instead
  of quietly using fabricated numbers.

  For local development/UI testing ONLY, you may explicitly pass
  --demo to train on synthetic data. This writes a SEPARATELY NAMED
  artifact (model_demo.pkl) so it can never be mistaken for, or
  silently overwrite, a model trained on real observations.

Usage:
    python train_model.py                # real data (default, safe)
    python train_model.py --demo         # explicit synthetic/demo model
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRegressor
except ImportError as e:
    print("=" * 70)
    print("TRAINING BLOCKED: lightgbm is not importable in this Python "
          "environment.")
    print(f"  ({type(e).__name__}: {e})")
    print()
    print("Fix: pip install -r requirements.txt  -- run in the SAME venv/")
    print("environment this script is being run with. The most common cause")
    print("is installing into one Python environment (e.g. system pip) and")
    print("running the script with another (e.g. a venv or conda env).")
    print("If `pip install lightgbm` itself fails to build, see")
    print("https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html")
    print("(e.g. macOS commonly needs `brew install libomp` first; minimal")
    print("Linux/Docker images commonly need `apt-get install libgomp1`).")
    print("=" * 70)
    sys.exit(1)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parent
REAL_DATASET = ROOT / "data" / "dataset" / "train_dataset.parquet"
DEMO_DATASET = ROOT / "data" / "demo" / "ocean_data_synthetic.csv"

FEATURES = ["lat", "lon", "day_of_year", "sst", "ssh", "sss"]
TARGETS = ["temp_50m", "temp_100m", "temp_200m", "temp_500m"]


def _make_base_model(n_train: int) -> LGBMRegressor:
    # LightGBM's default min_child_samples (20) silently refuses to split
    # further once a leaf has fewer than 20 rows -- fine for a large
    # dataset, but with a small real-observation dataset (a handful of
    # Argo floats) it means the model barely learns anything and every
    # split gets skipped. Scale it down to the training set size instead,
    # same fix used for the small real dataset elsewhere in this project.
    min_child = max(1, min(2, n_train - 1))
    return LGBMRegressor(n_estimators=200, learning_rate=0.03, random_state=42,
                          min_child_samples=min_child, verbose=-1)


def _fit_and_report(X_train, X_test, y_train, y_test):
    base = _make_base_model(len(X_train))
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = []
    print("\n--- Model Test Metrics (single held-out split) ---")
    for i, depth in enumerate(["50m", "100m", "200m", "500m"]):
        rmse = float(np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])))
        mae = float(mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]))
        r2 = float(r2_score(y_test.iloc[:, i], y_pred[:, i]))
        metrics.append({"Depth": depth, "RMSE (°C)": round(rmse, 3), "MAE (°C)": round(mae, 3), "R² Score": round(r2, 3)})
        print(f"Depth {depth:>4s} | RMSE: {rmse:.3f}°C | MAE: {mae:.3f}°C | R²: {r2:.3f}")
    return model, pd.DataFrame(metrics)


def _cross_validate(X, y, groups=None, n_splits=5):
    """K-fold (grouped by argo_wmo when possible, so no float leaks across
    folds) cross-validated metrics.

    A single 80/20 holdout on ~45 rows tests on ~9 points -- one unlucky
    (or lucky) float can swing R² wildly, so that number alone is not a
    trustworthy accuracy claim. Averaging over several folds, each one a
    genuinely different held-out slice, gives a far more honest estimate
    of how the model actually generalizes, at the cost of not being able
    to hold any single one of those folds out as a fixed "test_sample.csv"
    for the UI -- so we still do that separately (see train_real below),
    and report BOTH: this is the trustworthy accuracy number, that one is
    what the Model Validation tab has concrete rows to show.
    """
    n_groups = groups.nunique() if groups is not None else len(X)
    n_splits = max(2, min(n_splits, n_groups))
    if groups is not None and n_groups >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)

    per_depth = {depth: {"rmse": [], "mae": [], "r2": []} for depth in TARGETS}
    for train_idx, test_idx in split_iter:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        cv_model = MultiOutputRegressor(_make_base_model(len(X_tr)))
        cv_model.fit(X_tr, y_tr)
        y_pred = cv_model.predict(X_te)
        for i, depth in enumerate(TARGETS):
            per_depth[depth]["rmse"].append(float(np.sqrt(mean_squared_error(y_te.iloc[:, i], y_pred[:, i]))))
            per_depth[depth]["mae"].append(float(mean_absolute_error(y_te.iloc[:, i], y_pred[:, i])))
            per_depth[depth]["r2"].append(float(r2_score(y_te.iloc[:, i], y_pred[:, i])))

    rows = []
    print(f"\n--- {n_splits}-Fold Cross-Validated Metrics (the trustworthy number) ---")
    for depth, label in zip(TARGETS, ["50m", "100m", "200m", "500m"]):
        rmse_mean, rmse_std = float(np.mean(per_depth[depth]["rmse"])), float(np.std(per_depth[depth]["rmse"]))
        mae_mean = float(np.mean(per_depth[depth]["mae"]))
        r2_mean, r2_std = float(np.mean(per_depth[depth]["r2"])), float(np.std(per_depth[depth]["r2"]))
        rows.append({
            "Depth": label,
            "RMSE (°C)": round(rmse_mean, 3), "RMSE std": round(rmse_std, 3),
            "MAE (°C)": round(mae_mean, 3),
            "R² Score": round(r2_mean, 3), "R² std": round(r2_std, 3),
        })
        print(f"Depth {label:>4s} | RMSE: {rmse_mean:.3f}±{rmse_std:.3f}°C | MAE: {mae_mean:.3f}°C | R²: {r2_mean:.3f}±{r2_std:.3f}")
    return pd.DataFrame(rows)


def train_real():
    if not REAL_DATASET.exists():
        print("=" * 70)
        print("TRAINING BLOCKED: no validated real dataset found.")
        print(f"Expected: {REAL_DATASET}")
        print()
        print("Run the real-data pipeline first (requires internet access):")
        print("    python scripts/build_dataset.py --use-raw --max-profiles 200")
        print()
        print("That command fetches genuine Argo profiles + matched satellite")
        print("surface observations, runs the hard-gate validator, and only")
        print("writes train_dataset.parquet if it passes (see")
        print("docs/DATA_INTEGRITY.md). This script refuses to substitute")
        print("synthetic data for it.")
        print()
        print("For local UI/dev testing only, run: python train_model.py --demo")
        print("=" * 70)
        sys.exit(1)

    df = pd.read_parquet(REAL_DATASET)
    print(f"Loaded {len(df)} validated real observation rows from {REAL_DATASET.name}")

    X = df[FEATURES]
    y = df[TARGETS]
    groups = df["argo_wmo"] if "argo_wmo" in df.columns else None

    # Cross-validated metrics FIRST, before touching the holdout split below.
    # With ~45 total rows, a single 80/20 holdout tests on ~9 points -- too
    # few for R²/RMSE to mean much on their own (one unusual float can swing
    # it either way). This is the number to actually trust.
    cv_metrics_df = _cross_validate(X, y, groups=groups)

    # Group split by Argo float (argo_wmo) so the same float never appears
    # in both train and test -- prevents leakage between nearby profiles
    # from the same instrument. Falls back to a plain random split if the
    # grouped split can't be formed (e.g. too few unique floats for a
    # non-trivial 80/20 group partition) rather than crashing the run.
    X_train = X_test = y_train = y_test = None
    if groups is not None and groups.nunique() > 1:
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            if len(train_idx) == 0 or len(test_idx) == 0:
                raise ValueError("grouped split produced an empty train or test side")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        except (StopIteration, ValueError) as e:
            print(f"Note: grouped 80/20 split not usable ({e}); falling back to a random split.")
    if X_train is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, metrics_df = _fit_and_report(X_train, X_test, y_train, y_test)

    artifact = {
        "model": model,
        "metrics": metrics_df,
        "cv_metrics": cv_metrics_df,
        "feature_names": FEATURES,
        "data_mode": "real",
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "source_dataset": str(REAL_DATASET.name),
    }
    with open(ROOT / "model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    test_export = pd.concat([X_test, y_test], axis=1)
    # carry provenance columns through if present, so the app can show them
    for col in ["argo_wmo", "argo_cycle", "surface_distance_km", "surface_time_diff_hours"]:
        if col in df.columns:
            test_export[col] = df.loc[X_test.index, col]
    test_export.to_csv(ROOT / "test_sample.csv", index=False)

    print(f"\nSaved model.pkl (data_mode=real) and test_sample.csv ({len(test_export)} rows).")
    print("NOTE: 'metrics' in model.pkl is the single-holdout number shown in the")
    print("Model Validation tab (matches the concrete rows in test_sample.csv).")
    print("'cv_metrics' is the more trustworthy cross-validated estimate -- prefer")
    print("that one for any accuracy claim in a report, README, or demo.")


def train_demo():
    if not DEMO_DATASET.exists():
        print("No demo dataset found. Generating one now (clearly-labeled synthetic data)...")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_demo_data.py")], check=True)

    df = pd.read_csv(DEMO_DATASET)
    print(f"Loaded {len(df)} SYNTHETIC demo rows from {DEMO_DATASET.name}")
    print("⚠️  This model is for UI/development testing only. It is NOT trained")
    print("    on real observations and must never be presented as such.")

    X = df[FEATURES]
    y = df[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, metrics_df = _fit_and_report(X_train, X_test, y_train, y_test)

    artifact = {
        "model": model,
        "metrics": metrics_df,
        "feature_names": FEATURES,
        "data_mode": "demo_synthetic",
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "source_dataset": str(DEMO_DATASET.name),
    }
    # Deliberately different filenames so a demo model can NEVER be
    # mistaken for, or silently overwrite, a real one.
    with open(ROOT / "model_demo.pkl", "wb") as f:
        pickle.dump(artifact, f)

    test_export = pd.concat([X_test, y_test], axis=1)
    test_export.to_csv(ROOT / "test_sample_demo.csv", index=False)
    print(f"\nSaved model_demo.pkl and test_sample_demo.csv ({len(test_export)} rows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OceanEmbed subsurface temperature model")
    parser.add_argument("--demo", action="store_true", help="Train on synthetic demo data (writes model_demo.pkl, never model.pkl)")
    args = parser.parse_args()
    train_demo() if args.demo else train_real()
