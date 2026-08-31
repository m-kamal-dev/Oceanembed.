"""
Generate CLEARLY-LABELED synthetic demo data for UI/development testing.

This replaces the old generate_data.py, which tried a live OPeNDAP fetch
and then SILENTLY fell back to fabricated numbers under the same filename
(ocean_data.csv) that the app trained on -- with no indication to the user
that a fallback had happened. That is exactly the failure mode
docs/DATA_INTEGRITY.md warns against, so this script does the opposite:

  - It never attempts to pass synthetic values off as observations.
  - It never touches the real pipeline's output files.
  - It always writes to data/demo/ocean_data_synthetic.csv, a path whose
    name makes its status unambiguous, and every row carries
    is_synthetic=True plus a source_region label.
  - The app's demo-mode badges ("SYNTHETIC", "DEMO DATA") come from
    DATA_MODE == "demo_synthetic" in api_server.py, driven by which model
    artifact loaded (model.pkl vs model_demo.pkl) -- NOT from anything in
    this file -- so there's no path by which these rows can end up
    labeled as real observations.

For real, validated data use scripts/build_dataset.py --use-raw instead.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import REAL_DATA_REGIONS  # same bounding boxes the real pipeline targets

OUT_DIR = ROOT / "data" / "demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "ocean_data_synthetic.csv"


def generate(n_samples: int = 1800, seed: int = 42) -> pd.DataFrame:
    """Physics-informed but fully synthetic ocean profiles, spread across
    the same three regions the real pipeline can target (Arabian Sea, Bay
    of Bengal, wider Indian Ocean) -- so the DEMO map gives a sense of what
    broader coverage will look like once real fetches for those regions
    are actually run. Every number here is a formula output, not a
    measurement; `is_synthetic` and `source_region` make that traceable
    per-row, not just at the file level.
    """
    rng = np.random.default_rng(seed)
    region_keys = list(REAL_DATA_REGIONS.keys())
    # Roughly even split across the three regions, remainder to the first.
    per_region = n_samples // len(region_keys)
    counts = {k: per_region for k in region_keys}
    counts[region_keys[0]] += n_samples - per_region * len(region_keys)

    frames = []
    for key in region_keys:
        box = REAL_DATA_REGIONS[key]
        n = counts[key]
        lats = rng.uniform(box["min_lat"], box["max_lat"], n)
        lons = rng.uniform(box["min_lon"], box["max_lon"], n)
        days = rng.integers(1, 366, n)

        # Same latitudinal/seasonal warm-tropical-ocean trend used for all
        # three boxes -- a real fetch would show genuine regional
        # differences (e.g. Bay of Bengal's fresher surface layer from
        # river runoff); this synthetic generator does not model that,
        # which is exactly why it's demo data and not a substitute for
        # actually fetching Bay of Bengal observations.
        sst = 29.0 - (lats - 8.0) * 0.08 + np.sin(2 * np.pi * days / 365.0) * 1.2 + rng.normal(0, 0.3, n)
        ssh = 0.65 + (sst - 26.0) * 0.03 + rng.normal(0, 0.04, n)
        sss = 35.2 - np.abs(lons - 70.0) * 0.02 + rng.normal(0, 0.1, n)

        t50 = sst - 0.8 - (24.0 - lats) * 0.02 + rng.normal(0, 0.2, n)
        t100 = t50 - 2.5 - (ssh * 0.5) + rng.normal(0, 0.3, n)
        t200 = t100 - 4.1 + rng.normal(0, 0.4, n)
        t500 = t200 - 5.5 + rng.normal(0, 0.3, n)

        frames.append(pd.DataFrame({
            "lat": np.round(lats, 2), "lon": np.round(lons, 2), "day_of_year": days,
            "sst": np.round(sst, 2), "ssh": np.round(ssh, 3), "sss": np.round(sss, 2),
            "temp_50m": np.round(t50, 2), "temp_100m": np.round(t100, 2),
            "temp_200m": np.round(t200, 2), "temp_500m": np.round(t500, 2),
            "is_synthetic": True,
            "source_region": REAL_DATA_REGIONS[key]["label"],
        }))

    out = pd.concat(frames, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle region order


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} SYNTHETIC demo rows to {OUT_PATH}")
    print(f"Regions covered: {sorted(df['source_region'].unique())}")
    print("This file is for UI/development use only and is never read by")
    print("train_model.py's default (real) mode or by the validation hard gate.")
    print("Run `python train_model.py --demo` to (re)train model_demo.pkl on it.")
