"""
Central configuration for OceanEmbed.

Values that are used in more than one module (or that someone would
plausibly want to tune without hunting through the codebase) live here
instead of being duplicated as inline literals. Environment variables
override the defaults below so deployment settings never need a code
change.
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# ── Web server ──────────────────────────────────────────────────────
API_HOST = os.getenv("OCEANEMBED_HOST", "0.0.0.0")
API_PORT = int(os.getenv("OCEANEMBED_PORT", "5001"))
# Debug mode must be opted into explicitly; it must never be the
# default for something that could be deployed.
API_DEBUG = os.getenv("OCEANEMBED_DEBUG", "false").lower() in ("1", "true", "yes")

# ── Model / feature schema ─────────────────────────────────────────
FEATURE_COLUMNS = ["lat", "lon", "day_of_year", "sst", "ssh", "sss"]
TARGET_DEPTHS_M = [50, 100, 200, 500]

# ── Real-data matching constraints (data/match_surface_to_argo.py) ──
SPATIAL_MATCH_RADIUS_KM = 25.0
TEMPORAL_MATCH_WINDOW_HOURS = 24.0

# ── Geography ───────────────────────────────────────────────────────
EARTH_RADIUS_KM = 6371.0
# /api/predict-point no longer restricts to a bounding box around the
# training dataset -- it works at any ocean point on Earth, preferring
# (in order) a nearby validated observation, a live satellite fetch, or
# an explicitly-labeled statistical extrapolation. See
# get_surface_features_anywhere() in api_server.py.

# Reference year used to convert day-of-year back into a calendar date
# for display. The training data spans a single collection year.
REFERENCE_YEAR = 2026

# ── Real-data fetch regions (scripts/build_dataset.py --region) ─────
# Bounding boxes as (min_lat, max_lat, min_lon, max_lon), degrees.
# "arabian_sea" is the original single-region box this project shipped
# with (45 real matched rows, Jan 2025-Jan 2026). "bay_of_bengal" and
# "indian_ocean" are NOT fetched yet -- adding them here only makes the
# pipeline *capable* of pulling that data; it still requires actually
# running `python -m scripts.build_dataset --use-raw --region all` with
# live internet access to Ifremer/NOAA/NASA/ESA servers. Nothing in this
# repo fabricates rows for these regions to make coverage look bigger
# than it is -- see docs/DATA_INTEGRITY.md.
REAL_DATA_REGIONS = {
    "arabian_sea": {
        "label": "Arabian Sea",
        "min_lat": 8.0, "max_lat": 24.0,
        "min_lon": 60.0, "max_lon": 77.0,
    },
    "bay_of_bengal": {
        "label": "Bay of Bengal",
        "min_lat": 5.0, "max_lat": 22.0,
        "min_lon": 80.0, "max_lon": 100.0,
    },
    "indian_ocean": {
        # Broader North + equatorial Indian Ocean -- extends roughly
        # 200+ km beyond the coasts of India, Sri Lanka, and the Maldives
        # in every direction from the two regions above, without
        # reaching into the Southern Ocean or Pacific/Atlantic basins.
        "label": "Wider Indian Ocean (buffer around India)",
        "min_lat": -10.0, "max_lat": 25.0,
        "min_lon": 40.0, "max_lon": 100.0,
    },
}

# Tropical Cyclone Heat Potential proxy: the isotherm temperature (°C)
# whose depth is used as the risk indicator.
TCHP_ISOTHERM_C = 26.0

# ── Nemotron (optional AI copilot) ──────────────────────────────────
NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NEMOTRON_BASE_URL = "https://integrate.api.nvidia.com/v1"
