"""
OceanEmbed Backend API.
Wraps the trained ML model, loaded dataset, and optional Nemotron AI
copilot integration, and serves JSON endpoints + the static frontend.
"""
import datetime
import io
import logging
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.routing import BaseConverter

from config import (
    API_DEBUG,
    API_HOST,
    API_PORT,
    EARTH_RADIUS_KM,
    FEATURE_COLUMNS,
    NEMOTRON_BASE_URL,
    NEMOTRON_MODEL,
    REFERENCE_YEAR,
    SPATIAL_MATCH_RADIUS_KM,
    TCHP_ISOTHERM_C,
    TEMPORAL_MATCH_WINDOW_HOURS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from flask_cors import CORS
except ImportError:
    CORS = None
    logger.warning(
        "flask-cors not installed; CORS disabled (fine for same-origin use, "
        "install flask-cors if the frontend is served from a different origin)"
    )

# ── Load .env (optional) ─────────────────────────────────────────
# Lets NEMOTRON_API_KEY live in a local .env file instead of being
# exported by hand every time. Safe no-op if python-dotenv isn't
# installed or no .env file exists.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# ── Land mask (filters land-based observations) ───────────────
# Natural Earth 110m land polygons cover the whole globe, so this check
# (unlike the old dataset bounding-box check it replaces below) is what
# actually decides "can we predict here" -- ocean = yes, anywhere on Earth.
try:
    from land_mask import is_ocean, filter_ocean_points
except ImportError:
    logger.warning("land_mask module not found — no land filtering")
    def is_ocean(lat, lon):
        return True

    def filter_ocean_points(pts):
        return list(pts), []

# ── Live surface-data fetch (NASA/NOAA/ESA ERDDAP, real-time) ──────
# Used by predict_point/cyclone_track when a clicked point is far from
# the validated Arabian Sea training set, so predictions aren't limited
# to that one basin: we go get a real, live SST/SSH/SSS reading for
# wherever was clicked instead of just refusing or blindly extrapolating.
try:
    from data.surface_fetch import fetch_nearest_surface as _fetch_live_surface
except ImportError:
    logger.warning("data.surface_fetch not importable — live fetch fallback disabled")
    _fetch_live_surface = None

# ── Fix Windows unicode ──────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Live-fetch cache + bounded timeout ──────────────────────────
# ERDDAP round-trips (SST + SSH + SSS) take real seconds; caching by
# rounded (lat, lon, day) keeps repeat clicks near the same spot instant,
# and a hard wall-clock cap keeps one slow/unreachable server from ever
# hanging a map click -- it just falls through to extrapolation instead.
_LIVE_FETCH_CACHE: dict = {}
_LIVE_FETCH_CACHE_MAX = 500
LIVE_FETCH_TIMEOUT_SECONDS = float(os.getenv("OCEANEMBED_LIVE_FETCH_TIMEOUT", "12"))


def _live_fetch_with_timeout(lat: float, lon: float, dt, timeout: float = LIVE_FETCH_TIMEOUT_SECONDS):
    """Call fetch_nearest_surface with a hard wall-clock cap so a slow or
    unreachable satellite endpoint can never hang an API request -- returns
    None (same as "no data found") on timeout, error, or if the fetcher
    isn't available at all."""
    if _fetch_live_surface is None:
        return None
    cache_key = (round(lat, 2), round(lon, 2), getattr(dt, "date", lambda: dt)())
    cached = _LIVE_FETCH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import concurrent.futures
    # Note: deliberately NOT using the executor as a context manager --
    # `with` blocks on __exit__ until the worker thread finishes, which
    # would silently undo the timeout below (a slow ERDDAP call would
    # still hang the request for its full internal timeout). Submitting
    # standalone and shutting down with wait=False lets a slow call keep
    # running in the background harmlessly while we move on immediately.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_fetch_live_surface, lat, lon, dt)

    def _cache_when_done(f):
        # Runs even if we already gave up and returned extrapolation below --
        # a slow-but-successful fetch (e.g. widened SST search finishing at
        # 15s when we only waited 12s) still gets cached, so the NEXT click
        # near this point comes back instantly instead of retrying from
        # scratch. This is what makes retries/widening worth the extra time
        # instead of just moving the timeout problem around.
        try:
            r = f.result()
        except Exception:
            return
        if r is not None:
            if len(_LIVE_FETCH_CACHE) >= _LIVE_FETCH_CACHE_MAX:
                _LIVE_FETCH_CACHE.pop(next(iter(_LIVE_FETCH_CACHE)))
            _LIVE_FETCH_CACHE[cache_key] = r

    future.add_done_callback(_cache_when_done)
    try:
        result = future.result(timeout=timeout)
    except Exception as e:
        logger.info("Live surface fetch still running in background for (%.2f, %.2f) past %.0fs cap: %s", lat, lon, timeout, e)
        result = None
    finally:
        pool.shutdown(wait=False)

    return result


# ── Persistent live-observation pool ─────────────────────────────
# _LIVE_FETCH_CACHE above only avoids re-fetching the SAME rounded point
# twice -- it's process memory, gone on restart, and useless to anyone
# clicking a DIFFERENT nearby point. But every successful live fetch is a
# genuine real satellite reading, not synthetic -- so instead of using it
# once and discarding it, we persist it to disk here and treat it as a
# real anchor point for everyone's future clicks near it. Coverage grows
# automatically from ordinary map usage, with no manual re-run of the
# Argo pipeline needed. Kept clearly distinct from the Argo-validated
# training set (see validated_dataset in get_surface_features_anywhere):
# this is live satellite surface data, not a QC'd Argo subsurface
# profile, and it EXPIRES -- SST genuinely changes day to day, so a pool
# reading from a week ago is not the same claim as one from right now.
LIVE_POOL_PATH = Path(__file__).resolve().parent / "data" / "raw" / "surface" / "live_fetch_pool.csv"
LIVE_POOL_COLUMNS = ["lat", "lon", "sst", "ssh", "sss", "sst_source", "fetched_at"]
LIVE_POOL_MATCH_RADIUS_KM = SPATIAL_MATCH_RADIUS_KM   # same trust radius as the Argo dataset
LIVE_POOL_MAX_AGE_HOURS = TEMPORAL_MATCH_WINDOW_HOURS  # 24h -- same staleness rule used elsewhere
LIVE_POOL_RETENTION_DAYS = float(os.getenv("OCEANEMBED_LIVE_POOL_RETENTION_DAYS", "90"))

LIVE_POOL_DTYPES = {
    "lat": "float64", "lon": "float64", "sst": "float64",
    "ssh": "float64", "sss": "float64", "sst_source": "object", "fetched_at": "object",
}

_live_pool_lock = threading.Lock()
# An empty DataFrame built from just `columns=` has no dtype info -- every
# column silently defaults to `object`, which survives later pd.concat
# calls and makes numpy ufuncs (np.radians in _haversine_km) choke on what
# looks like float data. Pinning dtypes up front avoids that trap.
_live_pool_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in LIVE_POOL_DTYPES.items()})


def _load_live_pool() -> None:
    """Load previously-persisted live readings at startup, dropping
    anything past LIVE_POOL_RETENTION_DAYS so the file doesn't grow
    forever with readings too old to ever be reused anyway."""
    global _live_pool_df
    if not LIVE_POOL_PATH.exists():
        return
    try:
        loaded = pd.read_csv(LIVE_POOL_PATH)
        for col in ("lat", "lon", "sst", "ssh", "sss"):
            loaded[col] = pd.to_numeric(loaded[col], errors="coerce")
        loaded["fetched_at"] = pd.to_datetime(loaded["fetched_at"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=LIVE_POOL_RETENTION_DAYS)
        loaded = loaded.dropna(subset=["lat", "lon", "sst", "fetched_at"])
        loaded = loaded[loaded["fetched_at"] >= cutoff]
        _live_pool_df = loaded.reset_index(drop=True)
        logger.info("Loaded %d cached live satellite readings from %s", len(_live_pool_df), LIVE_POOL_PATH)
    except Exception as e:
        logger.warning("Could not load live-fetch pool (%s) -- starting empty: %s", LIVE_POOL_PATH, e)


def _append_to_live_pool(lat: float, lon: float, sst, ssh, sss, sst_source) -> None:
    """Record one successful live fetch: in memory immediately (so the
    very next request can already see it) and appended to disk (so it
    survives a restart)."""
    global _live_pool_df
    row = {
        "lat": lat, "lon": lon, "sst": sst, "ssh": ssh, "sss": sss,
        "sst_source": sst_source,
        "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    with _live_pool_lock:
        _live_pool_df = pd.concat([_live_pool_df, pd.DataFrame([row])], ignore_index=True)
        try:
            LIVE_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
            write_header = not LIVE_POOL_PATH.exists() or LIVE_POOL_PATH.stat().st_size == 0
            pd.DataFrame([row], columns=LIVE_POOL_COLUMNS).to_csv(
                LIVE_POOL_PATH, mode="a", header=write_header, index=False
            )
        except OSError as e:
            logger.warning("Could not persist live-fetch pool entry to %s: %s", LIVE_POOL_PATH, e)


def _nearest_live_pool_match(lat: float, lon: float) -> Optional[dict]:
    """Return the nearest still-fresh pooled live reading within
    LIVE_POOL_MATCH_RADIUS_KM, or None. Lets a click reuse a REAL
    satellite reading someone else's click already fetched nearby,
    instead of triggering a fresh multi-second ERDDAP round-trip (or
    falling all the way through to extrapolation)."""
    with _live_pool_lock:
        pool = _live_pool_df.copy()
    if pool.empty:
        return None

    fetched_at = pd.to_datetime(pool["fetched_at"], utc=True, errors="coerce")
    age_hours = (pd.Timestamp.now(tz="UTC") - fetched_at).dt.total_seconds() / 3600.0
    fresh = pool[age_hours <= LIVE_POOL_MAX_AGE_HOURS].reset_index(drop=True)
    if fresh.empty:
        return None

    dists = _haversine_km(lat, lon, fresh["lat"].values, fresh["lon"].values)
    idx = int(np.argmin(dists))
    nearest_km = float(dists[idx])
    if nearest_km > LIVE_POOL_MATCH_RADIUS_KM:
        return None

    row = fresh.iloc[idx]
    return {
        "sst": float(row["sst"]),
        "ssh": float(row["ssh"]) if pd.notna(row["ssh"]) else None,
        "sss": float(row["sss"]) if pd.notna(row["sss"]) else None,
        "sst_source": row.get("sst_source"),
        "distance_km": nearest_km,
    }


def _load_nemotron_key() -> Optional[str]:
    """Read the Nemotron API key from an env var, then fall back to
    .streamlit/secrets.toml. Returns None if neither is configured."""
    key = os.getenv("NEMOTRON_API_KEY")
    if key:
        return key
    secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        return None
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            logger.warning("secrets.toml present but no TOML parser available (need Python 3.11+ or `tomli`)")
            return None
    try:
        with open(secrets_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("NEMOTRON_API_KEY")
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s: %s", secrets_path, e)
        return None


NEMOTRON_API_KEY = _load_nemotron_key()
nemotron_client = None
if NEMOTRON_API_KEY:
    try:
        from openai import OpenAI
        nemotron_client = OpenAI(base_url=NEMOTRON_BASE_URL, api_key=NEMOTRON_API_KEY)
    except ImportError:
        logger.warning("openai package not installed; Nemotron copilot disabled")

# ── Load model & dataset ─────────────────────────────────────────
# DATA INTEGRITY: we load model.pkl (real) in preference to
# model_demo.pkl (synthetic), but we NEVER blur the two -- DATA_MODE
# below is read straight from the artifact's own data_mode field, not
# guessed from which file happened to load, so the UI can honestly
# label whichever one is actually running.
df = None
test_df = None          # held-out split only (test_sample.csv) -- kept for
                         # anything that specifically wants those exact rows
model = None
real_metrics_df = None
cv_metrics_df = None    # honest cross-validated metrics -- see train_model.py::_cross_validate
DATA_MODE = "missing"   # "real" | "demo_synthetic" | "missing"
SOURCE_DATASET = None

def load_artifacts() -> None:
    """Load the trained model artifact and its matching test dataset.

    Prefers model.pkl (real data) over model_demo.pkl (synthetic), but
    DATA_MODE always comes from the artifact's own data_mode field
    rather than being inferred from which file happened to load, so
    the UI can honestly label whichever one is actually running.

    Deliberately catches broadly (not just OSError/PickleError): a
    model.pkl built with MultiOutputRegressor(LGBMRegressor) needs
    lightgbm importable to unpickle at all, so a missing/broken
    lightgbm install raises ModuleNotFoundError here, not OSError. If
    that (or anything else) goes wrong, we log it clearly and keep the
    server up in a degraded "no model loaded" state instead of letting
    an exception at import time take the whole Flask app down before
    it can even start.
    """
    global df, test_df, model, real_metrics_df, cv_metrics_df, DATA_MODE, SOURCE_DATASET
    artifact = None
    # Explicit, honest opt-in to demo mode even when a real model.pkl is
    # present -- e.g. to show the denser demo map (1000+ clearly-labeled
    # synthetic points across Arabian Sea + Bay of Bengal + wider Indian
    # Ocean) without deleting the real artifact. Off by default: a real
    # model.pkl always wins unless this is explicitly set.
    force_demo = os.getenv("OCEANEMBED_FORCE_DEMO", "false").lower() in ("1", "true", "yes")
    if force_demo:
        logger.warning("OCEANEMBED_FORCE_DEMO is set -- loading model_demo.pkl (synthetic) "
                        "even though model.pkl (real) may also be present.")
    if not force_demo and os.path.exists("model.pkl"):
        try:
            with open("model.pkl", "rb") as f:
                artifact = pickle.load(f)
        except Exception as e:
            logger.error(
                "Could not load model.pkl (%s: %s). If this is "
                "ModuleNotFoundError for lightgbm, run `pip install -r "
                "requirements.txt` in the SAME environment/venv the server "
                "runs in -- a partial or wrong-venv install is the usual "
                "cause. Server will start anyway with predictions disabled.",
                type(e).__name__, e,
            )
    elif os.path.exists("model_demo.pkl"):
        try:
            with open("model_demo.pkl", "rb") as f:
                artifact = pickle.load(f)
        except Exception as e:
            logger.error("Could not load model_demo.pkl (%s: %s). Server will start anyway with predictions disabled.",
                         type(e).__name__, e)
    elif force_demo:
        logger.error("OCEANEMBED_FORCE_DEMO is set but model_demo.pkl doesn't exist yet -- "
                      "run `python train_model.py --demo` first.")

    if artifact is not None:
        model = artifact["model"]
        real_metrics_df = artifact["metrics"]
        # cv_metrics only exists for real-data runs (train_real() computes it;
        # train_demo() does not) -- absent for a demo_synthetic artifact.
        cv_metrics_df = artifact.get("cv_metrics")
        DATA_MODE = artifact.get("data_mode", "missing")
        SOURCE_DATASET = artifact.get("source_dataset")
        # `df` backs the map, dataset-stats, matching-quality, and the
        # predict-point IDW interpolation -- i.e. everything that should
        # reflect the FULL set of validated observations, not just the
        # ~20% GroupShuffleSplit held out for scoring. It previously
        # loaded test_sample.csv (the held-out split only -- 9 of 45 real
        # rows), which is why the map showed 9 dots instead of all 45.
        # test_sample.csv/test_sample_demo.csv are still loaded separately
        # into `test_df`, reserved for anything that specifically wants
        # "exactly the held-out rows the model was scored on".
        full_dataset_path = (
            Path("data") / "dataset" / "train_dataset.parquet" if DATA_MODE == "real"
            else Path("data") / "demo" / "ocean_data_synthetic.csv"
        )
        test_sample_path = "test_sample.csv" if DATA_MODE == "real" else "test_sample_demo.csv"
        try:
            df = (
                pd.read_parquet(full_dataset_path) if full_dataset_path.suffix == ".parquet"
                else pd.read_csv(full_dataset_path)
            )
        except FileNotFoundError:
            logger.error(
                "%s not found even though a %s model loaded -- falling back to %s",
                full_dataset_path, DATA_MODE, test_sample_path,
            )
            try:
                df = pd.read_csv(test_sample_path)
            except FileNotFoundError:
                logger.error("%s not found either", test_sample_path)
        try:
            test_df = pd.read_csv(test_sample_path)
        except FileNotFoundError:
            logger.error("%s not found even though a %s model loaded", test_sample_path, DATA_MODE)

        # Flag which rows of the full dataset are in the held-out test
        # split, so any per-point "predicted vs actual" comparison can
        # tell a fair held-out check apart from a point the model
        # actually trained on (where predicted≈actual would just reflect
        # memorization, not accuracy).
        if df is not None and test_df is not None and {"argo_wmo", "argo_cycle"}.issubset(df.columns) and {"argo_wmo", "argo_cycle"}.issubset(test_df.columns):
            held_out_keys = set(zip(test_df["argo_wmo"], test_df["argo_cycle"]))
            df["in_test_split"] = [
                (r["argo_wmo"], r["argo_cycle"]) in held_out_keys for _, r in df.iterrows()
            ]
        elif df is not None:
            df["in_test_split"] = False
    else:
        logger.warning("No model.pkl or model_demo.pkl loaded. Run train_model.py first.")


try:
    load_artifacts()
except Exception as e:
    # Belt-and-suspenders: load_artifacts() already catches per-file, but if
    # something outside that (e.g. a corrupt test_sample.csv) still throws,
    # fail into a running-but-degraded server rather than not starting at all.
    logger.error("Unexpected error during startup model load (%s: %s) -- starting with predictions disabled.",
                 type(e).__name__, e)
_load_live_pool()

app = Flask(__name__, static_folder="frontend", static_url_path="")
if CORS is not None:
    CORS(app)


# ── Rate limiting + optional API key ────────────────────────────
# No external dependency (works with no internet / no pip install): a
# small in-memory sliding-window counter per client IP. This is a
# known-limitations item from the earlier review -- the API had no
# throttling at all, so one script (accidental or not) could hammer
# every route, including the ones that make outbound satellite-fetch
# or paid Nemotron calls. Good enough for a hackathon demo; NOT a
# substitute for a real gateway/WAF in production.
_RATE_LIMIT_WINDOW_S = 60
_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("OCEANEMBED_RATE_LIMIT_PER_MIN", "60"))
_rate_limit_hits: dict = {}
_rate_limit_lock = threading.Lock()

# Routes that are cheap, static, or polled automatically by the UI on a
# timer -- exempt from the limiter so normal dashboard use never trips it.
_RATE_LIMIT_EXEMPT_PREFIXES = ("/api/status",)


def _client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr or "unknown"


@app.before_request
def _enforce_rate_limit():
    if not request.path.startswith("/api/"):
        return None
    if request.path.startswith(_RATE_LIMIT_EXEMPT_PREFIXES):
        return None
    ip = _client_ip()
    now = time.time()
    with _rate_limit_lock:
        hits = _rate_limit_hits.setdefault(ip, [])
        cutoff = now - _RATE_LIMIT_WINDOW_S
        while hits and hits[0] < cutoff:
            hits.pop(0)
        if len(hits) >= _RATE_LIMIT_MAX_REQUESTS:
            return jsonify({
                "error": "Rate limit exceeded",
                "detail": f"Max {_RATE_LIMIT_MAX_REQUESTS} requests/{_RATE_LIMIT_WINDOW_S}s per client.",
            }), 429
        hits.append(now)
        # Bound total memory even if many distinct IPs hit the server.
        if len(_rate_limit_hits) > 5000:
            _rate_limit_hits.pop(next(iter(_rate_limit_hits)))
    return None


# Optional API key: only enforced when OCEANEMBED_API_KEY is actually set,
# so the default hackathon-demo deployment (no env var configured) behaves
# exactly as before -- this is opt-in hardening, not a breaking change.
_API_KEY = os.getenv("OCEANEMBED_API_KEY")
_API_KEY_EXEMPT_PREFIXES = ("/api/status",)


@app.before_request
def _enforce_api_key():
    if not _API_KEY:
        return None
    if not request.path.startswith("/api/"):
        return None
    if request.path.startswith(_API_KEY_EXEMPT_PREFIXES):
        return None
    supplied = request.headers.get("X-API-Key")
    if supplied != _API_KEY:
        return jsonify({"error": "Missing or invalid X-API-Key header"}), 401
    return None


# Flask's built-in "float" converter doesn't match a leading "-", so
# /api/predict-point/<float:lat>/<float:lon> (and the two similar routes
# below) previously 404'd on any negative latitude or longitude -- i.e.
# the whole Southern Hemisphere and roughly the whole Western Hemisphere
# never even reached the domain/land logic. Predicting "everywhere"
# requires fixing this route matching too, not just the domain check.
class SignedFloatConverter(BaseConverter):
    regex = r"-?\d+(\.\d+)?"

    def to_python(self, value):
        return float(value)

    def to_url(self, value):
        return str(value)


app.url_map.converters["float"] = SignedFloatConverter

# ── Helpers ──────────────────────────────────────────────────────

def _date_from_day_of_year(day_of_year: int) -> datetime.datetime:
    """Convert a 1-indexed day-of-year into a calendar date within the
    dataset's REFERENCE_YEAR."""
    return datetime.datetime(REFERENCE_YEAR, 1, 1) + datetime.timedelta(days=day_of_year - 1)


def compute_prediction(row: pd.Series) -> tuple[dict, dict, dict]:
    """Run the model on a dataset row and return (predicted, actual, errors)
    dicts keyed by depth label."""
    lat = float(row["lat"])
    lon = float(row["lon"])
    day_of_year = int(row["day_of_year"])
    sst = float(row["sst"])
    ssh = float(row["ssh"])
    sss = float(row["sss"])
    features = pd.DataFrame([[lat, lon, day_of_year, sst, ssh, sss]], columns=FEATURE_COLUMNS)
    pred = model.predict(features)[0]
    predicted = {
        "surface": round(sst, 2),
        "50m": round(float(pred[0]), 2),
        "100m": round(float(pred[1]), 2),
        "200m": round(float(pred[2]), 2),
        "500m": round(float(pred[3]), 2),
    }
    actual = {
        "surface": round(sst, 2),
        "50m": round(float(row.get("temp_50m", 0)), 2),
        "100m": round(float(row.get("temp_100m", 0)), 2),
        "200m": round(float(row.get("temp_200m", 0)), 2),
        "500m": round(float(row.get("temp_500m", 0)), 2),
    }
    errors = {}
    for depth in ["50m", "100m", "200m", "500m"]:
        errors[depth] = round(abs(predicted[depth] - actual[depth]), 2)
    return predicted, actual, errors


def compute_insight(predicted: dict) -> dict:
    """Identify the depth interval with the steepest temperature gradient
    and classify it as a Weak/Moderate/Strong thermal signal."""
    depths = [0, 50, 100, 200, 500]
    temps = [predicted["surface"], predicted["50m"], predicted["100m"],
             predicted["200m"], predicted["500m"]]
    gradients = []
    for i in range(len(depths) - 1):
        dz = depths[i + 1] - depths[i]
        dt = temps[i + 1] - temps[i]
        gradients.append(abs(dt / dz))
    max_idx = int(np.argmax(gradients))
    max_g = gradients[max_idx]
    if max_g >= 0.05:
        level, title, indication = "Strong", "Strong thermal gradient detected", "Enhanced stratification"
    elif max_g >= 0.025:
        level, title, indication = "Moderate", "Moderate thermal gradient detected", "Possible stratification"
    else:
        level, title, indication = "Weak", "Weak thermal gradient detected", "Relatively mixed water column"
    return {"title": title, "indication": indication, "level": level,
            "depthStart": depths[max_idx], "depthEnd": depths[max_idx + 1],
            "gradient": round(max_g, 4)}


def compute_consistency(predicted: dict) -> dict:
    """Flag temperature inversions (depth where temp increases rather than
    decreases) in the predicted profile, which would indicate a modeling
    artifact rather than physically expected stratification."""
    temps = [predicted["surface"], predicted["50m"], predicted["100m"],
             predicted["200m"], predicted["500m"]]
    depths = [0, 50, 100, 200, 500]
    changes = [temps[i + 1] - temps[i] for i in range(len(temps) - 1)]
    inversions = [i for i, c in enumerate(changes) if c > 0]
    max_jump = max(abs(c) for c in changes)
    if len(inversions) == 0:
        return {"status": "Consistent", "message": "Temperature decreases continuously with depth.", "maxJump": round(max_jump, 2)}
    return {"status": "Review", "message": f"Temperature inversion between {depths[inversions[0]]}m and {depths[inversions[0] + 1]}m.", "maxJump": round(max_jump, 2)}


def compute_tchp(predicted: dict) -> tuple[float, str, str]:
    """Tropical Cyclone Heat Potential proxy shared by /api/disaster-risk,
    /api/predict-point and /api/cyclone-track: depth of the isotherm
    defined by TCHP_ISOTHERM_C (via linear interpolation of the predicted
    profile) + a risk tier. Returns (depth_m, risk_level, risk_note)."""
    depths = [0, 50, 100, 200, 500]
    temps = [predicted["surface"], predicted["50m"], predicted["100m"], predicted["200m"], predicted["500m"]]
    isotherm = TCHP_ISOTHERM_C
    d26 = None
    for i in range(len(depths) - 1):
        t0, t1 = temps[i], temps[i + 1]
        if (t0 - isotherm) * (t1 - isotherm) <= 0 and t0 != t1:
            frac = (isotherm - t0) / (t1 - t0)
            d26 = depths[i] + frac * (depths[i + 1] - depths[i])
            break
    if d26 is None:
        d26 = 500.0 if temps[-1] >= isotherm else 0.0

    if d26 >= 100:
        risk_level, risk_note = "Elevated", "Deep 26°C isotherm indicates a large warm-water reservoir available to fuel storm intensification if a cyclone tracks over this point."
    elif d26 >= 50:
        risk_level, risk_note = "Moderate", "Moderate warm-water depth; some potential to sustain cyclone intensification."
    else:
        risk_level, risk_note = "Low", "Shallow warm layer; limited heat reserve for cyclone intensification at this location."
    return round(d26, 1), risk_level, risk_note


def compute_mixed_layer_depth(predicted: dict) -> tuple[float, str, str]:
    """Mixed Layer Depth (MLD) proxy: the shallowest depth at which the
    predicted temperature has dropped 0.5°C below the surface value --
    a simplified version of the standard de Boyer Montegut et al. (2004)
    temperature-threshold MLD criterion (their default threshold is
    0.2°C; 0.5°C is used here because our depth grid is coarse -- 0, 50,
    100, 200, 500m -- and a 0.2°C threshold would almost always resolve
    to "shallower than our first sample", which would overstate
    precision we don't have).

    Disaster-management relevance: a shallow mixed layer means the
    surface is thin and wind-responsive -- more useful for search-and-
    rescue drift and oil-spill dispersion forecasting, and it also means
    a passing cyclone can entrain/upwell cooler water more easily,
    which tends to self-limit further intensification (see
    compute_cold_wake_potential below, which builds on this).
    """
    depths = [0, 50, 100, 200, 500]
    temps = [predicted["surface"], predicted["50m"], predicted["100m"], predicted["200m"], predicted["500m"]]
    threshold = predicted["surface"] - 0.5
    mld = None
    for i in range(len(depths) - 1):
        t0, t1 = temps[i], temps[i + 1]
        if t0 >= threshold and t1 < threshold and t0 != t1:
            frac = (t0 - threshold) / (t0 - t1)
            mld = depths[i] + frac * (depths[i + 1] - depths[i])
            break
    if mld is None:
        mld = 500.0 if temps[-1] >= threshold else 0.0

    if mld < 30:
        level, note = "Shallow", "Shallow mixed layer -- surface conditions respond quickly to wind and heating; drift/dispersion forecasts should use a thin near-surface layer."
    elif mld < 80:
        level, note = "Moderate", "Moderate mixed-layer depth, typical of open-ocean conditions away from strong upwelling or heavy freshwater influence."
    else:
        level, note = "Deep", "Deep mixed layer -- a thick layer of near-uniform warm water, which also means more heat available near the surface for cyclone intensification."
    return round(mld, 1), level, note


def compute_thermocline_index(predicted: dict) -> tuple[float, str, str, str]:
    """Thermocline steepness: the steepest°C-per-meter drop across the
    four depth bins (0-50, 50-100, 100-200, 200-500m) of the predicted
    profile, and which bin it occurs in.

    Disaster-management relevance: a sharp, shallow thermocline is a
    strong barrier to vertical mixing -- a cyclone crossing it can't
    easily draw up cold water to cool itself, so intensification is
    LESS self-limited (elevated risk). A weak/deep thermocline lets
    cold water reach the surface easily under wind-driven mixing --
    the storm tends to cool its own fuel source and self-limit
    (lower risk). This is the same physical mechanism NOAA/CIMSS use
    ocean heat content for operationally; this is a simplified proxy,
    not their actual model.
    """
    bins = [(0, 50), (50, 100), (100, 200), (200, 500)]
    vals = [predicted["surface"], predicted["50m"], predicted["100m"], predicted["200m"], predicted["500m"]]
    steepest = 0.0
    steepest_bin = bins[0]
    for i, (d0, d1) in enumerate(bins):
        grad = abs(vals[i] - vals[i + 1]) / (d1 - d0)
        if grad > steepest:
            steepest = grad
            steepest_bin = (d0, d1)

    if steepest >= 0.08:
        level = "Sharp"
        note = ("Sharp thermocline -- strong barrier to vertical mixing, so a cyclone here has less "
                "self-limiting cold-water feedback than usual. Combine with Cyclone Heat Potential above.")
    elif steepest >= 0.03:
        level = "Moderate"
        note = "Moderate stratification -- typical open-ocean thermocline strength."
    else:
        level = "Weak"
        note = ("Weak stratification -- wind-driven mixing can reach relatively deep water easily, which "
                "tends to cool the surface and self-limit cyclone intensification.")
    depth_range = f"{steepest_bin[0]}-{steepest_bin[1]}m"
    return round(steepest, 3), depth_range, level, note


def compute_cold_wake_potential(predicted: dict, mld_m: float) -> tuple[float, str, str]:
    """Cold-wake / self-limiting potential: an approximation of how much
    a cyclone's own wind-driven mixing could cool the surface at this
    point, estimated as the temperature difference between the surface
    and the water just below the mixed layer (using 100m as a fixed
    proxy for typical cyclone-driven mixing depth -- real mixing depth
    varies with storm intensity/translation speed, which this model has
    no way to know, so this is a fixed-depth approximation, not a
    forecast of an actual storm's cold wake).

    Disaster-management relevance: forecasters already use SST cooling
    ("cold wake") as a factor in intensity-forecast busts -- a storm
    that cools its own fuel source tends to weaken, and so does any
    subsequent storm crossing the same wake before it recovers. High
    cooling potential here is a PROTECTIVE signal (self-limiting); low
    cooling potential means the warm layer is deep/uniform enough that
    the storm can sustain intensity longer.
    """
    cooling_estimate = round(predicted["surface"] - predicted["100m"], 2)
    if cooling_estimate >= 4.0:
        level = "High (self-limiting)"
        note = ("Large surface-to-100m temperature drop -- wind-driven mixing here could cool the "
                "surface substantially, tending to weaken the storm and reduce risk to any system "
                "following the same track soon after.")
    elif cooling_estimate >= 1.5:
        level = "Moderate"
        note = "Some self-limiting cooling potential, but not enough to reliably weaken a strong system."
    else:
        level = "Low (sustaining)"
        note = ("Small surface-to-100m temperature difference -- little cold water available to mix up, "
                "so a cyclone here is less likely to weaken itself through cold-wake feedback.")
    return cooling_estimate, level, note


def compute_secondary_indicators(predicted: dict) -> dict:
    """Bundle the three additional subsurface-derived disaster indicators
    (beyond Cyclone Heat Potential) for the API response. All three are
    computed purely from the model's own predicted depth profile -- no
    additional data source is required."""
    mld_m, mld_level, mld_note = compute_mixed_layer_depth(predicted)
    grad, grad_range, grad_level, grad_note = compute_thermocline_index(predicted)
    cooling, cooling_level, cooling_note = compute_cold_wake_potential(predicted, mld_m)
    return {
        "mixedLayerDepth": {
            "depth_m": mld_m, "level": mld_level, "note": mld_note,
            "label": "Mixed Layer Depth",
        },
        "thermoclineIndex": {
            "gradient_c_per_m": grad, "depthRange": grad_range, "level": grad_level, "note": grad_note,
            "label": "Thermocline Steepness",
        },
        "coldWakePotential": {
            "cooling_estimate_c": cooling, "level": cooling_level, "note": cooling_note,
            "label": "Cold-Wake / Self-Limiting Potential",
        },
    }


def compute_broader_applications(predicted: dict, d26: float, insight: dict) -> dict:
    """The PS is filed under 'Disaster Management', so TCHP/cyclone risk is
    the headline number -- but the underlying reconstructed profile
    (surface -> 500m) is generically useful ocean-state information, not a
    cyclone-only artifact. This turns the same profile into three more
    reconstruction-derived reads that don't need any new model or data:
    fisheries habitat (thermocline depth drives pelagic fish aggregation),
    underwater acoustics/defense (thermocline depth bends sonar rays --
    classic 'thermocline shadow zone' problem for ASW), and a climate/
    heat-content proxy (warm-layer thickness as an ocean-warming signal).
    Not new science -- just naming what a subsurface temperature profile
    already implies, the same way compute_tchp() does for cyclones."""
    thermocline_depth = (insight["depthStart"] + insight["depthEnd"]) / 2

    if thermocline_depth <= 60:
        fisheries_note = ("Shallow thermocline — nutrient-rich water is close to the "
                           "surface, typically a productive zone for pelagic fish "
                           "(tuna, sardine) aggregation and a useful signal for "
                           "fishery advisories.")
        fisheries_tier = "Favorable"
    elif thermocline_depth <= 120:
        fisheries_note = ("Moderate thermocline depth — fish habitat is likely "
                           "compressed into a mid-depth band rather than near-surface.")
        fisheries_tier = "Moderate"
    else:
        fisheries_note = ("Deep, weak thermocline — surface waters are less "
                           "nutrient-rich; pelagic aggregation near the surface is "
                           "less likely here.")
        fisheries_tier = "Limited"

    if thermocline_depth <= 80:
        defense_note = ("Shallow thermocline bends sonar rays downward close to the "
                         "surface, creating a 'shadow zone' where a submarine below "
                         "the layer is harder to detect from surface sonar — relevant "
                         "to underwater surveillance/ASW planning.")
    else:
        defense_note = ("Deeper, more gradual thermocline — sound propagation is "
                         "closer to isothermal, giving more uniform sonar detection "
                         "range with less shadow-zone effect.")

    warm_layer_m = d26
    if warm_layer_m >= 100:
        climate_note = (f"Warm layer extends to ~{warm_layer_m:.0f} m — a thick "
                         "upper-ocean heat reservoir, the kind of signal used "
                         "(aggregated over time) to track ocean heat content and "
                         "monitor long-term warming trends.")
    else:
        climate_note = (f"Warm layer is thin (~{warm_layer_m:.0f} m) at this point and "
                         "time — less upper-ocean heat storage here right now.")

    return {
        "thermoclineDepth_m": round(thermocline_depth, 1),
        "fisheries": {"tier": fisheries_tier, "note": fisheries_note},
        "defenseAcoustics": {"note": defense_note},
        "climateMonitoring": {"note": climate_note},
        "disclaimer": "Illustrative reads derived from the same reconstructed profile, "
                       "for demo purposes — not calibrated fishery, acoustic, or "
                       "climate models.",
    }


# ── Serve frontend ───────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the dashboard's entry point."""
    return send_from_directory("frontend", "index.html")

@app.route("/<path:path>")
def static_files(path):
    """Serve the rest of the static frontend (JS/CSS/assets)."""
    return send_from_directory("frontend", path)


# ── API: Dataset Statistics ──────────────────────────────────────
@app.route("/api/dataset-stats")
def dataset_stats():
    """Summary statistics (ranges, counts, date span) for the loaded dataset,
    restricted to ocean-only locations."""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    # Count only ocean locations (land-filtered)
    unique_locs = df[["lat", "lon"]].drop_duplicates()
    all_points = [(float(r["lat"]), float(r["lon"])) for _, r in unique_locs.iterrows()]
    ocean_points, _ = filter_ocean_points(all_points)
    ocean_set = set(ocean_points)
    n_locs = len(ocean_points)
    n_obs = len(df[df.apply(lambda r: (float(r["lat"]), float(r["lon"])) in ocean_set, axis=1)])
    n_cols = 4  # target depths

    # Stats from ocean-only rows
    df_ocean = df[df.apply(lambda r: (float(r["lat"]), float(r["lon"])) in ocean_set, axis=1)]
    lat_min = round(float(df_ocean["lat"].min()), 4)
    lat_max = round(float(df_ocean["lat"].max()), 4)
    lon_min = round(float(df_ocean["lon"].min()), 4)
    lon_max = round(float(df_ocean["lon"].max()), 4)

    doy_min, doy_max = int(df_ocean["day_of_year"].min()), int(df_ocean["day_of_year"].max())
    date_min = _date_from_day_of_year(doy_min).strftime("%Y-%m-%d")
    date_max_raw = _date_from_day_of_year(doy_max)
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    date_max = min(date_max_raw, today).strftime("%Y-%m-%d")

    sst_min, sst_max, sst_mean = round(float(df_ocean["sst"].min()), 2), round(float(df_ocean["sst"].max()), 2), round(float(df_ocean["sst"].mean()), 2)
    ssh_min, ssh_max, ssh_mean = round(float(df_ocean["ssh"].min()), 4), round(float(df_ocean["ssh"].max()), 4), round(float(df_ocean["ssh"].mean()), 4)
    sss_min, sss_max, sss_mean = round(float(df_ocean["sss"].min()), 2), round(float(df_ocean["sss"].max()), 2), round(float(df_ocean["sss"].mean()), 2)

    if DATA_MODE == "real" and "argo_wmo" in df.columns:
        unique_floats = int(df_ocean["argo_wmo"].nunique()) if "argo_wmo" in df_ocean.columns else int(df["argo_wmo"].nunique())
        profile_cycles = int(df_ocean["argo_cycle"].nunique()) if "argo_cycle" in df_ocean.columns else int(df["argo_cycle"].nunique())
    else:
        # No real provenance in demo mode -- report honestly rather than a
        # capped placeholder number that used to be shown regardless of mode.
        unique_floats = None
        profile_cycles = None

    return jsonify({
        "totalRows": n_obs,
        "uniqueLocations": n_locs,
        "uniqueFloats": unique_floats,
        "profileCycles": profile_cycles,
        "targetDepths": n_cols,
        "latRange": {"min": lat_min, "max": lat_max},
        "lonRange": {"min": lon_min, "max": lon_max},
        "dateRange": {"min": date_min, "max": date_max},
        "sstRange": {"min": sst_min, "max": sst_max, "mean": sst_mean},
        "sshRange": {"min": ssh_min, "max": ssh_max, "mean": ssh_mean},
        "sssRange": {"min": sss_min, "max": sss_max, "mean": sss_mean},
        "datasetName": SOURCE_DATASET or "none",
        "region": "Arabian Sea / North Indian Ocean",
        "dataMode": DATA_MODE,
    })


# ── API: Observations ────────────────────────────────────────────
@app.route("/api/observations")
def get_observations():
    """List every unique ocean observation location in the loaded dataset."""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    unique_locs = df[["lat", "lon"]].drop_duplicates()

    # ── Apply land mask: only show ocean observations ───────────
    all_points = [(float(r["lat"]), float(r["lon"])) for _, r in unique_locs.iterrows()]
    ocean_points, land_points = filter_ocean_points(all_points)
    logger.info("Land mask: %d total -> %d ocean, %d land removed", len(all_points), len(ocean_points), len(land_points))
    ocean_set = set(ocean_points)

    obs = []
    idx = 0
    for _, row in unique_locs.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        if (lat, lon) not in ocean_set:
            continue
        matches = df[(df["lat"] == lat) & (df["lon"] == lon)]
        first = matches.iloc[0]
        day_of_year = int(first["day_of_year"])
        profile_date = _date_from_day_of_year(day_of_year).strftime("%Y-%m-%d")
        obs.append({
            "id": idx,
            "lat": lat,
            "lon": lon,
            "date": profile_date,
            "dayOfYear": day_of_year,
            "sst": round(float(first["sst"]), 2),
            "count": len(matches),
        })
        idx += 1
    return jsonify({"observations": obs, "total": len(obs)})


@app.route("/api/observation/<float:lat>/<float:lon>")
def get_observation(lat, lon):
    """Full detail (provenance, model prediction, insight) for one observation."""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    matches = df[(df["lat"] == lat) & (df["lon"] == lon)]
    if matches.empty:
        return jsonify({"error": "Observation not found"}), 404

    row = matches.iloc[0]
    day_of_year = int(row["day_of_year"])
    profile_date = _date_from_day_of_year(day_of_year).strftime("%Y-%m-%d")

    # Provenance (argoWMO, cycle, real matching distance/time, real source
    # labels) is ONLY available when this row actually came from the real
    # pipeline's output columns. It used to be fabricated deterministically
    # from lat/lon (a formula dressed up to look like a real WMO/cycle ID)
    # regardless of data mode -- that's removed. In demo mode we say so
    # explicitly instead of inventing plausible-looking IDs.
    has_provenance = DATA_MODE == "real" and "argo_wmo" in df.columns

    result = {
        "lat": lat, "lon": lon,
        "dayOfYear": day_of_year,
        "date": profile_date,
        "dataMode": DATA_MODE,
    }

    if has_provenance:
        result["profileTime"] = str(row.get("profile_time", profile_date))
        result["argoWMO"] = str(row.get("argo_wmo"))
        result["cycle"] = int(row.get("argo_cycle", 0))
        result["matching"] = {
            "distanceKm": round(float(row.get("surface_distance_km", 0)), 2),
            "timeDiffHours": round(float(row.get("surface_time_diff_hours", 0)), 2),
            "maxDistanceKm": SPATIAL_MATCH_RADIUS_KM,
            "maxTimeDiffHours": TEMPORAL_MATCH_WINDOW_HOURS,
            "violations": int(
                row.get("surface_distance_km", 0) > SPATIAL_MATCH_RADIUS_KM
                or row.get("surface_time_diff_hours", 0) > TEMPORAL_MATCH_WINDOW_HOURS
            ),
        }
        surface = {
            "sst": {"value": round(float(row["sst"]), 2), "unit": "°C", "source": str(row.get("sst_source", "unknown")), "classification": "SOURCE"},
            "ssh": {"value": round(float(row["ssh"]), 4), "unit": "m", "source": str(row.get("ssh_source", "unknown")), "classification": "SOURCE"},
            "sss": {"value": round(float(row["sss"]), 2), "unit": "PSU", "source": str(row.get("sss_source", "unknown")), "classification": "MATCHED"},
        }
    else:
        result["profileTime"] = f"{profile_date} (demo data — no real profile timestamp)"
        result["argoWMO"] = "N/A (demo data)"
        result["cycle"] = None
        result["matching"] = None
        surface = {
            "sst": {"value": round(float(row["sst"]), 2), "unit": "°C", "source": "Synthetic (demo)", "classification": "DEMO"},
            "ssh": {"value": round(float(row["ssh"]), 4), "unit": "m", "source": "Synthetic (demo)", "classification": "DEMO"},
            "sss": {"value": round(float(row["sss"]), 2), "unit": "PSU", "source": "Synthetic (demo)", "classification": "DEMO"},
        }
    result["surface"] = surface
    result["modelInputs"] = {
        "lat": lat, "lon": lon, "dayOfYear": day_of_year,
        "sst": round(float(row["sst"]), 2),
        "ssh": round(float(row["ssh"]), 4),
        "sss": round(float(row["sss"]), 2),
    }

    if model is not None:
        try:
            predicted, actual, errors = compute_prediction(row)
            in_test_split = bool(row.get("in_test_split", False))
            result["predicted"] = predicted
            result["actual"] = actual
            result["inTestSplit"] = in_test_split
            if in_test_split:
                # A fair comparison: the model never saw this point during
                # training, so predicted-vs-actual here is real evidence
                # of accuracy (see the crossValidated numbers on the Model
                # Validation tab for the aggregate version of this check).
                result["errors"] = errors
                result["accuracyNote"] = ("This point was held out from training, so the "
                                           "error above is a genuine accuracy check.")
            else:
                # This point WAS used to train the model -- showing a
                # near-perfect predicted/actual match here would look like
                # accuracy but would really just be memorization. Still
                # show the model's prediction (useful on its own) but
                # don't present the gap as an accuracy signal.
                result["errors"] = None
                result["accuracyNote"] = ("This point was used to TRAIN the model, so its "
                                           "predicted-vs-actual gap is not a fair accuracy "
                                           "check -- see the Model Validation tab's "
                                           "cross-validated numbers instead.")
            result["insight"] = compute_insight(predicted)
            result["consistency"] = compute_consistency(predicted)
            result["summary"] = {
                "surfaceTemp": predicted["surface"],
                "temp50": predicted["50m"],
                "temp100": predicted["100m"],
                "temp200": predicted["200m"],
                "temp500": predicted["500m"],
                "thermalGradient": round(abs(predicted["surface"] - predicted["500m"]) / 5.0, 2),
                "deepestLevel": "500m",
                "model": "Multi-output LightGBM",
            }
        except Exception as e:
            result["predictionError"] = str(e)

    return jsonify(result)


# ── API: Model Metrics ──────────────────────────────────────────
@app.route("/api/metrics")
def get_metrics():
    """Per-depth held-out test metrics AND the cross-validated metrics for
    the loaded model, both clearly labeled.

    IMPORTANT (see train_model.py::_cross_validate): with ~45 rows a single
    80/20 holdout tests on ~9 points, so 'holdout' R2/RMSE can swing widely
    on one unusual float and is NOT the number to trust for an accuracy
    claim. 'crossValidated' averages over several held-out folds and is the
    honest number -- this endpoint returns both, but any report/README/demo
    claim should quote crossValidated, not holdout.
    """
    if real_metrics_df is None:
        return jsonify({"error": "Metrics not available"}), 500
    metrics = real_metrics_df.to_dict(orient="records")
    rmse_values = [m["RMSE (\u00b0C)"] for m in metrics]
    r2_values = [m["R\u00b2 Score"] for m in metrics]
    overall = {
        "rmse": round(float(np.mean(rmse_values)), 3),
        "r2": round(float(np.mean(r2_values)), 3),
    }
    # MAE is only reported when the training run actually computed it
    # (see train_model.py). It is never derived from RMSE by a fixed
    # ratio -- that produces a plausible-looking but fabricated number.
    mae_key = "MAE (\u00b0C)"
    if metrics and mae_key in metrics[0]:
        overall["mae"] = round(float(np.mean([m[mae_key] for m in metrics])), 3)

    cv_metrics = None
    cv_overall = None
    if cv_metrics_df is not None:
        cv_metrics = cv_metrics_df.to_dict(orient="records")
        cv_overall = {
            "rmse": round(float(np.mean([m["RMSE (\u00b0C)"] for m in cv_metrics])), 3),
            "r2": round(float(np.mean([m["R\u00b2 Score"] for m in cv_metrics])), 3),
            "mae": round(float(np.mean([m["MAE (\u00b0C)"] for m in cv_metrics])), 3),
        }

    return jsonify({
        "depthMetrics": metrics,
        "overall": overall,
        "crossValidated": {
            "available": cv_metrics is not None,
            "depthMetrics": cv_metrics,
            "overall": cv_overall,
            "note": ("5-fold (grouped by Argo float) cross-validated metrics -- "
                     "this is the trustworthy accuracy number, not the single "
                     "holdout split above."),
        },
        "trustedMetric": "crossValidated" if cv_metrics is not None else "depthMetrics",
    })


# ── API: Feature Importance ─────────────────────────────────────
@app.route("/api/feature-importance")
def get_feature_importance():
    """Relative feature importance (averaged across the per-depth estimators)."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    feature_names = ["lat", "lon", "day_of_year", "sst", "ssh", "sss"]
    try:
        importances = [est.feature_importances_ for est in model.estimators_]
        avg = np.mean(importances, axis=0)
        pct = (avg / avg.sum()) * 100
        result = sorted(
            [{"name": n, "value": round(float(v), 1)} for n, v in zip(feature_names, pct)],
            key=lambda x: x["value"],
        )
        return jsonify({"features": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Matching Quality ───────────────────────────────────────
@app.route("/api/matching-quality")
def get_matching_quality():
    """Spatial/temporal matching-quality stats for the real Argo-satellite
    pairing, or an honest 'unavailable' response in demo mode."""
    # Computed from the ACTUAL loaded dataset -- these numbers used to be
    # hardcoded placeholders regardless of what data was loaded. If the
    # provenance columns aren't present (e.g. running in demo mode), we
    # say so plainly instead of showing fabricated statistics.
    required = {"surface_distance_km", "surface_time_diff_hours"}
    if df is None or DATA_MODE != "real" or not required.issubset(df.columns):
        return jsonify({
            "available": False,
            "reason": "demo_data" if DATA_MODE == "demo_synthetic" else "no_provenance_columns",
            "message": "Matching-quality statistics require the real, validated "
                       "Argo/satellite pipeline output. This build is currently "
                       f"running in '{DATA_MODE}' mode.",
        })

    dist = df["surface_distance_km"].dropna()
    tdiff = df["surface_time_diff_hours"].dropna()
    violations = int(((dist > SPATIAL_MATCH_RADIUS_KM) | (tdiff > TEMPORAL_MATCH_WINDOW_HOURS)).sum())
    return jsonify({
        "available": True,
        "constraints": {"maxDistanceKm": SPATIAL_MATCH_RADIUS_KM, "maxTimeDiffHours": TEMPORAL_MATCH_WINDOW_HOURS},
        "observed": {
            "meanDistanceKm": round(float(dist.mean()), 2) if len(dist) else None,
            "maxDistanceKm": round(float(dist.max()), 2) if len(dist) else None,
            "meanTimeDiffHours": round(float(tdiff.mean()), 2) if len(tdiff) else None,
            "maxTimeDiffHours": round(float(tdiff.max()), 2) if len(tdiff) else None,
            "violations": violations,
        },
        "compliance": f"{round(100 * (1 - violations / max(len(df), 1)), 1)}%",
    })


# ── API: System Status ──────────────────────────────────────────
@app.route("/api/status")
def get_status():
    """Report which data/model mode this build is currently running in."""
    labels = {
        "real": "Real Argo + Satellite Data (Validated)",
        "demo_synthetic": "Synthetic Demo Data (NOT real observations)",
        "missing": "No model trained yet",
    }
    return jsonify({
        "dataPipeline": {
            "status": "Ready" if df is not None else "Unavailable",
            "active": df is not None,
            "mode": DATA_MODE,
            "label": labels.get(DATA_MODE, DATA_MODE),
        },
        "mlModel": {"status": "Ready" if model else "Unavailable", "active": model is not None},
        "nemotron": {"status": "Ready" if nemotron_client else "Offline", "active": nemotron_client is not None},
        "datasetName": SOURCE_DATASET or "none",
        "lastProcessed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    })


# ── API: Disaster-Management Risk Index ───────────────────────────
# Tropical Cyclone Heat Potential (TCHP) is a real, operationally-used
# oceanographic metric (NOAA/CIMSS) linking upper-ocean warm-water volume
# to cyclone rapid-intensification risk. We approximate it here from the
# predicted profile: depth of the 26°C isotherm (D26) via linear
# interpolation, plus a simple heat-content proxy above it. This is what
# ties the model's output back to the "Disaster Management" PS category
# instead of just showing a temperature curve.
@app.route("/api/disaster-risk/<float:lat>/<float:lon>")
def disaster_risk(lat, lon):
    """Tropical Cyclone Heat Potential risk tier for an existing dataset point."""
    if df is None or model is None:
        return jsonify({"error": "Model/dataset not loaded"}), 500
    match = df[(df["lat"] == lat) & (df["lon"] == lon)]
    if match.empty:
        return jsonify({"error": "No observation at that location"}), 404
    row = match.iloc[0]
    predicted, actual, errors = compute_prediction(row)

    d26, risk_level, risk_note = compute_tchp(predicted)
    broader = compute_broader_applications(predicted, d26, compute_insight(predicted))
    secondary = compute_secondary_indicators(predicted)

    return jsonify({
        "location": {"lat": lat, "lon": lon},
        "isotherm26Depth_m": d26,
        "riskLevel": risk_level,
        "note": risk_note,
        "disclaimer": "Educational approximation of Tropical Cyclone Heat Potential "
                       "(TCHP) derived from the model's predicted profile. Not a "
                       "substitute for official IMD/JTWC cyclone forecasts.",
        "broaderApplications": broader,
        "secondaryIndicators": secondary,
    })


# ── Predict anywhere in the ocean, not just the Arabian Sea box ────
# GAP THIS CLOSES: earlier builds hard-rejected any click outside a
# bounding box around the (small, Arabian-Sea-only) training dataset --
# "Location is outside the validated data domain". That was honest about
# the training set's extent, but it made the map effectively unusable
# everywhere else. This version instead layers three sources, in order
# of trust, and is explicit in the response about which one answered:
#
#   1. VALIDATED  -- within SPATIAL_MATCH_RADIUS_KM of a real Argo/
#                    satellite-matched observation already in the dataset.
#   2. LIVE        -- outside that radius, so we fetch a real, live SST/
#                    SSH/SSS reading for the clicked point from NASA/NOAA/
#                    ESA ERDDAP servers (data/surface_fetch.py) and run
#                    the model on genuine current conditions there.
#   3. EXTRAPOLATED -- live fetch unavailable (offline dev machine, no
#                    satellite pass in the time window, endpoint down) --
#                    falls back to inverse-distance-weighted extrapolation
#                    from the nearest validated observations, clearly
#                    labeled as such with a downgraded confidence tier
#                    rather than silently pretending precision it doesn't
#                    have or refusing to answer at all.
#
# Points on land are still rejected (there is no subsurface ocean profile
# to predict there) -- see is_ocean() below, which uses a global land
# mask, not the old dataset-shaped box.
def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def interpolate_surface_features(lat, lon, k=4):
    """IDW-interpolate sst/ssh/sss at an arbitrary point from the k nearest
    loaded observations. Returns (sst, ssh, sss, nearest_km, n_used)."""
    locs = df[["lat", "lon", "sst", "ssh", "sss"]].drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    dists = _haversine_km(lat, lon, locs["lat"].values, locs["lon"].values)
    order = np.argsort(dists)[:min(k, len(locs))]
    nearest_km = float(dists[order[0]])

    if nearest_km < 0.5:
        row = locs.iloc[order[0]]
        return float(row["sst"]), float(row["ssh"]), float(row["sss"]), nearest_km, 1

    w = 1.0 / (dists[order] ** 2 + 1e-6)
    w = w / w.sum()
    sst = float(np.sum(w * locs.iloc[order]["sst"].values))
    ssh = float(np.sum(w * locs.iloc[order]["ssh"].values))
    sss = float(np.sum(w * locs.iloc[order]["sss"].values))
    return sst, ssh, sss, nearest_km, len(order)


def get_surface_features_anywhere(lat: float, lon: float, target_dt) -> dict:
    """Resolve sst/ssh/sss for ANY ocean point on Earth, preferring the most
    trustworthy source available (see module docstring above). Always
    returns a result -- never raises, never hard-fails a request.

    Order of preference: (1) Argo-validated dataset, (2) a still-fresh
    reading someone else's click already pulled from live satellite data
    nearby (see the persistent live pool above -- this is what lets real
    coverage grow from ordinary usage instead of staying fixed at whatever
    the last offline pipeline run produced), (3) a brand-new live fetch for
    this exact point, (4) honest statistical extrapolation."""
    sst, ssh, sss, nearest_km, n_used = interpolate_surface_features(lat, lon)

    if nearest_km <= SPATIAL_MATCH_RADIUS_KM:
        return {
            "sst": sst, "ssh": ssh, "sss": sss,
            "source": "validated_dataset",
            "nearestObsDistanceKm": nearest_km, "nObsUsed": n_used,
        }

    pooled = _nearest_live_pool_match(lat, lon)
    if pooled is not None:
        return {
            "sst": pooled["sst"],
            "ssh": pooled["ssh"] if pooled.get("ssh") is not None else ssh,
            "sss": pooled["sss"] if pooled.get("sss") is not None else sss,
            "source": "live_satellite",
            "nearestObsDistanceKm": pooled["distance_km"], "nObsUsed": 1,
            "sstSource": pooled.get("sst_source"),
            "reusedFromPool": True,
        }

    live = _live_fetch_with_timeout(lat, lon, target_dt)
    if live and live.get("sst") is not None:
        _append_to_live_pool(lat, lon, live["sst"], live.get("ssh"), live.get("sss"), live.get("sst_source"))
        return {
            "sst": live["sst"],
            "ssh": live["ssh"] if live.get("ssh") is not None else ssh,
            "sss": live["sss"] if live.get("sss") is not None else sss,
            "source": "live_satellite",
            "nearestObsDistanceKm": live.get("distance_km") or 0.0,
            "nObsUsed": 1,
            "sstSource": live.get("sst_source"),
        }

    # Live fetch unavailable -- honest fallback, not a failure.
    return {
        "sst": sst, "ssh": ssh, "sss": sss,
        "source": "extrapolated",
        "nearestObsDistanceKm": nearest_km, "nObsUsed": n_used,
    }


def _confidence_from_source(source: str, km: float, reused_from_pool: bool = False) -> tuple[str, str]:
    """Map (data source, distance to nearest validated observation) to an
    honest confidence tier + human-readable note."""
    if source == "live_satellite":
        if reused_from_pool:
            return "High", (f"Live NASA/NOAA/ESA satellite reading from an earlier nearby click, "
                             f"~{km:.0f} km away and still within the freshness window.")
        return "High", "Live NASA/NOAA/ESA satellite reading fetched for this exact point."
    if source == "validated_dataset":
        if km < 15:
            return "High", f"Within {km:.0f} km of a validated Arabian Sea observation."
        return "Medium", f"Interpolated from validated observations ~{km:.0f} km away."
    # extrapolated
    if km < 150:
        return "Low", (f"No live satellite data available right now; statistically "
                        f"extrapolated from an observation {km:.0f} km away — treat as indicative only.")
    return "Very Low", (f"No live satellite data available right now; nearest validated "
                         f"observation is {km:.0f} km away — sparse coverage, low confidence.")


def _day_of_year_from_date(date_str: Optional[str]) -> int:
    """Parse a YYYY-MM-DD date string to day-of-year, falling back to the
    dataset's latest day-of-year if the string is missing or invalid."""
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
    except (TypeError, ValueError):
        return int(df["day_of_year"].max())


def _prediction_uncertainty(source: str, reused_from_pool: bool = False) -> Optional[dict]:
    """Per-depth ± uncertainty band for a prediction, derived from the
    model's own cross-validated RMSE (see train_model.py::_cross_validate),
    not a fabricated number.

    This is a genuine limitation to be upfront about: with ~45 rows, we
    do not have enough data to fit a separate, stable quantile-regression
    model (a 0.1/0.9 quantile LightGBM trained on ~36 rows would itself
    have far higher variance than the point-estimate model) -- so rather
    than ship an unstable interval that looks precise but isn't, this
    uses the CV RMSE per depth as an honest 1-sigma band around the
    point prediction, widened for inputs that are farther from what the
    CV was actually measured on (CV only ever saw validated_dataset-
    quality surface inputs, so an extrapolated input carries additional
    uncertainty the CV number does not capture).
    """
    if cv_metrics_df is None:
        return None
    widen = {"validated_dataset": 1.0, "live_satellite": 1.15 if not reused_from_pool else 1.3,
             "extrapolated": 2.0}.get(source, 1.5)
    band = {}
    for _, row in cv_metrics_df.iterrows():
        depth_label = row["Depth"]
        rmse = float(row["RMSE (\u00b0C)"])
        band[depth_label] = round(rmse * widen, 2)
    return band


def _score_point(lat, lon, day_of_year, target_dt=None):
    """Shared core for /api/predict-point and /api/cyclone-track."""
    if target_dt is None:
        target_dt = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    feat = get_surface_features_anywhere(lat, lon, target_dt)
    sst, ssh, sss = feat["sst"], feat["ssh"], feat["sss"]
    features = pd.DataFrame([[lat, lon, day_of_year, sst, ssh, sss]],
                             columns=["lat", "lon", "day_of_year", "sst", "ssh", "sss"])
    pred = model.predict(features)[0]
    predicted = {
        "surface": round(sst, 2),
        "50m": round(float(pred[0]), 2), "100m": round(float(pred[1]), 2),
        "200m": round(float(pred[2]), 2), "500m": round(float(pred[3]), 2),
    }
    d26, risk_level, risk_note = compute_tchp(predicted)
    insight = compute_insight(predicted)
    broader = compute_broader_applications(predicted, d26, insight)
    secondary = compute_secondary_indicators(predicted)
    nearest_km = float(feat["nearestObsDistanceKm"])
    confidence, confidence_note = _confidence_from_source(feat["source"], nearest_km, feat.get("reusedFromPool", False))
    uncertainty = _prediction_uncertainty(feat["source"], feat.get("reusedFromPool", False))
    return {
        "sst": sst, "ssh": ssh, "sss": sss,
        "predicted": predicted,
        "predictedUncertainty": uncertainty,
        "isotherm26Depth_m": d26, "riskLevel": risk_level, "riskNote": risk_note,
        "nearestObsDistanceKm": round(nearest_km, 2), "nObsUsed": feat["nObsUsed"],
        "confidence": confidence, "confidenceNote": confidence_note,
        "source": feat["source"], "reusedFromPool": feat.get("reusedFromPool", False),
        "broaderApplications": broader,
        "secondaryIndicators": secondary,
    }


@app.route("/api/predict-point/<float:lat>/<float:lon>")
def predict_point(lat, lon):
    """Predict a full subsurface profile + cyclone heat-potential risk at
    ANY ocean lat/lon on Earth (see get_surface_features_anywhere above for
    how validated/live/extrapolated sources are chosen). The only hard
    rejection left is land, since there's no subsurface ocean profile to
    predict there."""
    if df is None or model is None:
        return jsonify({"error": "Model/dataset not loaded"}), 500
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify({"error": "lat/lon out of range"}), 400
    if not is_ocean(lat, lon):
        return jsonify({"error": "Location is on land or within the coastal buffer — no subsurface ocean prediction possible"}), 400

    date_str = request.args.get("date")
    day_of_year = _day_of_year_from_date(date_str)
    target_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    s = _score_point(lat, lon, day_of_year, target_dt)

    source_labels = {
        "validated_dataset": "Interpolated from validated Argo/satellite observations",
        "live_satellite": "Live satellite reading (NASA/NOAA/ESA) for this exact point",
        "extrapolated": "Statistically extrapolated (no live data available)",
    }
    return jsonify({
        "lat": lat, "lon": lon, "dayOfYear": day_of_year,
        "date": date_str or None,
        "dataMode": DATA_MODE,
        "isInterpolated": s["source"] != "live_satellite",
        "interpolation": {
            "method": source_labels.get(s["source"], s["source"]),
            "source": s["source"],
            "reusedFromPool": s.get("reusedFromPool", False),
            "nearestObsDistanceKm": s["nearestObsDistanceKm"],
            "confidence": s["confidence"],
            "note": s["confidenceNote"],
        },
        "modelInputs": {"lat": lat, "lon": lon, "dayOfYear": day_of_year,
                         "sst": round(s["sst"], 2), "ssh": round(s["ssh"], 4), "sss": round(s["sss"], 2)},
        "predicted": s["predicted"],
        "predictedUncertainty": {
            "band": s["predictedUncertainty"],
            "note": ("± band per depth, in °C, derived from the model's cross-validated "
                     "RMSE at that depth (see Model Validation tab), widened for lower-"
                     "confidence input sources. Not a formal quantile-regression interval -- "
                     "with ~45 training rows a separate quantile model would be less stable "
                     "than the point estimate itself, so this uses the honest number we do "
                     "have (CV RMSE) instead.") if s["predictedUncertainty"] else None,
        },
        "insight": compute_insight(s["predicted"]),
        "consistency": compute_consistency(s["predicted"]),
        "disasterRisk": {
            "isotherm26Depth_m": s["isotherm26Depth_m"],
            "riskLevel": s["riskLevel"],
            "note": s["riskNote"],
            "disclaimer": "Educational approximation of Tropical Cyclone Heat Potential "
                           "(TCHP) derived from the model's predicted profile. Not a "
                           "substitute for official IMD/JTWC cyclone forecasts.",
        },
        "secondaryIndicators": s["secondaryIndicators"],
        "broaderApplications": s["broaderApplications"],
    })


@app.route("/api/cyclone-track", methods=["POST"])
def cyclone_track():
    """NOVEL / not in reference implementations: batch-score an entire
    cyclone advisory track (lat/lon/date points, e.g. pasted from an
    IMD/JTWC bulletin) for subsurface heat-potential risk in one call, and
    flag the highest-risk segment. Turns a point-reconstruction demo into
    an actual disaster-response screening workflow."""
    if df is None or model is None:
        return jsonify({"error": "Model/dataset not loaded"}), 500
    body = request.get_json(force=True, silent=True) or {}
    points = body.get("points", [])
    if not isinstance(points, list) or not points:
        return jsonify({"error": 'Body must be {"points": [{"lat":.., "lon":.., "date": "YYYY-MM-DD"}, ...]}'}), 400
    if len(points) > 200:
        return jsonify({"error": "Too many points (max 200 per request)"}), 400

    results = []
    for i, p in enumerate(points):
        try:
            lat, lon = float(p["lat"]), float(p["lon"])
        except (KeyError, TypeError, ValueError):
            results.append({"index": i, "error": "Missing/invalid lat or lon"})
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            results.append({"index": i, "lat": lat, "lon": lon, "error": "lat/lon out of range"})
            continue
        if not is_ocean(lat, lon):
            results.append({"index": i, "lat": lat, "lon": lon, "error": "On land"})
            continue
        date_str = p.get("date")
        day_of_year = _day_of_year_from_date(date_str)
        target_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        s = _score_point(lat, lon, day_of_year, target_dt)
        results.append({
            "index": i, "lat": lat, "lon": lon, "date": date_str,
            "isotherm26Depth_m": s["isotherm26Depth_m"], "riskLevel": s["riskLevel"],
            "confidence": s["confidence"], "source": s["source"],
            "nearestObsDistanceKm": s["nearestObsDistanceKm"],
            "predicted": s["predicted"], "broaderApplications": s["broaderApplications"],
            "secondaryIndicators": s["secondaryIndicators"],
        })

    valid = [r for r in results if "error" not in r]
    peak = max(valid, key=lambda r: r["isotherm26Depth_m"]) if valid else None
    return jsonify({
        "dataMode": DATA_MODE,
        "points": results,
        "trackSummary": {
            "totalPoints": len(points),
            "scored": len(valid),
            "peakRisk": peak,
            "elevatedCount": sum(1 for r in valid if r["riskLevel"] == "Elevated"),
        },
        "disclaimer": "Educational approximation of Tropical Cyclone Heat Potential (TCHP) "
                       "along the supplied track. Not a substitute for official IMD/JTWC forecasts.",
    })


def _call_nemotron(system_prompt: str, user_content: str, temperature: float, max_tokens: int) -> str:
    """Shared Nemotron chat-completion call used by both the analysis and
    copilot-chat endpoints, so the model name and call shape live in one
    place. Raises whatever the underlying client raises; callers handle
    the user-facing error response."""
    completion = nemotron_client.chat.completions.create(
        model=NEMOTRON_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature, top_p=0.95, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return completion.choices[0].message.content


# ── API: Nemotron ───────────────────────────────────────────────
@app.route("/api/nemotron/analyze", methods=["POST"])
def nemotron_analyze():
    """Free-form scientific analysis of a prompt via the Nemotron model."""
    if not nemotron_client:
        return jsonify({
            "error": "Nemotron analysis unavailable. The ML prediction remains available.",
            "available": False,
        })
    data = request.get_json()
    prompt = data.get("prompt", "")
    system_prompt = (
        "You are OceanEmbed's scientific reasoning engine. Provide structured analysis "
        "with sections: SURFACE CONDITIONS, THERMAL STRUCTURE, DEPTH-DEPENDENT FEATURES, "
        "OCEANOGRAPHIC SIGNIFICANCE, MODEL LIMITATIONS, SUMMARY. Be concise and scientific. "
        "Distinguish observations from predictions."
    )
    try:
        result = _call_nemotron(system_prompt, prompt, temperature=0.2, max_tokens=600)
        return jsonify({"result": result, "available": True})
    except Exception as e:
        logger.warning("Nemotron analysis call failed: %s", e)
        return jsonify({"error": "Nemotron analysis timed out. Please retry.", "detail": str(e), "available": True}), 500


# ── API: Copilot Chat ─────────────────────────────────────────
@app.route("/api/copilot/chat", methods=["POST"])
def copilot_chat():
    """Context-aware chat with the Nemotron-backed OceanEmbed copilot."""
    if not nemotron_client:
        return jsonify({
            "error": "Copilot is currently offline. The ML prediction and observation data remain available.",
            "available": False,
        })
    data = request.get_json()
    user_msg = data.get("message", "")
    context = data.get("context", {})

    # Build context-aware system prompt
    system_prompt = (
        "You are OceanEmbed Copilot, an AI scientific assistant integrated into the OceanEmbed "
        "oceanographic intelligence platform. You help users understand ocean observations, "
        "subsurface temperature predictions, and the science behind them.\n\n"
        "GUIDELINES:\n"
        "- Be concise and helpful.\n"
        "- Use plain language accessible to non-specialists while being scientifically accurate.\n"
        "- Distinguish between observations (real measurements), predictions (ML model output), "
        "and your own interpretation.\n"
        "- When the user asks about a specific observation, use the provided context.\n"
        "- You can explain oceanography concepts like SST, SSH, SSS, thermocline, mixed layer, etc.\n"
        "- You can explain model metrics like RMSE, MAE, and R-squared.\n"
        "- Do NOT fabricate data. If context is not available, say so.\n"
        "- Keep responses under 300 words unless the user asks for more detail."
    )

    # Build user message with context if available
    full_msg = user_msg
    if context:
        ctx_parts = ["\n--- CURRENT OBSERVATION CONTEXT ---"]
        if "lat" in context and "lon" in context:
            ctx_parts.append(f"Location: {context['lat']}\u00b0N, {context['lon']}\u00b0E")
        if "date" in context:
            ctx_parts.append(f"Observation date: {context['date']}")
        if "argoWMO" in context:
            ctx_parts.append(f"Argo WMO: {context['argoWMO']}, Cycle: {context.get('cycle', 'N/A')}")
        if "surface" in context:
            s = context["surface"]
            if isinstance(s, dict):
                ctx_parts.append(f"SST: {s.get('sst',{}).get('value','?')} {s.get('sst',{}).get('unit','')} (source: {s.get('sst',{}).get('source','')})")
                ctx_parts.append(f"SSH: {s.get('ssh',{}).get('value','?')} {s.get('ssh',{}).get('unit','')} (source: {s.get('ssh',{}).get('source','')})")
                ctx_parts.append(f"SSS: {s.get('sss',{}).get('value','?')} {s.get('sss',{}).get('unit','')} (source: {s.get('sss',{}).get('source','')})")
        if "predicted" in context:
            p = context["predicted"]
            ctx_parts.append(f"ML Predicted temperatures: 50m={p.get('50m','?')}, 100m={p.get('100m','?')}, 200m={p.get('200m','?')}, 500m={p.get('500m','?')}\u00b0C")
        if "actual" in context:
            a = context["actual"]
            ctx_parts.append(f"Observed (Argo) temperatures: 50m={a.get('50m','?')}, 100m={a.get('100m','?')}, 200m={a.get('200m','?')}, 500m={a.get('500m','?')}\u00b0C")
        if "errors" in context:
            e = context["errors"]
            ctx_parts.append(f"Prediction errors: 50m={e.get('50m','?')}, 100m={e.get('100m','?')}, 200m={e.get('200m','?')}, 500m={e.get('500m','?')}\u00b0C")
        if "insight" in context:
            ins = context["insight"]
            ctx_parts.append(f"Thermal insight: {ins.get('title','')} ({ins.get('level','')} signal)")
        ctx_parts.append("--- END CONTEXT ---\n")
        full_msg = "\n".join(ctx_parts) + "\n\nUser question: " + user_msg

    try:
        result = _call_nemotron(system_prompt, full_msg, temperature=0.3, max_tokens=800)
        return jsonify({"result": result, "available": True})
    except Exception as e:
        logger.warning("Nemotron copilot call failed: %s", e)
        return jsonify({"error": "Copilot encountered an error. Please try again.", "detail": str(e), "available": True}), 500


if __name__ == "__main__":
    logger.info("OceanEmbed API starting on http://%s:%d (debug=%s)", API_HOST, API_PORT, API_DEBUG)
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
