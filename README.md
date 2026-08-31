# 🌊 OceanEmbed — Subsurface Ocean Temperature Reconstruction (SIH25066)

**Theme:** Disaster Management · **Category:** Software
Reconstructs subsurface ocean temperature (50m/100m/200m/500m) in the Arabian Sea
from surface satellite/in-situ observations (SST, SSH, SSS), and surfaces a
**Cyclone Heat Potential** indicator for disaster-management use.

> Where this idea sits in the field: satellite-to-subsurface temperature
> reconstruction is an active published research area (CNN/LightGBM, deep
> forest, ConvLSTM and graph-attention models all exist in the literature,
> some reaching R² ≈ 0.98 globally). This project's contribution isn't a new
> ML architecture — it's a **fully-traceable Indian-Ocean pipeline with a
> hard data-integrity gate** and a **direct disaster-response framing**
> (cyclone rapid-intensification risk), which most reference implementations
> don't provide.

## ✅ This build ships trained on real data

This app can run in exactly two modes, and it always tells you which one
you're in (see the header badge / `/api/status`):

| Mode | How it's produced | Model file | Use for |
|---|---|---|---|
| **Real** | `scripts/build_dataset.py --use-raw` → genuine Argo floats + NASA/NOAA/ESA satellite surface data, QC-filtered, hard-gate validated | `model.pkl` | Judging, real claims |
| **Demo** | `scripts/generate_demo_data.py` → formula-based synthetic numbers, explicitly labeled | `model_demo.pkl` | UI development only |

`train_model.py` will **refuse to train** (exit with instructions) if no
validated real dataset exists — it will never silently substitute synthetic
numbers the way the original prototype did.

**This repo ships pre-trained on real data.** `data/dataset/train_dataset.parquet`
is a genuine, hard-gate-validated dataset — 45 matched observations from 45
unique real Argo floats, paired with real NASA JPL MUR SST, NOAA NESDIS SSH,
and ESA SMOS / Argo in-situ SSS readings (see `data/dataset/validation_report.txt`
and `docs/REAL_DATA_DATASHEET.md` for the full audit trail). `model.pkl` was
trained directly on it, and `data_status.json` / `/api/status` report `mode:
real` accordingly. No demo/synthetic model ships in this build, so there is
no fallback path to accidentally demo the wrong one.

The dataset is real but still small (45 rows, 22 profile cycles) — honest
about that scale, not a limitation to hide. Growing it is just a matter of
internet access; see "Growing the real dataset further" below.

## 🚀 Quickstart

```bash
pip install -r requirements.txt

# 0. (Optional) Enable the Copilot panel — see "AI Copilot setup" below.
#    Skip this and the app still runs fine, just with the "AI Offline" badge.

# 1. Launch — model.pkl and the real dataset are already included.
python api_server.py
# open http://localhost:5001
```

### Growing the real dataset further

More real Argo profiles = a stronger model. On a machine with internet
access:
```bash
python scripts/build_dataset.py --use-raw --max-profiles 200   # fetch more real data
python train_model.py                                           # retrain on the larger set
python scripts/write_data_status.py                             # refresh the badge
python api_server.py
```
This re-runs the same hard-gate validator behind the bundled dataset, so the
result stays genuinely real — it just becomes a bigger genuinely-real
dataset. If you want to fall back to synthetic data purely for UI
development (never for judging), that path still exists too:
```bash
python scripts/generate_demo_data.py
python train_model.py --demo
python scripts/write_data_status.py
python api_server.py
```
`api_server.py` always prefers `model.pkl` over `model_demo.pkl` automatically
if both are present — remove or rename `model.pkl` first if you deliberately
want to preview the demo model instead.

### 🤖 AI Copilot setup

The Copilot chat panel calls NVIDIA's Nemotron model and needs one API key.
Without it, everything else (map, predictions, metrics) still works — the
panel just shows "AI Offline".

1. Get a free key at [build.nvidia.com](https://build.nvidia.com) → open
   any Nemotron model page → **Get API Key**.
2. Copy `.env.example` to `.env` (same folder as `api_server.py`) and paste
   the key in:
   ```bash
   cp .env.example .env
   # then edit .env and set NEMOTRON_API_KEY=<your real key>
   ```
   (`.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` works too,
   if you'd rather use that format — `api_server.py` checks both.)
3. Restart `python api_server.py`. The Copilot badge should switch from
   "AI Offline" to live within a few seconds.

`.env` and `secrets.toml` are already git-ignored — never commit your real
key. Have a screenshot or cached response ready as a plan B in case judging
Wi-Fi is unreliable.

## 📁 Structure

```
OceanEmbed/
├── config.py                     # Centralized constants (ports, matching
│                                  #   radius/window, reference year, model names)
├── data/
│   ├── argo_fetch.py             # Real Argo float discovery/download (argopy)
│   ├── argo_preprocess.py        # QC filtering + TEOS-10 depth conversion
│   ├── surface_fetch.py          # Real satellite surface obs (SST/SSH/SSS)
│   ├── match_surface_to_argo.py  # Spatial/temporal matching, ≤25km/≤24h
│   ├── raw/argo/                 # Downloaded NetCDF profiles (gitignored)
│   ├── processed/                # QC'd profiles (parquet)
│   ├── dataset/                  # Final matched + validated training set
│   └── demo/                     # Synthetic demo data ONLY — never mixed in
├── scripts/
│   ├── build_dataset.py          # End-to-end real pipeline orchestrator
│   ├── validate_dataset.py       # Hard gate: blocks training on fake data
│   ├── generate_demo_data.py     # Clearly-labeled synthetic data for UI dev
│   └── write_data_status.py      # Single source of truth for real/demo badge
├── train_model.py                # Real by default; --demo is explicit opt-in
├── inference/predict.py          # Standalone/batch inference wrapper
├── api_server.py                 # Flask API + serves frontend/
├── frontend/index.html           # Single dashboard UI (map, predictions,
│                                  #   provenance, cyclone heat potential, AI copilot)
├── land_mask.py                  # Filters land points from the map
├── land_data/                    # Natural Earth 110m land shapefile (bundled)
└── docs/DATA_INTEGRITY.md        # Full data-integrity policy & FAQ
```

Only **one** presentation app is used going forward: the Flask + HTML
dashboard (`api_server.py` + `frontend/`). The original project bundled a
Streamlit app, this Flask/HTML dashboard, *and* a duplicate nested copy of
the whole repo — that's now consolidated into the structure above. The old
Streamlit app and its old data generator (which had a silent synthetic-data
fallback) have been removed entirely rather than kept around as dead code.

## 🌀 Disaster-management framing

The dashboard computes an approximate **Tropical Cyclone Heat Potential**
indicator from the predicted profile — the depth at which predicted
temperature crosses 26°C (a real, operationally-used cyclone-intensification
proxy: a deep warm layer gives a storm more heat to draw on). This is what
ties the ML output to the *Disaster Management* PS category, rather than
showing an isolated temperature curve. See `/api/disaster-risk/<lat>/<lon>`.
This is an educational approximation, not an operational forecast — the UI
says so explicitly.

## 🌟 What's not available in comparable reference implementations

Published satellite-to-subsurface reconstruction work (and the original
prototype this was built on) evaluates the model **only at points that
already exist in the training dataset** — a plain lookup. That's a real
gap for a disaster-management tool: a live IMD/JTWC cyclone advisory gives
arbitrary lat/lon/time points that will essentially never land exactly on
a historical Argo-matched location.

- **`/api/predict-point/<lat>/<lon>`** — predicts a full subsurface
  profile at **any ocean point on Earth**, not just Arabian Sea dataset
  points. Three sources are tried in order of trust, and the response
  always says honestly which one answered:
  1. **Validated** — within 25 km of a real Argo/satellite-matched
     observation already in the dataset (IDW-interpolated).
  2. **Live** — outside that radius, so the server fetches a real,
     live SST/SSH/SSS reading for the clicked point from NASA/NOAA/ESA
     ERDDAP servers (`data/surface_fetch.py`) and runs the model on
     genuine current conditions there, with an 8-second cap so a slow
     satellite endpoint can never hang a map click.
  3. **Extrapolated** — live fetch unavailable (offline, no satellite
     pass in the time window, endpoint down) — falls back to inverse-
     distance-weighted extrapolation from the nearest validated
     observations, clearly labeled as such with a downgraded confidence
     tier, rather than either refusing or pretending precision it
     doesn't have.

  Only points on land (Natural Earth 110m land mask, which covers the
  whole globe) are rejected — there's no subsurface ocean profile to
  predict there. Click anywhere on the Observations map to try it.
- **`/api/cyclone-track`** — batch-scores an entire cyclone advisory track
  (a list of lat/lon/date points) for Tropical Cyclone Heat Potential in
  one call, and flags the highest-risk segment. See the **Cyclone Track
  Risk** tab. This turns a per-point reconstruction demo into an actual
  screening workflow a forecaster could paste a real advisory track into.

Both reuse the same trained model and the same TCHP proxy as the rest of
the app — they don't introduce a second, less-trustworthy code path.

## 🔍 What was fixed from the original prototype

1. **Silent synthetic fallback removed.** The old `generate_data.py` tried a
   live fetch and, on any failure, quietly wrote fabricated numbers to the
   same `ocean_data.csv` the model trained on — with no signal to the user.
   `train_model.py` now hard-fails instead, per the project's own
   `docs/DATA_INTEGRITY.md` policy.
2. **SSH/SSS were never real even on the "success" path.** Both were
   formula-generated proxies regardless of whether the Argo fetch succeeded.
   The real pipeline (`surface_fetch.py` + `match_surface_to_argo.py`) now
   pulls actual satellite SSH/SSS instead.
3. **`/api/matching-quality` was hardcoded** (fixed fake numbers like "0.39 km
   mean distance, 100% compliant") regardless of what dataset was loaded.
   It's now computed from the loaded dataset's real provenance columns, or
   honestly reports "not available in demo mode."
4. **`/api/status` said "Data Ready" for any dataframe**, real or fake. It
   now reports the actual mode (`real` / `demo_synthetic` / `missing`).
5. **Three overlapping UIs** (Streamlit, Flask+HTML, and a duplicated nested
   copy of the whole repo) consolidated into one.
6. **`requirements.txt`** now lists every package actually imported
   (flask, flask-cors, openai, argopy, gsw, shapely, pyshp, pyarrow were
   missing before).
7. **Map predictions were limited to a box around the training data,
   and negative coordinates 404'd.** `/api/predict-point` previously
   rejected any click outside a bounding box around the (small) Arabian
   Sea dataset, and separately used Flask's default `float` route
   converter, which doesn't match a leading `-` — so it 404'd on any
   negative latitude/longitude before the domain check even ran (most
   of the Southern and Western Hemispheres). Both are fixed: a custom
   signed-float route converter, plus the validated/live/extrapolated
   fallback chain described above, so any ocean point on Earth returns
   an honestly-labeled prediction instead of an error.

## Known limitations / next steps

- The Nemotron AI copilot is a single external API-key dependency — have a
  plan B (screenshots or a cached response) in case the venue Wi-Fi or the
  API is unavailable during judging.
- The core model (LightGBM MultiOutputRegressor on 6 tabular features) is a
  reasonable baseline but a plain regressor; if you have time, a
  spatiotemporal model (ConvLSTM/graph-attention, as in recent literature)
  would likely improve accuracy at depth, especially at 500m.
- Prediction uncertainty (`predictedUncertainty` in `/api/predict-point`,
  shown as "± X°C" per depth in the UI) is the model's cross-validated RMSE
  per depth, widened for lower-confidence input sources — not a true
  quantile-regression interval. With ~45 training rows, a separate
  quantile model would itself be less stable than the point estimate, so
  this is the honest number we actually have rather than a fabricated
  interval. Revisit once the dataset is larger.
- `/api/predict-point` and `/api/cyclone-track` prefer a live ERDDAP fetch
  for points far from the training set, but fall back to simple inverse-
  distance weighting from the k nearest loaded observations when live data
  isn't available. That's honest and fast, but a kriging/optimal-
  interpolation scheme (as satellite data providers use operationally)
  would give a statistically sounder uncertainty estimate than the current
  distance-only confidence tiers — worth doing if you have time before
  judging.
- The live-fetch path depends on outbound internet access to
  `coastwatch.pfeg.noaa.gov` at judging time; if the venue network blocks
  it, predictions far from the Arabian Sea will silently use the
  extrapolated fallback instead (still labeled honestly in the UI, just
  lower confidence) — worth testing on the actual venue Wi-Fi beforehand.
  There is currently no pre-fetched offline snapshot bundled for this case
  beyond the live-fetch pool the app builds up on its own from real usage
  (`data/raw/surface/live_fetch_pool.csv`).
- `land_mask.py` now ships with the real Natural Earth 110m land shapefile
  in `land_data/` (previously missing, so land filtering silently did
  nothing). If that folder is ever empty, land filtering fails safe (no
  filtering) rather than crash — check the server log for the
  `[OK] Land mask loaded` / `[WARN] Land mask unavailable` line to confirm
  which mode you're actually running in.
- Training data is currently a single basin (Arabian Sea) over ~13 months
  (45 matched Argo profiles). Extending to Bay of Bengal and a wider
  Indian Ocean domain is now a one-command operation --
  `python -m scripts.build_dataset --use-raw --region all --max-profiles 1000`
  (see `config.REAL_DATA_REGIONS`) -- but it still requires live internet
  access to Ifremer/NOAA/NASA/ESA to actually run; nothing in this repo
  fabricates rows for regions that haven't actually been fetched. Whatever
  row count comes back depends on real Argo float density in each box and
  your date window, not a fixed number.
- `/api/metrics` now returns both the single-holdout metrics and the
  5-fold cross-validated metrics (`crossValidated`), and the Model
  Validation tab shows both side by side, the CV one labeled as the
  trusted number. Quote the CV number, not the holdout one, in any
  report/poster/README claim.
- The API now has basic protection: an in-memory rate limiter (60
  req/min/IP by default, `OCEANEMBED_RATE_LIMIT_PER_MIN` env var) is
  always on, and an API key check (`X-API-Key` header) is available but
  off by default — set `OCEANEMBED_API_KEY` to turn it on. Neither is a
  substitute for a real API gateway in production.
- `tests/` now runs in CI on every push/PR via
  `.github/workflows/tests.yml`.
- Per-point "predicted vs actual" comparisons (`/api/observation/<lat>/<lon>`)
  now only score accuracy on points actually in the held-out test split
  (`inTestSplit: true`); for a point the model trained on, the UI shows the
  prediction but explicitly labels it "TRAINING POINT" and omits the error
  metric, so a near-perfect match there can't be mistaken for real accuracy.
- Three more subsurface-derived disaster indicators now sit alongside
  Cyclone Heat Potential (`/api/disaster-risk`, `/api/predict-point`,
  `/api/cyclone-track` → `secondaryIndicators`): **Mixed Layer Depth**
  (0.5°C-threshold proxy for the standard MLD criterion), **Thermocline
  Steepness** (which depth bin has the sharpest gradient, and whether that
  resists or allows cyclone-driven cold-water upwelling), and **Cold-Wake
  / Self-Limiting Potential** (surface-to-100m temperature difference as a
  proxy for how much a cyclone could cool its own fuel source). All three
  are derived purely from the model's own 5-depth predicted profile — no
  new data source required — and are simplified proxies, not the actual
  operational NOAA/CIMSS formulations; say so if asked in judging.
- **Demo mode now covers 1800 synthetic points** across all three regions
  (`scripts/generate_demo_data.py`, regenerate anytime with
  `python scripts/generate_demo_data.py`), every row tagged
  `is_synthetic=True` + `source_region` and never mixed into the real
  45-row dataset. It's off by default (a real `model.pkl` always wins);
  set `OCEANEMBED_FORCE_DEMO=true` to view it explicitly. If you retrain
  it (`python train_model.py --demo`) the demo map badges still say
  "SYNTHETIC/DEMO" everywhere — that labeling is driven by `DATA_MODE`,
  not by anything in the generator, so it can't accidentally end up
  looking like real observations.
