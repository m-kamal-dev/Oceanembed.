# Patch notes — why you were only seeing 9 dots

## Root cause (fixed)
`api_server.py` was loading `test_sample.csv` as the dataset that backs
the map, `/api/dataset-stats`, `/api/matching-quality`, and the
predict-point nearest-neighbor interpolation. `test_sample.csv` is only
the 20% GroupShuffleSplit held out for scoring the model — 9 of your 45
real, validated rows. The other 36 rows (the ones actually used to
*train* the model) were loaded correctly for training but never shown
on the map or used for interpolation at runtime.

**Fix**: `api_server.py` now loads the full validated dataset
(`data/dataset/train_dataset.parquet`, all 45 real rows) as `df` for
everything display/interpolation-related, and keeps a separate
`test_df` (from `test_sample.csv`) for anything that specifically wants
"exactly the held-out rows the model was scored on" (nothing currently
does — `/api/metrics` already reads pre-computed metrics straight out
of `model.pkl`, so this change is safe).

**To apply**: replace your `api_server.py` with the one in this patch
and restart `python api_server.py`. You'll immediately go from 9 dots
to 45 (all genuine Argo-matched real observations) — no internet
needed, no retraining needed, nothing about your data changes, just
what the app was reading.

## Getting past 45 real points (up to 500+)

45 is not an artificial cap — it's literally how many real Argo
profiles were fetched and passed the hard-gate validator so far (13
months, one bounding box). I don't have internet access in this
environment, so I can't run the live Argo/satellite fetch myself, but
your repo already ships the full real pipeline
(`scripts/build_dataset.py`). On a machine with internet access:

```bash
python scripts/build_dataset.py --use-raw \
    --start-date 2019-01-01 --end-date 2026-02-01 \
    --max-profiles 800
python train_model.py
python scripts/write_data_status.py
python api_server.py
```

Widening the date range to ~7 years (instead of 13 months) is the main
lever — Arabian Sea Argo float density means a single narrow window
just doesn't have 500 profiles in it. You can also nudge the bounding
box a little wider in `data/argo_fetch.py`'s defaults if you want to
pull in the Bay of Bengal or more open-ocean floats. This re-runs the
exact same hard-gate validator, so whatever comes out stays genuinely
real, just bigger.

## If you want 500+ points *right now* for UI/demo purposes

Per your own `docs/DATA_INTEGRITY.md` policy, synthetic data must stay
clearly labeled and separate from real training data — never presented
as real. I ran your existing `scripts/generate_demo_data.py` (pure
local computation, no internet needed) and it produced **5,000**
clearly-labeled synthetic rows at `data/demo/ocean_data_synthetic.csv`
(included in this patch). To actually see them in the app:

```bash
python train_model.py --demo
python scripts/write_data_status.py
# rename or move model.pkl aside first, since api_server.py always
# prefers model.pkl (real) over model_demo.pkl if both are present
python api_server.py
```

The UI will honestly badge this as "Demo/synthetic data" (not real) —
that's intentional, per your own project's integrity policy. Switch
back any time by restoring `model.pkl`.
