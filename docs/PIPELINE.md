OceanEmbed v2 — Pipeline Overview

This document explains the new observation-trained data pipeline and how to run it.

1) Fetch raw Argo profiles (optional)
   - data/raw/argo/ holds raw NetCDF/profile artifacts
   - Run: python -m data.argo_fetch (or scripts/build_dataset.py --use-raw)

2) Preprocess Argo
   - data/argo_preprocess.py will read raw files, apply Argo QC flags, convert pressure->depth (gsw when available), and interpolate to depths: 0,50,100,200,500,1000 m
   - Output: data/processed/argo_profiles.parquet
   - Run: python -m data.argo_preprocess or scripts/build_dataset.py

3) Fetch surface data
   - data/surface_fetch.py contains helpers for fetching SST/SSH/SSS from OPeNDAP/ERDDAP endpoints. Configure URLs in the module.

4) Match surface to Argo
   - data/match_surface_to_argo.py joins per-profile target temperatures with nearest/aggregated surface observations
   - Output: data/dataset/train_dataset.parquet

5) Train baseline
   - Use train_model.py: trains LightGBM multi-output baseline and saves model.pkl

6) Inference & App
   - inference/predict.py loads model.pkl and performs inference
   - streamlit app (app.py) should call inference/predict.py for real-mode predictions

Notes
- Use scripts/build_dataset.py to orchestrate steps 1-4. Start with --use-raw to try to fetch real Argo; pipeline has synthetic fallback when network/data are not available.
- Ensure required Python packages are installed. See requirements.txt (updated) for recommended packages.
- Nemotron integration is kept as an explanation layer only and requires NEMOTRON_API_KEY in .streamlit/secrets.toml
