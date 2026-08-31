# Data Integrity & Real-Observation Requirements

## Core Principle

**OceanEmbed training and validation datasets must contain only legitimate, traceable real-world observations — not synthetic, simulated, randomly generated, or fabricated values.**

This is essential for scientific credibility and for making valid claims to SIH judges.

## What This Means

### ✅ **Allowed Data Sources**

- **Argo Float Observations**: Quality-controlled subsurface temperature profiles from the Argo GDAC, with TEMP_QC or TEMP_ADJUSTED_QC flags applied.
- **Satellite/In-Situ Surface Data**: SST, SSH, SSS, wind vectors from established observational datasets (e.g., NOAA ERDDAP, Copernicus, OPeNDAP endpoints).
- **Archival Datasets**: Pre-published, reproducible, and documented ocean observations.

### ❌ **Not Allowed**

- Synthetic data (e.g., `generate_data.py` output)
- Simulated or modeled data (e.g., output from ocean models)
- Randomly generated numbers
- Manually fabricated test values
- Untraced or undocumented "demo" observations

---

## Data Pipeline Requirements

Every row in the training dataset must:

1. **Originate from real observations** with documented source and date
2. **Include provenance fields**:
   - `argo_wmo`: Argo float identifier
   - `argo_cycle`: Profile cycle number
   - `profile_time`: Timestamp of Argo profile
   - `surface_time`: Timestamp of matched surface observation
   - `surface_distance_km`: Distance between Argo profile and surface observation location
3. **Apply real QC filtering**: Argo TEMP_QC or TEMP_ADJUSTED_QC flags (accept 0, 1, 2 only)
4. **Respect physical constraints**:
   - Profiles must reach ≥ 500 m to be included (no extrapolation beyond observed depths)
   - Surface observations matched within ±24 hours and 25 km spatial radius
   - No temporal leakage (surface observations must be from same time or before Argo profile)

---

## Pre-Training Validation

Before training model.pkl, **you must run**:

```bash
python scripts/validate_dataset.py
```

or this will run automatically when you execute:

```bash
python scripts/build_dataset.py --use-raw
```

### The Validation Report

The validation report (`data/dataset/validation_report.txt`) must show:

```text
OCEANEMBED DATA VALIDATION REPORT
════════════════════════════════

DATASET SIZE
────────────
Total rows: 1234

PROVENANCE & SOURCES
────────────────────
Unique Argo floats (WMO): 45
Unique cycles: 234
Date range: 2023-01-01 → 2024-12-31

GEOGRAPHIC COVERAGE
────────────────────
Latitude: 8.00°N → 24.00°N
Longitude: 60.00°E → 77.00°E

SUBSURFACE TEMPERATURE TARGETS
────────────────────────────────
temperature_500m: min=2.50°C, max=28.90°C, mean=15.34°C

MISSING DATA SUMMARY
────────────────────
[list of missing % per column]

SYNTHETIC DATA DETECTION
────────────────────────
✅ No synthetic data patterns detected.
Dataset appears to contain real observations.

✅ DATASET VALIDATION PASSED
════════════════════════════════
```

### The Critical Check: `Synthetic rows: 0`

If the validator detects any of these signatures of synthetic data:

- All rows have identical temperature profiles
- Duplicate rows exceed 10% of the dataset
- Missing provenance fields (argo_wmo, argo_cycle) in >50% of rows
- Unrealistic value distributions (e.g., all values round numbers)

Then the pipeline will **raise a RuntimeError** and **block model training**:

```
❌ DATASET VALIDATION FAILED
Synthetic data detected in training dataset.
BLOCKING: You cannot train model.pkl on synthetic/fabricated data.
```

---

## Dataset Separation

### Real Training Data
- **File**: `data/dataset/train_dataset.parquet`
- **Source**: Real Argo + surface observations
- **Use**: Train model_real.pkl

### Synthetic/Demo Data (development only)
- **File**: `ocean_data_synthetic.csv` (clearly labeled as demo)
- **Source**: `generate_data.py`
- **Use**: Local development, UI testing, documentation
- **Never mixed** with real training data

---

## Reproducibility & Traceability

Every training example must be **reproducible**. This means:

1. **Source is documented**: e.g., "Argo float WMO-1234567, cycle 123"
2. **Preprocessing is deterministic**: Same code + same input data = same output
3. **No random choices**: No shuffling during pipeline (only in train/test split with explicit random seed)
4. **Data URLs are recorded**: Document which OPeNDAP endpoints were used for surface data

---

## Standard Training Workflow

```
REAL ARGO OBSERVATIONS
         ↓
   argo_fetch.py
   (download .nc files)
         ↓
   argo_preprocess.py
   (apply QC, depth conversion, interpolation)
         ↓
   surface_fetch.py
   (obtain SST/SSH/SSS from real endpoints)
         ↓
   match_surface_to_argo.py
   (spatial + temporal matching)
         ↓
   train_dataset.parquet (REAL DATA)
         ↓
   validate_dataset.py
   (pre-training validation report)
         ↓
   ✅ VALIDATION PASSED
         ↓
   train_model.py
   (LightGBM training on REAL data)
         ↓
   model_real.pkl
```

---

## Communicating to Judges

When presenting OceanEmbed to SIH judges, you can confidently say:

> "OceanEmbed is trained on quality-controlled Argo float observations paired with satellite-derived surface measurements. Every training row is traceable to its source (Argo WMO, cycle, timestamp) and surface observation (location, time, distance). The pipeline applies Argo QC flags and validates data completeness before model training. The final model is evaluated against independent Argo test observations."

You **cannot** say this unless the data integrity checks pass.

---

## FAQ

**Q: Can I use synthetic data for model development?**  
A: Yes, but keep it in `ocean_data_synthetic.csv` with clear labeling. Never mix it into real training.

**Q: What if real Argo download fails?**  
A: The pipeline should fail loudly with an error message, not silently fall back to synthetic data. This is intentional.

**Q: Can I interpolate 1000 m temperatures if the profile only reaches 900 m?**  
A: No. Use the actual observed depth. Extrapolation introduces bias and isn't defensible scientifically.

**Q: What if a surface observation is missing for a particular Argo profile?**  
A: That row is skipped from the training set. Missing surface data is acceptable; fabricating it is not.

**Q: How do I know if the validation passed?**  
A: Check the exit code: `echo $?` (0 = pass, 1 = fail). Also check `validation_report.txt` for a summary.

---

## References

- Argo data quality: https://argovis.colorado.edu/
- ERDDAP OPeNDAP endpoints: https://coastwatch.pfeg.noaa.gov/erddap/
- Copernicus Marine: https://data.marine.copernicus.eu/
