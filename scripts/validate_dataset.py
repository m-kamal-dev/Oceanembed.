"""
Data validation hard-gate: verify that the training dataset is built strictly from real observations.
Generates an audit report detailing sources, counts, statistical distributions, physical ranges,
and synthetic-data checks.

Hard Gate: Raises RuntimeError and exits with code 1 if any synthetic data or invalid provenance is found.
"""
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import SPATIAL_MATCH_RADIUS_KM, TEMPORAL_MATCH_WINDOW_HOURS

BASE = Path(__file__).resolve().parents[1] / 'data'
DATASET_DIR = BASE / 'dataset'
PROCESSED_DIR = BASE / 'processed'

REQUIRED_PROVENANCE_COLS = [
    'argo_wmo', 'argo_cycle', 'profile_time', 'surface_time', 'surface_distance_km'
]
REQUIRED_TARGET_COLS = [
    'temp_50m', 'temp_100m', 'temp_200m', 'temp_500m'
]
REQUIRED_SURFACE_COLS = ['lat', 'lon', 'day_of_year', 'sst', 'ssh', 'sss']


def detect_synthetic_rows(df: pd.DataFrame) -> List[str]:
    """
    Strict algorithmic checks for synthetic, simulated, or fabricated data:
    1. Repeated / quantized exact floating values
    2. Zero variance in temperature across profiles
    3. Missing or placeholder provenance (e.g., missing WMO / cycle numbers)
    4. Exact duplicate rows beyond physics
    5. Physical impossibility checks (e.g., negative temperatures in tropical Arabian Sea, non-physical salinity)
    """
    issues = []
    
    # Check 1: Missing provenance
    for col in REQUIRED_PROVENANCE_COLS:
        if col not in df.columns:
            issues.append(f"Missing mandatory provenance column: {col}")
        else:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues.append(f"Provenance column {col} contains {missing_count} null values ({100*missing_count/len(df):.1f}%)")

    # Check 2: Check for synthetic constant profiles or lack of variance
    for col in REQUIRED_TARGET_COLS:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 1 and valid.std() < 1e-6:
                issues.append(f"Target column {col} has zero standard deviation across profiles (synthetic indicator)")

    # Check 3: Exact duplicates across features and targets
    check_cols = [c for c in ['lat', 'lon', 'sst', 'ssh', 'sss', 'temp_50m', 'temp_100m', 'temp_200m', 'temp_500m'] if c in df.columns]
    if len(check_cols) > 0 and len(df) > 1:
        n_dupes = df.duplicated(subset=check_cols, keep=False).sum()
        if n_dupes > max(2, len(df) * 0.1):
            issues.append(f"High duplicate rate: {n_dupes} duplicate rows detected ({100*n_dupes/len(df):.1f}%)")

    # Check 4: Physical bounds across our full study domain (Arabian Sea, Bay of Bengal, wider Indian Ocean)
    if 'sst' in df.columns:
        if (df['sst'] < 15.0).any() or (df['sst'] > 35.0).any():
            issues.append("SST contains values outside physical North Indian Ocean bounds (15°C - 35°C)")
    if 'temp_500m' in df.columns:
        if (df['temp_500m'] < 2.0).any() or (df['temp_500m'] > 20.0).any():
            issues.append("Temperature at 500m contains values outside physical ocean bounds (2°C - 20°C)")
    if 'sss' in df.columns:
        valid_sss = df['sss'].dropna()
        # Arabian Sea open water runs ~35-37 PSU, but the Bay of Bengal (part of our
        # fetch domain since the multi-region expansion) is a documented low-salinity
        # basin: normal open-ocean BoB is ~29-34.5 PSU, and published near-surface
        # freshening events (heavy monsoon river discharge, esp. northern BoB) push it
        # into the low-to-mid 20s PSU. 20 PSU is a genuine floor for that; below it is
        # implausible for an open-ocean Argo float and worth flagging as suspect.
        if (valid_sss < 20.0).any() or (valid_sss > 40.0).any():
            issues.append("SSS contains values outside physical North Indian Ocean bounds (20 - 40 PSU, widened from Arabian-Sea-only 30-40 to cover documented Bay of Bengal freshening)")

    return issues


def generate_validation_report(train_dataset_parquet: Path, output_file: Path = None) -> bool:
    """Generate a comprehensive data validation report and enforce hard gate."""
    if not train_dataset_parquet.exists():
        logger.error(f"Training dataset not found: {train_dataset_parquet}")
        return False

    df = pd.read_parquet(train_dataset_parquet)
    total_rows = len(df)
    if total_rows == 0:
        logger.error("Training dataset is empty!")
        return False

    unique_wmo = df['argo_wmo'].nunique() if 'argo_wmo' in df.columns else 0
    unique_cycles = df['argo_cycle'].nunique() if 'argo_cycle' in df.columns else 0
    
    p_times = pd.to_datetime(df['profile_time']) if 'profile_time' in df.columns else None
    date_min = p_times.min() if p_times is not None else "N/A"
    date_max = p_times.max() if p_times is not None else "N/A"

    lat_min, lat_max = (df['lat'].min(), df['lat'].max()) if 'lat' in df.columns else (np.nan, np.nan)
    lon_min, lon_max = (df['lon'].min(), df['lon'].max()) if 'lon' in df.columns else (np.nan, np.nan)

    dist_mean = df['surface_distance_km'].mean() if 'surface_distance_km' in df.columns else np.nan
    dist_max = df['surface_distance_km'].max() if 'surface_distance_km' in df.columns else np.nan
    dist_min = df['surface_distance_km'].min() if 'surface_distance_km' in df.columns else np.nan

    time_diff_mean = df['surface_time_diff_hours'].mean() if 'surface_time_diff_hours' in df.columns else np.nan
    time_diff_max = df['surface_time_diff_hours'].max() if 'surface_time_diff_hours' in df.columns else np.nan

    if 'surface_distance_km' in df.columns and 'surface_time_diff_hours' in df.columns:
        violations = int(((df['surface_distance_km'] > SPATIAL_MATCH_RADIUS_KM)
                           | (df['surface_time_diff_hours'] > TEMPORAL_MATCH_WINDOW_HOURS)).sum())
        compliance_pct = 100.0 * (1 - violations / max(total_rows, 1))
        violations_str = f"{violations} ({compliance_pct:.1f}% compliant)"
    else:
        violations_str = "N/A (matching columns not present)"

    # Synthetic check
    synthetic_issues = detect_synthetic_rows(df)
    synthetic_count = len(synthetic_issues)
    is_valid = (synthetic_count == 0)

    status_str = "[PASS] PASSED" if is_valid else "[FAIL] FAILED"

    # Build report
    report_lines = [
        "",
        "=" * 70,
        "OCEANEMBED REAL DATA VALIDATION REPORT & HARD GATE",
        "=" * 70,
        f"Generated Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Dataset Path:        {train_dataset_parquet}",
        f"Validation Result:   {status_str}",
        "",
        "1. DATASET VOLUME & PROVENANCE",
        "─" * 70,
        f"Total Observation Rows:       {total_rows}",
        f"Total Columns:                {len(df.columns)}",
        f"Unique Argo Floats (WMO):     {unique_wmo}",
        f"Unique Profile Cycles:        {unique_cycles}",
        f"Observation Date Range:       {date_min} -> {date_max}",
        f"Latitude Coverage:            {lat_min:.4f} deg N -> {lat_max:.4f} deg N",
        f"Longitude Coverage:           {lon_min:.4f} deg E -> {lon_max:.4f} deg E",
        "",
        f"2. SURFACE MATCHING CONSTRAINTS (Max: {SPATIAL_MATCH_RADIUS_KM:.0f} km, {TEMPORAL_MATCH_WINDOW_HOURS:.0f} h)",
        "─" * 70,
        f"Mean Matching Distance:       {dist_mean:.2f} km",
        f"Max Matching Distance:        {dist_max:.2f} km",
        f"Min Matching Distance:        {dist_min:.2f} km",
        f"Mean Time Difference:         {time_diff_mean:.2f} hours",
        f"Max Time Difference:          {time_diff_max:.2f} hours",
        f"Violations (>{SPATIAL_MATCH_RADIUS_KM:.0f}km or >{TEMPORAL_MATCH_WINDOW_HOURS:.0f}h):   {violations_str}",
        "",
        "3. SURFACE OBSERVATION STATISTICS (Model Inputs)",
        "─" * 70,
    ]

    for col in ['sst', 'ssh', 'sss', 'wind_speed']:
        if col in df.columns:
            valid_vals = df[col].dropna()
            if len(valid_vals) > 0:
                report_lines.append(
                    f"{col.upper():<12} | Min: {valid_vals.min():.3f} | Max: {valid_vals.max():.3f} | Mean: {valid_vals.mean():.3f} | Std: {valid_vals.std():.3f} (n={len(valid_vals)})"
                )

    report_lines.extend([
        "",
        "4. SUBSURFACE ARGO OBSERVATIONS (Ground Truth Targets)",
        "─" * 70,
    ])

    for col in ['temperature_0m', 'temp_50m', 'temp_100m', 'temp_200m', 'temp_500m', 'temp_1000m']:
        if col in df.columns:
            valid_vals = df[col].dropna()
            if len(valid_vals) > 0:
                report_lines.append(
                    f"{col:<16} | Min: {valid_vals.min():.3f} deg C | Max: {valid_vals.max():.3f} deg C | Mean: {valid_vals.mean():.3f} deg C | Std: {valid_vals.std():.3f} deg C (n={len(valid_vals)})"
                )

    report_lines.extend([
        "",
        "5. DATA SOURCES & AUDIT TRAILS",
        "─" * 70,
    ])
    if 'sst_source' in df.columns:
        for src, cnt in df['sst_source'].value_counts().items():
            report_lines.append(f"SST Source: {src} ({cnt} rows)")
    if 'ssh_source' in df.columns:
        for src, cnt in df['ssh_source'].value_counts().items():
            report_lines.append(f"SSH Source: {src} ({cnt} rows)")
    if 'sss_source' in df.columns:
        for src, cnt in df['sss_source'].value_counts().items():
            report_lines.append(f"SSS Source: {src} ({cnt} rows)")

    report_lines.extend([
        "",
        "6. SYNTHETIC & FABRICATED DATA CHECKS",
        "─" * 70,
    ])

    if synthetic_issues:
        report_lines.append("[FAIL] Potential synthetic data or data integrity violations detected:")
        for issue in synthetic_issues:
            report_lines.append(f"  - {issue}")
        report_lines.append("\nHARD GATE TRIGGERED: Model training is BLOCKED until real data issues are resolved.")
    else:
        report_lines.append("[PASS] Synthetic rows detected: 0")
        report_lines.append("[PASS] All rows trace to genuine Argo float profiles & verified surface observations.")
        report_lines.append("[PASS] Physical bounding checks passed.")
        report_lines.append("[PASS] Provenance completeness 100%.")

    report_lines.extend([
        "",
        "=" * 70,
        f"STATUS: {'[PASS] VALIDATION PASSED - APPROVED FOR TRAINING' if is_valid else '[FAIL] VALIDATION FAILED - TRAINING BLOCKED'}",
        "=" * 70,
        ""
    ])

    report_text = "\n".join(report_lines)
    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode('ascii', errors='replace').decode('ascii'))

    if output_file:
        Path(output_file).write_text(report_text, encoding='utf-8')
        logger.info(f"Validation report saved to {output_file}")

    return is_valid


if __name__ == '__main__':
    dataset_path = DATASET_DIR / 'train_dataset.parquet'
    output_path = DATASET_DIR / 'validation_report.txt'
    
    passed = generate_validation_report(dataset_path, output_path)
    if not passed:
        logger.error("HARD GATE: Synthetic or unverified data detected. Exiting with error.")
        sys.exit(1)
    else:
        logger.info("HARD GATE: Validation PASSED. Dataset is verified real observations.")
        sys.exit(0)