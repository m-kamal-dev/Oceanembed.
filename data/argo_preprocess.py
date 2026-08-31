"""
Argo preprocessing: read raw NetCDFs (Argopy/xarray), apply Argo QC filtering, convert pressure->depth using TEOS-10 (gsw),
interpolate temperature onto fixed depth levels and produce a parquet table of profiles.
"""
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parents[1] / 'data'
RAW_DIR = BASE / 'raw' / 'argo'
PROCESSED_DIR = BASE / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DEPTHS = np.array([0, 50, 100, 200, 500, 1000], dtype=float)
# Require profile to cover at least near-surface (<=10m) and 500m for inclusion.
MIN_REQUIRED_MAX_DEPTH = 500.0
MAX_ALLOWED_MIN_DEPTH = 10.0


def pressure_to_depth(p, lat):
    """Convert pressure (dbar) to depth (m) using gsw when available.
    p: array-like pressures (dbar)
    lat: scalar latitude (degrees north)
    
    Uses TEOS-10 conversion via gsw.z_from_p (depth = -z).
    Logs an explicit WARNING if gsw is unavailable.
    """
    try:
        import gsw
        z = gsw.z_from_p(np.asarray(p, dtype=float), float(lat))  # returns negative z (height in m)
        return -np.asarray(z, dtype=float)
    except ImportError:
        logger.warning("GSW MODULE NOT AVAILABLE: Using approximate 1 dbar ≈ 1 m for pressure->depth. Install gsw for TEOS-10 accuracy.")
        return np.asarray(p, dtype=float)
    except Exception as e:
        logger.warning(f"gsw.z_from_p conversion failed ({e}); falling back to 1 dbar ≈ 1 m approximation")
        return np.asarray(p, dtype=float)


def _decode_qc_array(qc_arr):
    """Normalize QC array entries to single-char strings like '0'..'9'"""
    out = []
    for x in np.asarray(qc_arr).ravel():
        if isinstance(x, (bytes, np.bytes_)):
            try:
                out.append(x.decode('ascii'))
            except Exception:
                out.append(str(x))
        elif pd.isna(x):
            out.append('9')
        else:
            out.append(str(int(x)) if isinstance(x, (int, float, np.integer, np.floating)) and not np.isnan(x) else str(x))
    return np.array(out)


def _extract_profile_time(ds) -> Optional[pd.Timestamp]:
    """Extract profile timestamp from xarray Dataset and convert to pandas Timestamp."""
    for candidate in ['TIME', 'time', 'JULD', 'juld', 'JULD_LOCATION', 'PROFILE_TIME', 'DATE']:
        if candidate in ds.variables or candidate in ds.coords:
            try:
                tval = ds[candidate].values.ravel()[0]
                ts = pd.to_datetime(tval)
                if not pd.isna(ts):
                    return ts
            except Exception:
                continue
    return None


def process_netcdf_profile(nc_path: str) -> Optional[Dict[str, Any]]:
    """Read one NetCDF profile and return dict with metadata + interpolated temperatures or None if fails QC."""
    import xarray as xr
    p = Path(nc_path)
    if not p.exists() or p.stat().st_size < 100:
        return None

    try:
        ds = xr.open_dataset(str(p))
    except Exception as e:
        logger.warning(f"Failed to open NetCDF {p.name}: {e}")
        return None

    # 1. WMO (float identifier) and Cycle Number
    wmo = None
    cycle = None
    for candidate in ['PLATFORM_NUMBER', 'platform_number', 'WMO', 'wmo']:
        if candidate in ds.variables or candidate in ds.coords:
            try:
                raw_wmo = ds[candidate].values.ravel()[0]
                if isinstance(raw_wmo, (bytes, np.bytes_)):
                    wmo = raw_wmo.decode('utf-8').strip()
                else:
                    wmo = str(raw_wmo).strip()
                break
            except Exception:
                pass

    for candidate in ['CYCLE_NUMBER', 'cycle_number', 'CYCLE', 'cyc']:
        if candidate in ds.variables or candidate in ds.coords:
            try:
                cycle = int(ds[candidate].values.ravel()[0])
                break
            except Exception:
                pass

    # 2. Coordinates & Time
    lat = None
    lon = None
    for candidate in ['LATITUDE', 'latitude', 'lat']:
        if candidate in ds.variables or candidate in ds.coords:
            try:
                lat = float(ds[candidate].values.ravel()[0])
                break
            except Exception:
                pass

    for candidate in ['LONGITUDE', 'longitude', 'lon']:
        if candidate in ds.variables or candidate in ds.coords:
            try:
                lon = float(ds[candidate].values.ravel()[0])
                break
            except Exception:
                pass

    profile_time = _extract_profile_time(ds)

    if lat is None or lon is None or profile_time is None:
        logger.debug(f"Profile {p.name} missing coordinates or time; skipping")
        return None

    # 3. Temperature variable selection (prefer adjusted measurements if present)
    temp = None
    for candidate in ['TEMP_ADJUSTED', 'TEMP', 'temp_adjusted', 'temp', 'temperature']:
        if candidate in ds.variables:
            temp = ds[candidate].values.ravel()
            break
    if temp is None:
        logger.debug(f"No temperature variable in {p.name}")
        return None

    # 4. Temperature QC flags (standard Argo flags: '1'=good, '2'=probably good)
    temp_qc = None
    for candidate in ['TEMP_ADJUSTED_QC', 'TEMP_QC', 'temp_adjusted_qc', 'temp_qc']:
        if candidate in ds.variables:
            temp_qc = ds[candidate].values.ravel()
            break

    # 5. Pressure variable & QC
    pres = None
    for candidate in ['PRES_ADJUSTED', 'PRES', 'pres_adjusted', 'pres', 'pressure']:
        if candidate in ds.variables:
            pres = ds[candidate].values.ravel()
            break
    if pres is None:
        logger.debug(f"No pressure variable in {p.name}")
        return None

    pres_qc = None
    for candidate in ['PRES_ADJUSTED_QC', 'PRES_QC', 'pres_adjusted_qc', 'pres_qc']:
        if candidate in ds.variables:
            pres_qc = ds[candidate].values.ravel()
            break

    # 6. Salinity variable & QC (for in-situ surface salinity)
    psal = None
    for candidate in ['PSAL_ADJUSTED', 'PSAL', 'psal_adjusted', 'psal', 'salinity']:
        if candidate in ds.variables:
            psal = ds[candidate].values.ravel()
            break

    # Apply QC filtering: accept flags '1' (good) and '2' (probably good)
    accepted_qc = {'1', '2'}
    if temp_qc is not None:
        t_qc_chars = _decode_qc_array(temp_qc)
        t_good = np.isin(t_qc_chars, list(accepted_qc))
    else:
        t_good = ~np.isnan(temp)

    if pres_qc is not None:
        p_qc_chars = _decode_qc_array(pres_qc)
        p_good = np.isin(p_qc_chars, list(accepted_qc))
    else:
        p_good = ~np.isnan(pres)

    valid_mask = t_good & p_good & ~np.isnan(temp) & ~np.isnan(pres) & (pres >= 0)

    if np.sum(valid_mask) < 4:
        logger.debug(f"Profile {p.name} has too few QC-passed points: {np.sum(valid_mask)}")
        return None

    temp_valid = np.array(temp, dtype=float)[valid_mask]
    pres_valid = np.array(pres, dtype=float)[valid_mask]

    # Convert pressure (dbar) to depth (m) using TEOS-10 gsw
    depths = pressure_to_depth(pres_valid, lat)
    ok = ~np.isnan(temp_valid) & ~np.isnan(depths)
    if np.sum(ok) < 4:
        return None

    depths = depths[ok]
    temps = temp_valid[ok]

    # Sort strictly by increasing depth
    sort_idx = np.argsort(depths)
    depths = depths[sort_idx]
    temps = temps[sort_idx]

    # Remove duplicate depth points if any
    depths_unique, unique_idx = np.unique(depths, return_index=True)
    temps_unique = temps[unique_idx]

    min_d = float(depths_unique.min())
    max_d = float(depths_unique.max())

    # Check depth coverage: must reach >= 500m and start near surface (<= 10m)
    if min_d > MAX_ALLOWED_MIN_DEPTH or max_d < MIN_REQUIRED_MAX_DEPTH:
        logger.debug(f"Profile {p.name} depth coverage insufficient: {min_d:.1f}m - {max_d:.1f}m (requires <= {MAX_ALLOWED_MIN_DEPTH}m to >= {MIN_REQUIRED_MAX_DEPTH}m)")
        return None

    # Interpolate onto target depth levels strictly without extrapolation beyond observations
    try:
        from scipy.interpolate import interp1d
        f = interp1d(depths_unique, temps_unique, bounds_error=False, fill_value=np.nan)
        interp_temps = f(TARGET_DEPTHS)
    except Exception:
        logger.warning("scipy.interpolate not available; using numpy.interp")
        interp_temps = np.interp(TARGET_DEPTHS, depths_unique, temps_unique, left=np.nan, right=np.nan)

    # For 0m: if shallowest observed depth is <= 10m, map 0m to shallowest observation if NaN
    if np.isnan(interp_temps[0]) and min_d <= MAX_ALLOWED_MIN_DEPTH:
        interp_temps[0] = temps_unique[0]

    # In-situ surface salinity from float's CTD
    surface_salinity = np.nan
    if psal is not None:
        psal_valid = np.array(psal, dtype=float)[valid_mask][ok][sort_idx]
        valid_psal = psal_valid[~np.isnan(psal_valid)]
        if len(valid_psal) > 0:
            surface_salinity = float(valid_psal[0])

    # Core target depths: 50m, 100m, 200m, 500m
    core_idxs = [1, 2, 3, 4]  # 50m, 100m, 200m, 500m
    if np.isnan(interp_temps[core_idxs]).any():
        logger.debug(f"Profile {p.name} missing one of core target depths")
        return None

    rec = {
        'argo_wmo': str(wmo) if wmo is not None else p.stem.split('_')[0],
        'argo_cycle': int(cycle) if cycle is not None else 0,
        'source_file': p.name,
        'lat': float(lat),
        'lon': float(lon),
        'profile_time': profile_time,
        'temperature_0m': float(interp_temps[0]) if not np.isnan(interp_temps[0]) else np.nan,
        'temperature_50m': float(interp_temps[1]),
        'temperature_100m': float(interp_temps[2]),
        'temperature_200m': float(interp_temps[3]),
        'temperature_500m': float(interp_temps[4]),
        'temperature_1000m': float(interp_temps[5]) if not np.isnan(interp_temps[5]) else np.nan,
        'salinity_0m': float(surface_salinity) if not np.isnan(surface_salinity) else np.nan,
        'min_depth_m': min_d,
        'max_depth_m': max_d,
        'qc_passed_points': int(np.sum(valid_mask))
    }
    return rec


def build_processed_table(max_files: int = 1000) -> Optional[Path]:
    """Process files found in data/raw/argo/ and write data/processed/argo_profiles.parquet"""
    files = sorted(list(RAW_DIR.glob('*.nc')))[:max_files]
    if not files:
        logger.warning(f"No NetCDF files found in {RAW_DIR}")
        return None

    logger.info(f"Preprocessing {len(files)} raw NetCDF files (QC + TEOS-10 depth conversion)...")
    rows = []
    n_failed = 0
    for i, f in enumerate(files, start=1):
        try:
            out = process_netcdf_profile(str(f))
        except Exception as e:
            logger.warning(f"[{i}/{len(files)}] Unhandled error processing {f.name}: {e}")
            out = None
        if out is not None:
            rows.append(out)
        else:
            n_failed += 1
        if i % 25 == 0 or i == len(files):
            logger.info(f"[{i}/{len(files)}] processed so far — {len(rows)} passed QC, {n_failed} rejected/failed")

    if not rows:
        logger.warning("No processed profiles produced (all files failed QC or depth thresholds)")
        return None

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / 'argo_profiles.parquet'
    df.to_parquet(out_path, index=False)
    logger.info("=" * 60)
    logger.info("ARGO PREPROCESSING COMPLETE:")
    logger.info(f"  Files processed:      {len(files)}")
    logger.info(f"  Valid profiles:       {len(df)}")
    logger.info(f"  Unique WMO floats:    {df['argo_wmo'].nunique()}")
    logger.info(f"  Unique cycles:        {df['argo_cycle'].nunique()}")
    logger.info(f"  Depth range covered:  {df['min_depth_m'].min():.1f}m to {df['max_depth_m'].max():.1f}m")
    logger.info(f"  Saved path:           {out_path}")
    logger.info("=" * 60)
    return out_path


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    build_processed_table(max_files=200)

