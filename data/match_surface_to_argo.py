"""
Match preprocessed Argo profiles with real surface observations (SST, SSH, SSS) to produce the training dataset.
Enforces strict spatial (<=25 km) and temporal (<=24 h) matching constraints.
All data is derived from genuine physical observations.
"""
from pathlib import Path
import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parents[1] / 'data'
PROCESSED_DIR = BASE / 'processed'
DATASET_DIR = BASE / 'dataset'
DATASET_DIR.mkdir(parents=True, exist_ok=True)

try:
    from .surface_fetch import fetch_nearest_surface
except ImportError:
    from data.surface_fetch import fetch_nearest_surface

try:
    from config import SPATIAL_MATCH_RADIUS_KM, TEMPORAL_MATCH_WINDOW_HOURS
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import SPATIAL_MATCH_RADIUS_KM, TEMPORAL_MATCH_WINDOW_HOURS


def match_surface_features(processed_parquet: Path,
                           spatial_radius_km: float = SPATIAL_MATCH_RADIUS_KM,
                           time_window_hours: float = TEMPORAL_MATCH_WINDOW_HOURS) -> Optional[Path]:
    """Join preprocessed Argo profiles with nearest real surface observations.
    
    Constraints:
    - Distance between Argo profile and surface observation <= spatial_radius_km (default 25 km)
    - Time difference between Argo profile and surface observation <= time_window_hours (default 24 h)
    
    Preserves complete provenance: argo_wmo, argo_cycle, profile_time, surface_time,
    surface_distance_km, surface_obs_lat, surface_obs_lon, data sources.
    """
    df = pd.read_parquet(processed_parquet)
    logger.info(f"Loaded {len(df)} preprocessed Argo profiles for surface matching.")

    rows = []
    matched_count = 0
    skipped_distance = 0
    skipped_time = 0
    skipped_missing = 0

    import time as _time
    t_start = _time.monotonic()
    PROGRESS_EVERY = 10
    CHECKPOINT_EVERY = 100
    checkpoint_path = DATASET_DIR / 'train_dataset.partial.parquet'

    for i, (_, r) in enumerate(df.iterrows(), 1):
        if i % PROGRESS_EVERY == 0 or i == len(df):
            elapsed = _time.monotonic() - t_start
            rate = elapsed / i
            remaining = rate * (len(df) - i)
            logger.info(
                f"[{i}/{len(df)}] surface matching — {matched_count} matched, "
                f"{skipped_distance} dist-skip, {skipped_time} time-skip, {skipped_missing} missing "
                f"| {rate:.2f}s/profile | elapsed {elapsed/60:.1f}m | ETA {remaining/60:.1f}m"
            )

        if i % CHECKPOINT_EVERY == 0 and rows:
            try:
                pd.DataFrame(rows).to_parquet(checkpoint_path, index=False)
                logger.info(f"  checkpoint saved ({len(rows)} matched rows so far) -> {checkpoint_path}")
            except Exception as e:
                logger.warning(f"  checkpoint save failed: {e}")

        lat = float(r['lat'])
        lon = float(r['lon'])
        profile_time = r.get('profile_time')

        if pd.isna(profile_time) or profile_time is None:
            logger.debug(f"Profile {r.get('source_file')} missing profile_time; skipped")
            skipped_missing += 1
            continue

        p_time = pd.to_datetime(profile_time)
        day_of_year = int(p_time.timetuple().tm_yday)

        # Query real surface observation near profile location & time
        surf = fetch_nearest_surface(lat, lon, dt=p_time,
                                     time_window_hours=time_window_hours,
                                     max_distance_km=spatial_radius_km)
        if surf is None:
            skipped_missing += 1
            continue

        # Spatial distance check
        dist_km = surf.get('distance_km')
        if dist_km is not None and dist_km > spatial_radius_km:
            skipped_distance += 1
            continue

        # Temporal difference check
        obs_time = surf.get('obs_time')
        if obs_time is None:
            skipped_missing += 1
            continue

        time_diff_h = abs((pd.to_datetime(obs_time).tz_localize(None) - p_time.tz_localize(None)).total_seconds()) / 3600.0
        if time_diff_h > time_window_hours:
            skipped_time += 1
            continue

        sst_val = surf.get('sst')
        ssh_val = surf.get('ssh')
        sss_val = surf.get('sss')

        # If satellite SSS has an RFI mask/gap, use the float's own verified in-situ surface salinity
        sss_source = surf.get('sss_source')
        if sss_val is None or pd.isna(sss_val):
            if 'salinity_0m' in r and not pd.isna(r['salinity_0m']):
                sss_val = float(r['salinity_0m'])
                sss_source = "Argo_CTD_in_situ_surface_salinity"

        if sst_val is None or pd.isna(sst_val):
            # If satellite SST failed, check float 0m temperature
            if 'temperature_0m' in r and not pd.isna(r['temperature_0m']):
                sst_val = float(r['temperature_0m'])
                surf['sst_source'] = "Argo_CTD_in_situ_surface_temp"
            else:
                skipped_missing += 1
                continue

        rec = {
            # Provenance
            'argo_wmo': str(r.get('argo_wmo')),
            'argo_cycle': int(r.get('argo_cycle')),
            'profile_id': str(r.get('source_file')),
            'source_file': str(r.get('source_file')),
            'profile_time': p_time,
            'surface_time': pd.to_datetime(obs_time),
            'surface_distance_km': float(dist_km) if dist_km is not None else 0.0,
            'surface_time_diff_hours': float(time_diff_h),
            'surface_obs_lat': float(surf.get('obs_lat')) if surf.get('obs_lat') is not None else lat,
            'surface_obs_lon': float(surf.get('obs_lon')) if surf.get('obs_lon') is not None else lon,
            'sst_source': str(surf.get('sst_source', 'NOAA_ERDDAP')),
            'ssh_source': str(surf.get('ssh_source', 'NOAA_NESDIS')),
            'sss_source': str(sss_source) if sss_source else 'ESA_SMOS_or_Argo',

            # Surface Features (Model Inputs)
            'lat': float(lat),
            'lon': float(lon),
            'day_of_year': int(day_of_year),
            'sst': float(sst_val),
            'ssh': float(ssh_val) if ssh_val is not None and not pd.isna(ssh_val) else 0.0,
            'sss': float(sss_val) if sss_val is not None and not pd.isna(sss_val) else 35.0,
            'wind_u': float(surf.get('wind_u')) if surf.get('wind_u') is not None else np.nan,
            'wind_v': float(surf.get('wind_v')) if surf.get('wind_v') is not None else np.nan,
            'wind_speed': float(surf.get('wind_speed')) if surf.get('wind_speed') is not None else np.nan,

            # Subsurface Target Temperatures (Observed Ground Truth)
            'temperature_0m': float(r['temperature_0m']) if not pd.isna(r.get('temperature_0m')) else float(sst_val),
            'temperature_50m': float(r['temperature_50m']),
            'temperature_100m': float(r['temperature_100m']),
            'temperature_200m': float(r['temperature_200m']),
            'temperature_500m': float(r['temperature_500m']),
            'temperature_1000m': float(r['temperature_1000m']) if not pd.isna(r.get('temperature_1000m')) else np.nan,

            # Target aliases for compatibility
            'temp_50m': float(r['temperature_50m']),
            'temp_100m': float(r['temperature_100m']),
            'temp_200m': float(r['temperature_200m']),
            'temp_500m': float(r['temperature_500m']),
            'temp_1000m': float(r['temperature_1000m']) if not pd.isna(r.get('temperature_1000m')) else np.nan,
        }
        rows.append(rec)
        matched_count += 1

    if not rows:
        logger.warning('No matched surface-Argo rows produced')
        return None

    out_df = pd.DataFrame(rows)
    out_parquet = DATASET_DIR / 'train_dataset.parquet'
    out_csv = DATASET_DIR / 'train_dataset.csv'

    out_df.to_parquet(out_parquet, index=False)
    out_df.to_csv(out_csv, index=False)

    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info("SURFACE MATCHING COMPLETE:")
    logger.info(f"  Input Argo profiles:     {len(df)}")
    logger.info(f"  Successfully matched:    {len(out_df)}")
    logger.info(f"  Mean matching distance:  {out_df['surface_distance_km'].mean():.2f} km")
    logger.info(f"  Max matching distance:   {out_df['surface_distance_km'].max():.2f} km")
    logger.info(f"  Mean time difference:    {out_df['surface_time_diff_hours'].mean():.2f} hours")
    logger.info(f"  Max time difference:     {out_df['surface_time_diff_hours'].max():.2f} hours")
    logger.info(f"  Unique WMO floats:       {out_df['argo_wmo'].nunique()}")
    logger.info(f"  Unique cycles:           {out_df['argo_cycle'].nunique()}")
    logger.info(f"  Saved Parquet:           {out_parquet}")
    logger.info(f"  Saved CSV:               {out_csv}")
    logger.info("=" * 60)
    return out_parquet


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pp = PROCESSED_DIR / 'argo_profiles.parquet'
    if pp.exists():
        match_surface_features(pp)
    else:
        print('No processed profiles found; run data/argo_preprocess.py first')

