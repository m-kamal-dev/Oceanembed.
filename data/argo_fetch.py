"""
Argo fetch helper.
Uses Argopy to discover and download canonical Argo profile NetCDFs into data/raw/argo/.
Downloads genuine Argo observations (PRES, TEMP, PSAL, QC flags, WMO, CYCLE, TIME, LAT, LON)
via authoritative GDAC / ERDDAP servers.
"""
from pathlib import Path
import logging
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)
RAW_DIR = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'argo'
RAW_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHUNK_DAYS = 45  # don't split narrower than this even if still oversized


def _is_too_large_error(e: Exception) -> bool:
    """ERDDAP/proxy rejects oversized requests with HTTP 413. argopy wraps this as
    ErddapServerError, often with an empty message, so check the exception chain too."""
    if type(e).__name__ == 'ErddapServerError':
        return True
    if '413' in str(e):
        return True
    cause = e.__cause__ or e.__context__
    if cause is not None and ('413' in str(cause) or 'ClientResponseError' in type(cause).__name__):
        return True
    return False


def _fetch_index_chunk(DataFetcher, min_lon, max_lon, min_lat, max_lat, start_date, end_date):
    """Fetch the Argo index for one date sub-range; recursively bisects the date range
    on a 413 (payload too large) response and merges the halves. Returns a DataFrame
    (possibly empty) or None if this sub-range could not be fetched at all."""
    box = [min_lon, max_lon, min_lat, max_lat, 0.0, 1000.0, start_date, end_date]
    try:
        fetcher = DataFetcher(src='erddap')
        return fetcher.region(box).to_index()
    except Exception as e:
        span_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        if _is_too_large_error(e) and span_days > MIN_CHUNK_DAYS:
            mid_date = (pd.Timestamp(start_date) + (pd.Timestamp(end_date) - pd.Timestamp(start_date)) / 2).strftime('%Y-%m-%d')
            logger.warning(f"Index request too large for {start_date}..{end_date} ({type(e).__name__}); splitting at {mid_date} and retrying both halves")
            left = _fetch_index_chunk(DataFetcher, min_lon, max_lon, min_lat, max_lat, start_date, mid_date)
            right = _fetch_index_chunk(DataFetcher, min_lon, max_lon, min_lat, max_lat, mid_date, end_date)
            frames = [f for f in (left, right) if f is not None and len(f) > 0]
            return pd.concat(frames, ignore_index=True) if frames else None
        else:
            logger.error(f"Failed to fetch Argo index for {start_date}..{end_date} ({type(e).__name__}): {e or '<no message>'}")
            logger.exception("Full traceback for index fetch failure:")
            return None


def fetch_profiles_bbox(start_date: str, end_date: str,
                        min_lat: float = 8.0, max_lat: float = 24.0,
                        min_lon: float = 60.0, max_lon: float = 77.0,
                        max_profiles: int = 50) -> List[str]:
    """Fetch Argo profiles for a bounding box and date range using the Argopy API.
    1. Queries the Argo index for the spatial-temporal box [lon_min, lon_max, lat_min, lat_max, 0, 1000, start, end]
    2. Selects up to max_profiles across multiple WMO floats and cycles
    3. Fetches individual profiles into genuine xarray Datasets
    4. Saves canonical NetCDF files to data/raw/argo/WMO{wmo}_CYC{cyc}.nc
    
    Returns list of local NetCDF file paths.
    """
    try:
        from argopy import DataFetcher
    except ImportError as e:
        logger.error(f"argopy is not available: {e}")
        return []

    logger.info(f"Querying Argo index for box: Lon [{min_lon}, {max_lon}], Lat [{min_lat}, {max_lat}], Dates: {start_date} to {end_date}")
    
    idx_df = _fetch_index_chunk(DataFetcher, min_lon, max_lon, min_lat, max_lat, start_date, end_date)

    if idx_df is None or len(idx_df) == 0:
        logger.warning("Argopy returned zero profiles for the requested bbox/date range")
        return []

    # Chunking + retries can occasionally produce overlapping rows at chunk boundaries; drop exact dupes.
    dedupe_cols = [c for c in ['wmo', 'cyc', 'WMO', 'CYC'] if c in idx_df.columns]
    if dedupe_cols:
        idx_df = idx_df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)

    # Standardize column names (wmo, cyc, date, latitude, longitude)
    idx_df.columns = [c.lower() for c in idx_df.columns]
    
    total_discovered = len(idx_df)
    unique_wmos_disc = idx_df['wmo'].nunique() if 'wmo' in idx_df else 0
    logger.info(f"Discovered {total_discovered} real Argo profiles across {unique_wmos_disc} unique floats in region.")

    # Select profiles up to max_profiles, prioritizing float diversity
    if len(idx_df) > max_profiles:
        # Sample or select evenly across unique WMOs
        selected_rows = []
        grouped = idx_df.groupby('wmo')
        wmo_list = list(grouped.groups.keys())
        wmo_idx = 0
        cycle_indices = {w: 0 for w in wmo_list}
        
        while len(selected_rows) < max_profiles:
            w = wmo_list[wmo_idx % len(wmo_list)]
            w_group = grouped.get_group(w)
            c_idx = cycle_indices[w]
            if c_idx < len(w_group):
                selected_rows.append(w_group.iloc[c_idx])
                cycle_indices[w] += 1
            wmo_idx += 1
            if all(cycle_indices[w] >= len(grouped.get_group(w)) for w in wmo_list):
                break
        selected_df = pd.DataFrame(selected_rows)
    else:
        selected_df = idx_df.copy()

    logger.info(f"Selected {len(selected_df)} profiles across {selected_df['wmo'].nunique()} floats for download.")

    saved_profiles = []
    wmos_fetched = set()
    cycles_fetched = set()

    for i, (_, row) in enumerate(selected_df.iterrows(), 1):
        try:
            wmo = int(row['wmo'])
            cyc = int(row['cyc'])
            out_file = RAW_DIR / f"WMO{wmo}_CYC{cyc}.nc"

            # If already downloaded and valid NetCDF, reuse
            if out_file.exists() and out_file.stat().st_size > 500:
                logger.debug(f"Profile WMO {wmo} Cyc {cyc} already cached at {out_file.name}")
                saved_profiles.append(str(out_file))
                wmos_fetched.add(wmo)
                cycles_fetched.add((wmo, cyc))
                continue

            logger.info(f"[{i}/{len(selected_df)}] Fetching Argo profile WMO {wmo} Cycle {cyc}...")
            p_fetcher = DataFetcher(src='erddap')
            p_ds = p_fetcher.profile(wmo, cyc).to_xarray()

            if p_ds is not None and 'PRES' in p_ds and 'TEMP' in p_ds:
                p_ds.to_netcdf(out_file)
                saved_profiles.append(str(out_file))
                wmos_fetched.add(wmo)
                cycles_fetched.add((wmo, cyc))
                logger.info(f"Saved {out_file.name} ({out_file.stat().st_size / 1024:.1f} KB)")
            else:
                logger.warning(f"Profile WMO {wmo} Cycle {cyc} missing required variables; skipped")
        except Exception as e:
            logger.warning(f"Failed to fetch profile WMO {row.get('wmo')} Cycle {row.get('cyc')}: {e}")

    logger.info("=" * 60)
    logger.info("ARGO FETCH COMPLETE:")
    logger.info(f"  Profiles discovered: {total_discovered}")
    logger.info(f"  Profiles selected:   {len(selected_df)}")
    logger.info(f"  Profiles saved:      {len(saved_profiles)}")
    logger.info(f"  Unique WMO floats:   {len(wmos_fetched)}")
    logger.info(f"  Unique cycles:       {len(cycles_fetched)}")
    logger.info(f"  Saved directory:     {RAW_DIR}")
    logger.info("=" * 60)

    return saved_profiles


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    profiles = fetch_profiles_bbox(
        '2026-01-01', '2026-02-01',
        min_lon=68.0, max_lon=72.0,
        min_lat=12.0, max_lat=16.0,
        max_profiles=5
    )
    print(f"Fetched {len(profiles)} profiles")

