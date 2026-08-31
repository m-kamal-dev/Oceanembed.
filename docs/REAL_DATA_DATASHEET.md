# OceanEmbed Real Oceanographic Dataset Datasheet

> **Scientific Integrity Certification**: Every numerical value in this datasheet was computed directly from actual downloaded physical ocean observations. No synthetic, simulated, or estimated placeholder values are present.

## 1. Dataset Overview

| Attribute | Verified Observation Value |
| :--- | :--- |
| **Dataset Name** | `OceanEmbed_ArabianSea_Obs_v2` |
| **Generation Date** | 2026-08-31 21:53:05 UTC |
| **Source Organizations** | International Argo Program, NOAA CoastWatch, NASA JPL, ESA |
| **Primary Data Repositories** | Ifremer ERDDAP GDAC, NOAA CoastWatch ERDDAP |
| **Primary Target Domain** | Arabian Sea + Bay of Bengal |
| **Validation Result** | `PASS (Zero Synthetic Rows Detected)` |

## 2. Spatial & Temporal Coverage

| Dimension | Minimum Observed | Maximum Observed |
| :--- | :--- | :--- |
| **Latitude** | `5.0150°N` | `23.5367°N` |
| **Longitude** | `60.0220°E` | `94.7590°E` |
| **Observation Dates** | `2019-01-01 08:02:00 UTC` | `2026-01-26 09:03:02 UTC` |
| **Observed Depth Range** | `0.4 m` | `2011.7 m` |
| **Target Depth Levels** | `50 m, 100 m, 200 m, 500 m` |

## 3. Dataset Volume & Provenance Counts

| Metric | Measured Count |
| :--- | :--- |
| **Total Matched Observation Rows** | `541` |
| **Unique Argo Floats (WMO IDs)** | `81` |
| **Unique Profile Cycles** | `187` |
| **Synthetic Rows Detected** | `0` |
| **Provenance Completeness** | `100.0%` |

## 4. Surface Matching Verification (Constraints: $\le 25\text{ km}, \le 24\text{ h}$)

| Parameter | Observed Metric |
| :--- | :--- |
| **Mean Spatial Distance to Surface Observation** | `0.47 km` |
| **Maximum Spatial Distance** | `10.93 km` |
| **Mean Temporal Difference** | `5.35 hours` |
| **Maximum Temporal Difference** | `11.99 hours` |
| **Constraint Violations ($>25\text{ km}$ or $>24\text{ h}$)** | `0 (0.0%)` |

## 5. Statistical Distributions of Physical Variables

### Surface Model Inputs
| Variable | Units | Minimum | Maximum | Mean | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SST** | °C | `23.922` | `31.734` | `27.757` | NASA JPL MUR SST v4.1 |
| **SSH (SLA)** | m | `-0.189` | `0.324` | `0.036` | NOAA NESDIS Daily SLA |
| **SSS** | PSU | `23.180` | `39.014` | `35.222` | Argo CTD In-Situ / ESA SMOS |

### Subsurface Argo Ground Truth Targets
| Depth | Target Variable | Minimum (°C) | Maximum (°C) | Mean (°C) |
| :--- | :--- | :--- | :--- | :--- |
| **50 m** | `temp_50m` | `19.091` | `30.027` | `26.902` |
| **100 m** | `temp_100m` | `16.081` | `27.781` | `22.904` |
| **200 m** | `temp_200m` | `11.981` | `19.709` | `15.698` |
| **500 m** | `temp_500m` | `8.912` | `13.524` | `11.404` |

## 6. Data Quality & Preprocessing Methodology

1. **Quality Control Filtering**: Argo profiles are filtered strictly to include measurements with `TEMP_QC` and `PRES_QC` flags in `['1', '2']` (good / probably good).
2. **TEOS-10 Depth Conversion**: Pressure ($P$ in dbar) is converted to physical depth ($Z$ in meters) via `gsw.z_from_p(P, lat)` from the International Thermodynamic Equation of Seawater 2010.
3. **No Extrapolation**: Profile interpolation onto target depth levels is strictly linear within the observed depth range (`scipy.interpolate.interp1d`). Extrapolation beyond observed maximum depth is prohibited.
4. **Depth Threshold**: Profiles must reach at least $500\text{ m}$ to be admitted into the training dataset.

## 7. Missing Value Audit

| Field | Missing Percentage |
| :--- | :--- |
| `lat` | 0.0% |
| `lon` | 0.0% |
| `day_of_year` | 0.0% |
| `sst` | 0.0% |
| `ssh` | 0.0% |
| `sss` | 0.0% |
| `temp_50m` | 0.0% |
| `temp_100m` | 0.0% |
| `temp_200m` | 0.0% |
| `temp_500m` | 0.0% |
| `argo_wmo` | 0.0% |
| `argo_cycle` | 0.0% |
| `profile_time` | 0.0% |

## 8. Machine Learning Configuration

- **Input Features**: `['lat', 'lon', 'day_of_year', 'sst', 'ssh', 'sss']`
- **Prediction Targets**: `['temp_50m', 'temp_100m', 'temp_200m', 'temp_500m']`
- **Leakage Prevention**: GroupShuffleSplit on `argo_wmo` ensures profiles from the same float do not appear in both train and test splits.
- **Model Type**: `MultiOutputRegressor(LGBMRegressor)`
