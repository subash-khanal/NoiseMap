# Data prep: NoiseCapture → ML-ready table

Scripts to download and prepare [NoiseCapture](https://noise-planet.org/) (noise in dB, time, location) for modeling (e.g. temporally varying noise maps with AEF + XGBoost).

**Source:** [data.noise-planet.org/dump](https://data.noise-planet.org/dump/) (ODbL).

## 1. Download country zips

```bash
# From repo root (NoiseMap/) or from data_prep/
# List all available countries (227)
python data_prep/download_data.py --list-countries
# Or:  cd data_prep && python download_data.py --list-countries

# Download all countries into noisecapture_data/
python data_prep/download_data.py --output-dir noisecapture_data --all

# Download specific countries
python data_prep/download_data.py --output-dir noisecapture_data --countries Belgium Austria France
```

## 2. Prepare one table (GeoParquet / Parquet)

Reads all `*.zip` in the input directory, parses tracks (or points), adds time features, filters dB range, writes one Parquet/GeoParquet.

**Output columns:** `lon`, `lat`, `noise_level_dB`, `time_epoch`, `datetime_utc`, `hour`, `hour_sin`, `hour_cos`, `is_night`, `country`, and optionally `accuracy_m` (points), `geometry` (GeoParquet).

```bash
# From repo root; use paths relative to NoiseMap/
# Tracks (one row per session) → noisecapture_prepared.parquet
python data_prep/prep_data.py --input-dir noisecapture_data --output noisecapture_prepared.parquet

# Points (one row per second); subsample to 500k rows
python data_prep/prep_data.py --input-dir noisecapture_data --output noise_prepared.parquet --use-points --subsample 500000

# Plain Parquet (no geometry column)
python data_prep/prep_data.py --input-dir noisecapture_data --output out.parquet --no-geoparquet
```

**Options:**

- `--use-tracks` (default): one row per measurement session.
- `--use-points`: one row per second; use `--subsample N` to cap rows.
- `--max-accuracy-m`: drop points with GPS accuracy above this (meters).
- `--min-db`, `--max-db`: plausible LAeq range (default 30–110 dB).
- `--no-geoparquet`: write only lat/lon columns (no `geometry`); use if geopandas is not installed.

## Requirements

- `requests`, `pandas`, `pyarrow` (required).
- `geopandas`, `shapely` (optional, for GeoParquet with a geometry column).
