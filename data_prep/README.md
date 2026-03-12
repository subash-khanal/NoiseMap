# Data prep: NoiseCapture → one GeoParquet

Two scripts to get a single GeoParquet with everything (noise, time, AEF embeddings).

**Source:** [data.noise-planet.org/dump](https://data.noise-planet.org/dump/) (ODbL).

## 1. Download country zips

```bash
# List countries (227)
python download_data.py --list-countries

# Download all into noisecapture_data/
python download_data.py --output-dir noisecapture_data --all
```

## 2. Prepare one GeoParquet (noise + time + AEF_embed)

```bash
python prep_data.py --input-dir noisecapture_data --output noisecapture_prepared.parquet
```

Output: one GeoParquet with columns `lon`, `lat`, `noise_level_dB`, `time_epoch`, `datetime_utc`, `hour`, `hour_sin`, `hour_cos`, `is_night`, `country`, `AEF_embed`, `geometry`.

**Earth Engine:** Set `EARTHENGINE_CREDENTIALS` to your service account JSON path (e.g. in `~/.bashrc` or `~/.zshrc`), or run `earthengine authenticate` once.

## Requirements

- `requests`, `pandas`, `pyarrow`, `geopandas`, `shapely`, `earthengine-api`
