# NoiseMap

Temporally varying noise maps from [NoiseCapture](https://noise-planet.org/) (noise in dB, time, location). Data prep scripts produce a single table (GeoParquet/Parquet) for downstream ML (e.g. AEF embeddings + XGBoost).

**Data source:** [data.noise-planet.org/dump](https://data.noise-planet.org/dump/) (ODbL, 227 countries).

## Structure

- **data_prep/** — Download and prepare NoiseCapture zips into one table.
  - `download_data.py` — List countries; download country zips.
  - `prep_data.py` — Parse zips → GeoParquet/Parquet (lon, lat, noise_level_dB, time, country, time features).
  - See [data_prep/README.md](data_prep/README.md) for full usage.

## Setup

```bash
pip install -r requirements.txt
```

## Quick start

```bash
cd data_prep

# List available countries (227)
python download_data.py --list-countries

# Download all (or specific) countries
python download_data.py --output-dir ../noisecapture_data --all
# Or: --countries Belgium Austria France

# Prepare one table (run from repo root or data_prep)
python prep_data.py --input-dir ../noisecapture_data --output ../noisecapture_prepared.parquet
```

Raw zips go in `noisecapture_data/` (gitignored). You can also point `--input-dir` to an existing directory of zips (e.g. elsewhere on disk). Output table: `noisecapture_prepared.parquet` (or path you pass to `--output`).

## License

Code: use as you like. NoiseCapture data: ODbL ([noise-planet.org](https://noise-planet.org/)).
