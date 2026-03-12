# NoiseMap

One GeoParquet from [NoiseCapture](https://noise-planet.org/) (noise in dB, time, location, AEF embeddings) for downstream ML.

**Data source:** [data.noise-planet.org/dump](https://data.noise-planet.org/dump/) (ODbL, 227 countries).

## Scripts (data_prep/)

1. **download_data.py** — Download country zips.
2. **prep_data.py** — Build one GeoParquet with noise, time features, and AEF_embed.

## Quick start

```bash
pip install -r requirements.txt
cd data_prep

python download_data.py --output-dir ../noisecapture_data --all
python prep_data.py --input-dir ../noisecapture_data --output ../noisecapture_prepared.parquet
```

Set `EARTHENGINE_CREDENTIALS` to your Earth Engine service account JSON path, or run `earthengine authenticate` once. Result: `noisecapture_prepared.parquet` (GeoParquet with everything).

## License

Code: use as you like. NoiseCapture data: ODbL ([noise-planet.org](https://noise-planet.org/)).
