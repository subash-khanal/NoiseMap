# NoiseMap

Predict environmental noise (dB) from [AlphaEarth Foundations (AEF)](https://source.coop/tge-labs/aef) embeddings. **Why AEF?** AEF gives dense 64-d embeddings from satellite imagery; using them as input lets the model use land use and geography without hand-crafted features and **generalize spatially**—we can query AEF for any location and predict noise where we have no measurements, so we can build noisemaps for arbitrary regions and times.

## Pipeline (NoiseMap)

### 1. Prepare data

From the repo root:

```bash
cd data_prep

# Optional: list available countries (227)
python download_data.py --list-countries

# Download NoiseCapture country zips into a directory (e.g. ../noisecapture_data)
python download_data.py --output-dir ../noisecapture_data --all

# Build one GeoParquet and add AEF at each point (requires Earth Engine auth)
python prep_data.py --input-dir ../noisecapture_data --output ../noisecapture_prepared.parquet
```

**Result:** `noisecapture_prepared.parquet` (columns: `lon`, `lat`, `noise_level_dB`, `time_epoch`, `datetime_utc`, `hour`, `hour_sin`, `hour_cos`, `is_night`, `country`, `AEF_embed`, `geometry`).  
Set `EARTHENGINE_CREDENTIALS` to your Earth Engine service account JSON path, or run `earthengine authenticate` once before the last step.

### 2. Train model

From the repo root:

```bash
python train_noise_model.py \
  --data data_prep/noisecapture_prepared.parquet \
  --epochs 30 \
  --out-dir checkpoints
```

**Result:** `checkpoints/noise_mlp_checkpoint.pt` and `checkpoints/noise_mlp_preprocess.joblib`. Optional: set `WANDB_PROJECT=noise-map` (or use `--wandb-project`) for wandb logging.

### 3. Infer / dynamic noisemap

Open **dynamic_noisemap.ipynb** in [Google Colab](https://colab.research.google.com/) (or locally). The notebook downloads the parquet, checkpoint, and preprocess automatically, so you can skip steps 1–2. Run the cells: load data, load model, then draw a region on the map and run the demo to get a predicted noise (dB) heatmap for that area.

**Pre-built artifacts** (parquet, checkpoint, preprocess) are in this [Google Drive folder](https://drive.google.com/drive/folders/1lbL3juj2hkUhBTDJL78wvlb5F6SIDtAe); the notebook is configured to use them when run on Colab.

---

## Bonus

**aef_interactive_exploration.ipynb** — A separate notebook for exploring [AlphaEarth Foundations (AEF)](https://source.coop/tge-labs/aef) embeddings. It is not part of the noise pipeline. You can draw regions on a map, load AEF for those areas, and work with prepped OSM/mask data to inspect embeddings and tags. Useful if you want to learn simple ways to query and use AEF (e.g. with [aef-loader](https://pypi.org/project/aef-loader/)) for other geospatial tasks. Prepped data for Part 2 is also available in the same [Google Drive folder](https://drive.google.com/drive/folders/1lbL3juj2hkUhBTDJL78wvlb5F6SIDtAe) (`osm_prepped_data.pkl`).
