# NoiseMap

One GeoParquet from [NoiseCapture](https://noise-planet.org/) (noise in dB, time, location, AEF embeddings) for downstream ML. Train a PyTorch model with **train_noise_model.py**; use **noise_ml.ipynb** for inference (evaluate + heatmap demo).

**Data source:** [data.noise-planet.org/dump](https://data.noise-planet.org/dump/) (ODbL, 227 countries).

## Project layout

| Path | Purpose |
|------|--------|
| **environment.yml** | Conda env (Python 3.11, PyTorch + CUDA 12.1 for GPU). |
| **data_prep/** | Download zips, build parquet (noise + time + AEF_embed). |
| **train_noise_model.py** | Train the noise-prediction MLP (wandb); writes checkpoint + preprocess. |
| **noise_ml.ipynb** | Inference only: load saved model, EDA, evaluate, heatmap demo. |

## Scripts (data_prep/)

1. **download_data.py** — Download country zips.
2. **prep_data.py** — Build one GeoParquet with noise, time features, and AEF_embed.

## Conda environment

Use **Python 3.10 or 3.11**. The env installs **PyTorch with CUDA 12.1** for GPU by default.

```bash
conda env create -f environment.yml
conda activate noisemap
```

If your driver uses a different CUDA version (e.g. 11.8), edit `environment.yml` and set `pytorch-cuda=11.8` (or see [pytorch.org](https://pytorch.org)). For CPU-only, remove the `pytorch-cuda=12.1` line.

## Quick start

Create the conda env (GPU by default), then build the parquet:

```bash
conda env create -f environment.yml
conda activate noisemap

cd data_prep
python download_data.py --output-dir ../noisecapture_data --all
python prep_data.py --input-dir ../noisecapture_data --output ../noisecapture_prepared.parquet
```

Alternatively: `pip install -r requirements.txt` (no conda).

Set `EARTHENGINE_CREDENTIALS` to your Earth Engine service account JSON path, or run `earthengine authenticate` once. Result: `noisecapture_prepared.parquet` (GeoParquet with everything).

## Training (script) and inference (notebook)

All **model training** is in **train_noise_model.py**. The notebook **noise_ml.ipynb** is for **inference only** (load saved model, evaluate, heatmap).

### 1. Train (run once)

```bash
conda activate noisemap   # or: pip install -r requirements.txt
python train_noise_model.py --data data_prep/noisecapture_prepared.parquet --epochs 30 --out-dir checkpoints
```

This writes `checkpoints/noise_mlp_checkpoint.pt` and `checkpoints/noise_mlp_preprocess.joblib`.

### 2. Infer (notebook)

Open **noise_ml.ipynb**, set `CKPT_DIR = Path('checkpoints')` (or your output folder), then run:

- **Load data** + **EDA** (optional).
- **Load model** — loads checkpoint and preprocess from `CKPT_DIR`.
- **Evaluate** — test-set metrics and predicted vs actual plot.
- **Heatmap demo** — set `bbox` and `hour`; runs the model on AEF for that region and shows a noise (dB) map. For the heatmap, install: `pip install aef-loader odc-geo pyproj nest_asyncio`.

## License

Code: use as you like. NoiseCapture data: ODbL ([noise-planet.org](https://noise-planet.org/)).
