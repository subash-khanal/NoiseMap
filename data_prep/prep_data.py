#!/usr/bin/env python3
"""
Prepare NoiseCapture zip data into one GeoParquet with everything: noise, time features, and AEF_embed.

Single workflow: reads country zips from --input-dir, writes one GeoParquet to --output with columns
lon, lat, noise_level_dB, time_epoch, datetime_utc, hour, hour_sin, hour_cos, is_night, country, AEF_embed, geometry.

Earth Engine auth: set EARTHENGINE_CREDENTIALS to your service account JSON path, or run earthengine authenticate once.

Usage:
  python prep_data.py --input-dir noisecapture_data --output noisecapture_prepared.parquet
"""

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Optional: GeoParquet (geopandas); fallback to plain Parquet with lat/lon
try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


def parse_tracks_from_geojson(features: list) -> list[dict]:
    """Extract (lon, lat, noise_level, time_epoch) from track features."""
    rows = []
    for feat in features:
        prop = feat.get("properties", {})
        noise_level = prop.get("noise_level")
        time_epoch = prop.get("time_epoch")
        if noise_level is None or time_epoch is None:
            continue
        geom = feat.get("geometry")
        if geom and geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            lon = sum(c[0] for c in coords) / len(coords)
            lat = sum(c[1] for c in coords) / len(coords)
        elif geom and geom.get("type") == "Point":
            lon, lat = geom["coordinates"][0], geom["coordinates"][1]
        else:
            continue
        rows.append({
            "lon": lon, "lat": lat, "noise_level_dB": float(noise_level), "time_epoch": int(time_epoch)
        })
    return rows


def _aef_initialize_earth_engine(credentials_path: Optional[Path] = None):
    """
    Initialize Earth Engine API (used when --add-aef).
    If credentials_path is set, use that service account JSON key file (no browser login).
    Otherwise use default credentials (from earthengine authenticate or GOOGLE_APPLICATION_CREDENTIALS).
    """
    import sys
    import ee
    if credentials_path is not None:
        path = Path(credentials_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Earth Engine credentials file not found: {path}")
        with open(path) as f:
            key_data = json.load(f)
        service_account = key_data.get("client_email")
        if not service_account:
            raise ValueError(f"JSON key at {path} has no 'client_email'. Use a valid Google Cloud service account key.")
        credentials = ee.ServiceAccountCredentials(service_account, str(path))
        ee.Initialize(credentials)
        return
    try:
        ee.Initialize()
    except Exception as e:
        err_msg = str(e).lower()
        if "not signed up" in err_msg or "not registered" in err_msg:
            raise RuntimeError(
                "Earth Engine access denied: being logged into Google is not enough. "
                "You must register for Earth Engine (separate signup): open https://signup.earthengine.google.com/ "
                "with the SAME Google account you use here, submit the form, and wait for the approval email. "
                "Then run again. Details: https://developers.google.com/earth-engine/guides/access"
            ) from e
        print(f"Error initializing Earth Engine: {e}", file=sys.stderr)
        print("Attempting to authenticate...", file=sys.stderr)
        try:
            import ee
            ee.Authenticate()
            ee.Initialize()
        except Exception as e2:
            msg2 = str(e2).lower()
            if "not signed up" in msg2 or "not registered" in msg2:
                raise RuntimeError(
                    "Earth Engine access denied: being logged into Google is not enough. "
                    "Register at https://signup.earthengine.google.com/ with the same Google account, "
                    "submit the form, wait for the approval email, then run again. "
                    "See https://developers.google.com/earth-engine/guides/access"
                ) from e2
            print(f"Authentication failed: {e2}", file=sys.stderr)
            raise ImportError(
                "AEF requires earthengine-api. Install: pip install earthengine-api, then run: earthengine authenticate"
            ) from e2


def _aef_get_embeddings_batch(
    coordinates: list,
    year: int = 2024,
    collection_name: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
    chunk_size: int = 1000,
):
    """
    Get Alpha Earth embeddings for list of (lat, lon). Returns (embeddings (N,64), valid_mask (N,)).
    """
    import sys
    import ee
    n_coords = len(coordinates)
    embeddings = np.zeros((n_coords, 64), dtype=np.float32)
    valid_mask = np.zeros(n_coords, dtype=bool)
    collection = ee.ImageCollection(collection_name)
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = start_date.advance(1, "year")
    image = collection.filterDate(start_date, end_date).mosaic()
    n_batches = (n_coords + chunk_size - 1) // chunk_size
    iterator = range(0, n_coords, chunk_size)
    iterator = tqdm(iterator, total=n_batches, desc="AEF embeddings", unit=" batch")
    for i in iterator:
        chunk_end = min(i + chunk_size, n_coords)
        chunk_coords = coordinates[i:chunk_end]
        features = []
        for local_idx, (lat, lon) in enumerate(chunk_coords):
            global_idx = i + local_idx
            feat = ee.Feature(ee.Geometry.Point([lon, lat]), {"idx": global_idx})
            features.append(feat)
        fc = ee.FeatureCollection(features)
        try:
            samples = image.sampleRegions(collection=fc, scale=30, geometries=False)
            result = samples.getInfo()
            for feature in result.get("features", []):
                props = feature.get("properties", {})
                idx = props.get("idx")
                if idx is None:
                    continue
                vals = [props.get(f"A{b:02d}") for b in range(64)]
                if any(v is None for v in vals):
                    vals = [props.get(f"b{b}") for b in range(64)]
                if any(v is None for v in vals):
                    all_numeric = []
                    for k, v in sorted(props.items()):
                        if isinstance(v, (int, float)) and k not in ("idx", "system:index"):
                            all_numeric.append(v)
                    if len(all_numeric) >= 64:
                        vals = all_numeric[:64]
                    else:
                        continue
                embeddings[idx] = vals
                valid_mask[idx] = True
        except Exception as e:
            print(f"Error processing batch {i}-{chunk_end}: {e}", file=sys.stderr)
    return embeddings, valid_mask


def add_aef_embeddings(
    df: pd.DataFrame,
    year: int = 2024,
    collection_name: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
    chunk_size: int = 1000,
    credentials_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Add AEF_embed column (Earth Engine Alpha Earth). Each row: list of 64 floats or None.
    credentials_path: optional path to service account JSON key; if set, no browser login.
    """
    _aef_initialize_earth_engine(credentials_path)
    coords = list(zip(df["lat"].tolist(), df["lon"].tolist()))
    embeddings, valid_mask = _aef_get_embeddings_batch(coords, year=year, collection_name=collection_name, chunk_size=chunk_size)
    df = df.copy()
    df["AEF_embed"] = [
        embeddings[i].tolist() if valid_mask[i] else None
        for i in range(len(df))
    ]
    n_valid = int(valid_mask.sum())
    print(f"AEF: {n_valid} / {len(df)} rows have embeddings (year={year}).")
    return df


def load_all_zips(input_dir: Path) -> list[dict]:
    """Load all country zips (tracks); return list of row dicts with 'country' key."""
    zips = sorted(input_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No *.zip files in {input_dir}")
    all_rows = []
    for zip_path in tqdm(zips, desc="Countries", unit="zip"):
        country = zip_path.stem
        n_before = len(all_rows)
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                if not name.endswith(".geojson") or "tracks" not in name:
                    continue
                with z.open(name) as f:
                    data = json.load(f)
                features = data.get("features", [])
                chunk = parse_tracks_from_geojson(features)
                for row in chunk:
                    row["country"] = country
                all_rows.extend(chunk)
        tqdm.write(f"  {country}: +{len(all_rows) - n_before} rows")
    return all_rows


def main():
    p = argparse.ArgumentParser(description="Prepare NoiseCapture zips into one GeoParquet (noise + time + AEF_embed)")
    p.add_argument("--input-dir", type=Path, default=Path("noisecapture_data"), help="Directory of country zip files")
    p.add_argument("--output", type=Path, default=Path("noisecapture_prepared.parquet"), help="Output GeoParquet path")
    p.add_argument("--aef-credentials", type=Path, default=None, help="Earth Engine service account JSON (or set EARTHENGINE_CREDENTIALS)")
    args = p.parse_args()

    print(f"Loading zips from {args.input_dir}...")
    all_rows = load_all_zips(args.input_dir)
    print(f"Total rows: {len(all_rows)}")

    df = pd.DataFrame(all_rows)
    if "noise_level_dB" not in df.columns and "noise_level" in df.columns:
        df = df.rename(columns={"noise_level": "noise_level_dB"})

    df["datetime_utc"] = pd.to_datetime(df["time_epoch"], unit="ms", utc=True)
    df["hour"] = df["datetime_utc"].dt.hour + df["datetime_utc"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["is_night"] = (df["hour"] >= 22) | (df["hour"] < 6)
    df = df[(df["noise_level_dB"] >= 30) & (df["noise_level_dB"] <= 110)]

    creds_path = args.aef_credentials or os.environ.get("EARTHENGINE_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        creds_path = Path(creds_path)
    df = add_aef_embeddings(df, credentials_path=creds_path)

    base_cols = ["lon", "lat", "noise_level_dB", "time_epoch", "datetime_utc", "hour", "hour_sin", "hour_cos", "is_night", "country"]
    extra = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + extra]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_path = args.output
    if out_path.suffix.lower() != ".parquet" and out_path.suffix.lower() != ".geoparquet":
        out_path = out_path.with_suffix(".parquet")
    if HAS_GEOPANDAS:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326"))
        gdf.to_parquet(out_path, index=False)
        print(f"Wrote GeoParquet: {out_path}")
    else:
        df.to_parquet(out_path, index=False)
        print(f"Wrote Parquet (no geometry): {out_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
