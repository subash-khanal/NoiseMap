#!/usr/bin/env python3
"""
Prepare NoiseCapture zip data into a single GeoParquet (or Parquet) table for ML / analysis.

Output columns: lon, lat, noise_level_dB, time_epoch, datetime_utc, hour, hour_sin, hour_cos,
is_night, country, accuracy (if points), and geometry (Point, WGS84) for GeoParquet.

Usage:
  python prep_data.py --input-dir noisecapture_data --output noisecapture_prepared.parquet
  python prep_data.py --input-dir noisecapture_data --output out.geoparquet --use-points --subsample 500000
"""

import argparse
import json
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


def parse_points_from_geojson(features: list, max_accuracy_m: Optional[float] = None) -> list[dict]:
    """Extract (lon, lat, noise_level, time_epoch, accuracy) from point features."""
    rows = []
    for feat in features:
        prop = feat.get("properties", {})
        noise_level = prop.get("noise_level")
        time_epoch = prop.get("time_epoch")
        accuracy = prop.get("accuracy")
        if noise_level is None or time_epoch is None:
            continue
        if max_accuracy_m is not None and accuracy is not None and float(accuracy) > max_accuracy_m:
            continue
        geom = feat.get("geometry")
        if not geom or geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][0], geom["coordinates"][1]
        rows.append({
            "lon": lon, "lat": lat, "noise_level_dB": float(noise_level), "time_epoch": int(time_epoch),
            "accuracy_m": float(accuracy) if accuracy is not None else None,
        })
    return rows


def load_all_zips(
    input_dir: Path,
    use_tracks: bool = True,
    max_accuracy_m: Optional[float] = 100.0,
) -> list[dict]:
    """Load all country zips from input_dir; return list of row dicts (with 'country' key)."""
    zips = sorted(input_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No *.zip files in {input_dir}")

    all_rows = []
    for zip_path in tqdm(zips, desc="Countries", unit="zip"):
        country = zip_path.stem
        n_before = len(all_rows)
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                if not name.endswith(".geojson"):
                    continue
                if use_tracks and "tracks" not in name:
                    continue
                if not use_tracks and "points" not in name:
                    continue
                with z.open(name) as f:
                    data = json.load(f)
                features = data.get("features", [])
                chunk = (
                    parse_tracks_from_geojson(features)
                    if use_tracks
                    else parse_points_from_geojson(features, max_accuracy_m)
                )
                for row in chunk:
                    row["country"] = country
                all_rows.extend(chunk)
        added = len(all_rows) - n_before
        tqdm.write(f"  {country}: +{added} rows")
    return all_rows


def main():
    p = argparse.ArgumentParser(description="Prepare NoiseCapture zips into one GeoParquet/Parquet table")
    p.add_argument("--input-dir", type=Path, default=Path("noisecapture_data"), help="Directory of country zip files")
    p.add_argument("--output", type=Path, default=Path("noisecapture_prepared.parquet"), help="Output file (.parquet or .geoparquet)")
    p.add_argument("--use-tracks", action="store_true", default=True, help="Use tracks (one row per session) [default]")
    p.add_argument("--use-points", action="store_true", help="Use points (one row per second); more rows")
    p.add_argument("--max-accuracy-m", type=float, default=100.0, help="Drop points with GPS accuracy > this (meters)")
    p.add_argument("--min-db", type=float, default=30.0, help="Minimum plausible noise_level_dB")
    p.add_argument("--max-db", type=float, default=110.0, help="Maximum plausible noise_level_dB")
    p.add_argument("--subsample", type=int, default=None, help="Random subsample to this many rows (for points)")
    p.add_argument("--no-geoparquet", action="store_true", help="Write plain Parquet (lat/lon only), no geometry column")
    args = p.parse_args()

    use_tracks = args.use_tracks and not args.use_points
    print(f"Loading zips from {args.input_dir} (use_tracks={use_tracks})...")
    all_rows = load_all_zips(args.input_dir, use_tracks=use_tracks, max_accuracy_m=args.max_accuracy_m if not use_tracks else None)
    print(f"Total rows: {len(all_rows)}")

    df = pd.DataFrame(all_rows)
    # Standardise column name
    if "noise_level_dB" not in df.columns and "noise_level" in df.columns:
        df = df.rename(columns={"noise_level": "noise_level_dB"})

    # Time features (time_epoch in dump is milliseconds)
    df["datetime_utc"] = pd.to_datetime(df["time_epoch"], unit="ms", utc=True)
    df["hour"] = df["datetime_utc"].dt.hour + df["datetime_utc"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["is_night"] = (df["hour"] >= 22) | (df["hour"] < 6)

    # Filter dB range
    df = df[(df["noise_level_dB"] >= args.min_db) & (df["noise_level_dB"] <= args.max_db)]

    if args.subsample is not None and len(df) > args.subsample:
        df = df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows")

    # Column order: identity, measure, time, derived, meta
    base_cols = ["lon", "lat", "noise_level_dB", "time_epoch", "datetime_utc", "hour", "hour_sin", "hour_cos", "is_night", "country"]
    extra = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + extra]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    out_path = args.output
    written = False
    if out_path.suffix.lower() in (".parquet", ".geoparquet"):
        try:
            if HAS_GEOPANDAS and not args.no_geoparquet:
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326"))
                gdf.to_parquet(out_path, index=False)
                print(f"Wrote GeoParquet: {out_path}")
            else:
                df.to_parquet(out_path, index=False)
                print(f"Wrote Parquet: {out_path}")
            written = True
        except Exception as e:
            print(f"Parquet write failed ({e}). Writing CSV instead.", file=__import__("sys").stderr)
            out_path = out_path.with_suffix(".csv")
    if not written:
        df.to_csv(out_path, index=False)
        print(f"Wrote CSV: {out_path}")

    print(f"Columns: {list(df.columns)}")
    print(f"noise_level_dB: min={df['noise_level_dB'].min():.1f}, max={df['noise_level_dB'].max():.1f}")


if __name__ == "__main__":
    main()
