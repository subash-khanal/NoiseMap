#!/usr/bin/env python3
"""
Download NoiseCapture country zip files from https://data.noise-planet.org/dump/.

Usage:
  python download_data.py --output-dir noisecapture_data --all
  python download_data.py --output-dir noisecapture_data --countries Belgium Austria France
  python download_data.py --list-countries  # only print available countries, no download
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import requests

DUMP_BASE_URL = "https://data.noise-planet.org/dump"


def get_available_countries() -> list[str]:
    """Fetch dump index (HTML) and return sorted list of country names."""
    r = requests.get(f"{DUMP_BASE_URL}/")
    r.raise_for_status()
    countries = sorted(
        set(
            unquote(m).replace(".zip", "").split("/")[-1]
            for m in re.findall(r'href="([^"]+\.zip)"', r.text)
        )
    )
    return countries


def download_country(country: str, output_dir: Path, skip_existing: bool = True) -> bool:
    """Download one country zip. Returns True if downloaded or already exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{country}.zip"
    if skip_existing and zip_path.exists():
        return True
    url = f"{DUMP_BASE_URL}/{country.replace(' ', '%20')}.zip"
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=2**20):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Error {country}: {e}", file=sys.stderr)
        return False


def main():
    p = argparse.ArgumentParser(description="Download NoiseCapture country zips from data.noise-planet.org")
    p.add_argument("--output-dir", type=Path, default=Path("noisecapture_data"), help="Directory to save zip files")
    p.add_argument("--list-countries", action="store_true", help="Only list available countries and exit")
    p.add_argument("--all", action="store_true", help="Download all available countries")
    p.add_argument("--countries", nargs="+", help="Space-separated country names to download")
    p.add_argument("--no-skip-existing", action="store_true", help="Re-download even if zip already exists")
    args = p.parse_args()

    countries = get_available_countries()
    print(f"Available countries: {len(countries)}")

    if args.list_countries:
        for c in countries:
            print(c)
        return

    if args.all:
        to_download = countries
    elif args.countries:
        to_download = [c for c in args.countries if c in countries]
        missing = set(args.countries) - set(to_download)
        if missing:
            print(f"Warning: not in dump index: {missing}", file=sys.stderr)
    else:
        print("Use --all or --countries COUNTRY ... to download. Use --list-countries to see names.")
        return

    for i, country in enumerate(to_download, 1):
        if download_country(country, args.output_dir, skip_existing=not args.no_skip_existing):
            size_mb = (args.output_dir / f"{country}.zip").stat().st_size / 1e6
            print(f"[{i}/{len(to_download)}] {country}: {size_mb:.1f} MB")
        else:
            print(f"[{i}/{len(to_download)}] {country}: failed")


if __name__ == "__main__":
    main()
