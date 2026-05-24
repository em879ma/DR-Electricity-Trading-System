"""Download OPSD and Ember data to data/raw/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_download import download_ember, download_opsd


def main(force: bool = False) -> None:
    print("=== Downloading data ===")
    opsd_path = download_opsd(force=force)
    print(f"OPSD data: {opsd_path}")
    ember_path = download_ember(force=force)
    if ember_path:
        print(f"Ember data: {ember_path}")
    else:
        print("Ember data not available — will use proxies for energy-mix features.")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OPSD and Ember data.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args()
    main(force=args.force)
