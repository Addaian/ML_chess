#!/usr/bin/env python3
"""Download a month of Lichess rated games with Stockfish evaluations.

Usage:
    python3 -m training.download_data
    python3 -m training.download_data --file lichess_db_standard_rated_2024-06.pgn.zst
"""

import argparse
import os
import urllib.request

BASE_URL = "https://database.lichess.org/standard/"
DEFAULT_FILE = "lichess_db_standard_rated_2024-01.pgn.zst"


def download(filename: str, dest_dir: str = "training/data") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    url = BASE_URL + filename
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        return dest
    print(f"Downloading {url}")
    print("(Large file — several GB. Ctrl+C to cancel.)")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()
    return dest


def _progress(count, block_size, total_size):
    mb = count * block_size / 1_048_576
    total_mb = total_size / 1_048_576
    print(f"\r  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Lichess PGN database")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help="Filename from database.lichess.org/standard/")
    parser.add_argument("--dest", default="training/data", help="Destination directory")
    args = parser.parse_args()
    path = download(args.file, args.dest)
    print(f"Saved to: {path}")
