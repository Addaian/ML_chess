#!/usr/bin/env python3
"""Download a month of Lichess rated games with Stockfish evaluations.

Usage:
    python3 -m training.download_data
    python3 -m training.download_data --file lichess_db_standard_rated_2024-06.pgn.zst
"""

import argparse
import os
import subprocess

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
    subprocess.run(["curl", "-L", "--progress-bar", "-o", dest, url], check=True)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Lichess PGN database")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help="Filename from database.lichess.org/standard/")
    parser.add_argument("--dest", default="training/data", help="Destination directory")
    args = parser.parse_args()
    path = download(args.file, args.dest)
    print(f"Saved to: {path}")
