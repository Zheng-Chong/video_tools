#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def cleanup(root: Path, dry_run: bool) -> tuple[int, int]:
    """
    For each immediate subdirectory under `root` (movie folder):
      - if `asd_done.txt` does NOT exist in that folder
      - delete its `asd/` subfolder if present
    """
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    scanned = 0
    removed = 0

    for movie_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        scanned += 1
        done_flag = movie_dir / "asd_done.txt"
        asd_dir = movie_dir / "asd"

        if done_flag.exists():
            continue

        if asd_dir.exists() and asd_dir.is_dir():
            print(f"[REMOVE]{' (dry-run)' if dry_run else ''} {asd_dir}")
            removed += 1
            if not dry_run:
                shutil.rmtree(asd_dir)

    return scanned, removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove movie/asd directories when movie/asd_done.txt is missing."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Datasets/AVAGen"),
        help="Root directory that contains movie folders (default: Datasets/AVAGen)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting anything",
    )
    args = parser.parse_args()

    scanned, removed = cleanup(args.root, args.dry_run)
    print(f"[DONE] scanned={scanned} removed={removed} root={args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

