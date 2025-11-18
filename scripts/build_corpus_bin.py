#!/usr/bin/env python3
"""
Build a flat uint8 .bin file from one or more text directories.

- Reads all .txt files from --data-dir and any --extra-data-dir (repeatable).
- Concatenates documents with a sentinel between them (default: "\n\n<|doc|>\n\n").
- Writes bytes to --output (created/overwritten).
- Optionally caps to --max-bytes to make a smaller subset (e.g., ~10GiB).

The resulting .bin can be consumed by scripts/train_subtitles_transformer.py
via the --memmap-path flag (dtype=uint8).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Set

SENTINEL_DEFAULT = "\n\n<|doc|>\n\n"


def iter_txt_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.txt"))


def unique_ordered(paths: Sequence[Path]) -> List[Path]:
    seen: Set[Path] = set()
    out: List[Path] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def build_bin(
    data_dir: Path,
    output: Path,
    extra_data_dirs: Sequence[Path] | None = None,
    sentinel: str = SENTINEL_DEFAULT,
    max_bytes: int | None = None,
    encoding: str = "utf-8",
) -> int:
    directories: List[Path] = [data_dir]
    for extra in extra_data_dirs or []:
        try:
            ep = Path(extra)
        except TypeError:
            continue
        if ep not in directories:
            directories.append(ep)

    files: List[Path] = []
    for d in directories:
        files.extend(iter_txt_files(d))
    files = unique_ordered(files)

    if not files:
        raise FileNotFoundError(
            "No .txt files found under: " + ", ".join(str(d) for d in directories)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    sentinel_bytes = sentinel.encode(encoding)

    with open(output, "wb") as out:
        for i, path in enumerate(files):
            try:
                data = path.read_text(encoding=encoding, errors="ignore").strip()
            except Exception:
                data = ""
            if not data:
                continue
            payload = data.encode(encoding)

            # Prepend sentinel between documents (not before the first)
            if written > 0 and sentinel_bytes:
                chunk = sentinel_bytes
                if max_bytes is not None:
                    chunk = chunk[: max(0, max_bytes - written)]
                if chunk:
                    out.write(chunk)
                    written += len(chunk)
                    if max_bytes is not None and written >= max_bytes:
                        break

            # Write document bytes
            if max_bytes is None or written < max_bytes:
                remaining = None if max_bytes is None else max(0, max_bytes - written)
                if remaining is None:
                    out.write(payload)
                    written += len(payload)
                else:
                    chunk = payload[:remaining]
                    if chunk:
                        out.write(chunk)
                        written += len(chunk)
                if max_bytes is not None and written >= max_bytes:
                    break

            if (i + 1) % 500 == 0:
                print(f"Processed {i+1} files, bytes written: {written:,}")

    print(f"Done. Wrote {written:,} bytes to {output}")
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a flat uint8 corpus .bin")
    p.add_argument("--data-dir", type=Path, required=True, help="Primary text root directory")
    p.add_argument(
        "--extra-data-dir",
        type=Path,
        action="append",
        dest="extra_data_dirs",
        default=None,
        help="Additional text directory (repeatable)",
    )
    p.add_argument("--output", type=Path, required=True, help="Output .bin path")
    p.add_argument(
        "--sentinel",
        type=str,
        default=SENTINEL_DEFAULT,
        help="Document separator string (default matches trainer)",
    )
    p.add_argument(
        "--max-bytes",
        type=str,
        default=None,
        help="Optional cap for output size, supports suffixes k,m,g (e.g., 10g)",
    )
    return p.parse_args()


def parse_size(value: str | None) -> int | None:
    if value is None:
        return None
    s = value.strip().lower()
    if not s:
        return None
    unit = s[-1]
    num = s[:-1] if unit in {"k", "m", "g"} else s
    try:
        base = int(float(num))
    except ValueError:
        raise ValueError(f"Invalid --max-bytes value: {value}")
    if unit == "k":
        return base * 1024
    if unit == "m":
        return base * 1024 * 1024
    if unit == "g":
        return base * 1024 * 1024 * 1024
    return base


if __name__ == "__main__":
    args = parse_args()
    max_bytes = parse_size(args.max_bytes)
    build_bin(
        data_dir=args.data_dir,
        output=args.output,
        extra_data_dirs=args.extra_data_dirs,
        sentinel=args.sentinel,
        max_bytes=max_bytes,
    )
