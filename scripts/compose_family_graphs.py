#!/usr/bin/env python3
"""Compose GEFF predictions from one source directory per dataset family."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="FAMILY=DIR",
        help="Source GEFF directory for a family; repeat for each family.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    sources: dict[str, Path] = {}
    for specification in args.source:
        family, separator, raw_path = specification.partition("=")
        if not separator or not family or not raw_path:
            raise ValueError(f"Expected FAMILY=DIR, received {specification!r}")
        if family in sources:
            raise ValueError(f"Duplicate family: {family}")
        path = Path(raw_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        sources[family] = path

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty {args.output_dir}")
    if args.receipt.exists():
        raise FileExistsError(f"Refusing to overwrite {args.receipt}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    names: set[str] = set()
    for family, source_dir in sorted(sources.items()):
        paths = sorted(source_dir.glob(f"{family}_*.geff"))
        if not paths:
            raise FileNotFoundError(f"No {family}_*.geff graphs in {source_dir}")
        for source_path in paths:
            if source_path.name in names:
                raise RuntimeError(f"Duplicate dataset graph: {source_path.name}")
            names.add(source_path.name)
            destination = args.output_dir / source_path.name
            shutil.copytree(source_path, destination, copy_function=shutil.copy2)
            selected.append(
                {
                    "dataset": source_path.stem,
                    "family": family,
                    "source": str(source_path),
                    "source_sha256_tree": sha256_tree(source_path),
                    "output_sha256_tree": sha256_tree(destination),
                }
            )

    mismatches = [row for row in selected if row["source_sha256_tree"] != row["output_sha256_tree"]]
    if mismatches:
        raise RuntimeError(f"Copy verification failed for {len(mismatches)} graphs")
    receipt = {
        "status": "complete",
        "sources": {family: str(path) for family, path in sorted(sources.items())},
        "output_dir": str(args.output_dir.resolve()),
        "datasets": selected,
        "dataset_count": len(selected),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
