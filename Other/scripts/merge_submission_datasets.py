#!/usr/bin/env python3
"""Replace selected dataset blocks in one valid Biohub submission CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument("--use-alternative", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_blocks(path: Path) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    blocks: dict[str, list[dict[str, str]]] = defaultdict(list)
    order: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise RuntimeError(f"Unexpected schema in {path}: {reader.fieldnames}")
        for row in reader:
            dataset = row["dataset"]
            if dataset not in blocks:
                order.append(dataset)
            blocks[dataset].append(row)
    return order, dict(blocks)


def main() -> None:
    args = parse_args()
    base_order, base_blocks = read_blocks(args.base)
    _, alternative_blocks = read_blocks(args.alternative)
    selected = set(args.use_alternative)

    unknown = selected - set(base_blocks)
    missing = selected - set(alternative_blocks)
    if unknown:
        raise RuntimeError(f"Datasets absent from base submission: {sorted(unknown)}")
    if missing:
        raise RuntimeError(f"Datasets absent from alternative submission: {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    next_id = 0
    row_counts: dict[str, int] = {}
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for dataset in base_order:
            rows = alternative_blocks[dataset] if dataset in selected else base_blocks[dataset]
            row_counts[dataset] = len(rows)
            for source_row in rows:
                row = dict(source_row)
                row["id"] = str(next_id)
                writer.writerow(row)
                next_id += 1

    report = {
        "base": str(args.base),
        "base_sha256": sha256_file(args.base),
        "alternative": str(args.alternative),
        "alternative_sha256": sha256_file(args.alternative),
        "alternative_datasets": sorted(selected),
        "base_datasets": [dataset for dataset in base_order if dataset not in selected],
        "row_counts": row_counts,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": next_id,
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
