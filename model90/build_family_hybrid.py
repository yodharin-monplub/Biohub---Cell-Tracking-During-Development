#!/usr/bin/env python3
"""Build a deterministic Biohub submission by routing whole dataset families."""

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
    parser.add_argument("--alternative-family", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
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


def family(dataset: str) -> str:
    return dataset.split("_", 1)[0]


def main() -> None:
    args = parse_args()
    selected_families = set(args.alternative_family)
    base_order, base_blocks = read_blocks(args.base)
    alternative_order, alternative_blocks = read_blocks(args.alternative)
    if set(base_blocks) != set(alternative_blocks):
        raise RuntimeError({
            "base_only": sorted(set(base_blocks) - set(alternative_blocks)),
            "alternative_only": sorted(set(alternative_blocks) - set(base_blocks)),
        })
    if set(base_order) != set(alternative_order):
        raise RuntimeError("Base and alternative dataset coverage differs")

    available_families = {family(dataset) for dataset in base_order}
    unknown = selected_families - available_families
    if unknown:
        raise RuntimeError(f"Unknown alternative families: {sorted(unknown)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    next_id = 0
    sources: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for dataset in base_order:
            use_alternative = family(dataset) in selected_families
            rows = alternative_blocks[dataset] if use_alternative else base_blocks[dataset]
            sources[dataset] = "alternative" if use_alternative else "base"
            row_counts[dataset] = len(rows)
            for source_row in rows:
                row = dict(source_row)
                row["id"] = str(next_id)
                writer.writerow(row)
                next_id += 1

    report = {
        "status": "valid",
        "base": str(args.base),
        "base_sha256": sha256_file(args.base),
        "alternative": str(args.alternative),
        "alternative_sha256": sha256_file(args.alternative),
        "alternative_families": sorted(selected_families),
        "available_families": sorted(available_families),
        "dataset_sources": sources,
        "row_counts": row_counts,
        "rows": next_id,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
