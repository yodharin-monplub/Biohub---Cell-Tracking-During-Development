#!/usr/bin/env python3
"""Frozen final-division veto on exact model107 CSVs; never uses GT."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model118"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model109.extract import development_edges
from model93.repair_endpoints import dataset_blocks
from scripts.validate_submission import COLUMNS, validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    cohort = args.cohort
    cfg = json.loads((OUT / "config.json").read_text())
    threshold = cfg["minimum_repair_branch_probability"]
    if threshold != 0.4 or cfg["source_model"] != "model107":
        raise RuntimeError("Frozen model118 configuration changed")
    base = ROOT / "model107/results" / cohort
    source = base / "candidate.csv"
    saved = json.loads((base / "official_score.json").read_text())
    if saved["status"] != "valid_and_scored" or saved["skipped"] or sha(source) != saved["submission_sha256"]:
        raise RuntimeError("Unverified model107 input")
    out = OUT / "results" / cohort
    out.mkdir(parents=True, exist_ok=False)
    candidate = out / "candidate.csv"
    report = []
    totals = Counter()
    next_id = 0
    with candidate.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for movie, rows in dataset_blocks(source):
            cap_dir = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
            cap = cap_dir / f"{movie}.npz"
            companion = json.loads((cap_dir / f"{movie}.json").read_text())
            if sha(cap) != companion.get("file_sha256", companion.get("capture_sha256")):
                raise RuntimeError(f"Capture hash mismatch: {movie}")
            succ: dict[int, list[int]] = defaultdict(list)
            for row in rows:
                if row["row_type"] == "edge":
                    succ[int(row["source_id"])].append(int(row["target_id"]))
            forks = {a for a, bs in succ.items() if len(bs) == 2}
            if any(len(bs) > 2 for bs in succ.values()):
                raise RuntimeError(f"Invalid >2 daughter source: {movie}")
            with np.load(cap) as data:
                selected = ({(int(a), int(b)) for a, b, _ in data["post_ilp_edges"]}
                            if cohort == "confirmation" else set(development_edges(movie)))
                source_ids, target_ids, ps = data["source"], data["target"], data["probability"]
                mask = np.isin(source_ids, np.fromiter(forks, dtype=np.int64))
                probability = {(int(a), int(b)): float(p)
                               for a, b, p in zip(source_ids[mask], target_ids[mask], ps[mask], strict=True)}
            remove = set()
            details = []
            stats = Counter(forks=len(forks))
            for a in sorted(forks):
                bs = succ[a]
                old = [(a, b) for b in bs if (a, b) in selected]
                if len(old) != 1:
                    stats["forks_without_exactly_one_post_ilp_edge"] += 1
                    continue
                added = next((a, b) for b in bs if (a, b) not in selected)
                p = probability.get(added)
                if p is None:
                    stats["repair_probability_missing"] += 1
                    continue
                if p < threshold:
                    remove.add(added)
                    details.append({"source_id": a, "target_id": added[1], "probability": p})
            for row in rows:
                if row["row_type"] == "edge" and (int(row["source_id"]), int(row["target_id"])) in remove:
                    continue
                writer.writerow({**row, "id": next_id})
                next_id += 1
            stats["removed_division_edges"] = len(remove)
            totals.update(stats)
            report.append({"movie": movie, "stats": dict(stats), "edits": details})
            print(f"{cohort} {len(report)}/39 {movie}: removed={len(remove)}", flush=True)
    if len(report) != 39:
        raise RuntimeError("Wrong cohort coverage")
    before, after = validate(source), validate(candidate)
    if before["datasets"].keys() != after["datasets"].keys():
        raise RuntimeError("Movie set changed")
    if before["totals"]["nodes"] != after["totals"]["nodes"]:
        raise RuntimeError("Node count changed")
    if before["totals"]["edges"] - after["totals"]["edges"] != totals["removed_division_edges"]:
        raise RuntimeError("Edge-removal count mismatch")
    dump(out / "veto_report.json", {"status": "valid", "cohort": cohort, "threshold": threshold,
         "source_sha256": sha(source), "candidate_sha256": sha(candidate),
         "config_sha256": sha(OUT / "config.json"), "source_code_sha256": sha(Path(__file__)),
         "totals": dict(totals), "movies": report,
         "caveat": "Label-free final-graph ablation; full organizer score follows."})


if __name__ == "__main__":
    main()
