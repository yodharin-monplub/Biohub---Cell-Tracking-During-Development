"""One frozen label-free post-model118 orphan daughter addition."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model120"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model93.repair_endpoints import dataset_blocks
from scripts.validate_submission import COLUMNS, validate

SCALE = np.array((1.625, .40625, .40625), dtype=np.float64)


def select(movie, rows, cohort, cfg):
    nodes = {int(r["node_id"]): np.array([int(r[k]) for k in ("t", "z", "y", "x")], dtype=np.float64)
             for r in rows if r["row_type"] == "node"}
    succ, pred = defaultdict(list), defaultdict(list)
    edges = []
    for r in rows:
        if r["row_type"] == "edge":
            a, b = int(r["source_id"]), int(r["target_id"])
            edges.append((a, b))
            succ[a].append(b)
            pred[b].append(a)
    eligible = {}
    by_frame = Counter()
    for p in nodes:
        if len(succ[p]) != 1 or len(pred[p]) != 1:
            continue
        a, pp = succ[p][0], pred[p][0]
        if nodes[a][0] != nodes[p][0] + 1 or nodes[pp][0] != nodes[p][0] - 1:
            continue
        if len(succ[a]) != 1 or nodes[succ[a][0]][0] != nodes[p][0] + 2:
            continue
        eligible[p] = a
        by_frame[int(nodes[p][0])] += 1
    cap_dir = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
    cap = cap_dir / f"{movie}.npz"
    companion = json.loads((cap_dir / f"{movie}.json").read_text())
    if sha(cap) != companion.get("file_sha256", companion.get("capture_sha256")):
        raise RuntimeError("Neural capture hash mismatch")
    with np.load(cap) as data:
        source, target, values = data["source"], data["target"], data["probability"]
        mask = np.isin(source, np.fromiter(eligible, dtype=np.int64))
        probability = {(int(a), int(b)): float(p)
                       for a, b, p in zip(source[mask], target[mask], values[mask], strict=True)}
    xyz = {n: row[1:] * SCALE for n, row in nodes.items()}
    distance = lambda a, b: float(np.linalg.norm(xyz[a] - xyz[b]))
    proposals = []
    for (p, b), prob in probability.items():
        if prob < cfg["minimum_link_probability"] or b not in nodes or pred[b]:
            continue
        a = eligible[p]
        if b == a or nodes[b][0] != nodes[p][0] + 1:
            continue
        if len(succ[b]) != 1 or nodes[succ[b][0]][0] != nodes[p][0] + 2:
            continue
        parent_dist, sister_dist = distance(p, b), distance(a, b)
        if parent_dist > cfg["parent_orphan_max_um"] or sister_dist > cfg["sister_max_um"]:
            continue
        proposals.append((prob, p, b, parent_dist, sister_dist))
    proposals.sort(key=lambda row: (-row[0], row[1], row[2]))
    max_global = max(1, int(round(max(1, len(edges)) * cfg["global_fraction_cap"])))
    selected = []
    used_parent, used_orphan = set(), set()
    selected_by_frame = Counter()
    for prob, p, b, parent_dist, sister_dist in proposals:
        if len(selected) >= max_global:
            break
        frame = int(nodes[p][0])
        frame_cap = max(1, int(round(by_frame[frame] * cfg["frame_fraction_cap"])))
        if p in used_parent or b in used_orphan or selected_by_frame[frame] >= frame_cap:
            continue
        selected.append({"source_id": p, "target_id": b, "probability": prob,
                         "parent_distance_um": parent_dist, "sister_distance_um": sister_dist})
        used_parent.add(p)
        used_orphan.add(b)
        selected_by_frame[frame] += 1
    return selected, {"eligible_parents": len(eligible), "proposals": len(proposals),
                      "selected": len(selected), "global_cap": max_global,
                      "frames_selected": len(selected_by_frame)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    cohort = args.cohort
    cfg = json.loads((OUT / "frozen_config.json").read_text())
    if (cfg["source_model"], cfg["minimum_link_probability"], cfg["frame_fraction_cap"],
        cfg["global_fraction_cap"]) != ("model118", .6, .0076, .00375):
        raise RuntimeError("Frozen model120 configuration changed")
    source = ROOT / "model118/results" / cohort / "candidate.csv"
    saved = json.loads((ROOT / "model118/results" / cohort / "official_score.json").read_text())
    if saved["status"] != "valid_and_scored" or saved["skipped"] or sha(source) != saved["submission_sha256"]:
        raise RuntimeError("Model118 source not verified")
    out = OUT / "results" / cohort
    out.mkdir(parents=True, exist_ok=False)
    candidate = out / "candidate.csv"
    report, totals = [], Counter()
    next_id = 0
    with candidate.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for movie, rows in dataset_blocks(source):
            additions, stats = select(movie, rows, cohort, cfg)
            for row in rows:
                writer.writerow({**row, "id": next_id})
                next_id += 1
            for edge in additions:
                writer.writerow(dict(zip(COLUMNS, [next_id, movie, "edge", -1, -1, -1, -1, -1,
                                                    edge["source_id"], edge["target_id"]])))
                next_id += 1
            report.append({"movie": movie, "stats": stats, "additions": additions})
            totals.update(stats)
            print(f"{cohort} {len(report)}/39 {movie} added={len(additions)}", flush=True)
    if len(report) != 39:
        raise RuntimeError("Wrong movie coverage")
    before, after = validate(source), validate(candidate)
    if before["datasets"].keys() != after["datasets"].keys() or before["totals"]["nodes"] != after["totals"]["nodes"]:
        raise RuntimeError("Movie set or nodes changed")
    if after["totals"]["edges"] - before["totals"]["edges"] != totals["selected"]:
        raise RuntimeError("Added-edge count mismatch")
    dump(out / "selection_report.json", {"status": "valid", "cohort": cohort,
         "source_sha256": sha(source), "candidate_sha256": sha(candidate),
         "config_sha256": sha(OUT / "frozen_config.json"), "source_code_sha256": sha(Path(__file__)),
         "totals": dict(totals), "movies": report,
         "caveat": "Label-free offline full-graph candidate; scorer required."})


if __name__ == "__main__":
    main()
