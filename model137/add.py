"""Apply frozen train-only temporal gate to model120 proposals on model130."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model120.add import select
from model125.replay import load_graphs
from model136.temporal_pilot import GEOMETRY, IMAGE, image_features
from model93.repair_endpoints import dataset_blocks
from scripts.validate_submission import COLUMNS, validate

SCALE = np.asarray((1.625, .40625, .40625), dtype=np.float64)


def probability(row, trained):
    vector = np.asarray([row[key] for key in trained["feature_names"]], dtype=np.float64)
    normalized = (vector - np.asarray(trained["mean"])) / np.asarray(trained["scale"])
    return float(expit(trained["intercept"] + normalized @ np.asarray(trained["coefficients"])))


def case_for_proposal(proposal, nodes, successors, predecessors, get_frame):
    parent, orphan = int(proposal["source_id"]), int(proposal["target_id"])
    if len(successors[parent]) != 1 or len(predecessors[parent]) != 1:
        raise RuntimeError("Proposal parent topology drift")
    existing, previous = successors[parent][0], predecessors[parent][0]
    if len(successors[existing]) != 1 or len(successors[orphan]) != 1:
        raise RuntimeError("Proposal daughter continuation drift")
    next_existing, next_orphan = successors[existing][0], successors[orphan][0]
    t = nodes[parent][0]
    if (nodes[previous][0], nodes[existing][0], nodes[orphan][0],
        nodes[next_existing][0], nodes[next_orphan][0]) != (t - 1, t + 1, t + 1, t + 2, t + 2):
        raise RuntimeError("Proposal time sequence drift")
    xyz = lambda node: np.asarray(nodes[node][1:], dtype=np.float64) * SCALE
    distance = lambda a, b: float(np.linalg.norm(xyz(a) - xyz(b)))
    daughter_parent = distance(parent, existing)
    growth = distance(next_existing, next_orphan) - distance(existing, orphan)
    midpoint = float(np.linalg.norm((xyz(existing) + xyz(orphan)) / 2
                                    - (2 * xyz(parent) - xyz(previous))))
    row = {"t": t, "parent_orphan_um": float(proposal["parent_distance_um"]),
           "existing_daughter_parent_um": daughter_parent,
           "sister_um": float(proposal["sister_distance_um"]),
           "growth_um": growth, "midpoint_error_um": midpoint,
           "positions": {name: list(nodes[node]) for name, node in
                         (("parent", parent), ("previous_parent", previous),
                          ("existing_daughter", existing), ("orphan", orphan),
                          ("next_existing", next_existing), ("next_orphan", next_orphan))}}
    if abs(row["parent_orphan_um"] - distance(parent, orphan)) > 1e-8:
        raise RuntimeError("Proposal physical distance drift")
    if abs(row["sister_um"] - distance(existing, orphan)) > 1e-8:
        raise RuntimeError("Proposal sister distance drift")
    row.update(image_features(row, get_frame))
    if set(GEOMETRY + IMAGE) - set(row):
        raise RuntimeError("Missing frozen feature")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    cohort = args.cohort
    trained_path = ROOT / "model137/classifier.json"
    trained = json.loads(trained_path.read_text())
    pilot_path = ROOT / "model136/temporal_pilot.json"
    if (trained["status"] != "frozen" or trained["threshold"] != .95
            or trained["feature_names"] != list(GEOMETRY + IMAGE)
            or trained["source_model136_sha256"] != sha(pilot_path)
            or trained["source_model136_code_sha256"] != sha(ROOT / "model136/temporal_pilot.py")):
        raise RuntimeError("Frozen train-only image classifier changed")
    cfg_path = ROOT / "model120/frozen_config.json"
    cfg = json.loads(cfg_path.read_text())
    if (cfg["source_model"], cfg["minimum_link_probability"],
        cfg["frame_fraction_cap"], cfg["global_fraction_cap"]) != ("model118", .6, .0076, .00375):
        raise RuntimeError("Frozen proposal generator changed")
    source = ROOT / "model130/results" / cohort / "candidate.csv"
    control = json.loads((ROOT / "model130/results" / cohort / "official_score.json").read_text())
    old = ROOT / "model118/results" / cohort / "candidate.csv"
    old_score = json.loads((ROOT / "model118/results" / cohort / "official_score.json").read_text())
    names = {row["dataset"] for row in control["datasets"]}
    if (control["status"] != "valid_and_scored" or control["skipped"]
            or old_score["status"] != "valid_and_scored" or old_score["skipped"]
            or len(names) != 39 or sha(source) != control["submission_sha256"]
            or sha(old) != old_score["submission_sha256"]):
        raise RuntimeError("Scored control source changed")
    old_graphs = load_graphs(old, names)
    output = ROOT / "model137/results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate.csv"
    reports = []
    totals = Counter()
    next_id = 0
    with candidate.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for movie, rows in dataset_blocks(source):
            nodes = {int(row["node_id"]): tuple(int(row[key]) for key in ("t", "z", "y", "x"))
                     for row in rows if row["row_type"] == "node"}
            successors, predecessors = defaultdict(list), defaultdict(list)
            current_edges = set()
            for row in rows:
                if row["row_type"] == "edge":
                    a, b = int(row["source_id"]), int(row["target_id"])
                    successors[a].append(b)
                    predecessors[b].append(a)
                    current_edges.add((a, b))
            forbidden = set(old_graphs[movie]["edges"]) - current_edges
            proposals, proposal_stats = select(movie, rows, cohort, cfg)
            array = zarr.open(ROOT / "data/raw/train" / f"{movie}.zarr", mode="r")["0"]
            if tuple(array.shape) != (100, 64, 256, 256):
                raise RuntimeError("Unexpected raw image shape")
            @lru_cache(maxsize=8)
            def frame(t):
                return np.asarray(array[t])
            local = Counter()
            accepted = []
            evaluated = []
            # Frame order improves Zarr cache locality; original candidate order
            # and caps were already fixed by the frozen proposal generator.
            for proposal in sorted(proposals, key=lambda r: (nodes[int(r["source_id"])][0],
                                                            r["source_id"], r["target_id"])):
                parent, orphan = int(proposal["source_id"]), int(proposal["target_id"])
                if (parent, orphan) in forbidden:
                    local["prior_veto"] += 1
                    continue
                daughter = successors[parent][0]
                existing_distance = float(np.linalg.norm((np.asarray(nodes[parent][1:])
                                                         - np.asarray(nodes[daughter][1:])) * SCALE))
                if proposal["parent_distance_um"] < 3.0 or existing_distance > 5.2:
                    local["outside_train_support"] += 1
                    continue
                case = case_for_proposal(proposal, nodes, successors, predecessors, frame)
                value = probability(case, trained)
                local["image_scored"] += 1
                evaluated.append({"parent": parent, "orphan": orphan,
                                  "neural_probability": proposal["probability"],
                                  "image_score": value})
                if value >= trained["threshold"]:
                    accepted.append((parent, orphan))
                    local["accepted"] += 1
                else:
                    local["image_rejected"] += 1
            for row in rows:
                writer.writerow({**row, "id": next_id})
                next_id += 1
            for parent, orphan in accepted:
                writer.writerow(dict(zip(COLUMNS, (next_id, movie, "edge", -1, -1, -1,
                                                 -1, -1, parent, orphan), strict=True)))
                next_id += 1
            if local["accepted"] + local["image_rejected"] != local["image_scored"]:
                raise RuntimeError("Image gate count mismatch")
            if sum(local.values()) - local["image_scored"] != len(proposals):
                # Count categories excluding the 'image_scored' subtotal.
                raise RuntimeError("Proposal disposition count mismatch")
            reports.append({"movie": movie, "proposal_stats": proposal_stats,
                            "gate_counts": dict(local), "accepted_edges": accepted,
                            "evaluated": evaluated})
            totals.update(local)
            print(f"MODEL137 {cohort} {len(reports)}/39 {movie} proposals={len(proposals)}"
                  f" image_scored={local['image_scored']} accepted={local['accepted']}", flush=True)
    if len(reports) != 39 or {row["movie"] for row in reports} != names:
        raise RuntimeError("Wrong movie coverage")
    before, after = validate(source), validate(candidate)
    if set(before["datasets"]) != set(after["datasets"]) or before["totals"]["nodes"] != after["totals"]["nodes"]:
        raise RuntimeError("Control movie/node coverage changed")
    if after["totals"]["edges"] - before["totals"]["edges"] != totals["accepted"]:
        raise RuntimeError("Accepted edge count mismatch")
    dump(output / "selection_report.json", {"status": "valid", "cohort": cohort,
         "source_model130_sha256": sha(source), "source_model118_sha256": sha(old),
         "candidate_sha256": sha(candidate), "classifier_sha256": sha(trained_path),
         "config_sha256": sha(cfg_path), "source_code_sha256": sha(Path(__file__)),
         "totals": dict(totals), "movies": reports,
         "caveat": "Label-free real proposal replay; exact organizer scorer required."})


if __name__ == "__main__":
    main()
