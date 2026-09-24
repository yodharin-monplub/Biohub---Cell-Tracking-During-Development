"""Audit whether alleged orphan daughters have an independent raw antecedent.

This is diagnostic on already-reused validation cohorts, not threshold fitting.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha
from model125.replay import load_graphs

SCALE = np.asarray((1.625, .40625, .40625), dtype=np.float64)
FIELDS = ("orphan_motion_um", "candidate_prior_to_orphan_um",
          "candidate_prior_motion_error_um", "candidate_prior_to_parent_um",
          "candidate_earlier_motion_error_um", "candidate_prior_nearest_rank")


def summarize(rows):
    result = {"n": len(rows)}
    for field in FIELDS:
        vals = [float(row[field]) for row in rows if row[field] is not None]
        result[field] = {"n": len(vals), "min": min(vals) if vals else None,
                         "median": statistics.median(vals) if vals else None,
                         "max": max(vals) if vals else None}
    return result


def features(row, graph, raw):
    nodes = graph["nodes"]
    edges = graph["edges"]
    succ = defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
    parent = int(row["parent"])
    orphan = int(row["orphan"])
    t = int(row["t"])
    if len(succ[orphan]) != 1:
        raise RuntimeError("Orphan continuation changed")
    next_orphan = succ[orphan][0]
    if nodes[parent][0] != t or nodes[orphan][0] != t + 1 or nodes[next_orphan][0] != t + 2:
        raise RuntimeError("Proposal frame order changed")
    p = np.asarray(nodes[parent][1:], dtype=np.float64) * SCALE
    o = np.asarray(nodes[orphan][1:], dtype=np.float64) * SCALE
    o2 = np.asarray(nodes[next_orphan][1:], dtype=np.float64) * SCALE
    motion = float(np.linalg.norm(o2 - o))
    at_t = raw[raw[:, 0] == t]
    if not len(at_t):
        raise RuntimeError("No raw nodes in parent frame")
    physical = at_t[:, 1:].astype(np.float64) * SCALE
    from_parent = np.linalg.norm(physical - p, axis=1)
    # Exclude all alternative detector peaks within the parent's 3um core.
    candidates = physical[from_parent > 3.0]
    if not len(candidates):
        raise RuntimeError("No independent raw candidate at parent frame")
    backward = 2 * o - o2
    errors = np.linalg.norm(candidates - backward, axis=1)
    idx = int(np.argmin(errors))
    prior = candidates[idx]
    orphan_distances = np.linalg.norm(candidates - o, axis=1)
    rank = 1 + int(np.count_nonzero(orphan_distances < orphan_distances[idx]))
    earlier_error = None
    if t >= 1:
        at_earlier = raw[raw[:, 0] == t - 1]
        if len(at_earlier):
            physical_earlier = at_earlier[:, 1:].astype(np.float64) * SCALE
            # Independent linear trajectory inferred from raw t and orphan t+1.
            projected = 2 * prior - o
            earlier_error = float(np.min(np.linalg.norm(physical_earlier - projected, axis=1)))
    return {"orphan_motion_um": motion,
            "candidate_prior_to_orphan_um": float(np.linalg.norm(prior - o)),
            "candidate_prior_motion_error_um": float(errors[idx]),
            "candidate_prior_to_parent_um": float(np.linalg.norm(prior - p)),
            "candidate_earlier_motion_error_um": earlier_error,
            "candidate_prior_nearest_rank": rank}


def main():
    output = ROOT / "model136/antecedent_audit.json"
    if output.exists():
        raise FileExistsError("Existing antecedent audit")
    source = json.loads((ROOT / "model134/audit.json").read_text())
    if source["status"] != "complete":
        raise RuntimeError("Exact-label source incomplete")
    records = []
    for cohort in ("development", "confirmation"):
        source_rows = [row for row in source["cohorts"][cohort]["labeled_additions"]
                       if row["label"] in ("tp", "fp")]
        names = {row["movie"] for row in source_rows}
        graph_csv = ROOT / "model133/results" / cohort / "candidate.csv"
        if sha(graph_csv) != source["cohorts"][cohort]["source_model133_csv_sha256"]:
            raise RuntimeError("Scored graph source changed")
        graphs = load_graphs(graph_csv, names)
        by_movie = defaultdict(list)
        for row in source_rows:
            by_movie[row["movie"]].append(row)
        for movie in sorted(by_movie):
            capture_folder = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
            capture = capture_folder / f"{movie}.npz"
            receipt = json.loads((capture_folder / f"{movie}.json").read_text())
            if sha(capture) != receipt.get("file_sha256", receipt.get("capture_sha256")):
                raise RuntimeError("Frozen raw detector capture changed")
            with np.load(capture) as data:
                raw = data["coords"].copy()
            for row in by_movie[movie]:
                record = {"cohort": cohort, "family": row["family"], "movie": movie,
                          "label": row["label"], "parent": row["parent"],
                          "orphan": row["orphan"], "t": row["t"],
                          "probability": row["probability"],
                          **features(row, graphs[movie], raw)}
                records.append(record)
            print(f"ANTECEDENT {cohort} {movie} labeled={len(by_movie[movie])}", flush=True)
    expected = {"development": {"tp": 1, "fp": 15},
                "confirmation": {"tp": 5, "fp": 10}}
    for cohort, labels in expected.items():
        for label, count in labels.items():
            if sum(r["cohort"] == cohort and r["label"] == label for r in records) != count:
                raise RuntimeError("Exact labeled coverage changed")
    result = {"status": "complete", "samples": records,
              "summary": {cohort: {label: summarize([r for r in records
                                                    if r["cohort"] == cohort and r["label"] == label])
                                   for label in ("tp", "fp")}
                          for cohort in ("development", "confirmation")},
              "caveat": "31 reused validation labels; no threshold fitting or promotion."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "counts": expected}, indent=2))


if __name__ == "__main__":
    main()
