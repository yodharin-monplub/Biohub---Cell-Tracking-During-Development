"""Stage coverage of 13 previously mapped GT-positive orphan misses."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha
from model125.replay import load_graphs
from model137.add import SCALE, case_for_proposal, probability


def main():
    output = ROOT / "model139/audit.json"
    if output.exists():
        raise FileExistsError("Existing known-miss audit")
    source_path = ROOT / "model119/audit.json"
    source = json.loads(source_path.read_text())
    trained = json.loads((ROOT / "model137/classifier.json").read_text())
    if source["status"] != "complete" or trained["status"] != "frozen":
        raise RuntimeError("Known-miss source or classifier incomplete")
    reports = []
    for cohort in ("development", "confirmation"):
        positives = [r for r in source["cohorts"][cohort]["rows"]
                     if not r["second_daughter_occupied"]]
        if len(positives) != (6 if cohort == "development" else 7):
            raise RuntimeError("Known-miss count changed")
        graph_path = ROOT / "model130/results" / cohort / "candidate.csv"
        scored = json.loads((ROOT / "model130/results" / cohort / "official_score.json").read_text())
        if sha(graph_path) != scored["submission_sha256"]:
            raise RuntimeError("Scored model130 graph changed")
        names = {r["movie"] for r in positives}
        graphs = load_graphs(graph_path, names)
        selected = json.loads((ROOT / "model137/results" / cohort / "selection_report.json").read_text())
        selected_by_movie = {r["movie"]: r for r in selected["movies"]}
        for row in positives:
            movie = row["movie"]
            nodes = graphs[movie]["nodes"]
            successors, predecessors = defaultdict(list), defaultdict(list)
            for a, b in graphs[movie]["edges"]:
                successors[a].append(b)
                predecessors[b].append(a)
            parent, existing, orphan = (int(row[k]) for k in
                                        ("parent", "existing_daughter", "missing_daughter"))
            result = {"cohort": cohort, "movie": movie, "gt_parent": row["gt_parent"],
                      "parent": parent, "existing_daughter": existing, "orphan": orphan,
                      "neural_probability": row["alternative_neural_probability"],
                      "stage": "unknown", "image_score": None}
            if not all(node in nodes for node in (parent, existing, orphan)):
                result["stage"] = "node_missing"
            elif successors[parent] != [existing] or predecessors[orphan]:
                result["stage"] = "parent_or_orphan_topology"
            elif len(predecessors[parent]) != 1 or len(successors[existing]) != 1 or len(successors[orphan]) != 1:
                result["stage"] = "temporal_continuation"
            elif row["alternative_neural_probability"] is None:
                result["stage"] = "not_in_captured_top5"
            else:
                xyz = lambda node: np.asarray(nodes[node][1:], dtype=np.float64) * SCALE
                dist = lambda a, b: float(np.linalg.norm(xyz(a) - xyz(b)))
                proposal = {"source_id": parent, "target_id": orphan,
                            "parent_distance_um": dist(parent, orphan),
                            "sister_distance_um": dist(existing, orphan)}
                result.update(parent_distance_um=proposal["parent_distance_um"],
                              existing_distance_um=dist(parent, existing),
                              sister_distance_um=proposal["sister_distance_um"])
                if row["alternative_neural_probability"] < .6:
                    result["stage"] = "below_neural_floor_0p6"
                elif proposal["parent_distance_um"] > 15 or proposal["sister_distance_um"] > 20.5:
                    result["stage"] = "outside_broad_geometry"
                elif proposal["parent_distance_um"] < 3.0 or dist(parent, existing) > 5.2:
                    result["stage"] = "outside_train_support"
                else:
                    actual = next((r for r in selected_by_movie[movie]["evaluated"]
                                   if r["parent"] == parent and r["orphan"] == orphan), None)
                    if actual is None:
                        result["stage"] = "not_selected_after_caps_or_veto"
                    else:
                        array = zarr.open(ROOT / "data/raw/train" / f"{movie}.zarr", mode="r")["0"]
                        @lru_cache(maxsize=8)
                        def frame(t):
                            return np.asarray(array[t])
                        case = case_for_proposal(proposal, nodes, successors, predecessors, frame)
                        value = probability(case, trained)
                        if abs(value - actual["image_score"]) > 1e-8:
                            raise RuntimeError("Frozen real-proposal image score parity failed")
                        result["image_score"] = value
                        result["stage"] = "passes_image_0p8" if value >= .8 else "below_image_0p8"
            reports.append(result)
            print(f"MISS {cohort} {movie} {parent}->{orphan}: {result['stage']}", flush=True)
    totals = {cohort: dict(Counter(r["stage"] for r in reports if r["cohort"] == cohort))
              for cohort in ("development", "confirmation")}
    result = {"status": "complete", "source_model119_sha256": sha(source_path),
              "model137_classifier_sha256": sha(ROOT / "model137/classifier.json"),
              "totals": totals, "events": reports,
              "caveat": "Only 13 reused GT-positive known misses; stage diagnosis, no threshold selection."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
