"""Build explicit-GT temporal fork cases from movies outside both scored cohorts.

This is a feasibility dataset; it does not claim model133 candidate parity because
frozen neural captures do not yet exist for these 121 movies.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha
from scripts.audit_detector_division_candidates import load_graph

SCALE = np.asarray((1.625, .40625, .40625), dtype=np.float64)


def build_cases(movie, graph):
    nodes = {int(r["node_id"]): tuple(int(r[k]) for k in ("t", "z", "y", "x"))
             for r in graph.node_attrs().iter_rows(named=True)}
    successors, predecessors = defaultdict(list), defaultdict(list)
    for r in graph.edge_attrs(attr_keys=["source_id", "target_id"]).iter_rows(named=True):
        a, b = int(r["source_id"]), int(r["target_id"])
        successors[a].append(b)
        predecessors[b].append(a)
    by_frame = defaultdict(list)
    for node, attrs in nodes.items():
        by_frame[attrs[0]].append(node)
    xyz = {node: np.asarray(attrs[1:], dtype=np.float64) * SCALE for node, attrs in nodes.items()}
    def dist(a, b):
        return float(np.linalg.norm(xyz[a] - xyz[b]))
    results = []
    all_divisions = 0
    for parent, children in list(successors.items()):
        t = nodes[parent][0]
        if len(children) not in (1, 2):
            raise RuntimeError(f"GT outdegree unexpected: {movie}/{parent}")
        if len(children) == 2:
            all_divisions += 1
        if t < 1 or t > 97 or len(predecessors[parent]) != 1:
            continue
        previous = predecessors[parent][0]
        if nodes[previous][0] != t - 1 or any(nodes[d][0] != t + 1 for d in children):
            continue
        if any(len(successors[d]) != 1 or nodes[successors[d][0]][0] != t + 2
               for d in children):
            continue
        if len(children) == 2:
            ordered = sorted(children, key=lambda d: (dist(parent, d), d))
            pairs = [(ordered[0], ordered[1], "positive")]
        else:
            existing = children[0]
            if dist(parent, existing) > 5.2:
                continue
            pairs = []
            for orphan in by_frame[t + 1]:
                if orphan == existing or len(predecessors[orphan]) != 1:
                    continue
                # The explicit incoming GT edge establishes a different lineage.
                other_parent = predecessors[orphan][0]
                if other_parent == parent or nodes[other_parent][0] != t:
                    continue
                if len(successors[orphan]) != 1 or nodes[successors[orphan][0]][0] != t + 2:
                    continue
                if dist(parent, orphan) > 15.0:
                    continue
                pairs.append((existing, orphan, "negative"))
        for existing, orphan, label in pairs:
            parent_orphan = dist(parent, orphan)
            daughter_parent = dist(parent, existing)
            sister = dist(existing, orphan)
            next_existing, next_orphan = successors[existing][0], successors[orphan][0]
            growth = dist(next_existing, next_orphan) - sister
            midpoint_error = float(np.linalg.norm((xyz[existing] + xyz[orphan]) / 2
                                                  - (2 * xyz[parent] - xyz[previous])))
            geometry = (6.5 <= parent_orphan <= 15.0 and daughter_parent <= 5.2
                        and sister <= 20.5 and midpoint_error >= 3.0 and growth >= .5
                        and (sister >= 9.5 or growth >= 2.5))
            case = {"movie": movie, "family": movie.split("_")[0], "label": label,
                    "t": t, "parent": parent, "previous_parent": previous,
                    "existing_daughter": existing, "orphan": orphan,
                    "next_existing": next_existing, "next_orphan": next_orphan,
                    "prior_orphan": predecessors[orphan][0] if label == "negative" else None,
                    "parent_orphan_um": parent_orphan,
                    "existing_daughter_parent_um": daughter_parent,
                    "sister_um": sister, "growth_um": growth,
                    "midpoint_error_um": midpoint_error,
                    "model133_geometry": geometry,
                    "positions": {name: list(nodes[node]) for name, node in
                                  (("parent", parent), ("previous_parent", previous),
                                   ("existing_daughter", existing), ("orphan", orphan),
                                   ("next_existing", next_existing), ("next_orphan", next_orphan))}}
            results.append(case)
    return results, all_divisions


def main():
    output = ROOT / "model136/train_only_candidates.json"
    if output.exists():
        raise FileExistsError("Existing train-only candidate audit")
    audit = json.loads((ROOT / "model131/data_audit.json").read_text())
    if audit["status"] != "complete" or audit["totals"]["train_only_movies"] != 121:
        raise RuntimeError("Frozen train-only coverage audit missing")
    names = sorted(r["movie"] for r in audit["movies"] if r["scope"] == "train_only")
    if len(names) != 121 or len(set(names)) != 121:
        raise RuntimeError("Train-only movie coverage changed")
    all_cases = []
    totals = Counter()
    per_movie = []
    for index, movie in enumerate(names, 1):
        graph = load_graph(ROOT / "data/raw/train" / f"{movie}.geff")
        cases, divisions = build_cases(movie, graph)
        expected = next(r["annotated_divisions"] for r in audit["movies"] if r["movie"] == movie)
        if divisions != expected:
            raise RuntimeError(f"GT division count drift: {movie}")
        counts = Counter((r["label"], r["model133_geometry"]) for r in cases)
        per_movie.append({"movie": movie, "family": movie.split("_")[0],
                          "annotated_divisions": divisions,
                          "continuous_positive": sum(r["label"] == "positive" for r in cases),
                          "geometry_positive": counts[("positive", True)],
                          "continuous_negative": sum(r["label"] == "negative" for r in cases),
                          "geometry_negative": counts[("negative", True)]})
        totals.update({f"{label}_{'geometry' if geometry else 'other'}": n
                       for (label, geometry), n in counts.items()})
        all_cases.extend(cases)
        print(f"TRAIN-ONLY {index}/121 {movie}: positive={per_movie[-1]['continuous_positive']}"
              f"/{per_movie[-1]['geometry_positive']} negative={per_movie[-1]['continuous_negative']}"
              f"/{per_movie[-1]['geometry_negative']}", flush=True)
    if sum(row["annotated_divisions"] for row in per_movie) != 85:
        raise RuntimeError("Train-only positive base count changed")
    result = {"status": "complete", "source_model131_sha256": sha(ROOT / "model131/data_audit.json"),
              "movies": per_movie, "totals": dict(totals), "cases": all_cases,
              "caveat": "Explicit GT geometry audit only; lacks model1 detector and neural candidate distribution."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "totals": result["totals"]}, indent=2))


if __name__ == "__main__":
    main()
