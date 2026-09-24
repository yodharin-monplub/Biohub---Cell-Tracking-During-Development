"""Cap-aware train-only pilot for the frozen model142 close-orphan rule."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model142.select import qualifies

SCALE = np.asarray((1.625, .40625, .40625), dtype=np.float64)


def stage(movie, rows, cfg):
    capture = ROOT / "model140/capture" / f"{movie}.npz"
    with np.load(capture) as data:
        raw = data["coords"].copy()
        node_rows = data["post_ilp_nodes"].copy()
        edge_rows = data["post_ilp_edges"].copy()
    nodes = {int(row[0]): row[1:] for row in node_rows}
    successors, predecessors = defaultdict(list), defaultdict(list)
    for a, b, _ in edge_rows:
        successors[int(a)].append(int(b))
        predecessors[int(b)].append(int(a))
    by_frame = Counter()
    for parent, attrs in nodes.items():
        if len(successors[parent]) != 1 or len(predecessors[parent]) != 1:
            continue
        daughter, previous = successors[parent][0], predecessors[parent][0]
        if (int(nodes[daughter][0]) != int(attrs[0]) + 1
                or int(nodes[previous][0]) != int(attrs[0]) - 1
                or len(successors[daughter]) != 1
                or int(nodes[successors[daughter][0]][0]) != int(attrs[0]) + 2):
            continue
        by_frame[int(attrs[0])] += 1
    def is_raw(node):
        if node < 0 or node >= len(raw) or int(nodes[node][0]) != int(raw[node][0]):
            return False
        return float(np.linalg.norm((nodes[node][1:] - raw[node][1:]) * SCALE)) <= cfg["raw_node_match_max_um"]
    eligible = [row for row in rows if qualifies(row, cfg)]
    verified = [row for row in eligible if is_raw(int(row["parent"])) and is_raw(int(row["orphan"]))]
    verified.sort(key=lambda row: (-row["probability"], row["parent"], row["orphan"]))
    global_cap = max(1, int(round(max(1, len(edge_rows)) * cfg["global_fraction_cap"])))
    selected = []
    used_parent, used_orphan = set(), set()
    used_by_frame = Counter()
    for row in verified:
        if len(selected) >= global_cap:
            break
        parent, orphan = int(row["parent"]), int(row["orphan"])
        frame = int(nodes[parent][0])
        frame_cap = max(1, int(round(by_frame[frame] * cfg["frame_fraction_cap"])))
        if parent in used_parent or orphan in used_orphan or used_by_frame[frame] >= frame_cap:
            continue
        selected.append(row)
        used_parent.add(parent)
        used_orphan.add(orphan)
        used_by_frame[frame] += 1
    return eligible, verified, selected, sum(by_frame.values()), global_cap


def main():
    output = ROOT / "model142/pilot.json"
    if output.exists():
        raise FileExistsError("Existing cap-aware train-only pilot")
    cfg_path = ROOT / "model142/config.json"
    cfg = json.loads(cfg_path.read_text())
    audit_path = ROOT / "model140/audit.json"
    audit = json.loads(audit_path.read_text())
    if (audit["status"] != "complete" or cfg["source_model"] != "model130"
            or cfg["train_only_audit_sha256"] != sha(audit_path)
            or (cfg["minimum_link_probability"], cfg["rule_parent_orphan_max_um"],
                cfg["rule_sister_max_um"], cfg["sister_separation_growth_min_um"],
                cfg["frame_fraction_cap"], cfg["global_fraction_cap"]) != (
                    .3, 10.0, 14.0, 1.4, .0076, .00375)):
        raise RuntimeError("Frozen train-only model142 protocol changed")
    by_movie = defaultdict(list)
    for row in audit["proposals"]:
        by_movie[row["movie"]].append(row)
    if set(by_movie) != {r["movie"] for r in audit["movies"]}:
        raise RuntimeError("Wrong train-only movie coverage")
    reports = []
    all_rule = []
    all_selected = []
    for movie in sorted(by_movie):
        rule, verified, selected, eligible_count, cap = stage(movie, by_movie[movie], cfg)
        expected = next(r["proposal_stage"]["eligible_parents"]
                        for r in audit["movies"] if r["movie"] == movie)
        if eligible_count != expected:
            raise RuntimeError(f"Eligible parent/frame cap parity failed: {movie}")
        all_rule.extend(rule)
        all_selected.extend(selected)
        reports.append({"movie": movie, "family": movie.split("_")[0],
                        "broad": len(by_movie[movie]), "eligible_parents": eligible_count,
                        "rule": len(rule), "raw_verified": len(verified),
                        "selected": len(selected), "global_cap": cap,
                        "rule_labels": dict(Counter(r["gt_label"] for r in rule)),
                        "selected_labels": dict(Counter(r["gt_label"] for r in selected))})
        print(f"PILOT {movie}: rule={len(rule)} raw={len(verified)} selected={len(selected)}"
              f" labels={reports[-1]['selected_labels']}", flush=True)
    counts = Counter(r["gt_label"] for r in all_rule)
    if counts != Counter({"unknown": 1255, "explicit_division_positive": 10,
                          "positive_link_division_unverified": 5}):
        raise RuntimeError("Predeclared train-only rule counts changed")
    result = {"status": "train_only_stage_complete", "config_sha256": sha(cfg_path),
              "model140_audit_sha256": sha(audit_path), "movies": reports,
              "pre_cap_labels": dict(counts),
              "post_cap_labels": dict(Counter(r["gt_label"] for r in all_selected)),
              "post_cap_by_family": {family: dict(Counter(r["gt_label"] for r in all_selected
                                                         if r["movie"].startswith(family + "_")))
                                     for family in ("44b6", "6bba")},
              "selected_total": len(all_selected),
              "caveat": "Train-only post-ILP cap pilot; final model130 repair topology and sparse-GT unknowns differ."}
    dump(output, result)
    print(json.dumps({"status": result["status"], "selected_total": result["selected_total"],
                      "pre_cap_labels": result["pre_cap_labels"],
                      "post_cap_labels": result["post_cap_labels"],
                      "post_cap_by_family": result["post_cap_by_family"]}, indent=2))


if __name__ == "__main__":
    main()
