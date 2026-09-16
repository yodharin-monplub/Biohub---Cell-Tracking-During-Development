"""Explicit GT-positive one-link division opportunities on model118 graphs."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model119"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from scripts.audit_detector_division_candidates import build_graph

SCALE = np.array((1.625, .40625, .40625), dtype=np.float64)
COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def main():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError("Existing model119 audit")
    previous = json.loads((ROOT / "model117/results.json").read_text())
    if previous["status"] != "complete" or previous["source_model"] != "model107":
        raise RuntimeError("Model117 event audit not verified")
    report = {}
    for cohort in ("development", "confirmation"):
        scored = json.loads((ROOT / "model118/results" / cohort / "official_score.json").read_text())
        control = json.loads((ROOT / "model107/results" / cohort / "official_score.json").read_text())
        path = ROOT / "model118/results" / cohort / "candidate.csv"
        if scored["status"] != "valid_and_scored" or scored["skipped"] or sha(path) != scored["submission_sha256"]:
            raise RuntimeError("Model118 scored CSV not verified")
        scored_rows = {row["dataset"]: row for row in scored["datasets"]}
        control_rows = {row["dataset"]: row for row in control["datasets"]}
        if any((scored_rows[m]["division_tp"], scored_rows[m]["division_fn"]) !=
               (control_rows[m]["division_tp"], control_rows[m]["division_fn"]) for m in scored_rows):
            raise RuntimeError("Model117 division-event labels no longer stable")
        groups = {str(g["dataset"][0]): g for g in pl.read_csv(path, columns=COLUMNS).partition_by("dataset")}
        events_by_movie = {row["movie"]: row["events"] for row in previous["cohorts"][cohort]}
        if len(groups) != 39 or set(groups) != set(events_by_movie) or set(groups) != set(scored_rows):
            raise RuntimeError("Wrong cohort coverage")
        rows = []
        for movie in sorted(groups):
            group = groups[movie]
            _, id_map = build_graph(group)
            inverse = {v: k for k, v in id_map.items()}
            nrows = group.filter(pl.col("row_type") == "node")
            nodes = {int(r["node_id"]): np.array([int(r[k]) for k in ("t", "z", "y", "x")], dtype=np.float64)
                     for r in nrows.iter_rows(named=True)}
            succ: dict[int, list[int]] = defaultdict(list)
            pred: dict[int, list[int]] = defaultdict(list)
            for r in group.filter(pl.col("row_type") == "edge").iter_rows(named=True):
                a, b = int(r["source_id"]), int(r["target_id"])
                succ[a].append(b)
                pred[b].append(a)
            positives = [(int(g), event) for g, event in events_by_movie[movie].items()
                         if event["category"] == "all_three_present_one_direct_link" and not event["official_recovered"]]
            if not positives:
                continue
            cap_dir = ROOT / ("model100" if cohort == "development" else "model102") / "capture"
            cap = cap_dir / f"{movie}.npz"
            companion = json.loads((cap_dir / f"{movie}.json").read_text())
            if sha(cap) != companion.get("file_sha256", companion.get("capture_sha256")):
                raise RuntimeError("Captured neural score hash mismatch")
            with np.load(cap) as data:
                wanted = {inverse[event["matched_parent"]] for _, event in positives}
                wanted.update(inverse[q] for _, event in positives for q in event["daughter_incoming"][0] + event["daughter_incoming"][1])
                source, target, values = data["source"], data["target"], data["probability"]
                mask = np.isin(source, np.fromiter(wanted, dtype=np.int64))
                probability = {(int(a), int(b)): float(p)
                               for a, b, p in zip(source[mask], target[mask], values[mask], strict=True)}
            for gt_parent, event in positives:
                p = inverse[event["matched_parent"]]
                daughters = [inverse[d] for d in event["matched_daughters"]]
                correct = [d for d in daughters if d in succ[p]]
                if len(correct) != 1:
                    raise RuntimeError("Expected exactly one matched daughter in final graph")
                a = correct[0]
                b = next(d for d in daughters if d != a)
                old = pred[b][0] if len(pred[b]) == 1 else None
                xyz = lambda n: nodes[n][1:] * SCALE
                distance = lambda u, v: float(np.linalg.norm(xyz(u) - xyz(v)))
                next_a = succ[a][0] if len(succ[a]) == 1 and nodes[succ[a][0]][0] == nodes[p][0] + 2 else None
                next_b = succ[b][0] if len(succ[b]) == 1 and nodes[succ[b][0]][0] == nodes[p][0] + 2 else None
                alt = probability.get((p, b))
                incumbent = probability.get((old, b)) if old is not None else None
                rows.append({"movie": movie, "gt_parent": gt_parent, "parent": p, "existing_daughter": a,
                             "missing_daughter": b, "old_parent": old, "second_daughter_occupied": old is not None,
                             "parent_out_degree": len(succ[p]), "parent_has_predecessor": bool(pred[p]),
                             "alternative_neural_probability": alt, "incumbent_neural_probability": incumbent,
                             "neural_margin": None if alt is None or incumbent is None else alt - incumbent,
                             "existing_child_neural_probability": probability.get((p, a)),
                             "parent_missing_daughter_distance_um": distance(p, b),
                             "sister_distance_um": distance(a, b),
                             "both_daughters_continue": next_a is not None and next_b is not None,
                             "separation_growth_um": None if next_a is None or next_b is None
                             else distance(next_a, next_b) - distance(a, b),
                             "old_parent_predecessor": bool(pred[old]) if old is not None else None,
                             "old_parent_out_degree": len(succ[old]) if old is not None else None})
            print(f"{cohort} {movie}: positive_one_link={len(positives)}", flush=True)
        counts = Counter()
        for row in rows:
            counts["positive_one_link"] += 1
            counts["occupied"] += int(row["second_daughter_occupied"])
            counts["alternative_in_top5"] += int(row["alternative_neural_probability"] is not None)
            counts["alt_ge_0p7"] += int(row["alternative_neural_probability"] is not None
                                      and row["alternative_neural_probability"] >= .7)
            counts["alt_ge_0p8"] += int(row["alternative_neural_probability"] is not None
                                      and row["alternative_neural_probability"] >= .8)
            counts["alt_ge_0p9"] += int(row["alternative_neural_probability"] is not None
                                      and row["alternative_neural_probability"] >= .9)
            counts["alt_ge_0p8_and_margin_ge_0p4"] += int(row["alternative_neural_probability"] is not None
                                                      and row["alternative_neural_probability"] >= .8
                                                      and (not row["second_daughter_occupied"]
                                                           or (row["neural_margin"] is not None and row["neural_margin"] >= .4)))
        report[cohort] = {"rows": rows, "counts": dict(counts), "model118_csv_sha256": sha(path)}
        print(f"SUMMARY {cohort} {dict(counts)}", flush=True)
    dump(output, {"status": "complete", "cohorts": report, "source_sha256": sha(Path(__file__)),
                  "model118_notebook_sha256": sha(ROOT / "model118/submission.ipynb"),
                  "caveat": "GT-positive diagnostic only; no inference rule or whole-graph score. Both cohorts reused."})


if __name__ == "__main__":
    main()
