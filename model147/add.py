"""Restore only model130 midpoint-veto edges passing two-frame evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model125.replay import load_graphs
from model93.repair_endpoints import dataset_blocks
from model145.add import parts
from model145.select import select
from scripts.validate_submission import COLUMNS, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    cohort = parser.parse_args().cohort
    cfg_path = ROOT / "model147/config.json"
    cfg = json.loads(cfg_path.read_text())
    old_cfg = json.loads((ROOT / "model145/config.json").read_text())
    for key in ("minimum_link_probability", "parent_orphan_max_um", "sister_max_um",
                "rule_parent_orphan_max_um", "rule_sister_max_um",
                "sister_separation_growth_min_um", "raw_node_match_max_um",
                "frame_fraction_cap", "global_fraction_cap", "recovery_family"):
        if cfg[key] != old_cfg[key]:
            raise RuntimeError(f"Model145 selector changed: {key}")
    if (cfg["source_model"] != "model143" or cfg["control_model"] != "model130"
            or not cfg["preserve_model118_neural_veto"]
            or not cfg["preserve_model129_tight_sister_veto"]
            or not cfg["allow_model130_midpoint_veto_override"]):
        raise RuntimeError("Frozen veto-exception scope changed")
    stage_path = ROOT / "model147/stage_audit.json"
    stage = json.loads(stage_path.read_text())
    if stage["status"] != "diagnostic_only" or stage["config_sha256"] != sha(cfg_path):
        raise RuntimeError("Veto-stage provenance audit missing")
    paths = {name: ROOT / name / "results" / cohort / "candidate.csv"
             for name in ("model143", "model130", "model129", "model118")}
    scores = {name: json.loads((ROOT / name / "results" / cohort
                                / "official_score.json").read_text())
              for name in paths}
    names = {row["dataset"] for row in scores["model143"]["datasets"]}
    if (len(names) != 39 or any(scores[name]["status"] != "valid_and_scored"
                               or scores[name]["skipped"]
                               or sha(paths[name]) != scores[name]["submission_sha256"]
                               or {row["dataset"] for row in scores[name]["datasets"]} != names
                               or sha(paths[name]) != stage["cohorts"][cohort]["source_csv_sha256"][name]
                               for name in paths)):
        raise RuntimeError("Archived complete source score or CSV hash drift")
    controls = load_graphs(paths["model130"], names)
    predecessor = load_graphs(paths["model129"], names)
    old = load_graphs(paths["model118"], names)
    output = ROOT / "model147/results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate.csv"
    reports, totals = [], Counter()
    identifier = 0
    with candidate.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for movie, rows in dataset_blocks(paths["model143"]):
            if movie not in names:
                raise RuntimeError(f"Unexpected movie {movie}")
            nodes, current = parts(rows)
            base_nodes = controls[movie]["nodes"]
            base_edges = set(controls[movie]["edges"])
            stage129 = set(predecessor[movie]["edges"])
            stage118 = set(old[movie]["edges"])
            if (nodes != base_nodes or not base_edges <= current
                    or not base_edges <= stage129 or not stage129 <= stage118):
                raise RuntimeError(f"Graph provenance changed: {movie}")
            midpoint_removed = stage129 - base_edges
            tight_sister_removed = stage118 - stage129
            forbidden = (stage118 - current) - midpoint_removed
            if not tight_sister_removed <= forbidden | current:
                raise RuntimeError("Tight-sister veto not preserved")
            additions, stats = select(movie, rows, cohort, cfg, base_edges, forbidden)
            selected = {(row["parent"], row["orphan"]) for row in additions}
            if (movie.startswith("44b6_") and additions
                    or selected & tight_sister_removed
                    or any(edge in stage118 and edge not in midpoint_removed
                           for edge in selected)):
                raise RuntimeError("Model147 restored a forbidden edge")
            for row in rows:
                writer.writerow({**row, "id": identifier})
                identifier += 1
            for edge in additions:
                writer.writerow(dict(zip(COLUMNS,
                                         (identifier, movie, "edge", -1, -1, -1, -1, -1,
                                          edge["parent"], edge["orphan"]), strict=True)))
                identifier += 1
            reports.append({"movie": movie, "family": movie.split("_")[0],
                            "stats": stats, "midpoint_vetoed_edges": len(midpoint_removed),
                            "tight_sister_vetoed_edges": len(tight_sister_removed),
                            "restored_midpoint_edges": len(selected & midpoint_removed),
                            "additions": additions})
            totals.update(stats)
            totals["restored_midpoint_edges"] += len(selected & midpoint_removed)
            print(f"MODEL147 {cohort} {len(reports)}/39 {movie}: "
                  f"added={len(additions)} midpoint_restored={len(selected & midpoint_removed)}",
                  flush=True)
    if len(reports) != 39 or {row["movie"] for row in reports} != names:
        raise RuntimeError("Wrong movie coverage")
    before, after = validate(paths["model143"]), validate(candidate)
    if (set(before["datasets"]) != set(after["datasets"])
            or before["totals"]["nodes"] != after["totals"]["nodes"]
            or after["totals"]["edges"] - before["totals"]["edges"] != totals["selected"]):
        raise RuntimeError("Model143 graph preservation failed")
    dump(output / "selection_report.json", {"status": "valid", "cohort": cohort,
         "source_csv_sha256": {name: sha(path) for name, path in paths.items()},
         "candidate_sha256": sha(candidate), "config_sha256": sha(cfg_path),
         "stage_audit_sha256": sha(stage_path), "source_code_sha256": sha(Path(__file__)),
         "totals": dict(totals), "movies": reports,
         "caveat": "Label-free final graph; exact organizer scorer required."})


if __name__ == "__main__":
    main()
