"""GT-safe audit of expanded train-only 6bba orphan proposals."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from tracksdata.options import set_options

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model120.audit import movie_proposals
from model140.audit import graph_rows, gt_division_coverage
from model132.select import qualifies as high_confidence_rule
from model133.select import qualifies as broad_recovery_rule


def describe(values):
    return {"n": len(values),
            "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None}


def main():
    out = ROOT / "model144/audit.json"
    if out.exists():
        raise FileExistsError("Model144 audit already exists")
    cfg_path = ROOT / "model144/config.json"
    cfg = json.loads(cfg_path.read_text())
    capture_dir = ROOT / "model144/capture"
    manifest_path = capture_dir / "manifest.json"
    receipt_path = capture_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    by_movie = {row["movie"]: row for row in receipt["movies"]}
    if (receipt["status"] != "complete" or not receipt["preflight_pass"]
            or not receipt["all_train_only"] or len(by_movie) != 16
            or set(by_movie) != {cfg["preflight_movie"], *cfg["train_only_movies"]}
            or receipt["manifest_sha256"] != sha(manifest_path)
            or manifest["config_sha256"] != sha(cfg_path)
            or any(sha(capture_dir / f"{movie}.npz") != by_movie[movie]["capture_sha256"]
                   for movie in by_movie)):
        raise RuntimeError("Frozen 15-movie capture receipt or hashes invalid")
    prior_audit_path = ROOT / "model140/audit.json"
    prior = json.loads(prior_audit_path.read_text())
    if prior["status"] != "complete" or prior["source_capture_receipt_sha256"] != sha(
            ROOT / "model140/capture/receipt.json"):
        raise RuntimeError("Prior train-only audit drift")
    prior_movies = [row for row in prior["movies"] if row["family"] == "6bba"]
    if len(prior_movies) != 5:
        raise RuntimeError("Prior 6bba movie count changed")
    prior_names = {row["movie"] for row in prior_movies}
    if prior_names & set(cfg["train_only_movies"]):
        raise RuntimeError("Train-only captures overlap")
    high_cfg = json.loads((ROOT / "model132/config.json").read_text())
    broad_cfg = json.loads((ROOT / "model133/config.json").read_text())
    set_options(show_progress=False)
    reports = []
    proposals = [row for row in prior["proposals"] if row["movie"] in prior_names]
    total_divisions = sum(row["gt_division_count"] for row in prior_movies)
    for index, movie in enumerate(cfg["train_only_movies"], 1):
        with np.load(capture_dir / f"{movie}.npz") as archive:
            arrays = {key: archive[key].copy() for key in
                      ("post_ilp_nodes", "post_ilp_edges", "source", "target", "probability")}
        group = graph_rows(movie, arrays)
        rows, stage = movie_proposals(movie, group, set(), capture_dir)
        divisions = gt_division_coverage(movie, group, arrays)
        reports.append({"movie": movie, "family": "6bba",
                        "source_capture_sha256": by_movie[movie]["capture_sha256"],
                        "gt_division_count": len(divisions),
                        "one_direct_plus_orphan": sum(row["one_direct_plus_orphan"] for row in divisions),
                        "proposal_stage": stage,
                        "labels": dict(Counter(row["gt_label"] for row in rows)),
                        "gt_divisions": divisions})
        total_divisions += len(divisions)
        proposals.extend(rows)
        print(f"MODEL144 AUDIT {index}/15 {movie}: GT divisions={len(divisions)} "
              f"orphan cases={reports[-1]['one_direct_plus_orphan']} "
              f"proposals={len(rows)} labels={reports[-1]['labels']}", flush=True)
    if total_divisions != (cfg["expected_train_only_annotated_divisions"]
                           + sum(row["gt_division_count"] for row in prior_movies)):
        raise RuntimeError("Annotated division count changed")
    for row in proposals:
        row["model132_rule"] = high_confidence_rule(row, high_cfg)
        row["model133_rule"] = broad_recovery_rule(row, broad_cfg)
        row["model133_only"] = row["model133_rule"] and not row["model132_rule"]
    labels = dict(Counter(row["gt_label"] for row in proposals))
    stages = {}
    for stage in ("all", "model132_rule", "model133_only"):
        selected = proposals if stage == "all" else [row for row in proposals if row[stage]]
        stages[stage] = {
            "labels": dict(Counter(row["gt_label"] for row in selected)),
            "explicit_positive_growth_um": describe([
                row["sister_separation_growth_um"] for row in selected
                if row["gt_label"] == "explicit_division_positive"]),
            "explicit_contradiction_growth_um": describe([
                row["sister_separation_growth_um"] for row in selected
                if row["gt_label"] == "explicit_link_contradiction"]),
        }
    dump(out, {"status": "complete", "model144_receipt_sha256": sha(receipt_path),
               "model140_audit_sha256": sha(prior_audit_path),
               "source_code_sha256": sha(Path(__file__)),
               "movies": [*prior_movies, *reports], "stages": stages,
               "totals": {"movies": len(prior_movies) + len(reports),
                          "gt_divisions": total_divisions, "proposals": len(proposals),
                          "labels": labels}, "proposals": proposals,
               "caveat": "GT-safe sparse labels. Unknown proposals are NOT negatives; "
                         "model132/133 rule flags omit final-model veto/raw-ID/caps. "
                         "Train-only model1 post-ILP graph differs from model143."})
    print(json.dumps({"status": "complete", "stages": stages,
                      "gt_divisions": total_divisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
