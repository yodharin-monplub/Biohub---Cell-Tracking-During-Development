"""Re-threshold frozen model137 real-proposal scores at train-only OOF 0.80."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model93.repair_endpoints import dataset_blocks
from scripts.validate_submission import COLUMNS, validate

THRESHOLD = .80


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    cohort = args.cohort
    classifier_path = ROOT / "model137/classifier.json"
    classifier = json.loads(classifier_path.read_text())
    if (classifier["status"] != "frozen" or classifier["threshold"] != .95
            or classifier["oof_at_threshold"] != {"tp": 19, "fp": 0, "fn": 35, "tn": 257}):
        raise RuntimeError("Frozen model137 classifier changed")
    prior_report_path = ROOT / "model137/results" / cohort / "selection_report.json"
    prior = json.loads(prior_report_path.read_text())
    source = ROOT / "model130/results" / cohort / "candidate.csv"
    control = json.loads((ROOT / "model130/results" / cohort / "official_score.json").read_text())
    if (prior["status"] != "valid" or prior["cohort"] != cohort
            or prior["source_model130_sha256"] != sha(source)
            or prior["classifier_sha256"] != sha(classifier_path)
            or control["status"] != "valid_and_scored" or control["skipped"]
            or control["submission_sha256"] != sha(source)
            or len(prior["movies"]) != 39):
        raise RuntimeError("Real-proposal source or scored control changed")
    prior_scored = json.loads((ROOT / "model137/results" / cohort / "official_score.json").read_text())
    if (prior_scored["status"] != "valid_and_scored" or prior_scored["skipped"]
            or prior_scored["submission_sha256"] != prior["candidate_sha256"]):
        raise RuntimeError("Prior frozen replay score not verified")
    by_movie = {r["movie"]: r for r in prior["movies"]}
    names = {r["dataset"] for r in control["datasets"]}
    if set(by_movie) != names or len(names) != 39:
        raise RuntimeError("Wrong real-proposal movie coverage")
    output = ROOT / "model138/results" / cohort
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate.csv"
    reports = []
    total = Counter()
    identifier = 0
    with candidate.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for movie, rows in dataset_blocks(source):
            scores = by_movie[movie]["evaluated"]
            original = {tuple(edge) for edge in by_movie[movie]["accepted_edges"]}
            prior_recomputed = {(int(r["parent"]), int(r["orphan"])) for r in scores
                                if float(r["image_score"]) >= .95}
            if original != prior_recomputed:
                raise RuntimeError(f"Frozen model137 accepted-edge parity failed: {movie}")
            selected = [(int(r["parent"]), int(r["orphan"])) for r in scores
                        if float(r["image_score"]) >= THRESHOLD]
            if len(selected) != len(set(selected)) or not original <= set(selected):
                raise RuntimeError("New threshold lost prior edges or repeated a proposal")
            old_edges = {(int(row["source_id"]), int(row["target_id"]))
                         for row in rows if row["row_type"] == "edge"}
            if old_edges & set(selected):
                raise RuntimeError("Proposed edge already in control")
            for row in rows:
                writer.writerow({**row, "id": identifier})
                identifier += 1
            for a, b in selected:
                writer.writerow(dict(zip(COLUMNS, (identifier, movie, "edge", -1, -1,
                                                 -1, -1, -1, a, b), strict=True)))
                identifier += 1
            reports.append({"movie": movie, "evaluated": len(scores),
                            "prior_accepted": len(original), "accepted": len(selected),
                            "newly_accepted": len(selected) - len(original),
                            "selected_edges": selected})
            total.update({"evaluated": len(scores), "prior_accepted": len(original),
                          "accepted": len(selected), "newly_accepted": len(selected) - len(original)})
            print(f"MODEL138 {cohort} {len(reports)}/39 {movie}: accepted={len(selected)}"
                  f" (+{len(selected)-len(original)} over model137)", flush=True)
    before, after = validate(source), validate(candidate)
    if (set(before["datasets"]) != set(after["datasets"])
            or before["totals"]["nodes"] != after["totals"]["nodes"]
            or after["totals"]["edges"] - before["totals"]["edges"] != total["accepted"]):
        raise RuntimeError("Control graph preservation failed")
    dump(output / "selection_report.json", {"status": "valid", "cohort": cohort,
         "source_model130_sha256": sha(source), "prior_model137_report_sha256": sha(prior_report_path),
         "classifier_sha256": sha(classifier_path), "candidate_sha256": sha(candidate),
         "source_code_sha256": sha(Path(__file__)), "cutoff": THRESHOLD,
         "totals": dict(total), "movies": reports,
         "caveat": "One inference-only cutoff change; exact scorer required."})
    print(json.dumps({"cohort": cohort, "totals": dict(total)}))


if __name__ == "__main__":
    main()
