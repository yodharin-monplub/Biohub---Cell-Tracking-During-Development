"""Check inference-only proposal enumeration against archived model120 development audit."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha
from model93.repair_endpoints import dataset_blocks
from model132.select import enumerate_proposals, qualifies, select


def main():
    out = ROOT / "model132/pilot.json"
    if out.exists():
        raise FileExistsError("Existing model132 pilot")
    cfg = json.loads((ROOT / "model132/config.json").read_text())
    if cfg["source_model"] != "model130" or cfg["minimum_link_probability"] != .6:
        raise RuntimeError("Frozen config changed")
    archive = json.loads((ROOT / "model120/development_audit.json").read_text())
    if archive["status"] != "complete" or len(archive["proposals"]) != 50015:
        raise RuntimeError("Archived development pool changed")
    source = ROOT / "model118/results/development/candidate.csv"
    if sha(source) != archive["model118_csv_sha256"]:
        raise RuntimeError("Archived control CSV changed")
    by_movie = defaultdict(dict)
    for row in archive["proposals"]:
        key = (row["parent"], row["orphan"])
        if key in by_movie[row["movie"]]:
            raise RuntimeError("Duplicate archived proposal")
        by_movie[row["movie"]][key] = row
    reports = []
    totals = Counter()
    for index, (movie, rows) in enumerate(dataset_blocks(source), 1):
        raw, _, _ = enumerate_proposals(movie, rows, "development", cfg)
        observed = {(row["parent"], row["orphan"]): row for row in raw}
        earlier = by_movie[movie]
        if observed.keys() != earlier.keys():
            raise RuntimeError(f"Broad-pool identity parity failed: {movie} "
                               f"missing={len(earlier.keys()-observed.keys())} "
                               f"extra={len(observed.keys()-earlier.keys())}")
        for pair, row in observed.items():
            old = earlier[pair]
            for field in ("probability", "parent_distance_um", "existing_daughter_distance_um",
                          "sister_distance_um", "midpoint_prediction_error_um"):
                if abs(row[field] - old[field]) > 1e-9:
                    raise RuntimeError(f"Feature parity failed: {movie}/{pair}/{field}")
            for field in ("parent_probability_rank", "orphan_probability_rank"):
                if row[field] != old[field]:
                    raise RuntimeError(f"Rank parity failed: {movie}/{pair}/{field}")
        rule = {pair for pair, row in observed.items() if qualifies(row, cfg)}
        old_rule = {pair for pair, row in earlier.items() if qualifies(row, cfg)}
        if rule != old_rule:
            raise RuntimeError(f"Frozen-rule parity failed: {movie}")
        verified = {pair for pair in rule if observed[pair]["raw_ids_verified"]}
        selected, stats = select(movie, rows, "development", cfg)
        labels = Counter(earlier[pair]["gt_label"] for pair in rule)
        verified_labels = Counter(earlier[pair]["gt_label"] for pair in verified)
        reports.append({"movie": movie, "broad": len(raw), "rule": len(rule),
                        "raw_verified": len(verified), "selected_after_caps": len(selected),
                        "labels": dict(labels), "raw_verified_labels": dict(verified_labels),
                        "stats": stats})
        totals.update({"broad": len(raw), "rule": len(rule),
                       "raw_verified": len(verified), "selected_after_caps": len(selected)})
        totals.update({f"rule_{key}": value for key, value in labels.items()})
        totals.update({f"verified_{key}": value for key, value in verified_labels.items()})
        print(f"PILOT {index}/39 {movie}: broad={len(raw)} rule={len(rule)} verified={len(verified)} selected={len(selected)}", flush=True)
    if len(reports) != 39 or totals["broad"] != 50015 or totals["rule"] != 141:
        raise RuntimeError("Frozen 39-movie audit totals differ")
    result = {"status": "parity_pass_stage_only", "totals": dict(totals), "movies": reports,
              "source_model118_csv_sha256": sha(source),
              "source_archive_sha256": sha(ROOT / "model120/development_audit.json"),
              "config_sha256": sha(ROOT / "model132/config.json"),
              "caveat": "Development-only proposal parity and sparse labels; not a full model132 score."}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "totals": result["totals"]}, indent=2))


if __name__ == "__main__":
    main()
