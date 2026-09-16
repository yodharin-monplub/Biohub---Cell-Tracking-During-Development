"""Inference-only selected-edge burden before held-out labels or scoring."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model93.repair_endpoints import dataset_blocks
from model145.select import select


def edges(rows):
    return {(int(row["source_id"]), int(row["target_id"]))
            for row in rows if row["row_type"] == "edge"}


def main():
    output = ROOT / "model145/dry_run.json"
    if output.exists():
        raise FileExistsError("Existing model145 dry run")
    cfg_path = ROOT / "model145/config.json"
    cfg = json.loads(cfg_path.read_text())
    cohorts = {}
    for cohort in ("development", "confirmation"):
        paths = {name: ROOT / name / "results" / cohort / "candidate.csv"
                 for name in ("model143", "model130", "model118")}
        scores = {name: json.loads((ROOT / name / "results" / cohort
                                    / "official_score.json").read_text())
                  for name in paths}
        names = {row["dataset"] for row in scores["model143"]["datasets"]}
        if (len(names) != 39 or any(scores[name]["status"] != "valid_and_scored"
                                   or scores[name]["skipped"]
                                   or sha(paths[name]) != scores[name]["submission_sha256"]
                                   or {row["dataset"] for row in scores[name]["datasets"]} != names
                                   for name in paths)):
            raise RuntimeError("Scored source CSV drift")
        reports, total = [], Counter()
        generators = [dataset_blocks(paths[name]) for name in
                      ("model143", "model130", "model118")]
        for index, triple in enumerate(zip(*generators, strict=True), 1):
            (movie, rows), (base_movie, base_rows), (old_movie, old_rows) = triple
            if movie != base_movie or movie != old_movie or movie not in names:
                raise RuntimeError("Source movie order differs")
            current, base, old = edges(rows), edges(base_rows), edges(old_rows)
            if not base <= current:
                raise RuntimeError("Model143 does not contain model130")
            additions, stats = select(movie, rows, cohort, cfg, base, old - current)
            reports.append({"movie": movie, "family": movie.split("_")[0],
                            "stats": stats, "first_two": additions[:2]})
            total.update(stats)
            print(f"MODEL145 DRY {cohort} {index}/39 {movie}: "
                  f"new={len(additions)} rule={stats['rule_proposals']}", flush=True)
        if len(reports) != 39:
            raise RuntimeError("Incomplete dry run")
        cohorts[cohort] = {"totals": dict(total), "movies": reports,
                           "source_csv_sha256": {name: sha(path) for name, path in paths.items()}}
    dump(output, {"status": "inference_only_dry_run", "config_sha256": sha(cfg_path),
                  "source_code_sha256": sha(Path(__file__)), "cohorts": cohorts,
                  "caveat": "Selected-edge count only. No GT was read, no CSV candidate "
                            "was written, and no score or precision is implied."})
    print(json.dumps({cohort: result["totals"] for cohort, result in cohorts.items()}, indent=2))


if __name__ == "__main__":
    main()
