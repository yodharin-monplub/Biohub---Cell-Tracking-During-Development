"""Paired exact organizer comparison for model150 versus model143 and149."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model92.compare_official import compare, read_score


def main():
    output = ROOT / "model150/results/comparison.json"
    if output.exists():
        raise FileExistsError("Existing paired comparison")
    cohorts = {}
    for cohort in ("development", "confirmation"):
        candidate = ROOT / "model150/results" / cohort / "official_score.json"
        report = json.loads((ROOT / "model150/results" / cohort
                             / "selection_report.json").read_text())
        scored = json.loads(candidate.read_text())
        if (report["status"] != "valid" or scored["status"] != "valid_and_scored"
                or scored["skipped"] or scored["submission_sha256"] != report["candidate_sha256"]):
            raise RuntimeError("Candidate score not tied to complete selected graph")
        cohorts[cohort] = {
            "versus_model143": compare(
                read_score(ROOT / "model143/results" / cohort / "official_score.json"),
                read_score(candidate)),
            "versus_model149": compare(
                read_score(ROOT / "model149/results" / cohort / "official_score.json"),
                read_score(candidate))}
    gates = {cohort: rows["versus_model143"]["status"]
             == "eligible_for_independent_confirmation" for cohort, rows in cohorts.items()}
    status = "eligible_for_review" if all(gates.values()) else "not_promoted"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": status, "gates": gates, "cohorts": cohorts,
                                  "caveat": "Both score cohorts reused; no public/Kaggle claim."},
                                 indent=2) + "\n")
    print(json.dumps({"status": status,
                      "scores": {cohort: {"model143": rows["versus_model143"]["control"]["score"],
                                          "model149": rows["versus_model149"]["control"]["score"],
                                          "candidate": rows["versus_model143"]["candidate"]["score"]}
                                 for cohort, rows in cohorts.items()}}, indent=2))


if __name__ == "__main__":
    main()
