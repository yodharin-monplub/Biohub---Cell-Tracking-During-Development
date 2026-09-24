"""Paired exact organizer comparison for model145 versus model143."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model92.compare_official import compare, read_score


def main():
    output = ROOT / "model145/results/comparison.json"
    if output.exists():
        raise FileExistsError("Existing paired comparison")
    cohorts = {}
    for name in ("development", "confirmation"):
        base = ROOT / "model143/results" / name / "official_score.json"
        candidate = ROOT / "model145/results" / name / "official_score.json"
        report = json.loads((ROOT / "model145/results" / name / "selection_report.json").read_text())
        scored = json.loads(candidate.read_text())
        if (report["status"] != "valid" or scored["status"] != "valid_and_scored"
                or scored["skipped"] or scored["submission_sha256"] != report["candidate_sha256"]):
            raise RuntimeError("Candidate score not tied to complete selected graph")
        cohorts[name] = compare(read_score(base), read_score(candidate))
    gates = {name: row["status"] == "eligible_for_independent_confirmation"
             for name, row in cohorts.items()}
    status = "eligible_for_review" if all(gates.values()) else "not_promoted"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": status, "gates": gates, "cohorts": cohorts,
                                  "caveat": "Both score cohorts reused; no public/Kaggle claim."},
                                 indent=2) + "\n")
    print(json.dumps({"status": status,
                      "scores": {name: {"control": row["control"]["score"],
                                        "candidate": row["candidate"]["score"],
                                        "delta": row["delta"]}
                                 for name, row in cohorts.items()}}, indent=2))


if __name__ == "__main__":
    main()
