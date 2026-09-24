"""Compare complete exact organizer scores of model137 with model130."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model92.compare_official import compare, read_score


def main():
    output = ROOT / "model137/results/comparison.json"
    if output.exists():
        raise FileExistsError("Existing paired comparison")
    cohorts = {}
    for name in ("development", "confirmation"):
        base = ROOT / "model130/results" / name / "official_score.json"
        candidate = ROOT / "model137/results" / name / "official_score.json"
        receipt = json.loads((ROOT / "model137/results" / name / "selection_report.json").read_text())
        scored = json.loads(candidate.read_text())
        if (receipt["status"] != "valid" or scored["status"] != "valid_and_scored"
                or scored["skipped"] or scored["submission_sha256"] != receipt["candidate_sha256"]):
            raise RuntimeError("Candidate score not tied to complete label-free selection")
        cohorts[name] = compare(read_score(base), read_score(candidate))
    gates = {name: cohorts[name]["status"] == "eligible_for_independent_confirmation" for name in cohorts}
    status = "eligible_for_review" if all(gates.values()) else "not_promoted"
    result = {"status": status, "gates": gates, "cohorts": cohorts,
              "caveat": "Both cohorts reused; not untouched CV or public Kaggle score."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": status,
                      "scores": {name: {"control": row["control"]["score"],
                                        "candidate": row["candidate"]["score"],
                                        "delta": row["delta"]}
                                 for name, row in cohorts.items()}}, indent=2))


if __name__ == "__main__":
    main()
