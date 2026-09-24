"""Full paired score against best model130 and incremental model132 source."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model92.compare_official import compare, read_score


def main():
    output = ROOT / "model133/results/comparison.json"
    if output.exists():
        raise FileExistsError("Existing model133 comparison")
    cohorts = {}
    for name in ("development", "confirmation"):
        best = ROOT / "model130/results" / name / "official_score.json"
        source = ROOT / "model132/results" / name / "official_score.json"
        candidate = ROOT / "model133/results" / name / "official_score.json"
        receipt = json.loads((ROOT / "model133/results" / name / "selection_report.json").read_text())
        scored = json.loads(candidate.read_text())
        if receipt["status"] != "valid" or scored["submission_sha256"] != receipt["candidate_sha256"]:
            raise RuntimeError("Candidate score not tied to complete selection")
        cohorts[name] = {"versus_best_model130": compare(read_score(best), read_score(candidate)),
                         "incremental_versus_model132": compare(read_score(source), read_score(candidate))}
    gates = {name: row["versus_best_model130"]["status"] == "eligible_for_independent_confirmation"
             for name, row in cohorts.items()}
    status = "eligible_for_review" if all(gates.values()) else "not_promoted"
    result = {"status": status, "gates": gates, "cohorts": cohorts,
              "caveat": "Both cohorts informed rule design; no untouched validation or public Kaggle claim."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": status,
                      "scores": {name: {"model130": row["versus_best_model130"]["control"]["score"],
                                        "model132": row["incremental_versus_model132"]["control"]["score"],
                                        "model133": row["versus_best_model130"]["candidate"]["score"],
                                        "delta_vs_best": row["versus_best_model130"]["delta"]}
                                 for name, row in cohorts.items()}}, indent=2))


if __name__ == "__main__":
    main()
