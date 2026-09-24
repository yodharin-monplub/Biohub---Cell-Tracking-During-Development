"""Apply frozen promotion gates to both paired organizer-score cohorts."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model106"
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump


def main():
    rows = {}
    for cohort in ("development", "confirmation"):
        folder = OUT / "results" / cohort
        receipt = json.loads((folder / "replay_receipt.json").read_text())
        assert receipt["status"] == "complete" and receipt["model1_unchanged"]
        assert sha(folder / "candidate.csv") == receipt["candidate_sha256"]
        rows[cohort] = {}
        for control in ("model1", "model104"):
            item = json.loads((folder / f"comparison_{control}.json").read_text())
            rows[cohort][control] = item
    gates = {f"{cohort}_beats_{control}": rows[cohort][control]["status"] == "eligible_for_independent_confirmation"
             for cohort in rows for control in rows[cohort]}
    status = "eligible_for_review" if all(gates.values()) else "not_promoted"
    dump(OUT / "results/comparison.json", dict(status=status, gates=gates,
         cohorts={cohort: {control: {
             "control_score": rows[cohort][control]["control"]["score"],
             "candidate_score": rows[cohort][control]["candidate"]["score"],
             "delta": rows[cohort][control]["delta"],
             "families": {name: data["delta"] for name, data in rows[cohort][control]["families"].items()},
             "node_recall_delta": rows[cohort][control]["candidate"]["node_recall"] - rows[cohort][control]["control"]["node_recall"],
         } for control in rows[cohort]} for cohort in rows},
         caveat="Both movie groups have been reused. No public/Kaggle or untouched-holdout claim."))
    print(json.dumps(json.loads((OUT / "results/comparison.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
