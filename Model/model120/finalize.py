"""Paired official-score gate for the frozen model120 orphan branch addition."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model92.compare_official import compare, read_score


def main():
    out = ROOT / "model120/results"
    if (out / "comparison.json").exists():
        raise FileExistsError("Existing comparison")
    cohorts = {}
    for name in ("development", "confirmation"):
        control = ROOT / "model118/results" / name / "official_score.json"
        candidate = out / name / "official_score.json"
        selection = json.loads((out / name / "selection_report.json").read_text())
        scored = json.loads(candidate.read_text())
        if scored["submission_sha256"] != selection["candidate_sha256"]:
            raise RuntimeError("Scored candidate hash mismatch")
        cohorts[name] = compare(read_score(control), read_score(candidate))
    gates = {name: cohorts[name]["status"] == "eligible_for_independent_confirmation" for name in cohorts}
    status = "eligible_for_review" if all(gates.values()) else "not_promoted"
    dump(out / "comparison.json", {"status": status, "gates": gates, "cohorts": cohorts,
         "config_sha256": sha(ROOT / "model120/frozen_config.json"),
         "caveat": "Reused two39-movie cohorts; no public/Kaggle score or 0.97 claim."})
    print(json.dumps({"status": status, "scores": {name: {"control": v["control"]["score"],
                                                    "candidate": v["candidate"]["score"],
                                                    "delta": v["delta"]}
                                            for name, v in cohorts.items()}}, indent=2))


if __name__ == "__main__":
    main()
