"""Require paired full organizer score gain over model118."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import sha, dump
from model92.compare_official import compare, read_score


def main():
    output = ROOT / "model124/results"
    if (output / "comparison.json").exists():
        raise FileExistsError("Comparison already exists")
    cohorts = {}
    for name in ("development", "confirmation"):
        base = ROOT / "model118/results" / name / "official_score.json"
        candidate = output / name / "official_score.json"
        receipt = json.loads((output / name / "replay_receipt.json").read_text())
        scored = json.loads(candidate.read_text())
        if receipt["status"] != "complete" or scored["submission_sha256"] != receipt["candidate_sha256"]:
            raise RuntimeError("Scored output not tied to complete replay")
        cohorts[name] = compare(read_score(base), read_score(candidate))
    gates = {name: cohorts[name]["status"] == "eligible_for_independent_confirmation" for name in cohorts}
    status = "eligible_for_review" if all(gates.values()) else "not_promoted"
    dump(output / "comparison.json", {"status": status, "gates": gates, "cohorts": cohorts,
         "pilot_sha256": sha(ROOT / "model124/pilot.json"),
         "caveat": "Both cohorts are reused; no public/Kaggle or 0.97 claim."})
    print(json.dumps({"status": status,
                      "scores": {name: {"control": v["control"]["score"],
                                        "candidate": v["candidate"]["score"], "delta": v["delta"]}
                                 for name, v in cohorts.items()}}, indent=2))


if __name__ == "__main__":
    main()
