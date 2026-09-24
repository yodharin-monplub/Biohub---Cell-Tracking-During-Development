"""Aggregate exact organizer results for the frozen model118 division veto."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model92.compare_official import compare, read_score


def main():
    out = ROOT / "model118/results"
    if (out / "comparison.json").exists():
        raise FileExistsError("Existing comparison")
    cohorts = {}
    for name in ("development", "confirmation"):
        source = ROOT / "model107/results" / name / "official_score.json"
        candidate = out / name / "official_score.json"
        report = json.loads((out / name / "veto_report.json").read_text())
        scored = json.loads(candidate.read_text())
        if scored["submission_sha256"] != report["candidate_sha256"]:
            raise RuntimeError("Scored CSV does not match frozen transform")
        cohorts[name] = compare(read_score(source), read_score(candidate))
    gates = {name: cohorts[name]["status"] == "eligible_for_independent_confirmation"
             for name in cohorts}
    dump(out / "comparison.json", {"status": "eligible_for_review" if all(gates.values()) else "not_promoted",
         "gates": gates, "cohorts": cohorts, "config_sha256": sha(ROOT / "model118/config.json"),
         "model107_sha256": sha(ROOT / "model107/submission.ipynb"),
         "caveat": "Both 39-movie cohorts are reused; no Kaggle/public validation or 0.97 claim."})
    print(json.dumps({"status": "eligible_for_review" if all(gates.values()) else "not_promoted",
                      "scores": {name: {"control": v["control"]["score"],
                                         "candidate": v["candidate"]["score"], "delta": v["delta"]}
                                 for name, v in cohorts.items()}}, indent=2))


if __name__ == "__main__":
    main()
