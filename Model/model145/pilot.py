"""Held-out train-only capacity check for the frozen model145 rule."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha
from model145.select import qualifies


def main():
    output = ROOT / "model145/pilot.json"
    if output.exists():
        raise FileExistsError("Existing model145 train-only pilot")
    cfg_path = ROOT / "model145/config.json"
    cfg = json.loads(cfg_path.read_text())
    old_audit_path = ROOT / "model140/audit.json"
    new_audit_path = ROOT / "model144/audit.json"
    old = json.loads(old_audit_path.read_text())
    new = json.loads(new_audit_path.read_text())
    new_names = set(json.loads((ROOT / "model144/config.json").read_text())["train_only_movies"])
    if (old["status"] != "complete" or sha(old_audit_path) != cfg["train_only_source_audit_sha256"]
            or new["status"] != "complete" or new["model140_audit_sha256"] != sha(old_audit_path)
            or new["model144_receipt_sha256"] != sha(ROOT / "model144/capture/receipt.json")
            or len(new_names) != 15):
        raise RuntimeError("Frozen train-only source audits changed")
    groups = {}
    for label, names in (("selection_five", {row["movie"] for row in old["movies"]
                                               if row["family"] == "6bba"}),
                         ("heldout_fifteen", new_names)):
        proposals = [row for row in new["proposals"] if row["movie"] in names]
        selected = [row for row in proposals if qualifies(row, cfg)]
        per_movie = {movie: dict(Counter(row["gt_label"] for row in selected
                                          if row["movie"] == movie))
                     for movie in sorted(names)}
        groups[label] = {"movies": len(names), "all_proposals": len(proposals),
                         "selected": len(selected),
                         "labels": dict(Counter(row["gt_label"] for row in selected)),
                         "per_movie": per_movie,
                         "positive_movies": sum(v.get("explicit_division_positive", 0) > 0
                                                for v in per_movie.values())}
    result = {"status": "complete", "config_sha256": sha(cfg_path),
              "model140_audit_sha256": sha(old_audit_path),
              "model144_audit_sha256": sha(new_audit_path),
              "groups": groups,
              "caveat": "Sparse-GT unknowns are not negatives. This omits final-model "
                        "raw-ID/prior-veto/cap checks; exact full-graph scoring is required."}
    dump(output, result)
    print(json.dumps({"status": "complete", "groups": groups}, indent=2), flush=True)


if __name__ == "__main__":
    main()
