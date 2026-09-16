"""Train-only capacity check for pruning only model133 low-growth links."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha


def main():
    output = ROOT / "model148/pilot.json"
    if output.exists():
        raise FileExistsError("Existing model148 pilot")
    cfg_path = ROOT / "model148/config.json"
    cfg = json.loads(cfg_path.read_text())
    audit_path = ROOT / "model144/audit.json"
    audit = json.loads(audit_path.read_text())
    if (cfg["source_model"] != "model145" or cfg["target_branch"] != "model133"
            or cfg["recovery_family"] != "6bba" or cfg["minimum_sister_growth_um"] != 2.
            or audit["status"] != "complete"
            or audit["model144_receipt_sha256"] != sha(ROOT / "model144/capture/receipt.json")):
        raise RuntimeError("Frozen branch threshold or train-only provenance changed")
    rows = [row for row in audit["proposals"] if row["model133_only"]]
    kept = [row for row in rows
            if row["sister_separation_growth_um"] >= cfg["minimum_sister_growth_um"]]
    labels = lambda pool: dict(Counter(row["gt_label"] for row in pool))
    result = {"status": "complete", "config_sha256": sha(cfg_path),
              "model144_audit_sha256": sha(audit_path),
              "before": {"n": len(rows), "labels": labels(rows)},
              "after": {"n": len(kept), "labels": labels(kept)},
              "caveat": "Approximate model1 post-ILP proposal rule, not final model145 "
                        "selected edges. Sparse-GT unknowns are not negatives."}
    dump(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
