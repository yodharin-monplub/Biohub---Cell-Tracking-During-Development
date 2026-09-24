"""Freeze intermediate model133 growth threshold on train-only labels."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model100.capture import dump, sha


def main():
    output = ROOT / "model149/pilot.json"
    if output.exists():
        raise FileExistsError("Existing model149 pilot")
    cfg_path = ROOT / "model149/config.json"
    cfg = json.loads(cfg_path.read_text())
    old_cfg = json.loads((ROOT / "model148/config.json").read_text())
    if (cfg["minimum_sister_growth_um"] != 1.5
            or old_cfg["minimum_sister_growth_um"] != 2.
            or {key: value for key, value in cfg.items()
                if key != "minimum_sister_growth_um"}
            != {key: value for key, value in old_cfg.items()
                if key != "minimum_sister_growth_um"}):
        raise RuntimeError("More than the growth threshold changed")
    audit_path = ROOT / "model144/audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "complete":
        raise RuntimeError("Expanded train-only audit unavailable")
    before = [row for row in audit["proposals"] if row["model133_only"]]
    after = [row for row in before
             if row["sister_separation_growth_um"] >= cfg["minimum_sister_growth_um"]]
    labels = lambda rows: dict(Counter(row["gt_label"] for row in rows))
    result = {"status": "complete", "config_sha256": sha(cfg_path),
              "model144_audit_sha256": sha(audit_path),
              "before": {"n": len(before), "labels": labels(before)},
              "after": {"n": len(after), "labels": labels(after)},
              "caveat": "Sparse-GT unknown is not negative; approximate model1 post-ILP "
                        "rule coverage, not final model145 graph or precision."}
    dump(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
