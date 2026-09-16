#!/usr/bin/env python3
"""Archive the verified-empty first rental so a fresh rental can be recorded."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rental = json.loads((HERE / "rental.json").read_text())
stopped = json.loads((HERE / "manual_stop.json").read_text())
cleanup = json.loads((HERE / "empty_cleanup.json").read_text())
instance_id = int(rental["instance_id"])
if not (int(stopped.get("instance_id", -1)) == instance_id
        and int(cleanup.get("instance_id", -1)) == instance_id
        and stopped.get("data_uploaded") is False
        and stopped.get("training_started") is False
        and cleanup.get("status") == "empty_instance_destroyed"
        and cleanup.get("data_lost") is False):
    raise RuntimeError("First-attempt receipts do not prove safe archival")
if any((HERE / name).exists() for name in ("job_status.json", "upload.log", "retrieval.json")):
    raise RuntimeError("Run artifacts exist; refusing empty-attempt archival")
destination = HERE / f"attempt_{instance_id}_empty"
destination.mkdir(exist_ok=False)
names = ["rental.json", "manual_stop.json", "empty_cleanup.json"]
if (HERE / "watchdog_status.json").exists():
    names.append("watchdog_status.json")
for name in names:
    (HERE / name).rename(destination / name)
summary = {"status": "archived_verified_empty_attempt", "instance_id": instance_id,
           "files": names, "ready_for_fresh_rental": True}
(destination / "archive_receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
