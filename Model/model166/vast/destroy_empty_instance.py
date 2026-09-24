#!/usr/bin/env python3
"""Destroy only the stopped model166 instance proven empty before upload."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from client import call, sanitized_status, save_new

rental = json.loads((HERE / "rental.json").read_text())
stopped = json.loads((HERE / "manual_stop.json").read_text())
instance_id = int(rental["instance_id"])
if (int(stopped.get("instance_id", -1)) != instance_id
        or stopped.get("status") != "stopped_before_upload"
        or stopped.get("data_uploaded") is not False
        or stopped.get("training_started") is not False):
    raise RuntimeError("Empty-instance proof is incomplete")
for name in ("job_status.json", "upload.log", "retrieval.json"):
    if (HERE / name).exists():
        raise RuntimeError(f"Refusing deletion because {name} exists")
state = sanitized_status(instance_id)
if (state.get("cur_state") != "stopped"
        or state.get("intended_status") != "stopped"
        or state.get("actual_status") not in {"created", "exited", "stopped"}):
    raise RuntimeError(f"Instance is not stopped: {state}")
result = call("DELETE", f"instances/{instance_id}/")
if not result.get("success"):
    raise RuntimeError("Vast.ai did not acknowledge instance deletion")
receipt = {"status": "empty_instance_destroyed", "instance_id": instance_id,
           "updated_unix": time.time(), "data_lost": False,
           "reason": "avoid storage charges while external-data authorization is pending"}
save_new(HERE / "empty_cleanup.json", receipt)
print(json.dumps(receipt))
