#!/usr/bin/env python3
"""Restart the exact stopped replacement after explicit ID-specific approval."""
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
if instance_id != 51104608 or int(stopped.get("instance_id", -1)) != instance_id:
    raise RuntimeError("This restart gate is pinned to replacement instance 51104608")
if stopped.get("data_uploaded") is not False or stopped.get("training_started") is not False:
    raise RuntimeError("Instance is not proven empty")
state = sanitized_status(instance_id)
if (state.get("cur_state") != "stopped"
        or state.get("intended_status") != "stopped"
        or state.get("actual_status") not in {"created", "exited", "stopped"}):
    raise RuntimeError(f"Instance is not stopped: {state}")
result = call("PUT", f"instances/{instance_id}/", {"state": "running"})
if not result.get("success"):
    public_error = {key: result.get(key) for key in ("success", "msg", "error", "detail")
                    if key in result}
    raise RuntimeError(f"Vast.ai did not acknowledge restart: {public_error}")
receipt = {"status": "restart_requested_after_exact_authorization",
           "instance_id": instance_id, "updated_unix": time.time(),
           "authorized_payload": "81GB Biohub competition train images and ground-truth labels",
           "purpose": "model166 training only"}
save_new(HERE / "restart_receipt.json", receipt)
print(json.dumps(receipt))
