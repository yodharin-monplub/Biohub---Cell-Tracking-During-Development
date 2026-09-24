#!/usr/bin/env python3
"""Stop only the model166 rental and record the acknowledgement."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from client import call, save_new

rental = json.loads((HERE / "rental.json").read_text())
instance_id = int(rental["instance_id"])
result = call("PUT", f"instances/{instance_id}/", {"state": "stopped"})
if not result.get("success"):
    raise RuntimeError("Vast.ai did not acknowledge the stop request")
receipt = {"status": "stopped_before_upload", "instance_id": instance_id,
           "updated_unix": time.time(), "data_uploaded": False,
           "training_started": False, "disk_retained": True,
           "storage_charges_may_continue": True, "alarm_enabled": False}
save_new(HERE / "manual_stop.json", receipt)
print(json.dumps(receipt))
