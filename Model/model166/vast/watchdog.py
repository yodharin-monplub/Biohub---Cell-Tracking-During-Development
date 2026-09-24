#!/usr/bin/env python3
"""Quiet independent model166 rental stop guard; never deletes remote data."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from client import call


def main() -> None:
    rental = json.loads((HERE / "rental.json").read_text())
    instance_id = int(rental["instance_id"])
    while True:
        remaining = float(rental["hard_stop_deadline_unix"]) - time.time()
        if (HERE / "cleanup.json").exists():
            return
        if remaining <= 0 or (HERE / "request_stop").exists():
            result = call("PUT", f"instances/{instance_id}/", {"state": "stopped"})
            report = {"status": "stop_requested", "instance_id": instance_id,
                      "acknowledged": bool(result.get("success")),
                      "reason": "cost_deadline" if remaining <= 0 else "run_finished_or_failed",
                      "updated_unix": time.time(), "alarm_enabled": False}
            (HERE / "watchdog_status.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            return
        report = {"status": "armed", "instance_id": instance_id,
                  "seconds_to_stop": remaining, "monitor_interval_seconds": 600,
                  "updated_unix": time.time(), "alarm_enabled": False}
        (HERE / "watchdog_status.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        time.sleep(min(600, remaining))


if __name__ == "__main__":
    main()
