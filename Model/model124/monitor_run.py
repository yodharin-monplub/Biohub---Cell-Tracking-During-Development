"""Quiet 600-second supervisor for the full model124 paired score."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model124"


def main():
    status = OUT / "status.json"
    if status.exists():
        raise FileExistsError("Model124 status exists; inspect before any restart")
    started = time.time()
    with (OUT / "run.log").open("x") as log:
        child = subprocess.Popen(["bash", str(OUT / "run.sh")], cwd=ROOT,
                                 stdout=log, stderr=subprocess.STDOUT)
        while True:
            state = {"status": "running", "pid": child.pid, "supervisor_pid": os.getpid(),
                     "elapsed_seconds": time.time() - started, "updated_unix": time.time(),
                     "monitor_interval_seconds": 600, "alarm_enabled": False}
            status.write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps(state), flush=True)
            try:
                code = child.wait(timeout=600)
                break
            except subprocess.TimeoutExpired:
                pass
    state.update(status="complete" if code == 0 else "failed", exit_code=code,
                 elapsed_seconds=time.time() - started, updated_unix=time.time())
    comparison = OUT / "results/comparison.json"
    if code == 0 and comparison.exists():
        result = json.loads(comparison.read_text())
        state.update(decision=result["status"],
                     scores={name: values["candidate"]["score"] for name, values in result["cohorts"].items()})
    status.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
