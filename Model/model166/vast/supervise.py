#!/usr/bin/env python3
"""Upload, train, retrieve, verify, and clean up the one model166 rental."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from client import call, sanitized_status, save_new


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rental = json.loads((HERE / "rental.json").read_text())
    instance_id = int(rental["instance_id"])
    status_path = HERE / "job_status.json"
    if status_path.exists():
        raise FileExistsError("Inspect the prior model166 job before restarting")
    started = time.time()
    record = {"instance_id": instance_id, "status": "starting",
              "monitor_interval_seconds": 600, "alarm_enabled": False}

    def update(**values: object) -> None:
        record.update(values, updated_unix=time.time(), elapsed_seconds=time.time() - started)
        status_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)

    def run(command: list[str], label: str, timeout_seconds: float | None = None) -> None:
        update(status="running", stage=label)
        log_path = HERE / f"{label}.log"
        with log_path.open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            stage_started = time.time()
            try:
                while True:
                    budget_remaining = float(rental["hard_stop_deadline_unix"]) - time.time()
                    stage_remaining = None if timeout_seconds is None else timeout_seconds - (time.time() - stage_started)
                    if budget_remaining <= 0:
                        raise TimeoutError("Rental cost deadline reached")
                    if stage_remaining is not None and stage_remaining <= 0:
                        raise TimeoutError(label + " timed out")
                    wait_for = min(600.0, budget_remaining,
                                   stage_remaining if stage_remaining is not None else 600.0)
                    try:
                        code = child.wait(timeout=wait_for)
                        break
                    except subprocess.TimeoutExpired:
                        update(status="running", stage=label)
                if code:
                    raise RuntimeError(f"{label} exited {code}; inspect {log_path}")
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise

    try:
        deadline = time.time() + 300
        while True:
            state = sanitized_status(instance_id)
            if state["actual_status"] == "running" and state.get("public_ipaddr") and state.get("ports"):
                break
            if time.time() >= deadline:
                raise TimeoutError("Vast instance did not expose SSH within five minutes")
            time.sleep(15)
        if state["id"] != instance_id or state["label"] != rental["label"]:
            raise RuntimeError("Rental identity mismatch")
        host = str(state["public_ipaddr"])
        port = int(state["ports"]["22/tcp"][0]["HostPort"])
        ssh = ["ssh", "-i", "/home/msi/.ssh/runpod_codex", "-p", str(port),
               "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=accept-new",
               "-o", f"UserKnownHostsFile={HERE / 'known_hosts'}"]
        target = "root@" + host
        save_new(HERE / "connection.json", {"instance_id": instance_id,
                 "host": host, "port": port, "ssh_identity": "/home/msi/.ssh/runpod_codex"})
        run(ssh + [target, "mkdir -p /workspace/biohub"], "connection", 240)
        sources = [
            "data/raw/train",
            "data/public/support-pack/repo",
            "data/public/temporal-seed/weights/unet_transformer/split_0/edge_predictor_best.pth",
            "model166/train_continuation.py", "model166/run_remote.sh",
            "model166/vast/bootstrap_remote.sh",
        ]
        run(["rsync", "-aR", "--partial", "--info=progress2", "-e", shlex.join(ssh),
             *sources, target + ":/workspace/biohub/"], "upload")
        run(ssh + [target, "bash /workspace/biohub/model166/vast/bootstrap_remote.sh"],
            "bootstrap", 3600)
        run(ssh + [target, "bash /workspace/biohub/model166/run_remote.sh"],
            "training", 29_000)
        destination = ROOT / "model166/retrieved_output"
        destination.mkdir(parents=True, exist_ok=False)
        run(["rsync", "-a", "--partial", "-e", shlex.join(ssh),
             target + ":/workspace/biohub/model166/output/", str(destination) + "/"],
            "download", 1800)
        receipts = list(destination.rglob("training_receipt.json"))
        if len(receipts) != 1:
            raise RuntimeError("Expected exactly one downloaded training receipt")
        receipt = json.loads(receipts[0].read_text())
        checkpoint = Path(receipt["checkpoint"])
        local_checkpoint = receipts[0].parent / checkpoint.name
        if (receipt.get("status") != "trained_not_scored"
                or digest(local_checkpoint) != receipt.get("checkpoint_sha256")
                or receipt.get("initial_checkpoint_sha256")
                != "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"):
            raise RuntimeError("Downloaded checkpoint receipt verification failed")
        save_new(HERE / "retrieval.json", {"status": "verified", "instance_id": instance_id,
                 "checkpoint_sha256": receipt["checkpoint_sha256"],
                 "local_checkpoint": str(local_checkpoint)})
        stopped = call("PUT", f"instances/{instance_id}/", {"state": "stopped"})
        if not stopped.get("success"):
            raise RuntimeError("Vast did not acknowledge stop after verified retrieval")
        destroyed = call("DELETE", f"instances/{instance_id}/")
        if not destroyed.get("success"):
            raise RuntimeError("Vast did not acknowledge cleanup after verified retrieval")
        save_new(HERE / "cleanup.json", {"status": "destroyed_after_verified_download",
                 "instance_id": instance_id, "updated_unix": time.time()})
        update(status="complete", stage="checkpoint_verified_rental_destroyed")
    except BaseException as error:
        update(status="failed", error_type=type(error).__name__, error=str(error))
        try:
            stopped = call("PUT", f"instances/{instance_id}/", {"state": "stopped"})
            update(stop_requested=bool(stopped.get("success")), disk_retained=True,
                   storage_charges_continue=True)
        finally:
            (HERE / "request_stop").touch(exist_ok=True)
        raise


if __name__ == "__main__":
    main()
