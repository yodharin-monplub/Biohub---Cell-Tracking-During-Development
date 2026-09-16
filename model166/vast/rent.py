#!/usr/bin/env python3
"""Plan or explicitly rent one bounded Vast.ai model166 training instance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from client import call, save_new

IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"
LABEL = "biohub-model166-training"
DISK_GB = 150
MAX_DPH = 0.65
MAX_COST = 9.0
GPU_NAMES = ("RTX 4080S", "RTX 4090", "RTX 3090 Ti", "RTX 3090")
# This provider accepted the prior contract but the user could not start it.
# Never immediately re-rent the same unusable machine as a "replacement".
EXCLUDED_OFFER_IDS = frozenset({48470912})


def query() -> list[dict]:
    body = {
        "verified": {"eq": True}, "rentable": {"eq": True},
        "rented": {"eq": False}, "num_gpus": {"eq": 1},
        "gpu_name": {"in": list(GPU_NAMES)}, "gpu_ram": {"gte": 24000},
        "cpu_ram": {"gte": 60000}, "disk_space": {"gte": 160},
        "disk_bw": {"gte": 1000}, "inet_down": {"gte": 200},
        "inet_up": {"gte": 200}, "cuda_max_good": {"gte": 12.8},
        "direct_port_count": {"gte": 1}, "reliability2": {"gte": 0.99},
        "dph_total": {"lte": MAX_DPH}, "type": "ondemand",
        "allocated_storage": DISK_GB, "limit": 30,
    }
    return call("POST", "bundles/", body).get("offers", [])


def public_offer(row: dict) -> dict:
    keys = ("id", "gpu_name", "gpu_ram", "num_gpus", "cpu_ram",
            "cpu_cores_effective", "disk_space", "disk_bw", "inet_down",
            "inet_up", "reliability", "reliability2", "cuda_max_good",
            "geolocation", "dph_total", "dlperf", "dlperf_per_dphtotal",
            "storage_cost", "inet_up_cost", "inet_down_cost")
    return {key: row.get(key) for key in keys}


def choose(offers: list[dict]) -> dict:
    offers = [row for row in offers if int(row["id"]) not in EXCLUDED_OFFER_IDS]
    if not offers:
        raise RuntimeError("No qualifying Vast.ai offer; nothing rented")
    # Fast/value compromise. Prefer a 4080S or4090, then DLPerf/$, while
    # retaining3090-class fallback if the faster offers disappear.
    tier = {"RTX 4080S": 0, "RTX 4090": 0, "RTX 3090 Ti": 1, "RTX 3090": 2}
    return min(offers, key=lambda row: (
        tier.get(row.get("gpu_name"), 9),
        -float(row.get("dlperf_per_dphtotal") or 0),
        float(row["dph_total"])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="Actually create a billed instance; default is read-only")
    args = ap.parse_args()
    rental_path = HERE / "rental.json"
    if rental_path.exists():
        raise FileExistsError("Rental already recorded; never create a duplicate")
    offers = query()
    offer = choose(offers)
    plan = public_offer(offer)
    plan.update(image=IMAGE, allocated_disk_gb=DISK_GB,
                maximum_cost_usd=MAX_COST, execute=args.execute)
    if not args.execute:
        print(json.dumps({"status": "dry_run_no_rental", "selected": plan}, indent=2))
        return
    # Confirm image tag exists before accepting a billed offer.
    tag = IMAGE.split(":", 1)[1]
    url = "https://hub.docker.com/v2/repositories/pytorch/pytorch/tags/" + tag
    with urllib.request.urlopen(url, timeout=30) as response:
        if json.load(response).get("name") != tag:
            raise RuntimeError("Requested public image tag was not verified")
    existing = call("GET", "instances/").get("instances", [])
    if any(row.get("label") == LABEL for row in existing):
        raise RuntimeError("A model166 instance already exists")
    if float(offer["dph_total"]) > MAX_DPH:
        raise RuntimeError("Selected offer exceeds hourly cap")
    payload = {"image": IMAGE, "disk": DISK_GB, "runtype": "ssh_direct",
               "target_state": "running", "label": LABEL,
               "cancel_unavail": True}
    started = time.time()
    result = call("PUT", f"asks/{int(offer['id'])}/", payload)
    if not result.get("success") or not isinstance(result.get("new_contract"), int):
        raise RuntimeError("Vast.ai did not confirm instance creation")
    instance_id = result["new_contract"]
    # Never persist or print the returned per-instance API key.
    rate = float(offer["dph_total"])
    receipt = {"instance_id": instance_id, "offer": public_offer(offer),
               "label": LABEL, "created_unix": started,
               "hard_stop_deadline_unix": started + MAX_COST / rate * 3600,
               "maximum_cost_usd": MAX_COST, "disk_gb": DISK_GB,
               "image": IMAGE, "mode": "next_model_training_no_kaggle"}
    save_new(rental_path, receipt)
    print(json.dumps(receipt), flush=True)
    public_key = Path("/home/msi/.ssh/runpod_codex.pub").read_text().strip()
    if not public_key.startswith(("ssh-ed25519 ", "ssh-rsa ")):
        call("PUT", f"instances/{instance_id}/", {"state": "stopped"})
        raise RuntimeError("Local SSH public key invalid; instance stop requested")
    attached = call("POST", f"instances/{instance_id}/ssh", {"ssh_key": public_key})
    if not attached.get("success") and attached.get("msg") != "SSH key already associated with instance.":
        call("PUT", f"instances/{instance_id}/", {"state": "stopped"})
        raise RuntimeError("SSH attachment failed; instance stop requested")
    print("SSH public key attached; no private key or account credential uploaded.")


if __name__ == "__main__":
    main()
