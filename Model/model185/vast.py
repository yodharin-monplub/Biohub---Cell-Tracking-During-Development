#!/usr/bin/env python3
"""Vast.ai helper for model185 (user-authorized 2026-09-20 in chat: rental + data upload, cap $10).

Subcommands: offers | rent <offer_id> | status | stop | destroy | cost
State is kept in C:\\biohub_data\\work\\model185\\rental.json. The API key is read from .env and never printed.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(r"C:\biohub_data\work\model185\rental.json")
BASE = "https://console.vast.ai/api/v0/"
IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"
DISK_GB = 60
BUDGET_USD = 10.0


def key() -> str:
    for line in (ROOT.parent / "Other" / ".env").read_text().splitlines():
        if line.startswith("VAST_AI_API_KEY"):
            return line.split("=", 1)[1].split("#")[0].strip()
    raise RuntimeError("no VAST_AI_API_KEY in .env")


def call(method: str, path: str, payload: dict | None = None) -> dict:
    req = urllib.request.Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(),
                                 method=method, headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def offers():
    q = {"verified": {"eq": True}, "rentable": {"eq": True}, "num_gpus": {"eq": 1},
         "gpu_name": {"in": ["RTX 4090", "RTX 3090", "RTX 4080S", "RTX 3090 Ti"]}, "gpu_ram": {"gte": 23000},
         "cpu_ram": {"gte": 48000}, "disk_space": {"gte": DISK_GB + 20}, "reliability2": {"gte": 0.99},
         "inet_down": {"gte": 400}, "cuda_max_good": {"gte": 12.8}, "type": "on-demand",
         "order": [["dph_total", "asc"]], "limit": 12}
    for o in call("POST", "bundles/", q)["offers"]:
        print(o["id"], o["gpu_name"], f"${o['dph_total']:.3f}/h", f"down={o['inet_down']:.0f}", f"cpu={o['cpu_cores_effective']:.0f}",
              f"ram={o['cpu_ram'] / 1024:.0f}GB", o.get("geolocation"), f"rel={o['reliability2']:.3f}")


def rent(offer_id: int):
    if STATE.exists():
        raise SystemExit(f"rental state already exists: {STATE}")
    pub = (Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()
    body = {"image": IMAGE, "disk": DISK_GB, "runtype": "ssh_direct", "target_state": "running",
            "label": "biohub-model185", "cancel_unavail": True}
    try:
        r = call("PUT", f"asks/{offer_id}/", body)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"rent failed {exc.code}: {exc.read().decode()[:400]}")
    if not r.get("success"):
        raise SystemExit(f"rent failed: {r}")
    iid = int(r["new_contract"])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"instance_id": iid, "offer_id": offer_id, "rented_unix": time.time(), "budget_usd": BUDGET_USD}, indent=2))
    call("POST", f"instances/{iid}/ssh/", {"ssh_key": pub})
    print("rented instance", iid)


def instance() -> dict:
    iid = json.loads(STATE.read_text())["instance_id"]
    r = call("GET", f"instances/{iid}/")
    row = r.get("instances", r)
    return row if isinstance(row, dict) else next(x for x in row if int(x["id"]) == iid)


def status():
    row = instance()
    keep = ("id", "actual_status", "cur_state", "intended_status", "ssh_host", "ssh_port", "public_ipaddr", "ports",
            "gpu_name", "dph_total", "status_msg")
    out = {k: row.get(k) for k in keep}
    ports = row.get("ports") or {}
    direct = (ports.get("22/tcp") or [{}])[0].get("HostPort")
    out["direct_ssh"] = f"ssh -p {direct} root@{row.get('public_ipaddr')}" if direct else None
    st = json.loads(STATE.read_text())
    hours = (time.time() - st["rented_unix"]) / 3600
    out["hours"] = round(hours, 2)
    out["est_cost_usd"] = round(hours * float(row.get("dph_total") or 0), 2)
    print(json.dumps(out, indent=2))


def set_state(state: str):
    iid = json.loads(STATE.read_text())["instance_id"]
    print(call("PUT", f"instances/{iid}/", {"state": state}))


def destroy():
    iid = json.loads(STATE.read_text())["instance_id"]
    print(call("DELETE", f"instances/{iid}/"))
    STATE.rename(STATE.with_name(f"rental_destroyed_{iid}.json"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"offers": offers, "status": status, "stop": lambda: set_state("stopped"), "destroy": destroy}.get(cmd, lambda: rent(int(sys.argv[2])))()

