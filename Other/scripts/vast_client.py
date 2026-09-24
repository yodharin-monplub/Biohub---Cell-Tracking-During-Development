#!/usr/bin/env python3
"""Minimal Vast.ai client for this project (key from Other/.env; raw responses are never printed).

    python vast_client.py search [--max-dph 0.40]      cheap single-GPU 24 GB offers with fast network
    python vast_client.py rent <offer_id> [--disk 60]  create an instance (pytorch image), prints its id
    python vast_client.py status [instance_id]         instance state / ssh endpoint
    python vast_client.py destroy <instance_id>        destroy it (always do this when finished)
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / ".env"
IMAGE = "pytorch/pytorch:latest"
FIELDS = ["id", "label", "actual_status", "cur_state", "intended_status", "ssh_host", "ssh_port",
          "public_ipaddr", "gpu_name", "num_gpus", "gpu_ram", "cpu_ram", "dph_total", "disk_space",
          "inet_down", "inet_up", "status_msg"]


def key() -> str:
    for line in ENV.read_text().splitlines():
        if line.startswith("VAST_AI_API_KEY"):
            return line.split("=", 1)[1].split("#")[0].strip().strip('"')
    raise RuntimeError("VAST_AI_API_KEY missing from Other/.env")


def call(method: str, path: str, payload=None, params=None):
    url = "https://console.vast.ai/api/v0/" + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(), method=method,
        headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{exc.code} {exc.reason}: {exc.read()[:300].decode('utf-8', 'replace')}") from None


def search(max_dph: float):
    query = {"gpu_name": {"in": ["RTX 3090", "RTX 4090", "RTX A5000", "RTX 4080"]}, "num_gpus": {"eq": 1},
             "gpu_ram": {"gte": 20000}, "disk_space": {"gte": 60}, "inet_down": {"gte": 200},
             "inet_up": {"gte": 100}, "reliability2": {"gte": 0.97}, "rentable": {"eq": True},
             "cuda_max_good": {"gte": 12.4}, "dph_total": {"lte": max_dph},
             "type": "on-demand", "order": [["dph_total", "asc"]], "limit": 10}
    offers = call("GET", "bundles/", params={"q": json.dumps(query)})["offers"]
    for o in offers[:10]:
        print(f"{o['id']}  {o['gpu_name']:<10} ${o['dph_total']:.3f}/h  disk={o['disk_space']:.0f}GB  "
              f"down={o['inet_down']:.0f} up={o['inet_up']:.0f}  rel={o['reliability2']:.3f}  {o.get('geolocation')}")


def rent(offer_id: int, disk: int):
    body = {"client_id": "me", "image": IMAGE, "disk": disk, "runtype": "ssh_direc ssh_proxy", "label": "biohub-model202"}
    result = call("PUT", f"asks/{offer_id}/", payload=body)
    print(json.dumps({k: result.get(k) for k in ("success", "new_contract", "error")}))


def status(instance_id: int | None):
    if instance_id is None:
        rows = call("GET", "instances/")["instances"]
    else:
        response = call("GET", f"instances/{instance_id}/")
        rows = response.get("instances", response)
        rows = rows if isinstance(rows, list) else [rows]
    for row in rows:
        print(json.dumps({k: row.get(k) for k in FIELDS}))


def destroy(instance_id: int):
    print(json.dumps(call("DELETE", f"instances/{instance_id}/")))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("--max-dph", type=float, default=0.40)
    r = sub.add_parser("rent"); r.add_argument("offer_id", type=int); r.add_argument("--disk", type=int, default=60)
    t = sub.add_parser("status"); t.add_argument("instance_id", nargs="?", type=int)
    d = sub.add_parser("destroy"); d.add_argument("instance_id", type=int)
    args = ap.parse_args()
    if args.cmd == "search":
        search(args.max_dph)
    elif args.cmd == "rent":
        rent(args.offer_id, args.disk)
    elif args.cmd == "status":
        status(args.instance_id)
    else:
        destroy(args.instance_id)


if __name__ == "__main__":
    main()
