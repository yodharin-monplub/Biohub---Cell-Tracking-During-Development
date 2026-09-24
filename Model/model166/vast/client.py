"""Minimal Vast.ai client; credentials remain local and are never printed."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.request

BASE = "https://console.vast.ai/api/v0/"
KEY_PATHS = (Path("/home/msi/.config/vastai/vast_api_key"),
             Path("/home/msi/.vast_api_key"))


def call(method: str, path: str, payload: dict | None = None) -> dict:
    key_path = next((p for p in KEY_PATHS if p.is_file() and p.stat().st_size), None)
    if key_path is None:
        raise FileNotFoundError("No local Vast.ai API-key file")
    key = key_path.read_text().strip()
    if not key:
        raise RuntimeError("Empty Vast.ai API key")
    request = urllib.request.Request(
        BASE + path,
        data=None if payload is None else json.dumps(payload).encode(),
        method=method,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def save_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def sanitized_status(instance_id: int) -> dict:
    response = call("GET", f"instances/{instance_id}/")
    row = response.get("instances", response)
    if isinstance(row, list):
        row = next(r for r in row if int(r["id"]) == instance_id)
    keys = ("id", "label", "actual_status", "cur_state", "intended_status",
            "ssh_host", "ssh_port", "public_ipaddr", "ports", "gpu_name",
            "num_gpus", "gpu_ram", "cpu_ram", "dph_total", "disk_space",
            "status_msg")
    return {key: row.get(key) for key in keys}
