#!/usr/bin/env bash
set -euo pipefail

message="${1:-RunPod needs attention}"
notify-send -u critical -t 0 "Biohub RunPod" "${message}" 2>/dev/null || true
while true; do
  timeout 0.30 speaker-test -t sine -f 1000 >/dev/null 2>&1 || true
  sleep 4
done
