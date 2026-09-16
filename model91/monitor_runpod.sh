#!/usr/bin/env bash
set -u

workspace="/home/msi/resources/business folder/Coding/Kaggle/Biohub - Cell Tracking During Development"
host="root@213.181.111.2"
port="36101"
interval_seconds="${BIOHUB_MONITOR_INTERVAL_SECONDS:-600}"
alarm_enabled="${BIOHUB_MONITOR_ALARM_ENABLED:-0}"
log_path="${workspace}/model91/monitor.log"
alarm_pid_path="${workspace}/model91/alarm.pid"
echo "$$" > "${workspace}/model91/monitor.pid"

start_alarm() {
  local message="$1"
  if [[ "${alarm_enabled}" != "1" ]]; then
    printf '%s alarm_suppressed=%s\n' "$(date --iso-8601=seconds)" "${message}" >> "${log_path}"
    return
  fi
  if [[ -f "${alarm_pid_path}" ]]; then
    local old_pid
    old_pid="$(cat "${alarm_pid_path}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      return
    fi
  fi
  nohup bash "${workspace}/model77/local_beep_alarm.sh" "${message}" \
    > /tmp/model91_beep.log 2>&1 < /dev/null &
  echo "$!" > "${alarm_pid_path}"
}

while true; do
  timestamp="$(date --iso-8601=seconds)"
  remote="$({
    ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
      -p "${port}" "${host}" \
      'cd /workspace/biohub; if [[ -f model91/cloud_run/repaired_comparison.json ]]; then state=complete; elif pgrep -f "[m]odel91/run_cloud.sh" >/dev/null; then state=running; else state=stopped; fi; oof=$(find model91/cloud_run/tracking_repo/predictions -path "*/unet_transformer_val/split_0/*.geff" -type d 2>/dev/null | wc -l); cell=$(grep "EXECUTING NOTEBOOK CELL" model91/launcher.log 2>/dev/null | tail -n 1); printf "state=%s oof=%s cell=%s\n" "$state" "$oof" "$cell"'
  } 2>&1)"
  ssh_status=$?
  printf '%s ssh_status=%s %s\n' "${timestamp}" "${ssh_status}" "${remote}" >> "${log_path}"

  if [[ ${ssh_status} -eq 0 && "${remote}" == state=complete* ]]; then
    scp -F /dev/null -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
      -P "${port}" \
      "${host}:/workspace/biohub/model91/cloud_run/raw_official_score.json" \
      "${host}:/workspace/biohub/model91/cloud_run/repaired_comparison.json" \
      "${host}:/workspace/biohub/model91/cloud_run/submission_validation.json" \
      "${host}:/workspace/biohub/model91/cloud_run/submission.sha256" \
      "${host}:/workspace/biohub/model91/cloud_run/validator_results.csv" \
      "${host}:/workspace/biohub/model91/cloud_run/validator_results.sha256" \
      "${workspace}/model91/results/" >> "${log_path}" 2>&1
    copy_status=$?
    printf '%s receipt_copy_status=%s\n' "$(date --iso-8601=seconds)" "${copy_status}" >> "${log_path}"
    if [[ ${copy_status} -eq 0 ]]; then
      start_alarm "Model91 finished and receipts are local. The RunPod can be stopped after review."
    else
      start_alarm "Model91 finished, but receipt copy needs attention."
    fi
    exit 0
  fi

  if [[ ${ssh_status} -eq 0 && "${remote}" == state=stopped* ]]; then
    start_alarm "Model91 stopped without a final comparison. RunPod needs attention."
    exit 1
  fi

  sleep "${interval_seconds}"
done
