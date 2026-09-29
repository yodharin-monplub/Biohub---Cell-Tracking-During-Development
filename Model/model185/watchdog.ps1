# Cost guard for the model185 Vast.ai rental: stops the instance when estimated spend reaches $8.50
# (user cap $10). Checks every 10 minutes. Does not delete data; destroy is a separate manual step.
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$log = "C:\biohub_data\work\model185\watchdog.log"
while (Test-Path C:\biohub_data\work\model185\rental.json) {
  try {
    $s = & ..\Other\.venv\Scripts\python.exe model185\vast.py status | ConvertFrom-Json
    "$(Get-Date -Format o) status=$($s.actual_status) hours=$($s.hours) cost=$($s.est_cost_usd)" | Out-File -Append -Encoding utf8 $log
    if ([double]$s.est_cost_usd -ge 8.5) {
      "$(Get-Date -Format o) COST LIMIT: stopping instance" | Out-File -Append -Encoding utf8 $log
      & ..\Other\.venv\Scripts\python.exe model185\vast.py stop | Out-File -Append -Encoding utf8 $log
      break
    }
  } catch { "$(Get-Date -Format o) watchdog error: $($_.Exception.Message)" | Out-File -Append -Encoding utf8 $log }
  Start-Sleep -Seconds 600
}
