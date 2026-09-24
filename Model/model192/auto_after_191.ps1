# Waits for model191 to finish (its training_receipt.json is copied into Model\model191\weights), then runs
# prepare_and_push.ps1 once. Launch detached (WMI) so it survives Claude session restarts. Log:
# C:\biohub_data\work\model192\auto.log
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$log = 'C:\biohub_data\work\model192\auto.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
if (Test-Path 'C:\biohub_data\work\model192\pushed.flag') { Note 'already pushed; exiting'; exit }
Note 'waiting for model191'
while (-not (Test-Path 'Model\model191\weights\model191_scratch_all\training_receipt.json')) { Start-Sleep -Seconds 120 }
Start-Sleep -Seconds 30
Note 'model191 finished; running prepare_and_push.ps1'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File Model\model192\prepare_and_push.ps1 *>> $log
Note "prepare_and_push exit=$LASTEXITCODE"
if ($LASTEXITCODE -eq 0) { Get-Date -Format o | Out-File -Encoding ascii 'C:\biohub_data\work\model192\pushed.flag' }
