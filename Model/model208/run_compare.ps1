# model208: compare the division gate against the unchanged pipeline on the validator movies, base score only.
# Reuses the predictions already computed in C:\biohub_data\work\model208\off (28 GPU-minutes), so each pass is
# just post-processing + scoring. Launch detached via WMI. Results: <root>\cmp_<mode>\m208_summary.json
$ErrorActionPreference = "Continue"
$modelRoot = Split-Path -Parent $PSScriptRoot
Set-Location $modelRoot
$root = "C:\biohub_data\work\model208"
$log = "$root\compare.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
$py = (Resolve-Path "..\Other\.venv-gpu\Scripts\python.exe").Path

$env:BIOHUB_COMP_DIR = (Resolve-Path "..\Data\competition").Path
$env:BIOHUB_MODEL_ARTIFACTS = (Resolve-Path "..\Data\public_checkpoints\support-pack").Path
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path "..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt").Path
$env:BIOHUB_SECONDARY_ARTIFACT_MANIFEST = (Resolve-Path "..\Data\public_checkpoints\secondary-seed\ARTIFACT_MANIFEST.json").Path
$env:OMP_NUM_THREADS = "8"
$env:BIOHUB_VALIDATOR_ENABLE = "1"
$env:BIOHUB_M208_BASE_ONLY = "1"
$env:BIOHUB_M208_DIR = (Resolve-Path "model208").Path
$env:BIOHUB_M208_WEIGHTS = (Resolve-Path "model207\weights").Path
$env:BIOHUB_M208_CROPS = "C:\biohub_data\work\model207\crops.npz"

foreach ($mode in @("off", "veto")) {
  $work = "$root\cmp_$mode"
  if (Test-Path "$work\m208_summary.json") { Note "$mode already done"; continue }
  New-Item -ItemType Directory -Force $work | Out-Null
  Note "$mode : copying cached predictions"
  robocopy "$root\off\tracking_repo" "$work\tracking_repo" /E /NFL /NDL /NJH /NJS /NP | Out-Null
  robocopy "$root\off\secondary_seed_weights" "$work\secondary_seed_weights" /E /NFL /NDL /NJH /NJS /NP | Out-Null
  $env:BIOHUB_WORKING_DIR = $work
  $env:BIOHUB_M208_MODE = $mode
  Note "start $mode"
  Set-Location $work
  & $py (Join-Path $modelRoot "model208\run_nb.py") (Join-Path $modelRoot "model208\local_$mode.ipynb") *> "$work\run.log"
  Set-Location $modelRoot
  Note "end $mode exit=$LASTEXITCODE"
  $proxy = (Get-Content "$work\run.log" -Raw) -split "`r|`n" | Where-Object { $_ -match 'VALIDATOR base proxy' } | Select-Object -Last 1
  Note "$mode : $proxy"
}
Note "comparison finished"
