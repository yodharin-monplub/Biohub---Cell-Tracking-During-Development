# model208: score the division gate against ground truth on the notebook's own held-out validator movies.
# Pass 1 (mode off) also builds the prediction cache; later passes reuse it, so only the scoring differs.
# Launch detached via WMI. Results: C:\biohub_data\work\model208\<mode>\m208_summary.json
$ErrorActionPreference = "Continue"
$modelRoot = Split-Path -Parent $PSScriptRoot
Set-Location $modelRoot
$root = "C:\biohub_data\work\model208"
New-Item -ItemType Directory -Force $root | Out-Null
$log = "$root\run.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
$py = (Resolve-Path "..\Other\.venv-gpu\Scripts\python.exe").Path

$env:BIOHUB_COMP_DIR = (Resolve-Path "..\Data\competition").Path
$env:BIOHUB_MODEL_ARTIFACTS = (Resolve-Path "..\Data\public_checkpoints\support-pack").Path
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path "..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt").Path
$env:BIOHUB_SECONDARY_ARTIFACT_MANIFEST = (Resolve-Path "..\Data\public_checkpoints\secondary-seed\ARTIFACT_MANIFEST.json").Path
$env:OMP_NUM_THREADS = "8"
$env:BIOHUB_VALIDATOR_ENABLE = "1"
$env:BIOHUB_M208_DIR = (Resolve-Path "model208").Path
$env:BIOHUB_M208_WEIGHTS = (Resolve-Path "model207\weights").Path
$env:BIOHUB_M208_CROPS = "C:\biohub_data\work\model207\crops.npz"

foreach ($mode in @("off", "veto")) {
  $work = "$root\$mode"
  if (Test-Path "$work\m208_summary.json") { Note "$mode already done"; continue }
  New-Item -ItemType Directory -Force $work | Out-Null
  # share the expensive prediction cache across modes: only post-processing differs
  if ($mode -ne "off" -and (Test-Path "$root\off\tracking_repo")) {
    Note "$mode : reusing predictions from the off pass"
    robocopy "$root\off" $work /E /XF submission.csv m208_summary.json ppsweep_selected.json /NFL /NDL /NJH /NJS /NP | Out-Null
    Remove-Item "$work\biohub_live_resume\validator_base_state.json", "$work\biohub_live_resume\ppsweep_state.json" -ErrorAction SilentlyContinue
  }
  $env:BIOHUB_WORKING_DIR = $work
  $env:BIOHUB_M208_MODE = $mode
  Note "start $mode"
  Set-Location $work
  & $py (Join-Path $modelRoot "model208\run_nb.py") (Join-Path $modelRoot "model208\local_$mode.ipynb") *> "$work\run.log"
  Set-Location $modelRoot
  Note "end $mode exit=$LASTEXITCODE"
}
Note "model208 local evaluation finished"
