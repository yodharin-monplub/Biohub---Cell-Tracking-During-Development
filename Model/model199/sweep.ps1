# model199: local detection-threshold calibration on the 4 visible test movies (no Kaggle GPU).
# Runs the public reference first, then model193 (fine-tuned primary) at several thresholds, one after another,
# each in its own short working dir. Waits for model198 training to free the GPU. Launch detached via WMI.
$ErrorActionPreference = "Continue"
$modelRoot = Split-Path -Parent $PSScriptRoot
Set-Location $modelRoot
$root = "C:\biohub_data\work\model199"
New-Item -ItemType Directory -Force $root | Out-Null
$log = "$root\sweep.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
while (-not (Test-Path "model198\weights\model198_ft_secondary\training_receipt.json")) { Start-Sleep -Seconds 120 }
Start-Sleep -Seconds 60
$py = (Resolve-Path "..\Other\.venv-gpu\Scripts\python.exe").Path
& $py model199\make_local_variant.py *>> $log
$env:BIOHUB_COMP_DIR = (Resolve-Path "..\Data\competition").Path
$env:BIOHUB_MODEL_ARTIFACTS = (Resolve-Path "..\Data\public_checkpoints\support-pack").Path
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path "..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt").Path
$env:BIOHUB_SECONDARY_ARTIFACT_MANIFEST = (Resolve-Path "..\Data\public_checkpoints\secondary-seed\ARTIFACT_MANIFEST.json").Path
$env:OMP_NUM_THREADS = "8"
$env:BIOHUB_VALIDATOR_ENABLE = "0"
$ft = (Resolve-Path "model193\weights\model193_ft_primary\edge_predictor_best.pth").Path
$runs = @(
  @{ tag = "public_0965"; primary = ""; thr = "0.965" },
  @{ tag = "m193_0965"; primary = $ft; thr = "0.965" },
  @{ tag = "m193_0950"; primary = $ft; thr = "0.95" },
  @{ tag = "m193_0930"; primary = $ft; thr = "0.93" },
  @{ tag = "m193_0900"; primary = $ft; thr = "0.90" }
)
foreach ($r in $runs) {
  $work = "$root\$($r.tag)"
  if (Test-Path "$work\summary.json") { Note "$($r.tag) already done"; continue }
  New-Item -ItemType Directory -Force $work | Out-Null
  $env:BIOHUB_WORKING_DIR = $work
  $env:BIOHUB_M199_PRIMARY = $r.primary
  $env:BIOHUB_M199_DET_THRESHOLD = $r.thr
  Note "start $($r.tag)"
  Set-Location $work
  & $py (Join-Path $modelRoot "model199\run_variant.py") *> "$work\run.log"
  Set-Location $modelRoot
  Note "end $($r.tag) exit=$LASTEXITCODE"
}
Note "sweep finished"
