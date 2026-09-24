# model199 sweep5: model203 (half-gentle fine-tune) at 0.965 on the
# 4 visible test movies, after sweep.ps1 has finished (its last run is m193_0900). Diagnostic: does a gentler
# fine-tune keep the public model's recall on the hard movie 44b6_0b24845f (public 20,707 nodes, model193 14,758)?
# Launch detached via WMI.
$ErrorActionPreference = "Continue"
$modelRoot = Split-Path -Parent $PSScriptRoot
Set-Location $modelRoot
$root = "C:\biohub_data\work\model199"
$log = "$root\sweep5.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
while (-not (Test-Path "model203\weights\model203_ft_primary\training_receipt.json")) { Start-Sleep -Seconds 120 }
Start-Sleep -Seconds 60
$py = (Resolve-Path "..\Other\.venv-gpu\Scripts\python.exe").Path
$env:BIOHUB_COMP_DIR = (Resolve-Path "..\Data\competition").Path
$env:BIOHUB_MODEL_ARTIFACTS = (Resolve-Path "..\Data\public_checkpoints\support-pack").Path
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path "..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt").Path
$env:BIOHUB_SECONDARY_ARTIFACT_MANIFEST = (Resolve-Path "..\Data\public_checkpoints\secondary-seed\ARTIFACT_MANIFEST.json").Path
$env:OMP_NUM_THREADS = "8"
$env:BIOHUB_VALIDATOR_ENABLE = "0"
$runs = @(
  @{ tag = "m203_0965"; primary = (Resolve-Path "model203\weights\model203_ft_primary\edge_predictor_best.pth").Path; thr = "0.965" }
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
Note "sweep5 finished"
