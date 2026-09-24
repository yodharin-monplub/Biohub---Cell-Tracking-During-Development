# model199 sweep3: the gentle PAIR model196 (primary) + model198 (secondary) at 0.965, after sweep2. Based on sweep2.ps1;
# 4 visible test movies, after sweep.ps1 has finished (its last run is m193_0900). Diagnostic: does a gentler
# fine-tune keep the public model's recall on the hard movie 44b6_0b24845f (public 20,707 nodes, model193 14,758)?
# Launch detached via WMI.
$ErrorActionPreference = "Continue"
$modelRoot = Split-Path -Parent $PSScriptRoot
Set-Location $modelRoot
$root = "C:\biohub_data\work\model199"
$log = "$root\sweep3.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
while (-not (Test-Path "$root\m197_0965\summary.json")) { Start-Sleep -Seconds 120 }
Start-Sleep -Seconds 60
$py = (Resolve-Path "..\Other\.venv-gpu\Scripts\python.exe").Path
& $py model199\make_local_variant.py *>> $log
$env:BIOHUB_COMP_DIR = (Resolve-Path "..\Data\competition").Path
$env:BIOHUB_MODEL_ARTIFACTS = (Resolve-Path "..\Data\public_checkpoints\support-pack").Path
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path "..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt").Path
$env:BIOHUB_SECONDARY_ARTIFACT_MANIFEST = (Resolve-Path "..\Data\public_checkpoints\secondary-seed\ARTIFACT_MANIFEST.json").Path
$env:OMP_NUM_THREADS = "8"
$env:BIOHUB_VALIDATOR_ENABLE = "0"
$runs = @(
  @{ tag = "m196_m198_0965"; primary = (Resolve-Path "model196\weights\model196_ft_primary\edge_predictor_best.pth").Path; secondary = (Resolve-Path "model198\weights\model198_ft_secondary\edge_predictor_best.pth").Path; thr = "0.965" }
)
foreach ($r in $runs) {
  $work = "$root\$($r.tag)"
  if (Test-Path "$work\summary.json") { Note "$($r.tag) already done"; continue }
  New-Item -ItemType Directory -Force $work | Out-Null
  $env:BIOHUB_WORKING_DIR = $work
  $env:BIOHUB_M199_PRIMARY = $r.primary
  $env:BIOHUB_M199_DET_THRESHOLD = $r.thr
  $env:BIOHUB_M199_SECONDARY = $r.secondary
  Note "start $($r.tag)"
  Set-Location $work
  & $py (Join-Path $modelRoot "model199\run_variant.py") *> "$work\run.log"
  Set-Location $modelRoot
  Note "end $($r.tag) exit=$LASTEXITCODE"
}
Note "sweep3 finished"
