# model208: 40-MOVIE confirmation (20 per embryo type, all with annotated divisions) of the propose settings.
# Predicts once (~30-40 min on the laptop GPU), then re-scores the same graphs with the gate on.
# Launch detached via WMI. Result: C:\biohub_data\work\model208\cmp2\gate_comparison.json
$ErrorActionPreference = "Continue"
$modelRoot = Split-Path -Parent $PSScriptRoot
Set-Location $modelRoot
$root = "C:\biohub_data\work\model208"
$work = "$root\cmp7_val40"
New-Item -ItemType Directory -Force $work | Out-Null
$log = "$root\compare7.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
$py = (Resolve-Path "..\Other\.venv-gpu\Scripts\python.exe").Path

$env:BIOHUB_COMP_DIR = (Resolve-Path "..\Data\competition").Path
$env:BIOHUB_MODEL_ARTIFACTS = (Resolve-Path "..\Data\public_checkpoints\support-pack").Path
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path "..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt").Path
$env:BIOHUB_SECONDARY_ARTIFACT_MANIFEST = (Resolve-Path "..\Data\public_checkpoints\secondary-seed\ARTIFACT_MANIFEST.json").Path
$env:OMP_NUM_THREADS = "8"
$env:BIOHUB_VALIDATOR_ENABLE = "1"
$env:BIOHUB_WORKING_DIR = $work
$env:BIOHUB_M208_DIR = (Resolve-Path "model208").Path
$env:BIOHUB_M208_WEIGHTS = (Resolve-Path "model207\weights").Path
$env:BIOHUB_M208_CROPS = "C:\biohub_data\work\model207\crops.npz"
$env:BIOHUB_M208_NOTEBOOK = "local_propose.ipynb"
$env:BIOHUB_M208_PLAN = "p40n30:propose:0.40:30,p30n50:propose:0.30:50"
$env:BIOHUB_M208_VAL_N = "20"
$env:BIOHUB_M208_PROPOSE_CAP = "3"

# wait for the 8-movie grid to release the GPU/CPU
while (-not ((Get-Content "$root\compare6.log" -ErrorAction SilentlyContinue) -match "end exit")) { Start-Sleep -Seconds 120 }
Note "start"
Set-Location $work
& $py (Join-Path $modelRoot "model208\compare_in_process.py") *> "$work\run.log"
Note "end exit=$LASTEXITCODE"
Set-Location $modelRoot
