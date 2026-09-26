# model208: ONE 40-movie run scoring all six division variants with the OFFICIAL metric (replaces run_off40_a/b/c, which predicted the same movies three times; they died when the project moved on 2026-09-25).
# Predicts once (~30-40 min on the laptop GPU), then re-scores the same graphs with the gate on.
# Launch detached via WMI. Result: C:\biohub_data\work\model208\cmp2\gate_comparison.json
$ErrorActionPreference = "Continue"
$modelRoot = Split-Path -Parent $PSScriptRoot
Set-Location $modelRoot
$root = "C:\biohub_data\work\model208"
$work = "$root\cmp10all_val40"
New-Item -ItemType Directory -Force $work | Out-Null
$log = "$root\compare10all.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
$py = (Resolve-Path "..\Other\.venv-gpu\Scripts\python.exe").Path

$env:BIOHUB_COMP_DIR = (Resolve-Path "..\Data\competition").Path
$env:BIOHUB_MODEL_ARTIFACTS = (Resolve-Path "..\Data\public_checkpoints\support-pack").Path
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path "..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt").Path
$env:BIOHUB_SECONDARY_ARTIFACT_MANIFEST = (Resolve-Path "..\Data\public_checkpoints\secondary-seed\ARTIFACT_MANIFEST.json").Path
$env:OMP_NUM_THREADS = "8"
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = "1"  # Intel Fortran runtime (MKL/SciPy) otherwise aborts on a console CLOSE event (all runs died that way 2026-09-25 04:39)
$env:BIOHUB_VALIDATOR_ENABLE = "1"
$env:BIOHUB_WORKING_DIR = $work
$env:BIOHUB_M208_DIR = (Resolve-Path "model208").Path
$env:BIOHUB_M208_WEIGHTS = (Resolve-Path "model207\weights").Path
$env:BIOHUB_M208_CROPS = "C:\biohub_data\work\model207\crops.npz"
$env:BIOHUB_M208_NOTEBOOK = "local_propose.ipynb"
$env:BIOHUB_M208_PLAN = "off:off:,p40n30:propose:0.40:30,nosafe:nosafe:,veto40:veto:0.40,veto15:veto:0.15,p60n30:propose:0.60:30"
$env:BIOHUB_M208_VAL_N = "20"
$env:BIOHUB_M208_PROPOSE_CAP = "3"

# wait for the 8-movie grid to release the GPU/CPU

Note "start"
Set-Location $work
& $py (Join-Path $modelRoot "model208\compare_in_process.py") *> "$work\run.log"
Note "end exit=$LASTEXITCODE"
Set-Location $modelRoot
