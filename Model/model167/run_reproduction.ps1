# Windows port of run_reproduction.sh: run the exact public 0.947 notebook locally.
# Uses the GPU venv (.venv-gpu) and the ..\Data\competition / data\public junctions.
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
# Working dir must be a SHORT path outside OneDrive: geff/zarr temp files under the
# project path exceed the 260-char Windows limit (FileNotFoundError on *.partial).
$workDir = if ($env:BIOHUB_WORKING_DIR) { $env:BIOHUB_WORKING_DIR } else { "C:\biohub_data\work\model167" }
New-Item -ItemType Directory -Force $workDir | Out-Null

if (-not $env:BIOHUB_COMP_DIR) { $env:BIOHUB_COMP_DIR = Join-Path $repoRoot "..\Data\competition" }
$env:BIOHUB_WORKING_DIR = $workDir
if (-not $env:BIOHUB_MODEL_ARTIFACTS) { $env:BIOHUB_MODEL_ARTIFACTS = Join-Path $repoRoot "..\Data\public_checkpoints\support-pack" }
if (-not $env:BIOHUB_DEEPCENTER_CHECKPOINT) { $env:BIOHUB_DEEPCENTER_CHECKPOINT = Join-Path $repoRoot "..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt" }
if (-not $env:BIOHUB_SECONDARY_ARTIFACT_MANIFEST) { $env:BIOHUB_SECONDARY_ARTIFACT_MANIFEST = Join-Path $repoRoot "..\Data\public_checkpoints\secondary-seed\ARTIFACT_MANIFEST.json" }
# Bound BLAS/OMP threads: the organizer scorer idles on unbounded thread pools (VALIDATION_AUDIT 2026-09-14).
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "8" }

$py = if ($env:BIOHUB_PYTHON_BIN) { $env:BIOHUB_PYTHON_BIN } else { Join-Path $repoRoot "..\Other\.venv-gpu\Scripts\python.exe" }
Set-Location $workDir
& $py (Join-Path $repoRoot "model167\prepare_reproduction.py")
& $py (Join-Path $repoRoot "model167\execute_reproduction.py")
