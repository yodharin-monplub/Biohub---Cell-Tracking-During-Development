# Reciprocal testbed: waits for the fold1 checkpoint (trained on 71 x 44b6), predicts 16
# held-out 6bba movies with the exact 0.947 predictor (clean checkpoint as both seeds),
# then scores base / no_relink / no_linefit on them.
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:BIOHUB_COMP_DIR = (Resolve-Path ..\Data\competition).Path
$env:BIOHUB_WORKING_DIR = "C:\biohub_data\work\model167"
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path ..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt).Path
$env:OMP_NUM_THREADS = "3"
$runDir = "C:\biohub_data\work\model182\clean_80x125\model182_clean_fold1_seed20260914\split_1"
$ckpt = "$runDir\edge_predictor_best.pth"
$log = "C:\biohub_data\work\model182\heldout_fold1.log"
# 16 movies: mix of small/medium 6bba movies with many annotated edges and divisions
$stems = '6bba_48816121,6bba_afb141ff,6bba_cdcfe533,6bba_debd7bfa,6bba_df673a83,6bba_12665c0e,6bba_20852818,6bba_4ffd3da3,6bba_5c039895,6bba_7f87b3d8,6bba_969618f6,6bba_bb9f20c3,6bba_337b1b3a,6bba_062c8d37,6bba_085bf656,6bba_09961292'
while (-not (Test-Path "$runDir\training_receipt.json")) { Start-Sleep -Seconds 120 }
"$(Get-Date -Format o) receipt found; predicting held-out 6bba movies" | Out-File -Append -Encoding utf8 $log
if (-not (Test-Path "$runDir\config.json")) { Copy-Item ..\Data\public_checkpoints\support-pack\weights\unet_transformer\split_0\config.json "$runDir\config.json" }
& ..\Other\.venv-gpu\Scripts\python.exe model180\predict_stems.py --stems $stems --method unet_transformer_fold1clean --shard-tag fold1clean --weights $ckpt --env "BIOHUB_SECONDARY_WEIGHTS=$ckpt" *> C:\biohub_data\work\model182\predict_fold1clean.log
"$(Get-Date -Format o) predictor exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
$cfg = (Resolve-Path model180\configs\core4.json).Path
& ..\Other\.venv-gpu\Scripts\python.exe model180\sweep_configs.py --pred-root C:\biohub_data\work\model167\tracking_repo\predictions\yodha\unet_transformer_fold1clean --stems $stems --configs $cfg --csv-out model182\results\sweep_heldout_fold1.csv *> C:\biohub_data\work\model182\sweep_fold1clean.log
"$(Get-Date -Format o) sweep exit=$LASTEXITCODE; done" | Out-File -Append -Encoding utf8 $log
