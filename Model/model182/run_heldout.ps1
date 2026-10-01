# Waits for the fold0 training receipt, then predicts 24 held-out 44b6 movies with the
# EXACT 0.947 notebook predictor (clean checkpoint as both primary and secondary seed),
# then scores the production post-processing chain and its no-relink variant on them.
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:BIOHUB_COMP_DIR = (Resolve-Path ..\Data\competition).Path
$env:BIOHUB_WORKING_DIR = "C:\biohub_data\work\model167"
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path ..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt).Path
$env:OMP_NUM_THREADS = "3"
$runDir = "C:\biohub_data\work\model182\clean_80x125\model182_clean_fold0_seed20260914\split_0"
$receipt = "$runDir\training_receipt.json"
$ckpt = "$runDir\edge_predictor_best.pth"
$log = "C:\biohub_data\work\model182\heldout.log"
$stems = '44b6_d29c9ab2,44b6_3a861e03,44b6_d5e7d891,44b6_12dfb391,44b6_ddf577ad,44b6_668e0cc7,44b6_c96cfa10,44b6_8f5ab931,44b6_7e557709,44b6_7a302da0,44b6_d78e09d9,44b6_a2bb48bb,44b6_3bb3690f,44b6_8cc6506c,44b6_eb2880fc,44b6_949adeb1,44b6_996155de,44b6_587a1e22,44b6_8f9ecab4,44b6_c50204e0,44b6_415c0a3a,44b6_e28840c6,44b6_a21120c2,44b6_d2f34f90'
while (-not (Test-Path $receipt)) { Start-Sleep -Seconds 120 }
"$(Get-Date -Format o) receipt found; predicting held-out movies" | Out-File -Append -Encoding utf8 $log
if (-not (Test-Path "$runDir\config.json")) { Copy-Item ..\Data\public_checkpoints\support-pack\weights\unet_transformer\split_0\config.json "$runDir\config.json" }
& ..\Other\.venv-gpu\Scripts\python.exe model180\predict_stems.py --stems $stems --method unet_transformer_fold0clean --shard-tag fold0clean --weights $ckpt --env "BIOHUB_SECONDARY_WEIGHTS=$ckpt" *> C:\biohub_data\work\model182\predict_fold0clean.log
"$(Get-Date -Format o) predictor exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
$cfg = (Resolve-Path model180\configs\divgates.json).Path
& ..\Other\.venv-gpu\Scripts\python.exe model180\sweep_configs.py --pred-root C:\biohub_data\work\model167\tracking_repo\predictions\yodha\unet_transformer_fold0clean --stems $stems --configs $cfg --csv-out model182\results\sweep_heldout_fold0.csv *> C:\biohub_data\work\model182\sweep_fold0clean.log
"$(Get-Date -Format o) sweep exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
& ..\Other\.venv-gpu\Scripts\python.exe model180\edge_stage_diagnostic.py --pred-root C:\biohub_data\work\model167\tracking_repo\predictions\yodha\unet_transformer_fold0clean --stems $stems --csv-out model182\results\edge_stages_heldout_fold0.csv *> C:\biohub_data\work\model182\stages_fold0clean.log
"$(Get-Date -Format o) stages exit=$LASTEXITCODE; done" | Out-File -Append -Encoding utf8 $log
