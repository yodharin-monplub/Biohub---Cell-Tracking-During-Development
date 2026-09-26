# model187 honest test: primary = fold0 seed1, secondary = ensemble(seed2, seed3) via BIOHUB_TERTIARY_WEIGHTS.
# Same 24 held-out 44b6 movies and base post-processing as model182/186. Compare with 0.80606 (two seeds).
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:BIOHUB_COMP_DIR = (Resolve-Path ..\Data\competition).Path
$env:BIOHUB_WORKING_DIR = "C:\biohub_data\work\model167"
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path ..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt).Path
$env:OMP_NUM_THREADS = "3"
$seed1 = "C:\biohub_data\work\model182\clean_80x125\model182_clean_fold0_seed20260914\split_0\edge_predictor_best.pth"
$seed2 = "C:\biohub_data\work\model186\runs\model186_fold0_seed2\split_0\edge_predictor_best.pth"
$dir3 = "C:\biohub_data\work\model186\runs\model186_fold0_seed3\split_0"
$log = "C:\biohub_data\work\model187\chain.log"
New-Item -ItemType Directory -Force C:\biohub_data\work\model187, model187\results | Out-Null
$stems = '44b6_d29c9ab2,44b6_3a861e03,44b6_d5e7d891,44b6_12dfb391,44b6_ddf577ad,44b6_668e0cc7,44b6_c96cfa10,44b6_8f5ab931,44b6_7e557709,44b6_7a302da0,44b6_d78e09d9,44b6_a2bb48bb,44b6_3bb3690f,44b6_8cc6506c,44b6_eb2880fc,44b6_949adeb1,44b6_996155de,44b6_587a1e22,44b6_8f9ecab4,44b6_c50204e0,44b6_415c0a3a,44b6_e28840c6,44b6_a21120c2,44b6_d2f34f90'
while (-not (Test-Path "$dir3\training_receipt.json")) { Start-Sleep -Seconds 180 }
"$(Get-Date -Format o) seed3 ready; predicting triple-seed" | Out-File -Append -Encoding utf8 $log
if (-not (Test-Path "$dir3\config.json")) { Copy-Item ..\Data\public_checkpoints\support-pack\weights\unet_transformer\split_0\config.json "$dir3\config.json" }
& ..\Other\.venv-gpu\Scripts\python.exe model180\predict_stems.py --stems $stems --method unet_transformer_m187triple --shard-tag m187triple --weights $seed1 --env "BIOHUB_SECONDARY_WEIGHTS=$seed2" --env "BIOHUB_TERTIARY_WEIGHTS=$dir3\edge_predictor_best.pth" *> C:\biohub_data\work\model187\predict.log
"$(Get-Date -Format o) predictor exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
& ..\Other\.venv-gpu\Scripts\python.exe model180\sweep_configs.py --pred-root C:\biohub_data\work\model167\tracking_repo\predictions\yodha\unet_transformer_m187triple --stems $stems --configs (Resolve-Path model180\configs\base_only.json).Path --csv-out model187\results\heldout_fold0_tripleseed.csv *> C:\biohub_data\work\model187\sweep.log
"$(Get-Date -Format o) sweep exit=$LASTEXITCODE; done" | Out-File -Append -Encoding utf8 $log
