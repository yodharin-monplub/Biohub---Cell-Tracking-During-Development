# model189 follow-up: long-trained seed as PRIMARY + short seed777 (model186) as SECONDARY on the same 24
# held-out 44b6 movies. Mirrors "model185 cloud checkpoint as primary + public 50-epoch secondary" on Kaggle.
# Compare with long seed as both (0.81615) and two short seeds (0.80606).
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$work = "C:\biohub_data\work\model189"
$log = "$work\chain_mixed.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
$py = "..\Other\.venv-gpu\Scripts\python.exe"
$env:OMP_NUM_THREADS = "3"
Remove-Item Env:BIOHUB_TERTIARY_WEIGHTS -ErrorAction SilentlyContinue
$long = (Resolve-Path model189\weights\fold0_86x250\edge_predictor_best.pth).Path
$short = (Resolve-Path model186\weights\fold0_seed777\edge_predictor_best.pth).Path
$env:BIOHUB_COMP_DIR = (Resolve-Path ..\Data\competition).Path
$env:BIOHUB_WORKING_DIR = "C:\biohub_data\work\model167"
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path ..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt).Path
$stems = '44b6_d29c9ab2,44b6_3a861e03,44b6_d5e7d891,44b6_12dfb391,44b6_ddf577ad,44b6_668e0cc7,44b6_c96cfa10,44b6_8f5ab931,44b6_7e557709,44b6_7a302da0,44b6_d78e09d9,44b6_a2bb48bb,44b6_3bb3690f,44b6_8cc6506c,44b6_eb2880fc,44b6_949adeb1,44b6_996155de,44b6_587a1e22,44b6_8f9ecab4,44b6_c50204e0,44b6_415c0a3a,44b6_e28840c6,44b6_a21120c2,44b6_d2f34f90'
Note "predicting: long primary + short secondary"
& $py model180\predict_stems.py --stems $stems --method unet_transformer_m189mixed --shard-tag m189mixed --weights $long --env "BIOHUB_SECONDARY_WEIGHTS=$short" *> "$work\predict_mixed.log"
Note "predictor exit=$LASTEXITCODE"
& $py model180\sweep_configs.py --pred-root C:\biohub_data\work\model167\tracking_repo\predictions\yodha\unet_transformer_m189mixed --stems $stems --configs (Resolve-Path model180\configs\base_only.json).Path --csv-out model189\results\heldout_fold0_long_plus_short.csv *> "$work\sweep_mixed.log"
Note "sweep exit=$LASTEXITCODE; chain done"
