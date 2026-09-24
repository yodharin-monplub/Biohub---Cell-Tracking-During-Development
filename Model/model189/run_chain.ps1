# model189 chain: does a larger training budget help on an UNSEEN embryo?
# Train fold0 (6bba only) from random init with 4x the model182 budget (160 epochs x 250 iterations, about two
# passes over the 128 training movies per epoch: 40k iterations vs 80 x 125 = 10k), same seed 20260914, then predict the same 24 held-out 44b6 movies with the exact
# 0.947 predictor (single seed: primary = secondary) and score the base post-processing config.
# Compare with model182 one-seed held-out base 0.79657.
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$work = "C:\biohub_data\work\model189"
New-Item -ItemType Directory -Force $work | Out-Null
$log = "$work\chain.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
$py = "..\Other\.venv-gpu\Scripts\python.exe"
$env:OMP_NUM_THREADS = "3"
$env:BIOHUB_RUNS_RESUMED = "1"
Remove-Item Env:BIOHUB_TERTIARY_WEIGHTS -ErrorAction SilentlyContinue

$ftDir = "$work\runs\model189_fold0_160x250\split_0"
if (-not (Test-Path "$ftDir\training_receipt.json")) {
  Note "training fold0 160x250"
  & $py model184\train_stage.py --stage finetune --fold 0 --method model189_fold0_160x250 --output-root "$work\runs" --epochs 160 --max-iters 250 --seed 20260914 --max-wall-seconds 86400 --execute *> "$work\train_fold0.log"
  Note "train exit=$LASTEXITCODE"
}
$ckpt = "$ftDir\edge_predictor_best.pth"
if (-not (Test-Path $ckpt)) { Note "no checkpoint; stopping"; exit 1 }
if (-not (Test-Path "$ftDir\config.json")) { Copy-Item ..\Data\public_checkpoints\support-pack\weights\unet_transformer\split_0\config.json "$ftDir\config.json" }
New-Item -ItemType Directory -Force model189\weights\fold0_160x250 | Out-Null
Copy-Item "$ftDir\edge_predictor_best.pth", "$ftDir\config.json", "$ftDir\training_receipt.json" model189\weights\fold0_160x250\ -Force

$env:BIOHUB_COMP_DIR = (Resolve-Path ..\Data\competition).Path
$env:BIOHUB_WORKING_DIR = "C:\biohub_data\work\model167"
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path ..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt).Path
$stems = '44b6_d29c9ab2,44b6_3a861e03,44b6_d5e7d891,44b6_12dfb391,44b6_ddf577ad,44b6_668e0cc7,44b6_c96cfa10,44b6_8f5ab931,44b6_7e557709,44b6_7a302da0,44b6_d78e09d9,44b6_a2bb48bb,44b6_3bb3690f,44b6_8cc6506c,44b6_eb2880fc,44b6_949adeb1,44b6_996155de,44b6_587a1e22,44b6_8f9ecab4,44b6_c50204e0,44b6_415c0a3a,44b6_e28840c6,44b6_a21120c2,44b6_d2f34f90'
Note "predicting held-out movies"
& $py model180\predict_stems.py --stems $stems --method unet_transformer_m189fold0 --shard-tag m189fold0 --weights $ckpt --env "BIOHUB_SECONDARY_WEIGHTS=$ckpt" *> "$work\predict.log"
Note "predictor exit=$LASTEXITCODE"
New-Item -ItemType Directory -Force model189\results | Out-Null
& $py model180\sweep_configs.py --pred-root C:\biohub_data\work\model167\tracking_repo\predictions\yodha\unet_transformer_m189fold0 --stems $stems --configs (Resolve-Path model180\configs\base_only.json).Path --csv-out model189\results\heldout_fold0_160x250.csv *> "$work\sweep.log"
Note "sweep exit=$LASTEXITCODE; chain done"
