# model184 chain: convert synthetic -> (wait for GPU) -> pretrain 40 ep -> fine-tune fold0 80x125
# (warm start) -> predict the same 24 held-out 44b6 movies as model182 -> score base config.
# Compare with model182 from-scratch held-out base 0.79657.
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$work = "C:\biohub_data\work\model184"
$log = "$work\chain.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
$py = "..\Other\.venv-gpu\Scripts\python.exe"
$env:OMP_NUM_THREADS = "3"

if (-not (Test-Path "$work\synth\synthetic_splits.json")) {
  Note "converting synthetic sequences"
  & $py model184\convert_synthetic.py --src "$work\synthetic\sequences" --dst "$work\synth" --limit 400 --seed 0 --max-lineages 150 *> "$work\convert.log"
  Note "convert exit=$LASTEXITCODE"
}
# wait until the model182 fold1 chain has released the GPU (its predictor must have exited)
while (-not ((Get-Content C:\biohub_data\work\model182\heldout_fold1.log -ErrorAction SilentlyContinue) -match 'predictor exit')) { Start-Sleep -Seconds 180 }

$env:BIOHUB_RUNS_RESUMED = "1"
$pre = "$work\runs\model184_pretrain_syn400\split_0\edge_predictor_best.pth"
if (-not (Test-Path "$work\runs\model184_pretrain_syn400\split_0\training_receipt.json")) {
  Note "pretraining"
  & $py model184\train_stage.py --stage pretrain --data-dir "$work\synth" --splits "$work\synth\synthetic_splits.json" --method model184_pretrain_syn400 --epochs 40 --execute *> "$work\pretrain.log"
  Note "pretrain exit=$LASTEXITCODE"
}
$ftDir = "$work\runs\model184_ft_fold0\split_0"
if (-not (Test-Path "$ftDir\training_receipt.json")) {
  Note "fine-tuning fold0"
  & $py model184\train_stage.py --stage finetune --fold 0 --init $pre --method model184_ft_fold0 --epochs 80 --seed 20260914 --execute *> "$work\finetune_fold0.log"
  Note "finetune exit=$LASTEXITCODE"
}
$ckpt = "$ftDir\edge_predictor_best.pth"
if (-not (Test-Path "$ftDir\config.json")) { Copy-Item ..\Data\public_checkpoints\support-pack\weights\unet_transformer\split_0\config.json "$ftDir\config.json" }
$env:BIOHUB_COMP_DIR = (Resolve-Path ..\Data\competition).Path
$env:BIOHUB_WORKING_DIR = "C:\biohub_data\work\model167"
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path ..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt).Path
$stems = '44b6_d29c9ab2,44b6_3a861e03,44b6_d5e7d891,44b6_12dfb391,44b6_ddf577ad,44b6_668e0cc7,44b6_c96cfa10,44b6_8f5ab931,44b6_7e557709,44b6_7a302da0,44b6_d78e09d9,44b6_a2bb48bb,44b6_3bb3690f,44b6_8cc6506c,44b6_eb2880fc,44b6_949adeb1,44b6_996155de,44b6_587a1e22,44b6_8f9ecab4,44b6_c50204e0,44b6_415c0a3a,44b6_e28840c6,44b6_a21120c2,44b6_d2f34f90'
Note "predicting held-out movies with pretrained+finetuned fold0"
& $py model180\predict_stems.py --stems $stems --method unet_transformer_m184fold0 --shard-tag m184fold0 --weights $ckpt --env "BIOHUB_SECONDARY_WEIGHTS=$ckpt" *> "$work\predict_m184fold0.log"
Note "predictor exit=$LASTEXITCODE"
'{"base": {}}' | Set-Content -Encoding ascii model180\configs\base_only.json
New-Item -ItemType Directory -Force model184\results | Out-Null
& $py model180\sweep_configs.py --pred-root C:\biohub_data\work\model167\tracking_repo\predictions\yodha\unet_transformer_m184fold0 --stems $stems --configs (Resolve-Path model180\configs\base_only.json).Path --csv-out model184\results\heldout_fold0_pretrained.csv *> "$work\sweep_m184fold0.log"
Note "sweep exit=$LASTEXITCODE; chain done"
