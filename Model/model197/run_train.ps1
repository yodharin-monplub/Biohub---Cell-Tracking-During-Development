# model197: CONTINUE model193 (itself the public primary fine-tuned 24 ep at lr 2e-5) for 24 more epochs at lr 2e-5. Waits for model196 to free the GPU. (support-pack split_0, sha256 12f6881e...) on all 199
# train movies at low lr (2e-5), 24 epochs x 250 it x batch 2. Warm start keeps the public model's calibration
# (model190 showed our from-scratch models score 0.900 inside the 0.947 pipeline). Launch detached via WMI.
$ErrorActionPreference = "Continue"
$modelRoot = Split-Path -Parent $PSScriptRoot
Set-Location $modelRoot
$work = "C:\biohub_data\work\model197"
New-Item -ItemType Directory -Force $work | Out-Null
$log = "$work\chain.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
$py = "..\Other\.venv-gpu\Scripts\python.exe"
$env:OMP_NUM_THREADS = "3"
$env:BIOHUB_RUNS_RESUMED = "1"
while (-not (Test-Path "model196\weights\model196_ft_primary\training_receipt.json")) { Start-Sleep -Seconds 120 }
Start-Sleep -Seconds 60
$init = (Resolve-Path "model193\weights\model193_ft_primary\edge_predictor_best.pth").Path
$method = "model197_ft_primary_48ep_lr2e-5"
Note "start: init $init"
& $py model184\train_stage.py --stage finetune --fold all --init $init --method $method --output-root "$work\runs" --epochs 24 --max-iters 250 --lr 2e-5 --seed 20260926 --max-wall-seconds 36000 --execute *> "$work\train_$(Get-Date -Format yyyyMMdd_HHmmss).log"
Note "trainer exit=$LASTEXITCODE"
$dir = "$work\runs\$method\split_0"
if (Test-Path "$dir\training_receipt.json") {
  if (-not (Test-Path "$dir\config.json")) { Copy-Item ..\Data\public_checkpoints\support-pack\weights\unet_transformer\split_0\config.json "$dir\config.json" }
  New-Item -ItemType Directory -Force model197\weights\model197_ft_primary | Out-Null
  Copy-Item "$dir\edge_predictor_best.pth", "$dir\config.json", "$dir\training_receipt.json" model197\weights\model197_ft_primary\ -Force
  Note "done; checkpoint copied to Model\model197\weights\model197_ft_primary"
} else { Note "no training receipt" }
