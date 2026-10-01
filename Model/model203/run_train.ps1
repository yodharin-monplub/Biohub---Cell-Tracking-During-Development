# model203: HALF-GENTLE fine-tune (lr 1e-5, 6 epochs); waits for the model201 local check to free the GPU. of the public 50-epoch primary checkpoint (support-pack split_0, sha256 12f6881e...) on all 199
# train movies at low lr (2e-5), 24 epochs x 250 it x batch 2. Warm start keeps the public model's calibration
# (model190 showed our from-scratch models score 0.900 inside the 0.947 pipeline). Launch detached via WMI.
$ErrorActionPreference = "Continue"
$modelRoot = Split-Path -Parent $PSScriptRoot
Set-Location $modelRoot
$work = "C:\biohub_data\work\model203"
New-Item -ItemType Directory -Force $work | Out-Null
$log = "$work\chain.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
$py = "..\Other\.venv-gpu\Scripts\python.exe"
$env:OMP_NUM_THREADS = "3"
$env:BIOHUB_RUNS_RESUMED = "1"
while (-not (Test-Path "C:\biohub_data\work\model199\m201_0965\summary.json")) { Start-Sleep -Seconds 120 }
Start-Sleep -Seconds 60
$init = (Resolve-Path "..\Data\public_checkpoints\support-pack\weights\unet_transformer\split_0\edge_predictor_best.pth").Path
$method = "model203_ft_public_primary_lr1e-5_6ep"
Note "start: init $init"
& $py model184\train_stage.py --stage finetune --fold all --init $init --method $method --output-root "$work\runs" --epochs 6 --max-iters 250 --lr 1e-5 --seed 20260929 --max-wall-seconds 36000 --execute *> "$work\train_$(Get-Date -Format yyyyMMdd_HHmmss).log"
Note "trainer exit=$LASTEXITCODE"
$dir = "$work\runs\$method\split_0"
if (Test-Path "$dir\training_receipt.json") {
  if (-not (Test-Path "$dir\config.json")) { Copy-Item ..\Data\public_checkpoints\support-pack\weights\unet_transformer\split_0\config.json "$dir\config.json" }
  New-Item -ItemType Directory -Force model203\weights\model203_ft_primary | Out-Null
  Copy-Item "$dir\edge_predictor_best.pth", "$dir\config.json", "$dir\training_receipt.json" model203\weights\model203_ft_primary\ -Force
  Note "done; checkpoint copied to Model\model203\weights\model203_ft_primary"
} else { Note "no training receipt" }
