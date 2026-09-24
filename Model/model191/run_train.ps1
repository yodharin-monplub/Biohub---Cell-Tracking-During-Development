# model191: second LONG-trained all-data seed (all 199 train movies, random init, 160 epochs x 250 it x batch 2,
# seed 20260922) for a two-long-seed final ensemble with model185. Trained locally (RTX 4050, ~13-14 h).
# Self-resuming: if the trainer dies (other jobs on this laptop have killed it before), the loop restarts it
# from the last per-epoch checkpoint (--init, strict warm start; optimizer state is lost) for the remaining epochs.
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$work = "C:\biohub_data\work\model191"
New-Item -ItemType Directory -Force $work | Out-Null
$log = "$work\chain.log"
function Note($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
$py = "..\Other\.venv-gpu\Scripts\python.exe"
$env:OMP_NUM_THREADS = "3"
$env:BIOHUB_RUNS_RESUMED = "1"
$total = 160
$method = "model191_scratch_all_seed20260922"
$last = "$work\last_dir.txt"   # each segment needs a fresh run dir (the trainer refuses an existing one)
$dir = ""; if (Test-Path $last) { $dir = (Get-Content $last -Raw).Trim() }

# wait for the model189 mixed predictor to release the GPU
while (-not ((Get-Content C:\biohub_data\work\model189\chain_mixed.log -ErrorAction SilentlyContinue) -match 'predictor exit')) { Start-Sleep -Seconds 120 }

for ($segment = 0; $segment -lt 12; $segment++) {
  if ($dir -and (Test-Path "$dir\training_receipt.json")) { break }
  # epochs_offset.txt holds epochs whose logs were lost (before 2026-09-22 every relaunch reused train_seg0.log and
  # truncated it: 20 + 94 = 114 epochs). Each segment now writes its own train_run_<timestamp>.log.
  $done = 0
  if (Test-Path "$work\epochs_offset.txt") { $done = [int](Get-Content "$work\epochs_offset.txt" -Raw).Trim() }
  Get-ChildItem $work -Filter "train_run_*.log" -ErrorAction SilentlyContinue | ForEach-Object { $done += (Select-String -Path $_.FullName -Pattern '^\s*Epoch\s+\d+/').Count }
  $left = $total - $done
  if ($left -le 0) { break }
  $extra = @()
  if ($done -gt 0) {
    if ($dir -and (Test-Path "$dir\edge_predictor_best.pth")) { Copy-Item "$dir\edge_predictor_best.pth" "$work\resume_init.pth" -Force }
    if (-not (Test-Path "$work\resume_init.pth")) { Note "no checkpoint to resume from; stopping"; break }
    $extra = @("--init", "$work\resume_init.pth")
  }
  $segMethod = "${method}_seg$segment"; while (Test-Path "$work\runs\$segMethod") { $segMethod += "x" }
  $dir = "$work\runs\$segMethod\split_0"; $dir | Out-File -Encoding ascii $last
  Note "segment $segment : $done epochs done, training $left more in $segMethod"
  & $py model184\train_stage.py --stage finetune --fold all --method $segMethod --output-root "$work\runs" --epochs $left --max-iters 250 --seed (20260922 + $segment) --max-wall-seconds 86400 @extra --execute *> "$work\train_run_$(Get-Date -Format yyyyMMdd_HHmmss).log"
  Note "segment $segment exit=$LASTEXITCODE"
  Start-Sleep -Seconds 60
}
if ($dir -and (Test-Path "$dir\training_receipt.json")) {
  if (-not (Test-Path "$dir\config.json")) { Copy-Item ..\Data\public_checkpoints\support-pack\weights\unet_transformer\split_0\config.json "$dir\config.json" }
  New-Item -ItemType Directory -Force model191\weights\model191_scratch_all | Out-Null
  Copy-Item "$dir\edge_predictor_best.pth", "$dir\config.json", "$dir\training_receipt.json" model191\weights\model191_scratch_all\ -Force
  Copy-Item $log model191\weights\model191_scratch_all\segments.log -Force
  Note "done; checkpoint copied to Model\model191\weights"
} else { Note "gave up without a receipt" }
