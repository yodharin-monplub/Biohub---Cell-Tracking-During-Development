# Sequential GPU queue: waits for any running predictor to finish, then runs the next
# predict_stems.py variant on set C (4 example-test + 8 validator movies).
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:BIOHUB_COMP_DIR = (Resolve-Path ..\Data\competition).Path
$env:BIOHUB_WORKING_DIR = "C:\biohub_data\work\model167"
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path ..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt).Path
$env:OMP_NUM_THREADS = "3"
$stemsC = '44b6_0113de3b,44b6_0b24845f,6bba_05b6850b,6bba_05db0fb1,44b6_12dfb391,44b6_267148e4,44b6_2a2eff9f,44b6_341df25f,6bba_062c8d37,6bba_07e24132,6bba_085bf656,6bba_09961292'
$jobs = @(
  @{ method = 'unet_transformer_disapp15'; tag = 'disapp15'; extra = @('--ilp-disappearance-weight', '1.5') },
  @{ method = 'unet_transformer_det096';   tag = 'det096';   extra = @('--det-threshold', '0.96') },
  @{ method = 'unet_transformer_det097';   tag = 'det097';   extra = @('--det-threshold', '0.97') }
)
function Wait-Predictor {
  while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'predict_unet_transformer|predict_stems' }
    if (-not $busy) { return }
    Start-Sleep -Seconds 60
  }
}
foreach ($j in $jobs) {
  Wait-Predictor
  $log = "C:\biohub_data\work\model180_predict_$($j.tag).log"
  "$(Get-Date -Format o) starting $($j.method)" | Out-File -Append -Encoding utf8 C:\biohub_data\work\model180_gpu_queue.log
  $args = @('model180\predict_stems.py', '--stems', $stemsC, '--method', $j.method, '--shard-tag', $j.tag) + $j.extra
  & ..\Other\.venv-gpu\Scripts\python.exe @args *> $log
  "$(Get-Date -Format o) finished $($j.method) exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 C:\biohub_data\work\model180_gpu_queue.log
}
