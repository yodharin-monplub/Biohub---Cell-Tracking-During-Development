# Second GPU queue: waits for gpu_queue.ps1 to finish det097, scores queue-1 outputs with the
# no-relink post-processing, then runs association-parameter variants (predict + score).
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:BIOHUB_COMP_DIR = (Resolve-Path ..\Data\competition).Path
$env:BIOHUB_WORKING_DIR = "C:\biohub_data\work\model167"
$env:BIOHUB_DEEPCENTER_CHECKPOINT = (Resolve-Path ..\Data\public_checkpoints\deepcenter\weights\full_frame_center\best.pt).Path
$env:OMP_NUM_THREADS = "3"
$stemsC = '44b6_0113de3b,44b6_0b24845f,6bba_05b6850b,6bba_05db0fb1,44b6_12dfb391,44b6_267148e4,44b6_2a2eff9f,44b6_341df25f,6bba_062c8d37,6bba_07e24132,6bba_085bf656,6bba_09961292'
$qlog = "C:\biohub_data\work\model180_gpu_queue.log"
$root = "C:\biohub_data\work\model167\tracking_repo\predictions\yodha"
$cfg = (Resolve-Path model180\configs\nr_only.json).Path

function Score-Method($method) {
  $out = "model180\results\sweep_${method}_setC.csv"
  if (Test-Path $out) { return }
  "$(Get-Date -Format o) scoring $method" | Out-File -Append -Encoding utf8 $qlog
  & ..\Other\.venv-gpu\Scripts\python.exe model180\sweep_configs.py --pred-root "$root\$method" --stems $stemsC --configs $cfg --csv-out $out *> "C:\biohub_data\work\model180_sweep_$method.log"
  "$(Get-Date -Format o) scored $method exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $qlog
}

# wait for queue 1
while (-not ((Get-Content $qlog -ErrorAction SilentlyContinue) -match 'finished unet_transformer_det097')) { Start-Sleep -Seconds 120 }
foreach ($m in 'unet_transformer_disapp15','unet_transformer_det096','unet_transformer_det097') { Score-Method $m }

$jobs = @(
  @{ method = 'unet_transformer_edgethr040'; tag = 'edgethr040'; extra = @('--env', 'BIOHUB_DUAL_SEED_EDGE_THRESHOLD=0.40') },
  @{ method = 'unet_transformer_edgethr056'; tag = 'edgethr056'; extra = @('--env', 'BIOHUB_DUAL_SEED_EDGE_THRESHOLD=0.56') },
  @{ method = 'unet_transformer_bidir000';   tag = 'bidir000';   extra = @('--env', 'BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT=0.0') },
  @{ method = 'unet_transformer_bidir030';   tag = 'bidir030';   extra = @('--env', 'BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT=0.30') },
  @{ method = 'unet_transformer_secdet060';  tag = 'secdet060';  extra = @('--env', 'BIOHUB_SECONDARY_DETECTION_WEIGHT=0.60') },
  @{ method = 'unet_transformer_secdet100';  tag = 'secdet100';  extra = @('--env', 'BIOHUB_SECONDARY_DETECTION_WEIGHT=1.00') }
)
foreach ($j in $jobs) {
  $done = (Get-ChildItem "$root\$($j.method)" -Recurse -Filter *.geff -Directory -ErrorAction SilentlyContinue).Count
  if ($done -lt 12) {
    "$(Get-Date -Format o) starting $($j.method)" | Out-File -Append -Encoding utf8 $qlog
    $args = @('model180\predict_stems.py', '--stems', $stemsC, '--method', $j.method, '--shard-tag', $j.tag) + $j.extra
    & ..\Other\.venv-gpu\Scripts\python.exe @args *> "C:\biohub_data\work\model180_predict_$($j.tag).log"
    "$(Get-Date -Format o) finished $($j.method) exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $qlog
  }
  Score-Method $j.method
}
"$(Get-Date -Format o) queue2 complete" | Out-File -Append -Encoding utf8 $qlog
