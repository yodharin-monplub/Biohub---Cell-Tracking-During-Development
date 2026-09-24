# model196: run once the fine-tune has finished (Model\model196\weights\model196_ft_primary exists).
# Uploads the checkpoint as a private dataset (krittanutsomtuas, our team), builds the notebook with its hash pinned,
# and pushes the kernel (commit = quick dummy-data check). Run from the project root.
$ErrorActionPreference = 'Stop'
$py = 'Other\.venv\Scripts\python.exe'
$src = 'Model\model196\weights\model196_ft_primary'
foreach ($f in 'edge_predictor_best.pth', 'config.json') { if (-not (Test-Path "$src\$f")) { throw "model196 output missing: $src\$f" } }
$sha = (Get-FileHash "$src\edge_predictor_best.pth" -Algorithm SHA256).Hash.ToLower()
"model196 sha256 $sha"

$line = (Get-Content Other\.env | Where-Object { $_ -match '^KAGGLE_API_KEY=' }) -split '=', 2
$env:KAGGLE_API_TOKEN = $line[1].Split('#')[0].Trim().Trim('"')
$utf8 = New-Object System.Text.UTF8Encoding($false)

# 1. private dataset (short path outside OneDrive); the slug contains 'model196', which the notebook uses to find it
$ds = 'C:\biohub_data\work\model196\kaggle_dataset'
New-Item -ItemType Directory -Force $ds | Out-Null
Get-ChildItem $src -File | Copy-Item -Destination $ds -Force
[IO.File]::WriteAllText("$ds\dataset-metadata.json", '{"title": "Biohub model196 ft primary weights", "id": "krittanutsomtuas/biohub-model196-ft-primary-weights", "licenses": [{"name": "CC0-1.0"}]}', $utf8)
& $py -m kaggle.cli datasets create -p $ds
for ($i = 0; $i -lt 30; $i++) { $s = & $py -m kaggle.cli datasets status krittanutsomtuas/biohub-model196-ft-primary-weights; if ($s -match 'ready') { break }; Start-Sleep 10 }
if ($s -notmatch 'ready') { throw "dataset not ready: $s" }

# 2. notebook + monitor pin
& $py Model\model196\build.py $sha
$mon = Resolve-Path Model\model196\kaggle\monitor_once.py
[IO.File]::WriteAllText($mon, [IO.File]::ReadAllText($mon).Replace('__SHA196__', $sha), $utf8)

# 3. push (commit only sees the dummy test data; validator sweep disabled at commit)
& $py Model\model196\kaggle\push.py
