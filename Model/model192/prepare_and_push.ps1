# model192: run once model191 has finished training.
# Uploads the model191 checkpoint as a private dataset (krittanutsomtuas, our team), builds the notebook with both
# hashes pinned, and pushes the kernel (commit = quick dummy-data check). Run from the project root.
$ErrorActionPreference = 'Stop'
$py = 'Other\.venv\Scripts\python.exe'
$src = 'Model\model191\weights\model191_scratch_all'
$sha185 = 'ac8dd16249dbffc08e8da88eb1cd7228ed10d1625c22d696774487ed5ef41357'
foreach ($f in 'edge_predictor_best.pth', 'config.json') { if (-not (Test-Path "$src\$f")) { throw "model191 output missing: $src\$f" } }
$sha191 = (Get-FileHash "$src\edge_predictor_best.pth" -Algorithm SHA256).Hash.ToLower()
"model191 sha256 $sha191"

$line = (Get-Content Other\.env | Where-Object { $_ -match '^KAGGLE_API_KEY=' }) -split '=', 2
$env:KAGGLE_API_TOKEN = $line[1].Split('#')[0].Trim().Trim('"')
$utf8 = New-Object System.Text.UTF8Encoding($false)

# 1. private dataset (short path outside OneDrive)
$ds = 'C:\biohub_data\work\model191\kaggle_dataset'
New-Item -ItemType Directory -Force $ds | Out-Null
Get-ChildItem $src -File | Copy-Item -Destination $ds -Force
[IO.File]::WriteAllText("$ds\dataset-metadata.json", '{"title": "Biohub model191 scratch seed", "id": "krittanutsomtuas/biohub-model191-scratch-seed", "licenses": [{"name": "CC0-1.0"}]}', $utf8)
& $py -m kaggle.cli datasets create -p $ds
for ($i = 0; $i -lt 30; $i++) { $s = & $py -m kaggle.cli datasets status krittanutsomtuas/biohub-model191-scratch-seed; if ($s -match 'ready') { break }; Start-Sleep 10 }
if ($s -notmatch 'ready') { throw "dataset not ready: $s" }

# 2. notebook + monitor pin
& $py Model\model192\build.py $sha185 $sha191
$mon = Resolve-Path Model\model192\kaggle\monitor_once.py
[IO.File]::WriteAllText($mon, [IO.File]::ReadAllText($mon).Replace('__SHA191__', $sha191), $utf8)

# 3. push (commit only sees the dummy test data; validator sweep disabled at commit)
& $py Model\model192\kaggle\push.py
