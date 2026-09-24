# Download the competition data with kagglehub into Data/competition (a junction to C:\biohub_data\raw).
# Reads KAGGLE_API_KEY from .env; never prints it.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$kg = ((Get-Content Other\.env | Where-Object { $_ -match '^KAGGLE_API_KEY' }) -replace '^KAGGLE_API_KEY=\s*','' -replace '\s*#.*$','')
$env:KAGGLE_API_TOKEN = $kg
$env:KAGGLEHUB_CACHE = "C:\biohub_data\kagglehub_cache"
$log = "C:\biohub_data\download.log"
"start $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 $log
& Other\.venv\Scripts\python.exe Other\scripts\download_competition.py --output-dir Data\competition *>> $log
"exit $LASTEXITCODE $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 $log
