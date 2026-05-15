$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$projectRoot = (Resolve-Path -LiteralPath ".").Path
$runId = [System.Guid]::NewGuid().ToString("N")
$testTemp = Join-Path $projectRoot "temp\pytest-perf-$runId"

New-Item -ItemType Directory -Force -Path $testTemp | Out-Null

$env:PYTHONPATH = $projectRoot
$env:TEMP = $testTemp
$env:TMP = $testTemp

& $python -m pytest api\tests -m "perf" -q --basetemp $testTemp
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
