param(
    [string]$AppUrl = "http://127.0.0.1:8000",
    [string]$MlflowUrl = "http://127.0.0.1:5000",
    [string]$PostgresHost = "127.0.0.1",
    [int]$PostgresPort = 5432,
    [double]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$projectRoot = (Resolve-Path -LiteralPath ".").Path
$testTemp = Join-Path $projectRoot "temp\pytest-full-stack"

New-Item -ItemType Directory -Force -Path $testTemp | Out-Null

$env:PYTHONPATH = $projectRoot
$env:PAINEL_TEST_BASE_URL = $AppUrl
$env:PAINEL_TEST_MLFLOW_URL = $MlflowUrl
$env:TEMP = $testTemp
$env:TMP = $testTemp

& $python scripts\wait_for_stack.py --app-url $AppUrl --mlflow-url $MlflowUrl --postgres-host $PostgresHost --postgres-port $PostgresPort --timeout $TimeoutSeconds
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -m pytest api\tests -m "integration or e2e or recovery" -q --basetemp $testTemp
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
