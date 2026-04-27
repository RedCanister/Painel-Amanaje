param(
    [string]$AppUrl = "http://127.0.0.1:8000",
    [string]$MlflowUrl = "http://127.0.0.1:5000",
    [string]$PostgresHost = "127.0.0.1",
    [int]$PostgresPort = 5432,
    [double]$TimeoutSeconds = 180,
    [switch]$SkipStackCheck
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"

Write-Host "Generating manual signoff fixtures..." -ForegroundColor Cyan
& $python .\scripts\build_manual_test_fixtures.py

if (-not $SkipStackCheck) {
    Write-Host "Waiting for stack health..." -ForegroundColor Cyan
    & $python .\scripts\wait_for_stack.py --app-url $AppUrl --mlflow-url $MlflowUrl --postgres-host $PostgresHost --postgres-port $PostgresPort --timeout $TimeoutSeconds
}

Write-Host ""
Write-Host "Manual signoff package is ready." -ForegroundColor Green
Write-Host "Checklist: docs/manual-signoff-checklist.md"
Write-Host "Report template: docs/manual-signoff-report-template.md"
Write-Host "Fixture manifest: api/tests/fixtures/manual/manifest.json"
Write-Host "API: $AppUrl"
Write-Host "MLflow: $MlflowUrl"
Write-Host "Postgres: $PostgresHost`:$PostgresPort"
