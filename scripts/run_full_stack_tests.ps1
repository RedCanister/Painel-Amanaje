$ErrorActionPreference = "Stop"

param(
    [string]$AppUrl = "http://127.0.0.1:8000",
    [string]$MlflowUrl = "http://127.0.0.1:5000",
    [string]$PostgresHost = "127.0.0.1",
    [int]$PostgresPort = 5432,
    [double]$TimeoutSeconds = 180
)

$python = ".\.venv\Scripts\python.exe"
$env:PAINEL_TEST_BASE_URL = $AppUrl
$env:PAINEL_TEST_MLFLOW_URL = $MlflowUrl

& $python scripts\wait_for_stack.py --app-url $AppUrl --mlflow-url $MlflowUrl --postgres-host $PostgresHost --postgres-port $PostgresPort --timeout $TimeoutSeconds
& $python -m pytest api\tests -m "integration or e2e or recovery" -q
