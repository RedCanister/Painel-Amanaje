$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
& $python -m pytest api\tests -m "not e2e and not perf" -q
