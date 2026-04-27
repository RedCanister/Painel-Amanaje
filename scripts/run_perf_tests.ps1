$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
& $python -m pytest api\tests -m "perf" -q
