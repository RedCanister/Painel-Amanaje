from __future__ import annotations

import socket
import time
import urllib.error
import urllib.request


def _wait_for_http(url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if int(getattr(response, "status", 0) or 0) < 500:
                    return
        except Exception as exc:  # pragma: no cover - exercised by scripts, not unit tests.
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for HTTP endpoint '{url}'. Last error: {last_error}")


def _wait_for_tcp(host: str, port: int, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                return
        except OSError as exc:  # pragma: no cover - exercised by scripts, not unit tests.
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for TCP endpoint '{host}:{port}'. Last error: {last_error}")


def wait_for_stack(
    *,
    app_url: str,
    mlflow_url: str,
    postgres_host: str,
    postgres_port: int,
    timeout_seconds: float = 180.0,
) -> None:
    _wait_for_tcp(postgres_host, postgres_port, timeout_seconds)
    _wait_for_http(app_url, timeout_seconds)
    _wait_for_http(mlflow_url, timeout_seconds)
