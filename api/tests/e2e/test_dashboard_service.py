from __future__ import annotations

import json
import os
import time
import urllib.request

import pytest


pytestmark = pytest.mark.e2e


def _fetch(url: str, timeout: float = 8.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return int(getattr(response, "status", 0) or 0), body


def _wait_for_dashboard(base_url: str, timeout_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            status, _body = _fetch(f"{base_url}/extensions.json", timeout=4.0)
            if status < 500:
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for dashboard service at {base_url}.")


def test_dashboard_service_and_extensions_endpoint_are_available():
    dashboard_url = os.environ.get("PAINEL_TEST_DASHBOARD_URL", "http://127.0.0.1:8050").rstrip("/")
    _wait_for_dashboard(dashboard_url)

    status, payload_text = _fetch(f"{dashboard_url}/extensions.json")
    payload = json.loads(payload_text)

    assert status == 200
    assert set(payload).issuperset({"enabled", "root", "extensions", "errors"})
    assert isinstance(payload["extensions"], list)
    assert isinstance(payload["errors"], list)

    for path in ("/", "/extensions", "/embed"):
        page_status, page_body = _fetch(f"{dashboard_url}{path}")
        assert page_status == 200
        assert "Painel Amanaje Visualization" in page_body
