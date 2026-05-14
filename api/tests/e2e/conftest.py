from __future__ import annotations

import os
import urllib.request

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from playwright.sync_api import Browser, Page, sync_playwright


def _wait_for_http(url: str, timeout_seconds: float = 60.0) -> None:
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if int(getattr(response, "status", 0) or 0) < 500:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for '{url}'.")


@pytest.fixture(scope="session")
def e2e_base_url(full_stack_base_url: str) -> str:
    _wait_for_http(full_stack_base_url)
    return full_stack_base_url


@pytest.fixture()
def browser() -> Browser:
    headless = os.environ.get("PAINEL_E2E_HEADLESS", "1") != "0"
    with sync_playwright() as instance:
        browser = instance.chromium.launch(headless=headless)
        yield browser
        browser.close()


@pytest.fixture()
def page(browser: Browser, e2e_base_url: str) -> Page:
    context = browser.new_context(
        base_url=e2e_base_url,
        viewport={"width": 1440, "height": 960},
    )
    context.route(
        "https://fonts.googleapis.com/*",
        lambda route: route.fulfill(status=200, content_type="text/css", body=""),
    )
    context.route(
        "https://fonts.gstatic.com/*",
        lambda route: route.fulfill(status=200, body=b"", headers={"content-type": "font/woff2"}),
    )
    console_errors: list[str] = []
    network_failures: list[str] = []

    page = context.new_page()
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error" and "favicon" not in message.text.lower()
        else None,
    )
    page.on(
        "requestfailed",
        lambda request: network_failures.append(f"{request.method} {request.url}: {request.failure}")
        if "/favicon.ico" not in request.url
        and "fonts.googleapis.com" not in request.url
        and "fonts.gstatic.com" not in request.url
        and "ERR_ABORTED" not in str(request.failure or "")
        else None,
    )

    yield page

    context.close()
    assert not network_failures, f"Browser network failures detected: {network_failures}"
    assert not console_errors, f"Browser console errors detected: {console_errors}"
