from __future__ import annotations

import pytest


pytestmark = pytest.mark.e2e


def test_assistant_frontend_loads_example_catalog_through_browser(page):
    page.goto("/assistant", wait_until="networkidle")

    page.locator('[data-assistant-tab="examples"]').click()
    page.locator("#assistantExampleCatalog table").wait_for(state="visible")

    catalog = page.evaluate(
        """
        async () => {
            const response = await fetch('/examples/catalog?include_payloads=false');
            return await response.json();
        }
        """
    )

    assert catalog["status"] == "ok"
    assert catalog["summary"]["sample_count"] == catalog["summary"]["object_count"] * 5
    assert catalog["api_proof"]["frontend_route_count"] >= 10

    rendered_text = page.locator("#assistantExampleOutput").inner_text()
    assert "painel-amanaje-example-catalog-v1" in rendered_text
    assert page.locator("#assistantExampleCatalog tbody tr").count() >= catalog["summary"]["object_count"]
