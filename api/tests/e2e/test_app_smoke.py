from __future__ import annotations

import pytest

from .page_objects import AssistantPage, CreatePage, FeaturePage, ProductionPage, TrainingPage, UploadPage


pytestmark = pytest.mark.e2e


@pytest.mark.parametrize(
    ("page_cls", "path", "expected_text"),
    [
        (UploadPage, "/upload", "Data Upload"),
        (CreatePage, "/create", "Data Creation"),
        (FeaturePage, "/feature", "Feature Workspace"),
        (TrainingPage, "/training", "Global Training Context"),
        (ProductionPage, "/production", "Global Production Watch"),
        (AssistantPage, "/assistant", "Assistant Management"),
    ],
)
def test_primary_routes_render_core_workspace_surfaces(page, page_cls, path, expected_text):
    workspace = page_cls(page)
    workspace.goto(path)

    assert expected_text in page.content()


def test_upload_and_create_support_matrices_are_collapsible(page):
    upload_page = UploadPage(page)
    upload_page.goto("/upload")
    assert page.locator("#supportDetails").count() == 1
    upload_page.expand_support()
    assert page.locator("#supportMatrix .support-card").count() >= 2

    create_page = CreatePage(page)
    create_page.goto("/create")
    assert page.locator("#supportDetails").count() == 1
    create_page.expand_support()
    assert page.locator("#supportMatrix .support-card").count() >= 2


def test_json_blocks_gain_projectwide_collapsible_wrapper(page):
    page.goto("/upload", wait_until="networkidle")
    page.evaluate(
        """
        () => {
            const block = document.createElement('pre');
            block.className = 'json-block';
            block.textContent = JSON.stringify({items: Array.from({length: 25}, (_, index) => ({index, value: `item-${index}`}))}, null, 2);
            document.body.appendChild(block);
            window.AmanajeUI.enhanceCollapsibles(document.body);
        }
        """
    )

    assert page.locator("[data-collapsible-wrapper='true']").count() >= 1
    assert page.locator("[data-collapsible-toggle='true']").first.inner_text() in {"Expand", "Collapse"}


def test_training_and_production_global_context_controls_are_present(page):
    page.goto("/training", wait_until="networkidle")
    assert page.locator("#globalInferencePairId").count() == 1
    assert page.locator("#btnGlobalTrain").count() == 1
    assert page.locator("#btnGlobalRunStudy").count() == 1

    page.goto("/production", wait_until="networkidle")
    assert page.locator("#globalInferencePairId").count() == 1
    assert page.locator("#btnGlobalStartProduction").count() == 1
    assert page.locator("#btnGlobalRunSimulation").count() == 1


@pytest.mark.parametrize(
    ("viewport", "path"),
    [
        ({"width": 1440, "height": 960}, "/upload"),
        ({"width": 980, "height": 960}, "/feature"),
        ({"width": 720, "height": 960}, "/training"),
        ({"width": 560, "height": 960}, "/production"),
    ],
)
def test_workspace_layout_stays_within_available_viewport_width(page, viewport, path):
    page.set_viewport_size(viewport)
    page.goto(path, wait_until="networkidle")

    sizing = page.evaluate(
        """
        () => ({
            documentWidth: document.documentElement.scrollWidth,
            bodyWidth: document.body.scrollWidth,
            viewportWidth: window.innerWidth
        })
        """
    )

    assert sizing["documentWidth"] <= sizing["viewportWidth"] + 2
    assert sizing["bodyWidth"] <= sizing["viewportWidth"] + 2
