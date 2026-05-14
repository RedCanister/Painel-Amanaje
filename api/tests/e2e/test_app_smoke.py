from __future__ import annotations

import pytest

from .page_objects import (
    AssistantPage,
    CreatePage,
    FeaturePage,
    ProductionPage,
    SettingsPage,
    TrainingPage,
    UploadPage,
    VisualizationPage,
)


pytestmark = pytest.mark.e2e


def _computed_style_snapshot(page, selector):
    return page.locator(selector).evaluate(
        """
        (element) => {
            const style = getComputedStyle(element);
            return {
                borderRadius: parseFloat(style.borderTopLeftRadius || "0"),
                borderColor: style.borderTopColor,
                boxShadow: style.boxShadow,
                boxSizing: style.boxSizing,
                paddingTop: parseFloat(style.paddingTop || "0")
            };
        }
        """
    )


@pytest.mark.parametrize(
    ("page_cls", "path", "expected_text"),
    [
        (UploadPage, "/upload", "Data Upload"),
        (CreatePage, "/create", "Data Creation"),
        (FeaturePage, "/feature", "Feature Workspace"),
        (TrainingPage, "/training", "Global Training Context"),
        (TrainingPage, "/onnx", "ONNX Framework Workspace"),
        (ProductionPage, "/production", "Global Production Watch"),
        (VisualizationPage, "/visualization", "Visualization"),
        (VisualizationPage, "/plot", "Visualization"),
        (AssistantPage, "/assistant", "Assistant Management"),
        (SettingsPage, "/settings", "Global Settings"),
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
    page.wait_for_load_state("networkidle")

    create_page = CreatePage(page)
    create_page.goto("/create")
    assert page.locator("#supportDetails").count() == 1
    create_page.expand_support()
    assert page.locator("#supportMatrix .support-card").count() >= 2
    page.wait_for_load_state("networkidle")


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


def test_training_global_context_controls_are_present(page):
    page.goto("/training", wait_until="domcontentloaded")
    page.locator("main.container").wait_for(state="visible")
    assert page.locator("#globalInferencePairId").count() == 1
    assert page.locator("#btnGlobalTrain").count() == 1
    assert page.locator("#btnGlobalRunStudy").count() == 1
    assert page.locator("#globalDevice").count() == 1


def test_production_global_context_controls_are_present(page):
    page.goto("/production", wait_until="domcontentloaded")
    page.locator("main.container").wait_for(state="visible")
    assert page.locator("#globalInferencePairId").count() == 1
    assert page.locator("#btnGlobalStartProduction").count() == 1
    assert page.locator("#btnGlobalRunSimulation").count() == 1
    assert page.locator("#simulationContextStrip").count() == 1
    assert page.locator("#simulationFamilyStrip").count() == 1
    assert page.locator("#simulationMode").count() == 1
    assert page.locator("#simulationFamilyControls").count() == 1
    assert page.locator("#simulationBaselineSource").count() == 1
    assert page.locator("#simulationFeatureControls").count() == 1
    assert page.locator("#simulationScenarios").count() == 1
    assert page.locator("#simulationFamilyPayload").count() == 1


def test_visualization_loads_with_local_plotly_fallback_without_forcing_dashboard(page):
    page.goto("/visualization", wait_until="domcontentloaded")
    page.locator("main.container").wait_for(state="visible")

    assert page.locator("#plotSourceType").count() == 1
    assert page.locator("#galleryDeck").count() == 1
    assert page.locator("[data-plotly-fallback='true']").count() >= 1
    assert page.locator(".plotly-frame").count() == 0
    assert "Visualization Gallery" in page.content()


@pytest.mark.parametrize(
    ("path", "panel_selector", "control_selectors"),
    [
        ("/upload", "#uploadSection", ["#datasetObjectName", "#datasetType", "#datasetDescription", "#datasetSelect"]),
        ("/create", "#uploadSection", ["#assistantCreatePrompt", "#assistantCreateModel", "#datasetSelect"]),
        ("/feature", "section.feature-panel:has(#featureDatasetId)", ["#featureDatasetId", "#featureViewName", "#columnSearch", "#columnSort"]),
    ],
)
def test_red_domain_workspaces_keep_styled_panels_and_controls(page, path, panel_selector, control_selectors):
    page.goto(path, wait_until="networkidle")

    panel_style = _computed_style_snapshot(page, panel_selector)
    assert panel_style["borderRadius"] >= 14
    assert panel_style["boxShadow"] != "none"

    for selector in control_selectors:
        control_style = _computed_style_snapshot(page, selector)
        assert control_style["borderRadius"] >= 10
        assert control_style["boxSizing"] == "border-box"
        assert control_style["paddingTop"] >= 10
        assert control_style["borderColor"] != "rgb(118, 118, 118)"


@pytest.mark.parametrize(
    ("viewport", "path"),
    [
        ({"width": 1440, "height": 960}, "/upload"),
        ({"width": 980, "height": 960}, "/create"),
        ({"width": 980, "height": 960}, "/feature"),
        ({"width": 390, "height": 860}, "/create"),
        ({"width": 390, "height": 860}, "/feature"),
        ({"width": 720, "height": 960}, "/training"),
        ({"width": 560, "height": 960}, "/onnx"),
        ({"width": 560, "height": 960}, "/production"),
        ({"width": 560, "height": 960}, "/visualization"),
        ({"width": 560, "height": 960}, "/settings"),
    ],
)
def test_workspace_layout_stays_within_available_viewport_width(page, viewport, path):
    page.set_viewport_size(viewport)
    if path in {"/training", "/visualization"}:
        page.goto(path, wait_until="domcontentloaded")
        page.locator("main.container").wait_for(state="visible")
    else:
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


def test_assistant_toggle_expands_and_collapses_panel(page):
    page.goto("/feature", wait_until="networkidle")
    page.evaluate(
        """
        () => window.AmanajeUI.applyGlobalSettings({
            feature_flags: { assistant_visible: true, debug_mode: false }
        })
        """
    )
    toggle = page.locator("[data-assistant-toggle='feature']")
    panel = page.locator("[data-assistant-panel='feature']")
    toggle.wait_for(state="visible")

    assert toggle.get_attribute("aria-expanded") == "false"
    assert toggle.get_attribute("aria-controls") == panel.get_attribute("id")
    assert panel.is_hidden()

    toggle.click()
    assert toggle.get_attribute("aria-expanded") == "true"
    assert panel.is_visible()

    toggle.click()
    assert toggle.get_attribute("aria-expanded") == "false"
    assert panel.is_hidden()


def test_assistant_visibility_setting_hides_assistant_controls(page):
    page.goto("/feature", wait_until="networkidle")
    page.evaluate(
        """
        () => window.AmanajeUI.applyGlobalSettings({
            feature_flags: { assistant_visible: true, debug_mode: false }
        })
        """
    )
    container = page.locator("[data-assistant-collapsible='feature']")
    container.wait_for(state="visible")

    page.evaluate(
        """
        () => window.AmanajeUI.applyGlobalSettings({
            feature_flags: { assistant_visible: false, debug_mode: false }
        })
        """
    )
    assert container.evaluate("element => getComputedStyle(element).display") == "none"

    page.evaluate(
        """
        () => window.AmanajeUI.applyGlobalSettings({
            feature_flags: { assistant_visible: true, debug_mode: false }
        })
        """
    )
    assert container.evaluate("element => getComputedStyle(element).display") != "none"


def test_settings_debug_capture_exposes_stack_only_when_debug_mode_is_enabled(page):
    page.goto("/settings", wait_until="networkidle")
    debug_toggle = page.locator("#settingsDebugMode")

    debug_toggle.set_checked(False)
    quiet_payload = page.evaluate(
        """
        () => window.AmanajeUI.captureFrontendError(new Error('quiet capture'), { source: 'e2e' })
        """
    )
    assert "stack" not in quiet_payload
    assert "backend_debug" not in quiet_payload

    debug_toggle.set_checked(True)
    debug_payload = page.evaluate(
        """
        () => window.AmanajeUI.captureFrontendError(new Error('forced debug capture'), { source: 'e2e' })
        """
    )

    assert page.evaluate("window.AmanajeDebug") is True
    assert "Error: forced debug capture" in debug_payload["stack"]
    assert "forced debug capture" in page.locator("#settingsFrontendDebug").inner_text()


def test_settings_system_logs_panel_renders_terminal_view(page):
    page.goto("/settings", wait_until="networkidle")
    page.locator("#settingsLogsIncludeDocker").set_checked(False)
    page.locator("#settingsLogsRefresh").click()

    terminal = page.locator("#settingsLogsTerminal")
    terminal.wait_for(state="visible")

    assert "System Logs" in page.content()
    assert page.locator("#settingsLogSource").count() == 1
    assert page.locator("#settingsLogsAutoRefresh").is_checked()
    assert terminal.evaluate("element => element.clientWidth <= window.innerWidth")
