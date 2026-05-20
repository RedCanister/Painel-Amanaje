from __future__ import annotations

from uuid import uuid4

import pytest
from playwright.sync_api import expect

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
    return page.locator(selector).first.evaluate(
        """
        (element) => {
            const style = getComputedStyle(element);
            return {
                borderRadius: parseFloat(style.borderTopLeftRadius || "0"),
                borderColor: style.borderTopColor,
                boxShadow: style.boxShadow,
                boxSizing: style.boxSizing,
                paddingTop: parseFloat(style.paddingTop || "0"),
                fontFamily: style.fontFamily
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
        (SettingsPage, "/registry", "Model Registry - CRUD Management"),
        (AssistantPage, "/assistant", "Assistant Management"),
        (SettingsPage, "/settings", "Global Settings"),
    ],
)
def test_primary_routes_render_core_workspace_surfaces(page, page_cls, path, expected_text):
    workspace = page_cls(page)
    workspace.goto(path)

    assert expected_text in page.content()


def test_panel_workspace_saves_reloads_and_deletes_dashboard(page):
    panel_name = f"E2E Panel {uuid4().hex[:8]}"

    page.add_init_script("window.localStorage.setItem('amanajePanelMode', 'edit')")
    page.goto("/panel", wait_until="networkidle")
    page.locator("#panelModeEdit").wait_for(state="visible")
    page.locator("#panelModeEdit").click()
    page.locator("#panelDashboardName").wait_for(state="visible")
    page.locator("#panelNewDashboard").click()
    page.locator("#panelDashboardName").fill(panel_name)
    page.locator("#panelObjective").fill("Verify saved panel dashboard persistence.")
    page.locator("#panelWidgetKind").select_option("note")
    page.locator("#panelWidgetTitle").fill("E2E Note")
    page.locator("#panelWidgetConfig").fill('{"text": "Saved by Playwright E2E."}')
    page.locator("#panelAddWidget").click()

    expect(page.locator(".panel-widget-title", has_text="E2E Note")).to_be_visible()

    page.locator("#panelSaveDashboard").click()
    expect(page.locator("#statusMessage")).to_contain_text("Panel saved.", timeout=10000)

    page.locator("#panelNewDashboard").click()
    page.locator("#panelDashboardSelect").select_option(label=panel_name)
    expect(page.locator("#panelDashboardName")).to_have_value(panel_name, timeout=10000)
    expect(page.locator(".panel-widget-title", has_text="E2E Note")).to_be_visible()

    page.locator("#panelDeleteDashboard").click()
    expect(page.locator("#statusMessage")).to_contain_text("Panel deleted.", timeout=10000)


def test_panel_modes_sizes_and_dash_workspace_render(page):
    page.add_init_script("window.localStorage.removeItem('amanajePanelMode')")
    page.goto("/panel", wait_until="networkidle")
    page.locator("#panelModeRun").wait_for(state="visible")

    assert page.locator("#panelModeRun").count() == 1
    assert page.locator("#panelModeEdit").count() == 1
    assert page.locator("#panelWidgetSize option[value='compact']").count() == 0
    assert page.locator("#panelWidgetSize option[value='standard']").count() == 0

    page.locator("#panelModeEdit").click()
    page.locator("#panelNewDashboard").click()
    expect(page.locator(".panel-dash-frame")).to_be_visible()
    assert page.locator(".panel-widget select[data-panel-action='resize'] option[value='compact']").count() == 0
    assert page.locator(".panel-widget select[data-panel-action='resize'] option[value='standard']").count() == 0

    page.locator("#panelModeRun").click()
    expect(page.locator(".panel-sidebar")).to_be_hidden()
    expect(page.locator("#panelNewDashboard")).to_be_hidden()
    expect(page.locator(".panel-widget-actions[data-panel-edit-only='true']").first).to_be_hidden()
    expect(page.locator("#panelObjectiveRun")).to_be_visible()


def test_operational_atlas_shell_kicker_and_domain_navigation_render(page):
    page.goto("/settings", wait_until="networkidle")

    assert page.locator(".brand-kicker").inner_text().upper() == "OPERATIONAL ATLAS"
    assert page.locator(".nav-section").count() == 4
    for label in ("Data", "Models", "MLOps", "Management"):
        assert page.locator(".nav-section h3", has_text=label).count() == 1


@pytest.mark.parametrize(
    ("path", "selector"),
    [
        ("/upload", ".workspace-intro h2"),
        ("/create", ".workspace-intro h2"),
        ("/feature", ".workspace-intro h2"),
        ("/assistant", ".assistant-hero h2"),
        ("/settings", ".settings-header h2"),
        ("/registry", ".registry-header h2"),
        ("/training", ".global-context h3"),
        ("/onnx", ".onnx-context-panel h3"),
        ("/production", ".global-watch-panel h3"),
        ("/visualization", ".global-context h3"),
    ],
)
def test_atlas_page_titles_keep_heading_font_family(page, path, selector):
    page.goto(path, wait_until="domcontentloaded")
    page.locator(selector).first.wait_for(state="visible")

    style = _computed_style_snapshot(page, selector)
    assert "Fraunces" in style["fontFamily"]


@pytest.mark.parametrize(
    ("path", "selector"),
    [
        ("/training", ".panel-title"),
        ("/onnx", ".panel-title"),
        ("/production", ".panel-title"),
        ("/visualization", ".panel-title"),
    ],
)
def test_atlas_context_panels_keep_heading_font_family(page, path, selector):
    page.goto(path, wait_until="domcontentloaded")
    page.locator(selector).first.wait_for(state="visible")

    style = _computed_style_snapshot(page, selector)
    assert "Fraunces" in style["fontFamily"]


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


def test_production_interval_schedule_fields_follow_toggle(page):
    page.goto("/production", wait_until="domcontentloaded")
    page.locator("main.container").wait_for(state="visible")

    expect(page.locator("#watchScheduleFields")).to_be_hidden()
    page.locator("#watchScheduleEnabled").check()
    expect(page.locator("#watchScheduleFields")).to_be_visible()
    page.locator("#watchScheduleEnabled").uncheck()
    expect(page.locator("#watchScheduleFields")).to_be_hidden()


def test_panel_display_filters_collapse_and_filter_widget_is_available(page):
    page.add_init_script("window.localStorage.setItem('amanajePanelMode', 'edit')")
    page.goto("/panel", wait_until="networkidle")
    page.locator("#panelModeEdit").wait_for(state="visible")
    page.locator("#panelModeEdit").click()

    expect(page.locator(".panel-filter-details")).to_be_visible()
    expect(page.locator("#panelFilterColumn")).to_be_visible()
    page.locator(".panel-filter-summary-row").click()
    expect(page.locator("#panelFilterColumn")).to_be_hidden()
    page.locator(".panel-filter-summary-row").click()
    expect(page.locator("#panelFilterColumn")).to_be_visible()
    expect(page.locator("#panelWidgetKind option[value='filter_control']")).to_have_count(1)


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
