from __future__ import annotations

from pathlib import Path


UI_UTILITIES = Path(__file__).resolve().parents[1] / "static" / "js" / "ui-utilities.js"


def _source() -> str:
    return UI_UTILITIES.read_text(encoding="utf-8")


def test_json_explorer_public_helpers_are_exported():
    source = _source()

    assert "function normalizeJsonTable" in source
    assert "function renderJsonExplorer" in source
    assert "function renderPlotlyTableSpec" in source
    assert "function applyPlotTheme" in source
    assert "function plotIdentityMetadata" in source
    assert "normalizeJsonTable," in source
    assert "renderJsonExplorer," in source
    assert "renderPlotlyTableSpec," in source
    assert "applyPlotTheme," in source
    assert "plotIdentityMetadata," in source


def test_json_table_parser_covers_expected_shapes():
    source = _source()

    assert "Array.isArray(parsed.columns) && Array.isArray(parsed.rows)" in source
    assert "parsed.every(isPlainObject)" in source
    assert '["key", "value"]' in source
    assert 'reason: "not_tabular"' in source
    assert "compactJsonCell" in source


def test_json_explorer_keeps_common_and_plotly_modes():
    source = _source()

    assert 'label: "Table"' in source
    assert 'label: "Common JSON"' in source
    assert 'label: "Plotly Table"' in source
    assert "options.explorer !== false" in source
    assert "maxPlotlyRows" in source
    assert "maxPlotlyColumns" in source


def test_plot_renderer_has_theme_metadata_and_lightbox_hooks():
    source = _source()

    assert "getVisualizationTheme" in source
    assert "applyTraceTheme" in source
    assert "renderPlotMetadataBadges" in source
    assert "data-plot-lightbox" in source
    assert "openPlotImageLightbox" in source


def test_plotly_renderer_uses_manual_embed_fallback_for_dashboard_outages():
    source = _source()

    assert "function renderPlotlyFallback" in source
    assert 'data-plotly-fallback="true"' in source
    assert "Open interactive renderer" in source
    assert "data-plotly-frame-load" in source
    assert "Load embedded renderer" in source
