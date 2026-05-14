from __future__ import annotations

import base64
import json
import os
from typing import Any, Mapping
from urllib.parse import parse_qs

import dash
import plotly.graph_objects as go
import requests
from dash import Input, Output, dcc, html


API_INTERNAL_URL = os.getenv("AMANAJE_API_INTERNAL_URL", "http://localhost:8000").rstrip("/")
API_PUBLIC_URL = os.getenv("AMANAJE_API_PUBLIC_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("AMANAJE_DASHBOARD_REQUEST_TIMEOUT", "8"))

app = dash.Dash(__name__, suppress_callback_exceptions=True, title="Painel Amanaje Plotly")
server = app.server


def api_get(path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any] | list[Any]:
    url = f"{API_INTERNAL_URL}/{path.lstrip('/')}"
    try:
        response = requests.get(url, params=dict(params or {}), timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"status": "error", "detail": str(exc), "url": url}


def public_api_url(path: str) -> str:
    if path.startswith(("http://", "https://")):
        return path
    return f"{API_PUBLIC_URL}/{path.lstrip('/')}"


def option_rows(path: str, *, label_field: str = "name", value_field: str = "id") -> list[dict[str, str]]:
    payload = api_get(path)
    rows = payload if isinstance(payload, list) else payload.get("runs", [])
    if not isinstance(rows, list):
        return []
    options: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        value = row.get(value_field)
        if value in (None, ""):
            continue
        label = row.get(label_field) or row.get("run_id") or row.get("id") or value
        options.append({"label": str(label), "value": str(value)})
    return options


def split_figure(figure: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = dict(figure or {})
    return (
        {"data": payload.get("data", []), "layout": payload.get("layout", {})},
        dict(payload.get("config", {"displaylogo": False, "responsive": True})),
    )


def series_to_figure(plot: Mapping[str, Any]) -> dict[str, Any]:
    series = plot.get("series") if isinstance(plot.get("series"), list) else []
    labels = [str(item.get("label", "")) for item in series if isinstance(item, Mapping)]
    values = [float(item.get("value") or 0) for item in series if isinstance(item, Mapping)]
    if plot.get("plot_type") == "line":
        return go.Figure(
            data=[go.Scatter(x=labels, y=values, mode="lines+markers")],
            layout=go.Layout(template="plotly_white", title=plot.get("title") or "Plot"),
        ).to_dict()
    return go.Figure(
        data=[go.Bar(x=labels, y=values)],
        layout=go.Layout(template="plotly_white", title=plot.get("title") or "Plot"),
    ).to_dict()


def render_table(plot: Mapping[str, Any]) -> html.Div:
    rows = plot.get("rows") if isinstance(plot.get("rows"), list) else []
    columns = plot.get("columns") if isinstance(plot.get("columns"), list) else []
    if not columns and rows and isinstance(rows[0], Mapping):
        columns = list(rows[0].keys())
    return html.Div(
        [
            html.Table(
                [
                    html.Thead(html.Tr([html.Th(str(column)) for column in columns])),
                    html.Tbody(
                        [
                            html.Tr([html.Td(json.dumps(row.get(column)) if isinstance(row.get(column), (dict, list)) else str(row.get(column, ""))) for column in columns])
                            for row in rows[:20]
                            if isinstance(row, Mapping)
                        ]
                    ),
                ],
                className="dash-table",
            )
        ],
        className="dash-table-wrap",
    )


def render_plot_card(plot: Mapping[str, Any]) -> html.Article:
    title = str(plot.get("title") or "Generated Plot")
    description = str(plot.get("description") or "")
    header = [html.H3(title), html.P(description, className="muted") if description else None]
    header = [item for item in header if item is not None]

    if plot.get("kind") == "legacy_image":
        artifact = dict(plot.get("artifact") or {})
        src = public_api_url(str(artifact.get("url") or ""))
        body = html.A(html.Img(src=src, className="artifact-image"), href=src, target="_blank")
    elif plot.get("kind") == "table":
        body = render_table(plot)
    else:
        figure = plot.get("figure") if isinstance(plot.get("figure"), Mapping) else series_to_figure(plot)
        graph_figure, config = split_figure(figure)
        body = dcc.Graph(figure=graph_figure, config=config, className="dash-graph")

    return html.Article([*header, body], className="plot-card")


def render_plot_deck(plots: Any) -> html.Div:
    usable_plots = [plot for plot in plots if isinstance(plot, Mapping)] if isinstance(plots, list) else []
    if not usable_plots:
        return html.Div("No plots available for this selection yet.", className="muted")
    return html.Div([render_plot_card(plot) for plot in usable_plots], className="plot-grid")


def decode_embedded_figure(search: str) -> tuple[str, dict[str, Any]]:
    query = parse_qs((search or "").lstrip("?"))
    title = (query.get("title") or ["Plot"])[0]
    encoded = (query.get("figure") or [""])[0]
    if not encoded:
        return title, {}
    padded = encoded + ("=" * (-len(encoded) % 4))
    raw = base64.b64decode(padded.encode("ascii"))
    return title, json.loads(raw.decode("utf-8"))


def render_embed(search: str) -> html.Div:
    title, figure = decode_embedded_figure(search)
    graph_figure, config = split_figure(figure)
    return html.Div(
        [
            html.Div(title, className="embed-title"),
            dcc.Graph(figure=graph_figure, config=config, className="embed-graph"),
        ],
        className="embed-shell",
    )


def render_dashboard() -> html.Div:
    dataset_options = option_rows("/datasetmodel/list")
    model_options = option_rows("/learningmodel/list")
    run_payload = api_get("/runs/list")
    run_rows = run_payload.get("runs", []) if isinstance(run_payload, Mapping) else []
    run_options = [
        {"label": f"{row.get('run_type', 'run')} | {row.get('run_id')}", "value": str(row.get("run_id"))}
        for row in run_rows
        if isinstance(row, Mapping) and row.get("run_id")
    ]
    inference_payload = api_get("/production/history")
    inference_rows = inference_payload.get("runs", []) if isinstance(inference_payload, Mapping) else []
    inference_options = [
        {"label": row.get("name") or f"Inference #{row.get('id')}", "value": str(row.get("id"))}
        for row in inference_rows
        if isinstance(row, Mapping) and row.get("id") is not None
    ]

    return html.Div(
        [
            html.Header(
                [
                    html.H1("Painel Amanaje Plotly"),
                    html.P("Interactive plots rendered from the FastAPI registry, runtime artifacts, and analysis endpoints."),
                ],
                className="hero",
            ),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Datasets",
                        children=[
                            dcc.Dropdown(dataset_options, id="dataset-select", placeholder="Select a dataset"),
                            html.Div(id="dataset-plots", className="tab-body"),
                        ],
                    ),
                    dcc.Tab(
                        label="Features",
                        children=[
                            dcc.Dropdown(dataset_options, id="feature-dataset-select", placeholder="Select a dataset"),
                            html.Div(id="feature-plots", className="tab-body"),
                        ],
                    ),
                    dcc.Tab(
                        label="Models",
                        children=[
                            dcc.Dropdown(model_options, id="model-select", placeholder="Select a model"),
                            html.Div(id="model-plots", className="tab-body"),
                        ],
                    ),
                    dcc.Tab(
                        label="Training Runs",
                        children=[
                            dcc.Dropdown(run_options, id="run-select", placeholder="Select a run"),
                            html.Div(id="run-plots", className="tab-body"),
                        ],
                    ),
                    dcc.Tab(
                        label="Production",
                        children=[
                            dcc.Dropdown(inference_options, id="inference-select", placeholder="Select an inference pair"),
                            html.Div(id="inference-plots", className="tab-body"),
                        ],
                    ),
                    dcc.Tab(
                        label="Artifacts",
                        children=[
                            dcc.Dropdown(
                                [
                                    {"label": "All artifacts", "value": ""},
                                    {"label": "Loss curves", "value": "loss_curve"},
                                    {"label": "Metric comparisons", "value": "metric_comparison"},
                                    {"label": "Predictions vs actuals", "value": "predictions_vs_actuals"},
                                ],
                                id="artifact-kind-select",
                                value="",
                                clearable=False,
                            ),
                            html.Div(id="artifact-plots", className="tab-body"),
                        ],
                    ),
                ]
            ),
        ],
        className="dashboard-shell",
    )


app.layout = html.Div([dcc.Location(id="location"), html.Div(id="page")])


@app.callback(Output("page", "children"), Input("location", "pathname"), Input("location", "search"))
def render_page(pathname: str, search: str) -> html.Div:
    if pathname == "/embed":
        return render_embed(search)
    return render_dashboard()


@app.callback(Output("dataset-plots", "children"), Input("dataset-select", "value"))
def update_dataset_plots(dataset_id: str | None) -> html.Div:
    if not dataset_id:
        return html.Div("Select a dataset to inspect profile plots.", className="muted")
    payload = api_get("/analysis/data", {"dataset_id": dataset_id})
    return render_plot_deck(payload.get("plots", []) if isinstance(payload, Mapping) else [])


@app.callback(Output("feature-plots", "children"), Input("feature-dataset-select", "value"))
def update_feature_plots(dataset_id: str | None) -> html.Div:
    if not dataset_id:
        return html.Div("Select a dataset to inspect feature workspace plots.", className="muted")
    payload = api_get("/features/extract", {"dataset_id": dataset_id})
    return render_plot_deck(payload.get("plots", []) if isinstance(payload, Mapping) else [])


@app.callback(Output("model-plots", "children"), Input("model-select", "value"))
def update_model_plots(model_id: str | None) -> html.Div:
    if not model_id:
        return html.Div("Select a model to inspect model plots.", className="muted")
    payload = api_get("/analysis/model", {"model_id": model_id})
    return render_plot_deck(payload.get("plots", []) if isinstance(payload, Mapping) else [])


@app.callback(Output("run-plots", "children"), Input("run-select", "value"))
def update_run_plots(run_id: str | None) -> html.Div:
    if not run_id:
        return html.Div("Select a run to inspect saved plot artifacts.", className="muted")
    payload = api_get("/plots/artifacts", {"job_id": run_id})
    return render_plot_deck(payload.get("plots", []) if isinstance(payload, Mapping) else [])


@app.callback(Output("inference-plots", "children"), Input("inference-select", "value"))
def update_inference_plots(inference_id: str | None) -> html.Div:
    payload = api_get("/production/history")
    runs = payload.get("runs", []) if isinstance(payload, Mapping) else []
    selected = next((row for row in runs if isinstance(row, Mapping) and str(row.get("id")) == str(inference_id)), None)
    if not selected:
        return html.Div("Select an inference pair to inspect production plots.", className="muted")
    return render_plot_deck(selected.get("plots", []))


@app.callback(Output("artifact-plots", "children"), Input("artifact-kind-select", "value"))
def update_artifact_plots(kind: str | None) -> html.Div:
    params = {"kind": kind} if kind else {}
    payload = api_get("/plots/artifacts", params)
    return render_plot_deck(payload.get("plots", []) if isinstance(payload, Mapping) else [])


server.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; font-family: Inter, Arial, sans-serif; background: #f7fffd; color: #1f2937; }
            .dashboard-shell { max-width: 1180px; margin: 0 auto; padding: 28px; }
            .hero { margin-bottom: 24px; }
            .hero h1 { margin: 0 0 6px; color: #115e59; }
            .hero p, .muted { color: #64748b; }
            .tab-body { padding: 18px 0; }
            .plot-grid { display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
            .plot-card { background: white; border: 1px solid #d1fae5; border-radius: 12px; padding: 16px; box-shadow: 0 14px 32px rgba(15, 118, 110, 0.08); }
            .plot-card h3 { margin: 0 0 6px; color: #115e59; }
            .artifact-image { width: 100%; max-height: 440px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px; background: white; }
            .dash-table-wrap { max-height: 420px; overflow: auto; }
            .dash-table { width: 100%; border-collapse: collapse; font-size: 13px; }
            .dash-table th, .dash-table td { border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }
            .embed-shell { height: 100vh; display: grid; grid-template-rows: auto 1fr; background: white; }
            .embed-title { padding: 8px 12px; color: #475569; font-size: 13px; border-bottom: 1px solid #e5e7eb; }
            .embed-graph { min-height: 320px; height: 100%; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
