from __future__ import annotations

from dash import Input, Output, dcc, html


def register(app, api_client, context):
    model_select_id = context.component_id("model-select")
    output_id = context.component_id("summary")

    @app.callback(Output(output_id, "children"), Input(model_select_id, "value"))
    def update_model_summary(model_id):
        if not model_id:
            return html.Div("Select a model to inspect quality metrics.", className="muted")
        payload = api_client.get("/analysis/model", {"model_id": model_id})
        if not isinstance(payload, dict) or payload.get("status") == "error":
            return html.Div(payload.get("detail", "Model analysis is unavailable."), className="muted")
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
        plots = payload.get("plots") if isinstance(payload.get("plots"), list) else []
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Strong(str(key)),
                                html.Span(str(value), className="metric-value"),
                            ],
                            className="extension-metric",
                        )
                        for key, value in list(metrics.items())[:6]
                    ],
                    className="extension-metric-grid",
                ),
                html.Div(f"{len(plots)} related plot item(s) are available in the native Visualization gallery.", className="muted"),
            ]
        )

    models = api_client.get("/learningmodel/list")
    model_options = [
        {"label": row.get("name") or f"Model #{row.get('id')}", "value": str(row.get("id"))}
        for row in models
        if isinstance(row, dict) and row.get("id") is not None
    ] if isinstance(models, list) else []

    return {
        "layout": html.Div(
            [
                html.P("This sample stays inactive unless explicitly allowlisted. It demonstrates the local Python dashboard contract.", className="muted"),
                dcc.Dropdown(model_options, id=model_select_id, placeholder="Select a model"),
                html.Div(id=output_id, className="tab-body"),
            ],
            className="extension-example",
        ),
        "metadata": {
            "title": "Model Quality Overview",
            "description": "Example trusted dashboard for model metric summaries.",
        },
    }
