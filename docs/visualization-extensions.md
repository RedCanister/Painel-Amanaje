# Visualization Extensions

Painel Amanaje can load trusted local Dash dashboards inside the Visualization renderer. This is meant for personal dashboards, lab-specific inspection panels, and richer Python-native Plotly workflows that should live beside the standard plot gallery.

The feature is disabled by default.

## Runtime Flags

- `AMANAJE_DASH_EXTENSIONS_ENABLED=false`
  Enables local extension discovery when set to `true`.
- `AMANAJE_DASH_EXTENSIONS_DIR=/app/extensions`
  Directory scanned by the dashboard container.
- `AMANAJE_DASH_EXTENSIONS_ALLOWLIST=`
  Optional comma-separated extension ids. When present, only listed ids load.

The Docker Compose dashboard service already sets these defaults. Restart the dashboard container after changing them.

## Extension Layout

Each extension is a direct child directory of `dashboard/extensions`:

```text
dashboard/extensions/my_dashboard/
  manifest.json
  extension.py
```

`manifest.json`:

```json
{
  "id": "my_dashboard",
  "title": "My Dashboard",
  "description": "A local dashboard for one team workflow.",
  "version": "0.1.0",
  "author": "Team",
  "tags": ["local", "quality"],
  "entrypoint": "extension:register"
}
```

The manifest `id` must match the directory name and may only contain letters, numbers, dashes, and underscores.

## Python Contract

The entrypoint function receives the Dash app, an API client, and a namespaced context:

```python
from dash import Input, Output, dcc, html


def register(app, api_client, context):
    output_id = context.component_id("output")

    @app.callback(Output(output_id, "children"), Input(context.component_id("refresh"), "n_clicks"))
    def refresh(_clicks):
        payload = api_client.get("/production/history")
        return f"{len(payload.get('runs', []))} production runs"

    return html.Div([
        html.Button("Refresh", id=context.component_id("refresh")),
        html.Div(id=output_id),
    ])
```

Use `context.component_id("local-name")` for every Dash component id so multiple extensions cannot collide.

## Visualization Surface

FastAPI keeps `/visualization` as the user-facing page. The Custom Dashboards tab reads renderer metadata from `/extensions.json` and embeds selected dashboards from `/extensions/<extension_id>`.

If the renderer is unavailable or extensions are disabled, the native Visualization gallery, generator, object explorer, artifact browser, JSON tables, and fallback plot previews continue to work.

## Included Example

`dashboard/extensions/model_quality_overview` is an inactive sample. It has `"enabled": false`, so it only loads when extensions are enabled and the id is explicitly allowlisted:

```text
AMANAJE_DASH_EXTENSIONS_ENABLED=true
AMANAJE_DASH_EXTENSIONS_ALLOWLIST=model_quality_overview
```
