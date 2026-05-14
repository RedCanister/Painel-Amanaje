# Painel Amanaje Example Catalog

The example catalog exposes five samples for every registry, assistant workflow,
runtime request, legacy Pydantic, and v2 schema object.

Scale tiers:

- `simple`: 0.01 MB declared workload
- `small`: 0.1 MB declared workload
- `medium`: 1 MB declared workload
- `large`: 10 MB declared workload
- `complex`: 100 MB declared workload

The 100 MB tier is represented as metadata, realistic shapes, artifact paths,
and API payload contracts. The catalog intentionally avoids committing large
binary fixtures.

API:

```text
GET /examples/catalog
GET /examples/catalog?include_payloads=false
GET /examples/catalog/{object_name}
```

Export a snapshot:

```powershell
python scripts/export_example_catalog.py
python scripts/export_example_catalog.py --without-payloads
```

The Assistant Management page includes an Examples tab that fetches the catalog
through the browser and summarizes object coverage, API routes, frontend routes,
workflow chains, and validation status.
