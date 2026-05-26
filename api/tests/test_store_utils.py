import asyncio
import json
from types import SimpleNamespace

import pandas as pd

import main_app
from app.utils import store_utils


def _payload(response):
    return json.loads(response.body.decode("utf-8"))


def test_dados_normalizers_map_catalog_detail_and_resources():
    catalog = store_utils.normalize_dados_catalog_item(
        {
            "id": "abc",
            "title": "Catalog title",
            "nome": "catalog-name",
            "catalogacao": "Environment",
            "nomeOrganizacao": "Org",
            "ultimaAtualizacaoDados": "2026-01-01",
            "isAtualizado": True,
        }
    )

    assert catalog["external_id"] == "abc"
    assert catalog["title"] == "Catalog title"
    assert catalog["category"] == "Environment"
    assert catalog["organization"] == "Org"
    assert catalog["is_current"] is True

    detail = store_utils.normalize_dados_detail(
        {
            "id": "abc",
            "titulo": "Detailed dataset",
            "nome": "detailed-dataset",
            "organizacao": "Org",
            "licenca": "CC-BY",
            "temas": [{"title": "Health", "name": "health"}],
            "tags": [{"id": "1", "name": "tag", "display_name": "Tag"}],
            "recursos": [
                {
                    "id": "resource-1",
                    "titulo": "CSV resource",
                    "formato": "CSV",
                    "link": "https://example.gov/data.csv",
                    "tipo": "DADOS",
                    "tamanho": 123,
                    "quantidadeDownloads": 9,
                }
            ],
        }
    )

    assert detail["external_id"] == "abc"
    assert detail["category"] == "Health"
    assert detail["resources"][0]["external_id"] == "resource-1"
    assert detail["resources"][0]["format"] == "CSV"


def test_builtin_dados_provider_uses_custom_header_auth(tmp_path):
    providers = store_utils.list_providers(tmp_path / "providers.json")
    dados_provider = next(provider for provider in providers if provider["id"] == "dados_gov_br")

    assert dados_provider["token_env_var"] == "AMANAJE_DADOS_GOV_TOKEN"
    assert dados_provider["auth"] == {"type": "header", "header_name": "chave-api-dados-abertos"}
    assert dados_provider["raw_token_returned"] is False


def test_provider_storage_redacts_token_and_reports_configured_env(monkeypatch, tmp_path):
    monkeypatch.setenv("STORE_TEST_TOKEN", "super-secret")
    providers_path = tmp_path / "providers.json"

    provider = store_utils.save_provider(
        {
            "name": "Test Provider",
            "provider_type": "generic_rest",
            "base_url": "https://example.gov",
            "token_env_var": "store-test-token",
            "auth": {"type": "bearer", "token": "do-not-save"},
            "mappings": {"catalog_path": "/catalog"},
        },
        providers_path,
    )

    saved_text = providers_path.read_text(encoding="utf-8")
    assert "super-secret" not in saved_text
    assert "do-not-save" not in saved_text
    assert "STORE_TEST_TOKEN" in saved_text
    assert provider["token_configured"] is True
    assert provider["raw_token_returned"] is False


def test_provider_default_and_delete_management(monkeypatch, tmp_path):
    providers_path = tmp_path / "providers.json"
    monkeypatch.setenv("STORE_PROVIDER_TOKEN", "configured")
    provider = store_utils.save_provider(
        {
            "name": "Managed Provider",
            "provider_type": "generic_rest",
            "base_url": "https://example.gov",
            "token_env_var": "STORE_PROVIDER_TOKEN",
            "mappings": {"catalog_path": "/catalog"},
        },
        providers_path,
    )

    providers = store_utils.list_providers(providers_path)
    managed = next(item for item in providers if item["id"] == provider["id"])
    assert managed["is_default"] is True

    built_in = store_utils.set_default_provider("dados_gov_br", providers_path)
    assert built_in["is_default"] is True
    providers = store_utils.list_providers(providers_path)
    assert next(item for item in providers if item["id"] == "dados_gov_br")["is_default"] is True

    deleted = store_utils.delete_provider(provider["id"], providers_path)
    assert deleted["id"] == provider["id"]
    assert all(item["id"] != provider["id"] for item in store_utils.list_providers(providers_path))


def test_builtin_provider_cannot_be_deleted(tmp_path):
    try:
        store_utils.delete_provider("dados_gov_br", tmp_path / "providers.json")
    except store_utils.StoreValidationError as exc:
        assert "cannot be deleted" in str(exc)
    else:
        raise AssertionError("Expected built-in provider deletion to fail.")


def test_saved_dados_provider_is_normalized_to_official_api_paths(monkeypatch, tmp_path):
    providers_path = tmp_path / "providers.json"
    provider = store_utils.save_provider(
        {
            "name": "Dados Abertos",
            "provider_type": "generic_rest",
            "base_url": "https://dados.gov.br",
            "token_env_var": "CHAVE_API",
            "auth": {"type": "header", "header_name": "chave-api-dados-abertos"},
            "mappings": {},
        },
        providers_path,
    )

    resolved = store_utils.resolve_provider(provider["id"], providers_path)

    assert resolved["provider_type"] == "dados_gov_br"
    assert resolved["auth"] == {"type": "header", "header_name": "chave-api-dados-abertos"}
    assert resolved["openapi_url"] == "https://dados.gov.br/v3/api-docs"


def test_dados_provider_uses_official_list_url_for_saved_alias(monkeypatch, tmp_path):
    providers_path = tmp_path / "providers.json"
    provider = store_utils.save_provider(
        {
            "name": "Dados Abertos",
            "provider_type": "generic_rest",
            "base_url": "https://dados.gov.br",
            "token_env_var": "CHAVE_API",
            "auth": {"type": "header", "header_name": "chave-api-dados-abertos"},
        },
        providers_path,
    )
    calls = {}

    async def fake_get_json(provider, url, *, params=None, auth_required=False):
        calls["url"] = url
        calls["params"] = params
        calls["auth_required"] = auth_required
        return [{"id": "dataset-1", "titulo": "Dataset 1"}]

    monkeypatch.setattr(store_utils, "_http_get_json", fake_get_json)

    result = asyncio.run(store_utils.list_catalog(store_utils.resolve_provider(provider["id"], providers_path), page=2))

    assert calls["url"] == "https://dados.gov.br/dados/api/publico/conjuntos-dados"
    assert calls["params"]["pagina"] == 2
    assert calls["auth_required"] is True
    assert result["items"][0]["external_id"] == "dataset-1"


def test_provider_rejects_token_value_as_token_env_var(tmp_path):
    try:
        store_utils.save_provider(
            {
                "name": "Token Value Provider",
                "provider_type": "generic_rest",
                "base_url": "https://example.gov",
                "token_env_var": "aaa.bbb.ccc",
            },
            tmp_path / "providers.json",
        )
    except store_utils.StoreValidationError as exc:
        assert "not the token value" in str(exc)
    else:
        raise AssertionError("Expected token value to be rejected as token_env_var.")


def test_url_safety_blocks_local_private_and_file_urls():
    for url in [
        "file:///tmp/data.csv",
        "http://localhost/data.csv",
        "http://127.0.0.1/data.csv",
        "http://10.0.0.5/data.csv",
        "https://example.gov/../secret.csv",
    ]:
        try:
            store_utils.validate_public_http_url(url)
        except store_utils.StoreValidationError:
            continue
        raise AssertionError(f"Expected URL to be blocked: {url}")


def test_store_network_helpers_report_missing_httpx(monkeypatch):
    monkeypatch.setattr(store_utils, "httpx", None)

    try:
        asyncio.run(
            store_utils._http_get_json(
                {"provider_type": "generic_rest", "base_url": "https://example.gov"},
                "https://example.gov/catalog",
            )
        )
    except store_utils.StoreValidationError as exc:
        assert exc.status_code == 503
        assert "httpx is required" in str(exc)
        assert "rebuild the API image" in str(exc)
    else:
        raise AssertionError("Expected missing httpx to be reported as a StoreValidationError.")


def test_preview_resource_limits_rows_and_profiles_columns(monkeypatch):
    async def fake_download_resource(provider, dataset_external_id, resource_external_id):
        return store_utils.DownloadedResource(
            content=b"a,b\n1,x\n2,y\n3,z\n4,w\n5,q\n6,r\n",
            filename="sample.csv",
            resource={"external_id": resource_external_id, "title": "Sample", "url": "https://example.gov/sample.csv"},
            detail={"external_id": dataset_external_id, "title": "Dataset"},
            provider=dict(provider),
        )

    monkeypatch.setattr(store_utils, "download_resource", fake_download_resource)

    result = asyncio.run(
        store_utils.preview_resource(
            {"id": "generic", "provider_type": "generic_rest", "base_url": "https://example.gov"},
            "dataset-1",
            "resource-1",
        )
    )

    assert result["status"] == "ok"
    assert len(result["preview_rows"]) == 5
    assert result["row_count"] == 6
    assert result["columns"] == ["a", "b"]
    assert result["column_explorer"]


def test_preview_resource_reports_unsupported_resources(monkeypatch):
    async def fake_download_resource(provider, dataset_external_id, resource_external_id):
        return store_utils.DownloadedResource(
            content=b"not a supported table",
            filename="sample.bin",
            resource={"external_id": resource_external_id, "title": "Binary"},
            detail={"external_id": dataset_external_id},
            provider=dict(provider),
        )

    monkeypatch.setattr(store_utils, "download_resource", fake_download_resource)

    try:
        asyncio.run(store_utils.preview_resource({"id": "generic"}, "dataset-1", "resource-1"))
    except ValueError as exc:
        assert "Unsupported dataset file type" in str(exc)
    else:
        raise AssertionError("Expected unsupported resource preview to fail.")


def test_dados_view_resource_uses_direct_download_url(monkeypatch):
    async def fake_get_dataset_detail(provider, external_id):
        return {
            "external_id": external_id,
            "resources": [
                {
                    "external_id": "resource-1",
                    "title": "CSV Viewer",
                    "format": "CSV",
                    "url": "https://www.gov.br/example/data.csv/view",
                }
            ],
        }

    async def fake_get_bytes(provider, url, *, max_bytes, auth_required):
        assert url == "https://www.gov.br/example/data.csv/@@download/file"
        assert auth_required is True
        return b"a,b\n1,x\n", {"final_url": url, "content_type": "text/csv", "downloaded_bytes": 8}

    monkeypatch.setattr(store_utils, "get_dataset_detail", fake_get_dataset_detail)
    monkeypatch.setattr(store_utils, "_http_get_bytes", fake_get_bytes)

    result = asyncio.run(
        store_utils.preview_resource(
            {"id": "dados_abertos", "provider_type": "dados_gov_br"},
            "dataset-1",
            "resource-1",
        )
    )

    assert result["status"] == "ok"
    assert result["resource"]["url"].endswith("/@@download/file")
    assert result["preview_rows"] == [{"a": 1, "b": "x"}]


def test_csc_format_alias_is_parsed_as_csv(monkeypatch):
    async def fake_get_dataset_detail(provider, external_id):
        return {
            "external_id": external_id,
            "resources": [
                {
                    "external_id": "resource-1",
                    "title": "CSV resource with typo",
                    "format": "csc",
                    "url": "https://example.gov/download",
                }
            ],
        }

    async def fake_get_bytes(provider, url, *, max_bytes, auth_required):
        assert url == "https://example.gov/download"
        assert auth_required is False
        return b"a,b\n1,x\n", {"final_url": url, "content_type": "text/csv", "downloaded_bytes": 8}

    monkeypatch.setattr(store_utils, "get_dataset_detail", fake_get_dataset_detail)
    monkeypatch.setattr(store_utils, "_http_get_bytes", fake_get_bytes)

    result = asyncio.run(
        store_utils.preview_resource(
            {"id": "generic", "provider_type": "generic_rest"},
            "dataset-1",
            "resource-1",
        )
    )

    assert result["status"] == "ok"
    assert result["resource"]["file_extension"] == ".csv"
    assert result["preview_rows"] == [{"a": 1, "b": "x"}]


def test_pdf_resources_are_rejected_before_download(monkeypatch):
    called = False

    async def fake_get_dataset_detail(provider, external_id):
        return {
            "external_id": external_id,
            "resources": [
                {
                    "external_id": "resource-1",
                    "title": "PDF resource",
                    "format": "PDF",
                    "url": "https://www.gov.br/example/report.pdf/view",
                }
            ],
        }

    async def fake_get_bytes(*_args, **_kwargs):
        nonlocal called
        called = True
        return b"", {}

    monkeypatch.setattr(store_utils, "get_dataset_detail", fake_get_dataset_detail)
    monkeypatch.setattr(store_utils, "_http_get_bytes", fake_get_bytes)

    try:
        asyncio.run(
            store_utils.preview_resource(
                {"id": "dados_abertos", "provider_type": "dados_gov_br"},
                "dataset-1",
                "resource-1",
            )
        )
    except store_utils.StoreValidationError as exc:
        assert exc.status_code == 415
        assert "PDF" in str(exc)
        assert called is False
    else:
        raise AssertionError("Expected PDF resource preview to be rejected.")


def test_html_downloads_are_reported_as_unsupported_preview(monkeypatch):
    async def fake_download_resource(provider, dataset_external_id, resource_external_id):
        return store_utils.DownloadedResource(
            content=b"<!doctype html><html><body>Viewer</body></html>",
            filename="sample.csv",
            resource={"external_id": resource_external_id, "title": "Viewer", "url": "https://example.gov/sample.csv/view"},
            detail={"external_id": dataset_external_id},
            provider=dict(provider),
        )

    monkeypatch.setattr(store_utils, "download_resource", fake_download_resource)

    try:
        asyncio.run(store_utils.preview_resource({"id": "generic"}, "dataset-1", "resource-1"))
    except store_utils.StoreValidationError as exc:
        assert exc.status_code == 415
        assert "HTML/XML page" in str(exc)
    else:
        raise AssertionError("Expected HTML viewer response to be rejected.")


def test_store_materialize_route_creates_dataset_model(monkeypatch, tmp_path):
    output_path = tmp_path / "store_snapshot.csv"
    output_path.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
    dataframe = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    captured = {}

    class FakeRequest:
        async def json(self):
            return {
                "provider_id": "generic",
                "dataset_external_id": "dataset-1",
                "resource_external_id": "resource-1",
                "name": "store_dataset",
                "description": "Stored from test",
                "live": {"enabled": False},
            }

    async def fake_create_entry(db, orm, payload):
        captured["create_payload"] = payload
        return SimpleNamespace(id=42, **payload)

    async def fake_update_entry(db, orm, item_id, payload):
        captured["update_payload"] = payload
        base = dict(captured["create_payload"])
        base.update(payload)
        base.pop("id", None)
        return SimpleNamespace(id=item_id, **base)

    async def fake_materialize(provider, dataset_external_id, resource_external_id, *, dataset_name, dataset_dir):
        return {
            "output_path": output_path,
            "dataframe": dataframe,
            "parser_report": {"loader": "csv"},
            "provider": {"id": "generic"},
            "dataset": {"external_id": dataset_external_id, "title": "Dataset"},
            "resource": {"external_id": resource_external_id, "title": "Resource"},
            "connection_string": "store://generic/dataset-1/resource-1",
            "preview_rows": dataframe.head(5).to_dict(orient="records"),
            "columns": ["a", "b"],
        }

    monkeypatch.setattr(main_app.store_utils, "resolve_provider", lambda provider_id, providers_path: {"id": provider_id, "provider_type": "generic_rest"})
    monkeypatch.setattr(main_app.store_utils, "materialize_resource_files", fake_materialize)
    monkeypatch.setattr(main_app, "create_entry", fake_create_entry)
    monkeypatch.setattr(main_app, "update_entry", fake_update_entry)
    monkeypatch.setattr(main_app, "STORE_MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(main_app, "STORE_DATASET_DIR", tmp_path / "datasets")

    response = asyncio.run(main_app.store_materialize(FakeRequest(), db=object()))
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["dataset"]["id"] == 42
    assert captured["create_payload"]["connection_string"] == "store://generic/dataset-1/resource-1"
    assert captured["create_payload"]["shape"] == [2, 2]
    assert captured["create_payload"]["features_list"] == ["a", "b"]
    assert "manifest_path" in captured["update_payload"]["history"][-1]


def test_store_refresh_route_replaces_snapshot_and_updates_metadata(monkeypatch, tmp_path):
    output_path = tmp_path / "store_snapshot.csv"
    output_path.write_text("a,b\nold,x\n", encoding="utf-8")
    refreshed_dataframe = pd.DataFrame({"a": [10, 20, 30], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]})
    dataset = SimpleNamespace(
        id=55,
        name="live_store_dataset",
        path=str(output_path),
        connection_string="store://generic/dataset-1/resource-1",
        history=[{"operation": "store_materialize"}],
    )
    captured = {}

    class FakeRequest:
        pass

    async def fake_read(db, item_id):
        return dataset if item_id == 55 else None

    async def fake_refresh(provider, dataset_external_id, resource_external_id, *, output_path):
        refreshed_dataframe.to_csv(output_path, index=False)
        return {
            "output_path": output_path,
            "dataframe": refreshed_dataframe,
            "parser_report": {"loader": "csv"},
            "provider": {"id": "generic"},
            "dataset": {"external_id": dataset_external_id},
            "resource": {"external_id": resource_external_id},
            "preview_rows": refreshed_dataframe.head(5).to_dict(orient="records"),
            "columns": ["a", "b", "c"],
        }

    async def fake_update_entry(db, orm, item_id, payload):
        captured["update_payload"] = payload
        base = {**dataset.__dict__, **payload}
        base.pop("id", None)
        return SimpleNamespace(id=item_id, **base)

    monkeypatch.setattr(main_app.DatasetModel, "read", fake_read)
    monkeypatch.setattr(main_app.store_utils, "resolve_provider", lambda provider_id, providers_path: {"id": provider_id, "provider_type": "generic_rest"})
    monkeypatch.setattr(main_app.store_utils, "refresh_resource_file", fake_refresh)
    monkeypatch.setattr(main_app, "update_entry", fake_update_entry)
    monkeypatch.setattr(main_app, "STORE_MANIFEST_DIR", tmp_path / "manifests")

    response = asyncio.run(main_app.store_refresh_dataset(55, FakeRequest(), db=object()))
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert output_path.read_text(encoding="utf-8").startswith("a,b,c")
    assert captured["update_payload"]["shape"] == [3, 3]
    assert captured["update_payload"]["features_list"] == ["a", "b", "c"]
    assert captured["update_payload"]["history"][-1]["operation"] == "store_refresh"
