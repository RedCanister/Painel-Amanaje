from __future__ import annotations

import ipaddress
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import quote, unquote, urlencode, urljoin, urlparse, urlunparse

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - exercised by dependency-degraded tests
    httpx = None
import pandas as pd

from app.utils.io import save_json
from app.utils.tabular_utils import build_column_explorer, load_tabular_from_bytes


DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
BUILTIN_DADOS_PROVIDER_ID = "dados_gov_br"
BUILTIN_DADOS_TOKEN_ENV = "AMANAJE_DADOS_GOV_TOKEN"
SUPPORTED_PREVIEW_FORMATS = {
    "csc": ".csv",
    "csv": ".csv",
    "tsv": ".tsv",
    "txt": ".txt",
    "json": ".json",
    "jsonl": ".jsonl",
    "parquet": ".parquet",
    "xlsx": ".xlsx",
    "xls": ".xls",
}
UNSUPPORTED_RESOURCE_FORMATS = {
    "doc",
    "docx",
    "html",
    "htm",
    "pdf",
    "ppt",
    "pptx",
    "rar",
    "zip",
}


class StoreValidationError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400, errors: Optional[list[dict[str, Any]]] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.errors = errors or []


def _require_httpx():
    if httpx is None:
        raise StoreValidationError(
            "httpx is required for Store network operations. Install api/requirements.txt and rebuild the API image.",
            status_code=503,
        )
    return httpx


@dataclass
class DownloadedResource:
    content: bytes
    filename: str
    resource: dict[str, Any]
    detail: dict[str, Any]
    provider: dict[str, Any]


def _slugify(value: str, *, fallback: str = "store") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    return slug[:80] or fallback


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_provider_id(value: str) -> str:
    return _slugify(value, fallback="provider")


def _normalize_token_env_var(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    looks_like_jwt = len(text.split(".")) == 3 and all(part.strip() for part in text.split("."))
    looks_like_bearer = text.lower().startswith("bearer ")
    if looks_like_jwt or looks_like_bearer or len(text) > 128:
        raise StoreValidationError(
            "token_env_var must be the name of an environment variable, not the token value. "
            "Set the token in AMANAJE_DADOS_GOV_TOKEN and use that name here."
        )
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").upper()
    if not normalized or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", normalized):
        raise StoreValidationError("token_env_var must be an environment variable name such as AMANAJE_DADOS_GOV_TOKEN.")
    return normalized


def _builtin_dados_provider() -> dict[str, Any]:
    return {
        "id": BUILTIN_DADOS_PROVIDER_ID,
        "name": "dados.gov.br",
        "provider_type": "dados_gov_br",
        "base_url": "https://dados.gov.br",
        "openapi_url": "https://dados.gov.br/v3/api-docs",
        "token_env_var": BUILTIN_DADOS_TOKEN_ENV,
        "auth": {"type": "header", "header_name": "chave-api-dados-abertos"},
        "mappings": {},
        "built_in": True,
        "created_at": None,
        "updated_at": None,
    }


def _is_dados_gov_url(value: Any) -> bool:
    try:
        host = urlparse(str(value or "")).hostname
    except Exception:
        return False
    return str(host or "").strip().lower() == "dados.gov.br"


def _looks_like_dados_provider(provider: Mapping[str, Any]) -> bool:
    provider_type = str(provider.get("provider_type") or "").strip().lower()
    if provider_type == "dados_gov_br":
        return True
    auth = provider.get("auth") if isinstance(provider.get("auth"), Mapping) else {}
    header_name = str(auth.get("header_name") or "").strip().lower()
    return _is_dados_gov_url(provider.get("base_url")) and header_name == "chave-api-dados-abertos"


def _normalize_provider_kind(provider: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(provider)
    if not _looks_like_dados_provider(normalized):
        return normalized
    auth = dict(normalized.get("auth") or {})
    normalized["provider_type"] = "dados_gov_br"
    normalized["base_url"] = str(normalized.get("base_url") or "https://dados.gov.br").rstrip("/")
    normalized["openapi_url"] = normalized.get("openapi_url") or "https://dados.gov.br/v3/api-docs"
    normalized["auth"] = {
        "type": "header",
        "header_name": str(auth.get("header_name") or "chave-api-dados-abertos"),
    }
    return normalized


def load_saved_providers(providers_path: Path) -> dict[str, dict[str, Any]]:
    if not providers_path.exists():
        return {}
    try:
        payload = json.loads(providers_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    providers = payload.get("providers") if isinstance(payload, Mapping) else payload
    if not isinstance(providers, Mapping):
        return {}
    return {
        _normalize_provider_id(str(provider_id)): dict(provider)
        for provider_id, provider in providers.items()
        if isinstance(provider, Mapping)
    }


def _load_provider_document(providers_path: Path) -> dict[str, Any]:
    if not providers_path.exists():
        return {}
    try:
        payload = json.loads(providers_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {"providers": payload}


def load_default_provider_id(providers_path: Path) -> str | None:
    payload = _load_provider_document(providers_path)
    value = str(payload.get("default_provider_id") or "").strip()
    return _normalize_provider_id(value) if value else None


def _available_provider_ids(providers: Mapping[str, Mapping[str, Any]]) -> set[str]:
    return {BUILTIN_DADOS_PROVIDER_ID, *{_normalize_provider_id(str(provider_id)) for provider_id in providers}}


def _coerce_default_provider_id(default_provider_id: str | None, providers: Mapping[str, Mapping[str, Any]]) -> str:
    available_ids = _available_provider_ids(providers)
    normalized = _normalize_provider_id(default_provider_id or "") if default_provider_id else ""
    if normalized and normalized in available_ids:
        return normalized
    for provider_id, provider in providers.items():
        token_env_var = str(provider.get("token_env_var") or "").strip()
        if token_env_var and os.getenv(token_env_var):
            return _normalize_provider_id(str(provider_id))
    return BUILTIN_DADOS_PROVIDER_ID


def save_saved_providers(
    providers_path: Path,
    providers: Mapping[str, Mapping[str, Any]],
    *,
    default_provider_id: str | None = None,
) -> None:
    providers_path.parent.mkdir(parents=True, exist_ok=True)
    stored_default = default_provider_id if default_provider_id is not None else load_default_provider_id(providers_path)
    resolved_default = _coerce_default_provider_id(stored_default, providers)
    payload = {
        "providers": {str(key): _sanitize_provider_for_storage(dict(value)) for key, value in providers.items()},
        "default_provider_id": resolved_default,
        "updated_at": _utc_now(),
    }
    save_json(payload, providers_path)


def _sanitize_provider_for_storage(provider: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(provider)
    sanitized.pop("token", None)
    sanitized.pop("api_key", None)
    sanitized.pop("password", None)
    sanitized.pop("token_configured", None)
    sanitized.pop("raw_token_returned", None)
    auth = sanitized.get("auth")
    if isinstance(auth, Mapping):
        auth = dict(auth)
        for key in list(auth):
            if any(marker in str(key).lower() for marker in ("token", "secret", "password", "api_key")):
                auth.pop(key, None)
        sanitized["auth"] = auth
    return sanitized


def _redact_provider(provider: Mapping[str, Any]) -> dict[str, Any]:
    redacted = _sanitize_provider_for_storage(dict(provider))
    token_env_var = str(redacted.get("token_env_var") or "").strip()
    redacted["token_configured"] = bool(token_env_var and os.getenv(token_env_var))
    redacted["raw_token_returned"] = False
    return redacted


def redact_provider(provider: Mapping[str, Any]) -> dict[str, Any]:
    return _redact_provider(provider)


def list_providers(providers_path: Path) -> list[dict[str, Any]]:
    saved_providers = {
        provider_id: _normalize_provider_kind(provider)
        for provider_id, provider in load_saved_providers(providers_path).items()
    }
    default_provider_id = _coerce_default_provider_id(load_default_provider_id(providers_path), saved_providers)
    providers = {BUILTIN_DADOS_PROVIDER_ID: _builtin_dados_provider()}
    providers.update(saved_providers)
    redacted = []
    for provider_id, provider in providers.items():
        entry = _redact_provider(provider)
        entry["is_default"] = provider_id == default_provider_id
        redacted.append(entry)
    redacted.sort(key=lambda provider: (not bool(provider.get("is_default")), bool(provider.get("built_in")), str(provider.get("name") or "")))
    return redacted


def resolve_provider(provider_id: str, providers_path: Path) -> dict[str, Any]:
    normalized = _normalize_provider_id(provider_id)
    if normalized == BUILTIN_DADOS_PROVIDER_ID:
        return _builtin_dados_provider()
    providers = load_saved_providers(providers_path)
    provider = providers.get(normalized)
    if not provider:
        raise StoreValidationError(f"Store provider '{provider_id}' was not found.", status_code=404)
    provider = dict(provider)
    provider["id"] = normalized
    return _normalize_provider_kind(provider)


def save_provider(payload: Mapping[str, Any], providers_path: Path) -> dict[str, Any]:
    provider_type = str(payload.get("provider_type") or payload.get("providerType") or "generic_rest").strip().lower()
    if provider_type not in {"generic_rest", "openapi", "dados_gov_br"}:
        raise StoreValidationError("provider_type must be one of: generic_rest, openapi, dados_gov_br.")
    if provider_type == "dados_gov_br":
        provider_type = "dados_gov_br"

    name = str(payload.get("name") or "").strip()
    if not name:
        raise StoreValidationError("Provider name is required.")
    provider_id = _normalize_provider_id(str(payload.get("id") or name))
    if provider_id == BUILTIN_DADOS_PROVIDER_ID:
        raise StoreValidationError("The built-in dados.gov.br provider cannot be overwritten.")

    base_url = str(payload.get("base_url") or payload.get("baseUrl") or "").strip()
    if not base_url:
        raise StoreValidationError("Provider base_url is required.")
    validate_public_http_url(base_url, allow_path=True)

    openapi_url = str(payload.get("openapi_url") or payload.get("openapiUrl") or "").strip()
    if openapi_url:
        validate_public_http_url(openapi_url, allow_path=True)

    mappings = _coerce_mapping(payload.get("mappings") or payload.get("mapping") or {})
    auth = _coerce_mapping(payload.get("auth") or {})
    token_env_var = _normalize_token_env_var(payload.get("token_env_var") or payload.get("tokenEnvVar") or auth.get("token_env_var"))

    now = _utc_now()
    providers = load_saved_providers(providers_path)
    previous = providers.get(provider_id, {})
    provider = {
        "id": provider_id,
        "name": name,
        "provider_type": provider_type,
        "base_url": base_url.rstrip("/"),
        "openapi_url": openapi_url or None,
        "token_env_var": token_env_var,
        "auth": _sanitize_provider_for_storage({"auth": auth}).get("auth", {}),
        "mappings": mappings,
        "built_in": False,
        "created_at": previous.get("created_at") or now,
        "updated_at": now,
    }
    provider = _normalize_provider_kind(provider)
    providers[provider_id] = provider
    save_saved_providers(providers_path, providers)
    return _redact_provider(provider)


def set_default_provider(provider_id: str, providers_path: Path) -> dict[str, Any]:
    normalized = _normalize_provider_id(provider_id)
    if normalized == BUILTIN_DADOS_PROVIDER_ID:
        providers = load_saved_providers(providers_path)
        save_saved_providers(providers_path, providers, default_provider_id=BUILTIN_DADOS_PROVIDER_ID)
        provider = _builtin_dados_provider()
        result = _redact_provider(provider)
        result["is_default"] = True
        return result

    providers = load_saved_providers(providers_path)
    provider = providers.get(normalized)
    if not provider:
        raise StoreValidationError(f"Store provider '{provider_id}' was not found.", status_code=404)
    save_saved_providers(providers_path, providers, default_provider_id=normalized)
    result = _redact_provider(_normalize_provider_kind({**provider, "id": normalized}))
    result["is_default"] = True
    return result


def delete_provider(provider_id: str, providers_path: Path) -> dict[str, Any]:
    normalized = _normalize_provider_id(provider_id)
    if normalized == BUILTIN_DADOS_PROVIDER_ID:
        raise StoreValidationError("The built-in dados.gov.br provider cannot be deleted.", status_code=400)
    providers = load_saved_providers(providers_path)
    if normalized not in providers:
        raise StoreValidationError(f"Store provider '{provider_id}' was not found.", status_code=404)
    deleted = _redact_provider(_normalize_provider_kind({**providers.pop(normalized), "id": normalized}))
    current_default = load_default_provider_id(providers_path)
    next_default = None if current_default == normalized else current_default
    save_saved_providers(providers_path, providers, default_provider_id=next_default)
    return deleted


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise StoreValidationError(f"Invalid mapping JSON: {exc}") from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def validate_public_http_url(url: str, *, allow_path: bool = True) -> None:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise StoreValidationError("Only http and https URLs are allowed.")
    if not parsed.hostname:
        raise StoreValidationError("URL hostname is required.")
    host = parsed.hostname.strip().lower()
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".localhost") or host.endswith(".local"):
        raise StoreValidationError("Localhost and local network URLs are not allowed.")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved):
        raise StoreValidationError("Private, loopback, link-local, multicast, and reserved IP URLs are not allowed.")
    if not allow_path and (parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise StoreValidationError("URL must not include a path, query string, or fragment.")
    if ".." in Path(parsed.path or "/").parts:
        raise StoreValidationError("URL paths must not contain parent-directory traversal.")


def _join_provider_url(provider: Mapping[str, Any], path: str) -> str:
    base_url = str(provider.get("base_url") or "").rstrip("/") + "/"
    if re.match(r"^https?://", str(path or ""), flags=re.IGNORECASE):
        url = str(path)
    else:
        relative = str(path or "").lstrip("/")
        url = urljoin(base_url, relative)
    validate_public_http_url(url, allow_path=True)
    return url


def _token_for_provider(provider: Mapping[str, Any], *, required: bool = False) -> Optional[str]:
    token_env_var = str(provider.get("token_env_var") or "").strip()
    token = os.getenv(token_env_var) if token_env_var else None
    if required and not token:
        label = token_env_var or "configured token environment variable"
        raise StoreValidationError(
            f"{label} must be configured before this provider can be queried.",
            status_code=401,
        )
    return token


def _provider_auth(provider: Mapping[str, Any], *, required: bool = False) -> tuple[dict[str, str], dict[str, str]]:
    auth = dict(provider.get("auth") or {})
    auth_type = str(auth.get("type") or ("bearer" if provider.get("token_env_var") else "none")).strip().lower()
    token = _token_for_provider(provider, required=required or auth_type in {"bearer", "header", "query"} and bool(provider.get("token_env_var")))
    headers: dict[str, str] = {}
    params: dict[str, str] = {}
    if not token:
        return headers, params
    if auth_type == "bearer":
        headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "header":
        headers[str(auth.get("header_name") or "Authorization")] = token
    elif auth_type == "query":
        params[str(auth.get("query_param") or "token")] = token
    return headers, params


def _with_query_params(url: str, params: Mapping[str, Any]) -> str:
    clean_params = {str(key): value for key, value in params.items() if value not in (None, "")}
    if not clean_params:
        return url
    parsed = urlparse(url)
    existing = parsed.query
    extra = urlencode(clean_params, doseq=True)
    query = f"{existing}&{extra}" if existing else extra
    return urlunparse(parsed._replace(query=query))


async def _http_get_json(provider: Mapping[str, Any], url: str, *, params: Optional[Mapping[str, Any]] = None, auth_required: bool = False) -> Any:
    headers, auth_params = _provider_auth(provider, required=auth_required)
    request_url = _with_query_params(url, {**auth_params, **dict(params or {})})
    timeout = float(provider.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    client_module = _require_httpx()
    try:
        async with client_module.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(request_url, headers=headers)
    except client_module.HTTPError as exc:
        raise StoreValidationError(f"Unable to reach store provider: {exc}", status_code=502) from exc
    if response.status_code == 401:
        token_env_var = provider.get("token_env_var") or "the configured token environment variable"
        raise StoreValidationError(f"{token_env_var} must be configured with a valid token for this provider.", status_code=401)
    if response.status_code >= 400:
        raise StoreValidationError(f"Store provider returned HTTP {response.status_code}.", status_code=response.status_code)
    try:
        return response.json()
    except ValueError as exc:
        raise StoreValidationError("Store provider did not return JSON.", status_code=502) from exc


async def _http_get_bytes(
    provider: Mapping[str, Any],
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    auth_required: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    headers, auth_params = _provider_auth(provider, required=auth_required)
    request_url = _with_query_params(url, auth_params)
    timeout = float(provider.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    client_module = _require_httpx()
    try:
        async with client_module.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(request_url, headers=headers)
    except client_module.HTTPError as exc:
        raise StoreValidationError(f"Unable to download store resource: {exc}", status_code=502) from exc
    if response.status_code == 401:
        token_env_var = provider.get("token_env_var") or "the configured token environment variable"
        raise StoreValidationError(f"{token_env_var} must be configured with a valid token for this provider.", status_code=401)
    if response.status_code >= 400:
        raise StoreValidationError(f"Store resource returned HTTP {response.status_code}.", status_code=response.status_code)
    content = response.content
    if len(content) > max_bytes:
        raise StoreValidationError(f"Store resource exceeds the {max_bytes} byte download limit.", status_code=413)
    metadata = {
        "content_type": response.headers.get("content-type"),
        "content_length": response.headers.get("content-length"),
        "final_url": str(response.url),
        "downloaded_bytes": len(content),
    }
    return content, metadata


async def _http_head(provider: Mapping[str, Any], url: str, *, auth_required: bool = False) -> dict[str, Any]:
    headers, auth_params = _provider_auth(provider, required=auth_required)
    request_url = _with_query_params(url, auth_params)
    timeout = float(provider.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    client_module = _require_httpx()
    try:
        async with client_module.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.head(request_url, headers=headers)
            if response.status_code in {403, 405, 501}:
                range_headers = {**headers, "Range": "bytes=0-0"}
                async with client.stream("GET", request_url, headers=range_headers) as get_response:
                    if get_response.status_code < 400:
                        return {
                            "status_code": get_response.status_code,
                            "content_type": get_response.headers.get("content-type"),
                            "content_length": get_response.headers.get("content-length"),
                            "final_url": str(get_response.url),
                            "method": "GET",
                            "range_probe": True,
                        }
    except client_module.HTTPError as exc:
        raise StoreValidationError(f"Unable to validate store resource: {exc}", status_code=502) from exc
    if response.status_code == 401:
        token_env_var = provider.get("token_env_var") or "the configured token environment variable"
        raise StoreValidationError(f"{token_env_var} must be configured with a valid token for this provider.", status_code=401)
    if response.status_code >= 400 and response.status_code not in {405, 501}:
        raise StoreValidationError(f"Store resource validation returned HTTP {response.status_code}.", status_code=response.status_code)
    return {
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "content_length": response.headers.get("content-length"),
        "final_url": str(response.url),
    }


def _nested_get(payload: Any, path: Any, default: Any = None) -> Any:
    if not path:
        return payload
    if isinstance(path, (list, tuple)):
        parts = list(path)
    else:
        text = str(path)
        if text in {"$", "."}:
            return payload
        parts = [part for part in text.replace("[", ".").replace("]", "").split(".") if part]
    current = payload
    for part in parts:
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return default
        else:
            return default
        if current is None:
            return default
    return current


def _string_from_mapping(payload: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def normalize_dados_catalog_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "external_id": _string_from_mapping(item, "id"),
        "title": _string_from_mapping(item, "title", "titulo", "nome"),
        "name": _string_from_mapping(item, "nome", "title"),
        "category": _string_from_mapping(item, "catalogacao"),
        "organization": _string_from_mapping(item, "nomeOrganizacao", "organizacao"),
        "updated_at": _string_from_mapping(item, "ultimaAtualizacaoDados", "ultimaAlteracaoMetadados"),
        "is_current": item.get("isAtualizado"),
        "raw": dict(item),
    }


def normalize_dados_resource(resource: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "external_id": _string_from_mapping(resource, "id", "link", "url"),
        "title": _string_from_mapping(resource, "titulo", "name", "nomeArquivo", "id"),
        "description": _string_from_mapping(resource, "descricao", "description"),
        "url": _string_from_mapping(resource, "link", "url"),
        "format": _string_from_mapping(resource, "formato", "format"),
        "type": _string_from_mapping(resource, "tipo"),
        "size": resource.get("tamanho", resource.get("size")),
        "download_count": resource.get("quantidadeDownloads"),
        "file_name": _string_from_mapping(resource, "nomeArquivo"),
        "order": resource.get("numOrdem"),
        "updated_at": _string_from_mapping(resource, "dataUltimaAtualizacaoArquivo", "last_modified", "metadata_modified"),
        "raw": dict(resource),
    }


def normalize_dados_detail(detail: Mapping[str, Any]) -> dict[str, Any]:
    themes = detail.get("temas") if isinstance(detail.get("temas"), list) else []
    tags = detail.get("tags") if isinstance(detail.get("tags"), list) else []
    resources = detail.get("recursos") if isinstance(detail.get("recursos"), list) else []
    categories = [
        _string_from_mapping(theme, "title", "name")
        for theme in themes
        if isinstance(theme, Mapping)
    ]
    return {
        "external_id": _string_from_mapping(detail, "id"),
        "title": _string_from_mapping(detail, "titulo", "title", "nome"),
        "name": _string_from_mapping(detail, "nome", "titulo"),
        "description": _string_from_mapping(detail, "descricao"),
        "category": ", ".join(category for category in categories if category),
        "categories": [category for category in categories if category],
        "organization": _string_from_mapping(detail, "organizacao"),
        "license": _string_from_mapping(detail, "licenca"),
        "periodicity": _string_from_mapping(detail, "periodicidade"),
        "visibility": _string_from_mapping(detail, "visibilidade"),
        "updated_at": _string_from_mapping(detail, "dataUltimaAtualizacaoArquivo", "dataUltimaAtualizacaoMetadados"),
        "resources": [normalize_dados_resource(resource) for resource in resources if isinstance(resource, Mapping)],
        "tags": [_string_from_mapping(tag, "display_name", "name", "id") for tag in tags if isinstance(tag, Mapping)],
        "raw": dict(detail),
    }


def _generic_field(item: Mapping[str, Any], mappings: Mapping[str, Any], mapping_key: str, *fallback_keys: str) -> Any:
    configured = mappings.get(mapping_key)
    if configured:
        return _nested_get(item, configured)
    for key in fallback_keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_generic_catalog_item(item: Mapping[str, Any], mappings: Mapping[str, Any]) -> dict[str, Any]:
    external_id = _generic_field(item, mappings, "item_id", "id", "name", "slug", "url")
    title = _generic_field(item, mappings, "title", "title", "titulo", "name")
    return {
        "external_id": str(external_id or title or uuid.uuid4().hex),
        "title": str(title or external_id or "Untitled dataset"),
        "name": str(_generic_field(item, mappings, "name", "name", "slug") or title or external_id or ""),
        "category": str(_generic_field(item, mappings, "category", "category", "catalogacao", "theme") or ""),
        "organization": str(_generic_field(item, mappings, "organization", "organization", "org", "publisher") or ""),
        "updated_at": str(_generic_field(item, mappings, "updated_at", "updated_at", "modified", "last_modified") or ""),
        "is_current": _generic_field(item, mappings, "is_current", "is_current", "isAtualizado"),
        "raw": dict(item),
    }


def normalize_generic_resource(resource: Mapping[str, Any], mappings: Mapping[str, Any]) -> dict[str, Any]:
    external_id = _generic_field(resource, mappings, "resource_id", "id", "url", "link")
    title = _generic_field(resource, mappings, "resource_title", "title", "name", "titulo", "nomeArquivo")
    return {
        "external_id": str(external_id or title or uuid.uuid4().hex),
        "title": str(title or external_id or "Resource"),
        "description": str(_generic_field(resource, mappings, "resource_description", "description", "descricao") or ""),
        "url": str(_generic_field(resource, mappings, "resource_url", "url", "link", "download_url") or ""),
        "format": str(_generic_field(resource, mappings, "resource_format", "format", "formato") or ""),
        "type": str(_generic_field(resource, mappings, "resource_type", "type", "tipo") or ""),
        "size": _generic_field(resource, mappings, "resource_size", "size", "tamanho"),
        "download_count": _generic_field(resource, mappings, "resource_downloads", "downloads", "quantidadeDownloads"),
        "file_name": str(_generic_field(resource, mappings, "resource_file_name", "file_name", "nomeArquivo") or ""),
        "updated_at": str(_generic_field(resource, mappings, "resource_updated_at", "updated_at", "last_modified") or ""),
        "raw": dict(resource),
    }


def normalize_generic_detail(detail: Mapping[str, Any], mappings: Mapping[str, Any], *, fallback_item: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    base = normalize_generic_catalog_item(dict(fallback_item or detail), mappings)
    resources_value = _nested_get(detail, mappings.get("resources_path") or "resources", [])
    if not isinstance(resources_value, list):
        resources_value = []
    resources = [
        normalize_generic_resource(resource, mappings)
        for resource in resources_value
        if isinstance(resource, Mapping)
    ]
    base.update(
        {
            "description": str(_generic_field(detail, mappings, "description", "description", "descricao") or ""),
            "resources": resources,
            "raw": dict(detail),
        }
    )
    return base


async def list_catalog(provider: Mapping[str, Any], *, query: str = "", category: str = "", page: int = 1) -> dict[str, Any]:
    provider_type = str(provider.get("provider_type") or "").lower()
    page = max(int(page or 1), 1)
    if provider_type == "dados_gov_br":
        url = _join_provider_url(provider, "/dados/api/publico/conjuntos-dados")
        payload = await _http_get_json(
            provider,
            url,
            params={
                "nomeConjuntoDados": query or None,
                "dadosAbertos": "true",
                "isPrivado": "false",
                "pagina": page,
            },
            auth_required=True,
        )
        items = payload if isinstance(payload, list) else _nested_get(payload, "content", [])
        if not isinstance(items, list):
            items = []
        normalized = [normalize_dados_catalog_item(item) for item in items if isinstance(item, Mapping)]
        if category:
            needle = category.lower()
            normalized = [item for item in normalized if needle in str(item.get("category") or "").lower()]
        return {"provider": _redact_provider(provider), "page": page, "items": normalized}

    mappings = dict(provider.get("mappings") or {})
    catalog_path = mappings.get("catalog_path") or mappings.get("list_path") or "/"
    url = _join_provider_url(provider, str(catalog_path))
    params = {
        str(mappings.get("query_param") or "q"): query or None,
        str(mappings.get("category_param") or "category"): category or None,
        str(mappings.get("page_param") or "page"): page,
    }
    payload = await _http_get_json(provider, url, params=params)
    result_path = mappings.get("catalog_result_path") or mappings.get("items_path") or "items"
    items = _nested_get(payload, result_path, payload if isinstance(payload, list) else [])
    if not isinstance(items, list):
        items = []
    return {
        "provider": _redact_provider(provider),
        "page": page,
        "items": [normalize_generic_catalog_item(item, mappings) for item in items if isinstance(item, Mapping)],
    }


async def get_dataset_detail(provider: Mapping[str, Any], external_id: str) -> dict[str, Any]:
    provider_type = str(provider.get("provider_type") or "").lower()
    if provider_type == "dados_gov_br":
        url = _join_provider_url(provider, f"/dados/api/publico/conjuntos-dados/{quote(str(external_id), safe='')}")
        payload = await _http_get_json(provider, url, auth_required=True)
        if not isinstance(payload, Mapping):
            raise StoreValidationError("dados.gov.br dataset detail response was not an object.", status_code=502)
        return normalize_dados_detail(payload)

    mappings = dict(provider.get("mappings") or {})
    detail_path = str(mappings.get("detail_path") or mappings.get("item_path") or "")
    if not detail_path:
        raise StoreValidationError("Generic providers require mappings.detail_path to load dataset details.")
    url = _join_provider_url(provider, detail_path.replace("{id}", quote(str(external_id), safe="")))
    payload = await _http_get_json(provider, url)
    detail_payload = _nested_get(payload, mappings.get("detail_result_path") or "$", payload)
    if not isinstance(detail_payload, Mapping):
        raise StoreValidationError("Generic provider detail response was not an object.", status_code=502)
    return normalize_generic_detail(detail_payload, mappings)


def resource_from_detail(detail: Mapping[str, Any], resource_external_id: str) -> dict[str, Any]:
    resources = detail.get("resources") if isinstance(detail.get("resources"), list) else []
    for resource in resources:
        if not isinstance(resource, Mapping):
            continue
        if str(resource.get("external_id") or "") == str(resource_external_id):
            return dict(resource)
    raise StoreValidationError(f"Resource '{resource_external_id}' was not found in the selected dataset.", status_code=404)


def _embedded_resource_extension(value: Any) -> str:
    text = unquote(str(value or ""))
    for match in re.finditer(r"\.([A-Za-z0-9]{2,8})(?=$|[/?#])", text):
        extension = f".{match.group(1).lower()}"
        if extension in SUPPORTED_PREVIEW_FORMATS.values() or extension.lstrip(".") in UNSUPPORTED_RESOURCE_FORMATS:
            return extension
    suffix = Path(urlparse(text).path or text).suffix.lower()
    return suffix


def _resource_file_candidate(resource: Mapping[str, Any]) -> str:
    for key in ("file_name", "title"):
        value = str(resource.get(key) or "").strip()
        if value:
            return value
    path = unquote(urlparse(str(resource.get("url") or "")).path)
    match = re.search(r"([^/]+\.[A-Za-z0-9]{2,8})(?=$|/(?:view|@@download/file)$)", path)
    if match:
        return match.group(1)
    return Path(path).name or "store_resource"


def _resource_extension(resource: Mapping[str, Any]) -> str:
    candidates = [
        _embedded_resource_extension(resource.get("file_name")),
        _embedded_resource_extension(resource.get("url")),
        _embedded_resource_extension(resource.get("title")),
    ]
    for candidate in candidates:
        if candidate in SUPPORTED_PREVIEW_FORMATS.values():
            return candidate
        if candidate.lstrip(".") in UNSUPPORTED_RESOURCE_FORMATS:
            raise StoreValidationError(
                f"Selected resource format '{candidate.lstrip('.').upper()}' is not supported for Store preview or materialization. "
                "Choose a tabular resource such as CSV, TSV, JSON, JSONL, Parquet, or Excel.",
                status_code=415,
            )

    normalized_format = str(resource.get("format") or "").strip().lower().lstrip(".")
    if normalized_format in SUPPORTED_PREVIEW_FORMATS:
        return SUPPORTED_PREVIEW_FORMATS[normalized_format]
    if normalized_format in UNSUPPORTED_RESOURCE_FORMATS:
        raise StoreValidationError(
            f"Selected resource format '{normalized_format.upper()}' is not supported for Store preview or materialization. "
            "Choose a tabular resource such as CSV, TSV, JSON, JSONL, Parquet, or Excel.",
            status_code=415,
        )
    raise StoreValidationError(
        "Selected resource does not declare a supported tabular format. "
        "Choose a CSV, TSV, JSON, JSONL, Parquet, or Excel resource.",
        status_code=415,
    )


def _is_dados_resource_url(url: str) -> bool:
    host = str(urlparse(url).hostname or "").lower()
    return host.endswith("gov.br")


def _direct_resource_url(provider: Mapping[str, Any], url: str) -> str:
    if str(provider.get("provider_type") or "").lower() != "dados_gov_br" or not _is_dados_resource_url(url):
        return url
    parsed = urlparse(url)
    path = parsed.path
    if path.endswith("/view"):
        return urlunparse(parsed._replace(path=f"{path[:-5]}/@@download/file"))
    return url


def _filename_for_resource(resource: Mapping[str, Any]) -> str:
    candidate = _resource_file_candidate(resource)
    candidate = candidate.split("?")[0].strip() or "store_resource"
    suffix = Path(candidate).suffix.lower()
    extension = _resource_extension(resource)
    if suffix == extension:
        return candidate
    return f"{_slugify(candidate, fallback='store_resource')}{extension}"


def _resolved_resource_url(provider: Mapping[str, Any], resource: dict[str, Any]) -> str:
    url = str(resource.get("url") or "").strip()
    if not url:
        raise StoreValidationError("Selected resource does not include a download URL.")
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = _join_provider_url(provider, url)
    url = _direct_resource_url(provider, url)
    validate_public_http_url(url, allow_path=True)
    return url


async def download_resource(provider: Mapping[str, Any], dataset_external_id: str, resource_external_id: str) -> DownloadedResource:
    detail = await get_dataset_detail(provider, dataset_external_id)
    resource = resource_from_detail(detail, resource_external_id)
    original_url = str(resource.get("url") or "").strip()
    url = _resolved_resource_url(provider, resource)
    resource["catalog_url"] = original_url
    resource["url"] = url
    resource["file_extension"] = _resource_extension(resource)
    content, download_metadata = await _http_get_bytes(
        provider,
        url,
        max_bytes=int(provider.get("max_download_bytes") or DEFAULT_MAX_DOWNLOAD_BYTES),
        auth_required=str(provider.get("provider_type") or "").lower() == "dados_gov_br",
    )
    resource = {**resource, "download": download_metadata}
    return DownloadedResource(
        content=content,
        filename=_filename_for_resource(resource),
        resource=resource,
        detail=detail,
        provider=dict(provider),
    )


def _dataframe_preview_rows(dataframe: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    preview = dataframe.head(limit).copy()
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")


async def preview_resource(provider: Mapping[str, Any], dataset_external_id: str, resource_external_id: str) -> dict[str, Any]:
    downloaded = await download_resource(provider, dataset_external_id, resource_external_id)
    dataframe, parser_report = parse_downloaded_resource(downloaded)
    return {
        "status": "ok",
        "provider": _redact_provider(downloaded.provider),
        "dataset": downloaded.detail,
        "resource": downloaded.resource,
        "columns": [str(column) for column in dataframe.columns],
        "preview_rows": _dataframe_preview_rows(dataframe, limit=5),
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "parser_report": parser_report,
        "column_explorer": build_column_explorer(dataframe),
    }


def parse_downloaded_resource(downloaded: DownloadedResource) -> tuple[pd.DataFrame, dict[str, Any]]:
    content_prefix = downloaded.content[:512].lstrip().lower()
    if content_prefix.startswith((b"<!doctype html", b"<html", b"<?xml")):
        raise StoreValidationError(
            "Selected resource returned an HTML/XML page instead of a tabular file. "
            "Use a direct file download URL or choose another tabular resource.",
            status_code=415,
        )
    try:
        return load_tabular_from_bytes(downloaded.content, filename=downloaded.filename)
    except StoreValidationError:
        raise
    except ValueError as exc:
        raise StoreValidationError(str(exc), status_code=400) from exc
    except Exception as exc:
        raise StoreValidationError(f"Unable to parse selected resource as tabular data: {exc}", status_code=400) from exc


async def validate_resource_connection(provider: Mapping[str, Any], dataset_external_id: str, resource_external_id: str) -> dict[str, Any]:
    detail = await get_dataset_detail(provider, dataset_external_id)
    resource = resource_from_detail(detail, resource_external_id)
    original_url = str(resource.get("url") or "").strip()
    url = _resolved_resource_url(provider, resource)
    resource["catalog_url"] = original_url
    resource["url"] = url
    resource["file_extension"] = _resource_extension(resource)
    head = await _http_head(provider, url, auth_required=str(provider.get("provider_type") or "").lower() == "dados_gov_br")
    return {
        "status": "ok",
        "validated": True,
        "provider": _redact_provider(provider),
        "dataset": detail,
        "resource": resource,
        "connection": head,
        "validated_at": _utc_now(),
    }


def make_connection_string(provider_id: str, dataset_external_id: str, resource_external_id: str) -> str:
    return (
        f"store://{quote(str(provider_id), safe='')}/"
        f"{quote(str(dataset_external_id), safe='')}/"
        f"{quote(str(resource_external_id), safe='')}"
    )


def parse_connection_string(connection_string: str) -> dict[str, str]:
    parsed = urlparse(str(connection_string or ""))
    if parsed.scheme != "store":
        raise StoreValidationError("Dataset connection_string is not a store:// connection.")
    provider_id = unquote(parsed.netloc)
    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise StoreValidationError("Store connection string must include dataset and resource ids.")
    return {"provider_id": provider_id, "dataset_external_id": parts[0], "resource_external_id": parts[1]}


async def materialize_resource_files(
    provider: Mapping[str, Any],
    dataset_external_id: str,
    resource_external_id: str,
    *,
    dataset_name: str,
    dataset_dir: Path,
) -> dict[str, Any]:
    downloaded = await download_resource(provider, dataset_external_id, resource_external_id)
    dataframe, parser_report = parse_downloaded_resource(downloaded)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / f"{_slugify(dataset_name, fallback='store_dataset')}_{uuid.uuid4().hex[:8]}.csv"
    dataframe.to_csv(output_path, index=False, encoding="utf-8")
    connection_string = make_connection_string(str(provider.get("id")), dataset_external_id, resource_external_id)
    return {
        "output_path": output_path,
        "dataframe": dataframe,
        "parser_report": parser_report,
        "provider": _redact_provider(provider),
        "dataset": downloaded.detail,
        "resource": downloaded.resource,
        "connection_string": connection_string,
        "preview_rows": _dataframe_preview_rows(dataframe, limit=5),
        "columns": [str(column) for column in dataframe.columns],
    }


async def refresh_resource_file(
    provider: Mapping[str, Any],
    dataset_external_id: str,
    resource_external_id: str,
    *,
    output_path: Path,
) -> dict[str, Any]:
    downloaded = await download_resource(provider, dataset_external_id, resource_external_id)
    dataframe, parser_report = parse_downloaded_resource(downloaded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8")
    return {
        "output_path": output_path,
        "dataframe": dataframe,
        "parser_report": parser_report,
        "provider": _redact_provider(provider),
        "dataset": downloaded.detail,
        "resource": downloaded.resource,
        "preview_rows": _dataframe_preview_rows(dataframe, limit=5),
        "columns": [str(column) for column in dataframe.columns],
    }


def build_store_manifest(
    *,
    dataset_id: int,
    dataset_name: str,
    output_path: Path,
    connection_string: str,
    provider: Mapping[str, Any],
    external_dataset: Mapping[str, Any],
    resource: Mapping[str, Any],
    parser_report: Mapping[str, Any],
    live: Mapping[str, Any],
    manifest_dir: Path,
) -> dict[str, Any]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"dataset_{dataset_id}_store_manifest.json"
    manifest = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "snapshot_path": str(output_path),
        "connection_string": connection_string,
        "provider": _redact_provider(provider),
        "external_dataset": dict(external_dataset),
        "resource": dict(resource),
        "parser_report": dict(parser_report),
        "live": dict(live),
        "manifest_path": str(manifest_path),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    save_json(manifest, manifest_path)
    return manifest


def find_store_manifest_path(dataset_record: Any, manifest_dir: Path) -> Optional[Path]:
    history = getattr(dataset_record, "history", None)
    if isinstance(history, list):
        for entry in reversed(history):
            if isinstance(entry, Mapping) and entry.get("manifest_path"):
                candidate = Path(str(entry["manifest_path"]))
                if candidate.exists():
                    return candidate
    dataset_id = getattr(dataset_record, "id", None)
    if dataset_id is not None:
        candidate = manifest_dir / f"dataset_{dataset_id}_store_manifest.json"
        if candidate.exists():
            return candidate
    return None


def load_store_manifest(dataset_record: Any, manifest_dir: Path) -> dict[str, Any]:
    manifest_path = find_store_manifest_path(dataset_record, manifest_dir)
    if manifest_path is None:
        raise StoreValidationError("Store manifest was not found for this dataset.", status_code=404)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StoreValidationError(f"Unable to read store manifest: {exc}", status_code=500) from exc
    if not isinstance(payload, Mapping):
        raise StoreValidationError("Store manifest is invalid.", status_code=500)
    return dict(payload)
