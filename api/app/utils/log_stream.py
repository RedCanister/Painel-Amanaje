from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_LOG_TAIL = 160
MAX_LOG_TAIL = 500
DOCKER_TIMEOUT_SECONDS = 4

APP_LOG_SOURCES: dict[str, dict[str, str]] = {
    "app": {
        "label": "Application",
        "filename": "app.log",
    },
    "main_app": {
        "label": "FastAPI",
        "filename": "main_app.log",
    },
}

CONTAINER_SERVICES: dict[str, str] = {
    "api": "API",
    "postgres": "Postgres",
    "redis": "Redis",
    "airflow-webserver": "Airflow",
    "airflow-init": "Airflow Init",
    "mlflow": "MLflow",
    "pgadmin": "pgAdmin",
}

LOG_LINE_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\|\s+"
    r"(?P<logger>[^|]+)\s+\|\s+(?P<level>[A-Z]+)\s+\|\s+(?P<message>.*)$"
)
DOCKER_PREFIX_PATTERN = re.compile(r"^(?P<source>[^|]{1,80})\|\s?(?P<message>.*)$")
ISO_TIMESTAMP_PATTERN = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T[^\s]+)\s+(?P<message>.*)$")
KEY_VALUE_SECRET_PATTERN = re.compile(
    r"(?P<key>\b(?:api[_-]?key|authorization|bearer|credential|fernet[_-]?key|"
    r"password|passwd|private[_-]?key|secret|sql[_-]?alchemy[_-]?conn|token)\b)"
    r"(?P<sep>\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
JSON_SECRET_PATTERN = re.compile(
    r"(?P<prefix>\"(?:api[_-]?key|authorization|credential|fernet[_-]?key|password|passwd|"
    r"private[_-]?key|secret|sql[_-]?alchemy[_-]?conn|token)\"\s*:\s*)"
    r"(?P<value>\"[^\"]*\"|[^,\}\]]+)",
    re.IGNORECASE,
)
URL_CREDENTIAL_PATTERN = re.compile(
    r"(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<credentials>[^/\s:@]+:[^@\s/]+)@",
    re.IGNORECASE,
)
QUERY_SECRET_PATTERN = re.compile(
    r"(?P<prefix>[?&](?:api[_-]?key|password|secret|token)=)(?P<value>[^&\s]+)",
    re.IGNORECASE,
)
BEARER_PATTERN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/\-]+=*", re.IGNORECASE)


class LogSourceValidationError(ValueError):
    def __init__(self, source: str) -> None:
        super().__init__(f"Unknown log source: {source}")
        self.source = source


def normalize_tail(value: int | str | None = None) -> int:
    try:
        tail = int(value if value is not None else DEFAULT_LOG_TAIL)
    except (TypeError, ValueError):
        tail = DEFAULT_LOG_TAIL
    return max(1, min(tail, MAX_LOG_TAIL))


def redact_log_text(value: Any) -> str:
    text = str(value or "")
    text = URL_CREDENTIAL_PATTERN.sub(r"\g<scheme>[REDACTED]@", text)
    text = BEARER_PATTERN.sub("Bearer [REDACTED]", text)
    text = KEY_VALUE_SECRET_PATTERN.sub(lambda match: f"{match.group('key')}{match.group('sep')}[REDACTED]", text)
    text = JSON_SECRET_PATTERN.sub(lambda match: f"{match.group('prefix')}\"[REDACTED]\"", text)
    text = QUERY_SECRET_PATTERN.sub(lambda match: f"{match.group('prefix')}[REDACTED]", text)
    return text


def _tail_lines(path: Path, limit: int) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\r\n") for line in deque(handle, maxlen=limit)]


def _source_definition(log_dir: Path, activity_log_path: Path, compose_file: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = [
        {
            "id": "all",
            "label": "All Sources",
            "type": "combined",
            "available": True,
        }
    ]
    for source_id, metadata in APP_LOG_SOURCES.items():
        path = log_dir / metadata["filename"]
        sources.append(
            {
                "id": source_id,
                "label": metadata["label"],
                "type": "file",
                "available": path.exists(),
                "path": str(Path("logs") / metadata["filename"]),
            }
        )
    sources.append(
        {
            "id": "activity",
            "label": "Activity",
            "type": "activity",
            "available": activity_log_path.exists(),
            "path": str(Path("runtime_artifacts") / "monitoring" / activity_log_path.name),
        }
    )
    for service, label in CONTAINER_SERVICES.items():
        sources.append(
            {
                "id": f"container:{service}",
                "label": label,
                "type": "container",
                "available": compose_file.exists(),
                "service": service,
            }
        )
    return sources


def _known_source_ids() -> set[str]:
    return {"all", "activity", *APP_LOG_SOURCES.keys(), *(f"container:{service}" for service in CONTAINER_SERVICES)}


def _validate_source(source: str | None) -> str:
    normalized = str(source or "all").strip() or "all"
    if normalized not in _known_source_ids():
        raise LogSourceValidationError(normalized)
    return normalized


def _log_entry(
    *,
    source_id: str,
    source_label: str,
    source_type: str,
    message: str,
    timestamp: str | None = None,
    level: str | None = None,
    logger_name: str | None = None,
    raw: str | None = None,
) -> dict[str, Any]:
    redacted_message = redact_log_text(message)
    return {
        "source": source_id,
        "source_label": source_label,
        "source_type": source_type,
        "timestamp": redact_log_text(timestamp) if timestamp else None,
        "level": redact_log_text(level or ""),
        "logger": redact_log_text(logger_name or ""),
        "message": redacted_message,
        "raw": redact_log_text(raw if raw is not None else message),
    }


def _read_app_log(source_id: str, metadata: Mapping[str, str], path: Path, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"{metadata.get('label', source_id)} log file is not available yet."]

    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        for line in _tail_lines(path, limit):
            match = LOG_LINE_PATTERN.match(line)
            if match:
                entries.append(
                    _log_entry(
                        source_id=source_id,
                        source_label=metadata.get("label", source_id),
                        source_type="file",
                        timestamp=match.group("timestamp"),
                        level=match.group("level"),
                        logger_name=match.group("logger").strip(),
                        message=match.group("message"),
                        raw=line,
                    )
                )
            elif line.strip():
                entries.append(
                    _log_entry(
                        source_id=source_id,
                        source_label=metadata.get("label", source_id),
                        source_type="file",
                        message=line,
                        raw=line,
                    )
                )
    except OSError as exc:
        warnings.append(f"Unable to read {metadata.get('label', source_id)} log: {exc}")
    return entries, warnings


def _redact_mapping(value: Any, parent_key: str = "") -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child_value in value.items():
            key_text = str(key)
            if re.search(r"api[_-]?key|authorization|credential|password|passwd|private[_-]?key|secret|token", key_text, re.I):
                redacted[key_text] = "[REDACTED]"
            else:
                redacted[key_text] = _redact_mapping(child_value, key_text)
        return redacted
    if isinstance(value, list):
        return [_redact_mapping(item, parent_key) for item in value]
    if isinstance(value, str):
        return redact_log_text(value)
    return value


def _read_activity_log(path: Path, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], ["Activity log file is not available yet."]

    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        for line in _tail_lines(path, limit):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            safe_payload = _redact_mapping(payload)
            raw = json.dumps(safe_payload, ensure_ascii=True, sort_keys=True)
            entries.append(
                _log_entry(
                    source_id="activity",
                    source_label="Activity",
                    source_type="activity",
                    timestamp=str(safe_payload.get("timestamp") or ""),
                    level="INFO",
                    logger_name=str(safe_payload.get("event_type") or "activity"),
                    message=str(safe_payload.get("message") or safe_payload.get("event_type") or raw),
                    raw=raw,
                )
            )
    except OSError as exc:
        warnings.append(f"Unable to read activity log: {exc}")
    return entries, warnings


def _docker_command_prefixes() -> Iterable[list[str]]:
    if shutil.which("docker"):
        yield ["docker", "compose"]
    if shutil.which("docker-compose"):
        yield ["docker-compose"]


def _parse_docker_line(line: str) -> tuple[str, str, str | None]:
    source_label = "Container"
    message = line
    prefix_match = DOCKER_PREFIX_PATTERN.match(line)
    if prefix_match:
        source_label = prefix_match.group("source").strip()
        message = prefix_match.group("message").strip()

    timestamp = None
    timestamp_match = ISO_TIMESTAMP_PATTERN.match(message)
    if timestamp_match:
        timestamp = timestamp_match.group("timestamp")
        message = timestamp_match.group("message")
    return source_label, message, timestamp


def _read_docker_logs(compose_file: Path, services: list[str], limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    if not compose_file.exists():
        return [], ["Docker Compose file is not available from this runtime."]

    command_prefixes = list(_docker_command_prefixes())
    if not command_prefixes:
        return [], ["Docker CLI is unavailable from this runtime."]

    warnings: list[str] = []
    for prefix in command_prefixes:
        command = [
            *prefix,
            "-f",
            str(compose_file),
            "logs",
            "--no-color",
            "--timestamps",
            "--tail",
            str(limit),
            *services,
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=DOCKER_TIMEOUT_SECONDS,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            warnings.append(f"Docker logs unavailable through {' '.join(prefix)}: {exc}")
            continue

        if completed.returncode != 0:
            detail = redact_log_text((completed.stderr or completed.stdout or "").strip())
            warnings.append(f"Docker logs command failed through {' '.join(prefix)}: {detail or completed.returncode}")
            continue

        entries: list[dict[str, Any]] = []
        for line in (completed.stdout or "").splitlines()[-limit * max(1, len(services) or 1):]:
            if not line.strip():
                continue
            source_label, message, timestamp = _parse_docker_line(line)
            entries.append(
                _log_entry(
                    source_id=f"container:{source_label}",
                    source_label=source_label,
                    source_type="container",
                    timestamp=timestamp,
                    message=message,
                    raw=line,
                )
            )
        return entries, warnings

    return [], warnings or ["Docker logs are unavailable from this runtime."]


def _sort_timestamp(value: str | None) -> tuple[int, str]:
    if not value:
        return (0, "")
    normalized = value.replace("Z", "+00:00").replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(normalized)
        return (1, parsed.isoformat())
    except ValueError:
        return (1, value)


def collect_log_snapshot(
    *,
    log_dir: str | Path,
    activity_log_path: str | Path,
    compose_file: str | Path,
    source: str | None = "all",
    tail: int | str | None = DEFAULT_LOG_TAIL,
    include_docker: bool = True,
) -> dict[str, Any]:
    normalized_source = _validate_source(source)
    normalized_tail = normalize_tail(tail)
    log_root = Path(log_dir)
    activity_path = Path(activity_log_path)
    compose_path = Path(compose_file)
    warnings: list[str] = []
    entries: list[dict[str, Any]] = []

    if normalized_source == "all" or normalized_source in APP_LOG_SOURCES:
        for source_id, metadata in APP_LOG_SOURCES.items():
            if normalized_source not in {"all", source_id}:
                continue
            source_entries, source_warnings = _read_app_log(source_id, metadata, log_root / metadata["filename"], normalized_tail)
            entries.extend(source_entries)
            warnings.extend(source_warnings)

    if normalized_source in {"all", "activity"}:
        activity_entries, activity_warnings = _read_activity_log(activity_path, normalized_tail)
        entries.extend(activity_entries)
        warnings.extend(activity_warnings)

    if include_docker and (normalized_source == "all" or normalized_source.startswith("container:")):
        services: list[str] = []
        if normalized_source.startswith("container:"):
            services = [normalized_source.split(":", 1)[1]]
        docker_entries, docker_warnings = _read_docker_logs(compose_path, services, normalized_tail)
        entries.extend(docker_entries)
        warnings.extend(docker_warnings)

    entries = sorted(
        enumerate(entries),
        key=lambda item: (_sort_timestamp(item[1].get("timestamp")), item[0]),
    )
    trimmed_entries = [entry for _, entry in entries[-normalized_tail:]]
    for index, entry in enumerate(trimmed_entries, start=1):
        entry["id"] = f"{entry.get('source', 'log')}:{index}"

    return {
        "status": "ok",
        "generated_at": datetime.now().isoformat(),
        "tail": normalized_tail,
        "source": normalized_source,
        "include_docker": bool(include_docker),
        "sources": _source_definition(log_root, activity_path, compose_path),
        "entries": trimmed_entries,
        "warnings": warnings,
    }
