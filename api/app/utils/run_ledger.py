from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


DEFAULT_TERMINAL_CAP = 2000


def ensure_run_ledger_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _run_path(base_dir: str | Path, run_id: str) -> Path:
    return ensure_run_ledger_dir(base_dir) / f"{run_id}.json"


def create_run_entry(
    base_dir: str | Path,
    *,
    run_type: str,
    context: Optional[Mapping[str, Any]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    status: str = "queued",
) -> dict[str, Any]:
    run_id = f"{run_type}_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    created_message = f"{run_type} run created."
    terminal_line = {
        "index": 0,
        "timestamp": now,
        "stream": "system",
        "stage": "queued",
        "message": created_message,
    }
    payload = {
        "run_id": run_id,
        "run_type": run_type,
        "status": status,
        "stage": "queued",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "context": dict(context or {}),
        "parameters": dict(parameters or {}),
        "metrics": {},
        "artifacts": {},
        "result": {},
        "error": None,
        "progress": {
            "current_stage": "queued",
            "latest_event": created_message,
            "latest_timestamp": now,
            "timeline": [
                {
                    "status": status,
                    "stage": "queued",
                    "timestamp": now,
                    "message": created_message,
                }
            ],
            "counters": {},
        },
        "events": [
            {
                "status": status,
                "stage": "queued",
                "timestamp": now,
                "message": created_message,
            }
        ],
        "terminal": {
            "cap": DEFAULT_TERMINAL_CAP,
            "next_index": 1,
            "lines": [terminal_line],
        },
    }
    write_run_entry(base_dir, payload)
    return payload


def _coerce_terminal_cap(value: Any = None) -> int:
    try:
        cap = int(value if value is not None else DEFAULT_TERMINAL_CAP)
    except (TypeError, ValueError):
        cap = DEFAULT_TERMINAL_CAP
    return max(1, cap)


def _terminal_line(
    *,
    index: int,
    timestamp: str,
    stream: str,
    stage: str,
    message: str,
) -> dict[str, Any]:
    return {
        "index": int(index),
        "timestamp": timestamp,
        "stream": str(stream or "system"),
        "stage": str(stage or ""),
        "message": str(message or ""),
    }


def _append_terminal_line_to_payload(
    payload: dict[str, Any],
    *,
    message: str,
    stream: str = "system",
    stage: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict[str, Any]:
    terminal = dict(payload.get("terminal", {}) or {})
    cap = _coerce_terminal_cap(terminal.get("cap"))
    lines = [dict(line) for line in terminal.get("lines", []) if isinstance(line, Mapping)]
    try:
        next_index = int(terminal.get("next_index", len(lines)))
    except (TypeError, ValueError):
        next_index = len(lines)

    clean_message = str(message or "")
    if not clean_message:
        payload["terminal"] = {"cap": cap, "next_index": next_index, "lines": lines[-cap:]}
        return payload

    for raw_line in clean_message.splitlines() or [clean_message]:
        lines.append(
            _terminal_line(
                index=next_index,
                timestamp=timestamp or datetime.now().isoformat(),
                stream=stream,
                stage=stage if stage is not None else str(payload.get("stage") or ""),
                message=raw_line,
            )
        )
        next_index += 1

    payload["terminal"] = {
        "cap": cap,
        "next_index": next_index,
        "lines": lines[-cap:],
    }
    return payload


def read_run_entry(base_dir: str | Path, run_id: str) -> dict[str, Any] | None:
    path = _run_path(base_dir, run_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_run_entry(base_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path = _run_path(base_dir, str(payload["run_id"]))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return dict(payload)


def update_run_entry(
    base_dir: str | Path,
    run_id: str,
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    event_message: Optional[str] = None,
    merge: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload = read_run_entry(base_dir, run_id)
    if payload is None:
        raise FileNotFoundError(f"Run '{run_id}' was not found in the ledger.")

    now = datetime.now().isoformat()
    if status is not None:
        payload["status"] = status
        if status == "running" and not payload.get("started_at"):
            payload["started_at"] = now
        if status in {"completed", "failed", "cancelled"}:
            payload["completed_at"] = now
    if stage is not None:
        payload["stage"] = stage
    if merge:
        for key, value in merge.items():
            if key in {"context", "parameters", "metrics", "artifacts", "result"} and isinstance(value, Mapping):
                payload[key] = {**dict(payload.get(key, {}) or {}), **dict(value)}
            elif key == "progress" and isinstance(value, Mapping):
                existing_progress = dict(payload.get("progress", {}) or {})
                merged_progress = {**existing_progress, **dict(value)}
                if isinstance(existing_progress.get("counters"), Mapping) or isinstance(value.get("counters"), Mapping):
                    merged_progress["counters"] = {
                        **dict(existing_progress.get("counters", {}) or {}),
                        **dict(value.get("counters", {}) or {}),
                    }
                if isinstance(existing_progress.get("timeline"), list) and "timeline" not in value:
                    merged_progress["timeline"] = list(existing_progress.get("timeline", []) or [])
                payload[key] = merged_progress
            else:
                payload[key] = value
    payload["updated_at"] = now
    stage_changed = stage is not None
    status_changed = status is not None
    if event_message:
        payload.setdefault("events", []).append(
            {
                "status": payload.get("status"),
                "stage": payload.get("stage"),
                "timestamp": now,
                "message": event_message,
            }
        )
        _append_terminal_line_to_payload(
            payload,
            message=event_message,
            stream="system",
            stage=str(payload.get("stage") or ""),
            timestamp=now,
        )
    if event_message or stage_changed or status_changed:
        progress = dict(payload.get("progress", {}) or {})
        progress["current_stage"] = payload.get("stage")
        progress["latest_event"] = event_message or progress.get("latest_event") or ""
        progress["latest_timestamp"] = now
        timeline = list(progress.get("timeline", []) or [])
        timeline.append(
            {
                "status": payload.get("status"),
                "stage": payload.get("stage"),
                "timestamp": now,
                "message": event_message or "",
            }
        )
        progress["timeline"] = timeline
        payload["progress"] = progress
    write_run_entry(base_dir, payload)
    return payload


def append_run_terminal_line(
    base_dir: str | Path,
    run_id: str,
    *,
    message: str,
    stream: str = "system",
    stage: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict[str, Any]:
    payload = read_run_entry(base_dir, run_id)
    if payload is None:
        raise FileNotFoundError(f"Run '{run_id}' was not found in the ledger.")
    updated = _append_terminal_line_to_payload(
        payload,
        message=message,
        stream=stream,
        stage=stage,
        timestamp=timestamp,
    )
    write_run_entry(base_dir, updated)
    return updated


def build_run_terminal_lines(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    terminal = payload.get("terminal") if isinstance(payload, Mapping) else None
    if isinstance(terminal, Mapping) and isinstance(terminal.get("lines"), list) and terminal.get("lines"):
        return [
            _terminal_line(
                index=int(line.get("index", index)),
                timestamp=str(line.get("timestamp") or ""),
                stream=str(line.get("stream") or "system"),
                stage=str(line.get("stage") or ""),
                message=str(line.get("message") or ""),
            )
            for index, line in enumerate(terminal.get("lines", []))
            if isinstance(line, Mapping)
        ]

    synthesized: list[dict[str, Any]] = []

    def add_line(*, timestamp: Any, stream: str, stage: Any, message: Any) -> None:
        text = str(message or "")
        if not text:
            return
        for raw_line in text.splitlines() or [text]:
            synthesized.append(
                _terminal_line(
                    index=len(synthesized),
                    timestamp=str(timestamp or ""),
                    stream=stream,
                    stage=str(stage or ""),
                    message=raw_line,
                )
            )

    for event in payload.get("events", []) or []:
        if not isinstance(event, Mapping):
            continue
        add_line(
            timestamp=event.get("timestamp"),
            stream="system",
            stage=event.get("stage"),
            message=event.get("message"),
        )

    progress = payload.get("progress", {}) if isinstance(payload.get("progress"), Mapping) else {}
    for event in progress.get("timeline", []) or []:
        if not isinstance(event, Mapping):
            continue
        add_line(
            timestamp=event.get("timestamp"),
            stream="system",
            stage=event.get("stage"),
            message=event.get("message"),
        )

    result = payload.get("result", {}) if isinstance(payload.get("result"), Mapping) else {}
    add_line(
        timestamp=payload.get("updated_at"),
        stream="stdout",
        stage=payload.get("stage"),
        message=result.get("stdout"),
    )
    add_line(
        timestamp=payload.get("updated_at"),
        stream="stderr",
        stage=payload.get("stage"),
        message=result.get("stderr"),
    )
    add_line(
        timestamp=payload.get("completed_at") or payload.get("updated_at"),
        stream="stderr",
        stage=payload.get("stage"),
        message=payload.get("error"),
    )
    return synthesized[-DEFAULT_TERMINAL_CAP:]


def summarize_run_entry(payload: Mapping[str, Any]) -> dict[str, Any]:
    context = dict(payload.get("context", {}) or {})
    parameters = dict(payload.get("parameters", {}) or {})
    metrics = dict(payload.get("metrics", {}) or {})
    progress = payload.get("progress", {}) if isinstance(payload.get("progress"), Mapping) else {}
    counters = progress.get("counters", {}) if isinstance(progress.get("counters"), Mapping) else {}
    terminal = payload.get("terminal", {}) if isinstance(payload.get("terminal"), Mapping) else {}
    result = payload.get("result", {}) if isinstance(payload.get("result"), Mapping) else {}
    artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), Mapping) else {}
    events = payload.get("events", []) if isinstance(payload.get("events"), list) else []
    failure = payload.get("failure", {}) if isinstance(payload.get("failure"), Mapping) else {}

    return {
        "run_id": payload.get("run_id"),
        "run_type": payload.get("run_type"),
        "status": payload.get("status"),
        "stage": payload.get("stage"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "started_at": payload.get("started_at"),
        "completed_at": payload.get("completed_at"),
        "context": context,
        "parameters": parameters,
        "metrics": metrics,
        "error": payload.get("error"),
        "failure": {
            key: value
            for key, value in failure.items()
            if key not in {"traceback"}
        },
        "progress": {
            "current_stage": progress.get("current_stage"),
            "latest_event": progress.get("latest_event"),
            "latest_timestamp": progress.get("latest_timestamp"),
            "counters": dict(counters),
        },
        "queue": dict(payload.get("queue", {}) or {}),
        "control": dict(payload.get("control", {}) or {}),
        "source_run_id": payload.get("source_run_id"),
        "event_count": len(events),
        "terminal_summary": {
            "line_count": len(terminal.get("lines", []) or []),
            "next_index": terminal.get("next_index"),
            "cap": terminal.get("cap"),
        },
        "result_summary": {
            "status": result.get("status"),
            "keys": sorted(str(key) for key in result.keys())[:20],
            "key_count": len(result),
        },
        "artifact_summary": {
            "keys": sorted(str(key) for key in artifacts.keys())[:20],
            "key_count": len(artifacts),
        },
        "has_result": bool(result),
        "has_failure_traceback": bool(failure.get("traceback")),
    }


def list_run_entries(
    base_dir: str | Path,
    *,
    run_type: Optional[str] = None,
    status: Optional[str] = None,
    active_only: bool = False,
    run_ids: Optional[Iterable[str]] = None,
    model_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    study_id: Optional[int] = None,
    inference_id: Optional[int] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    directory = ensure_run_ledger_dir(base_dir)
    items: list[dict[str, Any]] = []
    requested_statuses = {
        item.strip()
        for item in str(status or "").split(",")
        if item.strip()
    }
    requested_ids = {str(item) for item in run_ids or [] if str(item).strip()}
    active_statuses = {"queued", "running", "paused", "cancel_requested"}
    for path in directory.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        context = dict(payload.get("context", {}) or {})
        payload_status = str(payload.get("status") or "")
        payload_run_id = str(payload.get("run_id") or "")
        if run_type and str(payload.get("run_type")) != str(run_type):
            continue
        if requested_ids and payload_run_id not in requested_ids:
            continue
        if requested_statuses and payload_status not in requested_statuses:
            continue
        if active_only and payload_status not in active_statuses:
            continue
        if model_id is not None and int(context.get("model_id", 0) or 0) != int(model_id):
            continue
        if dataset_id is not None and int(context.get("dataset_id", 0) or 0) != int(dataset_id):
            continue
        if study_id is not None and int(context.get("study_id", 0) or 0) != int(study_id):
            continue
        if inference_id is not None and int(context.get("inference_id", 0) or 0) != int(inference_id):
            continue
        items.append(payload)
    items.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
    if limit is not None and limit > 0:
        return items[:limit]
    return items
