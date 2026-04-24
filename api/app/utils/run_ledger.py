from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional


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
        "events": [
            {
                "status": status,
                "stage": "queued",
                "timestamp": now,
                "message": f"{run_type} run created.",
            }
        ],
    }
    write_run_entry(base_dir, payload)
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
            else:
                payload[key] = value
    payload["updated_at"] = now
    if event_message:
        payload.setdefault("events", []).append(
            {
                "status": payload.get("status"),
                "stage": payload.get("stage"),
                "timestamp": now,
                "message": event_message,
            }
        )
    write_run_entry(base_dir, payload)
    return payload


def list_run_entries(
    base_dir: str | Path,
    *,
    run_type: Optional[str] = None,
    model_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    study_id: Optional[int] = None,
    inference_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    directory = ensure_run_ledger_dir(base_dir)
    items: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        context = dict(payload.get("context", {}) or {})
        if run_type and str(payload.get("run_type")) != str(run_type):
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
    return items
