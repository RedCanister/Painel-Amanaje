from __future__ import annotations

import asyncio
import os
import re
import secrets
import socket
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Mapping


DEFAULT_QUEUE_NAME = os.getenv("AMANAJE_DEFAULT_QUEUE", "amanaje:default")
INTERACTIVE_QUEUE_NAME = os.getenv("AMANAJE_INTERACTIVE_QUEUE", "amanaje:interactive")
ASSISTANT_QUEUE_NAME = os.getenv("AMANAJE_ASSISTANT_QUEUE", "amanaje:assistant")
GPU_QUEUE_NAME = os.getenv("AMANAJE_GPU_QUEUE", "amanaje:gpu")
ALL_OPERATION_QUEUES = [INTERACTIVE_QUEUE_NAME, DEFAULT_QUEUE_NAME, ASSISTANT_QUEUE_NAME, GPU_QUEUE_NAME]
REDIS_URL = os.getenv("AMANAJE_REDIS_URL", os.getenv("REDIS_URL", "redis://redis:6379/0"))
WORKER_HOST_ID = os.getenv("AMANAJE_WORKER_HOST_ID", socket.gethostname())
WORKER_ROLE = os.getenv("AMANAJE_WORKER_ROLE", "cpu")
WORKER_CONCURRENCY = int(os.getenv("AMANAJE_WORKER_CONCURRENCY", "1") or "1")
WORKER_STALE_SECONDS = int(os.getenv("AMANAJE_WORKER_STALE_SECONDS", "120") or "120")
WORKER_NAME_PREFIX = os.getenv("AMANAJE_WORKER_NAME_PREFIX", "amanaje")

_SECRET_PATTERN = re.compile(
    r"(?i)(token|password|passwd|secret|api[_-]?key|authorization)(\s*[:=]\s*)([^\s,;]+)"
)


class QueueUnavailableError(RuntimeError):
    error_code = "queue_backend_unavailable"
    reason = "redis_rq_unavailable"

    def to_payload(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "domain": "queue_runtime",
            "queue_backend": "rq",
            "errors": [{"field": "queue", "reason": self.reason, "message": str(self)}],
            "recovery": [
                "Install redis and rq in the active API environment.",
                "Start Redis and point AMANAJE_REDIS_URL at the reachable Redis service.",
                "Rebuild the API and worker images after dependency changes.",
            ],
        }


def _load_rq() -> tuple[Any, Any]:
    try:
        from redis import Redis
        from rq import Queue

        return Redis, Queue
    except Exception as exc:
        raise QueueUnavailableError(f"Redis/RQ dependencies are not available: {exc}") from exc


def get_redis_connection() -> Any:
    Redis, _Queue = _load_rq()
    timeout = float(os.getenv("AMANAJE_REDIS_TIMEOUT_SECONDS", "2") or "2")
    connection = Redis.from_url(
        REDIS_URL,
        socket_connect_timeout=timeout,
        socket_timeout=timeout,
    )
    try:
        connection.ping()
    except Exception as exc:
        raise QueueUnavailableError(f"Redis is not reachable at {REDIS_URL}: {exc}") from exc
    return connection


def configured_queue_names(queue_names: list[str] | None = None) -> list[str]:
    names = list(queue_names or ALL_OPERATION_QUEUES)
    deduped: list[str] = []
    for name in names:
        normalized = str(name or "").strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _name_segment(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value or "").strip())
    return text.strip("-") or "worker"


def build_worker_name(queue_names: list[str] | None = None, *, base_name: str | None = None) -> str:
    queues = configured_queue_names(queue_names or [os.getenv("AMANAJE_WORKER_QUEUE", DEFAULT_QUEUE_NAME)])
    queue_slug = _name_segment("-".join(queue.replace(":", "-") for queue in queues))
    base = _name_segment(base_name or f"{WORKER_NAME_PREFIX}-{WORKER_HOST_ID}-{WORKER_ROLE}")
    hostname = _name_segment(socket.gethostname())
    suffix = secrets.token_hex(4)
    return f"{base}-{queue_slug}-{hostname}-{os.getpid()}-{suffix}"


def enqueue_callable(
    queue_name: str,
    callable_: Callable[..., Any],
    *,
    job_id: str,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    _Redis, Queue = _load_rq()
    connection = get_redis_connection()
    queue = Queue(queue_name, connection=connection)
    job = queue.enqueue(
        callable_,
        kwargs=dict(kwargs),
        job_id=job_id,
        job_timeout=os.getenv("AMANAJE_JOB_TIMEOUT", "2h"),
        result_ttl=int(os.getenv("AMANAJE_JOB_RESULT_TTL_SECONDS", "86400")),
        failure_ttl=int(os.getenv("AMANAJE_JOB_FAILURE_TTL_SECONDS", "604800")),
    )
    return {"backend": "rq", "queue": queue_name, "job_id": job.id}


def _queue_for_gpu_preference(prefer_gpu: bool) -> str:
    return GPU_QUEUE_NAME if prefer_gpu else DEFAULT_QUEUE_NAME


def enqueue_training_run(run_id: str, *, model_id: int, payload_data: Mapping[str, Any], prefer_gpu: bool = False) -> dict[str, Any]:
    return enqueue_callable(
        _queue_for_gpu_preference(prefer_gpu),
        run_training_job,
        job_id=run_id,
        kwargs={"run_id": run_id, "model_id": model_id, "payload_data": dict(payload_data)},
    )


def enqueue_study_run(run_id: str, *, study_id: int, payload_data: Mapping[str, Any], prefer_gpu: bool = False) -> dict[str, Any]:
    return enqueue_callable(
        _queue_for_gpu_preference(prefer_gpu),
        run_study_job,
        job_id=run_id,
        kwargs={"run_id": run_id, "study_id": study_id, "payload_data": dict(payload_data)},
    )


def enqueue_editor_execution(run_id: str, *, code: str, registry_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return enqueue_callable(
        INTERACTIVE_QUEUE_NAME,
        run_editor_execution_job,
        job_id=run_id,
        kwargs={"run_id": run_id, "code": code, "registry_context": dict(registry_context or {})},
    )


def enqueue_assistant_operation(run_id: str, *, operation: str, payload_data: Mapping[str, Any]) -> dict[str, Any]:
    return enqueue_callable(
        ASSISTANT_QUEUE_NAME,
        run_assistant_operation_job,
        job_id=run_id,
        kwargs={"run_id": run_id, "operation": operation, "payload_data": dict(payload_data)},
    )


def cancel_rq_job(run_id: str) -> dict[str, Any]:
    try:
        from rq.exceptions import NoSuchJobError
        from rq.job import Job
    except Exception as exc:
        raise QueueUnavailableError(f"Redis/RQ dependencies are not available: {exc}") from exc

    connection = get_redis_connection()
    try:
        job = Job.fetch(run_id, connection=connection)
    except NoSuchJobError:
        return {"backend": "rq", "job_id": run_id, "cancelled": False, "reason": "job_not_found"}

    job_status = str(job.get_status(refresh=True) or "unknown")
    if job_status in {"queued", "deferred", "scheduled"}:
        job.cancel()
        return {"backend": "rq", "job_id": run_id, "cancelled": True, "job_status": job_status}
    return {
        "backend": "rq",
        "job_id": run_id,
        "cancelled": False,
        "job_status": job_status,
        "reason": "cooperative_cancel_required",
    }


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _heartbeat_age_seconds(value: Any) -> float | None:
    parsed = _parse_datetime(value)
    if parsed is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())


def _safe_count(value: Any) -> int:
    try:
        raw = value() if callable(value) else value
        return int(raw or 0)
    except Exception:
        return 0


def _safe_job_ids(source: Any, *, limit: int | None = None) -> list[str]:
    ids: list[Any] = []
    for attr in ("get_job_ids", "job_ids"):
        try:
            value = getattr(source, attr, None)
            if value is None:
                continue
            ids = value() if callable(value) else value
            break
        except Exception:
            ids = []
    normalized = [str(item) for item in list(ids or [])]
    if limit is not None and limit > 0:
        return normalized[:limit]
    return normalized


def _safe_current_job_id(worker: Any) -> str | None:
    for attr in ("get_current_job_id", "current_job_id"):
        try:
            value = getattr(worker, attr, None)
            raw = value() if callable(value) else value
            if raw:
                return str(raw)
        except Exception:
            continue
    return None


def _worker_key_ttl(connection: Any, worker: Any) -> int | None:
    try:
        key = getattr(worker, "key", None) or f"rq:worker:{getattr(worker, 'name', '')}"
        return int(connection.ttl(key))
    except Exception:
        return None


def _is_worker_stale(*, heartbeat_age: float | None, key_ttl_seconds: int | None) -> bool:
    if key_ttl_seconds is not None and key_ttl_seconds > 0:
        return False
    if heartbeat_age is None:
        return True
    return heartbeat_age > WORKER_STALE_SECONDS


def _worker_meta(name: str) -> dict[str, str | None]:
    parts = str(name or "").split("-")
    if len(parts) < 4 or parts[0] != WORKER_NAME_PREFIX:
        return {"host_id": None, "role": None}
    return {"host_id": parts[1] or None, "role": parts[2] or None}


def _redact_failure_text(value: str) -> str:
    return _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", str(value or ""))


def worker_host_contract(queue_names: list[str] | None = None) -> dict[str, Any]:
    queues = configured_queue_names(queue_names or [DEFAULT_QUEUE_NAME])
    return {
        "host_id": WORKER_HOST_ID,
        "role": WORKER_ROLE,
        "queues": queues,
        "concurrency": WORKER_CONCURRENCY,
        "concurrency_model": "one RQ worker process per container; scale hosts/replicas for parallel jobs",
        "redis_url": REDIS_URL,
        "stale_after_seconds": WORKER_STALE_SECONDS,
    }


def _queue_counts(Queue: Any, connection: Any, queue_name: str) -> dict[str, Any]:
    from rq.registry import DeferredJobRegistry, FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry

    queue = Queue(queue_name, connection=connection)
    return {
        "name": queue_name,
        "queued_jobs": _safe_count(getattr(queue, "count", 0)),
        "started_jobs": _safe_count(StartedJobRegistry(queue_name, connection=connection).count),
        "deferred_jobs": _safe_count(DeferredJobRegistry(queue_name, connection=connection).count),
        "finished_jobs": _safe_count(FinishedJobRegistry(queue_name, connection=connection).count),
        "failed_jobs": _safe_count(FailedJobRegistry(queue_name, connection=connection).count),
    }


def get_queue_runtime_snapshot(queue_names: list[str] | None = None, *, limit: int = 50) -> dict[str, Any]:
    queues_to_check = configured_queue_names(queue_names)
    payload: dict[str, Any] = {
        "status": "unknown",
        "backend": "rq",
        "redis_url": REDIS_URL,
        "configured_queues": queues_to_check,
        "queues": [],
        "positions": {},
        "summary": {"queued_jobs": 0, "started_jobs": 0, "failed_jobs": 0},
    }
    try:
        _Redis, Queue = _load_rq()
        from rq.job import Job
        from rq.registry import FailedJobRegistry, StartedJobRegistry

        connection = get_redis_connection()
        queue_payloads: list[dict[str, Any]] = []
        positions: dict[str, dict[str, Any]] = {}
        summary = {"queued_jobs": 0, "started_jobs": 0, "failed_jobs": 0}
        for queue_name in queues_to_check:
            queue = Queue(queue_name, connection=connection)
            job_ids = _safe_job_ids(queue, limit=limit)
            jobs: list[dict[str, Any]] = []
            for index, job_id in enumerate(job_ids):
                job_payload: dict[str, Any] = {"job_id": job_id, "position": index + 1, "queue": queue_name}
                try:
                    job = Job.fetch(job_id, connection=connection)
                    job_payload.update(
                        {
                            "status": str(job.get_status(refresh=False) or "queued"),
                            "enqueued_at": _isoformat(getattr(job, "enqueued_at", None)),
                            "created_at": _isoformat(getattr(job, "created_at", None)),
                            "origin": getattr(job, "origin", queue_name),
                        }
                    )
                except Exception:
                    job_payload["status"] = "queued"
                jobs.append(job_payload)
                positions[job_id] = job_payload

            started_ids = _safe_job_ids(StartedJobRegistry(queue_name, connection=connection), limit=limit)
            failed_ids = _safe_job_ids(FailedJobRegistry(queue_name, connection=connection), limit=limit)
            counts = _queue_counts(Queue, connection, queue_name)
            summary["queued_jobs"] += int(counts["queued_jobs"])
            summary["started_jobs"] += int(counts["started_jobs"])
            summary["failed_jobs"] += int(counts["failed_jobs"])
            queue_payloads.append(
                {
                    **counts,
                    "jobs": jobs,
                    "started_job_ids": started_ids,
                    "failed_job_ids": failed_ids,
                }
            )

        payload.update({"status": "ok", "queues": queue_payloads, "positions": positions, "summary": summary})
    except QueueUnavailableError as exc:
        payload.update({"status": "unavailable", "error": str(exc), "error_code": exc.error_code})
    except Exception as exc:
        payload.update({"status": "error", "error": str(exc)})
    return payload


def get_worker_runtime_status(queue_names: list[str] | None = None) -> dict[str, Any]:
    queues_to_check = configured_queue_names(queue_names)
    payload: dict[str, Any] = {
        "status": "unknown",
        "backend": "rq",
        "redis_url": REDIS_URL,
        "configured_queues": queues_to_check,
        "worker_host_contract": worker_host_contract([DEFAULT_QUEUE_NAME]),
        "queues": [],
        "workers": [],
        "live_workers": [],
        "stale_workers": [],
        "summary": {"active_workers": 0, "live_workers": 0, "stale_workers": 0, "queued_jobs": 0},
    }
    try:
        _Redis, Queue = _load_rq()
        from rq import Worker

        connection = get_redis_connection()
        queue_payloads = []
        queued_jobs = 0
        for queue_name in queues_to_check:
            counts = _queue_counts(Queue, connection, queue_name)
            queued_jobs += int(counts["queued_jobs"])
            queue_payloads.append(counts)

        workers = []
        live_workers = []
        stale_workers = []
        for worker in Worker.all(connection=connection):
            worker_name = str(getattr(worker, "name", "") or "")
            worker_queues = [getattr(queue, "name", str(queue)) for queue in getattr(worker, "queues", [])]
            heartbeat = getattr(worker, "last_heartbeat", None)
            heartbeat_age = _heartbeat_age_seconds(heartbeat)
            key_ttl = _worker_key_ttl(connection, worker)
            stale = _is_worker_stale(heartbeat_age=heartbeat_age, key_ttl_seconds=key_ttl)
            meta = _worker_meta(worker_name)
            worker_payload = {
                "name": worker_name,
                "state": str(getattr(worker, "state", "unknown")),
                "queues": worker_queues,
                "host_id": meta["host_id"],
                "role": meta["role"],
                "birth_date": _isoformat(getattr(worker, "birth_date", None)),
                "last_heartbeat": _isoformat(heartbeat),
                "heartbeat_age_seconds": heartbeat_age,
                "key_ttl_seconds": key_ttl,
                "stale": stale,
                "current_job_id": _safe_current_job_id(worker),
                "successful_job_count": _safe_count(getattr(worker, "successful_job_count", 0)),
                "failed_job_count": _safe_count(getattr(worker, "failed_job_count", 0)),
                "total_working_time": _safe_count(getattr(worker, "total_working_time", 0)),
            }
            workers.append(worker_payload)
            if stale:
                stale_workers.append(worker_payload)
            else:
                live_workers.append(worker_payload)

        payload.update(
            {
                "status": "ok",
                "queues": queue_payloads,
                "workers": workers,
                "live_workers": live_workers,
                "stale_workers": stale_workers,
                "summary": {
                    "active_workers": len(live_workers),
                    "live_workers": len(live_workers),
                    "stale_workers": len(stale_workers),
                    "queued_jobs": queued_jobs,
                },
            }
        )
    except QueueUnavailableError as exc:
        payload.update({"status": "unavailable", "error": str(exc), "error_code": exc.error_code})
    except Exception as exc:
        payload.update({"status": "error", "error": str(exc)})
    return payload


def cleanup_stale_amanaje_workers(queue_names: list[str] | None = None) -> dict[str, Any]:
    queues_to_check = configured_queue_names(queue_names)
    result = {"status": "unknown", "removed": [], "skipped": [], "stale_after_seconds": WORKER_STALE_SECONDS}
    try:
        from rq import Worker

        connection = get_redis_connection()
        for worker in Worker.all(connection=connection):
            worker_name = str(getattr(worker, "name", "") or "")
            heartbeat_age = _heartbeat_age_seconds(getattr(worker, "last_heartbeat", None))
            key_ttl = _worker_key_ttl(connection, worker)
            if (
                not worker_name.startswith(f"{WORKER_NAME_PREFIX}-")
                or not _is_worker_stale(heartbeat_age=heartbeat_age, key_ttl_seconds=key_ttl)
            ):
                result["skipped"].append({"name": worker_name, "heartbeat_age_seconds": heartbeat_age, "key_ttl_seconds": key_ttl})
                continue
            try:
                register_death = getattr(worker, "register_death", None)
                if callable(register_death):
                    register_death()
                key = getattr(worker, "key", f"rq:worker:{worker_name}")
                connection.delete(key)
                connection.srem("rq:workers", key, worker_name)
                for queue_name in queues_to_check:
                    connection.srem(f"rq:workers:{queue_name}", key, worker_name)
                result["removed"].append({"name": worker_name, "heartbeat_age_seconds": heartbeat_age, "key_ttl_seconds": key_ttl})
            except Exception as exc:
                result["skipped"].append({"name": worker_name, "error": str(exc), "heartbeat_age_seconds": heartbeat_age, "key_ttl_seconds": key_ttl})
        result["status"] = "ok"
    except QueueUnavailableError as exc:
        result.update({"status": "unavailable", "error": str(exc), "error_code": exc.error_code})
    except Exception as exc:
        result.update({"status": "error", "error": str(exc)})
    return result


def _record_job_failure(run_id: str, exc: BaseException, *, stage: str = "failed") -> None:
    from app.utils.main_utils import RUN_LEDGER_DIR
    from app.utils.run_ledger import append_run_terminal_line, update_run_entry

    traceback_text = _redact_failure_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    failure = {
        "type": exc.__class__.__name__,
        "message": _redact_failure_text(str(exc)),
        "traceback": traceback_text,
        "worker_name": os.getenv("AMANAJE_EFFECTIVE_WORKER_NAME") or os.getenv("AMANAJE_WORKER_NAME"),
        "host_id": WORKER_HOST_ID,
        "role": WORKER_ROLE,
        "queue": os.getenv("AMANAJE_WORKER_QUEUE"),
        "job_id": run_id,
        "failed_at": datetime.now().isoformat(),
    }
    try:
        append_run_terminal_line(RUN_LEDGER_DIR, run_id, message=traceback_text, stream="stderr", stage=stage)
        update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="failed",
            stage=stage,
            merge={"error": failure["message"], "failure": failure},
            event_message=f"{failure['type']}: {failure['message']}",
        )
    except Exception:
        pass


def _run_with_failure_ledger(run_id: str, work: Callable[[], Any]) -> Any:
    try:
        return work()
    except Exception as exc:
        _record_job_failure(run_id, exc)
        raise


def run_training_job(run_id: str, model_id: int, payload_data: Mapping[str, Any]) -> None:
    def _work() -> None:
        import main_app

        asyncio.run(main_app._execute_training_run(run_id, model_id=model_id, payload_data=dict(payload_data)))

    _run_with_failure_ledger(run_id, _work)


def run_study_job(run_id: str, study_id: int, payload_data: Mapping[str, Any]) -> None:
    def _work() -> None:
        import main_app

        asyncio.run(main_app._execute_study_run(run_id, study_id=study_id, payload_data=dict(payload_data)))

    _run_with_failure_ledger(run_id, _work)


def run_editor_execution_job(run_id: str, code: str, registry_context: Mapping[str, Any] | None = None) -> None:
    from app.utils.editor_execution import execute_editor_code
    from app.utils.main_utils import RUN_LEDGER_DIR
    from app.utils.run_ledger import append_run_terminal_line, update_run_entry

    def _append_stdout(line: str) -> None:
        append_run_terminal_line(RUN_LEDGER_DIR, run_id, message=line, stream="stdout", stage="executing")

    def _append_stderr(line: str) -> None:
        append_run_terminal_line(RUN_LEDGER_DIR, run_id, message=line, stream="stderr", stage="executing")

    def _work() -> None:
        update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="running",
            stage="executing",
            event_message="Editor execution started.",
        )
        result = execute_editor_code(
            code,
            registry_context=registry_context,
            stdout_callback=_append_stdout,
            stderr_callback=_append_stderr,
        )
        if result.get("error"):
            failure = {
                "type": "EditorExecutionError",
                "message": "Editor execution completed with an error.",
                "traceback": _redact_failure_text(str(result.get("error") or "")),
                "worker_name": os.getenv("AMANAJE_EFFECTIVE_WORKER_NAME") or os.getenv("AMANAJE_WORKER_NAME"),
                "host_id": WORKER_HOST_ID,
                "role": WORKER_ROLE,
                "queue": INTERACTIVE_QUEUE_NAME,
                "job_id": run_id,
                "failed_at": datetime.now().isoformat(),
            }
            update_run_entry(
                RUN_LEDGER_DIR,
                run_id,
                status="failed",
                stage="failed",
                merge={"result": result, "error": result.get("error"), "failure": failure},
                event_message="Editor execution completed with an error.",
            )
            append_run_terminal_line(RUN_LEDGER_DIR, run_id, message=failure["traceback"], stream="stderr", stage="failed")
            return

        update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="completed",
            stage="completed",
            merge={
                "result": result,
                "metrics": {"variable_count": len(result.get("variables") or {})},
            },
            event_message="Editor execution completed.",
        )

    _run_with_failure_ledger(run_id, _work)


def run_assistant_operation_job(run_id: str, operation: str, payload_data: Mapping[str, Any]) -> None:
    def _work() -> None:
        import main_app

        asyncio.run(
            main_app._execute_assistant_operation_job(
                run_id,
                operation=operation,
                payload_data=dict(payload_data),
            )
        )

    _run_with_failure_ledger(run_id, _work)
