from __future__ import annotations

import asyncio
import os
import socket
from datetime import datetime
from typing import Any, Callable, Mapping


DEFAULT_QUEUE_NAME = os.getenv("AMANAJE_DEFAULT_QUEUE", "amanaje:default")
GPU_QUEUE_NAME = os.getenv("AMANAJE_GPU_QUEUE", "amanaje:gpu")
REDIS_URL = os.getenv("AMANAJE_REDIS_URL", os.getenv("REDIS_URL", "redis://redis:6379/0"))
WORKER_HOST_ID = os.getenv("AMANAJE_WORKER_HOST_ID", socket.gethostname())
WORKER_ROLE = os.getenv("AMANAJE_WORKER_ROLE", "cpu")
WORKER_CONCURRENCY = int(os.getenv("AMANAJE_WORKER_CONCURRENCY", "1") or "1")


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
        DEFAULT_QUEUE_NAME,
        run_editor_execution_job,
        job_id=run_id,
        kwargs={"run_id": run_id, "code": code, "registry_context": dict(registry_context or {})},
    )


def cancel_rq_job(run_id: str) -> dict[str, Any]:
    try:
        from rq.job import Job
        from rq.exceptions import NoSuchJobError
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


def enqueue_assistant_operation(run_id: str, *, operation: str, payload_data: Mapping[str, Any]) -> dict[str, Any]:
    return enqueue_callable(
        DEFAULT_QUEUE_NAME,
        run_assistant_operation_job,
        job_id=run_id,
        kwargs={"run_id": run_id, "operation": operation, "payload_data": dict(payload_data)},
    )


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _safe_count(value: Any) -> int:
    try:
        raw = value() if callable(value) else value
        return int(raw or 0)
    except Exception:
        return 0


def worker_host_contract(queue_names: list[str] | None = None) -> dict[str, Any]:
    queues = queue_names or [DEFAULT_QUEUE_NAME]
    return {
        "host_id": WORKER_HOST_ID,
        "role": WORKER_ROLE,
        "queues": queues,
        "concurrency": WORKER_CONCURRENCY,
        "concurrency_model": "one RQ worker process per container; scale hosts/replicas for parallel jobs",
        "redis_url": REDIS_URL,
    }


def get_worker_runtime_status(queue_names: list[str] | None = None) -> dict[str, Any]:
    queues_to_check = queue_names or [DEFAULT_QUEUE_NAME, GPU_QUEUE_NAME]
    payload: dict[str, Any] = {
        "status": "unknown",
        "backend": "rq",
        "redis_url": REDIS_URL,
        "configured_queues": queues_to_check,
        "worker_host_contract": worker_host_contract([DEFAULT_QUEUE_NAME]),
        "queues": [],
        "workers": [],
        "summary": {"active_workers": 0, "queued_jobs": 0},
    }
    try:
        _Redis, Queue = _load_rq()
        from rq import Worker
        from rq.registry import DeferredJobRegistry, FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry

        connection = get_redis_connection()
        queue_payloads = []
        queued_jobs = 0
        for queue_name in queues_to_check:
            queue = Queue(queue_name, connection=connection)
            count = _safe_count(getattr(queue, "count", 0))
            queued_jobs += count
            queue_payloads.append(
                {
                    "name": queue_name,
                    "queued_jobs": count,
                    "started_jobs": _safe_count(StartedJobRegistry(queue_name, connection=connection).count),
                    "deferred_jobs": _safe_count(DeferredJobRegistry(queue_name, connection=connection).count),
                    "finished_jobs": _safe_count(FinishedJobRegistry(queue_name, connection=connection).count),
                    "failed_jobs": _safe_count(FailedJobRegistry(queue_name, connection=connection).count),
                }
            )

        workers = []
        for worker in Worker.all(connection=connection):
            worker_queues = [getattr(queue, "name", str(queue)) for queue in getattr(worker, "queues", [])]
            workers.append(
                {
                    "name": getattr(worker, "name", None),
                    "state": str(getattr(worker, "state", "unknown")),
                    "queues": worker_queues,
                    "birth_date": _isoformat(getattr(worker, "birth_date", None)),
                    "last_heartbeat": _isoformat(getattr(worker, "last_heartbeat", None)),
                    "successful_job_count": _safe_count(getattr(worker, "successful_job_count", 0)),
                    "failed_job_count": _safe_count(getattr(worker, "failed_job_count", 0)),
                    "total_working_time": _safe_count(getattr(worker, "total_working_time", 0)),
                }
            )

        payload.update(
            {
                "status": "ok",
                "queues": queue_payloads,
                "workers": workers,
                "summary": {"active_workers": len(workers), "queued_jobs": queued_jobs},
            }
        )
    except QueueUnavailableError as exc:
        payload.update({"status": "unavailable", "error": str(exc), "error_code": exc.error_code})
    except Exception as exc:
        payload.update({"status": "error", "error": str(exc)})
    return payload


def run_training_job(run_id: str, model_id: int, payload_data: Mapping[str, Any]) -> None:
    import main_app

    asyncio.run(main_app._execute_training_run(run_id, model_id=model_id, payload_data=dict(payload_data)))


def run_study_job(run_id: str, study_id: int, payload_data: Mapping[str, Any]) -> None:
    import main_app

    asyncio.run(main_app._execute_study_run(run_id, study_id=study_id, payload_data=dict(payload_data)))


def run_editor_execution_job(run_id: str, code: str, registry_context: Mapping[str, Any] | None = None) -> None:
    from app.utils.editor_execution import execute_editor_code
    from app.utils.main_utils import RUN_LEDGER_DIR
    from app.utils.run_ledger import update_run_entry

    try:
        update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="running",
            stage="executing",
            event_message="Editor execution started.",
        )
        result = execute_editor_code(code, registry_context=registry_context)
        if result.get("error"):
            update_run_entry(
                RUN_LEDGER_DIR,
                run_id,
                status="failed",
                stage="failed",
                merge={"result": result, "error": result.get("error")},
                event_message="Editor execution completed with an error.",
            )
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
    except Exception as exc:
        update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="failed",
            stage="failed",
            merge={"error": str(exc)},
            event_message=str(exc),
        )
        raise


def run_assistant_operation_job(run_id: str, operation: str, payload_data: Mapping[str, Any]) -> None:
    import main_app

    asyncio.run(
        main_app._execute_assistant_operation_job(
            run_id,
            operation=operation,
            payload_data=dict(payload_data),
        )
    )
