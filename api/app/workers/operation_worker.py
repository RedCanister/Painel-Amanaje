from __future__ import annotations

import os
import sys

from app.utils.operation_queue import DEFAULT_QUEUE_NAME, REDIS_URL, get_redis_connection, worker_host_contract


def main() -> int:
    try:
        from rq import Worker
    except Exception as exc:
        print(f"RQ is not installed: {exc}", file=sys.stderr)
        return 2

    queue_names = sys.argv[1:] or [os.getenv("AMANAJE_WORKER_QUEUE", DEFAULT_QUEUE_NAME)]
    connection = get_redis_connection()
    worker_name = os.getenv("AMANAJE_WORKER_NAME") or f"amanaje-{os.getenv('AMANAJE_WORKER_HOST_ID', 'worker')}-{os.getpid()}"
    try:
        worker = Worker(queue_names, connection=connection, name=worker_name)
    except TypeError:
        worker = Worker(queue_names, connection=connection)
    print(
        "Starting Amanaje worker "
        f"name={worker_name} queues={queue_names} redis={REDIS_URL} "
        f"host={worker_host_contract(queue_names)}",
        flush=True,
    )
    print("Amanaje worker is ready and waiting for jobs.", flush=True)
    worker.work(logging_level=os.getenv("AMANAJE_WORKER_LOG_LEVEL", "INFO"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
