from __future__ import annotations

import json
import sys

from app.utils.operation_queue import DEFAULT_QUEUE_NAME, QueueUnavailableError, get_worker_runtime_status


def main() -> int:
    queue_names = sys.argv[1:] or [DEFAULT_QUEUE_NAME]
    payload = get_worker_runtime_status(queue_names)
    print(json.dumps(payload, default=str), flush=True)
    if payload.get("status") == "ok":
        return 0
    if payload.get("error_code") == QueueUnavailableError.error_code:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
