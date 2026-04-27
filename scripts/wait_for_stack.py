from __future__ import annotations

import argparse

from api.tests.support import wait_for_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for the Painel Amanaje full stack to become healthy.")
    parser.add_argument("--app-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mlflow-url", default="http://127.0.0.1:5000")
    parser.add_argument("--postgres-host", default="127.0.0.1")
    parser.add_argument("--postgres-port", type=int, default=5432)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wait_for_stack(
        app_url=args.app_url,
        mlflow_url=args.mlflow_url,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        timeout_seconds=args.timeout,
    )


if __name__ == "__main__":
    main()
