from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "api"
DEFAULT_OUTPUT = API_ROOT / "runtime_artifacts" / "examples" / "painel_amanaje_examples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Painel Amanaje example catalog as JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Destination JSON path.")
    parser.add_argument(
        "--without-payloads",
        action="store_true",
        help="Export metadata and payload field names without full sample payloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(API_ROOT) not in sys.path:
        sys.path.insert(0, str(API_ROOT))

    import main_app
    from app.utils.example_catalog import build_example_catalog

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = build_example_catalog(
        routes=main_app.app.routes,
        include_payloads=not args.without_payloads,
    )
    output_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": catalog["status"],
                "output": str(output_path),
                "object_count": catalog["summary"]["object_count"],
                "sample_count": catalog["summary"]["sample_count"],
                "route_count": catalog["api_proof"]["route_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
