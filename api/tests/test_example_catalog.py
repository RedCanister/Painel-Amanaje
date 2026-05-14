from __future__ import annotations

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import main_app
from app.utils.example_catalog import SCALE_TIERS, build_example_catalog


def _app_route_pairs() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for route in main_app.app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in sorted((route.methods or set()) - {"HEAD", "OPTIONS"}):
            pairs.add((method, route.path))
    return pairs


def test_example_catalog_builds_five_validated_samples_for_every_object():
    catalog = build_example_catalog(routes=main_app.app.routes)

    assert catalog["status"] == "ok"
    assert catalog["summary"]["samples_per_object"] == 5
    assert catalog["summary"]["validation_error_count"] == 0
    assert catalog["summary"]["sample_count"] == catalog["summary"]["object_count"] * 5
    assert catalog["summary"]["object_count"] >= 70

    expected_sizes = [tier["declared_size_mb"] for tier in SCALE_TIERS]
    families = {entry["family"] for entry in catalog["objects"]}
    assert {"registry_v1", "assistant_workflow", "runtime_requests", "legacy_pydantic", "registry_v2"} <= families

    for entry in catalog["objects"]:
        assert len(entry["samples"]) == 5
        assert [sample["declared_size_mb"] for sample in entry["samples"]] == expected_sizes
        assert all(sample.get("payload") for sample in entry["samples"])


def test_example_catalog_api_proof_covers_every_registered_fastapi_route():
    catalog = build_example_catalog(routes=main_app.app.routes, include_payloads=False)

    expected_pairs = _app_route_pairs()
    proof_pairs = {
        (route["method"], route["path"])
        for route in catalog["api_proof"]["routes"]
    }

    assert expected_pairs <= proof_pairs
    assert ("GET", "/examples/catalog") in proof_pairs
    assert ("GET", "/assistant") in proof_pairs
    assert ("POST", "/assistant/draft") in proof_pairs
    assert ("POST", "/training/{model_id}") in proof_pairs
    assert catalog["api_proof"]["route_count"] == len(proof_pairs)
    assert catalog["api_proof"]["frontend_route_count"] >= 10
    assert any(route["body_sample"] for route in catalog["api_proof"]["routes"] if route["method"] == "POST")


def test_example_catalog_routes_return_catalog_and_single_object_samples():
    client = TestClient(main_app.app)

    response = client.get("/examples/catalog?include_payloads=false")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["summary"]["sample_count"] == payload["summary"]["object_count"] * 5

    object_response = client.get("/examples/catalog/AssistantModel")
    object_payload = object_response.json()

    assert object_response.status_code == 200
    assert object_payload["status"] == "ok"
    assert object_payload["object"]["object_name"] == "AssistantModel"
    assert object_payload["object"]["sample_count"] == 5
    assert object_payload["object"]["samples"][-1]["payload"]["parameters"]["assistant"]["supported_draft_types"]

    missing_response = client.get("/examples/catalog/NoSuchObject")
    assert missing_response.status_code == 404
