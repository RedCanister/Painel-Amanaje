from __future__ import annotations

import asyncio
import json
import shutil
import uuid

import main_app


def _payload(response):
    return json.loads(response.body.decode("utf-8"))


def _use_workspace_run_dir(monkeypatch, request):
    run_dir = main_app.RUN_LEDGER_DIR / f"pytest_assistant_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    request.addfinalizer(lambda: shutil.rmtree(run_dir, ignore_errors=True))
    monkeypatch.setattr(main_app, "RUN_LEDGER_DIR", run_dir)
    return run_dir


def test_assistant_routes_are_registered():
    route_paths = {route.path for route in main_app.app.routes}

    assert {
        "/assistant/sessions",
        "/assistant/draft",
        "/assistant/review",
        "/assistant/approve",
        "/assistant/submit",
        "/execution/review",
        "/execution/run",
    }.issubset(route_paths)


def test_assistant_dataset_draft_can_be_reviewed_approved_and_prepared(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)

    draft_response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create a small synthetic dataset for model training.",
                target_type="dataset_generation",
            )
        )
    )
    draft_payload = _payload(draft_response)

    assert draft_response.status_code == 200
    assert draft_payload["status"] == "reviewed"
    assert draft_payload["review"]["approved"] is True
    assert draft_payload["draft"]["code"]
    assert "csv_text = df.to_csv(index=False)" in draft_payload["draft"]["code"]

    approval_response = asyncio.run(
        main_app.assistant_approve(
            main_app.AssistantApprovalRequest(
                run_id=draft_payload["run_id"],
                draft=main_app.WorkflowDraft.model_validate(draft_payload["draft"]),
                reviewer="pytest",
            )
        )
    )
    approval_payload = _payload(approval_response)

    assert approval_response.status_code == 200
    assert approval_payload["status"] == "approved"

    submit_response = asyncio.run(
        main_app.assistant_submit(
            main_app.AssistantSubmitRequest(run_id=draft_payload["run_id"], action="prepare"),
            db=object(),
        )
    )
    submit_payload = _payload(submit_response)

    assert submit_response.status_code == 200
    assert submit_payload["status"] == "submitted"
    assert submit_payload["submission"]["next_endpoint"] == "/execution/review"


def test_execution_review_blocks_file_and_process_access():
    class FakeRequest:
        async def json(self):
            return {"code": "import os\nos.system('echo unsafe')\nopen('x.txt', 'w')"}

    response = asyncio.run(main_app.execution_review(FakeRequest()))
    payload = _payload(response)

    assert response.status_code == 400
    assert payload["status"] == "needs_revision"
    assert payload["approved"] is False
    violation_codes = {violation["code"] for violation in payload["safety"]["violations"]}
    assert {"blocked-import", "blocked-attribute-call", "blocked-call"}.issubset(violation_codes)


def test_model_generation_draft_allows_project_model_dependencies(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)

    response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create a scikit-learn joblib model artifact.",
                target_type="model_generation",
            )
        )
    )
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["status"] == "reviewed"
    assert payload["review"]["approved"] is True
    assert payload["review"]["safety"]["profile"] == "model_generation"
    assert payload["review"]["safety"]["materialized_output_checks"]["joblib_bytes"] is True
    assert "joblib_bytes" in payload["draft"]["code"]


def test_execution_run_materializes_only_approved_dataset_outputs(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'feature_a': [1, 2], 'target': [0, 1]})\n"
        "csv_text = df.to_csv(index=False)\n"
        "name = 'assistant_dataset'\n"
        "description = 'ok'\n"
        "secret_value = 'hidden'\n"
    )

    response = asyncio.run(
        main_app.execution_run(
            main_app.ExecutionRunRequest(
                code=code,
                profile="dataset_generation",
                approved=True,
            )
        )
    )
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["status"] == "success"
    assert "csv_text" in payload["metadata"]
    assert "name" in payload["variables"]
    assert "secret_value" not in payload["variables"]


def test_registry_assistant_drafts_all_manual_registry_payloads(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)
    expected_fields = {
        "DatasetModel": "shape",
        "LearningModel": "parameters",
        "InferenceModel": "inference_params",
        "StudyModel": "study_params",
        "CodeModel": "code",
    }

    for registry_type, expected_field in expected_fields.items():
        response = asyncio.run(
            main_app.assistant_draft(
                main_app.AssistantDraftRequest(
                    prompt=f"Draft a {registry_type} registry entry.",
                    target_type="registry_object",
                    context={"registryType": registry_type, "name": f"{registry_type}_assistant"},
                )
            )
        )
        payload = _payload(response)

        assert response.status_code == 200
        assert payload["review"]["approved"] is True
        assert payload["draft"]["registry_type"] == registry_type
        assert expected_field in payload["draft"]["form_payload"]


def test_training_draft_requires_model_and_dataset_before_approval(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)

    draft_response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Launch a training run once the user picks a model and dataset.",
                target_type="training_run",
            )
        )
    )
    draft_payload = _payload(draft_response)

    assert draft_response.status_code == 200
    assert draft_payload["status"] == "needs_revision"
    assert draft_payload["review"]["approved"] is False
    assert "Choose a learning model and dataset." in draft_payload["review"]["required_actions"]

    approval_response = asyncio.run(
        main_app.assistant_approve(
            main_app.AssistantApprovalRequest(
                run_id=draft_payload["run_id"],
                draft=main_app.WorkflowDraft.model_validate(draft_payload["draft"]),
                reviewer="pytest",
            )
        )
    )
    approval_payload = _payload(approval_response)

    assert approval_response.status_code == 400
    assert approval_payload["status"] == "error"
    assert approval_payload["review"]["approved"] is False
