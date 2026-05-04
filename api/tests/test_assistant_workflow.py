from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import main_app
from app.models.assistant_objects import AssistantReferenceRequest
from app.models.model_objects import AssistantModel
from app.utils import assistant_llmops
from app.utils.assistant_provider import (
    AmanajeSLMProvider,
    OpenAICompatibleAssistantProvider,
    assistant_model_to_provider_config,
    get_assistant_provider_for_model,
)


def _payload(response):
    return json.loads(response.body.decode("utf-8"))


def _use_workspace_run_dir(monkeypatch, request):
    run_dir = main_app.RUN_LEDGER_DIR / f"pytest_assistant_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    assistant_root = run_dir / "assistant_artifacts"
    assistant_model_dir = assistant_root / "assistant_models"
    assistant_dataset_dir = assistant_root / "assistant_datasets"
    assistant_eval_dir = assistant_root / "assistant_evals"
    assistant_reference_dir = assistant_dataset_dir / "references"
    assistant_model_dir.mkdir(parents=True, exist_ok=True)
    assistant_dataset_dir.mkdir(parents=True, exist_ok=True)
    assistant_eval_dir.mkdir(parents=True, exist_ok=True)
    assistant_reference_dir.mkdir(parents=True, exist_ok=True)
    request.addfinalizer(lambda: shutil.rmtree(run_dir, ignore_errors=True))
    monkeypatch.setenv("AMANAJE_ASSISTANT_MLFLOW_ENABLED", "false")
    for env_name in (
        "AMANAJE_ASSISTANT_ACTIVE_PROVIDER",
        "AMANAJE_ASSISTANT_PROVIDER_CONFIGS",
        "AMANAJE_ASSISTANT_PROVIDER_CONFIG_PATH",
        "AMANAJE_SLM_ENABLED",
        "AMANAJE_SLM_BASE_URL",
        "AMANAJE_SLM_MODEL_NAME",
        "AMANAJE_SLM_API_KEY",
        "REMOTE_LAB_SLM_API_KEY",
    ):
        monkeypatch.delenv(env_name, raising=False)
    OpenAICompatibleAssistantProvider._cache.clear()
    OpenAICompatibleAssistantProvider._last_status_by_provider.clear()
    monkeypatch.setattr(main_app, "RUN_LEDGER_DIR", run_dir)
    monkeypatch.setattr(main_app, "ASSISTANT_MODEL_DIR", assistant_model_dir)
    monkeypatch.setattr(main_app, "ASSISTANT_DATASET_DIR", assistant_dataset_dir)
    monkeypatch.setattr(main_app, "ASSISTANT_EVAL_DIR", assistant_eval_dir)
    monkeypatch.setattr(main_app, "ASSISTANT_REFERENCE_DIR", assistant_reference_dir)
    return run_dir


def test_assistant_routes_are_registered():
    route_paths = {route.path for route in main_app.app.routes}

    assert {
        "/assistant",
        "/assistant/management/overview",
        "/assistant/sessions",
        "/assistant/draft",
        "/assistant/review",
        "/assistant/approve",
        "/assistant/submit",
        "/assistant/alignment/contracts",
        "/assistant/references",
        "/assistant/references/search",
        "/assistant/references/status",
        "/assistant/references/sync",
        "/assistant/feedback",
        "/assistant/provider/status",
        "/assistant/provider/test",
        "/assistant/datasets/rebuild",
        "/assistant/datasets/status",
        "/assistant/datasets/latest",
        "/assistant/models/list",
        "/assistant/models/{assistant_model_id}/status",
        "/assistant/models/{assistant_model_id}/test",
        "/assistant/models/{assistant_model_id}/training/curate",
        "/assistant/models/{assistant_model_id}/training/dataset/attach",
        "/assistant/contracts/workflow-draft.schema",
        "/assistant/evals/run",
        "/assistant/training/curate",
        "/assistantmodel/create",
        "/assistantmodel/list",
        "/execution/review",
        "/execution/run",
    }.issubset(route_paths)

    management_routes = {route["path"] for route in main_app._assistant_route_catalog()}
    assert "/assistant/management/overview" in management_routes
    assert "/execution/run" in management_routes
    assert "/mlflow/experiments" in management_routes


def test_assistant_training_dataset_snapshot_redacts_and_shapes(monkeypatch, request):
    run_dir = _use_workspace_run_dir(monkeypatch, request)
    interaction = {
        "label": "positive",
        "prompt": "Create a dataset. api_key=secret-value",
        "workflow_type": "dataset_generation",
        "context_pack_hash": "ctx-hash",
        "draft": {"title": "Dataset Draft"},
        "review": {"status": "approved", "approved": True},
        "user_edits": {"description": "Better metadata"},
    }
    interactions_path = main_app.ASSISTANT_DATASET_DIR / "interactions.jsonl"
    interactions_path.write_text(json.dumps(interaction) + "\n", encoding="utf-8")
    assistant_llmops.create_assistant_reference(
        main_app.ASSISTANT_REFERENCE_DIR,
        AssistantReferenceRequest(
            source_type="project_file",
            name="keyboard_rules",
            file_name="keyboard_rules.py",
            content_text="def keyboard_quality(row):\n    return row['switch_type'] == 'linear'\n",
            trust_level="project",
            tags=["dataset", "keyboard"],
            metadata={"source_path": "assistant_references/keyboard_rules.py", "source_extension": ".py"},
        ),
    )

    manifest = assistant_llmops.build_assistant_training_dataset_snapshot(
        main_app.ASSISTANT_DATASET_DIR,
        main_app.ASSISTANT_EVAL_DIR,
        run_dir,
        reference_dir=main_app.ASSISTANT_REFERENCE_DIR,
        registry_records={
            "datasets": [
                {
                    "id": 11,
                    "name": "keyboard_test_dataset",
                    "object_type": "dataset",
                    "dataset_type": "tabular",
                    "shape": [25, 4],
                    "features_list": ["switch_type", "layout", "price", "target"],
                    "path": "runtime_artifacts/uploads/keyboard_test_dataset.csv",
                }
            ],
            "learning_models": [
                {
                    "id": 12,
                    "name": "keyboard_classifier",
                    "object_type": "learning_model",
                    "model_type": "supervised_model",
                    "parameters": {"framework": "sklearn", "algorithm": "RandomForestClassifier"},
                    "metrics": {"accuracy": 0.91},
                    "input_features": ["switch_type", "layout", "price"],
                    "output_features": ["target"],
                    "path": "runtime_artifacts/models/keyboard_classifier.joblib",
                }
            ],
            "code_models": [
                {
                    "id": 13,
                    "name": "keyboard_dataset_script",
                    "object_type": "code_model",
                    "path": "runtime_artifacts/code/keyboard_dataset_script.py",
                    "code": {"script": "api_key=secret-value\n\ndef build_keyboard_dataset():\n    return 'keyboard csv'"},
                    "variables": {"rows": 25},
                }
            ],
        },
    )
    jsonl_path = Path(manifest["canonical_jsonl_path"])
    csv_path = Path(manifest["tabular_csv_path"])
    dataset_text = jsonl_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in dataset_text.splitlines() if line.strip()]
    source_types = {record["source_type"] for record in records}

    assert manifest["dataset_type"] == "assistant_training_dataset"
    assert manifest["materialization_format"] == "jsonl+csv"
    assert manifest["dataset_path"] == str(csv_path)
    assert jsonl_path.exists()
    assert csv_path.exists()
    assert manifest["record_count"] >= 1
    assert manifest["shape"] == [manifest["record_count"], len(manifest["features_list"])]
    assert manifest["dataset_hash"]
    assert manifest["tabular_hash"]
    assert manifest["generic_training_target"] == "assistant_quality_label"
    assert {"content", "registry_metadata", "asset_summary", "source_hash", "assistant_success", "assistant_quality_label"}.issubset(set(manifest["features_list"]))
    assert {"content_reference", "registry_dataset", "registry_learning_model", "registry_code_model"}.issubset(source_types)
    assert manifest["registry_record_counts"]["datasets"] == 1
    assert "keyboard_test_dataset" in dataset_text
    assert "keyboard_classifier" in dataset_text
    assert "build_keyboard_dataset" in dataset_text
    assert "keyboard_quality" in dataset_text
    assert "secret-value" not in dataset_text
    assert "secret-value" not in csv_text
    assert "[REDACTED]" in dataset_text
    assert "[REDACTED]" in csv_text

    dataset_record = SimpleNamespace(
        id=29,
        name="assistant_training_dataset_latest",
        dataset_type="assistant_training_dataset",
        path=str(csv_path),
        connection_string=str(manifest["manifest_path"]),
    )
    dataframe = main_app._utils._load_dataset_frame_from_record(dataset_record)
    assert list(dataframe.columns)[-1] == "assistant_quality_label"
    assert dataframe.shape == tuple(manifest["shape"])
    assert not any(isinstance(value, (dict, list)) for value in dataframe.head(20).to_numpy().reshape(-1))

    csv_path.unlink()
    assert not csv_path.exists()
    fallback_frame = main_app._utils._load_dataset_frame_from_record(dataset_record)
    assert csv_path.exists()
    assert list(fallback_frame.columns)[-1] == "assistant_quality_label"
    prepared = main_app._utils._prepare_dataset_for_training(
        dataset_record,
        SimpleNamespace(
            id=30,
            name="assistant_dataset_probe",
            model_type="supervised_model",
            parameters={"framework": "sklearn", "test_size": 0.4},
            input_features=None,
            output_features=None,
        ),
        {"framework": "sklearn", "test_size": 0.4},
    )
    assert prepared.output_feature == "assistant_quality_label"
    assert "assistant_quality_label" not in prepared.input_features


def test_assistant_training_registry_records_serializes_orm_rows_and_skips_non_mappings(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)
    dataset_row = main_app.DatasetORM(
        id=41,
        name="keyboard_rows",
        legacy_name="keyboard_rows",
        description="Keyboard test rows",
        object_type="dataset",
        size=0.1,
        path="runtime_artifacts/uploads/keyboard_rows.csv",
        dataset_type="tabular",
        shape=[10, 3],
        features_list=["switch", "layout", "target"],
    )
    assistant_dataset_row = main_app.DatasetORM(
        id=42,
        name="assistant_training_dataset_latest",
        legacy_name="assistant_training_dataset_latest",
        object_type="dataset",
        size=0.1,
        path="runtime_artifacts/assistant_datasets/latest.jsonl",
        dataset_type="assistant_training_dataset",
        shape=[10, 14],
    )
    learning_row = main_app.LearningORM(
        id=43,
        name="keyboard_classifier",
        object_type="learning_model",
        size=1.0,
        path="runtime_artifacts/models/keyboard_classifier.joblib",
        model_type="supervised_model",
        parameters={"framework": "sklearn"},
        metrics={"accuracy": 0.9},
        input_features=["switch", "layout"],
        output_features=["target"],
    )
    assistant_learning_row = main_app.LearningORM(
        id=44,
        name="assistant_model",
        object_type="learning_model",
        size=1.0,
        path="runtime_artifacts/assistant_models/model.json",
        model_type="assistant_model",
        parameters={"assistant": {"model_name": "amanaje"}},
        metrics={},
    )

    async def fake_get_all_entries(_db, orm_model):
        if orm_model is main_app.DatasetORM:
            return [dataset_row, assistant_dataset_row, "legacy scalar payload"]
        if orm_model is main_app.LearningORM:
            return [learning_row, assistant_learning_row]
        return []

    monkeypatch.setattr(main_app, "get_all_entries", fake_get_all_entries)

    assert isinstance(main_app._utils._json_payload(dataset_row), dict)
    records = asyncio.run(main_app._assistant_training_registry_records(object()))

    assert [record["name"] for record in records["datasets"]] == ["keyboard_rows"]
    assert records["datasets"][0]["features_list"] == ["switch", "layout", "target"]
    assert [record["name"] for record in records["learning_models"]] == ["keyboard_classifier"]
    assert records["learning_models"][0]["parameters"]["framework"] == "sklearn"


def test_assistant_model_packs_provider_config_into_learning_parameters():
    model = AssistantModel(
        id=7,
        name="panel_coding_assistant",
        description="Registry-backed assistant runtime",
        object_type="learning_model",
        size=0,
        path="runtime_artifacts/assistant_models/panel_coding_assistant",
        version=1,
        base_url="http://localhost:9100/v1",
        model_name="panel-coder",
        model_version="panel-coder-0.1",
        supported_draft_types=["dataset_generation", "registry_object"],
        temperature=0.35,
    )

    payload = model.model_dump()
    assistant_parameters = payload["parameters"]["assistant"]

    assert payload["model_type"] == "assistant_model"
    assert "base_url" not in payload
    assert assistant_parameters["base_url"] == "http://localhost:9100/v1"
    assert assistant_parameters["model_name"] == "panel-coder"
    assert assistant_parameters["supported_draft_types"] == ["dataset_generation", "registry_object"]

    config = assistant_model_to_provider_config(payload)
    assert config["type"] == "openai_compatible"
    assert config["base_url"] == "http://localhost:9100/v1"
    assert config["model_name"] == "panel-coder"
    assert config["temperature"] == 0.35


def test_registry_selected_local_assistant_model_can_create_draft():
    provider = get_assistant_provider_for_model(
        {
            "id": 8,
            "name": "deterministic_registry_assistant",
            "model_type": "assistant_model",
            "parameters": {
                "assistant": {
                    "provider_type": "local",
                    "model_version": "registry-local-0.1",
                }
            },
        }
    )

    draft = provider.create_draft(
        main_app.AssistantDraftRequest(
            prompt="Create a dataset about keyboards to test.",
            target_type="dataset_generation",
            assistant_model_id=8,
            provider=provider.name,
        )
    )

    assert draft.draft_type == "dataset_generation"
    assert "keyboards" in draft.prompt


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
    assert draft_payload["draft"]["model_version"]
    assert draft_payload["draft"]["prompt_template_version"]
    assert draft_payload["draft"]["context_pack_hash"]
    assert draft_payload["draft"]["provider_metadata"]["reference_pack_hash"]
    assert draft_payload["draft"]["provider_metadata"]["selected_reference_ids"]
    assert draft_payload["evaluation"]["checks"]["valid_workflow_draft"] is True
    assert draft_payload["evaluation"]["alignment_contracts"]["output_alignment"] is True
    assert draft_payload["evaluation"]["alignment_contracts"]["mlflow_alignment"] is True
    assert Path(draft_payload["artifact_paths"]["context_pack_path"]).exists()
    assert Path(draft_payload["artifact_paths"]["evaluation_path"]).exists()
    assert Path(draft_payload["artifact_paths"]["promotion_report_path"]).exists()
    assert draft_payload["mlflow_tracking"]["status"] == "disabled"
    context_pack = json.loads(Path(draft_payload["artifact_paths"]["context_pack_path"]).read_text(encoding="utf-8"))
    reference_types = {reference["source_type"] for reference in context_pack["content_references"]}
    assert {"alignment_contracts", "mlflow_tracking", "prompt_template"}.issubset(reference_types)

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
    assert approval_payload["ledger"]["result"]["training_example_path"]
    assert approval_payload["mlflow_tracking"]["status"] == "disabled"

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


def test_amanaje_slm_provider_adds_llmops_metadata_and_falls_back(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)

    response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create a dataset draft through the private assistant model.",
                target_type="dataset_generation",
                provider="amanaje_slm",
            )
        )
    )
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["draft"]["provider"] == "amanaje_slm"
    assert payload["draft"]["model_version"]
    assert payload["draft"]["prompt_template_version"] == "amanaje-assistant-template-v1"
    assert payload["draft"]["context_pack_id"]
    assert payload["draft"]["context_pack_hash"]
    assert payload["draft"]["evaluation_profile"] == "assistant-golden-v1"
    assert payload["draft"]["provider_metadata"]["fallback_provider"] == "local"
    assert payload["draft"]["artifacts"]["assistant_model_registry_payload"]["model_type"] == "assistant_model"


def test_amanaje_slm_provider_accepts_valid_mocked_runtime_json(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)
    monkeypatch.setenv("AMANAJE_SLM_ENABLED", "true")
    AmanajeSLMProvider._cache.clear()

    def fake_runtime(self, messages, config):
        return json.dumps(
            {
                "draft_type": "dataset_generation",
                "title": "Mocked SLM Dataset",
                "summary": "SLM generated a valid dataset draft.",
                "prompt": "Create a valid dataset.",
                "provider": "amanaje_slm",
                "code": (
                    "import pandas as pd\n"
                    "df = pd.DataFrame({'feature_a': [1], 'target': [0]})\n"
                    "csv_text = df.to_csv(index=False)\n"
                    "name = 'slm_dataset'\n"
                    "description = 'ok'\n"
                    "object_type = 'dataset'\n"
                    "dataset_type = 'dataset'\n"
                ),
                "execution_profile": "dataset_generation",
                "materialized_outputs": ["csv_text"],
            }
        )

    monkeypatch.setattr(AmanajeSLMProvider, "_call_openai_compatible_runtime", fake_runtime)

    response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create a valid dataset.",
                target_type="dataset_generation",
                provider="amanaje_slm",
            )
        )
    )
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["draft"]["provider"] == "amanaje_slm"
    assert payload["draft"]["provider_metadata"]["fallback_provider"] is None
    assert payload["draft"]["provider_metadata"]["runtime"] == "openai-compatible"
    assert payload["draft"]["title"] == "Mocked SLM Dataset"
    assert payload["review"]["approved"] is True


def test_amanaje_slm_provider_falls_back_on_invalid_json(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)
    monkeypatch.setenv("AMANAJE_SLM_ENABLED", "true")
    AmanajeSLMProvider._cache.clear()
    monkeypatch.setattr(AmanajeSLMProvider, "_call_openai_compatible_runtime", lambda *_args: "not-json")

    response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create a fallback dataset.",
                target_type="dataset_generation",
                provider="amanaje_slm",
            )
        )
    )
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["draft"]["provider_metadata"]["fallback_provider"] == "local"
    assert "invalid JSON" in payload["draft"]["provider_metadata"]["fallback_reason"]
    assert "csv_text" in payload["draft"]["code"]


def test_amanaje_slm_provider_falls_back_on_timeout_and_unsafe_code(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)
    monkeypatch.setenv("AMANAJE_SLM_ENABLED", "true")
    AmanajeSLMProvider._cache.clear()

    def timeout_runtime(self, messages, config):
        raise TimeoutError("slow model")

    monkeypatch.setattr(AmanajeSLMProvider, "_call_openai_compatible_runtime", timeout_runtime)
    timeout_response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create a timeout fallback dataset.",
                target_type="dataset_generation",
                provider="amanaje_slm",
            )
        )
    )
    timeout_payload = _payload(timeout_response)
    assert timeout_response.status_code == 200
    assert timeout_payload["draft"]["provider_metadata"]["fallback_provider"] == "local"

    AmanajeSLMProvider._cache.clear()

    def unsafe_runtime(self, messages, config):
        return json.dumps(
            {
                "draft_type": "dataset_generation",
                "title": "Unsafe",
                "summary": "Unsafe draft.",
                "prompt": "unsafe",
                "code": "import os\nos.system('echo no')\ncsv_text = 'a,b\\n1,2'",
                "execution_profile": "dataset_generation",
            }
        )

    monkeypatch.setattr(AmanajeSLMProvider, "_call_openai_compatible_runtime", unsafe_runtime)
    unsafe_response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create an unsafe draft.",
                target_type="dataset_generation",
                provider="amanaje_slm",
            )
        )
    )
    unsafe_payload = _payload(unsafe_response)
    assert unsafe_response.status_code == 200
    assert unsafe_payload["draft"]["provider_metadata"]["fallback_provider"] == "local"
    assert "failed safety review" in unsafe_payload["draft"]["provider_metadata"]["fallback_reason"]


def test_assistant_provider_status_can_run_golden_eval(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)

    response = asyncio.run(main_app.assistant_provider_status(run_eval=True))
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["providers"]["local"]["available"] is True
    assert payload["providers"]["amanaje_slm"]["provider"] == "amanaje_slm"
    assert payload["alignment_contracts"]["output_alignment"]
    assert payload["golden_eval"]["evaluation_profile"] == "assistant-golden-v1"
    assert payload["golden_eval"]["results"]
    assert Path(payload["golden_eval_artifacts"]["evaluation_suite_path"]).exists()
    assert payload["mlflow_tracking"]["status"] == "disabled"


def test_assistant_provider_registry_loads_external_config_and_redacts_secret(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)
    monkeypatch.setenv("REMOTE_LAB_SLM_API_KEY", "secret-provider-token")
    monkeypatch.setenv("AMANAJE_ASSISTANT_ACTIVE_PROVIDER", "remote_lab_slm")
    monkeypatch.setenv(
        "AMANAJE_ASSISTANT_PROVIDER_CONFIGS",
        json.dumps(
            {
                "providers": {
                    "remote_lab_slm": {
                        "type": "openai_compatible",
                        "enabled": True,
                        "base_url": "https://assistant-runtime.example.com/v1",
                        "model_name": "remote-amanaje-draft-model",
                        "model_version": "remote-0.1",
                        "api_key_env": "REMOTE_LAB_SLM_API_KEY",
                    }
                }
            }
        ),
    )

    status_response = asyncio.run(main_app.assistant_provider_status(run_eval=False))
    status_payload = _payload(status_response)

    assert status_response.status_code == 200
    assert status_payload["active_provider"] == "remote_lab_slm"
    assert "remote_lab_slm" in status_payload["providers"]
    remote_status = status_payload["providers"]["remote_lab_slm"]
    assert remote_status["type"] == "openai_compatible"
    assert remote_status["config"]["api_key_env"] == "[REDACTED]"
    assert remote_status["config"]["api_key_configured"] is True
    assert "secret-provider-token" not in json.dumps(status_payload)


def test_assistant_draft_uses_active_external_provider(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)
    monkeypatch.setenv("AMANAJE_ASSISTANT_ACTIVE_PROVIDER", "remote_lab_slm")
    monkeypatch.setenv(
        "AMANAJE_ASSISTANT_PROVIDER_CONFIGS",
        json.dumps(
            {
                "providers": {
                    "remote_lab_slm": {
                        "type": "openai_compatible",
                        "enabled": True,
                        "base_url": "http://remote.test/v1",
                        "model_name": "remote-amanaje-draft-model",
                        "model_version": "remote-0.1",
                    }
                }
            }
        ),
    )
    OpenAICompatibleAssistantProvider._cache.clear()

    def fake_runtime(self, messages, config):
        return json.dumps(
            {
                "draft_type": "dataset_generation",
                "title": "Remote Provider Dataset",
                "summary": "External provider generated a valid dataset draft.",
                "prompt": "Create a provider-selected dataset.",
                "provider": "remote_lab_slm",
                "code": (
                    "import pandas as pd\n"
                    "df = pd.DataFrame({'feature_a': [1], 'target': [0]})\n"
                    "csv_text = df.to_csv(index=False)\n"
                    "name = 'remote_dataset'\n"
                    "description = 'ok'\n"
                    "object_type = 'dataset'\n"
                    "dataset_type = 'dataset'\n"
                ),
                "execution_profile": "dataset_generation",
                "materialized_outputs": ["csv_text"],
            }
        )

    monkeypatch.setattr(OpenAICompatibleAssistantProvider, "_call_openai_compatible_runtime", fake_runtime)

    response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create a provider-selected dataset.",
                target_type="dataset_generation",
            )
        )
    )
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["draft"]["provider"] == "remote_lab_slm"
    assert payload["draft"]["model_version"] == "remote-0.1"
    assert payload["draft"]["provider_metadata"]["fallback_provider"] is None
    assert payload["draft"]["title"] == "Remote Provider Dataset"


def test_assistant_provider_diagnostics_and_schema_endpoint(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)

    diagnostic_response = asyncio.run(main_app.assistant_provider_test(provider="local"))
    diagnostic_payload = _payload(diagnostic_response)
    schema_response = asyncio.run(main_app.assistant_workflow_draft_schema())
    schema_payload = _payload(schema_response)

    assert diagnostic_response.status_code == 200
    assert diagnostic_payload["provider"] == "local"
    assert diagnostic_payload["workflow_draft_valid"] is True
    assert diagnostic_payload["safety_ok"] is True
    assert schema_response.status_code == 200
    assert "properties" in schema_payload
    assert "draft_type" in schema_payload["properties"]


def test_unknown_assistant_provider_fails_clearly(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)

    response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create something.",
                target_type="dataset_generation",
                provider="missing_provider",
            )
        )
    )
    payload = _payload(response)

    assert response.status_code == 400
    assert "missing_provider" in payload["detail"]


def test_assistant_alignment_contracts_endpoint():
    response = asyncio.run(main_app.assistant_alignment_contracts_endpoint())
    payload = _payload(response)

    assert response.status_code == 200
    assert set(payload["alignment_contracts"]) == {
        "output_alignment",
        "safety_alignment",
        "registry_alignment",
        "dependency_alignment",
        "mlflow_alignment",
    }
    assert payload["mlflow_experiment"] == "assistant/amanaje_slm"


def test_assistant_eval_endpoint_persists_suite(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)

    response = asyncio.run(main_app.assistant_run_evals(provider="amanaje_slm"))
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["status"] == "passed"
    assert payload["evaluation"]["passed"] is True
    assert Path(payload["artifact_paths"]["evaluation_suite_path"]).exists()
    assert payload["mlflow_tracking"]["status"] == "disabled"


def test_assistant_reference_upload_search_and_draft_context(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)

    reference_response = asyncio.run(
        main_app.assistant_create_reference(
            main_app.AssistantReferenceRequest(
                source_type="project_doc",
                name="Dataset naming guide",
                content_text="Use customer_activity_dataset for the name. api_key=super-secret",
                tags=["dataset_generation", "naming"],
            )
        )
    )
    reference_payload = _payload(reference_response)
    reference = reference_payload["reference"]

    assert reference_response.status_code == 200
    assert reference_payload["status"] == "created"
    assert reference["reference_id"].startswith("ref_")
    assert reference["content_hash"]
    assert Path(reference_payload["artifact_paths"]["metadata_path"]).exists()
    redacted_text_path = Path(reference_payload["artifact_paths"]["text_path"])
    assert redacted_text_path.exists()
    assert "super-secret" not in redacted_text_path.read_text(encoding="utf-8")

    search_response = asyncio.run(
        main_app.assistant_search_references(
            main_app.AssistantReferenceSearchRequest(query="naming", tags=["dataset_generation"])
        )
    )
    search_payload = _payload(search_response)
    assert any(item["reference_id"] == reference["reference_id"] for item in search_payload["references"])

    draft_response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create a customer activity dataset using the naming guide.",
                target_type="dataset_generation",
            )
        )
    )
    draft_payload = _payload(draft_response)
    context_pack = json.loads(Path(draft_payload["artifact_paths"]["context_pack_path"]).read_text(encoding="utf-8"))
    attached = [
        item
        for item in context_pack["content_references"]
        if item["reference_id"] == reference["reference_id"]
    ]

    assert draft_response.status_code == 200
    assert attached
    assert attached[0]["source_type"] == "project_doc"
    assert attached[0]["content_hash"] == reference["content_hash"]
    assert "super-secret" not in json.dumps(context_pack)
    assert context_pack["reference_pack_hash"]
    assert reference["reference_id"] in context_pack["selected_reference_ids"]


def test_assistant_reference_repository_sync_feeds_auto_context(monkeypatch, request):
    run_dir = _use_workspace_run_dir(monkeypatch, request)
    source_dir = run_dir / "managed_reference_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "dataset_rules.md").write_text(
        "Dataset rule: use customer_activity_dataset for generated training examples. password=hidden-value",
        encoding="utf-8",
    )
    monkeypatch.setenv("AMANAJE_ASSISTANT_REFERENCE_PATHS", str(source_dir))

    sync_response = asyncio.run(main_app.assistant_sync_references(max_files=10))
    sync_payload = _payload(sync_response)

    assert sync_response.status_code == 200
    assert sync_payload["status"] == "synced"
    assert sync_payload["created_count"] == 1
    assert sync_payload["created"][0]["content_hash"]

    status_response = asyncio.run(main_app.assistant_references_status())
    status_payload = _payload(status_response)
    assert status_response.status_code == 200
    assert status_payload["repository"]["stored_reference_count"] >= 1

    draft_response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create a customer activity dataset from the dataset rules.",
                target_type="dataset_generation",
            )
        )
    )
    draft_payload = _payload(draft_response)
    context_pack = json.loads(Path(draft_payload["artifact_paths"]["context_pack_path"]).read_text(encoding="utf-8"))
    project_file_refs = [item for item in context_pack["content_references"] if item["source_type"] == "project_file"]

    assert project_file_refs
    assert "hidden-value" not in json.dumps(context_pack)
    assert context_pack["reference_pack_hash"] == draft_payload["draft"]["provider_metadata"]["reference_pack_hash"]


def test_assistant_feedback_records_labeled_training_example(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)

    draft = main_app.get_assistant_provider("local").create_draft(
        main_app.AssistantDraftRequest(
            prompt="Create a draft that needs clearer metadata.",
            target_type="dataset_generation",
        )
    )
    feedback_response = asyncio.run(
        main_app.assistant_feedback(
            main_app.AssistantFeedbackRequest(
                draft=draft,
                label="rejected",
                reason="too_generic",
                notes="The name and summary need more project context.",
                user_edits={"summary": "Use a project-specific dataset description."},
            )
        )
    )
    feedback_payload = _payload(feedback_response)
    training_path = Path(feedback_payload["training_example_path"])

    assert feedback_response.status_code == 200
    assert feedback_payload["status"] == "recorded"
    assert feedback_payload["label"] == "rejected"
    assert training_path.exists()
    record = json.loads(training_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["label"] == "rejected"
    assert record["outcome"]["reason"] == "too_generic"
    assert record["user_edits"]["summary"]


def test_assistant_training_curation_redacts_sensitive_values(monkeypatch, request):
    _use_workspace_run_dir(monkeypatch, request)

    draft_response = asyncio.run(
        main_app.assistant_draft(
            main_app.AssistantDraftRequest(
                prompt="Create a synthetic dataset and remember this approval context.",
                target_type="dataset_generation",
                context={"api_key": "secret-value"},
            )
        )
    )
    draft_payload = _payload(draft_response)
    approval_response = asyncio.run(
        main_app.assistant_approve(
            main_app.AssistantApprovalRequest(
                run_id=draft_payload["run_id"],
                draft=main_app.WorkflowDraft.model_validate(draft_payload["draft"]),
                reviewer="pytest",
            )
        )
    )
    assert approval_response.status_code == 200

    curate_response = asyncio.run(main_app.assistant_curate_training())
    curate_payload = _payload(curate_response)

    assert curate_response.status_code == 200
    assert curate_payload["status"] == "curated"
    assert curate_payload["manifest"]["record_count"] >= 1
    curated_path = Path(curate_payload["manifest"]["curated_path"])
    assert curated_path.exists()
    curated_text = curated_path.read_text(encoding="utf-8")
    assert "secret-value" not in curated_text
    assert "[REDACTED]" in curated_text


def test_assistant_mlflow_event_logs_expected_tracking_payload(monkeypatch, request):
    run_dir = _use_workspace_run_dir(monkeypatch, request)
    monkeypatch.setenv("AMANAJE_ASSISTANT_MLFLOW_ENABLED", "true")
    monkeypatch.setattr(assistant_llmops, "MLFLOW_AVAILABLE", True)
    captured = {"params": None, "metrics": None, "json": None, "artifacts": []}

    class FakeRunInfo:
        run_id = "assistant-run-1"

    class FakeRun:
        info = FakeRunInfo()

    @contextmanager
    def fake_managed_run(*_args, **_kwargs):
        yield FakeRun()

    monkeypatch.setattr(assistant_llmops, "managed_run", fake_managed_run)
    monkeypatch.setattr(assistant_llmops, "active_run_id", lambda: "assistant-run-1")
    monkeypatch.setattr(assistant_llmops, "set_tags", lambda tags: captured.setdefault("tags", tags))
    monkeypatch.setattr(assistant_llmops, "log_params", lambda params: captured.update({"params": params}))
    monkeypatch.setattr(assistant_llmops, "log_metrics", lambda metrics: captured.update({"metrics": metrics}))
    monkeypatch.setattr(assistant_llmops, "log_json", lambda data, **_kwargs: captured.update({"json": data}))
    monkeypatch.setattr(assistant_llmops, "log_artifact", lambda path, **_kwargs: captured["artifacts"].append(path))

    draft = main_app.get_assistant_provider("local").create_draft(
        main_app.AssistantDraftRequest(prompt="Create csv_text.", target_type="dataset_generation")
    )
    review = main_app.review_workflow_draft(draft)
    evaluation = main_app.evaluate_draft_contract(draft, review)
    artifact = run_dir / "eval.json"
    artifact.write_text("{}", encoding="utf-8")

    result = assistant_llmops.log_assistant_mlflow_event(
        "draft_review",
        draft=draft,
        review=review,
        evaluation=evaluation,
        artifact_paths={"evaluation": artifact},
    )

    assert result["status"] == "logged"
    assert result["run_id"] == "assistant-run-1"
    assert captured["params"]["prompt_template_version"] == "amanaje-assistant-template-v1"
    assert captured["metrics"]["eval_pass_rate"] == 1.0
    assert captured["json"]["alignment_contracts"]["output_alignment"]
    assert str(artifact) in captured["artifacts"]


def test_assistant_frontend_panels_are_collapsed_by_default():
    project_root = Path(__file__).resolve().parents[1]
    create_html = (project_root / "templates" / "base_create.html").read_text(encoding="utf-8")
    feature_html = (project_root / "templates" / "base_feature.html").read_text(encoding="utf-8")
    registry_html = (project_root / "templates" / "base_purple.html").read_text(encoding="utf-8")
    ui_js = (project_root / "static" / "js" / "ui-utilities.js").read_text(encoding="utf-8")

    for key, source in {"create": create_html, "feature": feature_html, "registry": registry_html}.items():
        assert f'data-assistant-toggle="{key}"' in source
        assert f'data-assistant-panel="{key}" hidden' in source
        assert f'data-assistant-status="{key}"' in source

    assert "Attach Reference" not in create_html
    assert "Attach Reference" not in feature_html
    assert "Attach Reference" not in registry_html
    assert "btnAssistantCreateAttachReference" not in create_html
    assert "btnAssistantFeatureAttachReference" not in feature_html
    assert "btnRegistryAssistantAttachReference" not in registry_html
    assert "reference_ids:" not in (project_root / "static" / "js" / "create-page.js").read_text(encoding="utf-8")
    assert "assistant-reference-tools" not in (project_root / "static" / "style.css").read_text(encoding="utf-8")
    assert "initAssistantCollapsibles" in ui_js
    assert "setAssistantStatus" in ui_js
    assert "createAssistantReference" not in ui_js


def test_assistant_management_page_covers_routes_files_and_operations():
    project_root = Path(__file__).resolve().parents[1]
    assistant_html = (project_root / "templates" / "base_assistant.html").read_text(encoding="utf-8")
    assistant_js = (project_root / "static" / "js" / "assistant-page.js").read_text(encoding="utf-8")

    for label in (
        "Overview",
        "Models",
        "Datasets",
        "References",
        "Draft Lab",
        "Safety & Execution",
        "Evals & Training",
        "Routes & Files",
    ):
        assert label in assistant_html

    assert "/assistant/management/overview" in assistant_js
    assert "/assistant/draft" in assistant_js
    assert "/execution/run" in assistant_js
    assert "/assistant/evals/run" in assistant_js
    assert "/assistant/references/sync" in assistant_js
    assert "Attach Reference" not in assistant_html
    assert 'type="file"' not in assistant_html


def test_assistant_golden_eval_contracts_cover_core_workflows():
    provider = main_app.get_assistant_provider("amanaje_slm")
    cases = [
        main_app.AssistantDraftRequest(prompt="Generate csv_text.", target_type="dataset_generation", provider="amanaje_slm"),
        main_app.AssistantDraftRequest(prompt="Generate joblib_bytes.", target_type="model_generation", provider="amanaje_slm"),
        main_app.AssistantDraftRequest(
            prompt="Draft registry payload.",
            target_type="registry_object",
            provider="amanaje_slm",
            context={"registryType": "LearningModel"},
        ),
        main_app.AssistantDraftRequest(
            prompt="Prepare training.",
            target_type="training_run",
            provider="amanaje_slm",
            context={"modelId": 1, "datasetId": 2},
        ),
        main_app.AssistantDraftRequest(
            prompt="Prepare study.",
            target_type="study",
            provider="amanaje_slm",
            context={"studyId": 3},
        ),
    ]

    for request_payload in cases:
        draft = provider.create_draft(request_payload)
        review = main_app.review_workflow_draft(draft)
        evaluation = main_app.evaluate_draft_contract(draft, review)

        assert draft.context_pack_hash
        assert draft.model_version
        assert review.safety.ok is True
        assert evaluation["checks"]["valid_workflow_draft"] is True
        assert evaluation["checks"]["has_context_pack"] is True


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
    assert draft_payload["ledger"]["result"]["training_example_path"]

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
