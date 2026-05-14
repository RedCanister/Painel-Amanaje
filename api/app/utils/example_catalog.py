from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from fastapi.routing import APIRoute
from pydantic import BaseModel

from app.models import assistant_objects, model_objects, model_pydantic
from app.models.v2 import model_schemas as v2_schemas
from app.utils import main_utils


SCALE_TIERS: list[dict[str, Any]] = [
    {
        "tier": "simple",
        "declared_size_mb": 0.01,
        "rows": 12,
        "columns": 4,
        "complexity": "single-file smoke fixture",
    },
    {
        "tier": "small",
        "declared_size_mb": 0.1,
        "rows": 120,
        "columns": 8,
        "complexity": "small tabular workflow",
    },
    {
        "tier": "medium",
        "declared_size_mb": 1.0,
        "rows": 2400,
        "columns": 16,
        "complexity": "feature-engineering workflow",
    },
    {
        "tier": "large",
        "declared_size_mb": 10.0,
        "rows": 48000,
        "columns": 32,
        "complexity": "multi-stage registry workflow",
    },
    {
        "tier": "complex",
        "declared_size_mb": 100.0,
        "rows": 320000,
        "columns": 64,
        "complexity": "full production, assistant, study, and monitoring workflow",
    },
]

DOMAIN_SEEDS = [
    {
        "slug": "micro_cassava_quality",
        "title": "Micro Cassava Quality",
        "target": "quality_score",
        "features": ["starch_density", "drying_minutes", "soil_trace", "humidity_band"],
        "story": "tiny harvest-lab quality scoring",
    },
    {
        "slug": "solar_kiln_schedule",
        "title": "Solar Kiln Schedule",
        "target": "thermal_yield",
        "features": ["panel_angle", "cloud_index", "kiln_load", "vent_position"],
        "story": "solar kiln dispatch optimization",
    },
    {
        "slug": "neon_port_ledger",
        "title": "Neon Port Ledger",
        "target": "delay_risk",
        "features": ["dock_slot", "tariff_band", "fuel_spread", "crew_shift"],
        "story": "electric port logistics and risk routing",
    },
    {
        "slug": "textile_signal_forge",
        "title": "Textile Signal Forge",
        "target": "defect_probability",
        "features": ["thread_tension", "loom_speed", "dye_temperature", "pattern_entropy"],
        "story": "smart textile quality prediction",
    },
    {
        "slug": "rainfall_credit_mesh",
        "title": "Rainfall Credit Mesh",
        "target": "credit_resilience",
        "features": ["rainfall_gap", "market_distance", "cooperative_score", "reserve_ratio"],
        "story": "climate-aware microcredit resilience monitoring",
    },
]

DRAFT_TYPES = [
    "dataset_generation",
    "feature_operations",
    "model_generation",
    "registry_object",
    "training_run",
    "study",
]

REGISTRY_TYPES = [
    "DatasetModel",
    "LearningModel",
    "AssistantModel",
    "InferenceModel",
    "CodeModel",
    "StudyModel",
]

FRONTEND_ROUTES = [
    "/",
    "/upload",
    "/create",
    "/feature",
    "/training",
    "/optimization",
    "/editor",
    "/production",
    "/registry",
    "/assistant",
    "/settings",
]


def _tier(index: int) -> dict[str, Any]:
    return SCALE_TIERS[index % len(SCALE_TIERS)]


def _domain(index: int) -> dict[str, Any]:
    return DOMAIN_SEEDS[index % len(DOMAIN_SEEDS)]


def _stamp(index: int) -> str:
    base = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
    return (base + timedelta(minutes=index * 17)).isoformat()


def _history(index: int, action: str) -> list[dict[str, Any]]:
    tier = _tier(index)
    domain = _domain(index)
    return [
        {
            "operation": "example_catalog_seed",
            "action": action,
            "tier": tier["tier"],
            "declared_size_mb": tier["declared_size_mb"],
            "domain": domain["slug"],
            "when": _stamp(index),
        }
    ]


def _path(index: int, family: str, suffix: str) -> str:
    tier = _tier(index)
    domain = _domain(index)
    return f"runtime_artifacts/examples/{family}/{domain['slug']}_{tier['tier']}{suffix}"


def _registry_base(index: int, *, object_type: str, family: str, suffix: str = ".json") -> dict[str, Any]:
    tier = _tier(index)
    domain = _domain(index)
    return {
        "id": 1000 + index,
        "name": f"{domain['slug']}_{family}_{tier['tier']}",
        "description": f"{tier['complexity']} for {domain['story']}.",
        "object_type": object_type,
        "size": tier["declared_size_mb"],
        "path": _path(index, family, suffix),
        "date": _stamp(index),
        "version": index + 1,
        "history": _history(index, f"{family}_example"),
    }


def _dataset_payload(index: int) -> dict[str, Any]:
    tier = _tier(index)
    domain = _domain(index)
    dataset_types = ["dataset", "numerical_dataset", "timeseries_dataset", "text_dataset", "mixed_dataset"]
    features = domain["features"] + [domain["target"]]
    return {
        **_registry_base(index, object_type="dataset", family="dataset", suffix=".csv"),
        "dataset_type": dataset_types[index % len(dataset_types)],
        "shape": [tier["rows"], min(tier["columns"], len(features) + index + 1)],
        "has_features": True,
        "features_list": features,
        "connection_string": f"duckdb://examples/{domain['slug']}?tier={tier['tier']}",
    }


def _assistant_training_dataset_payload(index: int) -> dict[str, Any]:
    tier = _tier(index)
    payload = _dataset_payload(index)
    payload.update(
        {
            "name": f"{_domain(index)['slug']}_assistant_training_{tier['tier']}",
            "dataset_type": "assistant_training_dataset",
            "shape": [tier["rows"], 16],
            "path": _path(index, "assistant_training_dataset", ".jsonl"),
            "features_list": list(
                model_objects.AssistantTrainingDatasetModel.model_fields["features_list"].default_factory()
            ),
            "connection_string": _path(index, "assistant_training_dataset", "_manifest.json"),
        }
    )
    return payload


def _learning_payload(index: int) -> dict[str, Any]:
    tier = _tier(index)
    domain = _domain(index)
    model_types = [
        "learning_model",
        "supervised_model",
        "deep_learning_model",
        "transformative_model",
        "generative_model",
    ]
    frameworks = ["sklearn", "sklearn", "pytorch", "pytorch", "onnx"]
    return {
        **_registry_base(index, object_type="learning_model", family="learning_model", suffix=".joblib"),
        "model_type": model_types[index % len(model_types)],
        "parameters": {
            "framework": frameworks[index % len(frameworks)],
            "task_type": "regression" if index in {0, 4} else "classification",
            "input_features": domain["features"],
            "output_features": [domain["target"]],
            "estimator_class": ["DummyRegressor", "RandomForestClassifier", "FeedForwardNet", "TransformerEncoder", "ONNXRuntime"][index],
            "scale_tier": tier["tier"],
            "training_mode": ["manual", "queued", "study_seeded", "transfer", "production_shadow"][index],
        },
        "metrics": {
            "eval_accuracy": round(0.72 + index * 0.045, 4),
            "eval_f1": round(0.68 + index * 0.052, 4),
            "latency_ms": round(38.0 / (index + 1), 3),
        },
        "reference_data": _dataset_payload(index)["path"],
        "input_features": domain["features"],
        "output_features": [domain["target"]],
        "is_trained": index >= 1,
        "is_tested": index >= 2,
        "is_deployed": index >= 3,
    }


def _assistant_model_payload(index: int) -> dict[str, Any]:
    tier = _tier(index)
    domain = _domain(index)
    base = _learning_payload(index)
    base.update(
        {
            "name": f"{domain['slug']}_assistant_model_{tier['tier']}",
            "model_type": "assistant_model",
            "path": _path(index, "assistant_model", ".zip"),
            "parameters": {
                "assistant": {
                    "provider_type": "openai_compatible",
                    "runtime_kind": "pytorch_hf_server" if index >= 2 else "local_provider",
                    "model_name": f"amanaje-{domain['slug']}-{tier['tier']}",
                    "model_version": f"example-{index + 1}.0",
                    "base_model_name": "amanaje/example-base",
                    "supported_draft_types": DRAFT_TYPES[: min(len(DRAFT_TYPES), index + 2)],
                    "max_context_tokens": [1024, 2048, 4096, 8192, 16384][index],
                    "temperature": round(0.25 + index * 0.13, 2),
                    "training_profile": {
                        "dataset_path": _assistant_training_dataset_payload(index)["path"],
                        "target": "assistant_quality_label",
                    },
                    "reference_policy": {
                        "max_references": 2 + index,
                        "trust_levels": ["project", "manual", "runtime"][: min(3, index + 1)],
                    },
                    "behavior_profile": {
                        "mode": ["precise", "curious", "inventive", "skeptical", "orchestrator"][index],
                        "domain": domain["story"],
                    },
                },
                "artifact_manifest": {
                    "artifact_format": "assistant_model_bundle",
                    "scale_tier": tier["tier"],
                    "runtime_capabilities": {
                        "register": True,
                        "inspect": True,
                        "generate": True,
                        "train": index >= 3,
                    },
                },
            },
            "metrics": {
                "bundle_valid": index >= 1,
                "eval_pass_rate": round(0.76 + index * 0.045, 4),
                "draft_acceptance_rate": round(0.62 + index * 0.06, 4),
            },
            "reference_data": _assistant_training_dataset_payload(index)["path"],
            "input_features": ["prompt", "context_pack", "target_type"],
            "output_features": ["workflow_draft"],
            "provider_type": "openai_compatible",
            "runtime_kind": "pytorch_hf_server" if index >= 2 else "local_provider",
            "model_name": f"amanaje-{domain['slug']}-{tier['tier']}",
            "model_version": f"example-{index + 1}.0",
            "base_model_name": "amanaje/example-base",
            "supported_draft_types": DRAFT_TYPES[: min(len(DRAFT_TYPES), index + 2)],
            "max_context_tokens": [1024, 2048, 4096, 8192, 16384][index],
            "temperature": round(0.25 + index * 0.13, 2),
            "max_tokens": [256, 512, 1024, 2048, 4096][index],
            "enabled": True,
            "is_default": index == 0,
        }
    )
    return base


def _code_payload(index: int) -> dict[str, Any]:
    tier = _tier(index)
    domain = _domain(index)
    feature_list = ", ".join(repr(feature) for feature in domain["features"])
    script = (
        "import pandas as pd\n"
        f"features = [{feature_list}]\n"
        f"target = {domain['target']!r}\n"
        "def build_frame(rows=12):\n"
        "    return pd.DataFrame({name: range(rows) for name in features + [target]})\n"
    )
    return {
        **_registry_base(index, object_type="code_model", family="code_model", suffix=".py"),
        "code": {
            "language": "python",
            "script": script,
            "entrypoint": "build_frame",
            "api_calls": ["/features/preview", "/training/{model_id}", "/production/simulate"][: index + 1],
        },
        "variables": {
            "rows": tier["rows"],
            "columns": tier["columns"],
            "domain": domain["slug"],
            "declared_size_mb": tier["declared_size_mb"],
        },
    }


def _inference_payload(index: int) -> dict[str, Any]:
    domain = _domain(index)
    return {
        **_registry_base(index, object_type="inference_model", family="inference_model", suffix=".json"),
        "learning_model_id": 2000 + index,
        "dataset_id": 3000 + index,
        "input_features": domain["features"],
        "output_features": [domain["target"]],
        "inference_params": {
            "status": ["prepared", "validated", "active", "monitored", "drift_watch"][index],
            "batch_size": [1, 16, 64, 256, 1024][index],
            "thresholds": {
                "prediction_drift": round(0.04 + index * 0.01, 3),
                "data_drift": round(0.05 + index * 0.015, 3),
            },
            "linked_routes": ["/production/start", "/production/monitor", "/production/simulate"],
        },
    }


def _study_payload(index: int) -> dict[str, Any]:
    domain = _domain(index)
    payload = {
        **_inference_payload(index),
        "object_type": "study_model",
        "name": f"{domain['slug']}_study_{_tier(index)['tier']}",
        "path": _path(index, "study_model", ".json"),
        "sampler": ["RandomSampler", "TPESampler", "CmaEsSampler", "GridSampler", "NSGAIISampler"][index],
        "objective": ["minimize_loss", "maximize_accuracy", "maximize_f1", "minimize_latency", "maximize_resilience"][index],
        "best_trial": {
            "number": index,
            "value": round(0.84 + index * 0.025, 4),
            "state": "COMPLETE",
        },
        "best_params": {
            "learning_rate": [0.1, 0.03, 0.01, 0.003, 0.001][index],
            "hidden_dim": [8, 16, 32, 64, 128][index],
        },
        "study_params": {
            "n_trials": [3, 8, 21, 55, 144][index],
            "direction": "maximize" if index != 0 else "minimize",
            "metric": ["loss", "accuracy", "f1", "latency_ms", "credit_resilience"][index],
            "search_space": {
                "learning_rate": {"type": "float", "low": 0.0001, "high": 0.1},
                "hidden_dim": {"type": "categorical", "choices": [8, 16, 32, 64, 128]},
            },
        },
    }
    if index >= 2:
        payload["learning_model"] = _learning_payload(index)
        payload["dataset"] = _dataset_payload(index)
    return payload


def _content_reference_payload(index: int) -> dict[str, Any]:
    tier = _tier(index)
    domain = _domain(index)
    return {
        "reference_id": f"ref_example_{index + 1:02d}",
        "source_type": ["user_text", "project_doc", "project_file", "registry_snapshot", "runtime_artifact"][index],
        "name": f"{domain['title']} reference {tier['tier']}",
        "summary": f"Reference pack for {domain['story']}.",
        "trust_level": ["user", "project", "project", "runtime", "admin"][index],
        "uri": _path(index, "references", ".md"),
        "metadata": {
            "scale_tier": tier["tier"],
            "declared_size_mb": tier["declared_size_mb"],
            "source_hash_algorithm": "sha256",
        },
        "content_hash": f"sha256-example-{index + 1:02d}",
        "tags": [DRAFT_TYPES[index % len(DRAFT_TYPES)], domain["slug"], tier["tier"]],
        "created_at": _stamp(index),
    }


def _context_pack_payload(index: int) -> dict[str, Any]:
    registry_type = REGISTRY_TYPES[index % len(REGISTRY_TYPES)]
    return {
        "context_pack_id": f"ctx_example_{index + 1:02d}",
        "version": "assistant-context-pack-v1",
        "prompt": f"Prepare a {registry_type} example for {_domain(index)['story']}.",
        "workflow_goal": f"Exercise {registry_type} across the {SCALE_TIERS[index]['tier']} scale tier.",
        "target_type": DRAFT_TYPES[index % len(DRAFT_TYPES)],
        "provider": "local",
        "registry_type": registry_type,
        "form_values": {"registryType": registry_type, "scaleTier": _tier(index)["tier"]},
        "constraints": {"declared_size_mb": _tier(index)["declared_size_mb"], "max_runtime_seconds": 30 + index * 15},
        "safety_profile": DRAFT_TYPES[index % len(DRAFT_TYPES)],
        "allowed_imports": ["pandas", "numpy"] if index >= 1 else ["pandas"],
        "allowed_dependencies": ["scikit-learn"] if index >= 2 else [],
        "content_references": [_content_reference_payload(index)],
        "reference_pack_id": f"refpack_example_{index + 1:02d}",
        "reference_pack_hash": f"refpack-hash-{index + 1:02d}",
        "selected_reference_ids": [f"ref_example_{index + 1:02d}"],
        "reference_trust_levels": {f"ref_example_{index + 1:02d}": "project"},
        "compact_summary": f"{registry_type} context for {_domain(index)['slug']}.",
        "pack_hash": f"ctx-hash-{index + 1:02d}",
        "created_at": _stamp(index),
    }


def _registry_payload_for_type(registry_type: str, index: int) -> dict[str, Any]:
    mapping = {
        "DatasetModel": _dataset_payload,
        "LearningModel": _learning_payload,
        "AssistantModel": _assistant_model_payload,
        "InferenceModel": _inference_payload,
        "CodeModel": _code_payload,
        "StudyModel": _study_payload,
    }
    return mapping.get(registry_type, _dataset_payload)(index)


def _workflow_draft_payload(index: int) -> dict[str, Any]:
    registry_type = REGISTRY_TYPES[index % len(REGISTRY_TYPES)]
    draft_type = DRAFT_TYPES[index % len(DRAFT_TYPES)]
    domain = _domain(index)
    return {
        "draft_id": f"draft_example_{index + 1:02d}",
        "session_id": f"session_example_{index + 1:02d}",
        "draft_type": draft_type,
        "title": f"{domain['title']} {registry_type} Draft",
        "summary": f"Validated {draft_type} draft for {domain['story']}.",
        "prompt": f"Create a {registry_type} sample using the {domain['slug']} domain.",
        "status": "draft" if index < 2 else "reviewed",
        "provider": "local",
        "created_at": _stamp(index),
        "code": _code_payload(index)["code"]["script"] if draft_type in {"dataset_generation", "model_generation", "feature_operations"} else None,
        "operations": [
            {"type": "profile", "target": domain["slug"]},
            {"type": "validate", "registry_type": registry_type},
        ][: index + 1],
        "registry_payload": _registry_payload_for_type(registry_type, index),
        "registry_type": registry_type,
        "form_payload": _registry_payload_for_type(registry_type, index),
        "training_request": _training_request_payload(index) if draft_type == "training_run" else {},
        "study_request": _study_optimization_payload(index) if draft_type == "study" else {},
        "execution_profile": draft_type,
        "materialized_outputs": [_path(index, "materialized", ".json")],
        "artifacts": {"catalog_path": _path(index, "catalog", ".json")},
        "assumptions": [f"{domain['title']} fixtures are synthetic."],
        "risks": ["Declared 100 MB tier is metadata-only in source control."] if index == 4 else [],
        "context": {"registryType": registry_type, "scaleTier": _tier(index)["tier"]},
        "next_actions": ["review", "approve", "submit"][: min(3, index + 1)],
        "model_version": f"example-provider-{index + 1}",
        "prompt_template_version": "amanaje-assistant-template-v1",
        "context_pack_id": f"ctx_example_{index + 1:02d}",
        "context_pack_hash": f"ctx-hash-{index + 1:02d}",
        "evaluation_profile": "catalog-proof",
        "generation_provenance": _generation_provenance_payload(index),
        "provider_metadata": {"scale_tier": _tier(index)["tier"], "domain": domain["slug"]},
    }


def _generation_provenance_payload(index: int) -> dict[str, Any]:
    return {
        "source_type": "assistant",
        "provider": ["local", "amanaje_slm", "assistant_model_1", "assistant_model_2", "assistant_model_3"][index],
        "model_version": f"example-provider-{index + 1}",
        "prompt_template_version": "amanaje-assistant-template-v1",
        "context_pack_id": f"ctx_example_{index + 1:02d}",
        "context_pack_hash": f"ctx-hash-{index + 1:02d}",
        "fallback_provider": "local" if index >= 2 else None,
        "created_at": _stamp(index),
    }


def _safety_violation_payload(index: int) -> dict[str, Any]:
    return {
        "code": ["missing_review", "unsafe_import", "path_escape", "secret_literal", "unbounded_execution"][index],
        "severity": ["info", "low", "medium", "high", "critical"][index],
        "message": f"Example safety finding for {_domain(index)['story']}.",
        "line": index + 1,
        "name": f"catalog_rule_{index + 1}",
    }


def _safety_report_payload(index: int) -> dict[str, Any]:
    return {
        "ok": index < 3,
        "risk_level": ["none", "low", "medium", "high", "critical"][index],
        "profile": DRAFT_TYPES[index % len(DRAFT_TYPES)],
        "violations": [] if index < 2 else [_safety_violation_payload(index)],
        "warnings": [f"Review declared {SCALE_TIERS[index]['tier']} scale assumptions."] if index >= 1 else [],
        "allowed_imports": ["pandas", "numpy", "sklearn"][: min(3, index + 1)],
        "allowed_dependencies": ["pandas", "scikit-learn"][: min(2, index + 1)],
        "blocked_names": ["subprocess", "eval"] if index >= 3 else [],
        "materialized_output_checks": {"csv_text": index != 3, "joblib_bytes": index >= 2},
    }


def _training_request_payload(index: int) -> dict[str, Any]:
    return {
        "datasetId": 3000 + index,
        "parameters": {
            "framework": "pytorch" if index >= 2 else "sklearn",
            "input_features": _domain(index)["features"],
            "output_features": [_domain(index)["target"]],
            "epochs": [1, 3, 8, 13, 21][index],
            "batch_size": [4, 16, 32, 64, 128][index],
            "device": "cuda" if index == 4 else "cpu",
        },
        "inputType": ["Manual", "Dataset", "Study", "Transfer", "Production"][index],
        "studyId": 4000 + index if index >= 2 else None,
        "inferenceId": 5000 + index,
    }


def _study_optimization_payload(index: int) -> dict[str, Any]:
    return {
        "nTrials": [3, 8, 21, 55, 144][index],
        "outputDir": _path(index, "optuna", ""),
        "plotResults": index != 0,
        "inferenceId": 5000 + index,
        "preferGpu": index == 4,
    }


def _onnx_prepare_payload(index: int) -> dict[str, Any]:
    return {
        "model_id": 2000 + index,
        "dataset_id": 3000 + index,
        "study_id": 4000 + index if index >= 2 else None,
        "framework": "pytorch" if index >= 2 else "sklearn",
        "export_target": "onnx",
        "parameters": {
            "opset": 17,
            "dynamic_axes": index >= 2,
            "input_features": _domain(index)["features"],
        },
    }


def _legacy_dataset_payload(index: int) -> dict[str, Any]:
    payload = _dataset_payload(index)
    return {
        "id": payload["id"],
        "name": payload["name"],
        "description": payload["description"],
        "dataset_type": payload["dataset_type"],
        "source_type": ["regression", "classification", "clustering", "forecasting", "recommendation"][index],
        "size": payload["size"],
        "path": payload["path"],
        "date": payload["date"],
        "version": payload["version"],
        "history": payload["history"],
        "has_features": payload["has_features"],
        "features_list": payload["features_list"],
        "connection_string": payload["connection_string"],
    }


def _legacy_learning_payload(index: int) -> dict[str, Any]:
    payload = _learning_payload(index)
    return {
        "model_name": payload["name"],
        "size": payload["size"],
        "path": payload["path"],
        "description": payload["description"],
        "model_category": ["Regression", "Classification", "Forecasting", "Ranking", "Generative"][index],
        "date": payload["date"],
        "version": str(payload["version"]),
        "parameters": payload["parameters"],
        "metrics": payload["metrics"],
        "reference_data": payload["reference_data"],
        "input_data": payload["input_features"],
        "output_data": payload["output_features"],
        "is_trained": payload["is_trained"],
        "is_tested": payload["is_tested"],
        "is_deployed": payload["is_deployed"],
        "history": payload["history"],
    }


def _operation_payload(index: int) -> dict[str, Any]:
    return {
        "operation_name": f"{_domain(index)['slug']}_operation_{_tier(index)['tier']}",
        "description": f"Operation example for {_domain(index)['story']}.",
        "date": _stamp(index),
        "source_origin": ["manual", "assistant", "feature_workspace", "airflow", "production"][index],
    }


def _v2_object_payload(index: int, *, include_id: bool = False) -> dict[str, Any]:
    base = _registry_base(index, object_type="object", family="object", suffix=".json")
    return base if include_id else {key: value for key, value in base.items() if key != "id"}


def _v2_dataset_payload(index: int, *, include_id: bool = False) -> dict[str, Any]:
    payload = _dataset_payload(index)
    return payload if include_id else {key: value for key, value in payload.items() if key != "id"}


def _v2_learning_payload(index: int, *, include_id: bool = False) -> dict[str, Any]:
    payload = _learning_payload(index)
    return payload if include_id else {key: value for key, value in payload.items() if key != "id"}


def _v2_code_payload(index: int, *, include_id: bool = False) -> dict[str, Any]:
    payload = _code_payload(index)
    return payload if include_id else {key: value for key, value in payload.items() if key != "id"}


def _v2_study_payload(index: int, *, include_id: bool = False) -> dict[str, Any]:
    payload = {
        **_registry_base(index, object_type="study_model", family="study_model", suffix=".json"),
        "learning_model_id": 2000 + index,
        "dataset_id": 3000 + index,
        "sampler": ["RandomSampler", "TPESampler", "CmaEsSampler", "GridSampler", "NSGAIISampler"][index],
        "objective": ["minimize_loss", "maximize_accuracy", "maximize_f1", "minimize_latency", "maximize_resilience"][index],
        "best_trial": _study_payload(index)["best_trial"],
        "best_params": _study_payload(index)["best_params"],
        "study_params": _study_payload(index)["study_params"],
    }
    if include_id:
        payload["learning_model"] = {"id": 2000 + index, "name": _learning_payload(index)["name"]}
        payload["dataset"] = {"id": 3000 + index, "name": _dataset_payload(index)["name"]}
        return payload
    return {key: value for key, value in payload.items() if key != "id"}


def _sample_payload_for_class(model_class: type[BaseModel], index: int) -> dict[str, Any]:
    if model_class is model_objects.ObjectModel:
        return _registry_base(index, object_type="object", family="object")
    if model_class is model_objects.DatasetModel:
        return _dataset_payload(index)
    if model_class is model_objects.AssistantTrainingDatasetModel:
        return _assistant_training_dataset_payload(index)
    if model_class is model_objects.LearningModel:
        return _learning_payload(index)
    if model_class is model_objects.AssistantModel:
        return _assistant_model_payload(index)
    if model_class is model_objects.CodeModel:
        return _code_payload(index)
    if model_class is model_objects.InferenceModel:
        return _inference_payload(index)
    if model_class is model_objects.StudyModel:
        return _study_payload(index)

    if model_class is assistant_objects.AssistantSessionRequest:
        return {
            "title": f"{_domain(index)['title']} Session",
            "user_id": f"user-example-{index + 1}",
            "context": {"scale_tier": _tier(index)["tier"], "domain": _domain(index)["slug"]},
        }
    if model_class is assistant_objects.AssistantDraftRequest:
        return {
            "prompt": f"Draft a {REGISTRY_TYPES[index % len(REGISTRY_TYPES)]} for {_domain(index)['story']}.",
            "session_id": f"session_example_{index + 1:02d}",
            "workflow_goal": f"Exercise the {_tier(index)['tier']} tier.",
            "target_type": DRAFT_TYPES[index % len(DRAFT_TYPES)],
            "provider": "auto",
            "assistant_model_id": 1000 + index if index >= 2 else None,
            "reference_ids": [f"ref_example_{index + 1:02d}"],
            "context": {"registryType": REGISTRY_TYPES[index % len(REGISTRY_TYPES)]},
            "constraints": {"declared_size_mb": _tier(index)["declared_size_mb"]},
            "provider_overrides": {"temperature": round(0.2 + index * 0.1, 2)},
        }
    if model_class is assistant_objects.ContentReference:
        return _content_reference_payload(index)
    if model_class is assistant_objects.AssistantReferenceRequest:
        reference = _content_reference_payload(index)
        return {
            "source_type": reference["source_type"],
            "name": reference["name"],
            "summary": reference["summary"],
            "content_text": f"Reference instructions for {_domain(index)['story']}.",
            "file_name": f"{_domain(index)['slug']}_{_tier(index)['tier']}.md",
            "trust_level": reference["trust_level"],
            "uri": reference["uri"],
            "tags": reference["tags"],
            "metadata": reference["metadata"],
        }
    if model_class is assistant_objects.AssistantReferenceSearchRequest:
        return {
            "query": _domain(index)["slug"].replace("_", " "),
            "tags": [_tier(index)["tier"], DRAFT_TYPES[index % len(DRAFT_TYPES)]],
            "source_types": [_content_reference_payload(index)["source_type"]],
            "limit": [5, 10, 20, 30, 50][index],
        }
    if model_class is assistant_objects.AssistantContextPack:
        return _context_pack_payload(index)
    if model_class is assistant_objects.GenerationProvenance:
        return _generation_provenance_payload(index)
    if model_class is assistant_objects.WorkflowDraft:
        return _workflow_draft_payload(index)
    if model_class is assistant_objects.AssistantReviewRequest:
        return {"draft": _workflow_draft_payload(index), "context": {"reviewer": "example_catalog"}}
    if model_class is assistant_objects.AssistantFeedbackRequest:
        return {
            "draft": _workflow_draft_payload(index),
            "run_id": f"run_example_{index + 1:02d}",
            "label": ["positive", "corrected", "accepted", "rejected", "unsafe"][index],
            "reason": "catalog_proof",
            "notes": f"Feedback at {_tier(index)['tier']} scale.",
            "user_edits": {"summary": f"Tightened {_domain(index)['slug']} wording."},
        }
    if model_class is assistant_objects.SafetyViolation:
        return _safety_violation_payload(index)
    if model_class is assistant_objects.SafetyReport:
        return _safety_report_payload(index)
    if model_class is assistant_objects.ReviewDecision:
        return {
            "status": "approved" if index < 3 else "needs_revision",
            "approved": index < 3,
            "safety": _safety_report_payload(index),
            "messages": [f"Review message for {_tier(index)['tier']} scale."],
            "required_actions": [] if index < 3 else ["Reduce unsafe surface before execution."],
            "reviewed_at": _stamp(index),
        }
    if model_class is assistant_objects.AssistantApprovalRequest:
        return {
            "draft": _workflow_draft_payload(index),
            "run_id": f"run_example_{index + 1:02d}",
            "reviewer": "example-catalog",
            "notes": f"Approval proof for {_domain(index)['story']}.",
            "user_edits": {"approved_scale_tier": _tier(index)["tier"]},
        }
    if model_class is assistant_objects.AssistantSubmitRequest:
        return {
            "run_id": f"run_example_{index + 1:02d}",
            "action": ["prepare", "train", "optimize", "deploy", "monitor"][index],
            "payload": {"registry_type": REGISTRY_TYPES[index % len(REGISTRY_TYPES)], "scale_tier": _tier(index)["tier"]},
        }
    if model_class is assistant_objects.ExecutionRunRequest:
        return {
            "code": _code_payload(index)["code"]["script"],
            "profile": DRAFT_TYPES[index % len(DRAFT_TYPES)],
            "draft_id": f"draft_example_{index + 1:02d}",
            "approved": index < 4,
            "context": {"scale_tier": _tier(index)["tier"]},
        }

    if model_class is main_utils.ExecuteRequest:
        return {
            "code": _code_payload(index)["code"]["script"],
            "save": index >= 2,
            "objectName": f"{_domain(index)['slug']}_execution",
        }
    if model_class is main_utils.GenerateRequest:
        return {
            "prompt": f"Generate {_domain(index)['story']} assets at {_tier(index)['tier']} scale.",
            "operationId": ["datasets", "models", "assistant-models", "features", "studies"][index],
        }
    if model_class is main_utils.TrainingRequest:
        return _training_request_payload(index)
    if model_class is main_utils.StudyOptimizationRequest:
        return _study_optimization_payload(index)
    if model_class is main_utils.OnnxPrepareRequest:
        return _onnx_prepare_payload(index)
    if model_class is main_utils.ProductionStartRequest:
        return {"modelId": 2000 + index, "datasetId": 3000 + index, "inferenceId": 5000 + index}
    if model_class is main_utils.ProductionMonitorRequest:
        return {
            "modelId": 2000 + index,
            "datasetId": 3000 + index,
            "autoRetrain": index >= 3,
            "tolerance": round(0.02 + index * 0.01, 3),
            "inferenceId": 5000 + index,
        }

    if model_class is model_pydantic.DatasetModel:
        return _legacy_dataset_payload(index)
    if model_class is model_pydantic.StandardDataset:
        return {**_legacy_dataset_payload(index), "has_features": True}
    if model_class is model_pydantic.FeatureSet:
        return {**_legacy_dataset_payload(index), "features": _domain(index)["features"]}
    if model_class is model_pydantic.FeatureUnit:
        return {
            **_legacy_dataset_payload(index),
            "data_type": ["float", "integer", "category", "boolean", "embedding"][index],
            "origin_dataset": _dataset_payload(index)["name"],
        }
    if model_class is model_pydantic.SampleSet:
        return {
            **_legacy_dataset_payload(index),
            "features": _domain(index)["features"],
            "origin_dataset": _dataset_payload(index)["name"],
            "proportion": [0.05, 0.15, 0.35, 0.65, 1.0][index],
        }
    if model_class is model_pydantic.TemplateSet:
        return {
            **_legacy_dataset_payload(index),
            "features": _domain(index)["features"],
            "source_origin": ["manual", "pandas", "feature_store", "airflow", "assistant"][index],
        }
    if model_class is model_pydantic.LearningModel:
        return _legacy_learning_payload(index)
    if model_class is model_pydantic.ONNXModel:
        return {
            **_legacy_learning_payload(index),
            "onnx_file_path": _path(index, "onnx", ".onnx"),
            "onnx_graph": {"inputs": _domain(index)["features"], "outputs": [_domain(index)["target"]]},
            "source_framework": "pytorch" if index >= 2 else "sklearn",
        }
    if model_class is model_pydantic.TemplateModel:
        return {
            **_legacy_learning_payload(index),
            "template_origin": _path(index, "templates", ".py"),
            "source_framework": "pytorch" if index >= 2 else "sklearn",
        }
    if model_class is model_pydantic.DatasetForm:
        return {
            "name": _dataset_payload(index)["name"],
            "description": _dataset_payload(index)["description"],
            "source_type": _legacy_dataset_payload(index)["source_type"],
            "size": _tier(index)["declared_size_mb"],
        }
    if model_class is model_pydantic.LearningModelForm:
        return {
            "name": _learning_payload(index)["name"],
            "description": _learning_payload(index)["description"],
            "size": _tier(index)["declared_size_mb"],
            "source_code": _code_payload(index)["path"],
            "model_category": _legacy_learning_payload(index)["model_category"],
        }
    if model_class is model_pydantic.StudyForm:
        return {
            "study_name": _study_payload(index)["name"],
            "experiment_name": f"examples/{_domain(index)['slug']}",
            "description": _study_payload(index)["description"],
            "parameters": _study_payload(index)["study_params"],
            "model_used": [_learning_payload(index)["name"]],
            "data_used": [_dataset_payload(index)["name"]],
        }
    if model_class is model_pydantic.ParameterForm:
        return {
            "name": f"{_domain(index)['slug']}_parameters",
            "parameters": _learning_payload(index)["parameters"],
            "description": f"Parameter form for {_tier(index)['tier']} scale.",
            "related_model": _learning_payload(index)["name"],
        }
    if model_class is model_pydantic.ParameterSet:
        return {**_sample_payload_for_class(model_pydantic.ParameterForm, index), "date": _stamp(index)}
    if model_class is model_pydantic.StudySet:
        return {
            **_sample_payload_for_class(model_pydantic.StudyForm, index),
            "history": _history(index, "legacy_study_set"),
            "date": _stamp(index),
            "related_dataset": [_dataset_payload(index)["name"]],
        }
    if model_class is model_pydantic.DashboardConfig:
        return {
            "dashboard_name": f"{_domain(index)['slug']}_dashboard",
            "layout": {"columns": min(index + 1, 4), "density": ["tiny", "compact", "balanced", "dense", "war_room"][index]},
            "widgets": [
                {"type": "metric", "label": _domain(index)["target"]},
                {"type": "line", "label": "drift"},
            ],
            "description": f"Dashboard for {_domain(index)['story']}.",
            "date": _stamp(index),
        }
    if model_class is model_pydantic.ReportModel:
        return {
            "report_name": f"{_domain(index)['slug']}_report",
            "content": f"# {_domain(index)['title']}\n\nScale: {_tier(index)['tier']}.",
            "author": "example-catalog",
            "model_name": _learning_payload(index)["name"],
            "dataset_name": _dataset_payload(index)["name"],
            "related_study": _study_payload(index)["name"],
            "metrics": _learning_payload(index)["metrics"],
            "parameters": _learning_payload(index)["parameters"],
            "description": f"Report example for {_domain(index)['story']}.",
            "date": _stamp(index),
        }
    if model_class is model_pydantic.HistoryModel:
        return {
            "entity_name": _domain(index)["slug"],
            "entity_type": REGISTRY_TYPES[index % len(REGISTRY_TYPES)],
            "changes": _history(index, "history_change"),
            "dags": [{"dag_id": f"example_{_domain(index)['slug']}", "task_count": index + 1}],
            "date": _stamp(index),
            "changed_by": "example-catalog",
        }
    if model_class is model_pydantic.OperationModel:
        return _operation_payload(index)
    if model_class is model_pydantic.TransformationOperation:
        return {
            **_operation_payload(index),
            "parameters": {"fillna": 0, "scale": index >= 2},
            "input_features": _domain(index)["features"],
            "output_features": [f"{feature}_prepared" for feature in _domain(index)["features"]],
        }
    if model_class is model_pydantic.TestingOperation:
        return {
            **_operation_payload(index),
            "parameters": {"sample_rows": _tier(index)["rows"]},
            "test": {"name": "schema_and_drift_check", "threshold": 0.05 + index * 0.01},
        }
    if model_class is model_pydantic.ETLOperation:
        return {
            **_operation_payload(index),
            "parameters": {"mode": "incremental" if index >= 2 else "batch"},
            "input_data_sources": [_dataset_payload(index)["path"]],
            "output_data_targets": [_path(index, "etl", ".parquet")],
            "schedule": ["@once", "@daily", "@hourly", "*/15 * * * *", "*/5 * * * *"][index],
            "dags": [{"dag_id": f"etl_{_domain(index)['slug']}"}],
        }
    if model_class is model_pydantic.DatabaseOperation:
        return {
            **_sample_payload_for_class(model_pydantic.ETLOperation, index),
            "source_origin": "postgres",
        }
    if model_class is model_pydantic.ModelTrainingOperation:
        return {
            **_operation_payload(index),
            "parameters": _learning_payload(index)["parameters"],
            "input_dataset": _dataset_payload(index)["name"],
            "output_model": _learning_payload(index)["name"],
            "training_metrics": _learning_payload(index)["metrics"],
        }
    if model_class is model_pydantic.ModelEvaluationOperation:
        return {
            **_operation_payload(index),
            "parameters": {"split": "holdout", "tier": _tier(index)["tier"]},
            "input_model": _learning_payload(index)["name"],
            "evaluation_dataset": _dataset_payload(index)["name"],
            "evaluation_metrics": _learning_payload(index)["metrics"],
        }
    if model_class is model_pydantic.DeploymentOperation:
        return {
            **_operation_payload(index),
            "parameters": {"replicas": index + 1, "canary": index >= 3},
            "input_model": _learning_payload(index)["name"],
            "deployment_target": ["local", "docker", "mlflow", "k8s", "gpu-worker"][index],
            "status": ["planned", "built", "validated", "deployed", "monitored"][index],
        }

    if model_class is v2_schemas.ObjectCreate:
        return _v2_object_payload(index)
    if model_class is v2_schemas.ObjectUpdate:
        return {"description": f"Updated {_tier(index)['tier']} object example.", "history": _history(index, "v2_update")}
    if model_class is v2_schemas.ObjectRead:
        return _v2_object_payload(index, include_id=True)
    if model_class is v2_schemas.DatasetCreate:
        return _v2_dataset_payload(index)
    if model_class is v2_schemas.DatasetUpdate:
        return {"description": f"Updated {_tier(index)['tier']} dataset.", "shape": _dataset_payload(index)["shape"]}
    if model_class is v2_schemas.DatasetRead:
        return _v2_dataset_payload(index, include_id=True)
    if model_class is v2_schemas.LearningCreate:
        return _v2_learning_payload(index)
    if model_class is v2_schemas.LearningUpdate:
        return {"metrics": _learning_payload(index)["metrics"], "is_tested": index >= 2}
    if model_class is v2_schemas.LearningRead:
        return _v2_learning_payload(index, include_id=True)
    if model_class is v2_schemas.CodeCreate:
        return _v2_code_payload(index)
    if model_class is v2_schemas.CodeUpdate:
        return {"variables": _code_payload(index)["variables"]}
    if model_class is v2_schemas.CodeRead:
        return _v2_code_payload(index, include_id=True)
    if model_class is v2_schemas.StudyCreate:
        return _v2_study_payload(index)
    if model_class is v2_schemas.StudyUpdate:
        return {"study_params": _study_payload(index)["study_params"], "best_params": _study_payload(index)["best_params"]}
    if model_class is v2_schemas.StudyRead:
        return _v2_study_payload(index, include_id=True)

    raise KeyError(f"No example payload builder registered for {model_class.__module__}.{model_class.__name__}")


EXAMPLE_MODEL_GROUPS: list[tuple[str, list[type[BaseModel]]]] = [
    (
        "registry_v1",
        [
            model_objects.ObjectModel,
            model_objects.DatasetModel,
            model_objects.AssistantTrainingDatasetModel,
            model_objects.LearningModel,
            model_objects.AssistantModel,
            model_objects.CodeModel,
            model_objects.InferenceModel,
            model_objects.StudyModel,
        ],
    ),
    (
        "assistant_workflow",
        [
            assistant_objects.AssistantSessionRequest,
            assistant_objects.AssistantDraftRequest,
            assistant_objects.ContentReference,
            assistant_objects.AssistantReferenceRequest,
            assistant_objects.AssistantReferenceSearchRequest,
            assistant_objects.AssistantContextPack,
            assistant_objects.GenerationProvenance,
            assistant_objects.WorkflowDraft,
            assistant_objects.AssistantReviewRequest,
            assistant_objects.AssistantFeedbackRequest,
            assistant_objects.SafetyViolation,
            assistant_objects.SafetyReport,
            assistant_objects.ReviewDecision,
            assistant_objects.AssistantApprovalRequest,
            assistant_objects.AssistantSubmitRequest,
            assistant_objects.ExecutionRunRequest,
        ],
    ),
    (
        "runtime_requests",
        [
            main_utils.ExecuteRequest,
            main_utils.GenerateRequest,
            main_utils.TrainingRequest,
            main_utils.StudyOptimizationRequest,
            main_utils.OnnxPrepareRequest,
            main_utils.ProductionStartRequest,
            main_utils.ProductionMonitorRequest,
        ],
    ),
    (
        "legacy_pydantic",
        [
            model_pydantic.DatasetModel,
            model_pydantic.StandardDataset,
            model_pydantic.FeatureSet,
            model_pydantic.FeatureUnit,
            model_pydantic.SampleSet,
            model_pydantic.TemplateSet,
            model_pydantic.LearningModel,
            model_pydantic.ONNXModel,
            model_pydantic.TemplateModel,
            model_pydantic.DatasetForm,
            model_pydantic.LearningModelForm,
            model_pydantic.StudyForm,
            model_pydantic.ParameterForm,
            model_pydantic.ParameterSet,
            model_pydantic.StudySet,
            model_pydantic.DashboardConfig,
            model_pydantic.ReportModel,
            model_pydantic.HistoryModel,
            model_pydantic.OperationModel,
            model_pydantic.TransformationOperation,
            model_pydantic.TestingOperation,
            model_pydantic.ETLOperation,
            model_pydantic.DatabaseOperation,
            model_pydantic.ModelTrainingOperation,
            model_pydantic.ModelEvaluationOperation,
            model_pydantic.DeploymentOperation,
        ],
    ),
    (
        "registry_v2",
        [
            v2_schemas.ObjectCreate,
            v2_schemas.ObjectUpdate,
            v2_schemas.ObjectRead,
            v2_schemas.DatasetCreate,
            v2_schemas.DatasetUpdate,
            v2_schemas.DatasetRead,
            v2_schemas.LearningCreate,
            v2_schemas.LearningUpdate,
            v2_schemas.LearningRead,
            v2_schemas.CodeCreate,
            v2_schemas.CodeUpdate,
            v2_schemas.CodeRead,
            v2_schemas.StudyCreate,
            v2_schemas.StudyUpdate,
            v2_schemas.StudyRead,
        ],
    ),
]


def iter_example_model_classes() -> Iterable[tuple[str, type[BaseModel]]]:
    for family, classes in EXAMPLE_MODEL_GROUPS:
        for model_class in classes:
            yield family, model_class


def _object_key(model_class: type[BaseModel]) -> str:
    return f"{model_class.__module__}.{model_class.__name__}"


def _validated_payload(model_class: type[BaseModel], payload: dict[str, Any]) -> dict[str, Any]:
    if hasattr(model_class, "model_validate"):
        instance = model_class.model_validate(payload)
        return instance.model_dump(mode="json")
    instance = model_class.parse_obj(payload)
    return instance.dict()


def _route_body_sample(path: str, method: str) -> dict[str, Any] | None:
    method = method.upper()
    if method not in {"POST", "PUT", "PATCH"}:
        return None

    lowered = path.lower()
    if lowered == "/execute" or lowered == "/execute/jobs":
        return _sample_payload_for_class(main_utils.ExecuteRequest, 0)
    if lowered == "/generate":
        return _sample_payload_for_class(main_utils.GenerateRequest, 1)
    if lowered == "/settings/config":
        return {"feature_flags": {"debug_mode": True, "assistant_visible": True}}
    if lowered.startswith("/upload/"):
        return {
            "content_type": "multipart/form-data",
            "operationId": "datasets",
            "objectName": _dataset_payload(0)["name"],
            "file": _dataset_payload(0)["path"],
        }
    if lowered == "/assistant/sessions":
        return _sample_payload_for_class(assistant_objects.AssistantSessionRequest, 0)
    if lowered == "/assistant/draft":
        return _sample_payload_for_class(assistant_objects.AssistantDraftRequest, 1)
    if lowered == "/assistant/review":
        return _sample_payload_for_class(assistant_objects.AssistantReviewRequest, 2)
    if lowered == "/assistant/feedback":
        return _sample_payload_for_class(assistant_objects.AssistantFeedbackRequest, 3)
    if lowered == "/assistant/approve":
        return _sample_payload_for_class(assistant_objects.AssistantApprovalRequest, 2)
    if lowered == "/assistant/submit":
        return _sample_payload_for_class(assistant_objects.AssistantSubmitRequest, 2)
    if lowered == "/assistant/references":
        return _sample_payload_for_class(assistant_objects.AssistantReferenceRequest, 1)
    if lowered == "/assistant/references/search":
        return _sample_payload_for_class(assistant_objects.AssistantReferenceSearchRequest, 1)
    if lowered == "/execution/review" or lowered == "/execution/run":
        return _sample_payload_for_class(assistant_objects.ExecutionRunRequest, 0)
    if lowered == "/training/{model_id}":
        return _sample_payload_for_class(main_utils.TrainingRequest, 2)
    if lowered == "/studies/{study_id}/optimize":
        return _sample_payload_for_class(main_utils.StudyOptimizationRequest, 2)
    if lowered == "/onnx/prepare":
        return _sample_payload_for_class(main_utils.OnnxPrepareRequest, 2)
    if lowered == "/production/start":
        return _sample_payload_for_class(main_utils.ProductionStartRequest, 1)
    if lowered in {"/production/monitor", "/production/retrain"}:
        return _sample_payload_for_class(main_utils.ProductionMonitorRequest, 3)
    if lowered == "/production/simulate":
        return {"steps": 12, "amplitude": 0.05, "trend": "up", "scenario": {"scale_tier": "medium"}}
    if lowered in {"/features/preview", "/features/materialize"}:
        return {
            "datasetId": 3002,
            "limitRows": 25,
            "transforms": [
                {"type": "fill", "column": _domain(2)["features"][0], "value": 0},
                {"type": "sort", "column": _domain(2)["target"], "ascending": False},
            ],
        }
    if lowered.endswith("/create"):
        prefix = lowered.strip("/").split("/")[0]
        create_samples = {
            "objectmodel": _registry_base(0, object_type="object", family="object"),
            "datasetmodel": _dataset_payload(0),
            "assistanttrainingdatasetmodel": _assistant_training_dataset_payload(0),
            "learningmodel": _learning_payload(0),
            "assistantmodel": _assistant_model_payload(0),
            "inferencemodel": _inference_payload(0),
            "codemodel": _code_payload(0),
            "studymodel": _study_payload(0),
        }
        return create_samples.get(prefix)
    if lowered.endswith("/update/{item_id}"):
        return {"description": "Updated by the example catalog proof matrix."}
    return {"example": True, "scale_tier": "simple"}


def build_api_route_proof(routes: Iterable[Any] | None = None) -> dict[str, Any]:
    route_rows: list[dict[str, Any]] = []
    for route in routes or []:
        if not isinstance(route, APIRoute):
            continue
        methods = sorted(method for method in (route.methods or set()) if method not in {"HEAD", "OPTIONS"})
        for method in methods:
            body_sample = _route_body_sample(route.path, method)
            route_rows.append(
                {
                    "method": method,
                    "path": route.path,
                    "name": route.name,
                    "frontend_route": route.path in FRONTEND_ROUTES,
                    "body_sample": body_sample,
                    "proof_mode": "browser_render" if route.path in FRONTEND_ROUTES else "api_payload_contract",
                }
            )

    workflow_chains = [
        {
            "name": "dataset_to_training_to_production",
            "routes": ["/upload/{operation_id}", "/analysis/data", "/features/preview", "/training/{model_id}", "/production/start", "/production/monitor", "/production/simulate"],
        },
        {
            "name": "assistant_model_to_reviewed_workflow",
            "routes": ["/assistant/models/list", "/assistant/references/search", "/assistant/draft", "/assistant/review", "/assistant/approve", "/assistant/submit", "/assistant/feedback"],
        },
        {
            "name": "study_to_registry_to_mlflow",
            "routes": ["/studies/{study_id}/optimize", "/runs/list", "/plots/artifacts", "/mlflow/experiments", "/registry/download/{registry_type}/{item_id}"],
        },
        {
            "name": "guarded_execution",
            "routes": ["/execution/review", "/execution/run", "/assistant/training/curate", "/assistant/evals/run"],
        },
    ]
    return {
        "route_count": len(route_rows),
        "frontend_route_count": sum(1 for row in route_rows if row["frontend_route"]),
        "routes": route_rows,
        "frontend_routes": FRONTEND_ROUTES,
        "workflow_chains": workflow_chains,
        "note": "Mutation examples are payload contracts; they are intentionally not auto-submitted by the catalog endpoint.",
    }


def build_example_catalog(
    *,
    routes: Iterable[Any] | None = None,
    include_payloads: bool = True,
) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    validation_errors: list[dict[str, str]] = []

    for family, model_class in iter_example_model_classes():
        samples: list[dict[str, Any]] = []
        for index, tier in enumerate(SCALE_TIERS):
            raw_payload = _sample_payload_for_class(model_class, index)
            try:
                payload = _validated_payload(model_class, raw_payload)
            except Exception as exc:
                validation_errors.append(
                    {
                        "object_key": _object_key(model_class),
                        "tier": tier["tier"],
                        "error": str(exc),
                    }
                )
                payload = raw_payload

            sample = {
                "tier": tier["tier"],
                "declared_size_mb": tier["declared_size_mb"],
                "rows": tier["rows"],
                "columns": tier["columns"],
                "complexity": tier["complexity"],
                "storage_note": "Declared workload scale; fixtures avoid committing large binary blobs.",
            }
            if include_payloads:
                sample["payload"] = payload
            else:
                sample["payload_fields"] = sorted(payload.keys())
            samples.append(sample)

        objects.append(
            {
                "object_key": _object_key(model_class),
                "object_name": model_class.__name__,
                "family": family,
                "module": model_class.__module__,
                "sample_count": len(samples),
                "samples": samples,
            }
        )

    sample_count = sum(item["sample_count"] for item in objects)
    return {
        "status": "ok" if not validation_errors else "degraded",
        "version": "painel-amanaje-example-catalog-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "object_count": len(objects),
            "samples_per_object": len(SCALE_TIERS),
            "sample_count": sample_count,
            "scale_tiers": SCALE_TIERS,
            "families": {
                family: sum(1 for item in objects if item["family"] == family)
                for family, _classes in EXAMPLE_MODEL_GROUPS
            },
            "validation_error_count": len(validation_errors),
        },
        "objects": objects,
        "api_proof": build_api_route_proof(routes),
        "validation_errors": validation_errors,
    }


def find_example_object(
    object_name: str,
    *,
    routes: Iterable[Any] | None = None,
    include_payloads: bool = True,
) -> dict[str, Any] | None:
    normalized = object_name.strip().lower()
    if not normalized:
        return None

    catalog = build_example_catalog(routes=routes, include_payloads=include_payloads)
    for entry in catalog["objects"]:
        aliases = {
            entry["object_key"].lower(),
            entry["object_name"].lower(),
            f"{entry['family']}.{entry['object_name']}".lower(),
            f"{entry['module']}.{entry['object_name']}".lower(),
        }
        if normalized in aliases:
            return entry
    return None
