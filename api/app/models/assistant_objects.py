from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


ASSISTANT_DRAFT_TYPES = {
    "dataset_generation",
    "feature_operations",
    "model_generation",
    "registry_object",
    "study",
    "training_run",
}


def _default_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class AssistantSessionRequest(BaseModel):
    title: str = "Assistant Session"
    user_id: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)


class AssistantDraftRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    workflow_goal: Optional[str] = None
    target_type: Optional[str] = None
    provider: str = "auto"
    assistant_model_id: Optional[int | str] = None
    reference_ids: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    provider_overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt")
    @classmethod
    def _prompt_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt is required")
        return normalized

    @field_validator("target_type")
    @classmethod
    def _known_target_type(cls, value: Optional[str]) -> Optional[str]:
        if value in (None, ""):
            return None
        normalized = value.strip().lower()
        if normalized not in ASSISTANT_DRAFT_TYPES:
            allowed = ", ".join(sorted(ASSISTANT_DRAFT_TYPES))
            raise ValueError(f"target_type must be one of: {allowed}")
        return normalized


class ContentReference(BaseModel):
    reference_id: str = Field(default_factory=lambda: _default_id("ref"))
    source_type: str
    name: Optional[str] = None
    summary: str = ""
    trust_level: str = "project"
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class AssistantReferenceRequest(BaseModel):
    source_type: str = "user_text"
    name: Optional[str] = None
    summary: Optional[str] = None
    content_text: Optional[str] = None
    file_name: Optional[str] = None
    trust_level: str = "user"
    uri: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssistantReferenceSearchRequest(BaseModel):
    query: str = ""
    tags: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    limit: int = 20


class AssistantContextPack(BaseModel):
    context_pack_id: str = Field(default_factory=lambda: _default_id("ctx"))
    version: str = "assistant-context-pack-v1"
    prompt: str
    workflow_goal: Optional[str] = None
    target_type: str
    provider: str = "local"
    registry_type: Optional[str] = None
    form_values: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    safety_profile: str = "dataset_generation"
    allowed_imports: list[str] = Field(default_factory=list)
    allowed_dependencies: list[str] = Field(default_factory=list)
    content_references: list[ContentReference] = Field(default_factory=list)
    reference_pack_id: str = Field(default_factory=lambda: _default_id("refpack"))
    reference_pack_hash: str = ""
    selected_reference_ids: list[str] = Field(default_factory=list)
    reference_trust_levels: dict[str, str] = Field(default_factory=dict)
    compact_summary: str = ""
    pack_hash: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class GenerationProvenance(BaseModel):
    source_type: str = "assistant"
    provider: str
    model_version: Optional[str] = None
    prompt_template_version: Optional[str] = None
    context_pack_id: Optional[str] = None
    context_pack_hash: Optional[str] = None
    fallback_provider: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class WorkflowDraft(BaseModel):
    draft_id: str = Field(default_factory=lambda: _default_id("draft"))
    session_id: Optional[str] = None
    draft_type: str
    title: str
    summary: str
    prompt: str
    status: str = "draft"
    provider: str = "local"
    created_at: datetime = Field(default_factory=datetime.now)
    code: Optional[str] = None
    operations: list[dict[str, Any]] = Field(default_factory=list)
    registry_payload: dict[str, Any] = Field(default_factory=dict)
    registry_type: Optional[str] = None
    form_payload: dict[str, Any] = Field(default_factory=dict)
    training_request: dict[str, Any] = Field(default_factory=dict)
    study_request: dict[str, Any] = Field(default_factory=dict)
    execution_profile: str = "dataset_generation"
    materialized_outputs: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    next_actions: list[str] = Field(default_factory=list)
    model_version: Optional[str] = None
    prompt_template_version: Optional[str] = None
    context_pack_id: Optional[str] = None
    context_pack_hash: Optional[str] = None
    evaluation_profile: Optional[str] = None
    generation_provenance: dict[str, Any] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("draft_type")
    @classmethod
    def _known_draft_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ASSISTANT_DRAFT_TYPES:
            allowed = ", ".join(sorted(ASSISTANT_DRAFT_TYPES))
            raise ValueError(f"draft_type must be one of: {allowed}")
        return normalized


class AssistantReviewRequest(BaseModel):
    draft: WorkflowDraft
    context: dict[str, Any] = Field(default_factory=dict)


class AssistantFeedbackRequest(BaseModel):
    draft: WorkflowDraft
    run_id: Optional[str] = None
    label: str = "negative"
    reason: Optional[str] = None
    notes: Optional[str] = None
    user_edits: dict[str, Any] = Field(default_factory=dict)


class SafetyViolation(BaseModel):
    code: str
    severity: str
    message: str
    line: Optional[int] = None
    name: Optional[str] = None


class SafetyReport(BaseModel):
    ok: bool
    risk_level: str
    profile: str = "dataset_generation"
    violations: list[SafetyViolation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    allowed_imports: list[str] = Field(default_factory=list)
    allowed_dependencies: list[str] = Field(default_factory=list)
    blocked_names: list[str] = Field(default_factory=list)
    materialized_output_checks: dict[str, bool] = Field(default_factory=dict)


class ReviewDecision(BaseModel):
    status: str
    approved: bool = False
    safety: SafetyReport
    messages: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    reviewed_at: datetime = Field(default_factory=datetime.now)


class AssistantApprovalRequest(BaseModel):
    draft: WorkflowDraft
    run_id: Optional[str] = None
    reviewer: str = "local-user"
    notes: Optional[str] = None
    user_edits: dict[str, Any] = Field(default_factory=dict)


class AssistantSubmitRequest(BaseModel):
    run_id: str
    action: str = "prepare"
    payload: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


class ExecutionRunRequest(BaseModel):
    code: str
    profile: str = "dataset_generation"
    draft_id: Optional[str] = None
    approved: bool = False
    context: dict[str, Any] = Field(default_factory=dict)
