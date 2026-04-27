from __future__ import annotations

import ast
from typing import Any, Iterable, Mapping

from app.models.assistant_objects import ReviewDecision, SafetyReport, SafetyViolation, WorkflowDraft


ALLOWED_IMPORT_ROOTS = {
    "datetime",
    "math",
    "numpy",
    "pandas",
    "random",
    "sklearn",
    "statistics",
}

SAFETY_PROFILES = {
    "dataset_generation": {
        "allowed_imports": {"datetime", "math", "numpy", "pandas", "random", "statistics"},
        "allowed_dependencies": {"numpy", "pandas"},
        "required_outputs": {"csv_text"},
    },
    "model_generation": {
        "allowed_imports": {"base64", "io", "joblib", "numpy", "pandas", "sklearn"},
        "allowed_dependencies": {"joblib", "scikit-learn", "numpy", "pandas"},
        "required_outputs": {"joblib_bytes"},
    },
    "registry_object": {
        "allowed_imports": set(),
        "allowed_dependencies": set(),
        "required_outputs": set(),
    },
    "feature_operations": {
        "allowed_imports": set(),
        "allowed_dependencies": set(),
        "required_outputs": set(),
    },
    "training_run": {
        "allowed_imports": set(),
        "allowed_dependencies": set(),
        "required_outputs": set(),
    },
    "study": {
        "allowed_imports": set(),
        "allowed_dependencies": set(),
        "required_outputs": set(),
    },
}

BLOCKED_IMPORT_ROOTS = {
    "aiofiles",
    "asyncio",
    "base64",
    "builtins",
    "cloudpickle",
    "ctypes",
    "dill",
    "httpx",
    "importlib",
    "io",
    "joblib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "urllib",
}

BLOCKED_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

BLOCKED_ATTRIBUTE_NAMES = {
    "__bases__",
    "__class__",
    "__dict__",
    "__globals__",
    "__mro__",
    "__subclasses__",
    "chmod",
    "connect",
    "dump",
    "dumps",
    "loads",
    "mkdir",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "send",
    "system",
    "unlink",
    "write",
}

FILE_WRITER_METHODS = {
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_sql",
}


def _root_module(module_name: str | None) -> str:
    return str(module_name or "").split(".", 1)[0]


def _line(node: ast.AST) -> int | None:
    return getattr(node, "lineno", None)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _add_violation(
    violations: list[SafetyViolation],
    *,
    code: str,
    severity: str,
    message: str,
    node: ast.AST,
    name: str | None = None,
) -> None:
    violations.append(
        SafetyViolation(
            code=code,
            severity=severity,
            message=message,
            line=_line(node),
            name=name,
        )
    )


def _check_imports(tree: ast.AST, violations: list[SafetyViolation], allowed_roots: set[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _root_module(alias.name)
                if (root in BLOCKED_IMPORT_ROOTS and root not in allowed_roots) or root not in allowed_roots:
                    _add_violation(
                        violations,
                        code="blocked-import",
                        severity="high",
                        message=f"Import '{alias.name}' is not allowed in assistant-reviewed code.",
                        node=node,
                        name=alias.name,
                    )
        elif isinstance(node, ast.ImportFrom):
            root = _root_module(node.module)
            if (root in BLOCKED_IMPORT_ROOTS and root not in allowed_roots) or root not in allowed_roots:
                _add_violation(
                    violations,
                    code="blocked-import",
                    severity="high",
                    message=f"Import from '{node.module or ''}' is not allowed in assistant-reviewed code.",
                    node=node,
                    name=node.module,
                )


def _check_calls(tree: ast.AST, violations: list[SafetyViolation], profile: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in BLOCKED_CALL_NAMES:
                _add_violation(
                    violations,
                    code="blocked-call",
                    severity="critical",
                    message=f"Call to '{name}' is blocked in assistant-reviewed code.",
                    node=node,
                    name=name,
                )

            if isinstance(node.func, ast.Attribute):
                attribute_name = node.func.attr
                if attribute_name in BLOCKED_ATTRIBUTE_NAMES and not _is_allowed_profile_call(node.func, profile):
                    _add_violation(
                        violations,
                        code="blocked-attribute-call",
                        severity="high",
                        message=f"Attribute call '{attribute_name}' is blocked in assistant-reviewed code.",
                        node=node,
                        name=attribute_name,
                    )
                if attribute_name in FILE_WRITER_METHODS:
                    _add_violation(
                        violations,
                        code="file-write",
                        severity="high",
                        message=f"Method '{attribute_name}' writes external artifacts and requires a dedicated runtime path.",
                        node=node,
                        name=attribute_name,
                    )
                if attribute_name == "to_csv" and _to_csv_writes_to_path(node):
                    _add_violation(
                        violations,
                        code="file-write",
                        severity="high",
                        message="DataFrame.to_csv may only be used without a path, for example csv_text = df.to_csv(index=False).",
                        node=node,
                        name=attribute_name,
                    )


def _is_allowed_profile_call(function_node: ast.Attribute, profile: str) -> bool:
    if profile == "model_generation":
        if isinstance(function_node.value, ast.Name) and function_node.value.id == "joblib":
            return function_node.attr == "dump"
    return False


def _check_names(tree: ast.AST, violations: list[SafetyViolation]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            _add_violation(
                violations,
                code="dunder-name",
                severity="high",
                message=f"Dunder name '{node.id}' is blocked in assistant-reviewed code.",
                node=node,
                name=node.id,
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            _add_violation(
                violations,
                code="dunder-attribute",
                severity="high",
                message=f"Dunder attribute '{node.attr}' is blocked in assistant-reviewed code.",
                node=node,
                name=node.attr,
            )


def _to_csv_writes_to_path(node: ast.Call) -> bool:
    if node.args:
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or first_arg.value is not None:
            return True
    for keyword in node.keywords:
        if keyword.arg == "path_or_buf":
            if not isinstance(keyword.value, ast.Constant) or keyword.value.value is not None:
                return True
    return False


def _assigned_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)

        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                names.update(item.id for item in target.elts if isinstance(item, ast.Name))
    return names


def _profile_config(profile: str) -> dict[str, Any]:
    return SAFETY_PROFILES.get(profile, SAFETY_PROFILES["dataset_generation"])


def review_code_safety(
    code: str,
    *,
    allowed_imports: Iterable[str] | None = None,
    profile: str = "dataset_generation",
) -> SafetyReport:
    config = _profile_config(profile)
    allowed_roots = set(allowed_imports or config["allowed_imports"] or ALLOWED_IMPORT_ROOTS)
    allowed = sorted(allowed_roots)
    violations: list[SafetyViolation] = []
    warnings: list[str] = []
    output_checks: dict[str, bool] = {}

    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        violations.append(
            SafetyViolation(
                code="syntax-error",
                severity="critical",
                message=str(exc),
                line=exc.lineno,
            )
        )
        return SafetyReport(
            ok=False,
            risk_level="critical",
            violations=violations,
            warnings=warnings,
            profile=profile,
            allowed_imports=allowed,
            allowed_dependencies=sorted(config["allowed_dependencies"]),
            blocked_names=sorted(BLOCKED_CALL_NAMES),
            materialized_output_checks=output_checks,
        )

    _check_imports(tree, violations, allowed_roots)
    _check_calls(tree, violations, profile)
    _check_names(tree, violations)

    assigned_names = _assigned_names(tree)
    for required_name in sorted(config["required_outputs"]):
        output_checks[required_name] = required_name in assigned_names
        if not output_checks[required_name]:
            violations.append(
                SafetyViolation(
                    code="missing-output",
                    severity="high",
                    message=f"Required materialized output '{required_name}' is missing for {profile}.",
                    name=required_name,
                )
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.While, ast.AsyncFunctionDef)):
            warnings.append("Loops or async functions may need runtime limits before execution.")

    highest = _highest_risk(violation.severity for violation in violations)
    return SafetyReport(
        ok=not violations,
        risk_level=highest if violations else ("low" if code.strip() else "none"),
        profile=profile,
        violations=violations,
        warnings=warnings,
        allowed_imports=allowed,
        allowed_dependencies=sorted(config["allowed_dependencies"]),
        blocked_names=sorted(BLOCKED_CALL_NAMES),
        materialized_output_checks=output_checks,
    )


def _highest_risk(severities: Iterable[str]) -> str:
    rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    selected = "none"
    selected_rank = 0
    for severity in severities:
        severity_rank = rank.get(severity, 0)
        if severity_rank > selected_rank:
            selected = severity
            selected_rank = severity_rank
    return selected


def _positive_int(value: Any) -> bool:
    try:
        return int(value or 0) > 0
    except (TypeError, ValueError):
        return False


def review_workflow_draft(draft: WorkflowDraft | Mapping[str, Any]) -> ReviewDecision:
    workflow_draft = draft if isinstance(draft, WorkflowDraft) else WorkflowDraft.model_validate(draft)
    safety = review_code_safety(
        workflow_draft.code or "",
        profile=workflow_draft.execution_profile or workflow_draft.draft_type,
    )
    messages: list[str] = []
    required_actions: list[str] = []

    if not safety.ok:
        messages.append("The draft needs changes before approval.")
        required_actions.extend(violation.message for violation in safety.violations)

    if workflow_draft.draft_type == "feature_operations" and not workflow_draft.operations:
        messages.append("Feature operation drafts should include at least one transform before submission.")
        required_actions.append("Add feature operations or change the draft type.")

    if workflow_draft.draft_type == "training_run":
        request = workflow_draft.training_request
        if not request.get("modelId") or not request.get("datasetId"):
            messages.append("Training drafts need both modelId and datasetId before submission.")
            required_actions.append("Choose a learning model and dataset.")

    if workflow_draft.draft_type == "registry_object":
        payload = dict(workflow_draft.registry_payload or workflow_draft.form_payload or {})
        if not payload.get("name") or not payload.get("object_type") or not payload.get("path"):
            messages.append("Registry drafts need name, object_type, and path before submission.")
            required_actions.append("Complete required registry fields.")
        object_type = str(payload.get("object_type") or "")
        if object_type == "dataset" and not payload.get("shape"):
            required_actions.append("Dataset registry drafts require shape.")
        if object_type == "learning_model" and not isinstance(payload.get("parameters"), Mapping):
            required_actions.append("LearningModel registry drafts require parameters.")
        if object_type == "inference_model":
            if not _positive_int(payload.get("learning_model_id")) or not _positive_int(payload.get("dataset_id")):
                required_actions.append("InferenceModel registry drafts require valid learning_model_id and dataset_id.")
            if not isinstance(payload.get("inference_params"), Mapping):
                required_actions.append("InferenceModel registry drafts require inference_params.")
        if object_type == "study_model":
            if not _positive_int(payload.get("learning_model_id")) or not _positive_int(payload.get("dataset_id")):
                required_actions.append("StudyModel registry drafts require valid learning_model_id and dataset_id.")
            if not payload.get("sampler") or not payload.get("objective"):
                required_actions.append("StudyModel registry drafts require sampler and objective.")
        if object_type == "code_model":
            if not isinstance(payload.get("variables"), Mapping) or not isinstance(payload.get("code"), Mapping):
                required_actions.append("CodeModel registry drafts require variables and code objects.")
        if required_actions and not any("Registry draft" in message for message in messages):
            messages.append("Registry draft needs required fields before manual creation.")

    approved = safety.ok and not required_actions
    status = "approved" if approved else "needs_revision"
    if workflow_draft.draft_type == "dataset_generation" and safety.ok:
        messages.append("Dataset code is review-safe and ready for explicit execution approval.")
    if workflow_draft.draft_type == "model_generation" and safety.ok:
        messages.append("Model code is review-safe and ready for guarded execution.")

    return ReviewDecision(
        status=status,
        approved=approved,
        safety=safety,
        messages=messages,
        required_actions=required_actions,
    )
