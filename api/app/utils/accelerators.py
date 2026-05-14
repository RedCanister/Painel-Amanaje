from __future__ import annotations

import os
from typing import Any


def _load_torch() -> Any | None:
    try:
        import torch

        return torch
    except Exception:
        return None


def get_torch_accelerator_status() -> dict[str, Any]:
    torch = _load_torch()
    status: dict[str, Any] = {
        "torch_available": torch is not None,
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "fallback_policy": os.getenv("AMANAJE_GPU_FALLBACK_POLICY", "cpu"),
    }
    if torch is None:
        return status

    cuda = getattr(torch, "cuda", None)
    status["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        cuda_available = bool(cuda is not None and cuda.is_available())
    except Exception:
        cuda_available = False
    status["cuda_available"] = cuda_available

    try:
        device_count = int(cuda.device_count()) if cuda is not None else 0
    except Exception:
        device_count = 0
    status["device_count"] = device_count

    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        device: dict[str, Any] = {"index": index, "name": None, "total_memory_bytes": None}
        try:
            device["name"] = str(cuda.get_device_name(index))
        except Exception:
            device["name"] = f"cuda:{index}"
        try:
            props = cuda.get_device_properties(index)
            device["total_memory_bytes"] = int(getattr(props, "total_memory", 0) or 0)
        except Exception:
            device["total_memory_bytes"] = None
        devices.append(device)
    status["devices"] = devices
    return status


def resolve_torch_device(requested_device: Any = "auto") -> dict[str, Any]:
    normalized = str(requested_device or "auto").strip().lower()
    if normalized not in {"auto", "cuda", "cpu"}:
        normalized = "auto"

    status = get_torch_accelerator_status()
    cuda_available = bool(status.get("cuda_available"))
    resolved = "cpu"
    fallback_applied = False
    fallback_reason = None

    if normalized == "cpu":
        resolved = "cpu"
    elif normalized == "cuda":
        if cuda_available:
            resolved = "cuda"
        else:
            resolved = "cpu"
            fallback_applied = True
            fallback_reason = "cuda_unavailable"
    elif normalized == "auto":
        resolved = "cuda" if cuda_available else "cpu"
        fallback_reason = None if cuda_available else "auto_selected_cpu"

    return {
        **status,
        "requested_device": normalized,
        "resolved_device": resolved,
        "fallback_applied": fallback_applied,
        "fallback_reason": fallback_reason,
    }
