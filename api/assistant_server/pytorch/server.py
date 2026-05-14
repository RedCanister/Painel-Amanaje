from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Painel Amanaje PyTorch Assistant Server")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None


class ModelLoadRequest(BaseModel):
    model_name: str
    model_version: str | None = None
    bundle_dir: str | None = None
    extracted_dir: str | None = None
    model_artifact_path: str | None = None
    tokenizer_path: str | None = None
    chat_template_path: str | None = None
    chat_template: str | None = None
    generation_config: dict[str, Any] = Field(default_factory=dict)
    device: str = "auto"
    dtype: str = "auto"
    max_context_tokens: int | None = None


STATE: dict[str, Any] = {
    "loaded": False,
    "loading": False,
    "error": None,
    "manifest": {},
    "tokenizer": None,
    "model": None,
    "active_model": None,
    "loaded_at": None,
}


def _admin_token() -> str:
    return os.getenv("ASSISTANT_SERVER_ADMIN_TOKEN", "").strip()


def _require_admin_token(
    x_amanaje_assistant_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    expected = _admin_token()
    if not expected:
        return
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    supplied = (x_amanaje_assistant_token or bearer or "").strip()
    if supplied != expected:
        raise HTTPException(status_code=401, detail="Invalid assistant server admin token.")


def _default_bundle_dir() -> Path:
    return Path(os.getenv("ASSISTANT_MODEL_BUNDLE_DIR", ".")).resolve()


def _resolve_model_dir(payload: ModelLoadRequest | None = None) -> Path:
    if payload is not None:
        for value in (payload.model_artifact_path, payload.tokenizer_path, payload.chat_template_path):
            if value:
                candidate = Path(value).resolve()
                if candidate.exists():
                    return candidate.parent if candidate.is_file() else candidate
        for value in (payload.extracted_dir, payload.bundle_dir):
            if value:
                return Path(value).resolve()
    return _default_bundle_dir()


def _load_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / "assistant_model_manifest.json"
    if not path.exists():
        raise RuntimeError(f"assistant_model_manifest.json was not found in {bundle_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("assistant_model_manifest.json must be a JSON object.")
    return payload


def _release_loaded_model() -> None:
    STATE.update(
        {
            "loaded": False,
            "loading": False,
            "error": None,
            "manifest": {},
            "tokenizer": None,
            "model": None,
            "active_model": None,
            "loaded_at": None,
        }
    )
    try:
        import gc

        gc.collect()
    except Exception:
        pass


def _load_runtime(payload: ModelLoadRequest | None = None, *, force: bool = False) -> dict[str, Any]:
    requested_name = payload.model_name if payload is not None else os.getenv("ASSISTANT_MODEL_NAME", "assistant-model")
    if STATE["loaded"] and not force and (STATE.get("active_model") or {}).get("model_name") == requested_name:
        return _status_payload()

    try:
        STATE["loading"] = True
        STATE["error"] = None
        from transformers import AutoModelForCausalLM, AutoTokenizer

        bundle_dir = _resolve_model_dir(payload)
        manifest = _load_manifest(bundle_dir)
        model_name = requested_name or manifest.get("model_name") or bundle_dir.name
        device = (payload.device if payload is not None else os.getenv("ASSISTANT_DEVICE", manifest.get("device", "auto"))) or "auto"
        device_map = "auto" if str(device).lower() == "auto" else None
        tokenizer = AutoTokenizer.from_pretrained(bundle_dir, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            bundle_dir,
            device_map=device_map,
            trust_remote_code=False,
        )
        if payload and payload.chat_template and tokenizer is not None:
            tokenizer.chat_template = payload.chat_template
        elif payload and payload.chat_template_path:
            template_path = Path(payload.chat_template_path)
            if template_path.exists():
                tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        generation_config = {
            **dict(manifest.get("generation_config") or {}),
            **dict(payload.generation_config if payload else {}),
        }
        STATE.update(
            {
                "loaded": True,
                "loading": False,
                "error": None,
                "manifest": manifest,
                "tokenizer": tokenizer,
                "model": model,
                "active_model": {
                    "model_name": model_name,
                    "model_version": (payload.model_version if payload else None) or manifest.get("model_version"),
                    "bundle_dir": str(bundle_dir),
                    "device": device,
                    "dtype": payload.dtype if payload else os.getenv("ASSISTANT_DTYPE", manifest.get("dtype", "auto")),
                    "max_context_tokens": payload.max_context_tokens if payload else manifest.get("max_context_tokens"),
                    "generation_config": generation_config,
                },
                "loaded_at": datetime.now().isoformat(),
            }
        )
    except Exception as exc:
        STATE.update({"loaded": False, "loading": False, "error": str(exc), "tokenizer": None, "model": None})
        raise
    return _status_payload()


def _ensure_loaded() -> None:
    if STATE["loaded"]:
        return
    try:
        _load_runtime(
            ModelLoadRequest(
                model_name=os.getenv("ASSISTANT_MODEL_NAME", "assistant-model"),
                bundle_dir=str(_default_bundle_dir()),
            )
        )
    except Exception:
        return


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        return match.group(0)
    raise RuntimeError("Model output did not contain a JSON object.")


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    tokenizer = STATE.get("tokenizer")
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [message.model_dump() for message in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return "\n".join(f"{message.role}: {message.content}" for message in messages) + "\nassistant:"


def _status_payload() -> dict[str, Any]:
    active_model = STATE.get("active_model") or {}
    manifest = STATE.get("manifest") or {}
    return {
        "status": "ready" if STATE["loaded"] else ("loading" if STATE["loading"] else "unavailable"),
        "loaded": bool(STATE["loaded"]),
        "loading": bool(STATE["loading"]),
        "error": STATE["error"],
        "active_model": active_model,
        "model_name": active_model.get("model_name") or manifest.get("model_name") or os.getenv("ASSISTANT_MODEL_NAME"),
        "model_version": active_model.get("model_version") or manifest.get("model_version"),
        "bundle_dir": active_model.get("bundle_dir") or str(_default_bundle_dir()),
        "loaded_at": STATE.get("loaded_at"),
        "trust_remote_code": False,
        "one_active_model": True,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return _status_payload()


@app.get("/admin/models/status")
def admin_model_status(_: None = Depends(_require_admin_token)) -> dict[str, Any]:
    return _status_payload()


@app.post("/admin/models/load")
def admin_model_load(payload: ModelLoadRequest, _: None = Depends(_require_admin_token)) -> dict[str, Any]:
    try:
        return _load_runtime(payload, force=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/admin/models/unload")
def admin_model_unload(_: None = Depends(_require_admin_token)) -> dict[str, Any]:
    _release_loaded_model()
    return _status_payload()


@app.post("/v1/chat/completions")
def chat_completions(payload: ChatCompletionRequest) -> dict[str, Any]:
    _ensure_loaded()
    if not STATE["loaded"]:
        raise HTTPException(status_code=503, detail=STATE["error"] or "Assistant model is not loaded.")

    tokenizer = STATE["tokenizer"]
    model = STATE["model"]
    active_model = STATE.get("active_model") or {}
    generation_config = dict(active_model.get("generation_config") or {})
    prompt = _messages_to_prompt(payload.messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    max_new_tokens = int(
        payload.max_tokens
        or os.getenv("ASSISTANT_MAX_NEW_TOKENS", "")
        or generation_config.get("max_new_tokens")
        or 1600
    )
    temperature = float(
        payload.temperature
        if payload.temperature is not None
        else os.getenv("ASSISTANT_TEMPERATURE", "")
        or generation_config.get("temperature")
        or 0.2
    )
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 0.01),
        pad_token_id=tokenizer.eos_token_id,
    )
    output_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    content = tokenizer.decode(output_tokens, skip_special_tokens=True)
    content = _extract_json_object(content)
    json.loads(content)

    return {
        "id": "amanaje-pytorch-assistant",
        "object": "chat.completion",
        "model": payload.model or active_model.get("model_name") or "assistant-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
