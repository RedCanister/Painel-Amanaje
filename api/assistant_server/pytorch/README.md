# PyTorch/Hugging Face Assistant Server Example

This is a reference server for uploaded `AssistantModel` bundles. It is intentionally separate from the main FastAPI app: Painel Amanaje stores, validates, reviews, and routes drafts, while this service only generates strict `WorkflowDraft` JSON through an OpenAI-compatible API.

## Bundle Contract

The server expects a directory extracted from an AssistantModel `.zip` bundle. The zip must include:

- `assistant_model_manifest.json`
- PyTorch/Hugging Face weights such as `model.safetensors`, `pytorch_model.bin`, `adapter_model.safetensors`, or `model.pt`
- Tokenizer assets such as `tokenizer.json` or tokenizer config plus vocab/merges files
- A `chat_template`, `chat_template_path`, or `prompt_template_version`

Painel Amanaje checks this contract during `POST /upload/assistant-models`.

## Run

```powershell
cd api/assistant_server/pytorch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ASSISTANT_MODEL_BUNDLE_DIR="..\..\runtime_artifacts\assistant_models\bundles\your_bundle"
uvicorn server:app --host 0.0.0.0 --port 8080
```

Docker Compose starts this service as `assistant-server` and exposes it at `http://localhost:8091` for diagnostics. Other containers call it at `http://assistant-server:8080`.

Then configure Painel Amanaje or the uploaded AssistantModel with:

```json
{
  "provider_type": "openai_compatible",
  "runtime_kind": "pytorch_hf_server",
  "base_url": "http://localhost:8080/v1",
  "model_name": "your-assistant-model"
}
```

## Endpoints

- `GET /health`
- `POST /v1/chat/completions`
- `POST /admin/models/load`
- `POST /admin/models/unload`
- `GET /admin/models/status`

The chat response body follows the OpenAI chat-completions shape. `choices[0].message.content` must be a JSON object that validates as Painel Amanaje `WorkflowDraft`.

Admin endpoints use `X-Amanaje-Assistant-Token` or `Authorization: Bearer ...` when `ASSISTANT_SERVER_ADMIN_TOKEN` is configured.
