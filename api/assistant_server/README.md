# Painel Amanaje Assistant SLM Server Contract

Painel Amanaje treats assistant models as separate draft providers. A provider can run locally or on an external server as long as it speaks the OpenAI-compatible chat-completions protocol.

## Required Runtime Shape

- `POST /v1/chat/completions`
- Optional but recommended: `GET /health`
- Response content must be a single JSON object that validates as `WorkflowDraft`.
- The model must draft only. Painel Amanaje validates, reviews, executes, and persists.

## Provider Configuration

Set the active provider:

```powershell
$env:AMANAJE_ASSISTANT_ACTIVE_PROVIDER="amanaje_slm"
```

Configure one or more providers with JSON:

```powershell
$env:AMANAJE_ASSISTANT_PROVIDER_CONFIGS = Get-Content .\api\assistant_server\provider-config.example.json -Raw
```

Legacy variables still work for `amanaje_slm`:

```powershell
$env:AMANAJE_SLM_ENABLED="true"
$env:AMANAJE_SLM_BASE_URL="http://localhost:8080/v1"
$env:AMANAJE_SLM_MODEL_NAME="amanaje-assistant"
```

## Diagnostics

Use these Painel Amanaje endpoints before trusting a provider:

- `GET /assistant/provider/status`
- `POST /assistant/provider/test?provider=amanaje_slm`
- `GET /assistant/contracts/workflow-draft.schema`

The provider test endpoint sends a tiny draft request, validates the returned `WorkflowDraft`, runs safety review, and reports fallback behavior without creating registry objects.

## Registry-Backed Assistant Models

Painel Amanaje can also use an assistant runtime stored in the Model Registry. Create an `AssistantModel` entry, which is stored through the `LearningModel` table with `model_type="assistant_model"` for compatibility.

Minimal registry payload:

```json
{
  "name": "panel_coding_assistant",
  "description": "Local OpenAI-compatible assistant used for draft generation.",
  "object_type": "learning_model",
  "size": 0,
  "path": "runtime_artifacts/assistant_models/panel_coding_assistant",
  "version": 1,
  "model_type": "assistant_model",
  "parameters": {
    "assistant": {
      "provider_type": "openai_compatible",
      "base_url": "http://localhost:8080/v1",
      "model_name": "amanaje-assistant",
      "model_version": "amanaje-assistant-0.1",
      "temperature": 0.2,
      "max_tokens": 1600,
      "timeout_seconds": 20,
      "enabled": true
    }
  },
  "metrics": {},
  "reference_data": "runtime_artifacts/assistant_datasets/interactions.jsonl",
  "input_features": ["prompt", "context_pack", "target_type"],
  "output_features": ["workflow_draft"],
  "is_trained": false,
  "is_tested": false,
  "is_deployed": false
}
```

Useful endpoints:

- `GET /assistant/models/list`
- `GET /assistant/models/{assistant_model_id}/status`
- `POST /assistant/models/{assistant_model_id}/test`
- `POST /assistant/provider/test?assistant_model_id=1`

Normal assistant UI panels can select one of these registry models. If no model is selected, the configured active provider is used.

## Assistant Training Dataset

Painel Amanaje can materialize a redacted JSONL training dataset from assistant interactions, context packs, evaluations, run summaries, and internal references. The dataset is registered as an `AssistantTrainingDatasetModel`, which is stored as a DatasetModel with `dataset_type="assistant_training_dataset"`.

Useful endpoints:

- `POST /assistant/datasets/rebuild`
- `GET /assistant/datasets/status`
- `GET /assistant/datasets/latest`
- `POST /assistant/models/{assistant_model_id}/training/dataset/attach`

The rebuild endpoint writes a snapshot JSONL and manifest under `runtime_artifacts/assistant_datasets/snapshots/`, updates the latest AssistantTrainingDatasetModel registry row, and records a dataset hash that can be attached to an AssistantModel.

## Transfer-Aware LearningModel Training

The normal `POST /training/{model_id}` route now defaults to transfer-aware training. Set `parameters.training_mode` to control behavior:

- `"transfer"`: default; tries to load the latest compatible base artifact.
- `"fresh"`: ignores previous artifacts and trains from scratch.

Optional parameters:

- `base_model_id`: use another LearningModel as the transfer source.
- `base_artifact_path`: override the artifact path to load.
- `transfer_strategy`: defaults to `"auto"`.
- `warm_start_increment`: sklearn estimators with `warm_start` can increase `n_estimators` before continuing.

Each training run still creates a new run ledger entry, artifacts directory, and MLflow run. The registry row points to the latest artifact and metrics, while `parameters.training_lineage` and `history[]` preserve prior iterations.
