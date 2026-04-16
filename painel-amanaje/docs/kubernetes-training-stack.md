# Painel Amanaje on Kubernetes and Kubeflow

This document is a Kubernetes-first alternative to the stack defined in the repository-root `docker-compose.yaml`. It keeps the same logical components of Painel Amanaje, but replaces the Compose/Celery execution model with Kubernetes pods and Kubeflow-driven training jobs.

## Goal

The current Compose stack defines:

- PostgreSQL for metadata
- Redis for Celery messaging
- Airflow for orchestration
- MLflow for experiment tracking
- FastAPI for the application API
- Optional pgAdmin for database inspection

The Kubernetes version should preserve those responsibilities, but change the execution model:

- `CeleryExecutor` becomes `KubernetesExecutor`
- `airflow-worker` is removed as a long-running Celery worker
- Airflow launches Kubernetes pods for task execution
- model training runs inside Kubeflow-managed pods/jobs
- local bind mounts become persistent volumes or object storage

## Compose-to-Kubernetes mapping

| Compose element | Current role | Kubernetes replacement | Notes |
|---|---|---|---|
| `postgres` | Airflow/application metadata DB | `StatefulSet` + `Service` + `PersistentVolumeClaim` | Keep persistent storage |
| `pgadmin` | DB admin UI | `Deployment` + `Service` | Optional in non-dev clusters |
| `redis` | Celery broker | Removed | Not required with `KubernetesExecutor` |
| `airflow-init` | DB migrate + admin user bootstrap | `Job` or Airflow Helm init jobs | Run once during deployment |
| `airflow-webserver` | Airflow UI/API | `Deployment` + `Service` + `Ingress` | Public or internal route |
| `airflow-scheduler` | DAG scheduling | `Deployment` | Must be enabled in Kubernetes |
| `airflow-worker` | Celery worker pool | Removed | Replaced by task pods |
| `airflow-triggerer` | Deferrable task support | `Deployment` | Recommended to keep |
| `mlflow` | Experiment tracking server | `Deployment` + `Service` + PVC/object storage | Prefer external artifact store |
| `api` | FastAPI backend | `Deployment` + `Service` + `Ingress` | Mount shared data/model storage |
| `./dags`, `./plugins` | Local Airflow code mounts | Image bake, `git-sync`, or config volume | Avoid hostPath in cluster |
| `./data`, `./models`, `./mlruns` | Local project storage | PVCs and/or S3/MinIO buckets | Required for shared access |

## Target architecture

The Kubernetes version of Painel Amanaje should look like this:

1. Airflow runs inside the cluster using `KubernetesExecutor`.
2. The FastAPI service receives a training request from `/training/{model_id}`.
3. Airflow orchestrates validation, metadata preparation, and training submission.
4. The heavy training step is executed by Kubeflow, not by a Celery worker.
5. Kubeflow starts an isolated training pod or `PyTorchJob`.
6. The training container reads datasets from shared storage, logs metrics to MLflow, and writes model artifacts to the model store.
7. Airflow and the API update status in the application database.

This means the cluster no longer depends on a permanent Celery worker tier. Each workload is started as a pod only when needed.

## Airflow changes

In `docker-compose.yaml`, Airflow is configured with:

- `AIRFLOW__CORE__EXECUTOR: "CeleryExecutor"`
- `AIRFLOW__CELERY__BROKER_URL`
- `AIRFLOW__CELERY__RESULT_BACKEND`
- a commented `airflow-worker` service

In Kubernetes, that should change to:

- `AIRFLOW__CORE__EXECUTOR: "KubernetesExecutor"`
- remove Redis/Celery settings
- enable scheduler and triggerer
- let Airflow create short-lived Kubernetes pods for task execution

For Painel Amanaje, this is the direct replacement for the Celery worker concept: instead of a fixed worker pod, Airflow runs each eligible task inside its own Kubernetes pod.

### Recommended Airflow deployment model

Use the official Airflow Helm chart and configure it around Kubernetes execution:

```yaml
executor: KubernetesExecutor

config:
  AIRFLOW__CORE__LOAD_EXAMPLES: "False"
  AIRFLOW__CORE__DAGS_FOLDER: /opt/airflow/dags
  AIRFLOW__CORE__PLUGINS_FOLDER: /opt/airflow/plugins
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:${POSTGRES_PASSWORD}@postgres/airflow
  AIRFLOW__KUBERNETES__NAMESPACE: painel-amanaje

redis:
  enabled: false

workers:
  enabled: false

triggerer:
  enabled: true
```

### Airflow responsibilities in the Kubernetes version

Airflow should keep orchestration concerns:

- receive or proxy training requests
- validate dataset/model availability
- create MLflow run metadata
- submit Kubeflow runs
- monitor completion
- publish status back to the API/database

Airflow should not do the heavy training work inside the scheduler/webserver containers.

## Kubeflow changes for training

Kubeflow becomes the execution layer for model training. For Painel Amanaje, this is the cleanest replacement for "Celery worker does training".

### Recommended training pattern

Use one of these two patterns:

- Kubeflow Pipelines if training needs a multi-step workflow such as preprocessing, training, evaluation, and registration
- Kubeflow `PyTorchJob` if the main need is a dedicated distributed or isolated PyTorch training job

For the current project, Kubeflow Pipelines is the better control plane, and the actual training step can internally run a `PyTorchJob` when needed.

### Training pod contract

Each training run should receive:

- `model_id`
- `dataset_id`
- training hyperparameters
- optional `study_id`
- MLflow tracking URI
- storage locations for datasets and model artifacts

Each training run should produce:

- trained model artifact
- metrics and params in MLflow
- serialized training summary
- status update for the API layer

### Practical execution flow

1. `POST /training/{model_id}` arrives at FastAPI.
2. The API writes a job record and triggers an Airflow DAG or direct Airflow API call.
3. Airflow launches a Kubernetes pod for orchestration or directly submits a Kubeflow pipeline run.
4. Kubeflow starts the training environment in an isolated pod.
5. The training image imports the code from `api/app/utils/training.py` and related modules.
6. MLflow logs are written to the cluster MLflow service.
7. The trained model is saved to shared model storage.
8. Airflow marks the run complete and the API exposes the final status.

## Storage model

The Compose file relies on local mounts:

- `./data:/data`
- `./models:/models`
- `./mlruns:/mlruns`
- `./dags:/opt/airflow/dags`
- `./plugins:/opt/airflow/plugins`

In Kubernetes, those should become shared storage primitives:

| Current mount | Kubernetes equivalent | Recommendation |
|---|---|---|
| `./data` | PVC or object storage bucket | Prefer object storage for reproducible training inputs |
| `./models` | PVC or model artifact bucket | Prefer shared artifact storage |
| `./mlruns` | MLflow backend artifact store | Prefer S3/MinIO instead of local disk |
| `./dags` | git-sync sidecar or baked DAG image | Best for Airflow deployments |
| `./plugins` | baked image or config volume | Keep versioned with deployment |

For a real cluster, object storage is better than host-mounted volumes because both Airflow and Kubeflow need consistent access to the same datasets and artifacts.

## Networking and exposure

The Compose version publishes:

- Airflow on `8080`
- MLflow on `5000`
- API on `8000`
- pgAdmin on `5050`
- PostgreSQL on `5432`

In Kubernetes, expose them with:

- `ClusterIP` services for internal traffic
- `Ingress` for Airflow, API, MLflow, and optionally pgAdmin
- no public PostgreSQL service unless explicitly required

Typical internal service names:

- `postgres`
- `airflow-webserver`
- `mlflow`
- `painel-api`

## Secrets and configuration

The Compose file uses `.env` and inline environment variables. In Kubernetes, move them into:

- `Secret` for passwords, tokens, connection strings
- `ConfigMap` for non-sensitive runtime configuration

Examples:

- `POSTGRES_PASSWORD`
- `AIRFLOW_FERNET_KEY`
- MLflow tracking configuration
- application-level environment variables consumed by the API

## Suggested repository layout

If this architecture is implemented, a clean starting point is:

```text
k8s/
  namespace.yaml
  postgres/
    statefulset.yaml
    service.yaml
    pvc.yaml
  mlflow/
    deployment.yaml
    service.yaml
    pvc.yaml
  api/
    deployment.yaml
    service.yaml
    ingress.yaml
  airflow/
    values.yaml
  storage/
    data-pvc.yaml
    models-pvc.yaml
  secrets/
    app-secrets.yaml

kubeflow/
  pipelines/
    training_pipeline.py
  jobs/
    pytorchjob.yaml
```

## Recommended deployment order

1. Create namespace, secrets, and persistent storage.
2. Deploy PostgreSQL.
3. Deploy MLflow with persistent artifact storage.
4. Deploy Airflow with `KubernetesExecutor`.
5. Deploy Kubeflow components and the training pipeline/job definitions.
6. Deploy the FastAPI service.
7. Expose Airflow, API, and MLflow through Ingress.

## What changes relative to `docker-compose.yaml`

This Kubernetes version changes the operational model in three important ways:

1. Redis and Celery workers are no longer part of the runtime path.
2. Airflow becomes a Kubernetes-native orchestrator instead of a Compose-managed service group.
3. Training moves into Kubeflow-managed pods/jobs instead of staying inside long-lived worker containers.

## Recommended final state for Painel Amanaje

For this project, the most coherent cluster architecture is:

- Airflow on Kubernetes with `KubernetesExecutor`
- PostgreSQL as the shared metadata store
- MLflow as the experiment tracking service
- FastAPI as the user-facing control plane
- Kubeflow Pipelines for the training workflow
- `PyTorchJob` for isolated or distributed training when needed
- PVCs and/or object storage replacing local volume mounts

That preserves the intent of the current Compose stack, while making the training environment truly cloud-native and eliminating the dependency on Celery worker pods.
