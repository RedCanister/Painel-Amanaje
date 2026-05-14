# Painel Amanaje Worker Hosts

Painel Amanaje treats long operations as ledger-backed jobs. The API creates a run entry, enqueues work in Redis/RQ, and returns a `run_id`. Workers claim jobs from named queues and update the same run ledger as they progress.

## Current Queues

- `amanaje:default`: CPU-safe jobs such as editor execution, assistant operations, scikit-learn training, PyTorch `device=auto`, and studies that do not explicitly request CUDA.
- `amanaje:gpu`: CUDA-requested PyTorch training and study jobs. GPU access is granted only to GPU worker containers or GPU Kubernetes pods.

## Host Contract

Each worker host should advertise:

- `AMANAJE_WORKER_HOST_ID`: stable host or deployment name.
- `AMANAJE_WORKER_ROLE`: `cpu`, `gpu`, or a future specialization.
- `AMANAJE_WORKER_QUEUE`: queue consumed by the worker process.
- `AMANAJE_WORKER_CONCURRENCY`: declared process capacity. RQ workers execute one job at a time, so real parallelism comes from multiple worker processes, containers, or pods.
- `AMANAJE_REDIS_URL`: queue backend.

The API exposes `/runtime/workers` so the frontend can show queue depth, active worker count, and worker heartbeats.

## Scaling

For local Docker Compose, scale CPU parallelism with:

```powershell
docker compose up -d --scale api-worker-cpu=3
```

For GPU jobs, run the GPU profile on a machine with NVIDIA container runtime:

```powershell
docker compose --profile gpu up -d api-worker-gpu
```

For Kubernetes, use separate Deployments per queue. CPU workers scale by replicas. GPU workers use `resources.limits.nvidia.com/gpu: 1` and normally run one RQ worker process per pod.

## Why The Worker Looks "Stuck"

`api-worker-cpu` is expected to stay in the foreground. It prints that it is ready, then blocks while waiting for jobs. That is healthy behavior for `docker compose up`; use `docker compose up -d` when you want the terminal back.
