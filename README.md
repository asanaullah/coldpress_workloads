# Coldpress Workloads

Example workloads demonstrating [Coldpress](https://github.com/asanaullah/coldpress) capabilities for AI/HPC orchestration on Kubernetes.

## Overview

This repository contains real-world example workloads that showcase different Coldpress backends and use cases. Each example includes:
- **intent.yaml** - Coldpress configuration (target backend, dependencies, resources)
- **job-spec.yaml** - Vanilla Kubernetes Job definitions
- **Training scripts or workload code**

## Examples

### PyTorch DDP Training

Distributed PyTorch training using Data Parallel (DDP) across multiple GPUs.

**Backends:** JobSet, Kubeflow PyTorchJob

**Location:** `pytorch_ddp_training/`

**Run with JobSet:**
```bash
cd pytorch_ddp_training/
coldpress generate --intent intent_jobset.yaml
cd output/ddp-training-job/
./run.sh
```

**Run with Kubeflow:**
```bash
cd pytorch_ddp_training/
coldpress generate --intent intent_kubeflow.yaml
cd output/ddp-training-job/
./run.sh
```

**What it demonstrates:**
- Multi-replica DDP training (2 workers, 2 GPUs)
- Automatic DNS coordination via `${REPLICA_*}` macros
- ConfigMap mounting for training scripts
- Hardware discovery via init containers
- Persistent storage for checkpoints

---

### PyTorch Ray Training

Distributed PyTorch training using Ray Train framework.

**Backend:** KubeRay RayJob

**Location:** `pytorch_ray_training/`

**Run:**
```bash
cd pytorch_ray_training/
coldpress generate --intent intent_kuberay.yaml
cd output/ray-training-job/
./run.sh
```

**What it demonstrates:**
- Ray-based distributed training
- Automatic Ray cluster setup (head + workers)
- Resource scaling via replicas (2 pods → 4 GPUs total)
- Ray Train integration

---

### vLLM + GuideLLM Benchmark

Multi-task client-server workflow for LLM inference benchmarking.

**Backends:** JobSet, KServe InferenceService

**Location:** `vllm_guidellm_benchmark/`

**Run with JobSet:**
```bash
cd vllm_guidellm_benchmark/
coldpress generate --intent intent_jobset.yaml
cd output/vllm-benchmark-job/
./run.sh
```

**Run with KServe:**
```bash
cd vllm_guidellm_benchmark/
coldpress generate --intent intent_kserve.yaml
cd output/vllm-kserve-inference/
./run.sh
```

**What it demonstrates:**
- Task dependencies (`wait_for: ready`)
- Cross-task pod DNS resolution via `${REPLICA_*}` macros
- Automatic DNS service creation by JobSet
- Client waits for server readiness before starting
- vLLM inference server setup
- GuideLLM benchmark client automation

**Comparison with manual approach:**
- Manual: 100+ lines of bash for orchestration, polling, error handling
- Coldpress: 14 lines of YAML (intent file)

---

## Prerequisites

1. **Coldpress installed** - See [installation guide](https://github.com/asanaullah/coldpress#installation)
2. **Kubernetes cluster** with required operators:
   - Kueue operator (required for all examples)
   - JobSet operator (for JobSet examples)
   - Kubeflow Training Operator (for PyTorchJob examples)
   - KubeRay operator (for RayJob examples)
   - KServe (for InferenceService examples)
3. **Cluster setup completed** - Namespace, queues, RBAC configured via `coldpress-setup`

## Usage Pattern

All examples follow the same pattern:

```bash
# 1. Navigate to example directory
cd <example-directory>/

# 2. Generate manifests
coldpress generate --intent intent_<backend>.yaml

# 3. Navigate to output
cd output/<job-name>/

# 4. Review generated manifest
cat *.yaml

# 5. Run the job
./run.sh

# 6. Monitor progress
./monitor.sh

# 7. View logs
./logs.sh

# 8. Explore results interactively
./explore.sh

# 9. Copy results locally (optional)
./cp.sh

# 10. Clean up
./cleanup.sh
```

## Project Structure

```
coldpress_workloads/
├── pytorch_ddp_training/       # DDP training (JobSet, Kubeflow)
│   ├── intent_jobset.yaml
│   ├── intent_kubeflow.yaml
│   ├── job-spec.yaml
│   └── train.py
├── pytorch_ray_training/       # Ray training (KubeRay)
│   ├── intent_kuberay.yaml
│   ├── job-spec.yaml
│   └── train.py
└── vllm_guidellm_benchmark/    # LLM inference benchmark (JobSet, KServe)
    ├── intent_jobset.yaml
    ├── intent_kserve.yaml
    ├── job-spec.yaml
    ├── guidellm-client.yaml
    ├── vllm-server.yaml
    ├── service.yaml
    └── run.sh
```

## Contributing

To add a new workload example:

1. Create a new directory with a descriptive name
2. Add required files:
   - `intent_<backend>.yaml` - Coldpress configuration
   - `job-spec.yaml` - Vanilla Kubernetes Jobs
   - Training/workload scripts
3. Test with: `coldpress generate --intent intent_<backend>.yaml`
4. Document what the example demonstrates

## License

See LICENSE file.
