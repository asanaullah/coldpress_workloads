<!-- Assisted by: Gemini 3 -->
# Coldpress Workloads

This repository contains a collection of pre-defined workloads and discovery parsers (templates) for **Coldpress**, a Kubernetes-native optimization and orchestration framework designed to manage complex HPC-like workloads such as AI.

These templates define the execution environment, dependencies, and hardware requirements for various AI and HPC tasks, allowing standard users to easily submit complex jobs without worrying about the underlying infrastructure details.

## Repository Structure

* **`compute/`**: Contains `WorkloadTemplate` definitions for standard AI/HPC tasks.
    * **LLM Inference & Benchmarking**: `vllm-parser.yaml`, `guidellm-parser.yaml`, `deepti-qwen-parser.yaml`
    * **Distributed Training**: `pytorch-ddp-parser.yaml`
    * **HPC / Scientific**: `gromacs-parser.yaml`
    * **Interactive Environments**: `ml-dev-env.yaml`
    * **Networking/RDMA Benchmarks**: `rdma-perftest-server.yaml`, `rdma-perftest-client.yaml`
* **`discovery/`**: Contains `DiscoveryTemplate` definitions used by administrators to map and analyze cluster hardware.
    * `system-perf.yaml` (CPU states, turbo boost, NUMA)
    * `pcie-topology.yaml` (PCIe tree, link speeds)
    * `network-topology.yaml` (Logical/physical interfaces, MACs, MTU)
    * `rdma-connectivity.yaml` (RDMA NICs, ping latency maps)
    * `comprehensive.yaml` (Deep dive into kernel, networking, and security tunables)

## How to Use

### 1. Admin: Install Templates
Administrators operate with privileged access to define these templates in the cluster. To make a workload available to your users, apply it to the `coldpress-admin` namespace:

```bash
oc apply -f compute/deepti-qwen-parser.yaml

```

*(Note: Admins must also explicitly authorize user namespaces to access specific parsers via annotations).*

### 2. User: Submit a Workload

Users can launch jobs by submitting a `ColdpressResourceAllocator`  or `ComputeJ` Custom Resource. The user simply references the template name and provides the required high-level configuration parameters. The Coldpress Operator will automatically handle hardware allocation, queueing, and job orchestration.

**Example: Running Qwen2.5-Omni Inference**

Create a file named `run-qwen.yaml` with the following contents:

```yaml
apiVersion: coldpress.io/v1
kind: ColdpressResourceAllocator
metadata:
  name: deepti-qwen-test
  namespace: <user-namespace>
spec:
  storage:
    results: <user-pvc>
  tasks:
    - name: "qwen-inference-task"
      template: "deepti-qwen-parser"
      params:
        num_gpus: 4
        model_name: "Qwen/Qwen2.5-Omni-7B"
        prompt: "What do you see in this video?"
        max_new_tokens: 64

```

Submit the workload to the cluster:

```bash
oc apply -f run-qwen.yaml

```

Once submitted, Coldpress will allocate the required 4 GPUs on the most available node, execute the task, and deposit the final outputs (e.g., `output.txt`) into the `<user-pvc>` PersistentVolumeClaim under the `/data/coldpress_results/` directory.
