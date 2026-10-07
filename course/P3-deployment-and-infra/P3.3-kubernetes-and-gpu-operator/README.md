# P3.3 — Kubernetes + NVIDIA GPU Operator

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) on kind (Docker), with `mockllm` standing in for the GPU backend; ![T3](https://img.shields.io/badge/tier-T3%20AWS%20cluster-red) on EKS with the GPU Operator |
| **Time** | ≈30 min reading + ≈9 h hands-on |
| **Prerequisites** | P3.1, P3.2 |
| **Pinned** | GPU Operator `v26.7.1` (`NVIDIA/gpu-operator@cc6fd600`), k8s-device-plugin `v0.20.1`, dcgm-exporter `4.8.4` |

## Learning objectives

1. Use the Kubernetes objects a GPU server needs: Deployment, Service, probes, requests and limits, taints and tolerations, nodeSelector/affinity, PodDisruptionBudget, securityContext.
2. Explain what each GPU Operator component does: driver, container toolkit, device plugin, GPU Feature Discovery, DCGM exporter, MIG manager.
3. Schedule a pod onto a GPU (`nvidia.com/gpu: 1`), make readiness wait for model load, and drain in-flight streams on shutdown.

---

## 1. How a pod gets a GPU

```
pod spec: resources.limits["nvidia.com/gpu"] = 1   +   toleration for the GPU taint
   │ scheduler: only nodes advertising allocatable nvidia.com/gpu ≥ 1 (and tolerated taints) are candidates
   ▼
kubelet on the chosen node ── asks the NVIDIA device plugin ── "which GPU?" ── device plugin picks GPU-UUID-…
   │ container runtime + NVIDIA container toolkit inject /dev/nvidiaX + driver libs into the container
   ▼
container sees exactly that GPU (CUDA_VISIBLE_DEVICES set accordingly)
```

| Operator component | Job | On EKS AL2023 NVIDIA AMI |
|---|---|---|
| driver (DaemonSet) | installs the kernel driver on each GPU node | **disabled**: the AMI has it |
| container toolkit | configures containerd to inject GPUs | **disabled**: the AMI has it |
| **device plugin** | advertises `nvidia.com/gpu` to the kubelet and allocates GPUs to pods | on |
| **GFD** (GPU Feature Discovery) | labels nodes: `nvidia.com/gpu.product`, `.memory`, `.compute.major`, … | on |
| **DCGM exporter** | GPU metrics (util, memory, power, XID errors) for Prometheus (P3.5) | on |
| MIG manager | partitions A100/H100 GPUs (P4.4) | off for now |

`platform/deploy/gpu-operator-values.yaml` encodes the table. Its keys are UNVERIFIED against the chart at v26.7.1, so check `deployments/gpu-operator/values.yaml` in the pinned repo.

## 2. A GPU server's pod spec, line by line

Read `platform/deploy/base/vllm.yaml`. The non-obvious lines:

- **`startupProbe` with a 15-minute budget.** Loading 16 GB of weights and capturing CUDA graphs takes minutes. Without a startup probe, the liveness probe kills the pod mid-load, forever.
- **`readinessProbe`.** The Service only routes to the pod once `/health` answers.
- **`preStop: sleep 20` + `terminationGracePeriodSeconds: 120`.** When a pod terminates, endpoint removal and SIGTERM race. The sleep keeps the pod serving until the load balancers stop sending to it. Then vLLM finishes its in-flight streams within the grace period.
- **`maxUnavailable: 0, maxSurge: 1`.** A rollout adds the new pod before removing the old one, so capacity never dips. On a scale-from-zero GPU pool, the surge pod waits for a new node: plan your rollout time.
- **`/dev/shm` as a Memory emptyDir.** PyTorch and NCCL use shared memory, and the container default (64 MiB) breaks multi-process serving.
- **`readOnlyRootFilesystem`, `runAsNonRoot`, `drop: [ALL]`.** This satisfies the **restricted** Pod Security Standard set on the namespace.
- **The API key comes from a Secret,** never inline.

## 3. Scheduling on GPU properties

With GFD labels you can require hardware properties instead of instance names:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - { key: nvidia.com/gpu.memory, operator: Gt, values: ["23000"] }   # MiB; label name/units UNVERIFIED
```

Exercise 4 makes this work and verifies the actual label names on a real node (`kubectl get node -o json | jq .metadata.labels`).

> **Predict first.** You apply `vllm.yaml` to a cluster whose GPU pool is at 0 nodes. List every event, in order, until the pod is Ready, with a rough duration for each. Then compare with `kubectl get events -n s2s --sort-by=.lastTimestamp` on EKS. You'll reuse this timeline in P3.6 (cold starts). Something like:
> 1. pending, unschedulable
> 2. the autoscaler adds a node (1–3 min)
> 3. the node joins, and the device plugin advertises GPUs (~1 min)
> 4. image pull (P3.1)
> 5. weight load (P3.7)
> 6. graph capture
> 7. Ready

---

## Walkthrough

```bash
# T0 (kind)
kind create cluster --name s2s --config platform/deploy/kind/kind-config.yaml
docker build -f env/Dockerfile.platform -t s2s-platform:dev . && kind load docker-image s2s-platform:dev --name s2s
kubectl apply -k platform/deploy/overlays/kind
kubectl -n s2s rollout status deploy/mockllm
kubectl -n s2s port-forward svc/mockllm 8001:8001 &
curl -s localhost:8001/v1/completions -H 'content-type: application/json' -d '{"prompt":"hi","max_tokens":4}'
uv run python platform/deploy/policy.py
# T3 (EKS, after P3.2's make up && make addons)
kubectl get nodes -L nvidia.com/gpu.product
kubectl -n s2s create secret generic vllm-api-key --from-literal=key="$(openssl rand -hex 16)"
kubectl apply -k platform/deploy/overlays/eks
```

## What you should see

- kind: two mockllm pods on the `s2s/pool=gpu` worker; the policy check prints OK for every Deployment.
- EKS (`TODO(run-on: EKS g6.xlarge)`): the vLLM pod pending → a GPU node appears → GFD labels it → the pod goes Ready after the startup probe passes. Record the timeline.

## Exercises

```bash
uv run pytest course/P3-deployment-and-infra/P3.3-kubernetes-and-gpu-operator/exercises
```

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [Manifests pass policy (and kubeconform)](exercises/01-policy.md) | T0 | `test_manifests.py` |
| 2 | [Readiness gates on warm-up](exercises/02-readiness.md) | T0 (kind) | a Service with no endpoints until warm-up completes |
| 3 | [preStop drain: zero dropped streams during a rollout](exercises/03-drain.md) | T0 (kind) | `drain_test.py` streams through a rollout; 0 errors |
| 4 | [Schedule only on GPUs with ≥ 24 GB](exercises/04-gfd-affinity.md) | T3 | the pod lands on the right node, and the label names are recorded |

## Common mistakes

- Requesting `nvidia.com/gpu` in `requests` only. Extended resources must be set in `limits`; if `requests` are given, they must equal the limits.
- No toleration, so the pod sits Pending forever on a tainted GPU pool.
- A liveness probe without a startup probe, which restart-loops during model load.
- Forgetting `/dev/shm`. You get cryptic NCCL or dataloader errors.
- Treating `kubectl delete pod` as graceful for streams. Without preStop and a grace period, clients see cut streams.

## Go deeper

- GPU Operator docs (`NVIDIA/gpu-operator@cc6fd600` and docs.nvidia.com/datacenter/cloud-native/: UNVERIFIED from the build environment).
- k8s-device-plugin README: sharing (time-slicing) configs, which P4.4 uses.
- *Kubernetes Up & Running*: Pods, Deployments, Services, probes.
- EKS user guide: GPU AMIs.

**Next:** [P3.4 Serving stacks on Kubernetes](../P3.4-serving-stacks-on-k8s/README.md).
