# P3.6: Autoscaling, cold starts, spot (+ #3)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the scaling policy, KEDA on kind with the mock backend, and the spot-interruption simulation. ![T3](https://img.shields.io/badge/tier-T3%20AWS%20cluster-red) for the scale-up timeline and cold-start bench on EKS |
| **Time** | ≈30 min reading + ≈10 h hands-on |
| **Prerequisites** | P2.4 (#4 knee), P3.3 (probes, drain), P3.5 (Prometheus) |
| **You will build** | **#3**: a queue-based GPU autoscaler with KEDA, spot-first Karpenter NodePools with an on-demand fallback, and a measured cold-start breakdown |

## Learning objectives

1. Explain why CPU- or GPU-utilization HPA is the wrong signal for LLM serving, and use **queue depth + in-flight work** instead.
2. Write a KEDA `ScaledObject` with the Prometheus scaler, including scale-to-zero, and know what it costs.
3. Break a cold start into phases (node provisioning, image pull, weight load, warm-up) and attack each one.
4. Survive a **spot interruption** (two-minute notice, drain, retry) with zero failed requests, and fall back to on-demand capacity.

---

## 1. The wrong signal and the right one

| Signal | What goes wrong |
|---|---|
| CPU utilization | the CPU is idle while the GPU works, so it never scales |
| `DCGM_FI_DEV_GPU_UTIL` | it saturates at 100% with *one* request in decode (P3.5), long before throughput saturates. It's also flat from 40 to 400 concurrent users |
| request rate (req/s) | ignores request length: 10 req/s of 8k-token prompts is a different load from 10 req/s of chat |
| **running + waiting requests** | measures the work actually in the system. Waiting > 0 for long is the SLO risk |

The target per replica comes from your **#4 knee** (P2.4). If the knee is at concurrency *C\**, target about 0.8 · *C\**, so a replica runs just below where TTFT turns up.

```
desired = ceil( (Σ running + Σ waiting) / target_per_replica )
```

`platform/autoscaler/policy.py` makes this decision a pure function, with the rules the HPA `behavior` block encodes. Scale up immediately but bounded (+2 pods/min). Scale down only after the *maximum* demand over a 5-minute window has dropped, −1 pod per 2 min. Keep one warm replica until 10 minutes of zero demand. GPUs are slow and expensive to get back, so **asymmetry is the point**.

## 2. KEDA

KEDA turns the Prometheus query into an HPA external metric and adds **scale to zero**, which plain HPA can't do. See `platform/autoscaler/scaledobject.yaml`.

**The scale-from-zero problem.** At zero replicas, no vLLM pod exports `vllm:num_requests_waiting`. The demand signal has to come from something that is always up. Here that's the gateway's `gateway_inflight` gauge (#6), so a second trigger reads it. Requests that arrive during the cold start wait at the gateway, or get a fast `503 Retry-After` if you prefer failing quickly.

Scale-to-zero trade-off: you save idle GPU-hours (P3.5 "waste"), and the first user after idle pays a full cold start (§3). It's good for dev and batch, and rarely acceptable for an interactive SLO.

## 3. Cold-start anatomy

| Phase | Typical driver | Mitigation |
|---|---|---|
| node provisioning (Pending → node Ready) | EC2 launch + GPU driver/operator readiness | warm pool / over-provisioning "balloon" pods with a low PriorityClass; keep `min_size ≥ 1` in business hours |
| image pull | a 10+ GB CUDA + PyTorch image | pre-pull via a DaemonSet or bake into the AMI; smaller images (P3.1); a registry in-region |
| weight load | model GB ÷ effective GB/s from S3/disk (P3.7) | node-local NVMe cache, streaming loaders, safetensors mmap |
| warm-up | CUDA graph capture, torch.compile, KV allocation | keep the graph capture sizes small; measure with startup logs |

**Measure, don't guess.** `python -m autoscaler.coldstart --pod <pod>` prints the phase table from Kubernetes events. This module's bench is that table before and after the mitigations.

> **Predict first.** An 8B fp16 model is ≈16 GB. If S3 → node delivers *B* GB/s (measure it in P3.7), weight load alone is 16/*B* seconds. Add your image pull time (image size ÷ pull throughput) and a node launch of a few minutes. Which phase dominates? Write the prediction down before running the bench.

## 4. Spot

Spot GPUs are much cheaper, and AWS can reclaim them with a **two-minute** interruption notice (EC2 docs, UNVERIFIED wording). The plan:

1. **Detect.** Karpenter (or the AWS Node Termination Handler) watches the interruption events and cordons and drains the node.
2. **Drain gracefully.** The P3.3 `preStop` hook makes the pod fail readiness, and in-flight requests finish within `terminationGracePeriodSeconds`. Keep it under ~90 s to leave margin within the two minutes.
3. **Retry what's left.** The gateway (#6) retries requests that failed *before the first streamed byte* on another backend. Streams already in progress can't be retried transparently (P2.7): bound the max response length on spot replicas, or accept them as the error budget's use.
4. **Fall back.** `karpenter-nodepools.yaml` has a spot pool (weight 100) and an on-demand pool (weight 10). When spot capacity is unavailable, Karpenter launches on-demand. Diversify instance types in the spot pool to cut interruption rates.

P3.5's cost model has `spot_overhead`: the fraction of work redone after interruptions. Use it to check that spot is still cheaper once replays are counted.

---

## Walkthrough

```bash
uv run pytest course/P3-deployment-and-infra/P3.6-autoscaling/exercises      # T0: policy + spot simulation
# T0 on kind (P3.3 cluster with mockllm port-forwarded on 8001 + kube-prometheus-stack + KEDA):
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda -n keda --create-namespace --version 2.21.0        # version from SOURCES.md
kubectl apply -f platform/autoscaler/kind/scaledobject-mock.yaml
uv run python -m loadgen.cli --url http://localhost:8001 --rates 2 8 16 --duration 120   # watch: kubectl -n s2s get hpa,pods -w
```

The T3 path on EKS is in [aws.md](aws.md).

## What you should see

- `test_policy.py`: all pass after exercise 1. The flapping test shows the stabilized policy changes replica count far fewer times than the naive formula on a noisy trace.
- kind: at 2 req/s the mock stays at 1 replica. At 16 req/s it grows to `max`. About 10 min after the load stops, it goes back to 0 (or your `minReplicaCount`).
- EKS (`TODO(run-on: EKS g6.xlarge)`): a timeline where pods go Pending (no GPU node), a node appears, the image pulls, weights load, and the pod becomes Ready. The TTFT spike during that window is the cold start users see.

## Bench (T3)

| cold start phase | before (s) | after mitigations (s) |
|---|---|---|
| created → scheduled (node provisioning) | `TODO(run-on: EKS g6.xlarge)` | |
| pulling → pulled (image) | `TODO(run-on: EKS g6.xlarge)` | |
| started → ready (weights + warm-up) | `TODO(run-on: EKS g6.xlarge)` | |
| **total** | | |

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [The scaling policy as a pure function](exercises/01-policy.md) | T0 | `test_policy.py` |
| 2 | [KEDA ScaledObject on kind with the mock backend](exercises/02-keda-kind.md) | T0 | `check_scale_log.py` on your `kubectl get hpa -w` log |
| 3 | [#3 on EKS: load ramp and scale-up timeline](exercises/03-eks-timeline.md) | T3 | `check_scale_log.py` + the cold-start table |
| 4 | *(hard)* [Simulated spot interruption with zero failed requests](exercises/04-spot.md) | T0 (sim) / T3 | `test_spot_sim.py` |

## Common mistakes

- Scaling on GPU utilization: one decode request reads 100%.
- `cooldownPeriod` too short: scale to zero, then a cold start for the next user a minute later.
- No scale-from-zero signal: KEDA queries a metric that disappears with the last pod.
- `terminationGracePeriodSeconds` longer than the spot notice: the node is killed mid-drain anyway.
- One instance type in the spot pool: one capacity crunch takes everything.

## Go deeper

- KEDA docs: ScaledObject spec, Prometheus scaler (`kedacore/keda@v2.21.0`).
- Karpenter docs: NodePools, disruption, interruption handling (UNVERIFIED for the version you install).
- Modular handbook: *Fast scaling*. Silicon to Scale ch. 15.

**Next:** [P3.7 Weight storage and lazy loading](../P3.7-weight-delivery/README.md).
