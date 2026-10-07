# P3.4 — Serving stacks on Kubernetes (+ #1)

| | |
|---|---|
| **Tier** | ![T3](https://img.shields.io/badge/tier-T3%20AWS%20cluster-red) EKS with 1–2 GPU nodes (`infra/aws/eks`, `gpu_max = 2`); manifest checks are T0 |
| **Time** | ≈35 min reading + ≈14 h hands-on (≈6 GPU-node-hours) |
| **Prerequisites** | P3.3, P2.3 (#4), P2.7 (#6) |
| **Pinned** | KServe `v0.21.0`, NVIDIA Dynamo `v1.5.0`, llm-d `v0.10.0`, Ray `2.59.0` (all in SOURCES.md) |
| **You will build** | **#1**: a self-hosted inference cluster serving an open model behind #6, with health checks and a reproducible latency report |

## Learning objectives

1. Compare four ways to run LLM servers on Kubernetes: a plain Deployment, **KServe**, **NVIDIA Dynamo**, **llm-d** and **Ray Serve**, and know what each adds.
2. Explain **prefill–decode disaggregation** (DistServe) and when it pays.
3. Explain **KV/prefix-aware routing**, and why round-robin is wrong for LLMs with shared prefixes.
4. Ship #1 and its latency report.

---

## 1. What each stack adds over a Deployment

| Stack | Core idea | You get | You pay |
|---|---|---|---|
| plain Deployment + Service (P3.3) | K8s primitives | full control, minimal moving parts | you build autoscaling, routing and canaries yourself |
| **KServe** | an `InferenceService` CRD over serving runtimes | a standard API, canaries/traffic splits, scale-to-zero (with Knative), a vLLM-based HF runtime | CRDs + a controller. Knative mode adds complexity; RawDeployment mode is simpler |
| **Ray Serve** (KubeRay `RayService`) | Python-first composition of deployments | multi-model pipelines, Python pre/post-processing, autoscaling on ongoing requests | a Ray cluster to operate; one more layer between you and vLLM |
| **NVIDIA Dynamo** | distributed inference framework | **disaggregated** prefill/decode, **KV-aware routing**, KV transfer, a planner | a newer, faster-moving project; more components |
| **llm-d** | K8s-native, vLLM-based, with an Envoy inference scheduler | **prefix/KV-aware endpoint picking**, disaggregation guides | the Gateway API + inference extension stack |

Use the plain Deployment unless you need what a stack adds. Measure what it adds with #4.

## 2. Prefill–decode disaggregation

Prefill is compute-bound and bursty. Decode is memory-bound and steady (P1.2). On one GPU they interfere: a long prefill stalls every decode (P2.3 §2). Chunked prefill mitigates that. **Disaggregation** (DistServe, 2401.09670) removes the interference entirely:

- prefill workers compute the KV cache and **ship it** over the network (NVLink, RDMA or TCP) to decode workers
- decode workers only decode
- each pool is sized and parallelized for its own phase

It pays when:
- prompts are long and outputs are moderate (prefill-heavy)
- you have strict ITL SLOs under bursty prefill load
- the interconnect makes KV transfer cheap relative to recomputing it

It costs:
- KV transfer bandwidth: 128 KiB/token for an 8B GQA model in bf16 (P1.2), so a 4k prompt is 512 MiB per request
- operational complexity
- at small scale, idle imbalance between the pools

> **Predict first.** An 8B model, 4,000-token prompts, 200-token outputs, KV at 128 KiB/token. Each request ships 4000 × 128 KiB ≈ 500 MiB from prefill to decode. Over 25 Gb/s TCP (≈3 GB/s), that is ≈ 0.17 s per request **added to TTFT**. With an L4's prefill of ~0.5–1 s for 4k tokens (P1.4 §4), is disaggregation worth it on TCP for 1 req/s? For 10 req/s? Write your reasoning down, then measure with Dynamo (exercise 4).

## 3. KV/prefix-aware routing

With prefix caching (P2.3), *which replica* gets a request matters: send turn 6 of a conversation to the replica that holds turns 1–5 in its cache, and prefill drops to the new message. A round-robin Service scatters conversations, so each replica's cache holds fragments and the hit rates collapse. llm-d's scheduler and Dynamo's router score replicas by expected cache hits and load. Your #6 gateway can approximate it with consistent hashing on a conversation ID. That is a P3.4 exercise.

## 4. #1: the cluster, as a checklist

`platform/deploy` plus the gateway make up #1. It is "done" when **every** item below is true and recorded in `results/p34_cluster.md`:

- [ ] EKS from `infra/aws/eks` (`make check` clean); GPU node group min 0
- [ ] GPU Operator, DCGM exporter and kube-prometheus-stack running (`make addons`)
- [ ] vLLM Deployment (`overlays/eks`) Ready, with the probes and drain behaviour of P3.3
- [ ] the #6 gateway in front, the only entry point, with tenant keys from a Secret; no unauthenticated endpoint
- [ ] a rolling update with **0 failed requests** (`drain_test.py` running inside the cluster)
- [ ] a latency report from #4 via P2.4's `report.py`, with all required fields
- [ ] `make down` executed, and the teardown confirmed by `cost.sh`

---

## Walkthrough (T3)

```bash
cd infra/aws/eks && make up && make addons && cd -
kubectl -n s2s create secret generic vllm-api-key --from-literal=key="$S2S_API_KEY"
kubectl apply -k platform/deploy/overlays/eks                     # baseline: plain Deployment
kubectl -n s2s port-forward svc/vllm 8000:8000 &
PYTHONPATH=platform python -m loadgen.cli --url http://127.0.0.1:8000 --rates 1 2 4 --duration 60 --out results/p34_deploy.json
# then KServe (install per its docs at v0.21.0), Ray Serve (KubeRay), Dynamo and llm-d (platform/deploy/{dynamo,llmd}/README.md)
```

## What you should see

`TODO(run-on: EKS 2× g6.xlarge)`. Expected shapes:

- the plain Deployment and KServe RawDeployment are within noise of each other: same engine, thin wrapper
- Ray Serve adds a small per-request overhead
- with a shared-prefix workload, llm-d's prefix-aware routing gives a higher cache hit rate and lower TTFT than a round-robin Service over the same replicas
- with long prompts under load, Dynamo's disaggregated mode lowers p90 ITL, at some TTFT cost

## Exercises

| # | Exercise | Tier | Check |
|---|---|---|---|
| 1 | [Same model, two stacks, one table](exercises/01-two-stacks.md) | T3 | `results/p34_compare.md` from `loadgen` JSONs |
| 2 | [Rolling update with zero dropped requests](exercises/02-zero-drop.md) | T3 (rehearse on kind) | `drain_test.py` exit code 0 |
| 3 | [#1 checklist + report](exercises/03-cluster-report.md) | T3 | `check_cluster_report.py` checks the required sections |
| 4 | [Disaggregated vs aggregated on Dynamo or llm-d](exercises/04-disaggregation.md) | T3 (2 GPUs) | the table + an explanation tied to §2's arithmetic |
| 5 | [Session-affinity routing in #6](exercises/05-affinity-routing.md) | T0 | `test_affinity.py`: the same session sticks to one backend; fails over when it dies |

## Common mistakes

- Comparing stacks at different vLLM versions or flags. Pin the engine image across all of them.
- KServe in Knative (serverless) mode before you need scale-to-zero: more moving parts for the same latency.
- Disaggregating on a slow network and blaming the framework.
- Using session affinity *without* failover: a dead replica black-holes its sessions.

## Go deeper

- DistServe (2401.09670) · the Modular handbook's *Prefill-decode disaggregation*, *Inference routing* and *Distributed inference* pages.
- KServe `docs/`, Dynamo `docs/`, llm-d `docs/` and `guides/`, the Ray Serve docs (pinned tags in SOURCES.md).
- `aws-samples/awsome-inference@d3236c8c`: EKS inference examples.
- Inference Engineering Academy, fleet/Dynamo course: **UNVERIFIED** (see GAPS.md).

**Next:** [P3.5 Observability](../P3.5-observability/README.md).

## Animation

[`animations/p3-request-path.html`](../../../animations/p3-request-path.html): request → gateway → queue → autoscaler → pod, with a cold vs warm toggle.
