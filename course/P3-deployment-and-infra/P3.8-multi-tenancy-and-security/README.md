# P3.8: Multi-tenancy and security (+ #8, stretch)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for everything: keys and rotation, the audit log, the noisy-neighbor test (mock backend), and NetworkPolicies on kind. ![T3](https://img.shields.io/badge/tier-T3%20AWS%20cluster-red) optional, to repeat the NetworkPolicy and IRSA checks on EKS |
| **Time** | ≈30 min reading + ≈8 h hands-on |
| **Prerequisites** | P2.7 (#6 gateway: keys, rate limits, token budgets), P3.3 (namespaces, Pod Security), P3.2 (IAM, Secrets Manager) |
| **You will build** | **#8**: key rotation with no downtime, network isolation, a tamper-evident audit log, and a test proving one tenant can't break another's SLO |

## Learning objectives

1. Choose a tenancy model (shared pool, dedicated deployment, or MIG slice) from isolation needs and cost.
2. Layer the controls: **identity** (keys, IAM), **quota** (rate, tokens, concurrency), **network** (NetworkPolicy), **runtime** (Pod Security, seccomp, sandboxes).
3. Handle secrets properly: Secrets Manager/SSM → pod at runtime, least-privilege roles (IRSA / EKS Pod Identity), nothing in Git.
4. Make actions auditable with an append-only, tamper-evident log.

---

## 1. Tenancy models

| Model | Isolation | Cost efficiency | Use when |
|---|---|---|---|
| **shared pool** (one deployment, gateway-enforced quotas) | logical: keys, quotas, separate metrics | best: bursts share the GPUs | internal teams, same trust level |
| **dedicated deployment** per tenant (own namespace, own pods) | process + network + scheduling | worse: each tenant's idle capacity is waste | a noisy or regulated tenant, a custom model or LoRA set |
| **MIG slice** per tenant (P4.4) | hardware partition: memory, cache and SMs | good for small models | small models with strict isolation on A100/H100-class GPUs |

The KV cache is the subtle shared resource in a shared pool. **Prefix caching shares KV blocks across requests**, and a timing side channel can reveal whether another tenant sent the same prefix. Per-tenant cache salting (vLLM's `cache_salt` request field, UNVERIFIED for your version) or per-tenant deployments close it.

## 2. The layers

| Layer | Control | Where in this repo |
|---|---|---|
| identity | per-tenant API keys stored as **hashes**, rotated in two phases | `platform/tenancy/keys.py`, gateway `previous_key_sha256` |
| quota | rate (token bucket), tokens per window, models allowed | #6 gateway (P2.7) |
| network | default-deny, explicit allows, IMDS blocked | `platform/tenancy/networkpolicies.yaml` |
| runtime | Pod Security `restricted`, non-root, read-only rootfs, drop ALL, seccomp | `platform/deploy/policy.py` (P3.3), `seccomp-inference.json` |
| cloud identity | IRSA / Pod Identity: one role per ServiceAccount, scoped to one S3 prefix or one secret | P3.2 `infra/aws/eks` |
| audit | hash-chained JSONL plus an off-box checkpoint | `platform/tenancy/audit.py` |

**Secrets.** The HF token and backend API keys live in Secrets Manager or SSM Parameter Store (SecureString). They reach the pod via the Secrets Store CSI driver or an External Secrets operator, using a role that can read **that one secret**. Never put them in the repo, a ConfigMap, a Terraform variable default, or a container image layer. P3.1's Dockerfile policy and `policy.py`'s `secretKeyRef` rule enforce part of this.

**Tool and code execution.** If the platform runs model-generated code (an agent's tools), that code is untrusted. seccomp and dropped capabilities **narrow** the kernel surface. They are not a sandbox. Run untrusted code in **gVisor** (a `runsc` RuntimeClass) or **Firecracker** microVMs, in a namespace with no credentials and egress only through an allow-listed proxy.

## 3. Key rotation without downtime

```
phase 1: config key_sha256 = H(new), previous_key_sha256 = [H(old)]   → roll gateway (2 replicas, maxUnavailable 0)
         deliver `new` via the secret store; the tenant switches whenever it's ready
phase 2: once metrics show the old key unused, remove H(old)          → roll gateway
```

There is never a moment when the tenant's working key is rejected. The same two-phase pattern rotates backend keys and the audit HMAC key. For the HMAC key, record which key generation signed each record.

## 4. The audit log

Each record stores `prev` and `hash = HMAC(key, prev ‖ canonical(record))`. Changing, deleting or reordering any record breaks the chain from that point on. Truncating the tail doesn't break it, so periodically copy the latest hash somewhere the writer can't modify. S3 with Object Lock, or a separate account's CloudWatch Logs, both work. Log **who / what / outcome / token counts**. Never log prompts, completions or full keys.

> **Predict first.** Tenant A (rps limit 5) floods 100 concurrent requests. Tenant B sends one request per second. Without the gateway, what happens to B's TTFT on a backend with 4 sequence slots? With the gateway, how many of A's requests get 429? Exercise 4 measures both.

---

## Walkthrough

```bash
uv run pytest course/P3-deployment-and-infra/P3.8-multi-tenancy-and-security/exercises
PYTHONPATH=platform uv run python -c "from tenancy import keys; k=keys.generate(); print(k); print(keys.key_hash(k))"
# kind (P3.3 cluster):
kubectl apply -k platform/deploy/overlays/kind && kubectl apply -f platform/tenancy/networkpolicies.yaml
bash course/P3-deployment-and-infra/P3.8-multi-tenancy-and-security/exercises/np_probe.sh
```

## What you should see

- pytest: rotation, audit and noisy-neighbor tests pass. The audit verifier names the exact line that was edited, deleted or reordered.
- `np_probe.sh`: `gateway → mockllm: allowed`, `stray pod → mockllm: blocked`, `mockllm → 169.254.169.254: blocked`. If everything shows **allowed**, your CNI doesn't enforce NetworkPolicy.

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [NetworkPolicy tests on kind](exercises/01-netpol.md) | T0 | `test_netpol.py` (static) + `np_probe.sh` (live) |
| 2 | [Key rotation without downtime](exercises/02-rotation.md) | T0 | `test_rotation.py` |
| 3 | [Tamper-evident audit log](exercises/03-audit.md) | T0 | `test_audit.py` (your `verify`) |
| 4 | *(hard)* [Noisy neighbor: one tenant can't break another's SLO](exercises/04-noisy-neighbor.md) | T0 | `test_noisy.py` |

## Common mistakes

- NetworkPolicies on a CNI that ignores them: everything is still allowed, and nobody notices without a probe.
- Forgetting egress. Default-deny ingress only still lets a compromised pod reach IMDS and the internet.
- Keys in a ConfigMap, a Helm values file or a Terraform default.
- One IAM role for every pod.
- Rate limiting by requests only: one tenant sends 32k-token prompts within its rps and starves everyone. Limit tokens too, and per-tenant concurrency.
- An audit log the service can rewrite, with no external checkpoint.

## Go deeper

- Kubernetes docs: NetworkPolicy, Pod Security Standards, RuntimeClass (UNVERIFIED).
- AWS docs: Secrets Manager, IAM Roles for Service Accounts / EKS Pod Identity, VPC CNI network policies.
- gVisor and Firecracker design docs.
- Modular handbook: *InferenceOps and management*.

**Next:** [P3.9 Second cloud](../P3.9-second-cloud/README.md).
