# tenancy: #8, the secure multi-tenant layer (stretch)

| File | What |
|---|---|
| `keys.py` | key generation (prefixed, 256-bit), hash-only storage, two-phase rotation using the gateway's `previous_key_sha256` |
| `audit.py` | append-only JSONL audit log with an HMAC hash chain, plus a verifier that detects edits, deletions, reordering and (with a checkpoint) truncation |
| `networkpolicies.yaml` | default-deny for `s2s`, then DNS, ingress → gateway → backends, Prometheus scrape, and HTTPS-only backend egress with IMDS blocked |
| `seccomp-inference.json` | a deny-list seccomp profile for inference pods. The README explains when you need gVisor or Firecracker instead |

Built in [P3.8](../../course/P3-deployment-and-infra/P3.8-multi-tenancy-and-security/README.md).
