# P3.8 quiz

<details><summary><b>1.</b> Why store API keys as SHA-256 hashes, and why is a fast hash fine here when it isn't for passwords?</summary>

A leaked config or DB dump then doesn't leak usable keys. These keys are 256-bit random values, so brute force is infeasible whatever the hash speed. Passwords are low-entropy and need a slow KDF.
</details>

<details><summary><b>2.</b> What does default-deny egress protect against that default-deny ingress doesn't?</summary>

A compromised pod calling out: stealing node credentials from IMDS, exfiltrating data, or reaching other internal services.
</details>

<details><summary><b>3.</b> A hash chain detects edits and deletions. What does it not detect on its own, and how do you fix that?</summary>

Truncation of the tail, or a whole rewrite by someone who holds the key. Fix it by checkpointing the latest hash somewhere the writer can't modify (S3 Object Lock, another account), and keeping the HMAC key out of the writer's reach where you can.
</details>

<details><summary><b>4.</b> Is seccomp a sandbox for running model-generated code?</summary>

No. It narrows the syscall surface, but the code still shares the host kernel. Use gVisor or Firecracker, with no credentials and controlled egress.
</details>

<details><summary><b>5.</b> Two tenants share one vLLM deployment with prefix caching. What leaks, and what are the options?</summary>

Timing: a cache hit is faster, which reveals that someone already sent that prefix. Options are per-tenant cache salting, disabling prefix caching across tenants, or separate deployments.
</details>
