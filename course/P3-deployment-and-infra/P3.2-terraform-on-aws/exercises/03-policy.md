# Exercise 3: policy test, no inbound and no public API (T0)

`test_policy_tf.py` reads the Terraform **source** (no terraform binary, no credentials) and enforces the guardrails every AWS guide in this course promises:

- `infra/aws/single-node` has no `ingress` blocks (access is SSM only)
- IMDSv2 is required, and the root volume is encrypted
- the EKS API is never open to `0.0.0.0/0`, and `endpoint_public_access_cidrs` is set
- GPU nodes carry the `nvidia.com/gpu` taint

1. Run it: `uv run pytest course/P3-deployment-and-infra/P3.2-terraform-on-aws/exercises/test_policy_tf.py`.
2. Break each rule once on purpose, for example by adding an `ingress { from_port = 22 … }` block, and confirm the matching test fails with a readable message. Revert.
3. **Your task:** implement the skipped test at the bottom of the file. Remove its `skip` and make it pass.
4. **Think:** text checks are brittle. A `dynamic "ingress"` block, a module input, or a value from a variable can slip through. What does `terraform show -json` on a **plan** give you that source greps can't? Sketch the same rules as conftest/OPA policies over the plan JSON, and list one rule that only a plan-level check can enforce.
