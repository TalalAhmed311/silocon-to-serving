# Exercise 1 — Static checks (T0)

1. Install `terraform` ≥ 1.6, `tflint` and `checkov`.
2. Run `bash tools/ci_terraform.sh`. Fix any input names `terraform validate` rejects in `infra/aws/eks/main.tf`, using the module README at v21.26.0 as the reference. Commit the fixes; this is a real contribution to the course.
3. Run `tflint --recursive infra/` and `checkov -d infra/`. For each checkov finding, fix it **or** write one line explaining why it is acceptable for a lab (for example: "single NAT gateway: cost over HA"). Keep that list in `infra/aws/CHECKOV.md`.
