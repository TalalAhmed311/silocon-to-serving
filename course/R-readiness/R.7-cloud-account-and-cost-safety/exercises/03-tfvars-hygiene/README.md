# Exercise 3 — tfvars hygiene (medium, T0)

Two helpers that protect you from the most common cloud mistake: committing a secret.

- `check_tfvars(text)` returns one finding per suspicious line: an AWS access key id, a literal `password` or `token`, or any `aws_secret_access_key`. Comments, empty values and harmless values are not findings.
- `gitignore_covers(lines, path)` says whether a `.gitignore` ignores a path. It must treat `terraform.tfvars.example` as **not** ignored, and `terraform.tfvars` as ignored.

The test builds its fake key by concatenating strings, so no real-looking credential sits in the repo. Never paste a real key into a test or a tfvars file.

```bash
uv run pytest course/R-readiness/R.7-cloud-account-and-cost-safety/exercises/03-tfvars-hygiene
```

When it passes, check your real repo. The single-node example file says `terraform.tfvars` is git-ignored, so confirm it:

```bash
git check-ignore -v infra/aws/single-node/terraform.tfvars
```

If that prints nothing, the file is **not** ignored. Add `terraform.tfvars`, `*.tfstate*` and `.terraform/` to your `.gitignore` before you run `make up`, and run `gitignore_covers` over your new lines to prove it. State files can hold sensitive values, so they belong in the same list.
