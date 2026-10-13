# Exercise 2 — Account checklist (T0, no test)

Work through this list once, in order, before [P2.1 on AWS](../../../../P2-serving-engines/P2.1-first-serve-with-vllm/aws.md). Tick each box only when you have seen the evidence. Keep the evidence (a screenshot or the command output, with account ids blurred) in your lab notes. Never paste keys or tokens there.

## The account

- [ ] **Root user has MFA.** Sign in as root once, add an authenticator-app MFA device, then sign out. Evidence: the console's security page lists an MFA device for the root user.
- [ ] **Root has no access keys.** Evidence: the root user's security page shows no access keys.
- [ ] **An everyday identity exists.** An IAM Identity Center user in a group with only the permissions you need for this course. You never sign in as root again. Evidence: you can open the AWS access portal with this user.
- [ ] **The identity user has MFA too.**
- [ ] **A region is chosen and written down.** One region for the whole course, close to you, where the instance type you want is offered. Evidence: the region name in your lab notes.

## Money

- [ ] **A budget with an email alert exists.** Either created by hand (AWS Budgets) or by applying `infra/aws/guardrails`. Evidence: the budget is listed, and you received or can see the subscribed email address.
- [ ] **You wrote down the hourly price.** Look up the on-demand price of your instance type in your region on the EC2 pricing page. Evidence: `____ $/h` in your lab notes, with the date. Do not copy a price from this repo; there isn't one.
- [ ] **You ran exercise 1 with that price.** The cost of a 4-hour session and of forgetting the instance for a weekend is in your notes.

## GPU quota

- [ ] **A quota increase is requested.** Service Quotas, Amazon EC2, "Running On-Demand G and VT instances", at least 4 vCPUs (a `g6.xlarge` has 4). Evidence: the request is listed. Approval can take minutes to days, so do this first.
- [ ] **The request is approved.** Evidence: the applied quota value is 4 or more.

## Your laptop

- [ ] **AWS CLI v2 installed.** `aws --version` prints a version.
- [ ] **Session Manager plugin installed.** `session-manager-plugin` prints a version or help text.
- [ ] **SSO login works.** `aws configure sso`, then `aws sso login --profile <your-profile>`.
- [ ] **`aws sts get-caller-identity --profile <your-profile>` works.** It must show your SSO identity (a role session), not root and not an IAM user with long-term keys.
- [ ] **Terraform installed.** `terraform version` prints a version.
- [ ] **`terraform.tfvars` is git-ignored** (exercise 3) and holds no secrets.

## The repo's guardrails

- [ ] **You read `infra/aws/single-node/README.md`.** You can say what the idle watchdog does and what it does not delete.
- [ ] **`make check` passes** in `infra/aws/single-node` (offline checks: formatting, validation, no `ingress` block).
- [ ] **`infra/aws/guardrails` is applied**, or you created an equivalent budget by hand. (The single-node stack also creates its own budget when you run `make up`.)

## Dry run (no GPU launched)

- [ ] `make up` is **not** run yet. Instead run `terraform plan -var-file=terraform.tfvars` in `infra/aws/single-node` and read the plan: you can find the instance, the disk, the IAM role, and you can confirm the security group has no inbound rules. `TODO(run)`
- [ ] You know the three commands for the day: `make up`, `make ssm`, `make down`, and you have set a phone alarm for the end of your session.

When every box is ticked, start P2.1.
