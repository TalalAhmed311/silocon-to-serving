# R.7 — A safe cloud account

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) to read and do the exercises. Creating the account is free; **you must finish it before P2.1** |
| **Time** | ≈30 min reading + ≈2 h hands-on (account setup, plus waiting for a quota decision) |
| **Prerequisites** | [R.3](../R.3-terminal-git-build/README.md) (terminal, environment variables), [R.6](../R.6-http-and-networking/README.md) (ports and localhost). Or pass the self-checks in [Start here](../README.md) |
| **You will build** | a cost calculator, a tfvars secrets check, and a ticked account checklist |

Already comfortable with AWS accounts, budgets and Terraform? Answer the lesson's checkpoints. If you get them all right, skip to [P0.1](../../P0-systems-primer/P0.1-c-cpp-for-systems/README.md), and keep the [account checklist](exercises/02-account-checklist/README.md) for P2.1.

## Learning objectives

By the end of this module you can:

1. Estimate the cost of a GPU session, and of forgetting to stop it, with `hours × hourly price`.
2. Set up an AWS account so that the root user is locked and daily work uses an SSO identity.
3. Choose a region, and request the GPU quota that new accounts do not have.
4. Create a budget alarm, and name what still costs money when an instance is "off".
5. Reach an instance with Session Manager instead of an open SSH port.
6. Run the course's stack exactly as [P2.1 on AWS](../../P2-serving-engines/P2.1-first-serve-with-vllm/aws.md) does: `make up`, `make ssm`, `make down`, then prove nothing is left.
7. Keep secrets out of the repo.

## Why this matters

From P2.1 on, the course runs real GPU servers on rented machines. A rented GPU bills every hour it exists, whether or not you use it. The usual way to lose money is not a big experiment. It is one forgotten instance over a weekend. Everything here exists to make that hard.

---

## 1. The bill is the risk

Buying a data-centre GPU costs a great deal and needs a server to hold it. Renting one by the hour costs nothing until you start it. The trade is that the meter runs.

```
cost = hours running × hourly price           (per instance)
```

The price depends on the instance type and the region. **This repo lists no prices on purpose.** Look up the on-demand price for your instance type in your region on the EC2 pricing page and write it in your notes. In this module, every price is an "example value, check the current price page".

A 3-hour session is `3 × price`. A forgotten instance is `24 × days × price`. A weekend of forgetting is 48 hours, which can be ten times the whole session. Exercise 1 makes you compute both.

## 2. Root user and everyday identity

When you create an AWS account you get a **root user**: an email and password that can do anything, including closing the account. Treat it like a master key.

1. Sign in as root once. Add **MFA** (multi-factor authentication, usually an authenticator app). Make sure root has **no access keys**.
2. Turn on **IAM Identity Center** and create a user and a group for yourself with only the permissions the course needs.
3. From then on sign in to the AWS access portal with that user. The CLI uses short-lived credentials from it (`aws configure sso`).

Short-lived credentials matter: a leaked key that expires in hours is much less harmful than a key that works forever.

## 3. Regions, zones and GPU quotas

A **region** is a geographic cluster of data centres (for example `us-east-1`). It has several **availability zones** (separate buildings). Prices, instance types and quotas differ by region. Pick one region, write it down, and use it for the whole course.

New accounts usually have a **quota of 0 vCPUs** for GPU instance families ("Running On-Demand G and VT instances"). Your first `make up` fails with a quota error until you ask for more. A `g6.xlarge` has 4 vCPUs, so request at least 4 (8 for headroom). Approval can take minutes or days, so **request it on day one**.

## 4. Budgets and alarms

An **AWS Budget** watches your spend and emails you at thresholds you choose. It is an alarm, not a brake: AWS does not stop your instance when you pass it. The repo's [`infra/aws/guardrails`](../../../infra/aws/guardrails/main.tf) stack creates a monthly budget that emails at 80% of the forecast and at 100% of actual spend. The single-node stack also creates one. Billing data lags by hours, so an alarm can arrive late. That is why the course adds an idle auto-stop (step 8).

## 5. What costs money when "off"

| State | Still billed |
|---|---|
| running | the instance, its disk, any public IP |
| **stopped** | **the disk (EBS)** and any snapshots; the instance itself is not billed |
| terminated (`make down`) | nothing from the stack, if everything was in the stack |
| any state | **NAT gateways, Elastic IPs that are allocated, load balancers, snapshots you took by hand** |

"Stop" is a pause, not a delete. Only `make down` removes the disk. The course's `make stop` keeps your data and your disk cost.

## 6. Access without an open port

The classic way in is SSH on port 22, open to the internet and guarded by a key. Bots scan for open port 22 all day. **Session Manager (SSM)** works the other way round: an agent on the instance opens an outbound connection to AWS, and you get a shell through AWS after you sign in. The course's security group has **no inbound rules**, so there is nothing to scan. Servers on the instance bind to `127.0.0.1`, and `make forward` tunnels `localhost:8000` on your laptop to that port.

## 7. Terraform in one step

**Terraform** lets you describe cloud resources in files. You run it in three verbs:

- `terraform plan`: show what would change (a diff, nothing happens);
- `terraform apply`: make the cloud match the files;
- `terraform destroy`: delete everything the files created.

Terraform remembers what it made in a **state** file. `terraform state list` shows the resources it still owns. [P3.2](../../P3-deployment-and-infra/P3.2-terraform-on-aws/README.md) teaches it properly. Here you only run a given stack.

## 8. The course's stack lifecycle

Every lab is the same loop (from [`infra/aws/single-node/README.md`](../../../infra/aws/single-node/README.md)):

```bash
cd infra/aws/single-node
make up         # terraform apply: instance, disk, role, budget, idle alarm
make ssm        # shell on the instance
make forward    # in a second terminal: localhost:8000 -> the instance
make down       # terraform destroy: delete everything
```

The cost meter runs only from `make up` to `make down`. Two guards catch a forgotten session: an **idle watchdog** stops the instance after 30 idle minutes (GPU under 5% and no SSM session), and a CloudWatch alarm stops it after 3 hours of near-zero CPU. Stopping is not deleting: the disk is still billed until `make down`.

## 9. Teardown checklist

After every session:

1. `make down` finished with `Destroy complete`.
2. `terraform state list` prints nothing.
3. `../scripts/cost.sh` shows no running instance tagged `Project=silicon-to-serving`.
4. The console's EC2 pages (instances, volumes, snapshots, Elastic IPs) are empty in your region.
5. The billing page shows no growing line you do not recognise. Billing lags, so check again the next day.

## 10. Secrets

- Never commit keys, tokens or passwords. Not in code, not in `terraform.tfvars`, not in a test.
- Use SSO (`aws configure sso`, `aws sso login`) rather than long-lived keys. If you must use environment variables, set them in your shell for the session only.
- Keep `terraform.tfvars` and state files in `.gitignore`. Commit only `terraform.tfvars.example`.
- Pass the Hugging Face token through SSM Parameter Store, as the single-node README does, never through a file in the repo.

If a key leaks, deactivate it in the console first, then clean the repo history.

---

## Exercises

| # | Exercise | Difficulty | Check |
|---|---|---|---|
| 1 | [Cost calculator](exercises/01-cost-calc/README.md) | easy | `uv run pytest course/R-readiness/R.7-cloud-account-and-cost-safety/exercises/01-cost-calc` |
| 2 | [Account checklist](exercises/02-account-checklist/README.md) | do before P2.1 | tick every box with evidence |
| 3 | [tfvars hygiene](exercises/03-tfvars-hygiene/README.md) | medium | `uv run pytest course/R-readiness/R.7-cloud-account-and-cost-safety/exercises/03-tfvars-hygiene` |

Verify all reference solutions: `S2S_SOLUTIONS=1 uv run pytest course/R-readiness/R.7-cloud-account-and-cost-safety`.

Every real cloud action here is `TODO(run)`: this build environment has no AWS access.

## Quiz

[quiz.md](quiz.md)

## Go further

All links are UNVERIFIED from the build environment (egress blocked). Check them.

- [AWS Budgets user guide](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html)
- [IAM Identity Center user guide](https://docs.aws.amazon.com/singlesignon/latest/userguide/what-is.html)
- [Systems Manager Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html)
- [HashiCorp: Get started with Terraform on AWS](https://developer.hashicorp.com/terraform/tutorials/aws-get-started)

## Next

Continue to [P0.1](../../P0-systems-primer/P0.1-c-cpp-for-systems/README.md). Do the account checklist before [P2.1](../../P2-serving-engines/P2.1-first-serve-with-vllm/aws.md).
