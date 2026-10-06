# P3.2 quiz

<details><summary><b>1.</b> Why must Terraform state be remote and locked?</summary>

So everyone applies against the same view of reality, and two applies never run at once and corrupt it. State can also contain secrets, so it doesn't belong in git.
</details>

<details><summary><b>2.</b> Why taint the GPU node group?</summary>

So only pods that explicitly tolerate the taint (GPU workloads) schedule there. Otherwise a stray CPU pod can occupy an expensive GPU node, or keep it from scaling to zero.
</details>

<details><summary><b>3.</b> What does IMDSv2 (<code>http_tokens = required</code>) protect against?</summary>

SSRF-style theft of instance-role credentials. Fetching them needs a PUT-obtained session token, which simple request forgery can't produce.
</details>

<details><summary><b>4.</b> Which costs keep running when the GPU pool is at 0 nodes?</summary>

The EKS control plane, the NAT gateway (hourly plus per GB), the system node, EBS volumes, and any load balancers.
</details>

<details><summary><b>5.</b> Why does <code>make down</code> delete K8s LoadBalancer services before <code>terraform destroy</code>?</summary>

K8s created the ELBs and their ENIs outside Terraform's state. They hold onto subnets and security groups, so the VPC destroy hangs until they're gone.
</details>
