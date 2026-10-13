# R.7 quiz

<details><summary><b>1.</b> A GPU instance has an example price of 2.00 per hour (made-up). You run a 3-hour lab and then forget it until Monday, 60 hours after you started. What does the weekend cost beyond the lab?</summary>

**114.** The instance ran 60 hours, so the bill is `60 × 2.00 = 120`. The lab itself was `3 × 2.00 = 6`, so the forgotten time cost `120 − 6 = 114`. The price is an example; the formula `hours × price` is what to keep.
</details>

<details><summary><b>2.</b> Why create an IAM Identity Center user instead of using the root user every day?</summary>

Root can do anything, including closing the account, and cannot be limited. If its password or session leaks, everything is lost. An Identity Center user has only the permissions you grant and gets short-lived credentials, so a leak is smaller and expires. Keep root locked behind MFA and use it almost never.
</details>

<details><summary><b>3.</b> Your first <code>make up</code> fails with a vCPU limit error on a brand-new account. What happened and what do you do?</summary>

New accounts often start with a quota of 0 vCPUs for GPU (G and VT) instances. Request an increase of at least 4 vCPUs in Service Quotas for your region and wait for approval. Do this before the lab day.
</details>

<details><summary><b>4.</b> You stop an instance with <code>make stop</code> and go home. What do you still pay for?</summary>

The disk (EBS volume) and any snapshots, because stopping only pauses the instance. Other resources that bill while idle are allocated Elastic IPs, NAT gateways and load balancers. <code>make down</code> removes the disk.
</details>

<details><summary><b>5.</b> An AWS Budget with an email alert is set to 100. Your spend reaches 101. Does AWS stop your instances?</summary>

No. A budget is an alarm: it emails you. It does not stop anything, and billing data arrives with a delay. That is why the course adds an idle watchdog that stops the instance on its own, and why you tear down after every session.
</details>

<details><summary><b>6.</b> Why does the course use Session Manager and keep port 22 closed?</summary>

With SSM the instance opens an outbound connection to AWS and you sign in through AWS, so the security group needs no inbound rules. A closed port cannot be scanned or brute-forced. Servers also bind to <code>127.0.0.1</code> and are reached through a tunnel, so no inference endpoint is public.
</details>

<details><summary><b>7.</b> What do <code>terraform plan</code>, <code>apply</code> and <code>destroy</code> do, and what is the state?</summary>

<code>plan</code> shows the difference between your files and reality without changing anything. <code>apply</code> makes the cloud match the files. <code>destroy</code> deletes everything Terraform created. The state is Terraform's record of the resources it owns; <code>terraform state list</code> empty after a destroy is evidence of a clean teardown.
</details>

<details><summary><b>8.</b> Your <code>terraform.tfvars</code> contains <code>db_password = "..."</code> and the file is not in <code>.gitignore</code>. Name two fixes.</summary>

Keep secrets out of the file (use SSM Parameter Store or a secrets manager and pass them at runtime), and add <code>terraform.tfvars</code> to <code>.gitignore</code>, committing only <code>terraform.tfvars.example</code>. If it was already committed, rotate the secret: deleting the file does not remove it from git history.
</details>
