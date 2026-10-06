# Exercise 2 — The S3 gateway endpoint (T0 + T3)

Without a gateway endpoint, every S3 GET from a private-subnet node goes out through the NAT gateway, which bills per GB processed. A 16 GB model pulled by 10 nodes is 160 GB through the NAT.

1. **T0:** read `aws_vpc_endpoint.s3` in `infra/aws/eks/main.tf`. Run `terraform plan` (it needs credentials, or read the code) and confirm the endpoint is attached to the **private** route tables.
2. **T3:** with the cluster up, pull a 1 GB object from S3 in a pod. Check the NAT gateway's `BytesOutToDestination` CloudWatch metric before and after. Then temporarily remove the endpoint (`terraform apply -target`), repeat, and record both. Put the endpoint back.
3. Write the per-GB arithmetic for your region in your notes.
