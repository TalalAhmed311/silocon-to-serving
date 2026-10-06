# Exercise 1: NetworkPolicy tests on kind (T0)

1. Read `platform/tenancy/networkpolicies.yaml`. `test_netpol.py` checks it statically: default-deny in both directions, backends reachable only from the gateway and Prometheus, and backend egress limited to 443 with IMDS and private ranges excluded.
2. Apply it to your kind cluster and run `np_probe.sh`. Every line must match its expectation.
3. **Break it on purpose.** Delete the `backends` policy and re-run the probe. Which lines flip? Now delete `default-deny` instead. Restore both.
4. Add a rule so Prometheus (namespace `monitoring`) can scrape the **gateway** on 9000. Write the static test first, then the policy, then confirm with a probe from a pod in `monitoring`.

If the probe says `allowed` everywhere, your CNI isn't enforcing policies. Check your kind version, or install Calico.
