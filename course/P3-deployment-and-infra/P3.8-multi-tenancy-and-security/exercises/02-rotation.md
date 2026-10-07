# Exercise 2: key rotation without downtime (T0)

`test_rotation.py` runs the gateway with a tenant mid-rotation: `key_sha256 = H(new)`, `previous_key_sha256 = [H(old)]`.

1. Make the tests pass. They need `platform/tenancy/keys.py` (provided) and the gateway's `previous_key_sha256` support (provided). Read both.
2. **Your task:** `test_no_failed_requests_during_rollout` is marked xfail. It simulates a rolling update with two gateway instances, A (old config) and B (rotated config), behind a client that alternates between them. The client switches from the old key to the new one halfway. Write the assertion that no request fails, then remove the xfail.
3. **Your task:** the gateway can't tell you when the old key is unused. Add a `key_generation` label (`current` / `previous`) to `gateway_requests_total`, then write the PromQL that proves it's safe to finish the rotation.

On EKS the same thing is a `kubectl apply -k` of the updated ConfigMap, then `kubectl rollout restart deploy/gateway` with `maxUnavailable: 0`.
