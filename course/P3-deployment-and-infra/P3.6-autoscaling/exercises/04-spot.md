# Exercise 4 (hard): spot interruption with zero failed requests

## Part A: simulation (T0)

`test_spot_sim.py::test_drain_then_kill` reclaims one of two backends mid-run with a **graceful** shutdown (stop accepting, finish in-flight). The gateway retries refused connections on the survivor. It passes as shipped: read it and understand why.

Now write `test_hard_kill_streams` (remove the `xfail`):

1. Use **streaming** requests (`"stream": true`) and kill the spot backend **without** draining. Hint: run the mock in a subprocess (`uvicorn mockllm.server:app --app-dir platform --port …`) and `SIGKILL` it.
2. Measure how many requests fail. They are exactly the streams that had started on that backend: the gateway must not retry after the first byte (P2.7).
3. Add a mitigation and show the failure count drops to the budget you state in the test docstring. Options: the gateway emits a final SSE error event the client can resume from, or the client retries with the partial output as prompt continuation, or spot replicas cap `max_tokens` so streams are short.

## Part B: the real thing (T3, optional)

On EKS with Karpenter and the two NodePools, use AWS FIS to send a spot interruption to the GPU node during a #4 ramp. Pass: the #4 summary shows zero non-2xx responses for non-streaming traffic, and the replacement node came from the **on-demand** pool if spot capacity was exhausted. Check with `kubectl get nodes -L karpenter.sh/capacity-type`. `TODO(run-on: EKS + Karpenter)`.
