# Exercise 4 (hard): a fair scheduler across MPS clients

MPS lets kernels from different processes run concurrently, but it doesn't divide SM time fairly: a client with bigger or more kernels takes more.

1. Measure the unfairness: 4 MPS clients, one of them at 4× the request rate. Compute share-weighted Jain from per-tenant tok/s.
2. Fix it with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` per client (MPS docs, UNVERIFIED variable name for your driver). Give each client 25% and re-measure. What does it cost in aggregate throughput?
3. Fix it **above** the GPU instead: route all four tenants through the #6 gateway with per-tenant concurrency limits (P3.8 exercise 4). Compare the two fixes on fairness, p99 ITL and aggregate tok/s.
