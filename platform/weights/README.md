# weights: #5, model weight delivery

| File | What |
|---|---|
| `manifest.py` | the registry contract: `<root>/<model>/<version>/manifest.json` with size and sha256 for each file. Includes `build` and `verify` |
| `cache.py` | node-local LRU cache: fetch to a temp dir, verify, atomic rename; size limit; pinned versions never evicted. `DirStore` (T0) and `S3Store` |
| `bench.py` | `method | model GB | load s | effective GB/s` for cold/warm reads, safetensors, mmap, S3 at several concurrencies, and to-GPU |
| `daemonset-prefetch.yaml` | warms the cache on every GPU node, so a scale-up skips the S3 download |

Built in [P3.7](../../course/P3-deployment-and-infra/P3.7-weight-delivery/README.md).
