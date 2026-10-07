# P3.7: Weight storage and lazy loading (+ #5)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the manifest, the verifier, the LRU cache and the local loader bench (tiny model from P0.5). ![T2](https://img.shields.io/badge/tier-T2%20single%20GPU-orange) / ![T3](https://img.shields.io/badge/tier-T3%20AWS%20cluster-red) for S3 throughput, streaming loaders and to-GPU |
| **Time** | ≈30 min reading + ≈8 h hands-on |
| **Prerequisites** | P0.2 (page cache, mmap, pinned memory), P3.6 (cold-start phases) |
| **You will build** | **#5**: a versioned weight registry with checksummed manifests, a node-local NVMe LRU cache, and a loader benchmark |

## Learning objectives

1. Trace where load time goes: **S3 → network → local disk → page cache → host memory → GPU**, and name the bottleneck at each hop.
2. Read a sharded safetensors checkpoint (`model.safetensors.index.json` → shards), and explain why safetensors can be mmapped and loaded lazily.
3. Use a streaming loader (Run:ai Model Streamer via vLLM `--load-format runai_streamer`) and compare it with download-then-load.
4. Build **#5**: immutable versions, manifests with checksums, a size-bounded LRU cache with atomic installs.

---

## 1. The hops

| Hop | Bound by | How to see it |
|---|---|---|
| S3 → instance | per-connection throughput × concurrency, up to the instance's network bandwidth | `bench.py --s3 … --concurrency 1 4 16 64` |
| network → local disk | NVMe write (instance store) vs EBS gp3 provisioned throughput | `iostat -x 1` during a download |
| disk → page cache | sequential read speed; **cold** vs **warm** | `bench.py` read-cold vs read-warm |
| page cache → host tensors | memcpy, or nothing with mmap | safetensors vs mmap rows |
| host → GPU | PCIe; pinned vs pageable (P0.2) | `--gpu` row |

The slowest hop sets the speed. That's usually S3 at low concurrency, or EBS gp3 at its default throughput. A single stream from S3 is far below the instance's network bandwidth, so parallel ranged GETs are the first lever.

> **Predict first.** For your instance (look up its network bandwidth and the gp3 throughput you provisioned), which hop limits a 16 GB model load? Write down the expected seconds for: S3 at 1 stream, S3 at 16 streams, gp3 read, page cache read. Then run the bench.

## 2. safetensors in one paragraph

A safetensors file is an 8-byte header length, a JSON header (`name → dtype, shape, data_offsets`), then raw bytes. No pickle, so loading runs no code. Offsets are explicit, so you can **mmap** the file and hand out tensors that point into it. Pages are read only when touched (P0.2 page faults). Big models are split into shards, and `model.safetensors.index.json`'s `weight_map` says which shard holds each tensor. `manifest.safetensors_shards()` reads it.

## 3. Streaming loaders

Download-then-load runs the hops **sequentially**: the whole model reaches disk, then gets read back. A streaming loader overlaps them. It reads byte ranges from S3 with many concurrent requests straight into host buffers, then copies to the GPU, without touching the disk. vLLM integrates the Run:ai Model Streamer:

```bash
vllm serve s3://bucket/models/llama3-8b/REV --load-format runai_streamer \
  --model-loader-extra-config '{"concurrency": 16}'
```

The flag names and config keys come from vLLM's `docs/models/extensions/runai_model_streamer.md` at the pinned SHA (SOURCES.md). Re-check them if you change versions. Exercise 4 compares the two approaches.

## 4. #5: the registry and the cache

```
s3://<registry>/models/<model>/<version>/manifest.json    ← {total_bytes, files: [{path, size, sha256}]}
s3://<registry>/models/<model>/<version>/*.safetensors …
```

- **Immutable versions.** Never overwrite a version: a rollout points replicas at a new version, and a rollback points them back. Bucket versioning plus a deny on `s3:PutObject` to existing versions enforces it.
- **Checksums.** Verify on install, not on every load. A half-downloaded shard otherwise fails as a confusing CUDA or shape error an hour later.
- **Cache.** `platform/weights/cache.py`: fetch into a temp dir, verify, **atomic rename**. Readers never see a partial copy, and a crash leaves only garbage in `.tmp/`. LRU eviction never removes a pinned version, meaning one a running replica uses.
- **Prefetch.** `daemonset-prefetch.yaml` warms each GPU node's cache, so a P3.6 scale-up skips the S3 hop entirely.

---

## Walkthrough

```bash
uv run pytest course/P3-deployment-and-infra/P3.7-weight-delivery/exercises
uv run python platform/engine/v0/tools/make_tiny_llama.py /tmp/tiny        # P0.5's tiny model
PYTHONPATH=platform uv run python -m weights.manifest build /tmp/tiny --model tiny --version v1
PYTHONPATH=platform uv run python -m weights.manifest verify /tmp/tiny
PYTHONPATH=platform uv run python -m weights.bench --dir /tmp/tiny
```

The T2/T3 path is in [aws.md](aws.md).

## What you should see

- `verify` prints `OK`. Flip one byte in a shard and it reports `sha256: <shard>`. Truncate one and it reports `size:`.
- Tiny-model bench (T0): every row finishes in milliseconds. The **ratio** of read-warm to safetensors to mmap is the point, not the absolute numbers.
- On `g6.xlarge` with an 8B model (`TODO(run-on: g6.xlarge)`): S3 GB/s grows with concurrency until it flattens at a plateau set by the instance network or the disk. The streamer row beats download-then-load in total time-to-ready.

## Bench

| method | model GB | load s | effective GB/s |
|---|---|---|---|
| read-cold / read-warm / safetensors / mmap | | | `TODO(run-on: g6.xlarge)` |
| s3-download c=1 / 4 / 16 / 64 | | | `TODO(run-on: g6.xlarge)` |
| vLLM default (download, then load) | | | `TODO(run-on: g6.xlarge)` |
| vLLM `runai_streamer` | | | `TODO(run-on: g6.xlarge)` |

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [Manifest + checksum verifier](exercises/01-verifier.md) | T0 | `test_verifier.py` |
| 2 | [LRU NVMe cache with a size limit](exercises/02-lru.md) | T0 | `test_lru.py` (your `plan_evictions`) + `test_cache.py` (the whole cache) |
| 3 | [S3 throughput vs concurrency](exercises/03-s3-curve.md) | T2/T3 | `check_s3_curve.py` on your results JSON |
| 4 | *(hard)* [Stream shards to the GPU vs download-then-load](exercises/04-stream.md) | T2 | your bench row + the comparison table |

## Common mistakes

- Benchmarking "disk" reads that are really page-cache hits. Drop caches (root) or use a file larger than RAM.
- One S3 stream, then concluding "S3 is slow".
- Overwriting a version in place: two replicas end up with different weights under the same name.
- Verifying with a size check only.
- `hostPath` caches without a size limit: the node's disk fills, and the kubelet starts evicting pods.

## Go deeper

- `huggingface/safetensors` README (format) · `run-ai/runai-model-streamer` docs · vLLM model loading docs at the pinned SHA.
- P0.2 for mmap, page faults and pinned memory.
- Silicon to Scale ch. 15 (deployment and scaling). Modular handbook *Fast scaling*.

**Next:** [P3.8 Multi-tenancy and security](../P3.8-multi-tenancy-and-security/README.md).
