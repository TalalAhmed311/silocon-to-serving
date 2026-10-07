# Exercise 4 (hard): stream shards to the GPU vs download-then-load (T2)

Two ways to get an 8B model from S3 into GPU memory:

- **A. download-then-load:** `WeightCache.get()` to NVMe, then `vllm serve <dir>` (or `bench.py --gpu` on the dir).
- **B. stream:** `vllm serve s3://… --load-format runai_streamer` (aws.md has the commands). Optionally, write your own minimal streamer: concurrent ranged GETs of each shard into pinned host buffers, parse the safetensors header from the first bytes, and `cudaMemcpyAsync` each tensor as its byte range lands. Use `torch.frombuffer` on the pinned buffer, then `.to("cuda", non_blocking=True)`.

Fill in:

| method | model GB | time to weights-on-GPU (s) | time to first token (s) | peak host RAM | local disk used |
|---|---|---|---|---|---|
| A cold (empty cache) | | | | | |
| A warm (cache hit) | | | | | |
| B streamer | | | | | 0 |
| B your streamer (optional) | | | | | 0 |

`TODO(run-on: g6.xlarge)`

**Questions:** When does A-warm beat B? Think repeated restarts on the same node, which is P3.6's scale-up case. What does B cost in S3 request charges per load at your concurrency and chunk size? Which would you choose for scale-from-zero, and which for a rolling update?
