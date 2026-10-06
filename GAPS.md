# GAPS

This file lists everything marked `UNVERIFIED` or `TODO(run-on: …)`, with the exact step that closes it. It is updated after every phase.

## A. Hosts blocked from the Stage 1 build environment

The build sandbox's egress policy denied these hosts. Re-run `python tools/check_links.py --external` from a machine with normal internet access to close them.

| Host / source | Why it matters | How to close |
|---|---|---|
| leetgpu.com | per-problem URLs in all six `leetgpu-map.md` files | open https://leetgpu.com/challenges in a browser; confirm the titles match `AlphaGPU/leetgpu-challenges@37a1253` and record the URL pattern; then regenerate the maps |
| www.inference-engineering.xyz | anchor resource (P1 foundations, P2 engines, P3 fleet/Dynamo) | open it in a JS-capable browser; record the course/lesson index and pin the access date; until then no lesson depends on it |
| handbook.modular.com | anchor; content verified through the repo, but the site URLs are not | spot-check 5 page URLs and the interactives listed in SOURCES.md §2.1 |
| vvinjamu.github.io | anchor; verified through the repo | open the index and the diagrams page once |
| arxiv.org, usenix.org | 15 paper citations | open each ID in SOURCES.md §4 and confirm title/authors |
| docs.nvidia.com, nvidia.com | **whitepaper numbers for every roofline** (P1.4) | download the A100, H100, Ada (L4/L40S) and Blackwell whitepapers and the T4/A10G datasheets; record page numbers in `gpu_specs.yaml` |
| docs.aws.amazon.com, aws.amazon.com | instance specs and **prices** | check each instance in §5 (`g4dn.xlarge`, `g5.xlarge`, `g6.xlarge`, `g6e`, a 4×L4 `g6.12xlarge`-class instance, `p4d.24xlarge`, `p5`) in the learner's region |
| docs.vllm.ai, docs.sglang.ai, docs.ray.io, keda.sh, karpenter.sh, triton-lang.org | rendered docs | the repo `docs/` folders at the pinned SHAs were used instead; spot-check the rendered pages |
| kipp.ly, siboehm.com, horace.io, en.algorithmica.org, huggingface.co/spaces, jax-ml.github.io, sre.google, opentelemetry.io, prometheus.io, grafana.com, pytorch.org, intel.com, arm.com, agner.org, brendangregg.com, learncpp.com, csapp.cs.cmu.edu, pages.cs.wisc.edu, cloud.google.com, modal.com, docs.lambda.ai, docs.coreweave.com | articles, books, docs | open each link once |

## B. Facts to check at build time (not yet needed by any built code)

- Choose the P0.5 tiny Llama-architecture model and the P2 default 7–8B model. Check each license and pin it by HF revision SHA.
- Silicon to Scale's "Inference Fast-Track": the prompt says chapters 01, 06, 08, 09, 11, 13 and 17. The book's front matter lists Path 2 as 6, 11, 8 → 7, 9, 17 → 14, 5. The syllabi follow the book. Confirm this is what you want.
- P3.9 second cloud: the default is GKE. Confirm, or pick Modal, Lambda or CoreWeave.
- P4.3 primary distributed path: Slurm (ParallelCluster) or Ray on EKS. Confirm one.
- `awsome-distributed-training`: its latest tag is a pre-reorg snapshot. Re-check its paths at build time of P4.

## C. Runs that need hardware (`TODO(run-on: …)`)

None are pending yet, because no module is built. Every T2/T3 benchmark added in Stage 4 is listed here with its exact command.
