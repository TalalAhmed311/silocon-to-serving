# P3.7 quiz

<details><summary><b>1.</b> List the hops from S3 to GPU memory and the usual bottleneck.</summary>

S3 → instance network → local disk → page cache → host tensors → PCIe → GPU. A single S3 stream or an EBS gp3 volume at its default throughput usually limits it, so use many concurrent ranged GETs and NVMe.
</details>

<details><summary><b>2.</b> Why can a safetensors file be mmapped and loaded lazily, but a pickle checkpoint can't?</summary>

safetensors has a JSON header with explicit dtype, shape and byte offsets for each tensor, followed by raw bytes, so a tensor is just a view into the file. Pickle has to execute code to rebuild objects, which is also why it's unsafe.
</details>

<details><summary><b>3.</b> Why install into a temp dir and rename, instead of downloading in place?</summary>

The rename is atomic, so readers see either no version or a complete, verified one, never a partial one. A crash leaves only garbage in `.tmp/`.
</details>

<details><summary><b>4.</b> Why must model versions be immutable?</summary>

Otherwise two replicas can serve different weights under the same name, and rollback becomes impossible. Each upload gets a new version, and deploys point at versions.
</details>

<details><summary><b>5.</b> Your "disk read" benchmark shows 20 GB/s on gp3. What happened?</summary>

You measured the page cache, not the disk. Drop caches (as root), or read a file larger than RAM.
</details>
