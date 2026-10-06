# P0.2 quiz

<details><summary><b>1.</b> With 4 KiB pages and 4-level paging, how many memory reads does a TLB miss cost in the worst case, before the actual load?</summary>

**Four**, one per level (PML4, PDPT, PD, PT). The page-walk caches and the ordinary L1/L2 often serve some of them, which is why the average is better than the worst case.
</details>

<details><summary><b>2.</b> Why does <code>mmap()</code> of a 16 GB file return in microseconds?</summary>

It only creates virtual mappings, marked not-present. No data is read until each page is touched and faults.
</details>

<details><summary><b>3.</b> Minor vs major page fault?</summary>

**Minor:** the page is already in RAM (for example in the page cache), and the kernel only installs the mapping. **Major:** the page must be read from storage first.
</details>

<details><summary><b>4.</b> Two processes <code>mmap</code> the same model file read-only. How many copies of the weights are in RAM?</summary>

**One.** Both map the same page-cache pages. With `read()` into private buffers there would be two, plus the page cache.
</details>

<details><summary><b>5.</b> What is TLB reach, and why do huge pages raise it 512×?</summary>

Reach = entries × page size. A 2 MiB page covers 512 4 KiB pages with one entry.
</details>

<details><summary><b>6.</b> Why does <code>cudaMemcpyAsync</code> from pageable memory not overlap with compute the way it does from pinned memory?</summary>

The DMA engine needs physical pages that won't move. For pageable memory, the driver must first copy the data into a pinned staging buffer with the CPU, which makes the transfer effectively synchronous. Pinned memory can be DMA'd directly.
</details>

<details><summary><b>7.</b> A safetensors header says a tensor's <code>data_offsets</code> are <code>[0, 1 &lt;&lt; 40]</code> in a 1 MB file. What should a loader do?</summary>

Reject the file. Never form a pointer from untrusted offsets without checking `end <= data_size` and that `end - begin` equals `numel × dtype_size`.
</details>

<details><summary><b>8.</b> Why time "mmap + touch" instead of just "mmap"?</summary>

Because the cost moves from the `mmap` call to the page faults at first touch. Timing `mmap` alone measures almost nothing.
</details>
