# Exercise 1 — Verify `gpu_specs.yaml` (mandatory)

Every roofline, calculator and "% of peak" figure in the rest of the course reads `gpu_specs.yaml`. The values in it were written from memory and are marked `UNVERIFIED`. Your job:

1. For each GPU, open the document named in `source:`. For the A10G, use AWS's G5 instance page or the A10G product brief.
2. Check every number: SM count, memory, bandwidth, L2, FP32, and **dense** FP16/BF16/FP8/INT8 tensor throughput. Halve the sparse headline figures.
3. Fix whatever is wrong. Fill the `null`s you can source.
4. Set `status: VERIFIED (<document title>, <page or table>, <YYYY-MM-DD>)`.

**Test:** `test_specs.py` checks that the file loads, that every entry has a source, and that every `VERIFIED` entry carries a document, a page and a date. It can't check the numbers themselves. That part is on you, and it is exactly the "verify before you cite" rule.
