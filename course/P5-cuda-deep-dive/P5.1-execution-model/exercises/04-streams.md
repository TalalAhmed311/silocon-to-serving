# Exercise 4 (hard): overlap H2D, compute and D2H with 3 streams (T2)

1. Run `p5.1_05_streams_overlap` and record serial vs pipelined time.
2. `nsys profile -o overlap ./build/p5/p5.1_05_streams_overlap`, then open it in the Nsight Systems UI (copy the `.nsys-rep` back from the instance) and screenshot the timeline: H2D, kernel and D2H of different chunks running at once.
3. Vary the chunk count (2, 4, 8, 16, 32) and plot time vs chunks. Why does it stop improving, and why does it eventually get worse?
4. Replace the pinned buffer with `malloc` memory. What does the timeline show now, and why (04_pinned_vs_pageable)?

Deliverable: the timeline screenshot + the chunk-count table. `TODO(run-on: g4dn.xlarge)`
