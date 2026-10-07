# P5.4 quiz

<details><summary><b>1.</b> Why does sequential addressing (rung 3) beat interleaved (rung 1)?</summary>

The active threads are contiguous, so whole warps retire with no divergence, and `s[tid] + s[tid+s]` hits distinct banks.
</details>

<details><summary><b>2.</b> Why is rung 6 the biggest single step?</summary>

Each thread accumulates many elements in registers with 16-byte loads, so there are enough bytes in flight to saturate DRAM, plus one atomic per block instead of millions of partial writes and a second launch.
</details>

<details><summary><b>3.</b> Why must decoupled look-back assign tile ids with an atomic counter?</summary>

A tile spins waiting for its predecessors. If ids followed `blockIdx`, a waiting resident block could depend on a block that isn't scheduled yet, which deadlocks. Counter ids guarantee every predecessor has already started.
</details>

<details><summary><b>4.</b> Hillis–Steele is O(n log n) work. Why use it at all?</summary>

Inside a warp it's 5 shuffle steps with no shared memory or barriers, and its low depth beats work-efficiency at that scale.
</details>

<details><summary><b>5.</b> Two runs of the atomic reduction differ in the 7th digit. Bug?</summary>

No. Float addition isn't associative, and block completion order varies. Use the deterministic two-pass version if you need reproducibility.
</details>
