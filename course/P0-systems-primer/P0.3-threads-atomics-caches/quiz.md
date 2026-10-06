# P0.3 quiz

<details><summary><b>1.</b> Name two things threads of one process share and two things they don't.</summary>

**Shared:** the address space (heap, globals, mmap'd files) and file descriptors. **Not shared:** each thread has its own stack and its own registers (including the instruction pointer). Thread-local storage is per thread.
</details>

<details><summary><b>2.</b> Why is <code>counter++</code> on a shared <code>int</code> wrong even on a single-socket laptop?</summary>

It is a load, an add and a store. Two threads can both load the same value and both store value+1, losing an update. In C++ that is a data race and therefore undefined behavior.
</details>

<details><summary><b>3.</b> What does a release store + acquire load pair guarantee?</summary>

If the acquire load reads the value written by the release store, then every write the releasing thread made *before* the store is visible to the acquiring thread *after* the load.
</details>

<details><summary><b>4.</b> Two threads increment <i>different</i> variables and still slow each other down. Why?</summary>

**False sharing.** The variables share a 64-byte cache line. Each write needs exclusive ownership of the line, so it invalidates the other core's copy and the line ping-pongs between cores. Pad each variable to its own line.
</details>

<details><summary><b>5.</b> Why must <code>cv.wait</code> be given a predicate?</summary>

Spurious wakeups, and notifications that happen before the wait starts (lost wakeups). The predicate re-checks the real condition under the lock.
</details>

<details><summary><b>6.</b> Why does a thread pool run the task <i>after</i> releasing the queue lock?</summary>

Otherwise one long task would block every other worker from dequeuing, which serializes the pool. Running user code under the lock also risks deadlock if the task itself submits work.
</details>

<details><summary><b>7.</b> In D2, why must <code>allocate_many</code> check the free count under the same lock as the pops?</summary>

Otherwise another thread could take blocks between the check and the pops, and the request would end half-satisfied, breaking the all-or-nothing guarantee. A scheduler relies on that guarantee to admit a request only if its whole KV need fits.
</details>

<details><summary><b>8.</b> What is copy-on-write in a paged KV cache, and when does it happen?</summary>

Two sequences share a block (for example a common prefix, or beam/parallel sampling). When one of them must append to or modify that block, it gets a fresh block with a copy of the data, and the shared block's ref count drops by one. If it is the sole owner, it writes in place.
</details>

<details><summary><b>9.</b> On Linux, which NUMA node does a page of a <code>malloc</code>'d buffer live on?</summary>

By default, the node of the CPU that first *touched* (wrote) the page. Initialization order decides placement.
</details>
