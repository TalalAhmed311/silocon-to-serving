/* P0.3 — Threads, atomics and caches. From "two threads, one counter" to a thread pool and the KV block allocator (D2). */
(function () {
  const F = S2S.fmt;
  // deterministic pseudo-random numbers for the simulator's interleaving
  const rng = (seed) => () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

  S2S.lesson({
    id: "p0-3", n: "P0.3", title: "Threads, atomics and caches",
    subtitle: "Systems primer · first principles · T0, any laptop",
    kicker: "Lesson · ≈ 50 min",
    headline: "Two threads, two counters, one very slow line",
    intro: `<p>An inference server does many things at once: it accepts requests, forms batches, runs the model and hands out cache memory. Each of those runs on a <b>thread</b>, and threads share memory. Sharing memory is what makes them fast, and also what makes them wrong or slow when you are careless.</p>
<p>This lesson builds the toolkit step by step: why <code>counter++</code> breaks, how mutexes and atomics fix it, why caches make two threads slow each other down even when they never touch the same variable, and finally the two components you will reuse for the rest of the course: a <b>thread pool</b> and a <b>block allocator</b> (D2), the ancestor of vLLM's KV-cache manager.</p>`,
    facts: ["11 steps", "5 checkpoints", "1 simulator", "5 exercises"],
    legend: [["k", "shared data"], ["v", "per-thread"], ["q", "synchronisation"], ["ok", "hit / correct"], ["hot", "conflict / lost"]],
    prev: "p0-2", next: "p0-4",
    steps: [
      { rail: "threads", title: "A process is an address space; a thread is a worker inside it",
        body: `<p>In P0.2 every process got its own page tables, so its own private address space. A <b>process</b> is that address space plus one or more <b>threads</b>. A thread is a running sequence of instructions: it has its own <b>stack</b> (local variables, return addresses) and its own <b>registers</b>, including the instruction pointer that says where it is in the code.</p>
<p>Everything else is shared by the threads of one process: the heap, global variables, memory-mapped files (your model weights), open files.</p>
<table><tr><th></th><th>process</th><th>thread</th></tr>
<tr><td>address space</td><td>its own page tables</td><td>shared with its process</td></tr>
<tr><td>creating one</td><td><code>fork</code>: new page tables</td><td>a stack + a kernel task</td></tr>
<tr><td>talking</td><td>pipes, sockets, shared memory</td><td>plain memory</td></tr>
<tr><td>one crashes</td><td>the others survive</td><td>the whole process dies</td></tr></table>
<p>A thread is cheaper to create because no address space is built: <code>examples/01_proc_vs_thread.c</code> times both on your machine. Serving systems use both. vLLM, for example, runs its API server and its engine core as separate processes, and uses threads or async tasks inside each.</p>`,
        scene(G) {
          G.text(24, 36, "two processes: two separate address spaces", { size: 13, color: "q" });
          [24, 336].forEach((x, p) => {
            G.rect(x, 48, 280, 120, { stroke: "q", rx: 10 });
            G.text(x + 12, 68, p ? "process B" : "process A", { size: 12, color: "muted" });
            G.box(x + 12, 80, 130, 32, "page table", { fill: "w", size: 12 });
            G.box(x + 12, 122, 130, 32, "heap · globals", { fill: "k", size: 12 });
            G.box(x + 154, 80, 114, 74, "1 thread", { fill: "v", size: 12 });
          });
          G.text(24, 204, "one process, three threads: one shared address space", { size: 13, color: "ok" });
          G.rect(24, 216, 592, 186, { stroke: "ok", rx: 10 });
          G.box(40, 230, 560, 40, "shared: heap · globals · mmap'd weights · open files", { fill: "k", size: 13 });
          const th = [];
          for (let i = 0; i < 3; i++) {
            const x = 40 + i * 195, g = G.group();
            G.box(x, 300, 170, 40, `thread ${i}: stack`, { fill: "v", size: 12, parent: g });
            G.box(x, 348, 170, 34, "registers · IP", { stroke: "muted", color: "muted", size: 12, parent: g });
            G.arrow(x + 85, 298, x + 85, 274, { color: "ink", w: 1.5, parent: g });
            th.push(g);
          }
          G.from(th, { opacity: 0, y: 16, stagger: 0.2, duration: 0.35 });
          G.caption("threads share all memory; each has only its own stack and registers");
        } },

      { rail: "the race", title: "counter++ is three steps, and threads interleave them",
        body: `<p>Two threads each count tokens into one shared variable: <code>counter++</code>. It looks like one operation. The CPU actually does three: <b>load</b> the value into a register, <b>add</b> one, <b>store</b> it back.</p>
<p>The operating system can pause a thread between any two instructions, and on a multi-core CPU two threads truly run at the same time. One unlucky order:</p>
<div class="eq">counter = 5
T0: load  → r0 = 5
T1: load  → r1 = 5
T0: add, store → counter = 6
T1: add, store → counter = 6    (should be 7)</div>
<p>One increment vanished: a <b>lost update</b>. Two threads touching the same memory, at least one writing, with nothing ordering them, is a <b>data race</b>. In C++ a data race is <i>undefined behaviour</i>: the compiler assumes it never happens, so the result may be even worse than a lost count. ThreadSanitizer (<code>-fsanitize=thread</code>) detects races while the program runs. Exercise 1 hands you a racy <code>TokenStats</code> to fix.</p>`,
        check: { q: "Two threads each run counter++ 1,000,000 times on a shared int, with no synchronisation. What final value should you expect?",
          options: ["Exactly 2,000,000", "Some value up to 2,000,000, usually less, different on each run", "Exactly 1,000,000"], answer: 1,
          why: "Every time the two load-add-store sequences overlap, one increment is lost. How often that happens depends on timing, so the total changes from run to run. And since it is a data race, the C++ standard does not even promise that much." },
        scene(G) {
          G.text(40, 40, "thread 0", { size: 13, color: "v" }); G.text(268, 40, "memory: counter", { size: 13, color: "k" }); G.text(440, 40, "thread 1", { size: 13, color: "v" });
          G.label(24, 64, "time ↓", { size: 11 });
          const rows = [[0, "load → r0 = 5", "5"], [1, "load → r1 = 5", "5"], [0, "add → r0 = 6", "5"], [0, "store r0", "6"], [1, "add → r1 = 6", "6"], [1, "store r1", "6"]];
          const out = [];
          rows.forEach(([t, s, m], i) => {
            const y = 74 + i * 44, g = G.group();
            G.box(t ? 430 : 40, y, 170, 34, s, { stroke: "v", color: "v", size: 12, parent: g });
            G.box(268, y, 120, 34, m, { fill: i === 5 ? "hot" : "k", size: 14, parent: g });
            G.arrow(t ? 428 : 212, y + 17, t ? 392 : 264, y + 17, { color: "muted", parent: g });
            out.push(g);
          });
          G.text(24, 358, "two increments ran, but the counter went 5 → 6", { size: 14, color: "hot" });
          G.text(24, 384, "one update is lost: a data race", { size: 13, color: "hot" });
          G.from(out, { opacity: 0, y: -8, stagger: 0.3, duration: 0.25 });
          G.caption("both threads read 5 before either wrote back");
        } },

      { rail: "mutex", title: "A mutex lets one thread in at a time",
        body: `<p>The simplest fix: make the three steps happen with nobody else in between. A <b>mutex</b> (mutual exclusion) is a lock. Only one thread can hold it; others that ask wait until it is released. The code between lock and unlock is the <b>critical section</b>.</p>
<div class="eq">std::mutex mu;
{
  std::lock_guard g(mu);  // lock, or wait
  ++counter;              // critical section
}                         // g destroyed: unlock</div>
<p><code>std::lock_guard</code> is RAII from P0.1: the destructor unlocks, so an early return or an exception can't leave the lock held.</p>
<p>It is correct, but look at the timeline. With four threads hammering one counter, only one is ever doing useful work; the others queue. Four threads are no faster than one, and each hand-off of the lock between cores costs extra. The rules that follow: keep critical sections short, and <b>never run long or unknown code while holding a lock</b>. That one returns in the thread pool.</p>`,
        legend: [["k", "holds the lock"], ["hot", "waiting"]],
        scene(G) {
          G.text(24, 40, "4 threads, 1 mutex, 1 counter", { size: 13 });
          const segs = [];
          for (let t = 0; t < 4; t++) {
            const y = 64 + t * 48;
            G.text(24, y + 21, `T${t}`, { size: 13, color: "v" });
            for (let s = 0; s < 14; s++) {
              const mine = s % 4 === t;
              const r = G.rect(60 + s * 40, y, 36, 30, { fill: mine ? "k" : "hot", rx: 4, opacity: mine ? 0.95 : 0.3 });
              if (mine) segs.push(r);
            }
          }
          G.arrow(60, 268, 620, 268, { color: "muted" }); G.label(620, 286, "time →", { anchor: "end", size: 11 });
          G.text(24, 316, "at any moment exactly one thread is inside", { size: 13 });
          G.text(24, 342, "the others wait: 4 threads ≈ 1 thread, plus hand-offs", { size: 13, color: "hot" });
          G.text(24, 386, "{ std::lock_guard g(mu); ++counter; }", { size: 13, color: "q" });
          G.from(segs, { opacity: 0, stagger: 0.06, duration: 0.15 });
          G.caption("a mutex makes the result correct by making the threads take turns");
        } },

      { rail: "atomics", title: "Atomics: one indivisible read-modify-write",
        body: `<p>For a single variable there is a cheaper tool. <code>std::atomic&lt;uint64_t&gt;</code> offers operations the hardware performs as <b>one indivisible step</b>. <code>counter.fetch_add(1)</code> compiles on x86 to a single <code>lock xadd</code> instruction: the core holds the variable's cache line exclusively while it loads, adds and stores, so no other core can slip in between.</p>
<p>Not every update is a single instruction. Exercise 1 also tracks the largest batch seen. "If bigger, store it" is a check then an act, and another thread can act in between. The tool is <b>compare-and-swap</b> (CAS): "if the value is still what I last saw, replace it; otherwise tell me the new value".</p>
<div class="eq">uint64_t cur = max_batch.load();
while (b &gt; cur &amp;&amp;
       !max_batch.compare_exchange_weak(cur, b)) {
  // CAS failed: cur now holds the fresh value; retry
}</div>
<p>If another thread raised the maximum in the meantime, the CAS fails, <code>cur</code> is refreshed, and the loop re-checks <code>b &gt; cur</code>. Every lock-free data structure is built from this loop.</p>`,
        check: { q: "Why is <code>if (b &gt; max.load()) max.store(b);</code> wrong even though max is a std::atomic?",
          options: ["Atomic loads are not allowed inside an if", "Another thread can store a larger value between the load and the store, and this thread then overwrites it with a smaller one", "store() is not atomic"], answer: 1,
          why: "Each call is atomic on its own, but the pair is not. Between the check and the store another thread can raise max; the store then replaces it with a smaller b. compare_exchange makes check-and-store one atomic step." },
        scene(G) {
          G.text(24, 36, "counter++ on a plain int: three steps, gaps in between", { size: 13, color: "hot" });
          ["load", "add", "store"].forEach((s, i) => G.box(24 + i * 150, 50, 120, 36, s, { stroke: "hot", color: "hot", size: 13 }));
          const gap = G.box(150, 96, 140, 28, "T1 sneaks in here", { stroke: "hot", color: "hot", size: 11, dash: "4 3" });
          G.text(24, 160, "fetch_add: one indivisible step", { size: 13, color: "ok" });
          const at = G.box(24, 174, 420, 40, "lock xadd: load + add + store, line held", { fill: "ok", size: 13 });
          G.text(24, 252, "compare-and-swap loop (atomic max)", { size: 13, color: "q" });
          G.box(24, 266, 150, 36, "cur = max.load()", { stroke: "q", color: "q", size: 12 });
          G.box(206, 266, 110, 36, "b > cur ?", { stroke: "q", color: "q", size: 12 });
          G.box(350, 266, 160, 36, "CAS(cur → b)", { fill: "q", size: 12 });
          G.box(540, 266, 76, 36, "done", { fill: "ok", size: 12 });
          G.arrow(176, 284, 202, 284, { color: "q" }); G.arrow(318, 284, 346, 284, { color: "q" }); G.label(320, 276, "yes", { size: 10 });
          G.arrow(512, 284, 536, 284, { color: "ok" }); G.label(512, 276, "ok", { size: 10, color: "ok" });
          G.path("M 430 304 C 430 360, 262 360, 262 308", { color: "hot", arrow: true, dash: "4 3" });
          G.label(300, 372, "failed: cur = fresh value, check again", { size: 11, color: "hot" });
          G.path("M 300 266 C 300 232, 578 232, 578 262", { color: "muted", arrow: true, dash: "2 3" }); G.label(470, 232, "no: nothing to do", { size: 10 });
          G.pulse(gap, { repeat: 4 }); G.from(at, { opacity: 0, scale: 0.8, transformOrigin: "center", duration: 0.5, delay: 0.3 });
          G.caption("atomics fuse the steps; CAS turns check-then-act into one step");
        } },

      { rail: "coherence", title: "Every core has its own copy: cache coherence",
        body: `<p>P0.1 said memory moves in 64-byte cache lines. Each core keeps recently used lines in its own private caches (L1, usually L2), with a larger L3 shared by all cores. So the same line can sit in two cores' caches at once. If core 0 writes its copy, core 1's copy is now stale.</p>
<p>Hardware keeps the copies consistent with a <b>coherence protocol</b>. The textbook one, <b>MESI</b>, gives each line in each cache one of four states:</p>
<table><tr><th>state</th><th>meaning</th></tr>
<tr><td><b>M</b> modified</td><td>only I have it, and I changed it</td></tr>
<tr><td><b>E</b> exclusive</td><td>only I have it, unchanged</td></tr>
<tr><td><b>S</b> shared</td><td>others may have read-only copies</td></tr>
<tr><td><b>I</b> invalid</td><td>my copy is stale: don't use it</td></tr></table>
<p>The one rule that matters here: <b>to write a line, a core must own it (M or E)</b>. If other cores hold copies, it first sends them an invalidation (a "read for ownership") and waits for the line to arrive. An L1 hit takes a few cycles; moving a line from another core takes tens to over a hundred cycles, depending on the CPU.</p>`,
        legend: [["ok", "M / E: owned"], ["v", "S: shared"], ["hot", "I: invalid"], ["k", "data"]],
        scene(G) {
          [[24, "core 0", "M", "ok"], [346, "core 1", "I", "hot"]].forEach(([x, n, st, col]) => {
            G.rect(x, 40, 270, 150, { stroke: "ink", rx: 10 });
            G.text(x + 14, 64, n, { size: 13 });
            G.text(x + 14, 92, "private L1 / L2", { size: 11, color: "muted" });
            for (let i = 0; i < 8; i++) G.rect(x + 14 + i * 26, 110, 24, 34, { fill: "k", rx: 3, opacity: st === "I" ? 0.25 : 0.9 });
            G.box(x + 230, 110, 30, 34, st, { fill: col, size: 15 });
            G.label(x + 14, 166, "line 0x1000 (64 B)", { size: 11 });
          });
          const inv = G.path("M 480 194 C 440 236, 200 236, 160 196", { color: "hot", w: 2.5, arrow: true });
          G.label(320, 252, "core 0's write: invalidate core 1's copy", { anchor: "middle", size: 11, color: "hot" });
          G.box(24, 268, 592, 48, "shared L3 cache", { fill: "w", size: 13 });
          G.box(24, 336, 592, 40, "RAM", { stroke: "muted", color: "muted", size: 13 });
          G.text(24, 404, "write needs ownership: other copies must be invalidated first", { size: 13 });
          G.pulse(inv, { repeat: 5 });
          G.caption("core 0 wrote the line, so core 1's copy became invalid");
        } },

      { rail: "false sharing", title: "False sharing: different variables, same line",
        body: `<p>Now give each thread its <i>own</i> counter. No data race any more, so this should scale perfectly:</p>
<div class="eq">struct { std::atomic&lt;long&gt; a, b; } c;  // 16 B
// thread 0: c.a++ forever
// thread 1: c.b++ forever</div>
<p>But <code>a</code> and <code>b</code> sit in the same 64-byte line. Coherence works on whole lines, not variables. Core 0 writes <code>a</code>: it takes the line and invalidates core 1's copy. Core 1 writes <code>b</code>: it takes the line back. Count eight alternating writes: the first fetches the line from memory, and <b>each of the other seven moves it between cores</b>. The threads share no data, yet they fight over every write. That is <b>false sharing</b>.</p>
<p>The fix is to give each counter a line of its own:</p>
<div class="eq">struct alignas(64) Padded {
  std::atomic&lt;long&gt; v;   // 8 B used
};                        // + 56 B padding
static_assert(sizeof(Padded) == 64);</div>
<p>This is P0.1's padding used on purpose: 56 wasted bytes per counter buy back the whole speed-up. Exercise 2 pads <code>PerWorkerCounters</code>; <code>bench/false_sharing_bench</code> prints the packed and padded ns/op on your machine. On Linux, <code>perf c2c</code> finds contended lines directly. Some CPUs (Apple M-series) move 128 bytes at a time for this purpose, so <code>std::hardware_destructive_interference_size</code> is the portable constant where available.</p>`,
        check: { q: "Eight worker threads each increment their own 8-byte atomic counter, stored packed in an array that starts on a 64-byte boundary. How many cache lines do the threads fight over?",
          options: ["None: every thread has its own counter", "One: all eight counters live in the same 64-byte line", "Eight: one per counter"], answer: 1,
          why: "8 counters × 8 bytes = 64 bytes = exactly one line. Every write by any thread needs that line in M state, so all eight threads take turns owning it. Padding each counter to 64 bytes gives eight lines and ends the fight." },
        legend: [["k", "a (thread 0)"], ["v", "b (thread 1)"], ["hot", "line transfer"], ["ok", "hit"]],
        scene(G) {
          G.text(24, 36, "packed: a and b in one 64-byte line", { size: 13, color: "hot" });
          G.box(24, 50, 130, 36, "core 0: a++", { stroke: "k", color: "k", size: 12 });
          G.box(486, 50, 130, 36, "core 1: b++", { stroke: "v", color: "v", size: 12 });
          for (let i = 0; i < 8; i++) G.box(176 + i * 36, 52, 32, 32, i === 0 ? "a" : i === 1 ? "b" : "", { fill: i === 0 ? "k" : i === 1 ? "v" : "line", size: 12 });
          const ping = G.path("M 120 90 C 220 122, 420 122, 520 90", { color: "hot", w: 2.5, arrow: true });
          G.path("M 520 94 C 420 150, 220 150, 120 94", { color: "hot", w: 2.5, arrow: true });
          G.label(320, 162, "the line moves on every write", { anchor: "middle", size: 12, color: "hot" });
          const w = [];
          for (let i = 0; i < 8; i++) w.push(G.box(24 + i * 74, 174, 66, 28, i ? "move" : "fetch", { fill: i ? "hot" : "w", size: 11 }));
          G.label(24, 222, "8 alternating writes → 7 line transfers between cores", { size: 12, color: "ink" });
          G.text(24, 258, "padded: alignas(64), one line each", { size: 13, color: "ok" });
          for (let l = 0; l < 2; l++) for (let i = 0; i < 8; i++) G.rect(24 + l * 310 + i * 36, 272, 32, 32, { fill: i ? "line" : l ? "v" : "k", rx: 4, opacity: i ? 0.5 : 1 });
          G.label(24, 322, "line 0: a + 56 B padding", { size: 11 }); G.label(334, 322, "line 1: b + 56 B padding", { size: 11 });
          for (let i = 0; i < 8; i++) G.box(24 + i * 74, 336, 66, 30, i < 2 ? "fetch" : "hit", { fill: i < 2 ? "w" : "ok", size: 11 });
          G.label(24, 386, "after each core's first fetch: every write is an L1 hit", { size: 12, color: "ink" });
          G.from(w, { opacity: 0, stagger: 0.15, duration: 0.2 }); G.pulse(ping, { repeat: 6 });
          G.caption("coherence tracks lines, so neighbours on a line contend");
        } },

      { rail: "share nothing", title: "The fastest shared write is the one you don't make",
        body: `<p>Rank the three ways to count with T threads by how much they write to shared lines. Take example 03: 20,000,000 increments split over 4 threads.</p>
<table><tr><th>strategy</th><th>shared-line writes</th></tr>
<tr><td><b>mutex</b> around <code>++</code></td><td>20 M lock + 20 M unlock writes on the lock's line, 20 M writes to the counter's line, and threads waiting their turn</td></tr>
<tr><td><b>one atomic</b>, <code>fetch_add</code></td><td>20 M atomic writes to one line, which moves whenever a different core writes next</td></tr>
<tr><td><b>per-thread</b> local count, add once at the end</td><td>4 (one per thread, at the end)</td></tr></table>
<div class="eq">long local = 0;          // in a register
for (...) ++local;       // no sharing at all
total.fetch_add(local);  // once per thread</div>
<p>All three give exactly the same total. <code>examples/03_atomics.cpp</code> times them; the expected order is per-thread &lt; atomic &lt; mutex, with the mutex degrading most as threads are added.</p>
<p>The same idea runs through the rest of the course: give each worker its own slice of the output (each thread computes its own rows of a matrix product), and combine once at the end. Communication, not arithmetic, is what limits parallel speed-up.</p>`,
        scene(G) {
          G.text(24, 36, "4 threads × 5,000,000 increments: writes to shared lines", { size: 13 });
          const rows = [["mutex", 40, "hot", "≥ 40 M"], ["one atomic", 20, "v", "20 M"], ["per-thread", 0.000004, "ok", "4"]];
          const bars = [];
          rows.forEach(([n, val, col, lab], i) => {
            const y = 66 + i * 80;
            G.text(24, y + 26, n, { size: 13 });
            bars.push(G.rect(160, y, Math.max(3, (val / 40) * 440), 40, { fill: col, rx: 4 }));
            G.text(170 + Math.max(3, (val / 40) * 440) - (val > 10 ? 90 : -4), y + 26, lab, { size: 13, color: val > 10 ? "bg" : "ink" });
          });
          G.label(160, 310, "bar length ∝ shared-line writes (counts, not times)", { size: 11 });
          G.text(24, 352, "same final total in all three", { size: 13, color: "ok" });
          G.text(24, 380, "expected time order: per-thread < atomic < mutex", { size: 13 });
          G.from(bars, { attr: { width: 0 }, stagger: 0.2, duration: 0.5 });
          G.caption("combine once at the end instead of sharing on every step");
        } },

      { rail: "acquire · release", title: "Publishing data: acquire and release",
        body: `<p>Atomics have a second job: <b>ordering</b>. A typical hand-off: one thread fills a buffer, then raises a flag; another waits for the flag, then reads the buffer.</p>
<div class="eq">// producer thread
data = 42;
ready.store(true, release);

// consumer thread
while (!ready.load(acquire)) {}
use(data);               // guaranteed to see 42</div>
<p>Why is ordering a question at all? To go fast, both the compiler and the CPU may reorder memory operations that look independent: <code>data</code> and <code>ready</code> are different addresses. Without a rule, the consumer could see <code>ready == true</code> and still read the old <code>data</code>.</p>
<p>The rule: a <b>release</b> store makes every write that came before it visible to any thread that <b>acquire</b>-loads the same atomic and sees the stored value. Release is "everything above stays above"; acquire is "everything below stays below".</p>
<ul><li><code>memory_order_seq_cst</code> (the default): strongest and simplest. Use it unless a profiler says otherwise.</li>
<li><code>memory_order_relaxed</code>: atomicity only, no ordering. Right for statistics counters (the false-sharing bench uses it), wrong for a flag that publishes data.</li></ul>
<p><code>volatile</code> is <i>not</i> a synchronisation tool in C++. Exercise 5's lock-free queue is exactly this producer/consumer pair, checked under TSan.</p>`,
        check: { q: "The consumer waits with ready.load(memory_order_relaxed), then reads data. What can go wrong?",
          options: ["Nothing: the flag is atomic, so the data is safe", "It may see ready == true but read the old value of data, because relaxed gives no ordering", "The load never returns"], answer: 1,
          why: "Relaxed makes the flag itself race-free but orders nothing around it. Only the release store paired with an acquire load guarantees that the writes before the store are visible after the load." },
        scene(G) {
          G.text(24, 36, "producer thread", { size: 13, color: "v" }); G.text(350, 36, "consumer thread", { size: 13, color: "v" });
          G.box(24, 56, 250, 40, "data = 42", { fill: "k", size: 13 });
          G.box(24, 150, 250, 40, "ready.store(true, release)", { fill: "q", size: 12 });
          G.line(24, 124, 274, 124, { color: "q", dash: "5 4" }); G.label(24, 118, "writes above can't move below", { size: 10, color: "q" });
          G.box(350, 150, 266, 40, "ready.load(acquire) == true", { fill: "q", size: 12 });
          G.line(350, 214, 616, 214, { color: "q", dash: "5 4" }); G.label(350, 230, "reads below can't move above", { size: 10, color: "q" });
          G.box(350, 248, 266, 40, "use(data) → 42", { fill: "ok", size: 13 });
          const sw = G.arrow(276, 170, 346, 170, { color: "ink", w: 2.5 });
          G.label(311, 206, "pairs with", { anchor: "middle", size: 11, color: "ink" });
          G.path("M 276 66 C 326 66, 316 268, 346 268", { color: "ok", w: 1.5, dash: "3 3", arrow: true });
          G.label(350, 310, "data = 42 is visible after the acquire", { size: 11, color: "ok" });
          G.text(24, 364, "with relaxed instead: ready may be true while data still reads 0", { size: 13, color: "hot" });
          G.pulse(sw, { repeat: 4 });
          G.caption("release publishes everything before it to whoever acquires the flag");
        } },

      { rail: "thread pool", title: "A thread pool: create workers once, feed them chunks",
        body: `<p>P0.5's engine does every matrix product on several threads. Llama-3-8B has 7 weight matrices per layer and 32 layers, so a token needs 224 matrix products plus the output layer. Creating fresh threads for each one would spend more time starting threads than computing.</p>
<p>A <b>thread pool</b> starts N workers once. They sleep on a <b>condition variable</b> until a task appears in a shared queue:</p>
<div class="eq">for (;;) {
  { std::unique_lock lk(mu_);
    cv_.wait(lk, [&amp;]{
      return stop_ || !q_.empty(); });
    if (stop_ &amp;&amp; q_.empty()) return;
    task = pop(q_); }      // unlock here
  task();                  // run OUTSIDE the lock
}</div>
<p>Two details from the module's common mistakes: always <code>wait</code> with a predicate (wake-ups can be spurious), and run the task after unlocking, or one long task blocks every other worker.</p>
<p><code>parallel_for(n, fn)</code> splits <code>[0, n)</code> into one contiguous chunk per worker: <code>chunks = min(n, workers)</code>, <code>step = ceil(n / chunks)</code>. For n = 10 and 4 workers, step = 3: [0,3) [3,6) [6,9) [9,10). It then waits on a countdown until every chunk is done. Contiguous chunks mean each thread streams its own memory and writes its own output lines, so no false sharing. A 4096-row matrix-vector product on 8 workers gives each 512 rows. You write this in exercise 3; the reference is <code>examples/thread_pool.hpp</code>.</p>`,
        scene(G) {
          G.text(24, 36, "parallel_for(10, fn) on 4 workers: step = ceil(10 / 4) = 3", { size: 13 });
          const cols = ["k", "v", "q", "ok"], ch = [[0, 3], [3, 6], [6, 9], [9, 10]];
          for (let i = 0; i < 10; i++) {
            const c = ch.findIndex(([b, e]) => i >= b && i < e);
            G.box(24 + i * 58, 52, 52, 36, String(i), { fill: cols[c], size: 13 });
          }
          G.text(24, 130, "queue (mutex + condition variable)", { size: 12, color: "muted" });
          G.rect(24, 140, 592, 52, { stroke: "muted", rx: 8 });
          const tasks = ch.map(([b, e], i) => G.box(36 + i * 146, 150, 132, 32, `fn(${b}, ${e})`, { fill: cols[i], size: 12 }));
          const ws = [];
          for (let i = 0; i < 4; i++) {
            const g = G.group();
            G.arrow(102 + i * 146, 194, 102 + i * 146, 232, { color: "muted", parent: g });
            G.box(36 + i * 146, 236, 132, 60, `worker ${i}`, { stroke: cols[i], color: cols[i], size: 13, parent: g });
            G.label(102 + i * 146, 316, `${ch[i][1] - ch[i][0]} item${ch[i][1] - ch[i][0] > 1 ? "s" : ""}`, { anchor: "middle", size: 11, parent: g });
            ws.push(g);
          }
          G.text(24, 360, "countdown 4 → 3 → 2 → 1 → 0, then parallel_for returns", { size: 13 });
          G.text(24, 388, "workers are created once and reused for every call", { size: 13, color: "ok" });
          G.from(tasks, { opacity: 0, x: -20, stagger: 0.15, duration: 0.3 }); G.from(ws, { opacity: 0, stagger: 0.15, duration: 0.3, delay: 0.5 });
          G.caption("contiguous chunks: each worker owns its own range of the data");
        } },

      { rail: "D2 allocator", title: "D2: a block allocator with reference counts",
        body: `<p>A language model being served keeps a growing amount of memory per conversation (P1.2 calls it the <b>KV cache</b>: 128 KiB per token for Llama-3-8B). Allocating one big region per conversation wastes memory, because nobody knows in advance how long a reply will be. Serving engines instead cut memory into fixed-size <b>blocks</b>, say 16 tokens each, and give each conversation a list of block ids. Exercise 4 builds that bookkeeping, with integer ids standing in for memory.</p>
<ul><li><b>Free list</b>: a stack of free ids. <code>allocate()</code> pops one in O(1); <code>free()</code> pushes it back. Last freed, first reused (LIFO), because its memory is likely still in cache.</li>
<li><b>Reference counts</b>: two conversations that start with the same prompt can <b>share</b> the prompt's blocks. <code>share()</code> adds an owner; <code>free()</code> removes one and returns the block at 0.</li>
<li><b>Copy-on-write</b>: before writing into a shared block, a sequence gets its own copy.</li></ul>
<p>Worked example, 8 blocks, free stack <code>[0 … 7]</code> with 7 on top. Sequence A has a 40-token prompt: ceil(40 / 16) = 3 blocks, so it pops 7, 6, 5 (block 5 is half full). B is forked from A and shares all three: refcounts 2, 2, 2. B now appends token 41, which goes into block 5, but 5 is shared, so <code>copy_on_write(5)</code> pops block 4 for B and drops 5's count to 1.</p>
<div class="eq">A = [7, 6, 5]   B = [7, 6, 4]
refcount: 7→2  6→2  5→1  4→1   free: 0 1 2 3
num_free + blocks in use = 4 + 4 = 8 ✓</div>
<p>All of this runs under one mutex. <code>allocate_many(n)</code> must check "at least n free" and pop them <i>under the same lock</i>: a scheduler admits a request only if all its blocks fit. P6.3 turns this class into the real paged KV cache.</p>`,
        check: { q: "allocate_many(3) checks num_free() ≥ 3 under the lock, unlocks, then locks again to pop three blocks. What can go wrong?",
          options: ["Nothing, as long as each step holds the lock", "Another thread can take blocks between the check and the pops, so the request ends half-allocated", "The refcounts become negative"], answer: 1,
          why: "Each step is safe on its own, but the decision is based on a state that can change before the pops. All-or-nothing needs the check and the pops in one critical section; this is the same check-then-act problem as the atomic max." },
        legend: [["v", "shared (refcount 2)"], ["k", "owned by A"], ["q", "owned by B"], ["line", "free"]],
        scene(G) {
          G.text(24, 36, "8 blocks after: A allocates 3, B forks, B appends", { size: 13 });
          const rc = [0, 0, 0, 0, 1, 1, 2, 2], col = ["line", "line", "line", "line", "q", "k", "v", "v"];
          const bl = [];
          for (let i = 0; i < 8; i++) {
            bl.push(G.box(24 + i * 74, 52, 66, 50, `#${i}`, { fill: col[i], size: 14, boxOpacity: col[i] === "line" ? 0.6 : 1, color: col[i] === "line" ? "muted" : "bg" }));
            G.label(57 + i * 74, 122, `ref ${rc[i]}`, { anchor: "middle", size: 11, color: rc[i] ? "ink" : "muted" });
          }
          G.text(24, 170, "sequence A", { size: 13 }); G.text(24, 236, "sequence B", { size: 13 });
          [[7, "v"], [6, "v"], [5, "k"]].forEach(([b, c], i) => G.box(140 + i * 60, 150, 52, 32, `#${b}`, { fill: c, size: 12 }));
          [[7, "v"], [6, "v"], [4, "q"]].forEach(([b, c], i) => G.box(140 + i * 60, 216, 52, 32, `#${b}`, { fill: c, size: 12 }));
          G.label(330, 170, "40 tokens: 16 + 16 + 8", { size: 11 });
          const cow = G.box(330, 216, 280, 32, "token 41 → copy_on_write(5) → #4", { stroke: "q", color: "q", size: 11 });
          G.text(24, 296, "free stack (top on the right):", { size: 12, color: "muted" });
          for (let i = 0; i < 4; i++) G.box(260 + i * 56, 278, 48, 30, String(i), { stroke: "muted", color: "muted", size: 12 });
          G.text(24, 348, "num_free 4 + in use 4 = 8 blocks ✓", { size: 14, color: "ok" });
          G.label(24, 376, "free() at refcount 0 throws: no double free, no use-after-free", { size: 12, color: "ink" });
          G.from(bl, { opacity: 0, y: -10, stagger: 0.08, duration: 0.25 }); G.pulse(cow, { repeat: 4 });
          G.caption("shared prefix blocks are stored once and counted twice");
        } },

      { rail: "NUMA", title: "On big servers, memory has a home",
        body: `<p>A laptop has one memory system. A large server often has two CPU <b>sockets</b>, each with its own memory controllers and its own RAM. A core can reach the other socket's memory over the link between the sockets, but more slowly and with less bandwidth. This is <b>NUMA</b>: non-uniform memory access.</p>
<p>Which socket's memory does a page live on? By default Linux puts it on the node of the core that <b>first touches</b> it: P0.2's demand paging again. So the thread that <i>initialises</i> a buffer decides where it lives. A buffer filled by one thread and then read by workers on the other socket is remote for all of them.</p>
<div class="eq">numactl --hardware          # show the nodes
# run on node 0's cores and memory only:
numactl --cpunodebind=0 --membind=0 ./app</div>
<p>GPU servers have the same issue one level up: each GPU hangs off one socket's PCIe lanes, so the pinned host buffers that feed it (P0.2) should live in that socket's memory. <code>nvidia-smi topo -m</code> shows the layout (P4). On a laptop this is explanation only; measuring it needs a multi-socket machine (tier T3, optional).</p>`,
        legend: [["ok", "local access"], ["hot", "remote access"], ["k", "memory"], ["w", "socket"]],
        scene(G) {
          [[24, "socket 0"], [356, "socket 1"]].forEach(([x, n], s) => {
            G.rect(x, 44, 260, 150, { fill: "w", opacity: 0.35, rx: 10 });
            G.text(x + 14, 66, n, { size: 13 });
            for (let c = 0; c < 4; c++) G.box(x + 14 + c * 60, 82, 52, 40, `core ${s * 4 + c}`, { stroke: "ink", size: 11 });
            G.box(x + 14, 140, 232, 36, "memory controller", { stroke: "muted", color: "muted", size: 12 });
            G.box(x, 250, 260, 50, `node ${s} RAM`, { fill: "k", size: 13 });
            G.line(x + 130, 178, x + 130, 248, { color: "muted", w: 3 });
          });
          G.line(286, 110, 354, 110, { color: "muted", w: 6 }); G.label(320, 100, "link", { anchor: "middle", size: 11 });
          const loc = G.path("M 66 124 L 66 246", { color: "ok", w: 3, arrow: true });
          const rem = G.path("M 124 124 L 124 131 L 584 131 L 584 246", { color: "hot", w: 3, arrow: true, dash: "6 4" });
          G.label(76, 220, "local", { size: 12, color: "ok" }); G.label(320, 224, "remote: across the link, slower", { anchor: "middle", size: 12, color: "hot" });
          G.box(24, 326, 160, 40, "GPU on PCIe", { stroke: "q", color: "q", size: 12 });
          G.line(104, 302, 104, 324, { color: "q", w: 2 });
          G.text(200, 344, "first touch decides a page's node", { size: 13 });
          G.text(200, 368, "initialise data on the thread that uses it", { size: 13, color: "ok" });
          G.pulse(rem, { repeat: 4 });
          G.caption("the same load costs more when the page lives on the other socket");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Threads share their process's memory; each has only its own stack and registers.</li>
<li><code>counter++</code> is load, add, store. Unsynchronised, threads lose updates (a data race). Mutexes make threads take turns; atomics make one update indivisible; CAS loops make check-then-act atomic.</li>
<li>Caches are kept coherent per 64-byte line, and a write needs the line exclusively. Two threads writing neighbouring variables fight over the line: false sharing. Pad with <code>alignas(64)</code>.</li>
<li>Release/acquire pairs publish data from one thread to another; relaxed gives atomicity only.</li>
<li>A thread pool creates workers once and splits loops into contiguous chunks. D2 manages fixed-size blocks with a free stack, reference counts and copy-on-write: the core of a paged KV cache.</li></ul>
<p>Next, P0.4 looks inside a single core: how one instruction can work on 8 numbers at once, and how to tell whether a program is limited by arithmetic or by memory.</p>`,
    sim: {
      title: "Who owns the line? A false-sharing simulator",
      intro: "Each thread increments its own count as fast as it can. Choose where the counts live. Every write needs its cache line in the writing core's cache: if another core wrote it last, the line must move (red); otherwise it is a hit (green). Press Run to watch 48 writes in a random interleaving. The lower chart estimates speed-up versus one thread for every strategy.",
      height: 320,
      controls: [
        { id: "s", label: "where the counts live", type: "select", value: "packed", options: [["shared", "one shared atomic"], ["packed", "per-thread, packed"], ["padded", "per-thread, padded"], ["local", "local, add at end"]] },
        { id: "T", label: "threads (one per core)", min: 1, max: 16, value: 4 },
        { id: "line", label: "cache line size", type: "select", value: 64, options: [[64, "64 B (x86, most ARM)"], [128, "128 B (some Apple M-series cases)"]] },
        { id: "b", label: "increments per turn with the line (example value)", min: 1, max: 32, value: 4 },
        { id: "h", label: "L1 hit, cycles (example value)", min: 1, max: 10, value: 4 },
        { id: "c", label: "line transfer between cores, cycles (example value)", min: 20, max: 300, step: 10, value: 80 },
      ],
      run: { label: "Run 48 writes", frames: 48, ms: 70 },
      draw(G, v, t) {
        const T = v.T, L = v.line, h = v.h, c = v.c, B = v.b;
        const perLine = L / 8;
        // threads that share each thread's line
        const group = (s, T) => s === "shared" ? T : s === "packed" ? Math.min(T, perLine) : 1;
        const lineOf = (s, core) => s === "shared" ? 0 : s === "packed" ? Math.floor(core / perLine) : core;
        const speedup = (s, T) => {
          if (s === "local" || s === "padded") return T;
          const k = group(s, T), p = (k - 1) / k / B;
          return (T / k) * h / (h + p * c);           // T/k independent lines, each serialised
        };
        // timeline of 48 writes
        const r = rng(7 + T * 31), N = 48, shown = Math.max(0, Math.round(N * t));
        const owner = {}, res = [];
        let core = 0;
        for (let i = 0; i < N; i++) {
          if (i % B === 0) core = Math.floor(r() * T);
          const ln = lineOf(v.s, core);
          let kind;
          if (v.s === "local") kind = "hit";
          else if (owner[ln] === undefined) kind = "cold";
          else kind = owner[ln] === core ? "hit" : "move";
          owner[ln] = core; res.push([core, kind]);
        }
        const lanes = Math.min(T, 8), lh = Math.min(16, 120 / lanes);
        G.label(10, 14, `write timeline (first ${lanes} of ${T} cores shown) · green = hit · red = line moved · grey = first fetch`, { size: 11 });
        for (let k = 0; k < lanes; k++) { G.label(10, 34 + k * lh + lh / 2, `c${k}`, { size: 10 }); G.rect(36, 24 + k * lh, 600, lh - 2, { fill: "line", rx: 2, opacity: 0.35 }); }
        let moves = 0, hits = 0;
        for (let i = 0; i < shown; i++) {
          const [core, kind] = res[i];
          if (kind === "move") moves++; if (kind === "hit") hits++;
          if (core < lanes) G.rect(38 + i * 12.4, 25 + core * lh, 10, lh - 4, { fill: kind === "hit" ? "ok" : kind === "move" ? "hot" : "muted", rx: 2 });
        }
        // speed-up chart
        const x0 = 50, y0 = 300, W = 560, H = 120;
        const X = (n) => x0 + ((n - 1) / 15) * W, Y = (s) => y0 - ((Math.max(-6, Math.min(4, Math.log2(s))) + 6) / 10) * H;
        G.axes(x0, y0 - H, W, H, { ylabel: "speed-up vs 1 thread (log)" });
        [1, 4, 8, 16].forEach((n) => G.label(X(n), y0 + 14, String(n), { anchor: "middle", size: 10 }));
        [[1 / 64, "1/64×"], [1 / 8, "1/8×"], [1, "1×"], [16, "16×"]].forEach(([s, t]) => { G.label(x0 - 6, Y(s) + 4, t, { anchor: "end", size: 10 }); G.line(x0, Y(s), x0 + W, Y(s), { color: "line", dash: "2 4", w: 1 }); });
        G.label(400, y0 + 14, "threads →", { anchor: "middle", size: 10 });
        const curves = [["padded", "ok"], ["packed", "v"], ["shared", "hot"]];
        curves.forEach(([s, col]) => {
          let d = ""; for (let n = 1; n <= 16; n++) d += (n > 1 ? " L " : "M ") + X(n).toFixed(1) + " " + Y(speedup(s, n)).toFixed(1);
          G.path(d, { color: col, w: v.s === s || (v.s === "local" && s === "padded") ? 3 : 1.5, opacity: v.s === s || (v.s === "local" && s === "padded") ? 1 : 0.55 });
        });
        [["padded / local", "ok"], ["packed", "v"], ["one shared", "hot"]].forEach(([n, col], i) => { G.rect([300, 420, 500][i], 165, 10, 10, { fill: col, rx: 2 }); G.label([314, 434, 514][i], 174, n, { size: 11 }); });
        const sp = speedup(v.s, T);
        G.circle(X(T), Y(sp), 5, { fill: "ink" });
        const k = v.s === "local" || v.s === "padded" ? 1 : group(v.s, T), p = (k - 1) / k / B, cyc = h + p * c;
        const mem = v.s === "shared" ? 8 : v.s === "packed" ? 8 * T : v.s === "padded" ? L * T : 0;
        return [
          { title: "This configuration", rows: [["threads per contested line", F.num(k)], ["writes that move the line (long run)", (100 * p).toFixed(0) + "%"], ["cycles per increment, per line", F.num(cyc, 1)], ["estimated speed-up vs 1 thread", F.num(sp, 2) + "×"]],
            gauge: [[Math.min(1, sp / T), "ok"], [Math.max(0, 1 - sp / T), "hot"]], gaugeText: `share of ideal ${T}× scaling`, chip: [sp >= 0.8 * T, sp >= 0.8 * T ? "scales with threads" : sp < 1 ? "slower than one thread" : "contention limits scaling"] },
          { title: "This run (48 writes)", rows: [["line moves", F.num(moves)], ["hits", F.num(hits)], ["first fetches", F.num(shown - moves - hits)]] },
          { title: "Memory for the counters", rows: [["bytes", F.bytes(mem)], ["cache lines", F.num(Math.ceil(mem / L))]], html: `<p class="note">Padding spends ${L - 8} bytes per counter to buy one line per thread.</p>` },
          { title: "Model", html: `<p class="note">Threads take turns in a random order, making several increments per turn. A line shared by k threads moves at the start of (k − 1)/k of the turns, and its writes are serialised: each costs hit + (k − 1)/(k × turn length) × transfer cycles. Independent lines run in parallel. Cycle costs are example values; real CPUs add prefetching, store buffers and contention effects. Measure with <code>bench/false_sharing_bench</code>.</p>` },
        ];
      },
    },
    practice: {
      intro: `All five exercises build as one CMake project. Run the commands from the module folder <code>course/P0-systems-primer/P0.3-threads-atomics-caches</code>. Add <code>-DS2S_TSAN=ON</code> to the configure step to build everything under ThreadSanitizer (recommended for 1, 4 and 5), or <code>-DS2S_USE_SOLUTIONS=ON</code> to test the reference solutions.`,
      items: [
        { title: "Fix the data race", tier: "T0 · easy", goal: "Make TokenStats::record() thread-safe: an atomic total and a compare-exchange loop for the maximum. Exact counts, and TSan stays silent.",
          cmd: "cmake -S exercises -B build/ex && cmake --build build/ex -j && ctest --test-dir build/ex -R 01-fix-the-race --output-on-failure" },
        { title: "Pad the counters", tier: "T0 · easy", goal: "Give every Slot of PerWorkerCounters its own 64-byte line, then run the false-sharing bench and record the speed-up.",
          cmd: "ctest --test-dir build/ex -R 02-pad-counters --output-on-failure" },
        { title: "ThreadPool::parallel_for", tier: "T0 · medium", goal: "Split [0, n) into contiguous chunks, wait on a countdown, and rethrow the first exception after all chunks finish.",
          cmd: "ctest --test-dir build/ex -R 03-thread-pool --output-on-failure" },
        { title: "D2: KV block allocator", tier: "T0 · medium", goal: "Free stack, refcounts, share, free, all-or-nothing allocate_many and copy_on_write, all thread-safe and checked by a seeded 8-thread stress test.",
          cmd: "ctest --test-dir build/ex -R 04-kv-block-allocator --output-on-failure" },
        { title: "Lock-free SPSC queue", tier: "T0 · hard", goal: "A bounded single-producer/single-consumer ring with two atomic indices and release/acquire ordering; nothing lost, order kept, TSan silent.",
          cmd: "cmake -S exercises -B build/ex-tsan -DS2S_TSAN=ON && cmake --build build/ex-tsan -j && ctest --test-dir build/ex-tsan -R 05-spsc-queue --output-on-failure" },
      ],
      labs: [
        { label: "Examples: process vs thread cost, false sharing, mutex vs atomic vs per-thread, the thread pool (cmake -S examples -B build/examples -DCMAKE_BUILD_TYPE=Release)", path: "course/P0-systems-primer/P0.3-threads-atomics-caches/examples/" },
        { label: "Reading perf stat, perf c2c and flame graphs", path: "course/P0-systems-primer/P0.3-threads-atomics-caches/examples/05_perf_walkthrough.md" },
        { label: "Benchmarks: packed vs padded ns/op, allocator allocs/s vs threads", path: "course/P0-systems-primer/P0.3-threads-atomics-caches/bench/" },
        { label: "The thread pool reused by P0.4 and P0.5", path: "course/P0-systems-primer/P0.3-threads-atomics-caches/examples/thread_pool.hpp" },
        { label: "Animation: MESI states and false sharing", path: "animations/p0-false-sharing.html" },
        { label: "Self-check questions", path: "course/P0-systems-primer/P0.3-threads-atomics-caches/quiz.md" },
      ],
    },
  });
})();
