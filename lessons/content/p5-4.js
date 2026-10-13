/* P5.4 — Reductions and scans: the tree, Harris's ladder, warp shuffles, scan, decoupled look-back. */
(function () {
  const F = S2S.fmt;
  const VALS = [3, 1, 7, 0, 4, 1, 6, 3];   // sum 25; inclusive scan 3,4,11,11,15,16,22,25
  const strip = (G, x, y, n, cw, ch, fill, o = {}) => G.grid(x, y, 1, n, cw, ch, (r, c) => fill(c), o);

  // 64 threads (two warps of 32) over successive halving steps; active(tid, s) decides who works
  function threadRows(G, x0, y0, steps, active, label) {
    const cw = 8; x0 += 36;
    G.rect(x0 - 2, y0 - 16, 32 * cw + 2, 10, { stroke: "blue", rx: 2, opacity: 0.8 });
    G.rect(x0 + 32 * cw + 2, y0 - 16, 32 * cw - 2, 10, { stroke: "pink", rx: 2, opacity: 0.8 });
    G.label(x0 + 100, y0 - 8, "warp 0", { size: 9, color: "blue" }); G.label(x0 + 32 * cw + 100, y0 - 8, "warp 1", { size: 9, color: "pink" });
    const out = [];
    steps.forEach((s, i) => {
      const y = y0 + i * 26;
      G.label(24, y + 13, label(s), { size: 10 });
      out.push(strip(G, x0, y, 64, cw, 18, (c) => (active(c, s) ? "k" : "line"), { gap: 1 }));
    });
    return out;
  }

  S2S.lesson({
    id: "p5-4", n: "P5.4", title: "Reductions and scans",
    subtitle: "CUDA deep dive · first principles · T2, one NVIDIA GPU (sm_75+)",
    kicker: "Lesson · ≈ 55 min",
    headline: "Adding a million numbers without waiting in line",
    intro: `<p>Summing an array looks inherently serial: each addition needs the previous total. This lesson shows how a GPU does it in about 20 steps instead of a million, using a <b>tree</b>, and then climbs Mark Harris's six-rung ladder to see where a naive GPU reduction loses time (divergence, bank conflicts, idle threads, too few bytes in flight). It then covers the <b>scan</b> (running totals), the other basic parallel pattern, and the trick that makes CUB's scan fast: <b>decoupled look-back</b>. Reductions and scans are inside softmax, layer norms, sampling and attention, so you will use these ideas in every later lesson.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "active / working"], ["line", "idle"], ["v", "result"], ["blue", "warp 0"], ["pink", "warp 1"], ["hot", "problem"], ["ok", "good"]],
    prev: "p5-3", next: "p5-5",
    steps: [
      { rail: "the tree", title: "A sum is a tree, not a line",
        body: `<p>A <b>reduction</b> turns many values into one with an operation like add, max or min. The serial loop <code>s += a[i]</code> takes n steps, because each add needs the one before it. But addition can be regrouped: <code>(a + b) + (c + d)</code> equals <code>((a + b) + c) + d</code> on paper. So pair the values up and add all the pairs <b>at the same time</b>, then pair the results, and so on:</p>
<div class="eq">[3 1 7 0 4 1 6 3]   8 values
[ 4   7   5   9 ]   4 adds, all in parallel
[   11     14   ]   2 adds
[       25      ]   1 add  -> total</div>
<p>8 values need 3 rounds, because each round halves the count: rounds = log2(n). For n = 2^26 (67 million floats, 256 MiB) that is <b>26 rounds</b>, instead of 67 million sequential steps. The total number of additions is still n - 1, so no work is wasted; the tree just exposes the parallelism.</p>
<p>The cost on a GPU is not the additions (they are nearly free) but <i>how the threads and memory behave while doing them</i>. A reduction reads each value once and does one add, so it is a <b>memory-bound</b> kernel (P5.2): the best any implementation can do is read the array at full memory bandwidth. Every rung of the ladder below is a step toward that ceiling.</p>`,
        scene(G) {
          G.text(24, 36, "sum 8 values in 3 rounds", { size: 14 });
          const lv = [VALS, [4, 7, 5, 9], [11, 14], [25]];
          const cw = 60, boxes = [];
          lv.forEach((vals, L) => {
            const span = 8 / vals.length, y = 70 + L * 84;
            vals.forEach((v, i) => {
              const cx = 40 + (i * span + (span - 1) / 2) * 70;
              boxes.push(G.box(cx, y, cw, 36, String(v), { fill: L === 3 ? "v" : L === 0 ? "k" : "blue", size: 15, rx: 5 }));
              if (L > 0) {
                const pa = 40 + ((i * 2) * (span / 2) + (span / 2 - 1) / 2) * 70 + cw / 2, pb = 40 + ((i * 2 + 1) * (span / 2) + (span / 2 - 1) / 2) * 70 + cw / 2;
                G.line(pa, y - 48, cx + cw / 2, y, { color: "muted", w: 1 });
                G.line(pb, y - 48, cx + cw / 2, y, { color: "muted", w: 1 });
              }
            });
            G.label(560, y + 22, L === 0 ? "n = 8" : L === 3 ? "total" : `round ${L}`, { size: 11 });
          });
          G.label(24, 392, "log2(8) = 3 rounds;  n = 2^26 would need 26", { size: 13, color: "v" });
          G.from(boxes, { opacity: 0, y: 10, stagger: 0.04, duration: 0.25 });
          G.caption("pairs add in parallel; each round halves what is left");
        } },

      { rail: "rung 1", title: "Rung 1: a shared-memory tree and its idle lanes",
        body: `<p>On the GPU, a block loads its slice of the array into shared memory (one element per thread), then runs the tree there. The textbook first version halves the work each round using an <b>interleaved</b> pattern: in round with step <i>s</i> = 1, 2, 4, ..., thread <i>tid</i> adds its neighbour <i>s</i> places to the right, but only if <code>tid % (2*s) == 0</code>:</p>
<div class="eq">for (int s = 1; s &lt; blockDim.x; s *= 2) {
  if (tid % (2 * s) == 0) sm[tid] += sm[tid + s];
  __syncthreads();
}   // result in sm[0]</div>
<p>It is correct, and slow. Look at <i>which</i> threads work: round 1 uses every second thread, round 2 every fourth, and so on. Those active threads are <b>spread across every warp</b>. A warp issues one instruction for all 32 lanes (P5.1), so a warp with a single working lane still takes a whole instruction slot, and the lanes that fail the <code>if</code> are masked off (divergence). In the drawing, both warps stay occupied for the first 5 rounds even though the number of useful lanes falls from 32 to 1.</p>
<p>This kernel is the baseline. The exercise gives you rungs 1 and 2 and asks you to build 3 to 6.</p>`,
        scene(G) {
          G.text(24, 34, "rung 1: tid % (2s) == 0  (64 threads, 2 warps)", { size: 13 });
          const rows = threadRows(G, 24, 74, [1, 2, 4, 8, 16, 32], (t, s) => t % (2 * s) === 0, (s) => "s=" + s);
          [32, 16, 8, 4, 2, 1].forEach((n, i) => G.label(578, 74 + i * 26 + 13, n + " work", { size: 10, color: i < 5 ? "hot" : "muted" }));
          G.label(24, 270, "active threads are spread over BOTH warps for 5 of 6 rounds", { size: 12, color: "hot" });
          G.label(24, 294, "a warp with 1 busy lane still costs a full instruction", { size: 12 });
          G.label(24, 330, "teal = thread adds this round; grey = masked off", { size: 12 });
          G.from(rows, { opacity: 0, stagger: 0.12, duration: 0.3 });
          G.caption("few working lanes, but every warp stays on the hardware");
        } },

      { rail: "rungs 2-3", title: "Rungs 2 and 3: keep active threads together, address sequentially",
        body: `<p>Fix 1 (rung 2): instead of <i>skipping</i> threads, give the work to the <b>first</b> threads by computing a strided index: thread <i>tid</i> works on <code>idx = 2*s*tid</code>, for <code>tid &lt; blockDim/(2s)</code>. Active threads are now contiguous, so whole warps retire. But a new problem appears: thread <i>tid</i> touches shared-memory word <code>2*s*tid</code>, a stride of 2s words, and with 32 banks (P5.2) a stride of 2s puts <code>gcd(2s, 32)</code> lanes in one bank: 2-way, then 4-way, 8-way, 16-way conflicts as <i>s</i> grows.</p>
<p>Fix 2 (rung 3): make the <i>stride shrink</i> instead: start with s = blockDim/2 and halve it. Thread <i>tid</i> adds <code>sm[tid + s]</code> into <code>sm[tid]</code> for <code>tid &lt; s</code>:</p>
<div class="eq">for (int s = blockDim.x / 2; s &gt; 0; s &gt;&gt;= 1) {
  if (tid &lt; s) sm[tid] += sm[tid + s];
  __syncthreads();
}</div>
<p>Consecutive threads touch consecutive words, so no bank conflicts, and the active threads are the lowest ids, so as <i>s</i> falls below 32 whole warps go idle together. Worked count for a 256-thread block: round 1 uses 128 threads = 4 full warps; round 2 uses 64 = 2 warps; round 3 uses 1 warp; after that a single warp with fewer and fewer lanes.</p>
<p>This rung needs the barrier: <code>__syncthreads()</code> must be reached by <b>all</b> threads, so it sits <i>outside</i> the <code>if (tid &lt; s)</code>. Putting it inside is a deadlock or undefined behaviour.</p>`,
        check: { q: "In the sequential-addressing loop, where must __syncthreads() go?",
          options: ["Inside if (tid < s), only the active threads need to sync", "Outside the if, so every thread of the block reaches it", "Once, after the loop"], answer: 1,
          why: "A barrier waits until every thread of the block arrives. If threads with tid >= s skip it, the active ones wait forever (deadlock) or behaviour is undefined. And it is needed after every round, because the next round reads sums that other threads just wrote." },
        scene(G) {
          G.text(24, 34, "rung 3: tid < s, s halves from 32  (64 threads)", { size: 13 });
          const rows = threadRows(G, 24, 74, [32, 16, 8, 4, 2, 1], (t, s) => t < s, (s) => "s=" + s);
          G.label(578, 74 + 13, "w1 idle", { size: 10, color: "ok" });
          G.label(24, 270, "after the first round warp 1 retires; the rest shrinks inside warp 0", { size: 12, color: "ok" });
          G.label(24, 294, "words tid and tid + s: consecutive, so no bank conflicts", { size: 12 });
          G.text(24, 336, "rung 2 (strided index) did this to the banks:", { size: 12 });
          G.label(24, 358, "s=1: 2-way   s=2: 4-way   s=4: 8-way   s=8: 16-way   (gcd(2s, 32) lanes per bank)", { size: 11, color: "hot" });
          G.from(rows, { opacity: 0, stagger: 0.12, duration: 0.3 });
          G.caption("contiguous active threads: whole warps leave, banks stay clean");
        } },

      { rail: "rungs 4-5", title: "Rungs 4 and 5: don't waste the first round, finish with shuffles",
        body: `<p><b>Rung 4: add two elements while loading.</b> In rung 3, half the threads do nothing in round 1 (they only loaded a value). Instead let each thread load <i>two</i> elements, add them in registers, and store the sum. One block now covers twice the data with the same number of threads, so you need half as many blocks and write half as many partial results.</p>
<p><b>Rung 5: warp shuffles for the tail.</b> When only one warp is left (s &le; 32) the data fits in 32 registers, one per lane. Lanes can read each other's registers directly with <code>__shfl_down_sync(mask, v, offset)</code>: lane <i>i</i> receives the value held by lane <i>i + offset</i>. No shared memory, no <code>__syncthreads()</code>:</p>
<div class="eq">for (int o = 16; o &gt; 0; o &gt;&gt;= 1)
  v += __shfl_down_sync(0xffffffffu, v, o);
// lane 0 now holds the sum of the warp's 32 values</div>
<p>8-lane toy with our values <code>[3 1 7 0 4 1 6 3]</code>: offset 4 gives lanes 0-3 the values <code>[7 2 13 3]</code>; offset 2 gives lane 0 <code>20</code> and lane 1 <code>5</code>; offset 1 gives lane 0 <code>25</code>. Five steps (offsets 16, 8, 4, 2, 1) handle a real warp. To combine the warps of a block, each warp's lane 0 writes its sum to one shared slot, one <code>__syncthreads()</code>, and the first warp shuffles those slots.</p>
<p><b>Why the mask?</b> Since Volta, lanes of a warp are <i>not guaranteed</i> to run in lockstep (independent thread scheduling). The old trick of unrolling the last steps with <code>volatile</code> shared memory and no barrier is therefore wrong. The <code>_sync</code> functions and <code>__syncwarp()</code> state which lanes take part and make the exchange safe.</p>`,
        check: { q: "Why is the old \"volatile shared memory, no barrier\" unrolled warp reduction unsafe on modern GPUs?",
          options: ["volatile is no longer supported", "Since Volta, lanes in a warp are not guaranteed to execute in lockstep, so you need __shfl_*_sync or __syncwarp()", "Shared memory is slower than registers"], answer: 1,
          why: "Independent thread scheduling lets lanes of one warp progress at different times. Code that silently assumed lockstep can read a neighbour's value before it is written. The _sync primitives name the participating lanes and guarantee the exchange." },
        scene(G) {
          G.text(24, 34, "__shfl_down_sync: lane i reads lane i + offset (8 lanes drawn)", { size: 12 });
          const lanes = [VALS.slice()];
          let cur = VALS.slice(), nxt;
          [4, 2, 1].forEach((o) => { nxt = cur.map((v, i) => (i + o < 8 ? v + cur[i + o] : v)); lanes.push(nxt); cur = nxt; });
          const info = [["start", ""], ["offset 4", "lanes 0-3"], ["offset 2", "lanes 0-1"], ["offset 1", "lane 0"]];
          const all = [];
          lanes.forEach((vals, L) => {
            const y = 56 + L * 80;
            G.label(24, y + 22, info[L][0], { size: 11 });
            vals.forEach((v, i) => {
              const useful = L === 0 || i < 8 / 2 ** L;
              all.push(G.box(110 + i * 62, y, 54, 32, String(v), { fill: L === 3 && i === 0 ? "v" : useful ? "k" : "line", size: 13, rx: 4 }));
              if (L > 0 && useful) { const o = [4, 2, 1][L - 1]; G.line(110 + (i + o) * 62 + 27, y - 48, 110 + i * 62 + 27, y, { color: "muted", w: 1, arrow: true }); }
            });
            G.label(110 + 8 * 62 + 4, y + 22, "", { size: 10 });
          });
          G.label(24, 392, "lane 0 ends with 25; real warps do 5 steps (16, 8, 4, 2, 1)", { size: 12, color: "v" });
          G.from(all, { opacity: 0, stagger: 0.02, duration: 0.2 });
          G.caption("values move between registers: no shared memory, no barrier");
        } },

      { rail: "rung 6", title: "Rung 6: many elements per thread, enough bytes in flight",
        body: `<p>After rung 5 the kernel is correct and tidy, but still too slow. The reason is not arithmetic: a DRAM access takes hundreds of cycles, and the memory system only reaches full bandwidth if <b>enough requests are in flight at once</b>. Little's law says it plainly: <code>bytes in flight = bandwidth x latency</code>. Example values: 320 GB/s (T4, gpu_specs.yaml, UNVERIFIED) and an assumed latency of 600 ns:</p>
<div class="eq">in flight = 320e9 B/s x 600e-9 s = 192,000 B
per SM (40 SMs) = 4,800 B
scalar 4 B loads : 1,200 per SM
float4 16 B loads:   300 per SM</div>
<p>A block that only loads one or two floats per thread, then spends a long time in the tree, leaves the memory idle. Rung 6 changes the shape of the kernel:</p>
<ul><li><b>Grid-stride loop</b> (P5.1): launch a GPU-sized grid and let each thread accumulate <i>many</i> elements in a register, so loads keep coming.</li>
<li><b><code>float4</code> loads</b> (P5.2): 16 bytes per lane, fewer instructions, more bytes outstanding per request.</li>
<li><b>One <code>atomicAdd</code> per block</b> for the final step, instead of writing a partial for every block and launching a second kernel.</li></ul>
<p><b>Predict-first answer.</b> At n = 2^26 floats: every rung reads 256 MiB. Rung 1 with 256-thread blocks also writes 2^26 / 256 = 262,144 partials (1 MiB) and a second kernel reads them back, about 0.8% extra. So rung 1 is <i>not</i> slower because it moves more bytes. It is slower because its work per byte is badly shaped (divergence, conflicts, idle lanes, few loads in flight). If it were bandwidth-bound it would tie with rung 6, and Speed of Light in ncu (P5.3) shows it is not near the memory peak.</p>
<p>Rung 6 and <code>cub::DeviceReduce::Sum</code> should both land near the read bandwidth. The L3 exit check is rung 6 within 10% of CUB. <code>d4/reduce.cuh</code> has all six rungs.</p>`,
        scene(G) {
          G.text(24, 34, "memory requests in flight (each box = one outstanding load)", { size: 12 });
          G.label(24, 62, "rungs 1-5: load a little, then think", { size: 12, color: "hot" });
          const few = strip(G, 24, 70, 40, 14, 22, (c) => (c < 6 || (c > 20 && c < 24) ? "k" : "line"), { gap: 1.5 });
          G.label(24, 108, "DRAM idle between bursts", { size: 11, color: "hot" });
          G.label(24, 150, "rung 6: grid-stride + float4: always loading", { size: 12, color: "ok" });
          const many = strip(G, 24, 158, 40, 14, 22, () => "ok", { gap: 1.5 });
          G.label(24, 196, "DRAM busy: bandwidth-bound, the best possible", { size: 11, color: "ok" });
          G.text(24, 250, "bytes in flight = bandwidth x latency", { size: 14 });
          G.label(24, 276, "320e9 B/s x 600e-9 s = 192,000 B  (example values, UNVERIFIED)", { size: 12 });
          G.label(24, 300, "per SM: 4,800 B = 1,200 scalar or 300 float4 loads", { size: 12 });
          G.label(24, 344, "bytes moved at n = 2^26: rung 1 ~258 MiB, rung 6 256 MiB", { size: 12, color: "v" });
          G.from([few, many], { opacity: 0, stagger: 0.3, duration: 0.4 });
          G.caption("same bytes, different overlap: speed is set by loads in flight");
        } },

      { rail: "float order", title: "Why the last digit changes: floating-point order",
        body: `<p>The tree adds numbers in a different order than the serial loop. On paper that is fine. In floating point it is not, because <b>float addition is not associative</b>: every add rounds to the nearest representable value. A float32 has a 24-bit significand, so above 2^24 = 16,777,216 it can no longer represent every integer:</p>
<div class="eq">a = 16,777,216 (= 2^24), b = 1, c = 1

(a + b) + c = 16,777,216 + 1 = 16,777,216 (rounds)
              16,777,216 + 1 = 16,777,216
a + (b + c) = 16,777,216 + 2 = 16,777,218 (exact)</div>
<p>The two orders differ by 2. In a real sum of millions of values the differences are tiny, around the last digits, but they exist. Consequences:</p>
<ul><li>A GPU reduction will not match a CPU loop bit for bit. Compare with a <b>tolerance scaled to the magnitude</b> of the data, as the test does (against a double-precision reference).</li>
<li>With <code>atomicAdd</code> from many blocks, the <i>order</i> in which blocks finish varies from run to run, so the result can differ between two runs of the same program. That is expected, not a bug.</li>
<li>If you need the same bits every run (reproducible training, debugging), use the deterministic two-pass version: fixed partial per block, then a second fixed-order reduction.</li></ul>
<p>The same effect is why softmax and norm kernels (P5.5) accumulate in float32 even when the data is half precision.</p>`,
        check: { q: "Two runs of the atomic-add reduction on the same data differ in the 7th significant digit. Is it a bug?",
          options: ["Yes: a race condition corrupts the sum", "No: float addition is not associative and block completion order varies between runs", "Yes: atomicAdd on floats is not supported"], answer: 1,
          why: "atomicAdd is correct (no update is lost), but the order in which blocks add into the total is not fixed, and different orders round differently. Use the two-pass deterministic version if you need identical bits." },
        scene(G) {
          G.text(24, 34, "a = 2^24, b = 1, c = 1 in float32", { size: 14 });
          G.text(24, 80, "(a + b) + c", { size: 13 });
          G.box(24, 94, 150, 34, "a + b = 2^24", { stroke: "hot", color: "hot", size: 12, rx: 5 });
          G.label(180, 116, "1 is lost to rounding", { size: 11, color: "hot" });
          G.box(24, 138, 150, 34, "+ c = 2^24", { fill: "hot", size: 12, rx: 5 });
          G.label(180, 160, "= 16,777,216", { size: 12, color: "hot" });
          G.text(340, 80, "a + (b + c)", { size: 13 });
          G.box(340, 94, 150, 34, "b + c = 2", { stroke: "ok", color: "ok", size: 12, rx: 5 });
          G.box(340, 138, 150, 34, "a + 2 = 2^24 + 2", { fill: "ok", size: 12, rx: 5 });
          G.label(340, 192, "= 16,777,218  (exact)", { size: 12, color: "ok" });
          G.text(24, 250, "different order, different rounding", { size: 14, color: "v" });
          G.label(24, 280, "-> never compare with ==; use a magnitude-scaled tolerance", { size: 12 });
          G.label(24, 304, "-> atomics: block order varies, so bits can vary per run", { size: 12 });
          G.label(24, 328, "-> need exact repeatability? deterministic two-pass", { size: 12 });
          G.caption("the tree is a different sum order, so a different rounding");
        } },

      { rail: "scan", title: "Scan: every running total at once",
        body: `<p>A reduction gives one number. A <b>scan</b> (prefix sum) gives <i>all</i> the running totals: output element <i>i</i> is the sum of the inputs up to <i>i</i>.</p>
<div class="eq">in             : 3  1  7  0  4  1  6  3
inclusive scan : 3  4 11 11 15 16 22 25
exclusive scan : 0  3  4 11 11 15 16 22</div>
<p><b>Inclusive</b> includes element <i>i</i> itself; <b>exclusive</b> sums only the elements before it (the first output is the identity, 0 for add). The last inclusive value equals the reduction (25).</p>
<p>Why care? A scan converts a <i>sequential-looking</i> question ("where does my output go?") into something every thread can answer at once:</p>
<ul><li><b>Stream compaction:</b> keep only elements that pass a test. Flags <code>1 0 1 1 0</code> exclusively scanned give each kept element its output slot.</li>
<li><b>Variable-size outputs:</b> how many tokens each sequence in a batch contributed, so the offset of sequence <i>j</i> is the exclusive scan of the lengths (batching, P2).</li>
<li><b>Sampling:</b> a cumulative probability distribution is a scan of the probabilities; top-p sampling cuts it at 0.9.</li>
<li>Radix sort, histograms and many tree algorithms.</li></ul>
<p>The serial scan needs n sequential steps again. The next steps show how to parallelize it, which is harder than the reduction because every output needs a different partial sum, and the algorithms trade <b>work</b> (total additions) against <b>depth</b> (rounds that must happen one after another).</p>`,
        scene(G) {
          const inc = []; VALS.reduce((a, v, i) => (inc[i] = a + v), 0);
          const exc = inc.map((v, i) => v - VALS[i]);
          const rowFn = (y, name, vals, col) => {
            G.label(24, y + 22, name, { size: 11 });
            return vals.map((v, i) => G.box(150 + i * 56, y, 50, 34, String(v), { fill: col, size: 14, rx: 4 }));
          };
          const a = rowFn(60, "input", VALS, "k");
          const b = rowFn(140, "inclusive", inc, "v");
          const c = rowFn(220, "exclusive", exc, "blue");
          G.label(24, 300, "inclusive[i] = in[0] + ... + in[i]", { size: 12 });
          G.label(24, 322, "exclusive[i] = in[0] + ... + in[i-1]", { size: 12 });
          G.label(24, 352, "last inclusive value = the reduction = 25", { size: 12, color: "v" });
          G.from([a, b, c], { opacity: 0, y: 8, stagger: 0.25, duration: 0.3 });
          G.caption("a scan keeps every running total, not just the last");
        } },

      { rail: "work vs depth", title: "Two scans: Hillis-Steele and Blelloch",
        body: `<p>Two classic parallel scans show the central trade-off.</p>
<p><b>Hillis-Steele.</b> In round <i>d</i> = 1, 2, 4, ..., every element adds the element <i>d</i> places to its left (if it exists). After log2(n) rounds every element holds its inclusive total:</p>
<div class="eq">in   : 3  1  7  0  4  1  6  3
d=1  : 3  4  8  7  4  5  7  9
d=2  : 3  4 11 11 12 12 11 14
d=4  : 3  4 11 11 15 16 22 25   (inclusive scan)</div>
<p>Depth log2(n), but <b>work</b> (additions) is n - 1, n - 2, n - 4, ... summed: for n = 8 that is 7 + 6 + 4 = 17, in general <code>n log2 n - n + 1</code>. Every thread stays busy and needs no complicated indexing, but it does about log n times more additions than the serial loop.</p>
<p><b>Blelloch (up-sweep, down-sweep).</b> First do the reduction tree (the up-sweep, n - 1 adds), then walk back down the tree distributing the partial sums (the down-sweep, another n - 1 adds). Work is about 2(n - 1): <b>O(n)</b>, "work-efficient", at the price of depth 2 log2 n.</p>
<div class="eq">n = 1024:
  Hillis-Steele: 1024 x 10 - 1023 = 9,217 adds
  Blelloch     : 2 x 1023         = 2,046 adds
  serial       : 1023 adds (but depth 1023)</div>
<p>Which wins depends on scale. Within a <b>warp</b> (n = 32) Hillis-Steele with shuffles is five steps and no shared memory or barriers, so its extra work is cheap and its low depth wins. Across a <b>block</b>, work-efficiency starts to matter. What D4 does: each thread scans 8 items serially in registers (cheap, n/8 threads needed), the 256 thread totals are scanned with warp shuffles, and the result is added back.</p>`,
        check: { q: "Hillis-Steele does about n log2 n additions, much more than n - 1. Why is it still used inside a warp?",
          options: ["It is always the fastest scan", "At 32 lanes it takes 5 shuffle steps with no shared memory or barriers, so low depth beats work-efficiency", "It needs less memory than Blelloch at any size"], answer: 1,
          why: "Work-efficiency matters when extra additions cost real time. Inside a warp the additions are nearly free and the 5 steps run in registers without barriers, so depth dominates. Across larger scales the extra log n factor stops being free." },
        scene(G) {
          G.text(24, 34, "Hillis-Steele on 8 values (each row = one round)", { size: 13 });
          const rows = [VALS.slice()];
          [1, 2, 4].forEach((d) => { const p = rows[rows.length - 1]; rows.push(p.map((v, i) => (i - d >= 0 ? v + p[i - d] : v))); });
          const names = ["in", "d=1", "d=2", "d=4"], all = [];
          rows.forEach((vals, L) => {
            const y = 54 + L * 62;
            G.label(24, y + 20, names[L], { size: 11 });
            vals.forEach((v, i) => {
              const d = L ? [1, 2, 4][L - 1] : 0;
              all.push(G.box(80 + i * 62, y, 54, 30, String(v), { fill: L === 0 ? "k" : i - d >= 0 ? "blue" : "line", size: 13, rx: 4 }));
              if (L && i - d >= 0) G.line(80 + (i - d) * 62 + 27, y - 32, 80 + i * 62 + 27, y, { color: "muted", w: 1 });
            });
          });
          G.label(24, 320, "adds:  n=8: 7 + 6 + 4 = 17  (Blelloch ~ 14)", { size: 12 });
          const w = (v) => v * 0.055;
          G.label(24, 346, "n = 1024  HS", { size: 11 }); G.rect(130, 334, w(9217) , 14, { fill: "hot", rx: 2 });
          G.label(24, 368, "          Blelloch", { size: 11 }); G.rect(130, 356, w(2046), 14, { fill: "ok", rx: 2 });
          G.label(130 + w(9217) + 6, 346, "9,217", { size: 11 }); G.label(130 + w(2046) + 6, 368, "2,046", { size: 11 });
          G.from(all, { opacity: 0, stagger: 0.015, duration: 0.2 });
          G.caption("every element adds its left neighbour at distance d: log n rounds");
        } },

      { rail: "large n", title: "Scan for any n: scan-then-propagate",
        body: `<p>A block can scan one <b>tile</b> (D4 uses 2,048 elements). Blocks cannot talk to each other during a kernel, so how do you scan an array of 2^26 elements? The <b>scan-then-propagate</b> recipe uses three phases, each a kernel launch:</p>
<ol><li>Every block scans its own tile and writes the tile's <b>total</b> to a small array.</li>
<li>Scan the array of tile totals (small; recurse with the same function).</li>
<li>Every block adds its tile's <b>carry-in</b> (the scanned total of all earlier tiles) to each of its elements.</li></ol>
<div class="eq">tiles of 4, all inputs = 1:
phase 1 tile scans : 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4
tile totals        : 4 4 4 4
phase 2 exclusive  : 0 4 8 12   (carry-in per tile)
phase 3 add carry  : 1..4 | 5..8 | 9..12 | 13..16</div>
<p>It works, and it is <b>O(n)</b> work, but count the memory traffic. Phase 1 reads n and writes n; phase 3 reads those n again and writes n again: about <b>4n</b> element accesses, plus the small totals. Since scan is memory-bound (P5.2), traffic is time: 4n takes about twice as long as the ideal 2n (read the input once, write the output once). The next step shows how to reach 2n.</p>`,
        scene(G) {
          G.text(24, 34, "scan-then-propagate (tiles of 4, inputs all 1)", { size: 13 });
          const cw = 33, all = [];
          const rowTiles = (y, name, f, col) => {
            G.label(24, y - 6, name, { size: 11 });
            for (let t = 0; t < 4; t++) for (let i = 0; i < 4; i++) all.push(G.box(24 + t * 150 + i * (cw + 3), y, cw, 28, String(f(t, i)), { fill: col(t), size: 12, rx: 3 }));
          };
          const tc = (t) => ["k", "blue", "q", "pink"][t];
          rowTiles(70, "1  scan each tile (read n, write n)", (t, i) => i + 1, tc);
          for (let t = 0; t < 4; t++) { all.push(G.box(24 + t * 150, 134, 138, 26, "total " + 4, { stroke: tc(t), color: tc(t), size: 11, rx: 4 })); }
          G.label(24, 128, "   totals  ->  2  scan the totals: carry-in 0, 4, 8, 12", { size: 11 });
          for (let t = 0; t < 4; t++) all.push(G.box(24 + t * 150, 190, 138, 26, "carry " + 4 * t, { fill: tc(t), size: 11, rx: 4 }));
          rowTiles(262, "3  add carry-in (read n, write n)", (t, i) => 4 * t + i + 1, tc);
          G.label(24, 330, "traffic: n + n  then  n + n  =  4n  (ideal: 2n)", { size: 13, color: "hot" });
          G.label(24, 354, "3 kernel launches, the data crosses DRAM twice", { size: 12 });
          G.from(all, { opacity: 0, stagger: 0.01, duration: 0.15 });
          G.caption("correct and O(n), but every element is read and written twice");
        } },

      { rail: "look-back", title: "Single pass: decoupled look-back",
        body: `<p>Why do tiles wait for phase 2? Because tile <i>j</i> needs the total of tiles 0 to <i>j-1</i>. <b>Decoupled look-back</b> (Merrill and Garland, 2016; the algorithm in CUB) lets tiles <i>find that out while the kernel runs</i>, in one pass over the data (read n, write n: 2n).</p>
<p>Each tile has a <b>status word</b> in global memory holding a flag and a value:</p>
<ul><li><b>X</b>: not ready yet.</li>
<li><b>A</b> (aggregate): the tile's own total is known, its prefix is not.</li>
<li><b>P</b> (prefix): the sum of everything up to and including this tile is known.</li></ul>
<p>A tile scans its data, immediately publishes <b>A</b> with its total, then <b>looks back</b> at its predecessors: if tile <i>j-1</i> shows P, add that value and stop; if it shows A, add it and look one tile further; if X, wait (spin). Once it has the sum of all earlier tiles it publishes <b>P</b> for itself, adds the carry to its elements and writes them. Tile 0 has nothing before it and publishes P at once, so the chain of Ps starts and later tiles rarely need to look back far.</p>
<p>Two details that make it work (and make it hard to get right):</p>
<ul><li><b>Tile ids come from an atomic counter</b>, not <code>blockIdx</code>. A tile spins waiting for earlier tiles, so those must already be running. With <code>blockIdx</code> a waiting resident block could depend on a block that the hardware has not scheduled yet: deadlock. The counter hands out ids in the order tiles actually start.</li>
<li><b>Flag and value are written atomically</b> as one 64-bit word, so a reader never sees a flag that says "ready" with an old value.</li></ul>
<p><code>d4/scan.cuh</code> implements it with comments, and the test runs it repeatedly, because races appear as occasional wrong results.</p>`,
        check: { q: "Why must decoupled look-back assign tile ids with an atomic counter instead of using blockIdx?",
          options: ["Atomics are faster than reading blockIdx", "A tile spins waiting for its predecessors, so each predecessor must have already started; blockIdx order does not guarantee that", "blockIdx is not available in a scan kernel"], answer: 1,
          why: "If a resident block waits on a lower-numbered block that the hardware has not scheduled yet (every SM slot is full of waiters), nobody makes progress: deadlock. The atomic counter gives ids in actual start order, so everything a tile waits for has begun." },
        scene(G) {
          G.text(24, 28, "tiles publish status words; tile 4 looks back", { size: 13 });
          const st = ["P", "P", "A", "A", "?", "X"], val = ["4", "8", "4", "4", "", ""];
          const col = { P: "ok", A: "v", X: "line", "?": "k" };
          const all = [];
          for (let t = 0; t < 6; t++) {
            const x = 24 + t * 100;
            all.push(G.box(x, 110, 90, 40, "tile " + t, { fill: col[st[t]], size: 12, rx: 5 }));
            G.label(x + 4, 170, t === 4 ? "(me)" : st[t] === "X" ? "X not ready" : st[t] === "A" ? "A aggregate " + val[t] : "P prefix " + val[t], { size: 11, color: t === 4 ? "k" : "ink" });
          }
          G.path("M 394 106 C 394 70, 350 70, 336 106", { color: "k", w: 2, arrow: true });
          G.path("M 394 106 C 394 60, 250 60, 236 106", { color: "k", w: 2, arrow: true });
          G.path("M 394 106 C 394 50, 150 50, 136 106", { color: "k", w: 2, arrow: true });
          G.label(24, 200, "tile 4: sees tile 3 = A (4)  ->  add, look further", { size: 12 });
          G.label(24, 222, "         tile 2 = A (4)  ->  add, look further", { size: 12 });
          G.label(24, 244, "         tile 1 = P (8)  ->  add, stop", { size: 12 });
          G.text(24, 286, "carry-in = 4 + 4 + 8 = 16", { size: 14, color: "v" });
          G.label(24, 312, "then publish P = 16 + own total", { size: 12 });
          G.label(24, 350, "one pass: read n, write n = 2n   (vs 4n for 3 phases)", { size: 13, color: "ok" });
          G.from(all, { opacity: 0, y: 8, stagger: 0.05, duration: 0.25 });
          G.caption("a tile reads predecessors' flags until it meets a prefix");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A reduction is a tree: log2(n) rounds, n - 1 additions, memory-bound, so the ceiling is read bandwidth.</li>
<li>The ladder: interleaved tree (divergence), strided index (bank conflicts), sequential addressing, two elements per thread, warp shuffles for the tail, then grid-stride + float4 + one atomic per block for enough bytes in flight.</li>
<li>Since Volta use <code>__shfl_*_sync</code> and <code>__syncwarp()</code>; never rely on warp lockstep. <code>__syncthreads()</code> must be reached by every thread.</li>
<li>Float addition is not associative: tree and atomic orders change the last digits; test with tolerances or use a deterministic two-pass version.</li>
<li>A scan keeps every running total. Hillis-Steele has depth log n but n log n work; Blelloch has O(n) work; scan-then-propagate costs 4n traffic; decoupled look-back reaches 2n in one pass using an atomic tile counter and per-tile status words.</li></ul>`,
    sim: {
      title: "Reduction and scan cost calculator",
      intro: "Pick the array size and block shape. The reduction panel counts rounds, partial results and global bytes for rung 1 (one element per thread, partials plus a second kernel) and rung 6 (one atomic per block), then the lower-bound time at a bandwidth peak. The scan panel compares the work of three algorithms and the traffic of the two large-n strategies. Peak bandwidth is an example value (T4: 320 GB/s, gpu_specs.yaml, UNVERIFIED).",
      height: 300,
      controls: [
        { id: "lg", label: "n = 2^k, k", min: 10, max: 28, step: 1, value: 26 },
        { id: "b", label: "threads per block", min: 64, max: 1024, step: 64, value: 256 },
        { id: "peak", label: "peak bandwidth, GB/s (example)", min: 50, max: 1000, step: 10, value: 320 },
        { id: "sm", label: "SMs (rung 6 launches 8 blocks per SM)", min: 1, max: 160, step: 1, value: 40 },
      ],
      draw(G, v) {
        const n = 2 ** v.lg, B = v.b;
        const rounds = v.lg;
        const partials1 = Math.ceil(n / B), blocks6 = v.sm * 8;
        const bytes1 = 4 * n + 4 * partials1 * 2, bytes6 = 4 * n + 4 * blocks6;
        const t1 = bytes1 / (v.peak * 1e9), t6 = bytes6 / (v.peak * 1e9);
        const hs = n * Math.log2(n) - n + 1, bl = 2 * (n - 1), ser = n - 1;
        G.label(10, 14, "scan work (additions), relative to the serial n - 1", { size: 11 });
        const mx = Math.max(hs / ser, bl / ser, 1), sc = 460 / mx;
        [["serial", ser, "line"], ["Blelloch", bl, "ok"], ["Hillis-Steele", hs, "hot"]].forEach(([nm, w, c], i) => {
          const y = 28 + i * 40;
          G.text(10, y + 17, nm, { size: 12 });
          G.rect(120, y + 2, Math.max(2, (w / ser) * sc), 22, { fill: c, rx: 3 });
          G.text(126 + (w / ser) * sc, y + 18, F.num(w / ser, 2) + "x", { size: 12 });
        });
        G.label(10, 168, "scan memory traffic for large n (element accesses)", { size: 11 });
        [["3-phase", 4, "v"], ["look-back", 2, "ok"]].forEach(([nm, w, c], i) => {
          const y = 180 + i * 36;
          G.text(10, y + 17, nm, { size: 12 });
          G.rect(120, y + 2, w * 100, 22, { fill: c, rx: 3 });
          G.text(126 + w * 100, y + 18, w + "n", { size: 12 });
        });
        G.label(10, 270, `n = ${F.num(n)} elements = ${F.bytes(4 * n)}`, { size: 11 });
        return [
          { title: "Reduction tree", rows: [["elements", F.num(n)], ["rounds (log2 n)", String(rounds)], ["additions (n - 1)", F.num(n - 1)]] },
          { title: "Rung 1 (partials + 2nd kernel)", rows: [["partials", F.num(partials1)], ["bytes moved", F.bytes(bytes1)], ["extra vs read-once", F.num((100 * (bytes1 - 4 * n)) / (4 * n), 2) + "%"], ["time >=", F.ms(t1)]] },
          { title: "Rung 6 (one atomic per block)", rows: [["blocks", F.num(blocks6)], ["bytes moved", F.bytes(bytes6)], ["time >=", F.ms(t6)]],
            html: `<p class="note">Lower bounds at the example peak. Real rung 1 is slower than its bound because of divergence, conflicts and too few loads in flight; rung 6 should approach it.</p>` },
          { title: "Scan", rows: [["Hillis-Steele adds", F.si(hs, 2)], ["Blelloch adds", F.si(bl, 2)], ["traffic 3-phase", "4n = " + F.bytes(16 * n)], ["traffic look-back", "2n = " + F.bytes(8 * n)]] },
        ];
      },
      note: "Models: additions counted as n - 1, n log2 n - n + 1 and 2(n - 1). Scan traffic counts 4-byte reads and writes of the full array only (totals arrays ignored). Times assume the reduction is DRAM-read-bound.",
    },
    practice: {
      intro: "Build once from the repository root (see the shared P5 page): <code>cmake -S course/P5-cuda-deep-dive -B build/p5 -DCMAKE_CUDA_ARCHITECTURES=native &amp;&amp; cmake --build build/p5 -j</code>. Run the three examples first; they print GB/s per rung, three warp reductions that must agree, and 3-phase vs look-back vs CUB scans.",
      items: [
        { title: "Climb the reduction ladder", tier: "T2 · medium", goal: "Rungs 1 and 2 are given in <b>kernel.cuh</b>. Add rungs 3 to 6 behind the same <b>reduce_sum(rung, ...)</b> launcher. The test checks every rung on awkward sizes (1, 300, 513, 2^22 + 5) against a double-precision sum with a magnitude-scaled tolerance. Explain each jump with one ncu metric (bank conflicts for rungs 2 and 3).",
          cmd: "ctest --test-dir build/p5 -R p5.4_01-reduce-ladder --output-on-failure && ./build/p5/p5.4_01-reduce-ladder --bench" },
        { title: "Reduction within 10% of CUB (L3 exit check)", tier: "T2 · hard", goal: "<b>--bench</b> prints each rung's GB/s at 2^24, 2^26 and 2^28, then <b>rung 6 / cub</b> and PASS if it is within 10% of <b>cub::DeviceReduce::Sum</b>. Record the table in the module README.",
          cmd: "./build/p5/p5.4_01-reduce-ladder --bench" },
        { title: "Block scan and scan-then-propagate", tier: "T2 · hard", goal: "Implement <b>inclusive_scan(in, out, n)</b>: scan a 2,048-element tile (8 serial items per thread, shuffle scan of the 256 thread totals), then handle any n by scanning tile totals and adding carry-ins. Optional: decoupled look-back, compared with d4/scan.cuh. The test is bit-exact (small integers stored as floats).",
          cmd: "ctest --test-dir build/p5 -R p5.4_03-scan --output-on-failure && ./build/p5/p5.4_03-scan --bench" },
        { title: "Top-k by per-block selection and merge", tier: "T2 · hard", goal: "Brief in <b>exercises/04-topk.md</b>: select the k largest per block, then merge the per-block candidates. Report a bench table.",
          cmd: "cat course/P5-cuda-deep-dive/P5.4-reductions-and-scans/exercises/04-topk.md" },
      ],
      labs: [
        { label: "Examples: reduction ladder vs CUB, three warp reductions, scans", path: "course/P5-cuda-deep-dive/P5.4-reductions-and-scans/examples/" },
        { label: "AWS walkthrough (g4dn.xlarge, about 2 instance-hours; cost, auto-stop and teardown on the shared page)", path: "course/P5-cuda-deep-dive/P5.4-reductions-and-scans/aws.md" },
        { label: "Shared P5 setup: build, Nsight counter access, teardown", path: "course/P5-cuda-deep-dive/aws-common.md" },
        { label: "D4 reduction ladder (6 rungs)", path: "platform/kernels/include/d4/reduce.cuh" },
        { label: "D4 scan with decoupled look-back", path: "platform/kernels/include/d4/scan.cuh" },
        { label: "Standalone animation: tree vs shuffle reduction", path: "animations/p5-reduction.html" },
      ],
    },
  });
})();
