/* L3 — Reductions and scans (Lane B). Concepts behind the L3 LeetGPU problems: tree reduction, sequential addressing,
   per-thread accumulation, warp shuffles, block reduce + one atomic, scan (Hillis-Steele, Blelloch, multi-tile),
   stream compaction, bitonic sort and top-k tournaments, CSR SpMV. */
(function () {
  const F = S2S.fmt;
  const LV = "laneB-cuda/L3-reductions-and-scans";
  const mono = (h) => h.replace(/<code>(.*?)<\/code>/g, '<span style="font-family:var(--f-mono);font-size:13px">$1</span>');
  const P = (slug, title, diff, prio, goal) => ({
    title: `#${slug.split("-")[0]} ${title}`, tier: `${diff} · ${prio}`,
    goal: mono(`${goal} Hints and reference solution: <code>${LV}/solutions/${slug}/</code>`),
    cmd: `ctest --test-dir build/laneB -R '^L3_${slug}$' --output-on-failure`,
  });
  // a row of numeric boxes; vals may contain null (drawn as a faint dot); hl(i) gives a colour for a highlighted cell
  const nums = (G, x, y, vals, cw, ch, hl, base = "k") => vals.map((v, i) => {
    const c = hl ? hl(i) : null;
    return v === null ? G.box(x + i * (cw + 4), y, cw, ch, "·", { fill: "line", size: 12, boxOpacity: 0.4 })
      : G.box(x + i * (cw + 4), y, cw, ch, String(v), { fill: c || base, size: 12, boxOpacity: c ? 1 : 0.5 });
  });

  S2S.lesson({
    id: "l3", n: "L3", title: "Reductions and scans",
    subtitle: "Lane B · CUDA on LeetGPU · T1 browser or T2 harness",
    kicker: "Lesson · ≈ 70 min",
    headline: "Many numbers in, one (or a running total) out",
    intro: `<p>So far each thread owned its own output. A <b>reduction</b> breaks that rule: N inputs become one number (a sum, a maximum, a count). A <b>scan</b> keeps N outputs, but output i depends on every input before it. Threads must cooperate, and cooperation is where GPU programs get slow or wrong.</p>
<p>The dark panel climbs the standard ladder: tree, shared-memory tree, per-thread accumulation, warp shuffles, block reduce with one atomic, then scans (two algorithms and the multi-block version), and the things scans build: stream compaction, sorting and top-k, and a sparse matrix-vector product. The Simulate tab shows where the additions of a big reduction actually happen.</p>`,
    facts: ["11 steps", "5 checkpoints", "1 simulator", "7 core problems"],
    legend: [["k", "value / thread"], ["v", "just updated"], ["hot", "invalid or wasted"], ["ok", "result"]],
    prev: "l2", next: "l4",
    steps: [
      { rail: "tree", title: "A sum is a tree, not a chain",
        body: `<p>Summing N numbers on a CPU is a chain: <code>s = s + x[i]</code>, N − 1 steps, each waiting for the one before. On a GPU we want thousands of threads busy. The way out is a property of addition: it is <b>associative</b>, <code>(a + b) + c = a + (b + c)</code>, so we may add in any grouping.</p>
<p>Group the numbers in pairs, add each pair at the same time, then pair up the results, and repeat. Eight numbers take 3 levels instead of 7 steps:</p>
<div class="eq">[3 1 4 1 5 9 2 6]   8 values
[4 5 14 8]           level 1: 4 adds at once
[9 22]               level 2: 2 adds at once
[31]                 level 3: 1 add</div>
<p>The number of levels is log₂ N. For N = 2<sup>24</sup> (16.8 million values) that is 24 levels, though in practice we do most of the adding serially in each thread first (step 3). The total number of additions is still N − 1. The tree does not save work, it saves <i>time</i>, by making each level parallel.</p>
<p>Anything associative works the same way: max, min, product, count, bitwise or. Floating-point addition is only <i>almost</i> associative (step 5).</p>`,
        scene(G) {
          const lv = [[3, 1, 4, 1, 5, 9, 2, 6], [4, 5, 14, 8], [9, 22], [31]];
          const cw = 52, step = 70, x0 = 24;
          lv.forEach((vals, L) => {
            const span = Math.pow(2, L), y = 64 + L * 78;
            vals.forEach((v, i) => {
              const cx = x0 + (i * span + (span - 1) / 2) * step;
              G.box(cx, y, cw, 36, String(v), { fill: L === 3 ? "ok" : "k", size: 14, boxOpacity: 0.55 + 0.15 * L });
              if (L > 0) { for (const j of [2 * i, 2 * i + 1]) { const pspan = span / 2, pcx = x0 + (j * pspan + (pspan - 1) / 2) * step; G.line(pcx + cw / 2, y - 42, cx + cw / 2, y, { color: "muted", w: 1 }); } }
            });
            G.label(594, y + 22, ["input", "L1", "L2", "L3"][L], { size: 11 });
          });
          G.text(24, 372, "serial chain: 7 dependent steps", { size: 14, color: "hot" });
          G.text(24, 398, "tree: 3 levels, each level fully parallel", { size: 14, color: "ok" });
          G.caption("same 7 additions, regrouped so each level runs at once");
        } },

      { rail: "shared tree", title: "In shared memory: halve the stride each step",
        body: `<p>A block of 256 threads loads 256 values into shared memory and reduces them in place. Step 1: thread t adds the value 128 slots away into its own slot. Now the first 128 slots hold pair sums. Step 2: stride 64. And so on down to stride 1, when slot 0 holds the block's sum: 8 steps, a <code>__syncthreads()</code> after each (L2 explained why).</p>
<div class="eq">for (int s = blockDim.x / 2; s &gt; 0; s &gt;&gt;= 1) {
  if (t &lt; s) sm[t] += sm[t + s];
  __syncthreads();
}</div>
<p>The address pattern matters. The tempting textbook version has thread t work when <code>t % (2 * s) == 0</code> and add <code>sm[t + s]</code>. It gives the same sum, but the active threads are scattered across <i>every</i> warp, so every warp diverges and runs at a fraction of its lanes. Harris's reduction ladder (Sources) measures this. With <code>t &lt; s</code>, the active threads are always the first s, so whole warps are either fully busy or fully idle (and finished warps cost nothing). Reads also stay in consecutive banks (no conflicts, as in L2).</p>
<p>Note the active thread count halves every step: 128, 64, 32, 16, 8, 4, 2, 1. Half of the block is idle after step 1. That is the next problem.</p>`,
        check: { q: "In the shared-memory tree, why use the guard if (t < s) rather than if (t % (2*s) == 0) for the thread that does sm[t] += sm[t + s]?",
          options: ["The first form adds fewer numbers", "The first keeps the active threads together, so whole warps are busy or idle; the second leaves a few active lanes in every warp (divergence)", "The second form is not associative"], answer: 1,
          why: "Both forms perform the same additions. With t < s, threads 0 … s−1 work: the active set is a prefix, so entire warps drop out as s shrinks and the others have no divergence. With t % (2s) == 0, active threads are spread out (every 2nd, 4th, 8th thread), so every warp keeps issuing instructions with most lanes masked off." },
        scene(G) {
          const n = 16, cw = 32, x0 = 28;
          G.text(24, 32, "16 slots; thread t adds sm[t + s] into sm[t]", { size: 13 });
          [8, 4, 2, 1].forEach((s, r) => {
            const y = 80 + r * 84;
            G.label(x0, y - 24, `step ${r + 1}: s = ${s} → threads 0 … ${s - 1} active`, { color: "ink", size: 12 });
            for (let i = 0; i < n; i++) G.box(x0 + i * 36, y, cw, 30, "", { fill: i < s ? "k" : "line", boxOpacity: i < s ? 0.9 : 0.35 });
            for (let t = 0; t < s; t++) { const xa = x0 + (t + s) * 36 + 16, xb = x0 + t * 36 + 16; G.path(`M ${xa} ${y} C ${xa} ${y - 22}, ${xb} ${y - 22}, ${xb} ${y - 2}`, { color: "v", w: 1, arrow: true }); }
          });
          G.text(24, 412, "active threads: always a prefix → whole warps retire", { size: 13, color: "ok" });
          G.caption("each step halves the active prefix; slot 0 ends with the sum");
        } },

      { rail: "per-thread work", title: "Let every thread add many values before anyone cooperates",
        body: `<p>If one thread loads one value, half the block idles after the first step, and a million-element sum needs thousands of blocks, each writing a partial result that a second kernel must add. Both costs vanish if each thread first accumulates many inputs in a <b>register</b>, using the grid-stride loop from L1:</p>
<div class="eq">float acc = 0;
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i &lt; n; i += gridDim.x * blockDim.x)
  acc += in[i];     // register-serial</div>
<p>Example: N = 2<sup>24</sup>, 320 blocks of 256 threads (the reference uses 8 × the SM count, here 40 SMs as on a T4, an example value) = 81,920 threads. Each thread adds about 16,777,216 / 81,920 = 204.8 numbers. Only then do the 256 per-thread totals in a block cooperate. Of the N − 1 additions, all but 81,919 happen serially inside threads.</p>
<p>Two payoffs. The kernel now reads each element once, in coalesced order, so it is <b>memory-bound</b>: time ≈ 4 N bytes ÷ bandwidth. For 2<sup>24</sup> floats that is 67.1 MB, or 0.22 ms at 300 GB/s (example value, check your GPU's spec sheet). The reference solution also uses <code>float4</code> loads (16 bytes per lane), which cuts the number of load instructions by 4 and keeps lanes busy.</p>
<p>That is the exit check of this level: your sum must come within 10% of <code>cub::DeviceReduce::Sum</code>, which is bound by the same bandwidth.</p>`,
        scene(G) {
          const T = 8, N = 40, colr = ["k", "v", "q", "ok", "blue", "pink", "hot", "w"];
          G.text(24, 32, "8 threads, grid-stride: thread t sums t, t+8, t+16 …", { size: 13 });
          const cells = Array.from({ length: N }, (_, i) => G.box(24 + (i % 20) * 30, 56 + Math.floor(i / 20) * 34, 26, 28, String(i), { fill: colr[i % T], size: 10, boxOpacity: 0.8 }));
          G.label(24, 134, "input elements, coloured by the thread that adds them", { size: 11 });
          const acc = Array.from({ length: T }, (_, t) => G.box(24 + t * 74, 168, 66, 40, `acc ${t}`, { fill: colr[t], size: 12 }));
          for (let t = 0; t < T; t++) G.arrow(57 + t * 74, 142, 57 + t * 74, 164, { color: colr[t] });
          G.label(24, 242, "each acc is a register: no sync, no memory traffic", { size: 11, color: "ink" });
          G.text(24, 306, "then only 8 values need the tree", { size: 14, color: "v" });
          for (let t = 0; t < T; t++) G.arrow(57 + t * 74, 252, 57 + t * 74, 276, { color: "muted", dash: "3 3" });
          
          G.text(24, 340, "2^24 values, 81,920 threads: 204.8 adds per thread", { size: 13, color: "ink" });
          G.text(24, 366, "cooperative adds: 81,919 of 16,777,215 (0.5%)", { size: 13, color: "ok" });
          G.text(24, 392, "→ time ≈ bytes / bandwidth", { size: 14, color: "v" });
          G.from(cells, { opacity: 0, stagger: 0.012, duration: 0.15 });
          G.caption("most of the sum happens in registers, in coalesced order");
        } },

      { rail: "warp shuffle", title: "Warp shuffles: swap registers without shared memory",
        body: `<p>Inside a warp the 32 lanes already run in lock-step, so they should not need shared memory and barriers to talk. They don't. <code>__shfl_down_sync(mask, v, d)</code> returns the value of <code>v</code> held by the lane <i>d positions above</i> mine. It is a register-to-register copy across lanes, in one instruction. (<code>mask = 0xffffffff</code> says all 32 lanes take part; the <code>_sync</code> variants are required on current GPUs.)</p>
<div class="eq">for (int d = 16; d &gt; 0; d &gt;&gt;= 1)
  v += __shfl_down_sync(0xffffffffu, v, d);
// lane 0 now holds the sum of all 32 lanes</div>
<p>Five steps (offsets 16, 8, 4, 2, 1) sum 32 values: log₂ 32 = 5. Trace it on 8 lanes with offsets 4, 2, 1. Lane i adds lane i + d. Lanes with no lane above them just add their own value to itself and are ignored:</p>
<div class="eq">lanes   0  1  2  3  4  5  6  7
start   3  1  4  1  5  9  2  6
d = 4   8 10  6  7  ·  ·  ·  ·     (3+5 …)
d = 2  14 17  ·  ·
d = 1  31                           (14+17)</div>
<p>Only lane 0 holds the answer. This is the same pattern you saw in L2's Count Array Element: <code>warp_sum</code> at the top of <code>${LV}/solutions/4-reduction/kernel.cu</code>. <code>__shfl_xor_sync</code> is the sibling that makes every lane end with the total (a butterfly), useful when all lanes need it, as in softmax.</p>`,
        check: { q: "A warp of 32 lanes runs the five-step __shfl_down_sync reduction (offsets 16, 8, 4, 2, 1). Which lane holds the full sum at the end, and how many shared-memory barriers were needed?",
          options: ["Lane 31; five barriers", "Lane 0; none", "Every lane; one barrier"], answer: 1,
          why: "Each step pulls a value from a higher lane into a lower one, so partial sums accumulate toward lane 0. Lanes of a warp execute the shuffle together and it moves register values directly, so no shared memory and no __syncthreads() are involved. (A xor-based butterfly would leave the total in every lane.)" },
        scene(G) {
          G.text(24, 32, "8 lanes shown; lane i adds the value from lane i + d", { size: 13 });
          const rows = [[3, 1, 4, 1, 5, 9, 2, 6], [8, 10, 6, 7, null, null, null, null], [14, 17, null, null, null, null, null, null], [31, null, null, null, null, null, null, null]];
          const labs = ["start", "d = 4", "d = 2", "d = 1"], cw = 54, x0 = 70;
          for (let L = 0; L < 4; L++) {
            const y = 56 + L * 92;
            G.label(14, y + 24, labs[L], { size: 12, color: "ink" });
            nums(G, x0, y, rows[L], cw, 36, (i) => (L > 0 && rows[L][i] !== null ? "v" : null));
            if (L < 3) { const d = [4, 2, 1][L]; for (let i = 0; i < 8 - d && rows[L + 1][i] !== null; i++) G.arrow(x0 + (i + d) * (cw + 4) + cw / 2, y + 38, x0 + i * (cw + 4) + cw / 2, y + 86, { color: "muted", w: 1 }); }
          }
          G.label(x0, 396, "dots: values no longer needed; lane 0 holds 31", { size: 11 });
          G.caption("log₂ 8 = 3 shuffles; no shared memory, no barrier");
        } },

      { rail: "block reduce", title: "Warps first, then one warp of warps, then one atomic",
        body: `<p>A block has up to 32 warps (1,024 threads). Combine them in two levels:</p>
<ol><li>Every warp reduces its own 32 values with shuffles. Lane 0 of each warp writes its sum into <code>__shared__ float warp_part[32]</code>.</li>
<li><code>__syncthreads()</code>. Then warp 0 loads <code>warp_part[lane]</code> (zero for lanes ≥ number of warps) and reduces <i>that</i> with five more shuffles.</li></ol>
<div class="eq">v = warp_sum(v);                      // 5 shuffles
if (lane == 0) warp_part[w] = v;
__syncthreads();
v = (threadIdx.x &lt; nw) ? warp_part[lane] : 0.f;
if (w == 0) v = warp_sum(v);          // 5 more</div>
<p>With 256 threads: 8 warps, so 10 shuffle rounds and one barrier per block, regardless of how many elements each thread added first. Thread 0 then issues <code>atomicAdd(out, block_total)</code>. With 8 × SMs blocks (320 here) that is 320 atomics in total, not millions. This is the whole of <code>reduce_sum</code> in the reference.</p>
<p><b>Order and rounding.</b> Float addition rounds, so grouping changes the answer. In float32, 10<sup>8</sup> + 1 rounds to 10<sup>8</sup> (floats near 10<sup>8</sup> are 8 apart), so (10<sup>8</sup> + 1) − 10<sup>8</sup> = 0 but (10<sup>8</sup> − 10<sup>8</sup>) + 1 = 1. The atomics land in whatever order blocks finish, so the last bits of the result can differ between runs. If you need bitwise repeatability, write one partial per block and sum them in a second fixed-order pass.</p>
<p><b>Fused map + reduce.</b> Dot Product (#17) and Mean Squared Error (#27) are the same kernel with the loop body changed: <code>acc += a[i] * b[i]</code>, or <code>acc += (a[i] - b[i])²</code>. Fusing saves writing and re-reading the elementwise result.</p>`,
        check: { q: "A reduction over 2^24 floats runs 320 blocks of 256 threads. Each block reduces its threads' sums with shuffles and shared memory, then thread 0 does one atomicAdd. How many global atomics does the whole kernel issue?",
          options: ["320: one per block", "81,920: one per thread", "16,777,216: one per element"], answer: 0,
          why: "Each thread first accumulates about 205 elements in a register, the block combines its 256 thread sums with shuffles plus one barrier, and only thread 0 touches global memory. That is one atomic per block: 320. A per-thread or per-element atomic would serialize on the single output address." },
        scene(G) {
          G.text(24, 32, "block of 256 threads = 8 warps; each: 5 shuffles", { size: 13 });
          const w = Array.from({ length: 8 }, (_, i) => G.box(24 + i * 74, 52, 66, 40, `warp ${i}`, { fill: "k", size: 11, boxOpacity: 0.85 }));
          
          for (let i = 0; i < 8; i++) G.arrow(57 + i * 74, 94, 57 + i * 74, 134, { color: "k" });
          const sm = Array.from({ length: 8 }, (_, i) => G.box(24 + i * 74, 138, 66, 34, `part[${i}]`, { fill: "v", size: 11 }));
          G.label(24, 190, "__shared__ warp_part[ ]", { size: 11 });
          const bar = G.rect(24, 204, 584, 6, { fill: "hot", rx: 3 }); G.label(24, 226, "__syncthreads()", { size: 11, color: "hot" });
          G.box(24, 244, 140, 40, "warp 0 loads 8 parts", { fill: "q", size: 11 });
          G.arrow(166, 264, 206, 264, { color: "q" });
          G.box(210, 244, 140, 40, "5 shuffles → total", { fill: "k", size: 11 });
          G.arrow(352, 264, 392, 264, { color: "k" });
          G.box(396, 244, 190, 40, "atomicAdd(out, total)", { fill: "ok", size: 12 });
          G.text(24, 330, "10 shuffles + 1 barrier + 1 atomic per block", { size: 14, color: "ink" });
          G.text(24, 356, "320 blocks → 320 atomics for any N", { size: 14, color: "ok" });
          G.text(24, 392, "float order varies → last bits can differ per run", { size: 13, color: "hot" });
          G.from(sm, { opacity: 0, y: -10, stagger: 0.05, duration: 0.2 }); G.pulse(bar, { repeat: 5 });
          G.caption("the cooperation cost is tiny next to the memory reads");
        } },

      { rail: "scan", title: "Scan: every running total at once",
        body: `<p>A <b>scan</b> (prefix sum) turns <code>[3 1 4 1 5 9 2 6]</code> into running totals. The <b>inclusive</b> scan includes the element itself: <code>[3 4 8 9 14 23 25 31]</code>. The <b>exclusive</b> scan stops just before it: <code>[0 3 4 8 9 14 23 25]</code>. The last inclusive value is the total.</p>
<p>Serially it is one pass: <code>out[i] = out[i-1] + in[i]</code>. Each result needs the previous one, a chain again. Can it be parallel? Yes, with extra additions. <b>Hillis–Steele</b>: in step k, every element adds the element 2<sup>k</sup> positions to its left (if there is one):</p>
<div class="eq">in     3  1  4  1  5  9  2  6
o = 1  3  4  5  5  6 14 11  8     x[i] += x[i-1]
o = 2  3  4  8  9 11 19 17 22     x[i] += x[i-2]
o = 4  3  4  8  9 14 23 25 31     x[i] += x[i-4]</div>
<p>Three steps for eight elements (log₂ 8), every thread busy each step. Check one value: after o = 2, element 6 is 17. It has added inputs 3 … 6: 1 + 5 + 9 + 2 = 17. ✓ After o = 4 every element has reached back to input 0.</p>
<p>The price is <b>work</b>: step k does N − 2<sup>k</sup> additions, 7 + 6 + 4 = 17 for N = 8 (versus 7 serially), and about N log₂ N in general. For 2,048 elements that is about 10 × as many additions as the serial scan (20,481 versus 2,047). It is a fine choice for small sizes: the reference code uses it to scan 256 per-thread totals. For a big array we want something that does only O(N) work.</p>`,
        check: { q: "Hillis–Steele scan of 8 elements: how many additions in total, compared with the serial scan?",
          options: ["7 (same as serial)", "17 (7 + 6 + 4 versus 7 serially)", "24 (8 per step, 3 steps) versus 7"], answer: 1,
          why: "At offset o, only elements with index ≥ o have a left neighbour to add: 8 − 1 = 7 at o = 1, 8 − 2 = 6 at o = 2, 8 − 4 = 4 at o = 4. Total 17. (Hardware may still issue 24 thread-slots, but 7 of them are masked off.) The serial scan does 7 additions. More work, in 3 steps instead of 7." },
        scene(G) {
          const rows = [[3, 1, 4, 1, 5, 9, 2, 6], [3, 4, 5, 5, 6, 14, 11, 8], [3, 4, 8, 9, 11, 19, 17, 22], [3, 4, 8, 9, 14, 23, 25, 31]];
          const labs = ["input", "o = 1", "o = 2", "o = 4"], cw = 54, x0 = 70;
          G.text(24, 32, "Hillis–Steele: x[i] += x[i − o], o = 1, 2, 4", { size: 13 });
          for (let L = 0; L < 4; L++) {
            const y = 56 + L * 92;
            G.label(14, y + 24, labs[L], { size: 12, color: "ink" });
            nums(G, x0, y, rows[L], cw, 36, (i) => (L > 0 && rows[L][i] !== rows[L - 1][i] ? "v" : null));
            if (L < 3) { const o = [1, 2, 4][L]; for (let i = o; i < 8; i++) G.arrow(x0 + (i - o) * (cw + 4) + cw / 2, y + 38, x0 + i * (cw + 4) + cw / 2, y + 86, { color: "muted", w: 1 }); }
          }
          G.text(24, 408, "17 adds in 3 steps; serial: 7 adds in 7 steps", { size: 13, color: "ink" });
          G.caption("amber cells changed this step; each step doubles how far back it reaches");
        } },

      { rail: "Blelloch", title: "Blelloch: sweep up, then sweep down, in O(N) work",
        body: `<p>Hillis–Steele wastes work because it recomputes overlapping sums. <b>Blelloch's scan</b> avoids that in two sweeps over a tree.</p>
<p><b>Up-sweep</b> (reduce): build the sum tree in place, so the last slot holds the total. <b>Down-sweep</b>: put 0 in the last slot, then go down the tree: each node passes its own left-sum to its right child and gives its left child what it received from above. After the sweep slot i holds the <i>exclusive</i> prefix. Same example:</p>
<div class="eq">in      3 1 4 1 5 9 2 6
up 1    3 4 4 5 5 14 2 8
up 2    3 4 4 9 5 14 2 22
up 3    3 4 4 9 5 14 2 31     (total)
clear   3 4 4 9 5 14 2 0
down 1  3 4 4 0 5 14 2 9      (stride 4)
down 2  3 0 4 4 5 9 2 23      (stride 2)
down 3  0 3 4 8 9 14 23 25    (stride 1)</div>
<p>One down-sweep step is: <code>t = a[l]; a[l] = a[r]; a[r] += t;</code>. Check <i>down 1</i>: l = 3, r = 7: t = 9, a[3] = 0, a[7] = 0 + 9 = 9. ✓</p>
<p>Work: the up-sweep does N − 1 = 7 additions, the down-sweep another 7: 14 ≈ 2N, against Hillis–Steele's N log N. The price is 2 log₂ N steps, and in shared memory the strided access creates bank conflicts that need padding (L2). The reference solution does something simpler: each thread scans 8 consecutive items serially in registers, then the 256 thread totals are scanned across the block, and the per-item prefixes are fixed up. Most of the work is register-serial and cheap.</p>`,
        scene(G) {
          const rows = [[3, 1, 4, 1, 5, 9, 2, 6], [3, 4, 4, 5, 5, 14, 2, 8], [3, 4, 4, 9, 5, 14, 2, 22], [3, 4, 4, 9, 5, 14, 2, 31], [3, 4, 4, 9, 5, 14, 2, 0], [3, 4, 4, 0, 5, 14, 2, 9], [3, 0, 4, 4, 5, 9, 2, 23], [0, 3, 4, 8, 9, 14, 23, 25]];
          const labs = ["input", "up 1", "up 2", "up 3", "clear", "down 1", "down 2", "down 3"], cw = 54, x0 = 80;
          G.text(24, 30, "up-sweep builds the total; down-sweep distributes prefixes", { size: 13 });
          rows.forEach((r, L) => {
            const y = 40 + L * 43;
            G.label(14, y + 24, labs[L], { size: 11, color: L < 4 ? "k" : "q" });
            nums(G, x0, y, r, cw, 34, (i) => (L > 0 && r[i] !== rows[L - 1][i] ? "v" : null), L < 4 ? "k" : "q");
          });
          G.rect(x0 - 3, 40 + 7 * 43 - 3, 8 * (cw + 4) + 2, 40, { stroke: "ok", sw: 2, rx: 6 });
          G.label(x0, 40 + 8 * 43 + 14, "exclusive scan of the input; adds: 7 up + 7 down = 14", { size: 12, color: "ok" });
          G.caption("O(N) work in 2·log₂ N steps; amber = changed this step");
        } },

      { rail: "many blocks", title: "Scanning an array bigger than one block",
        body: `<p>A block can only scan its own tile. The reference solution handles any N in three phases (<code>inclusive_scan</code> in the L3 scan solutions):</p>
<ol><li><b>Scan each tile</b> independently (256 threads × 8 items = 2,048 elements) and write each tile's <i>total</i> to a small array.</li>
<li><b>Scan the tile totals</b>, recursively with the same code (they are 2,048 times fewer).</li>
<li><b>Add the carry</b>: tile t adds the scanned total of tiles 0 … t − 1 to every element.</li></ol>
<p>Four toy tiles of four numbers show it:</p>
<div class="eq">tile 0: 1 2 3 4 → local 1 3 6 10    total 10
tile 1: 5 6 7 8 → local 5 11 18 26  total 26
tile 2: 1 1 1 1 → local 1 2 3 4     total 4
carries (scan of totals) = 0 10 36
+ carry: 1 → 1 3 6 10, 2 → 15 21 28 36,
         3 → 37 38 39 40</div>
<p>The last value, 40 = 10 + 26 + 4, is the grand total. For N = 2<sup>24</sup>: 2<sup>24</sup> / 2,048 = 8,192 tiles, then 8,192 / 2,048 = 4 totals, then one tile: three levels.</p>
<p>The cost is traffic. Scanning tiles reads and writes every element, and adding the carry reads and writes them all again: about 4N elements moved, versus the 2N (read once, write once) that is the minimum. The state of the art, <b>decoupled look-back</b> (Merrill and Garland, NVIDIA technical report; used by CUB's <code>DeviceScan</code>), reaches the minimum in a single pass: each tile publishes its aggregate with a status flag, and a tile that needs its carry <i>looks back</i> over its predecessors' flags instead of waiting for a whole separate pass. Read the paper after you have written the three-phase version, and compare against CUB with <code>--bench</code>.</p>`,
        scene(G) {
          const tiles = [[1, 2, 3, 4], [5, 6, 7, 8], [1, 1, 1, 1]];
          const loc = [[1, 3, 6, 10], [5, 11, 18, 26], [1, 2, 3, 4]], fin = [[1, 3, 6, 10], [15, 21, 28, 36], [37, 38, 39, 40]];
          const cw = 36, gx = [24, 214, 404], rowsY = [58, 142, 262];
          G.label(24, 44, "1  tiles of input", { color: "ink", size: 12 });
          tiles.forEach((t, k) => nums(G, gx[k], rowsY[0], t, cw, 30, null, "k"));
          G.label(24, 128, "2  scan inside each tile; totals = last value", { color: "ink", size: 12 });
          loc.forEach((t, k) => nums(G, gx[k], rowsY[1], t, cw, 30, (i) => (i === 3 ? "v" : null), "k"));
          G.label(24, 200, "3  scan the totals (10, 26, 4) → carries 0, 10, 36", { color: "ink", size: 12 });
          [0, 10, 36].forEach((c, k) => G.box(gx[k], 206, 80, 28, `carry ${c}`, { fill: "q", size: 12 }));
          G.label(24, 258, "4  add each tile's carry to all of its elements", { color: "ink", size: 12 });
          fin.forEach((t, k) => nums(G, gx[k], 268, t, cw, 30, (i) => (k > 0 ? "ok" : null), "k"));
          G.text(24, 346, "tiles of 2,048: 2^24 → 8,192 → 4 → 1", { size: 14, color: "v" });
          G.text(24, 372, "three-phase: ~4N elements moved; look-back: 2N", { size: 14, color: "ink" });
          G.caption("local scans, scan of totals, add carry: green cells got a carry");
        } },

      { rail: "compaction", title: "Scan as a building block: stream compaction",
        body: `<p><b>Stream compaction</b> keeps only the elements that satisfy a condition and packs them together in order. It sounds sequential: where does element i go? It goes to position <i>(number of kept elements before i)</i>, and that is exactly an exclusive scan of 0/1 flags.</p>
<div class="eq">in     5 -1  7  0  3 -2 -4  9   keep if &gt; 0
flags  1  0  1  0  1  0  0  1
scan   0  1  1  2  2  3  3  3   (exclusive)
out    [5 7 3 9]   count = 3 + 1 = 4</div>
<p>Three kernels: <code>flag</code> (one thread per element), the scan (reused from #16 with <code>int +</code>), and <code>scatter</code>: <code>if (flag[i]) out[pos[i]] = in[i]</code>. All threads write to different slots, and the writes are in increasing order, so they coalesce well. The scan's last value gives the count.</p>
<p>The same pattern is everywhere. <b>Radix sort</b> scans digit histograms to find each key's slot. <b>Segmented scans</b> (#70) carry a flag array to restart sums at segment boundaries. A serving engine uses compaction to drop finished sequences from a batch, which you will meet in P6. The reference <code>${LV}/solutions/72-stream-compaction/kernel.cu</code> shows the fused-variant hint too: <code>__ballot_sync</code> plus <code>__popc</code> count the kept lanes of a warp without shared memory.</p>`,
        check: { q: "Keep the elements > 0 of [2, -1, 4, -3, 6, 8]. What is the exclusive-scan value of the flag of element 8 (the last one), and so where does it go in the output?",
          options: ["3: it goes to out[3]", "4: it goes to out[4]", "5: it keeps its own index"], answer: 0,
          why: "Flags are 1 0 1 0 1 1, and the exclusive scan is 0 1 1 2 2 3. Three kept elements (2, 4, 6) come before the 8, so it is written to out[3] and the output is [2, 4, 6, 8]. The scan value is a count of kept elements before you, not the index you started at." },
        scene(G) {
          const cw = 54, x0 = 70;
          const rows = [["input", [5, -1, 7, 0, 3, -2, -4, 9]], ["flag > 0", [1, 0, 1, 0, 1, 0, 0, 1]], ["excl. scan", [0, 1, 1, 2, 2, 3, 3, 3]]];
          G.text(24, 30, "keep elements > 0, in order", { size: 13 });
          rows.forEach(([n, r], k) => { G.label(8, 78 + k * 66, n, { size: 11, color: "ink" }); nums(G, x0 + 20, 56 + k * 66, r, cw - 8, 34, (i) => (k > 0 && (i === 0 || i === 2 || i === 4 || i === 7) ? "v" : null), "k"); });
          const keep = [0, 2, 4, 7];
          keep.forEach((src, j) => G.arrow(x0 + 20 + src * (cw - 4) + 23, 258 - 0, x0 + 20 + j * (cw - 4) + 23, 318, { color: "ok", w: 1.5 }));
          G.label(8, 340, "out", { size: 11, color: "ink" });
          nums(G, x0 + 20, 322, [5, 7, 3, 9], cw - 8, 34, () => "ok");
          G.text(24, 392, "position of a kept element = its scan value", { size: 14, color: "v" });
          G.text(24, 416, "count = last scan value + last flag = 4", { size: 13, color: "ink" });
          G.caption("a scan of the flags gives every survivor its output slot");
        } },

      { rail: "sort and top-k", title: "Sorting networks and top-k tournaments",
        body: `<p><b>Bitonic sort</b> is a sorting algorithm with no data-dependent branches: a fixed network of <i>compare-exchange</i> steps (look at two slots, put the smaller first). Because the network is the same for every input, every thread knows what to do. Stage (k, j) compares slot i with slot i XOR j and orders them ascending if <code>(i &amp; k) == 0</code>, descending otherwise.</p>
<p>For a power-of-two size n = 2<sup>m</sup> there are m(m + 1)/2 stages, each doing n/2 compare-exchanges. For a 2,048-element tile (m = 11): 11 × 12 / 2 = 66 stages × 1,024 = 67,584 compare-exchanges, all inside shared memory with a barrier per stage. Total work is O(N log² N), more than a merge sort's O(N log N), but it parallelizes perfectly. For big arrays, a radix sort (#36; CUB's <code>DeviceRadixSort</code>) beats it. The reference sorts 2,048-element tiles in shared memory, then handles larger merges: stages with distance j ≥ 2,048 in global memory, the rest back in shared memory.</p>
<p><b>Top-k</b> (the k largest values) does not need a full sort. Sort each tile, keep only its top K' (K' = k rounded up to a power of two), and repeat on the survivors: the true top-k can never be dropped, because an element dropped from a tile has at least K' larger elements beside it. Example: N = 2<sup>24</sup>, k = 100, K' = 128, tiles of 2,048 (16 times shrink per round):</p>
<div class="eq">16,777,216 → 8,192 tiles × 128 = 1,048,576
 1,048,576 →   512 tiles × 128 =    65,536
    65,536 →    32 tiles × 128 =     4,096
     4,096 →     2 tiles × 128 =       256
       256 → 1 tile: sort, take the first 100</div>
<p>Five launches. Sampling the top-k of a ~128k-token vocabulary is this problem (P2). Large k (&gt; 1,024 here) cannot shrink the set; the reference leaves that fallback as <code>TODO(learner)</code>: a full sort, or a radix-select using histograms (L2) and scans.</p>`,
        scene(G) {
          const ns = [16777216, 1048576, 65536, 4096, 256, 100], labs = ["N = 16,777,216", "1,048,576", "65,536", "4,096", "256", "top 100"];
          G.text(24, 32, "top-k tournament: sort tiles of 2,048, keep 128 each", { size: 13 });
          const bars = ns.map((n, i) => {
            const w = Math.log2(n) * 20, y = 56 + i * 46;
            G.label(24, y - 4, labs[i], { size: 11, color: "ink" });
            return G.rect(24, y, w, 24, { fill: i === 5 ? "ok" : "k", rx: 5, opacity: 0.9 });
          });
          bars.forEach((b, i) => { if (i) G.label(24 + Math.log2(ns[i]) * 20 + 10, 56 + i * 46 + 17, i < 5 ? "round " + i : "answer", { size: 11 }); });
          G.label(24, 344, "bar length = log₂(size)", { size: 11 });
          G.text(24, 380, "bitonic tile: 66 stages × 1,024 = 67,584 compare-exchanges", { size: 12, color: "v" });
          G.text(24, 404, "5 rounds, each shrinks the set 16×", { size: 13, color: "ok" });
          G.from(bars, { scaleX: 0, transformOrigin: "left center", stagger: 0.15, duration: 0.3 });
          G.caption("each round discards everything that cannot be in the top k");
        } },

      { rail: "SpMV", title: "Sparse matrix-vector: one warp per row",
        body: `<p>Most entries of some matrices (graphs, pruned weights) are zero. <b>CSR</b> (compressed sparse row) stores only the nonzeros in three arrays: <code>vals</code> (the nonzero values, row by row), <code>col_idx</code> (the column of each), and <code>row_ptr</code> (where each row starts in those arrays). Row r owns entries <code>row_ptr[r] … row_ptr[r+1] − 1</code>.</p>
<div class="eq">A = [5 0 0 1]    row_ptr = [0 2 3 3 6]
    [0 2 0 0]    col_idx = [0 3 1 0 1 2]
    [0 0 0 0]    vals    = [5 1 2 4 3 6]
    [4 3 6 0]
x = [1 2 3 4]  →  y = [9 4 0 28]
row 0: 5·1 + 1·4 = 9     row 3: 4·1 + 3·2 + 6·3 = 28</div>
<p>Each output y[r] is a dot product, so a reduction per row. One thread per row is the obvious mapping, but neighbouring threads then read different rows' <code>vals</code>, which are far apart (uncoalesced), and a long row stalls its thread while short rows finish. <b>Warp per row</b> fixes both: the 32 lanes stride across the row's nonzeros (<code>k = start + lane; k += 32</code>), so reads of <code>vals</code> and <code>col_idx</code> are contiguous, then a five-step shuffle reduction adds the lanes' partial sums, and lane 0 writes y[r].</p>
<p>The reads <code>x[col_idx[k]]</code> are a <b>gather</b>: random access into x. Read it through the read-only cache (<code>__ldg</code>); how well it works depends on the matrix.</p>
<p>When rows are very short (average below about 4 nonzeros), a warp is mostly idle, so the reference switches to thread per row (<code>solve</code> checks the average). SpMV does 2 FLOPs per 8 bytes of <code>vals</code> and <code>col_idx</code> (0.25 FLOP/byte, before counting x), far below what any GPU needs to be compute-bound. Judge it in GB/s against copy bandwidth, as in L1.</p>`,
        scene(G) {
          const A = [[5, 0, 0, 1], [0, 2, 0, 0], [0, 0, 0, 0], [4, 3, 6, 0]];
          G.text(24, 30, "CSR: only nonzeros are stored", { size: 13 });
          for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) G.box(24 + c * 42, 46 + r * 38, 38, 34, A[r][c] ? String(A[r][c]) : "", { fill: A[r][c] ? (r === 3 ? "v" : "k") : "line", size: 13, boxOpacity: A[r][c] ? 0.9 : 0.3 });
          G.label(24, 212, "A (row 3 highlighted)", { size: 11 });
          G.label(220, 58, "vals", { size: 11, color: "ink" }); G.label(220, 114, "col_idx", { size: 11, color: "ink" }); G.label(220, 170, "row_ptr", { size: 11, color: "ink" });
          [5, 1, 2, 4, 3, 6].forEach((v, i) => G.box(290 + i * 50, 42, 46, 30, String(v), { fill: i >= 3 ? "v" : "k", size: 13, boxOpacity: 0.9 }));
          [0, 3, 1, 0, 1, 2].forEach((v, i) => G.box(290 + i * 50, 98, 46, 30, String(v), { fill: i >= 3 ? "v" : "k", size: 13, boxOpacity: 0.7 }));
          [0, 2, 3, 3, 6].forEach((v, i) => G.box(290 + i * 50, 154, 46, 30, String(v), { fill: i >= 3 ? "v" : "line", size: 13, boxOpacity: 0.8 }));
          G.label(290 + 3 * 50 - 6, 204, "row 3 = entries 3 … 5", { size: 11, color: "v" });
          G.text(24, 260, "warp per row: lane l handles entries 3 + l, 3 + l + 32 …", { size: 13, color: "ink" });
          [0, 1, 2].forEach((l) => G.box(24 + l * 90, 276, 82, 32, `lane ${l}`, { fill: "v", size: 12 }));
          G.label(300, 296, "lanes 3 … 31: no entry, add 0", { size: 11 });
          G.label(24, 330, "4·1 = 4        3·2 = 6        6·3 = 18", { size: 12, color: "ink" });
          G.text(24, 366, "5 shuffles: 4 + 6 + 18 = 28 → y[3]", { size: 15, color: "ok" });
          G.text(24, 394, "coalesced vals/col_idx, one reduction per row", { size: 13, color: "ink" });
          G.caption("consecutive lanes read consecutive nonzeros of one row");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>An associative operation regroups into a tree: N − 1 additions in log₂ N dependent levels.</li>
<li>Fast reductions do most of the adding in registers (grid-stride), then use warp shuffles (5 steps, no barrier), a tiny shared-memory step across warps, and one atomic per block. The kernel ends memory-bound.</li>
<li>Float sums change with grouping; atomics make the last bits order-dependent.</li>
<li>Scan: Hillis–Steele is simple but O(N log N) work; Blelloch is O(N); multiple tiles need scan-of-totals and a carry (or decoupled look-back).</li>
<li>Scan turns "where does my output go?" into arithmetic: compaction, radix sort, segmented ops. Bitonic networks sort in shared memory; top-k keeps only each tile's best K'.</li>
<li>CSR SpMV: one warp per row, then shuffle-reduce.</li></ul>
<p>Next, L4 fuses elementwise ops and normalizations (softmax, layer norm, RMSNorm): reductions embedded in the middle of a bigger kernel.</p>`,
    sim: {
      title: "Where do the additions of a big sum happen?",
      intro: "Pick the array size and the launch shape of the reference reduction (grid-stride accumulate → warp shuffles → per-warp slots → atomic per block). The bars show how the N − 1 additions split across the four stages. The panels give the launch, the minimum memory traffic and the number of cooperative operations.",
      height: 300,
      controls: [
        { id: "lg", label: "log₂ N (elements)", min: 16, max: 30, step: 1, value: 24 },
        { id: "sms", label: "SM count (example: T4 has 40, check yours)", min: 1, max: 132, step: 1, value: 40 },
        { id: "bps", label: "blocks per SM", min: 1, max: 16, step: 1, value: 8 },
        { id: "thr", label: "threads per block", type: "select", value: 256, options: [[128, "128"], [256, "256"], [512, "512"], [1024, "1024"]] },
        { id: "bw", label: "memory bandwidth GB/s (example value)", min: 50, max: 1000, step: 10, value: 300 },
      ],
      draw(G, v) {
        const N = Math.pow(2, v.lg), thr = v.thr, warpsPB = thr / 32;
        let blocks = v.sms * v.bps;
        const need = Math.ceil(N / 4 / thr);               // float4 loads: one thread per 4 elements at most
        if (need < blocks) blocks = Math.max(1, need);
        const T = blocks * thr, W = blocks * warpsPB;
        const reg = Math.max(0, N - T), warp = Math.max(0, T - W), blk = Math.max(0, W - blocks), atom = Math.max(0, blocks - 1);
        const stages = [["registers", reg, "k"], ["shuffles", warp, "q"], ["warp 0", blk, "v"], ["atomics", atom, "hot"]];
        const x0 = 100, wmax = 520;
        G.text(20, 22, "share of the N − 1 additions (bar length = fraction)", { size: 11 });
        stages.forEach(([n, c, col], i) => {
          const y = 44 + i * 58, frac = c / (N - 1);
          G.label(20, y + 16, n, { size: 11, color: "ink" });
          G.rect(x0, y, wmax, 24, { stroke: "line", rx: 5 });
          G.rect(x0, y, Math.max(2, wmax * frac), 24, { fill: col, rx: 5 });
          G.label(x0 + 6, y + 40, `${F.num(Math.round(c))} adds (${(frac * 100).toFixed(frac < 0.001 ? 4 : 2)}%)`, { size: 11 });
        });
        G.label(20, 290, "everything except the first bar is cooperation", { size: 11 });
        const bytes = 4 * N, ms = bytes / (v.bw * 1e9) * 1e3, naiveBlocks = Math.ceil(N / thr);
        return [
          { title: "Launch", rows: [["blocks", F.num(blocks)], ["threads", F.num(T)], ["elements per thread", (N / T).toFixed(1)], ["warps per block", String(warpsPB)]],
            chip: [N / T >= 8, N / T >= 8 ? "plenty of serial work per thread" : "little per-thread work: cooperation dominates"] },
          { title: "Memory-bound estimate", rows: [["bytes read", F.bytes(bytes)], ["time ≥ bytes / BW", `${ms.toFixed(3)} ms`]],
            html: `<p class="note">A lower bound: every element read once at the stated bandwidth (an example value: use your GPU's spec sheet or the harness's measured copy). Real kernels also pay launch time.</p>` },
          { title: "Cooperation", rows: [["shuffle rounds per block", String(5 * (warpsPB > 1 ? 2 : 1))], ["barriers per block", warpsPB > 1 ? "1" : "0"], ["global atomics", F.num(blocks)], ["one element per thread instead: blocks", F.num(naiveBlocks)]],
            html: `<p class="note">The last row is the block count of a tree that loads one element per thread: each block would write a partial that another pass must add.</p>` },
        ];
      },
      note: "Model: N − 1 additions split as registers N − T, warp shuffles T − W, warp 0 W − B, atomics B − 1 (T threads, W warps, B blocks). Block count is capped so each thread has at least one float4.",
    },
    practice: {
      intro: `<p>Solve each problem on <a href="https://leetgpu.com/challenges">LeetGPU</a> first (T1, in the browser). We don't reproduce the statements; read them there. Then run the local reference and test (T2, needs an NVIDIA GPU and CUDA ≥ 12.4). Build once from the repository root:</p>
<p><code>cmake -S laneB-cuda -B build/laneB -DCMAKE_BUILD_TYPE=Release &amp;&amp; cmake --build build/laneB -j</code></p>
<p>Run <code>--bench</code> on the binary directly, for example <code>./build/laneB/L3_4-reduction --bench</code>. <b>Exit check:</b> your sum of 2<sup>24</sup> to 2<sup>28</sup> floats is within 10% of <code>cub::DeviceReduce::Sum</code> (CCCL v3.5.0 as pinned in the repo), median of 100 runs, on the same GPU (<code>TODO(run-on: g4dn.xlarge)</code> until run). Note #29 ships with a <code>TODO(learner)</code> for k &gt; 1,024.</p>`,
      items: [
        P("4-reduction", "Reduction", "medium", "core · exit check", "Grid-stride <code>float4</code> accumulate, warp shuffles, one atomic per block; compare with CUB."),
        P("17-dot-product", "Dot Product", "medium", "core", "The same reduction with a fused multiply in the loop body."),
        P("16-prefix-sum", "Prefix Sum", "medium", "core", "Tile scan with 8 serial items per thread, scan of totals, carry add; then read about decoupled look-back."),
        P("72-stream-compaction", "Stream Compaction", "medium", "core", "Flags, scan, scatter; the scan value is the output slot."),
        P("29-top-k-selection", "Top K Selection", "medium", "core", "Tile sort and keep K' per round; implement the k &gt; 1,024 fallback and add a test with k = 5000."),
        P("15-sorting", "Sorting", "hard", "core", "Bitonic network: shared-memory stages for j &lt; 2,048, global stages above."),
        P("18-sparse-matrix-vector-multiplication", "Sparse Matrix-Vector Multiplication", "medium", "core", "CSR, warp per row, shuffle reduce; thread per row when rows are short."),
        P("27-mean-squared-error", "Mean Squared Error", "medium", "practice", "Fused squared difference plus reduction."),
        P("70-segmented-prefix-sum", "Segmented Exclusive Prefix Sum", "medium", "practice", "A scan whose operator restarts at flagged boundaries."),
        P("47-subarray-sum", "Subarray Sum", "medium", "practice", "A range sum as a difference of two prefix sums."),
        P("48-2d-subarray-sum", "2D Subarray Sum", "medium", "practice", "The 2D variant: scan rows, then columns."),
        P("38-nearest-neighbor", "Nearest Neighbor", "medium", "practice", "Brute-force distances with a block-level argmin reduction."),
        P("35-monte-carlo-integration", "Monte Carlo Integration", "medium", "practice", "Per-thread random samples, then a sum reduction."),
        P("49-3d-subarray-sum", "3D Subarray Sum", "medium", "stretch", "One more scan axis."),
        P("51-max-subarray-sum", "Max Subarray Sum", "medium", "stretch", "A scan over (sum, best) pairs with an associative combine."),
        P("36-radix-sort", "Radix Sort", "hard", "stretch", "Digit histograms and a scan give every key its slot."),
        P("71-parallel-merge", "Parallel Merge", "medium", "stretch", "Merge-path partitioning."),
        P("75-sparse-matrix-dense-matrix-multiplication", "Sparse Matrix-Dense Matrix Multiplication", "medium", "stretch", "CSR times a dense matrix: reuse each loaded nonzero across columns."),
        P("82-linear-recurrence", "Linear Recurrence", "medium", "stretch", "Scan with an associative linear operator."),
        P("110-gae-reverse-scan", "Parallel Reverse Scan (GAE)", "medium", "stretch", "A scan run from the right."),
      ],
      labs: [
        { label: "Level syllabus, objectives and exit check", path: `${LV}/syllabus.md` },
        { label: "Problem map with links and priorities", path: `${LV}/leetgpu-map.md` },
        { label: "The harness: CUDA_CHECK, time_gpu, measure_copy_gbs, report", path: "laneB-cuda/harness/" },
        { label: "Reduction animation (Lane A P5.4)", path: "animations/p5-reduction.html" },
        { label: "AWS T2 box for L1–L4 (T4, sm_75): instance_type = \"g4dn.xlarge\", then make down", path: "infra/aws/single-node" },
      ],
    },
  });
})();
