/* P5.5 — Softmax and norms. Row-wise kernels are memory-bound: count row-passes, then remove them (online softmax, registers, fusion). */
(function () {
  const F = S2S.fmt;
  // Example bandwidths from course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/gpu_specs.yaml (UNVERIFIED there).
  const GPUS = { t4: { name: "T4", bw: 320e9 }, l4: { name: "L4", bw: 300e9 }, a100: { name: "A100-80GB", bw: 2039e9 } };
  // a labelled, filled box in the passes timeline: R = read, W = write
  const blk = (G, x, y, w, label, kind) => G.box(x, y, w, 26, label, { fill: kind === "R" ? "k" : "v", size: 11, rx: 4 });

  S2S.lesson({
    id: "p5-5", n: "P5.5", title: "Softmax and norms",
    subtitle: "CUDA deep dive · first principles · T2, one NVIDIA GPU (sm_75+)",
    kicker: "Lesson · ≈ 45 min",
    headline: "Do less than read the row three times",
    intro: `<p>Softmax and RMSNorm sit between almost every pair of matrix multiplications in a transformer. They do hardly any arithmetic, so their speed is set by one thing: how many times the kernel reads and writes each row in DRAM. This lesson builds both operations from their definitions, shows why softmax needs a numerical trick, and then removes row-passes one at a time: the <b>online softmax</b> that merges its max and its sum, the row kept in <b>registers</b>, and <b>fusion</b> with neighbouring operations. It ends with the precision rules (store in bf16, accumulate in fp32) and how to derive a test tolerance instead of guessing one.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "read from DRAM"], ["v", "write to DRAM"], ["q", "running max m"], ["ok", "result / saved"], ["hot", "overflow / wasted"]],
    prev: "p5-4", next: "p5-6",
    steps: [
      { rail: "softmax", title: "Softmax turns scores into probabilities",
        body: `<p>A model's last layer produces one real number per vocabulary word, the <b>logits</b>. Attention produces one score per earlier token. Both need to become <b>probabilities</b>: positive numbers that add up to 1, larger where the score is larger. <b>Softmax</b> does that by exponentiating, then dividing by the total:</p>
<div class="eq">softmax(x)_i = e^(x_i) / Σ_j e^(x_j)

x = [2, 1, 0]
e^x = [7.389, 2.718, 1.000]      sum = 11.107
p   = [0.665, 0.245, 0.090]      sum = 1</div>
<p>The exponential makes every value positive and turns differences into ratios: a score 1 higher gets e ≈ 2.72 times the probability. Softmax works on each <b>row</b> independently. For a batch of tokens, the logits are a matrix <code>rows × cols</code> (cols = vocabulary size, or the number of attention keys) and one kernel handles every row.</p>
<p>The arithmetic per element is one exponential, one add and one divide. A GPU does thousands of those in the time it takes to fetch one number from DRAM. Remember that; the rest of the lesson follows from it.</p>`,
        scene(G) {
          const x = [2, 1, 0], p = [0.665, 0.245, 0.09];
          G.text(40, 40, "logits x", { size: 14 });
          G.text(380, 40, "probabilities p", { size: 14, color: "ok" });
          const base = 330, h = 40;
          const b1 = x.map((v, i) => G.rect(50 + i * 80, base - (v + 0.4) * h * 1.6, 56, (v + 0.4) * h * 1.6, { fill: "q", rx: 4 }));
          x.forEach((v, i) => { G.text(78 + i * 80, base - (v + 0.4) * h * 1.6 - 8, String(v), { anchor: "middle", size: 14 }); G.label(78 + i * 80, base + 18, "x" + i, { anchor: "middle" }); });
          const b2 = p.map((v, i) => G.rect(390 + i * 80, base - v * 260, 56, v * 260, { fill: "ok", rx: 4 }));
          p.forEach((v, i) => { G.text(418 + i * 80, base - v * 260 - 8, v.toFixed(3), { anchor: "middle", size: 14 }); G.label(418 + i * 80, base + 18, "p" + i, { anchor: "middle" }); });
          G.arrow(290, 180, 365, 180, { color: "ink", w: 2 });
          G.box(262, 130, 106, 40, "softmax", { stroke: "ink", size: 14 });
          G.label(262, 205, "e^x, then ÷ sum", { size: 12 });
          G.label(262, 223, "sum of p = 1", { size: 12 });
          G.label(40, 372, "1 exp + 1 add + 1 divide per element: almost no maths", { size: 13, color: "ink" });
          G.from(b1.concat(b2), { attr: { height: 0 }, duration: 0.5, stagger: 0.08 });
          G.caption("positive, sums to 1, bigger score gets bigger share");
        } },

      { rail: "overflow", title: "exp overflows, so subtract the max first",
        body: `<p>Real logits are not small. In fp32 the largest finite number is about 3.4 × 10<sup>38</sup>, and e<sup>x</sup> reaches that at x ≈ 88.7. Logits of 100 or 1000 occur (the course test uses values up to 1000). Then <code>exp(x)</code> is <code>inf</code>, and <code>inf / inf</code> is <code>NaN</code>.</p>
<p>The fix uses one fact: multiplying the numerator and denominator by the same constant changes nothing. Pick the constant <code>e^(-m)</code> where <code>m = max(x)</code>:</p>
<div class="eq">softmax(x)_i = e^(x_i - m) / Σ_j e^(x_j - m)
where m = max(x)

x = [1000, 999]
naive:   e^1000 = inf  → inf / inf = NaN
shifted: x - m = [0, -1]
         e^x = [1.000, 0.368]    sum = 1.368
         p   = [0.731, 0.269]</div>
<p>After the shift the largest exponent is e<sup>0</sup> = 1, so the sum is at least 1 (never zero, never inf) and every term is at most 1. The answer is mathematically identical. It costs one extra thing: you must know the row maximum <i>before</i> you can start summing. That dependency is what makes softmax interesting to implement.</p>`,
        check: { q: "In fp32, softmax of [1000, 999] is computed as exp(x) / Σ exp(x) with no shift. What comes out?",
          options: ["[0.731, 0.269], exp just saturates harmlessly", "NaN: exp(1000) is inf, and inf / inf is NaN", "[0, 0], because the values underflow"], answer: 1,
          why: "exp(1000) is far above fp32's largest value (about 3.4e38), so it becomes inf. The sum is inf too, and inf / inf is NaN. Subtracting the max first gives exponents 0 and -1, which are safe, and the same probabilities mathematically." },
        scene(G) {
          G.text(24, 36, "without the shift", { size: 14, color: "hot" });
          G.row(24, 50, ["x = 1000", "x = 999"], { w: 100, h: 34, gap: 12, stroke: "muted", color: "ink", size: 13 });
          G.arrow(74, 88, 74, 118, { color: "hot", w: 2 });
          G.arrow(186, 88, 186, 118, { color: "hot", w: 2 });
          G.row(24, 122, ["e^x = inf", "e^x = inf"], { w: 100, h: 34, gap: 12, fill: "hot", size: 13 });
          G.arrow(130, 160, 130, 186, { color: "hot", w: 2 });
          G.box(60, 190, 140, 34, "inf / inf = NaN", { fill: "hot", size: 13 });
          G.text(340, 36, "with the shift (m = 1000)", { size: 14, color: "ok" });
          G.row(340, 50, ["x - m = 0", "x - m = -1"], { w: 120, h: 34, gap: 12, stroke: "muted", color: "ink", size: 13 });
          G.arrow(400, 88, 400, 118, { color: "ok", w: 2 });
          G.arrow(532, 88, 532, 118, { color: "ok", w: 2 });
          G.row(340, 122, ["1.000", "0.368"], { w: 120, h: 34, gap: 12, fill: "ok", size: 13 });
          G.arrow(466, 160, 466, 186, { color: "ok", w: 2 });
          G.box(386, 190, 160, 34, "p = 0.731, 0.269", { fill: "ok", size: 13 });
          G.rect(24, 262, 592, 120, { stroke: "line", rx: 8 });
          G.label(40, 288, "fp32 largest finite value ≈ 3.4e38 ≈ e^88.7", { size: 13, color: "ink" });
          G.label(40, 312, "e^(x-m) ≤ 1 and the max term is exactly 1", { size: 13, color: "ink" });
          G.label(40, 336, "so the sum is in [1, cols]: never 0, never inf", { size: 13, color: "ink" });
          G.label(40, 360, "same answer: both sides were multiplied by e^(-m)", { size: 13, color: "muted" });
          G.caption("subtracting the row max keeps every exponent at or below 0");
        } },

      { rail: "3 passes", title: "Safe softmax reads the row three times",
        body: `<p>"Safe softmax" needs the max before the sum, and the sum before it can divide. So a straightforward kernel makes three passes over the row:</p>
<ol><li><b>pass 1:</b> read the row, find <code>m</code>.</li><li><b>pass 2:</b> read the row again, compute <code>s = Σ e^(x - m)</code>.</li><li><b>pass 3:</b> read it a third time, write <code>e^(x - m) / s</code>.</li></ol>
<p>That is 3 reads and 1 write. Softmax does so little arithmetic that the time is the bytes moved divided by the memory bandwidth, the same bound you used for GEMM in P5.6. Take 65,536 rows of 1,024 fp32 values (a row is 4 KiB):</p>
<div class="eq">one row-pass = 65,536 × 1,024 × 4 B = 268.4 MB

safe        3R + 1W = 4 passes = 1,074 MB
online      2R + 1W = 3 passes =   805 MB
registers   1R + 1W = 2 passes =   537 MB

at 320 GB/s (T4 example value, UNVERIFIED):
  safe 3.36 ms · online 2.52 ms · registers 1.68 ms</div>
<p>These are lower bounds: the time if the kernel ran at the quoted peak bandwidth. Small rows can be re-read from the GPU's caches, which the numbers ignore; the point is that the version with fewer passes can never be slower, and the best version must touch the data exactly twice (read it, write the result). The goal of the next three steps is to get from 4 passes to 2.</p>`,
        scene(G) {
          const rows = [["safe", [["R", "max"], ["R", "sum"], ["R", "÷"], ["W", ""]], "1,074 MB · 3.36 ms"], ["online", [["R", "m, d"], ["R", "÷"], ["W", ""]], "805 MB · 2.52 ms"], ["registers", [["R", "all"], ["W", ""]], "537 MB · 1.68 ms"]];
          G.text(24, 34, "row-passes over DRAM (65,536 × 1,024 fp32)", { size: 13 });
          rows.forEach(([name, blocks, tot], i) => {
            const y = 60 + i * 108;
            G.text(24, y + 18, name, { size: 14 });
            const bs = blocks.map(([kind, lbl], j) => blk(G, 120 + j * 100, y, 92, (kind === "R" ? "read " : "write ") + lbl, kind));
            G.label(120, y + 54, tot, { size: 13, color: "ink" });
            G.from(bs, { opacity: 0, x: -20, stagger: 0.1, duration: 0.25, delay: i * 0.2 });
          });
          G.label(24, 392, "time = bytes / bandwidth: every removed pass is a saving", { size: 13, color: "ink" });
          G.caption("320 GB/s is a T4 example value, UNVERIFIED; times are lower bounds");
        } },

      { rail: "online", title: "Online softmax: keep the max and the sum together",
        body: `<p>Can the sum be computed in the same pass as the max? The sum needs the final max, which we do not know until the end. The trick is to <b>keep the sum correct for the max seen so far</b>, and fix it whenever the max grows.</p>
<p>Keep two numbers: <code>m</code>, the max so far, and <code>d = Σ e^(x_i - m)</code> over the elements so far. Read the next element <code>v</code>:</p>
<ul><li>If <code>v ≤ m</code>: the max does not change, so <code>d ← d + e^(v - m)</code>.</li>
<li>If <code>v &gt; m</code>: every old term <code>e^(x_i - m)</code> must become <code>e^(x_i - v)</code>. That is the old term times <code>e^(m - v)</code>. So <code>d ← d · e^(m - v) + 1</code> (the new term is <code>e^(v - v) = 1</code>), and <code>m ← v</code>.</li></ul>
<div class="eq">x = [2, 5, 3]
start      m = 2   d = 1                (just x0)
read 5 &gt; 2  d = 1 · e^(2-5) + 1 = 1.050   m = 5
read 3 ≤ 5  d = 1.050 + e^(3-5) = 1.185

check: e^(2-5) + e^(5-5) + e^(3-5)
     = 0.050 + 1 + 0.135 = 1.185  ✓</div>
<p>After one pass over the row, <code>(m, d)</code> is exactly what the three-pass version needed two passes to compute. A second pass divides: <code>e^(x - m) / d</code>. Total: 2 reads and 1 write. This is the <b>online normalizer</b> of Milakov and Gimelshein (2018), and it is the same rescaling idea FlashAttention uses in P5.8. The animation <code>animations/p5-online-softmax.html</code> steps through it.</p>`,
        check: { q: "Online softmax reads x = [1, 4] left to right. What are (m, d) at the end?",
          options: ["m = 4, d = 2", "m = 4, d ≈ 1.05", "m = 1, d ≈ 1.05"], answer: 1,
          why: "Start m = 1, d = 1. Reading 4 > 1 rescales the old sum by e^(1-4) = 0.0498 and adds the new term 1, so d = 1.0498 and m = 4. Check directly: e^(1-4) + e^(4-4) = 0.0498 + 1." },
        scene(G) {
          const x = [2, 5, 3];
          G.text(24, 34, "stream the row left to right", { size: 13 });
          x.forEach((v, i) => G.box(24 + i * 90, 46, 74, 34, "x = " + v, { stroke: "muted", color: "ink", size: 13 }));
          const states = [["m = 2", "d = 1", "first element", "line"], ["m = 5", "d = 1.050", "5 > 2: rescale old d by e^(2-5)", "hot"], ["m = 5", "d = 1.185", "3 ≤ 5: just add e^(3-5)", "ok"]];
          const gs = states.map(([m, d, why, col], i) => {
            const y = 112 + i * 92;
            G.text(24, y + 20, "after x" + i, { size: 12, color: "muted" });
            const g1 = G.box(110, y, 90, 34, m, { fill: "q", size: 14 });
            const g2 = G.box(210, y, 120, 34, d, { fill: col === "line" ? "w" : col, size: 14 });
            G.label(110, y + 58, why, { size: 12, color: col === "hot" ? "hot" : "ink" });
            if (i > 0) G.arrow(155, y - 22, 155, y - 2, { color: "muted" });
            return [g1, g2];
          });
          G.label(360, 146, "d is always", { size: 13, color: "ink" });
          G.label(360, 166, "Σ e^(x_i - m)", { size: 13, color: "ink" });
          G.label(360, 186, "for the max m", { size: 13, color: "ink" });
          G.label(360, 206, "seen so far", { size: 13, color: "ink" });
          G.box(360, 290, 240, 40, "final: m = 5, d = 1.185", { stroke: "ok", color: "ok", size: 13 });
          G.label(360, 350, "one read replaces two passes", { size: 12 });
          gs.forEach((g, i) => G.from(g, { opacity: 0, y: 12, duration: 0.3, delay: 0.3 * i }));
          G.caption("when the max grows, the old sum is rescaled to the new max");
        } },

      { rail: "merge", title: "Partial states merge, so lanes can share a row",
        body: `<p>A row of 128,256 values (Llama 3's vocabulary) is too long for one thread. Many threads each scan a slice, then their partial results must be combined. For a plain sum that is just <code>a + b</code>. For softmax each slice has its own <code>(m, d)</code>, and the two sums are relative to different maxes, so they cannot simply be added. Bring both to the larger max <code>M</code> first:</p>
<div class="eq">(m1, d1) ⊕ (m2, d2)
    = ( M,  d1·e^(m1-M) + d2·e^(m2-M) )
  where M = max(m1, m2)

(2, 3) ⊕ (5, 1):  M = 5
  d = 3 · e^(2-5) + 1 · e^(5-5)
    = 3 × 0.0498 + 1 = 1.149</div>
<p>This is the single-element rule from the last step, generalised: a lone element is the state <code>(x, 1)</code>. The operation is <b>associative</b> (merging in any grouping gives the same result), just like <code>+</code>, which is exactly what a parallel reduction needs. So the tree reduction from P5.4 applies unchanged: each lane scans its slice, then lanes merge pairwise with <code>__shfl_xor_sync</code>, then warps merge through shared memory. The repo's version is <code>md_merge</code> and <code>warp_md</code> in <code>platform/kernels/include/d4/softmax.cuh</code>.</p>
<p>One edge case matters in real models. A fully masked slice has <code>m = -inf</code>. Then <code>-inf - (-inf)</code> is NaN. <code>md_merge</code> guards it: an empty state contributes <code>d = 0</code>.</p>`,
        check: { q: "Merge the partial states (m, d) = (2, 3) and (5, 1). What is the merged d?",
          options: ["4", "3·e^(2-5) + 1 ≈ 1.149", "3 + 1·e^(5-2) ≈ 23.1"], answer: 1,
          why: "Both sums must be expressed relative to the new max M = 5. The first state's terms shrink by e^(2-5) = 0.0498, so 3 contributes 0.149, and the second state already uses m = 5, so its 1 stays. Adding 4 would mix terms measured against different maxes." },
        scene(G) {
          const xs = [2, 5, 3, 1];
          G.text(24, 32, "4 lanes, one element each → shuffle merge", { size: 13 });
          const lane = xs.map((v, i) => G.box(24 + i * 150, 50, 130, 44, `lane ${i}: (${v}, 1)`, { fill: "w", size: 13 }));
          const l1 = [G.box(54, 156, 190, 44, "(5, 1.050)", { fill: "q", size: 14 }), G.box(354, 156, 190, 44, "(3, 1.135)", { fill: "q", size: 14 })];
          [[89, 149], [239, 149], [389, 449], [539, 449]].forEach(([a, b], i) => G.arrow(a, 96, (i < 2 ? 149 : 449), 154, { color: "muted" }));
          G.label(150, 222, "e^(2-5) + 1", { size: 12, anchor: "end" }); G.label(458, 222, "1 + e^(1-3)", { size: 12 });
          const l2 = G.box(194, 280, 250, 48, "(5, 1.203)", { fill: "ok", size: 16 });
          G.arrow(149, 202, 280, 278, { color: "muted" }); G.arrow(449, 202, 358, 278, { color: "muted" });
          G.label(194, 346, "1.050 + 1.135 · e^(3-5) = 1.050 + 0.154", { size: 12, color: "ink" });
          G.label(194, 366, "= direct sum e^-3 + 1 + e^-2 + e^-4 ✓", { size: 12, color: "ink" });
          G.from(l1, { opacity: 0, duration: 0.35, delay: 0.3 });
          G.from(l2, { opacity: 0, duration: 0.35, delay: 0.7 });
          G.caption("associative merge: any tree of lanes gives the same (m, d)");
        } },

      { rail: "registers", title: "One warp per row, the row in registers",
        body: `<p>Online softmax still reads the row twice: once for <code>(m, d)</code>, once to write <code>e^(x-m)/d</code>. If the whole row fits on chip, the second read can come from <b>registers</b> instead of DRAM, so DRAM sees exactly 1 read and 1 write: the floor.</p>
<p>How big a row fits? Give the row to one <b>warp</b> (32 lanes, P5.1). Lane number <code>lane</code> holds elements <code>lane, lane + 32, lane + 64, ...</code>, <code>VPL</code> values per lane (<code>softmax_v3</code> in <code>platform/kernels/include/d4/softmax.cuh</code>). With 32 values per lane the warp holds 32 × 32 = <b>1,024 columns</b>, 32 registers per lane. Interleaving by 32 means each <code>k</code> step is 32 consecutive floats: one coalesced 128-byte read (P5.2).</p>
<div class="eq">float v[VPL];                    // in registers
for k: v[k] = row[k*32 + lane];  // coalesced
m = warp_max(max of v);          // shuffles
for k: v[k] = __expf(v[k] - m);
s   = warp_sum(sum of v);
for k: out[k*32 + lane] = v[k] / s;</div>
<p>Because a row belongs to one warp, there is no <code>__syncthreads()</code>: warps never wait for each other, and all the reductions are shuffles. For 65,536 rows the launch is 65,536 × 32 / 256 = <b>8,192 blocks</b> of 8 warps. The cost is the 1,024-column ceiling: a 131,072-column row needs 1,024 registers per lane, which does not exist (the hardware limit is 255). Then the code falls back to the block-wide online version (<code>softmax_v2</code>).</p>`,
        scene(G) {
          G.text(24, 32, "a warp owns one row: lane l holds l, l+32, l+64, ...", { size: 13 });
          G.label(24, 54, "k = 0 → 31 reads (32 lanes → 32 consecutive floats = 128 B)", { size: 12 });
          const cw = 17, x0 = 40;
          const rows = [0, 1, 2, 3];
          const cells = [];
          rows.forEach((k, ri) => {
            G.label(x0 - 28, 84 + ri * 24 + 12, "k=" + k, { size: 11 });
            for (let l = 0; l < 32; l++) cells.push(G.rect(x0 + l * cw, 74 + ri * 24, cw - 2, 20, { fill: ri === 0 ? "k" : "w", rx: 2, opacity: ri === 0 ? 1 : 0.7 }));
          });
          G.label(x0 - 28, 84 + 4 * 24 + 12, " ⋮", { size: 12 });
          G.label(x0 - 28, 84 + 5 * 24 + 12 + 6, "k=31", { size: 11 });
          for (let l = 0; l < 32; l++) cells.push(G.rect(x0 + l * cw, 74 + 5 * 24 + 8, cw - 2, 20, { fill: "w", rx: 2, opacity: 0.7 }));
          G.label(x0, 74 + 6 * 24 + 26, "lane 0 ... lane 31", { size: 11 });
          G.rect(24, 270, 592, 118, { stroke: "line", rx: 8 });
          G.label(40, 294, "32 lanes × 32 values = 1,024 columns per warp-row", { size: 13, color: "ink" });
          G.label(40, 316, "DRAM: 1 read + 1 write per element (the floor)", { size: 13, color: "ok" });
          G.label(40, 338, "reductions: __shfl_xor_sync only, no __syncthreads", { size: 13, color: "ink" });
          G.label(40, 360, "longer rows (> 1,024 cols) fall back to the block version", { size: 13, color: "hot" });
          G.from(cells, { opacity: 0, stagger: 0.003, duration: 0.2 });
          G.caption("each k-slice is one coalesced 128-byte read");
        } },

      { rail: "RMSNorm", title: "RMSNorm: scale each row to unit size",
        body: `<p>Stacking dozens of layers makes activations drift to large or tiny magnitudes. A <b>normalization</b> layer rescales each token's vector (one row) before the next matmul. Llama-style models use <b>RMSNorm</b> (root mean square norm):</p>
<div class="eq">rms(x) = sqrt( mean(x²) + eps )
y_i    = x_i / rms(x) · w_i
(w: learned, one value per column)

x = [3, 4, 0, -5]
mean(x²) = (9 + 16 + 0 + 25) / 4 = 12.5
rms      = sqrt(12.5) = 3.536   (eps negligible)
x / rms  = [0.849, 1.131, 0, -1.414]
check: mean((x/rms)²)
     = (0.72 + 1.28 + 0 + 2.0) / 4 = 1.0 ✓</div>
<p>After dividing by the rms the row has root-mean-square 1, whatever its input scale; then <code>w</code> sets the learned scale per column. <code>eps</code> is a small constant that keeps the division finite for an all-zero row. LayerNorm also subtracts the mean; RMSNorm skips that step, which saves a reduction and works as well in practice.</p>
<p>As a kernel: one <b>block per row</b> (a Llama-3-8B row is 4,096 values, 16 KiB in fp32). Each thread accumulates squares of its slice (with <code>float4</code> loads, 16 bytes per instruction, from P5.2), a block reduction (P5.4) gives the sum, thread 0 computes <code>rsqrt(sum/cols + eps)</code>, and every thread scales its slice. The arithmetic is about 3 FLOPs per element against 4 to 8 bytes moved, so, like softmax, <b>it is bandwidth-bound</b> and the floor is the copy kernel.</p>`,
        scene(G) {
          const x = [3, 4, 0, -5], y = [0.849, 1.131, 0, -1.414];
          const zero1 = 150, zero2 = 150, s = 14;
          const bars = (x0, vals, col, ztop) => vals.map((v, i) => {
            const h = Math.abs(v) * s;
            return G.rect(x0 + i * 70, v >= 0 ? ztop - h : ztop, 48, Math.max(h, 1), { fill: col, rx: 3 });
          });
          G.text(40, 34, "x  (rms = 3.536)", { size: 14 });
          G.text(360, 34, "y = x / rms  (rms = 1)", { size: 14, color: "ok" });
          G.line(30, zero1, 320, zero1, { color: "line" });
          G.line(350, zero2, 630, zero2, { color: "line" });
          const b1 = bars(50, x, "q", zero1), b2 = bars(370, y, "ok", zero2);
          x.forEach((v, i) => G.text(74 + i * 70, v >= 0 ? zero1 - v * s - 8 : zero1 - v * s + 18, String(v), { anchor: "middle", size: 13 }));
          y.forEach((v, i) => G.text(394 + i * 70, v >= 0 ? zero2 - v * s - 8 : zero2 - v * s + 18, v.toFixed(2), { anchor: "middle", size: 13 }));
          G.arrow(320, 120, 350, 120, { color: "ink", w: 2 });
          G.label(40, 262, "1. each thread: Σ x² over its slice", { size: 13, color: "ink" });
          G.label(40, 284, "2. block sum → 50 / 4 = 12.5", { size: 13, color: "ink" });
          G.label(40, 306, "3. scale = rsqrt(12.5 + eps) = 0.283", { size: 13, color: "ink" });
          G.label(40, 328, "4. each thread: y = x · scale · w", { size: 13, color: "ink" });
          G.label(40, 360, "one block per row · float4 loads · fp32 accumulation", { size: 12 });
          G.from(b1.concat(b2), { attr: { height: 0 }, duration: 0.4, stagger: 0.05 });
          G.caption("same shape, unit rms: the norm only rescales the row");
        } },

      { rail: "fusion", title: "Fusion removes whole passes, not just launches",
        body: `<p>In a transformer block the data flow before each norm is: add the layer's output into the residual stream, then normalize it. Written as two kernels:</p>
<div class="eq">A (add):  r = x + r     rd x, rd r, wr r   = 3
B (norm): y = rmsnorm(r) rd r, wr y        = 2
                                   total   = 5

fused:    rd x, rd r, wr r, wr y           = 4</div>
<p>Kernel B has to re-read <code>r</code> only because kernel A wrote it to DRAM and ended. Fusing keeps the new <code>r</code> in registers or shared memory and normalizes it right there. That is 4 passes instead of 5: <b>20% less traffic, so up to 1.25× faster</b> for a bandwidth-bound op, plus one fewer kernel launch. (The weight <code>w</code> is read for every row but is small and stays in cache, so it is left out of the count.)</p>
<div class="eq">16,384 × 4,096 × 2 B (bf16) = 134.2 MB/pass
unfused 5 passes = 671 MB
fused   4 passes = 537 MB</div>
<p>Fusion is the general lesson of memory-bound GPU code: an intermediate that is written to DRAM and read straight back is a pass you can delete. <code>fused_add_rmsnorm</code> in <code>platform/kernels/include/d4/norms.cuh</code> does this; vLLM has the same op for the same reason, and swapping one such op into a serving engine is the capstone of this phase (P5.10). One detail: it normalizes the value that was <i>stored</i> (rounded to bf16), so its output matches the unfused path bit for bit.</p>`,
        check: { q: "Residual add + RMSNorm as two kernels moves 5 row-passes. Fused it moves 4. What is the best-case speed-up for a bandwidth-bound op?",
          options: ["About 1.25× (5/4)", "About 2×", "None: fusion only helps compute-bound kernels"], answer: 0,
          why: "Time is bytes divided by bandwidth, and bytes fall from 5 passes to 4, so time falls by 5/4 = 1.25× at best. The saving comes from not writing and re-reading r through DRAM. Fusion helps most precisely when a kernel is memory-bound." },
        scene(G) {
          const w = 58, x0 = 24;
          G.text(24, 34, "unfused: two kernels, 5 row-passes", { size: 13, color: "hot" });
          const sep = [["R", "x"], ["R", "r"], ["W", "r"], ["R", "r"], ["W", "y"]];
          const s1 = sep.map(([k, l], i) => blk(G, x0 + i * (w + 6) + (i >= 3 ? 14 : 0), 50, w, (k === "R" ? "rd " : "wr ") + l, k));
          G.label(x0, 96, "kernel A: add", { size: 11 }); G.label(x0 + 3 * (w + 6) + 14, 96, "kernel B: norm", { size: 11 });
          G.rect(x0 + 3 * (w + 6) + 7, 44, 1, 40, { fill: "muted" });
          G.text(24, 160, "fused: one kernel, 4 row-passes", { size: 13, color: "ok" });
          const fu = [["R", "x"], ["R", "r"], ["W", "r"], ["W", "y"]];
          const s2 = fu.map(([k, l], i) => blk(G, x0 + i * (w + 6), 176, w, (k === "R" ? "rd " : "wr ") + l, k));
          G.box(x0 + 4 * (w + 6) + 6, 176, 186, 26, "sum stays in registers", { stroke: "ok", color: "ok", size: 11, rx: 4 });
          G.rect(24, 250, 592, 134, { stroke: "line", rx: 8 });
          G.label(40, 276, "the second read of r is a pass the fused kernel never makes", { size: 13, color: "ink" });
          G.label(40, 300, "bytes: 5 → 4  (16,384 × 4,096 bf16: 671 MB → 537 MB)", { size: 13, color: "ink" });
          G.label(40, 324, "best case 5/4 = 1.25× faster, and 1 launch fewer", { size: 13, color: "ok" });
          G.label(40, 348, "w is re-read per row but stays in cache: left out", { size: 12 });
          G.label(40, 368, "cyan = DRAM read · amber = DRAM write", { size: 12 });
          G.from(s1.concat(s2), { opacity: 0, y: 10, stagger: 0.06, duration: 0.25 });
          G.caption("a value written then re-read is a pass fusion can delete");
        } },

      { rail: "cross-entropy", title: "Fused cross-entropy: one read of the biggest tensor",
        body: `<p>Training needs a loss. For a language model it is the <b>cross-entropy</b>: how surprised the model was by the correct next token, <code>-log p[target]</code>. The obvious code computes the softmax, stores the probabilities, then takes the log of one entry. For Llama 3's vocabulary (128,256) and 4,096 token positions the logits are the largest tensor in the model:</p>
<div class="eq">4,096 × 128,256 × 4 B (fp32) = 2.10 GB/pass
naive: read twice + write probs
     = 3 passes ≈ 6.3 GB, then a gather</div>
<p>Algebra removes it. With <code>p = e^(x - m) / d</code>:</p>
<div class="eq">-log p[t] = -(x_t - m) + log d
          = m + log(d) - x_t
(m + log d is called the logsumexp)

x = [2, 1, 0], target t = 0:
  m = 2, d = 1 + 0.368 + 0.135 = 1.503
  loss = 2 + ln 1.503 - 2 = 0.408   (= -ln 0.665 ✓)</div>
<p>So a kernel needs only the online <code>(m, d)</code> of the row plus the one logit at the target column: <b>one read of the logits, no probabilities written</b>. A thread that sees column <code>t</code> saves its value; the block merges <code>(m, d)</code> as in step 5. Exercise 4 builds this.</p>
<p>It is also more accurate. A confident wrong prediction, logits <code>[110, 0]</code> with target 1, has <code>p[1] = e^-110 ≈ 1.7e-48</code>. That is below the smallest fp32 value (about 1.4e-45), so it rounds to 0 and <code>-log 0 = inf</code>. The logsumexp form gives <code>110 + log(1 + e^-110) - 0 ≈ 110</code>, the right finite loss.</p>`,
        scene(G) {
          G.text(24, 34, "naive: 3 passes over a 2.10 GB tensor", { size: 13, color: "hot" });
          const nb = [["R", "m, d"], ["R", "x"], ["W", "p"]].map(([k, l], i) => blk(G, 24 + i * 112, 50, 104, (k === "R" ? "read " : "write ") + l, k));
          G.label(24, 98, "≈ 6.3 GB moved, then gather p[t]", { size: 12, color: "ink" });
          G.text(24, 140, "fused: 1 pass, no probabilities stored", { size: 13, color: "ok" });
          const fb = G.box(24, 156, 104, 26, "read x", { fill: "k", size: 11, rx: 4 });
          G.label(24, 202, "≈ 2.1 GB moved: m, d and x[t] built on the fly", { size: 12, color: "ink" });
          G.rect(24, 236, 592, 150, { stroke: "line", rx: 8 });
          G.text(40, 262, "logits [110, 0], target = 1", { size: 13 });
          G.box(40, 276, 300, 34, "-log(1.7e-48) → -log 0 = inf", { stroke: "hot", color: "hot", size: 12 });
          G.box(40, 320, 300, 34, "110 + log 1 - 0 = 110", { fill: "ok", size: 12 });
          G.label(356, 290, "e^-110 is below fp32's smallest", { size: 12, color: "hot" });
          G.label(356, 306, "positive value (~1.4e-45): it is 0", { size: 12, color: "hot" });
          G.label(356, 338, "logsumexp never forms the tiny p", { size: 12, color: "ok" });
          G.from(nb.concat([fb]), { opacity: 0, x: -20, stagger: 0.1, duration: 0.25 });
          G.caption("loss = m + log d - x[target]: fewer bytes and no underflow");
        } },

      { rail: "precision", title: "Store in bf16, accumulate in fp32, derive the tolerance",
        body: `<p>Activations are often stored as 16-bit numbers to halve the bytes. Both formats have a sign, an exponent (the range) and a mantissa (the precision):</p>
<table><tr><th>format</th><th>exp bits</th><th>significant bits</th><th>unit roundoff u</th><th>largest value</th></tr>
<tr><td>fp32</td><td>8</td><td>24</td><td>2<sup>-24</sup> ≈ 6.0e-8</td><td>3.4e38</td></tr>
<tr><td>fp16</td><td>5</td><td>11</td><td>2<sup>-11</sup> ≈ 4.9e-4</td><td><b>65,504</b></td></tr>
<tr><td>bf16</td><td>8</td><td>8</td><td>2<sup>-8</sup> ≈ 3.9e-3</td><td>3.4e38</td></tr></table>
<p><b>u</b> is the largest relative error from rounding one number to the format. bf16 keeps fp32's range but only 8 significant bits (7 stored plus an implicit 1). fp16 is more precise but overflows above 65,504, which a softmax numerator or a squared activation can exceed.</p>
<p>The rule is <b>store in 16 bits, accumulate in fp32</b>. Accumulating in bf16 breaks a sum quietly: above 256, bf16 numbers are 2 apart, so <code>256 + 1</code> rounds back to 256 and the sum stops growing. Summing 4,096 ones would return 256. (fp16 stalls at 2,048.)</p>
<p><b>Deriving a tolerance.</b> Count the roundings on the path. For RMSNorm with bf16 storage and fp32 math: the inputs are already bf16 (the test rounds them before computing the reference), the fp32 maths adds about 1e-7, and the output is rounded <i>once</i> to bf16. So the relative error is at most about u = 3.9e-3. A test <code>rtol = 1e-2</code> (about 2.5u) is therefore principled, not a number picked to make the test pass. Exercise 3 asks you to write this derivation down.</p>`,
        check: { q: "A kernel sums 4,096 copies of 1.0 using a bf16 accumulator. What does it return?",
          options: ["4,096", "About 4,000", "256: above 256 the spacing is 2, so adding 1 rounds back down"], answer: 2,
          why: "bf16 has 8 significant bits, so between 256 and 512 representable values are 2 apart. 256 + 1 = 257 is exactly halfway between 256 and 258 and rounds to the even neighbour 256, so the sum never moves again. Accumulating in fp32 (24 bits) avoids this." },
        scene(G) {
          const fmts = [["fp32", 1, 8, 23, "range 3.4e38 · u 6.0e-8"], ["fp16", 1, 5, 10, "max 65,504 · u 4.9e-4"], ["bf16", 1, 8, 7, "range 3.4e38 · u 3.9e-3"]];
          const unit = 17;
          G.text(24, 32, "bit layout (one box per bit)", { size: 13 });
          fmts.forEach(([name, s, e, m, note], i) => {
            const y = 54 + i * 62, x0 = 70;
            G.text(24, y + 18, name, { size: 13 });
            G.rect(x0, y, s * unit - 1, 24, { fill: "hot", rx: 2 });
            G.rect(x0 + s * unit, y, e * unit - 1, 24, { fill: "v", rx: 2 });
            G.rect(x0 + (s + e) * unit, y, m * unit - 1, 24, { fill: "k", rx: 2 });
            G.label(x0, y + 42, note, { size: 11 });
          });
          G.label(70, 244, "red: sign · amber: exponent (range) · teal: mantissa (precision)", { size: 11 });
          G.rect(24, 262, 592, 128, { stroke: "line", rx: 8 });
          G.text(40, 288, "sum of ones in a bf16 accumulator:", { size: 13 });
          G.label(40, 312, "255 + 1 = 256, then 256 + 1 = 257 → rounds to 256", { size: 12, color: "ink" });
          G.label(40, 332, "→ the total stalls at 256, however many ones follow", { size: 12, color: "hot" });
          G.text(40, 362, "fp32 accumulator: exact up to 16,777,216", { size: 13, color: "ok" });
          G.caption("16-bit storage halves the bytes; fp32 accumulators keep sums honest");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Softmax is exp then normalize; subtract the row max first or fp32 overflows to inf and NaN.</li>
<li>Softmax and norms do a few FLOPs per byte, so the metric is DRAM row-passes and GB/s against copy bandwidth, not FLOP/s.</li>
<li>Online softmax keeps (m, d) in one pass, rescaling d by e^(m - v) when the max grows; the merge of two states is associative, so lanes and warps can combine them.</li>
<li>A warp can hold a 1,024-column row in registers: one read, one write, no <code>__syncthreads</code>. Longer rows fall back to the block version.</li>
<li>Fusion deletes passes: add + RMSNorm goes from 5 to 4, and cross-entropy from 3 passes over a 2 GB tensor to 1 (m + log d - x[target]).</li>
<li>Store 16-bit, accumulate fp32, and derive tolerances from the unit roundoff: bf16 u = 2<sup>-8</sup>, so rtol 1e-2 is about 2.5u.</li></ul>`,
    sim: {
      title: "Row-pass calculator",
      intro: "Pick an operation, a shape, a data type and a GPU's example bandwidth. Each variant is described by how many times it moves a full tensor through DRAM; the chart shows the lower-bound time those bytes need at peak bandwidth.",
      height: 260,
      controls: [
        { id: "op", label: "operation", type: "select", value: "softmax", options: [["softmax", "softmax: safe / online / registers"], ["addnorm", "residual add + RMSNorm: unfused / fused"], ["ce", "cross-entropy: materialize / fused"]] },
        { id: "rows", label: "rows (tokens)", type: "select", value: 65536, options: [[1024, "1,024"], [4096, "4,096"], [16384, "16,384"], [65536, "65,536"]] },
        { id: "cols", label: "columns", type: "select", value: 1024, options: [[256, "256"], [1024, "1,024"], [4096, "4,096 (Llama-3-8B hidden)"], [32000, "32,000"], [128256, "128,256 (Llama 3 vocab)"]] },
        { id: "dtype", label: "storage type", type: "select", value: 4, options: [[4, "fp32 (4 B)"], [2, "bf16 / fp16 (2 B)"]] },
        { id: "gpu", label: "GPU bandwidth (example values)", type: "select", value: "t4", options: [["t4", "T4 · 320 GB/s"], ["l4", "L4 · 300 GB/s"], ["a100", "A100-80GB · 2,039 GB/s"]] },
      ],
      draw(G, v) {
        const g = GPUS[v.gpu], pass = v.rows * v.cols * v.dtype, MB = (b) => F.si(b, 1) + "B";
        let variants, note;
        if (v.op === "softmax") {
          const fits = v.cols <= 1024;
          variants = [["safe 3R+1W", 4, "hot"], ["online 2R+1W", 3, "v"], [fits ? "registers 1R+1W" : "registers: too long, online", fits ? 2 : 3, fits ? "ok" : "w"]];
          note = fits ? "" : "A row longer than 1,024 columns does not fit in one warp's registers, so v3 falls back to the 3-pass online version.";
        } else if (v.op === "addnorm") {
          variants = [["unfused 5 passes", 5, "hot"], ["fused 4 passes", 4, "ok"]];
          note = "Counts x, r reads and writes of r, y; the weight vector stays in cache.";
        } else {
          variants = [["materialize p: 3 passes", 3, "hot"], ["fused: 1 pass", 1, "ok"]];
          note = "Materialized = online softmax (2R + 1W of the logits), then a gather of one entry per row. Fused reads the logits once.";
        }
        const times = variants.map(([, p]) => (p * pass) / g.bw), tmax = Math.max(...times);
        const W = 560, x0 = 50, bw = Math.min(110, (W - 40) / variants.length - 24), gap = 28, base = 205, hmax = 125;
        G.axes(x0 - 10, 20, W, base - 20, {});
        variants.forEach(([name, p, col], i) => {
          const h = (hmax * times[i]) / tmax, x = x0 + 14 + i * (bw + gap);
          G.rect(x, base - h, bw, h, { fill: col, rx: 4 });
          G.text(x + bw / 2, base - h - 20, F.ms(times[i]), { anchor: "middle", size: 14 });
          G.label(x + bw / 2, base - h - 5, MB(p * pass), { anchor: "middle", size: 11 });
          G.label(x + bw / 2, base + 18, name, { anchor: "middle", size: 11 });
        });
        G.label(x0 - 10, 14, "lower-bound time at peak bandwidth", { size: 11 });
        const best = times[times.length - 1], worst = times[0];
        return [
          { title: "Traffic", rows: [["one row-pass", MB(pass)], ...variants.map(([n, p]) => [n, MB(p * pass)])] },
          { title: `Time on ${g.name}`, rows: [["bandwidth (example)", F.num(g.bw / 1e9) + " GB/s"], ...variants.map(([n], i) => [n, F.ms(times[i])]), ["best vs first", (worst / best).toFixed(2) + "×"]],
            html: `<p class="note">A lower bound: bytes ÷ peak bandwidth, with caches ignored. Real kernels reach a fraction of peak (the copy kernel is the achievable ceiling), but the ratios between variants are what the pass counts predict. Bandwidths are example values from gpu_specs.yaml, UNVERIFIED. ${note}</p>` },
        ];
      },
    },
    practice: {
      intro: `Build once per GPU session (see <code>course/P5-cuda-deep-dive/aws-common.md</code>): <code>cmake -S course/P5-cuda-deep-dive -B build/p5 -DCMAKE_CUDA_ARCHITECTURES=native &amp;&amp; cmake --build build/p5 -j</code>. Add <code>-DS2S_USE_SOLUTIONS=ON</code> (separate build dir) to run the tests against the reference kernels. Predict the GB/s from the pass counts before you measure. Results are not yet recorded in the repo (<code>TODO(run-on: g4dn.xlarge)</code>).`,
      items: [
        { title: "Online softmax in GB/s (L4 exit check)", tier: "T2", goal: "Implement softmax_rows: safe 3-pass, online 2-pass, then one warp per row in registers. The test includes logits in [500, 1000]. --bench reports GB/s for 1024 and 32000 columns.",
          cmd: "ctest --test-dir build/p5 -R p5.5_01-online-softmax --output-on-failure" },
        { title: "RMSNorm within X% of copy", tier: "T2", goal: "One block per row, float4 loads, fp32 accumulation. Measure a simple baseline first, write down X (the gap you commit to closing) in your notes, then optimize. fp32 test, rtol = 1e-5.",
          cmd: "ctest --test-dir build/p5 -R p5.5_02-rmsnorm --output-on-failure" },
        { title: "A bf16 tolerance you can justify", tier: "T0 + T2", goal: "Derive the rtol from the unit roundoff and the number of roundings on the path, run the fp16/bf16 example, and write a 4 to 6 line derivation. See the tests in platform/kernels/tests/test_softmax_norms.cu.",
          cmd: "./build/p5/p5.5_04_fp16_bf16 && ctest --test-dir build/d4 -R softmax --output-on-failure" },
        { title: "Fused cross-entropy (hard)", tier: "T2 · hard", goal: "loss = logsumexp - logit[target] in one read, vocabularies up to 128,256. Never compute log(softmax). --bench compares against half the copy bandwidth (read-only).",
          cmd: "ctest --test-dir build/p5 -R p5.5_04-cross-entropy --output-on-failure" },
      ],
      labs: [
        { label: "Softmax ladder: safe vs online vs registers, in GB/s", path: "course/P5-cuda-deep-dive/P5.5-softmax-and-norms/examples/01_softmax_ladder.cu" },
        { label: "RMSNorm: fp32 scalar vs float4 vs bf16", path: "course/P5-cuda-deep-dive/P5.5-softmax-and-norms/examples/02_rmsnorm.cu" },
        { label: "Fused residual add + RMSNorm vs two kernels", path: "course/P5-cuda-deep-dive/P5.5-softmax-and-norms/examples/03_fused_residual_rmsnorm.cu" },
        { label: "fp16 vs bf16 rounding and range", path: "course/P5-cuda-deep-dive/P5.5-softmax-and-norms/examples/04_fp16_bf16.cu" },
        { label: "The D4 kernels: softmax and norms", path: "platform/kernels/include/d4/softmax.cuh" },
        { label: "The D4 kernels: RMSNorm and fused add + RMSNorm", path: "platform/kernels/include/d4/norms.cuh" },
        { label: "Online softmax animation (running max and sum, merge)", path: "animations/p5-online-softmax.html" },
        { label: "AWS guide (g4dn.xlarge, about 2 instance-hours)", path: "course/P5-cuda-deep-dive/P5.5-softmax-and-norms/aws.md" },
        { label: "Reading: Milakov and Gimelshein, Online normalizer calculation for softmax (2018)", path: "https://arxiv.org/abs/1805.02867" },
      ],
    },
  });
})();
