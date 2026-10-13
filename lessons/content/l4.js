/* L4 — Fused elementwise ops and norms (Lane B). Concepts behind the L4 LeetGPU problems: fusion as byte counting,
   stable and online softmax, block-per-row kernels, RMSNorm/LayerNorm, Welford variance, fused residual+norm,
   cross entropy via log-sum-exp, number formats and vector loads, gated activations, RoPE. */
(function () {
  const F = S2S.fmt;
  const LV = "laneB-cuda/L4-fused-elementwise-and-norms";
  const mono = (h) => h.replace(/<code>(.*?)<\/code>/g, '<span style="font-family:var(--f-mono);font-size:13px">$1</span>');
  const P = (slug, title, diff, prio, goal) => ({
    title: `#${slug.split("-")[0]} ${title}`, tier: `${diff} · ${prio}`,
    goal: mono(`${goal} Hints and reference solution: <code>${LV}/solutions/${slug}/</code>`),
    cmd: `ctest --test-dir build/laneB -R '^L4_${slug}$' --output-on-failure`,
  });
  const nums = (G, x, y, vals, cw, ch, hl, base = "k", gap = 4) => vals.map((v, i) => {
    const c = hl ? hl(i) : null;
    return G.box(x + i * (cw + gap), y, cw, ch, String(v), { fill: c || base, size: 12, boxOpacity: c ? 1 : 0.5 });
  });
  const erf = (x) => { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; };

  S2S.lesson({
    id: "l4", n: "L4", title: "Fused elementwise ops and norms",
    subtitle: "Lane B · CUDA on LeetGPU · T1 browser or T2 harness",
    kicker: "Lesson · ≈ 70 min",
    headline: "Count the passes over memory",
    intro: `<p>The middle of a transformer layer is not matrix multiplication. It is softmax, normalization, residual adds, gated activations and rotary embeddings: small operations on large tensors. Each one does a few arithmetic operations per element it reads, so each is <b>memory-bound</b> (L1) and its cost is the number of times it passes over memory.</p>
<p>The dark panel teaches the one idea that makes these kernels fast: <b>fuse</b> operations so a value is read once, used for everything, and written once. Along the way: stable and <i>online</i> softmax, row-wise RMSNorm and LayerNorm (reductions from L3 inside a bigger kernel), the numerics of variance and of fp16/bf16, gated activations and RoPE. The Simulate tab counts bytes for fused and unfused versions.</p>`,
    facts: ["11 steps", "5 checkpoints", "1 simulator", "9 core problems"],
    legend: [["k", "read"], ["v", "write / updated"], ["hot", "problem"], ["ok", "good"]],
    prev: "l3", next: "l5",
    steps: [
      { rail: "fusion", title: "Every kernel boundary is a round trip to DRAM",
        body: `<p>Suppose a layer applies three elementwise operations to a tensor of N floats: <i>op1</i>, <i>op2</i>, <i>op3</i>. Written as three kernels, each one reads its input from global memory and writes its output back. The intermediate results travel to DRAM and straight back.</p>
<div class="eq">unfused:  3 kernels * (read 4N + write 4N) = 24N bytes
fused:    1 kernel  *  read 4N + write 4N  =  8N bytes</div>
<p>The arithmetic is the same. The bytes are one third. Since the kernels are memory-bound, time ≈ bytes ÷ bandwidth, so the fused version is about 3 times faster. It also launches one kernel instead of three, saving launch overhead.</p>
<p>The rule for this whole level: <b>count the row-passes</b>. One <i>pass</i> is one read or write of every element involved. Each fusion in this lesson removes passes: online softmax from 4 to 3 (or 2), residual-add plus norm from 5 to 4, SwiGLU from 5 to 3. Intermediate values stay in registers or shared memory (L2), where reuse is free.</p>
<p>Fusion has limits: the fused kernel needs the intermediate to be computable where it is consumed (same thread, same block), and it uses more registers or shared memory. That is why reductions (a whole row must be summed before anyone can scale) need care, as the next steps show.</p>`,
        scene(G) {
          const dram = (x, y, w, t) => { G.box(x, y, w, 34, t, { fill: "w", size: 12 }); };
          G.text(24, 30, "unfused: three kernels", { size: 13, color: "hot" });
          dram(24, 46, 590, "global memory (DRAM)");
          const xs = [60, 260, 460], ops = ["op 1", "op 2", "op 3"];
          xs.forEach((x, i) => {
            G.box(x, 128, 100, 38, ops[i], { fill: "k", size: 13 });
            G.arrow(x + 30, 82, x + 30, 126, { color: "k", w: 2 }); G.arrow(x + 70, 128, x + 70, 84, { color: "v", w: 2 });
          });
          G.label(24, 190, "6 trips: 3 reads + 3 writes = 24N bytes", { size: 12, color: "hot" });
          G.text(24, 250, "fused: one kernel", { size: 13, color: "ok" });
          dram(24, 266, 590, "global memory (DRAM)");
          G.box(180, 348, 280, 38, "op 1 · op 2 · op 3  (registers)", { fill: "ok", size: 13 });
          G.arrow(250, 302, 250, 346, { color: "k", w: 2 }); G.arrow(390, 348, 390, 304, { color: "v", w: 2 });
          G.label(24, 410, "2 trips: 1 read + 1 write = 8N bytes, 3× less", { size: 12, color: "ok" });
          G.caption("the intermediate results never leave the chip");
        } },

      { rail: "stable softmax", title: "Softmax overflows unless you subtract the max",
        body: `<p>A language model ends with <b>logits</b>, one score per vocabulary entry. <b>Softmax</b> turns scores into probabilities: <code>p_i = e^{x_i} / Σ_j e^{x_j}</code>. All p are positive and sum to 1.</p>
<p>The exponential grows fast. The largest float32 is about 3.4 × 10<sup>38</sup>, and e<sup>x</sup> reaches that at x ≈ 88.7 (ln of 3.4 × 10<sup>38</sup>). Logits of 100 or more are normal for a confident model; e<sup>100</sup> is <b>inf</b>, and inf / inf is NaN.</p>
<p>The fix uses a property of the formula: subtracting the same number c from every x leaves the result unchanged, because e<sup>x−c</sup> = e<sup>x</sup> · e<sup>−c</sup> and the e<sup>−c</sup> cancels top and bottom. Choose c = max(x). Now the biggest exponent is e<sup>0</sup> = 1, and nothing overflows:</p>
<div class="eq">x = [1000, 1001, 1002]        e^1002 overflows
x - max = [-2, -1, 0]
e^(x-max) = [0.1353, 0.3679, 1.0]    sum = 1.5032
p = [0.0900, 0.2447, 0.6652]          (sums to 1)</div>
<p>The straightforward stable kernel therefore makes three passes over the row: find the max, sum the exponentials, then write <code>e^{x−max} / sum</code>. That is 3 reads and 1 write: 4 row-passes. The next step removes one read.</p>`,
        scene(G) {
          G.text(24, 32, "x = [1000, 1001, 1002]", { size: 14 });
          G.box(24, 56, 180, 36, "e^x  → inf, NaN", { fill: "hot", size: 13 });
          G.label(214, 78, "e^88.7 is already the float32 maximum", { size: 11, color: "hot" });
          G.text(24, 134, "subtract max = 1002", { size: 14, color: "v" });
          nums(G, 24, 150, [-2, -1, 0], 60, 34, null, "k", 8);
          G.label(24, 208, "x − max", { size: 11 });
          nums(G, 24, 220, ["0.135", "0.368", "1.000"], 60, 34, () => "v", "k", 8);
          G.label(24, 278, "e^(x − max): never above 1", { size: 11 });
          G.text(24, 306, "divide by the sum 1.503", { size: 14, color: "ok" });
          const p = [0.09, 0.2447, 0.6652];
          p.forEach((v, i) => { G.rect(24 + i * 70, 374 - v * 80, 56, v * 80, { fill: "ok", rx: 4 }); G.label(30 + i * 70, 390, v.toFixed(3), { size: 11 }); });
          G.label(24, 408, "probabilities (sum 1)", { size: 11 });
          G.rect(330, 330, 280, 70, { stroke: "line", rx: 8 });
          G.label(342, 350, "3 passes over the row:", { size: 12, color: "ink" });
          G.label(342, 368, "max → sum → write", { size: 12 }); G.label(342, 386, "3 reads + 1 write = 4 passes", { size: 12, color: "hot" });
          G.caption("the shift changes nothing mathematically and prevents overflow");
        } },

      { rail: "online softmax", title: "Online softmax: update the sum as the max changes",
        body: `<p>Can we get max and sum in <i>one</i> read? The sum depends on the max, and the max is only known at the end. The trick: keep a running pair <code>(m, d)</code>, the max so far and the sum of <code>e^{x_j − m}</code> so far. When a new value v arrives:</p>
<ul><li>If <code>v ≤ m</code>: the max is unchanged, so <code>d += e^{v − m}</code>.</li>
<li>If <code>v &gt; m</code>: the max moves up. All old terms were measured against the old max. Rescale them by <code>e^{m − v}</code> (which is &lt; 1), then add the new term, which is <code>e^0 = 1</code>: <code>d = d · e^{m − v} + 1; m = v</code>.</li></ul>
<div class="eq">x = [1, 3, 2]          start m = -inf, d = 0
v = 1: new max   m = 1, d = 1
v = 3: new max   d = 1 * e^(1-3) + 1 = 1.1353, m = 3
v = 2: smaller   d = 1.1353 + e^(2-3) = 1.5032
direct: 0.1353 + 1 + 0.3679 = 1.5032</div>
<p>Same answer, one read. Two partial states merge the same way. Given (m<sub>a</sub>, d<sub>a</sub>) and (m<sub>b</sub>, d<sub>b</sub>): <code>M = max(m_a, m_b)</code> and <code>D = d_a · e^{m_a − M} + d_b · e^{m_b − M}</code>. That merge is associative, so it works in a thread loop, in a warp shuffle tree, and across blocks: everything from L3 applies. The same trick, applied to attention scores, is FlashAttention (L6).</p>
<p>With the final (M, D) the output pass writes <code>e^{x − M} / D</code>. Total: <b>2 reads and 1 write</b> (3 row-passes) instead of 4.</p>`,
        check: { q: "Online softmax state is (m, d) = (3, 1.1353) and the next value is v = 5. What is the new state?",
          options: ["(5, 1.1353 + e^(5-3)) = (5, 8.52)", "(5, 1.1353 · e^(3-5) + 1) = (5, 1.154)", "(3, 1.1353 + e^(5-3))"], answer: 1,
          why: "5 is a new maximum, so the old sum (measured against max 3) must be rescaled to the new max by e^(3−5) = 0.1353: 1.1353 × 0.1353 = 0.1536, and the new term contributes e^0 = 1. d = 1.1536 (rounded: 1.154) and m = 5. Adding e^(5−3) to the old sum would mix two different reference points." },
        scene(G) {
          G.text(24, 32, "x = [1, 3, 2], keeping (m, d)", { size: 14 });
          const hdr = ["", "m", "d", ""], rows = [["start", "−∞", "0", ""], ["v = 1", "1", "1", "new max"], ["v = 3", "3", "1·e^−2 + 1 = 1.135", "new max: rescale old d"], ["v = 2", "3", "1.135 + e^−1 = 1.503", "smaller: just add"]];
          rows.forEach((r, i) => {
            const y = 60 + i * 56;
            G.box(24, y, 80, 40, r[0], { fill: "w", size: 12 });
            G.box(112, y, 70, 40, r[1], { fill: i === 0 ? "w" : "v", size: 14, boxOpacity: i && r[1] !== rows[i - 1][1] ? 1 : 0.55 });
            G.box(190, y, 190, 40, r[2], { fill: i === 0 ? "w" : "k", size: 12, boxOpacity: i ? 0.9 : 0.5 });
            G.label(392, y + 24, r[3], { size: 11, color: r[3].startsWith("new") ? "v" : "ink" });
          });
          G.text(24, 306, "check: e^−2 + e^0 + e^−1 = 1.503 ✓", { size: 14, color: "ok" });
          G.rect(24, 328, 580, 70, { stroke: "line", rx: 8 });
          G.label(36, 350, "new max v:   d ← d · e^(m − v) + 1,  m ← v", { size: 12, color: "v" });
          G.label(36, 370, "else:           d ← d + e^(v − m)", { size: 12, color: "k" });
          G.label(36, 390, "merge:  D = d_a·e^(m_a−M) + d_b·e^(m_b−M)", { size: 12, color: "ink" });
          G.caption("one read gives both the max and a consistent sum");
        } },

      { rail: "block per row", title: "One block per row: threads, then warps, then block",
        body: `<p>A softmax row can be thousands to over a hundred thousand elements long (a vocabulary), and every row is independent. The reference kernel gives <b>one block to each row</b> (<code>${LV}/solutions/5-softmax/kernel.cu</code>):</p>
<ol><li>Each thread strides over the row (<code>c = threadIdx.x; c += blockDim.x</code>), keeping its own <code>(m, d)</code>. Reads are coalesced (L2).</li>
<li>The block finds the row max <code>M</code> (warp shuffles, then one shared slot per warp, as in L3, but with <code>__shfl_xor_sync</code> so every thread ends with the answer).</li>
<li>Each thread rescales its sum, <code>d · e^{m − M}</code>, and the block sums them to <code>D</code>.</li>
<li>A second loop over the row writes <code>e^{x − M} / D</code>.</li></ol>
<p>The row is read twice (once to build the state, once to write) and written once. Example: 1,024 rows of 32,768 fp32 values is 128 MiB. Four passes (the 3-read version) move 512 MiB, three move 384 MiB, a 1.33× saving. If the row fits in registers or shared memory (32,768 floats is 128 KiB, too big for the 48 KB the course kernels use, but 4,096 floats = 16 KiB fits), the second read is on-chip and the total is 2 passes, 256 MiB: 2× less than the naive version. GPU caches can also serve the second read from L2 for modest row sizes, which is why you must measure.</p>
<p><b>Exit check for L4:</b> run the softmax <code>--bench</code> on rows of 4k to 32k elements and report GB/s and the percentage of the measured copy bandwidth. Count 8 bytes per element if you treat the minimum as one read plus one write, and say which count you used.</p>`,
        scene(G) {
          G.text(24, 30, "one row, 8 threads shown: each keeps (m, d)", { size: 13 });
          Array.from({ length: 8 }, (_, t) => G.box(24 + t * 74, 46, 66, 36, `m${t}, d${t}`, { fill: "k", size: 11, boxOpacity: 0.85 }));
          for (let t = 0; t < 8; t++) G.arrow(57 + t * 74, 84, 57 + t * 74, 108, { color: "muted", w: 1 });
          G.box(24, 110, 584, 30, "block: M = max of m_t · D = Σ d_t · e^(m_t − M)  (shuffles + shared slots)", { fill: "v", size: 12 });
          G.text(24, 184, "row traffic, drawn to scale (R = read, W = write)", { size: 13 });
          const lanes = [["3 reads + write (naive)", ["R", "R", "R", "W"], "hot"], ["online: 2 reads + write", ["R", "R", "W"], "v"], ["row kept on-chip", ["R", "W"], "ok"]];
          lanes.forEach(([nm, seg, c], i) => {
            const y = 214 + i * 56;
            G.label(24, y - 4, nm, { size: 11, color: c });
            seg.forEach((s, j) => G.box(24 + j * 112, y, 108, 30, s === "R" ? "read row" : "write row", { fill: s === "R" ? "k" : "q", size: 11, boxOpacity: 0.9 }));
          });
          G.text(24, 400, "4 → 3 → 2 passes", { size: 15, color: "ok" });
          G.caption("each removed pass is bandwidth you get back");
        } },

      { rail: "RMSNorm & LayerNorm", title: "Normalization: rescale each row to a standard size",
        body: `<p>Activations in a deep network drift in scale from layer to layer. A <b>normalization</b> layer rescales each token's vector (one row of length <code>cols</code>) so the next layer sees a stable size. Two versions matter.</p>
<p><b>RMSNorm</b> (Llama): divide by the root mean square of the row, then multiply by a learned weight vector w:</p>
<div class="eq">y_i = x_i * rsqrt(mean(x^2) + eps) * w_i
x = [1 2 3 4], w = 1, eps ~ 0
mean(x^2) = (1 + 4 + 9 + 16) / 4 = 7.5
rsqrt(7.5) = 0.3651
y = [0.365 0.730 1.095 1.461]    (mean of y^2 = 1)</div>
<p><b>LayerNorm</b> (GPT-2, BERT) also subtracts the mean first, and adds a learned bias β: <code>y = (x − μ) / √(σ² + ε) · γ + β</code>, where σ² is the <i>population</i> variance (divide by n, not n − 1).</p>
<div class="eq">x = [1 2 3 4]:  mean 2.5
var = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
std = 1.118
y = [-1.342 -0.447 0.447 1.342]</div>
<p>Both are a <b>row reduction followed by an elementwise scale</b>: the same one-block-per-row, strided-loop and block-reduce pattern as softmax. RMSNorm needs one sum (Σx²); LayerNorm needs two statistics. In Llama-3-8B the row is 4,096 wide (hidden size, <code>platform/capacity/presets/llama3-8b.json</code>), ε is typically 1e-5 (the repo's NumPy reference defaults to it), and the reference model code is <code>rmsnorm</code> in <code>platform/engine/v0/reference/llama_numpy.py</code>.</p>
<p>The reference CUDA kernel <code>${LV}/solutions/50-rms-normalization/kernel.cu</code> adds <code>float4</code> loads (when <code>cols % 4 == 0</code>): 4,096 floats = 1,024 <code>float4</code>, so 512 threads take 2 each. Use <code>rsqrtf</code> once per row, not per element.</p>`,
        scene(G) {
          const draw = (x, y, vals, c, lab) => {
            G.label(x, y - 64, lab, { size: 11, color: "ink" });
            G.line(x - 4, y, x + 164, y, { color: "muted", w: 1 });
            const sc = 20;
            vals.forEach((v, i) => G.rect(x + i * 40, v >= 0 ? y - v * sc : y, 32, Math.abs(v) * sc, { fill: c, rx: 3, opacity: 0.9 }));
            vals.forEach((v, i) => G.label(x + i * 40 + 2, v >= 0 ? y + 14 : y + Math.abs(v) * sc + 14, v.toFixed(2), { size: 10 }));
          };
          G.text(24, 32, "same row x = [1, 2, 3, 4]", { size: 14 });
          draw(24, 150, [1, 2, 3, 4], "w", "input x");
          draw(24, 300, [0.365, 0.73, 1.095, 1.461], "k", "RMSNorm: x · rsqrt(7.5)");
          draw(340, 300, [-1.342, -0.447, 0.447, 1.342], "q", "LayerNorm: (x − 2.5) / 1.118");
          G.label(340, 100, "RMSNorm: one sum (Σx²)", { size: 12, color: "k" });
          G.label(340, 120, "LayerNorm: mean and variance", { size: 12, color: "q" });
          G.label(340, 140, "both: reduce a row, then scale it", { size: 12, color: "ink" });
          G.text(24, 388, "one block per row, strided loop, block reduce", { size: 13, color: "v" });
          G.caption("RMSNorm keeps the sign pattern; LayerNorm also centres the row");
        } },

      { rail: "variance", title: "Variance without cancellation: Welford and Chan",
        body: `<p>The textbook shortcut is one pass: accumulate Σx and Σx², then <code>var = E[x²] − E[x]²</code>. It is fast and it can be badly wrong. If the mean is large compared with the spread, the two terms are nearly equal and their difference is all rounding noise (<b>catastrophic cancellation</b>). Example: values around 10,000 with spread 1. E[x²] is about 10<sup>8</sup>, and float32 numbers near 10<sup>8</sup> are 8 apart, so the true variance 0.67 comes out as 0, 8 or even negative. A negative variance makes <code>rsqrt</code> return NaN. The test suite's <code>large_mean</code> case checks for this.</p>
<p><b>Welford's update</b> avoids subtracting big numbers. Keep (n, mean, M2), where M2 = Σ(x − mean)². For each new x:</p>
<div class="eq">n += 1;  delta = x - mean;
mean += delta / n;
M2 += delta * (x - mean);      // var = M2 / n
x = [2 4 6]:
x=2: n=1 mean=2 M2=0
x=4: n=2 delta=2 mean=3 M2=0 + 2*(4-3) = 2
x=6: n=3 delta=3 mean=4 M2=2 + 3*(6-4) = 8
var = 8/3 = 2.667   (check: 4 + 0 + 4 = 8)</div>
<p>Each thread runs Welford over its strided slice. To merge two states, use <b>Chan's formula</b>: <code>delta = mean_b − mean_a</code>, <code>n = n_a + n_b</code>, <code>mean = mean_a + delta · n_b / n</code>, <code>M2 = M2_a + M2_b + delta² · n_a · n_b / n</code>. Check with A = [2, 4] (n = 2, mean = 3, M2 = 2) and B = [6] (n = 1, mean = 6, M2 = 0): delta = 3, mean = 3 + 3 · 1/3 = 4, M2 = 2 + 0 + 9 · 2 · 1/3 = 8. ✓ Same result, and the merge is associative, so the shuffle tree from L3 applies, as in <code>${LV}/solutions/113-layer-normalization/kernel.cu</code>.</p>`,
        check: { q: "Welford state after the values [2, 4] is (n, mean, M2) = (2, 3, 2). The next value is 6. What is the state afterwards?",
          options: ["(3, 4, 8)", "(3, 4, 2)", "(3, 3, 8)"], answer: 0,
          why: "delta = 6 − 3 = 3; n = 3; mean = 3 + 3/3 = 4; M2 = 2 + delta · (6 − new mean) = 2 + 3 · 2 = 8. The population variance is 8/3 = 2.667, which matches the deviations from the mean 4: (−2)² + 0² + 2² = 8." },
        scene(G) {
          G.text(24, 30, "naive:  var = E[x²] − E[x]²", { size: 13, color: "hot" });
          G.box(24, 46, 590, 34, "values near 10,000, spread 1: E[x²] ≈ E[x]² ≈ 1.0002 × 10⁸", { fill: "w", size: 12 });
          G.label(24, 100, "float32 spacing near 10⁸ is 8: the true 0.67 is lost → 0, 8, or negative", { size: 12, color: "hot" });
          G.text(24, 146, "Welford on [2, 4, 6]", { size: 13, color: "ok" });
          const hdr = ["x", "n", "mean", "M2"], rows = [[2, 1, 2, 0], [4, 2, 3, 2], [6, 3, 4, 8]];
          hdr.forEach((h, j) => G.label(24 + j * 80, 168, h, { size: 12, color: "ink" }));
          rows.forEach((r, i) => r.forEach((v, j) => G.box(24 + j * 80, 176 + i * 40, 72, 34, String(v), { fill: j === 3 ? "v" : "k", size: 14, boxOpacity: 0.85 })));
          G.label(350, 196, "δ = x − mean(old)", { size: 11 }); G.label(350, 216, "mean += δ / n", { size: 11 });
          G.label(350, 236, "M2 += δ · (x − mean(new))", { size: 11 });
          G.text(24, 322, "Chan merge A=(2, 3, 2) with B=(1, 6, 0)", { size: 13, color: "q" });
          G.label(24, 346, "δ = 3 · mean = 3 + 3·1/3 = 4", { size: 12 });
          G.label(24, 366, "M2 = 2 + 0 + 3²·2·1/3 = 8 ✓ (same as one pass)", { size: 12 });
          G.text(24, 404, "subtract small deviations, never big squares", { size: 13, color: "ok" });
          G.caption("Welford is stable; Chan lets threads and warps merge their states");
        } },

      { rail: "residual + norm", title: "Fuse the residual add into the norm: one fewer pass",
        body: `<p>Each transformer sub-layer adds its output to a <b>residual</b> stream: <code>r = x + residual</code>, and the next sub-layer starts with a norm of <code>r</code>. As separate kernels:</p>
<ol><li>add: read x, read residual, write r (3 row-passes)</li>
<li>norm: read r, write y (2 row-passes)</li></ol>
<p>That is 5 passes. The fused kernel (<code>${LV}/solutions/83-fused-residual-rmsnorm/kernel.cu</code>) loads x and the residual once, computes <code>r</code>, writes it (the next layer's residual needs it), <b>keeps it in shared memory</b> while it sums the squares, and then scales from shared memory instead of re-reading DRAM:</p>
<div class="eq">unfused: 3 + 2 = 5 passes
fused:   read x, read res, write r, write y = 4 passes
8192 tokens * 4096 * 4 B = 128 MiB per pass
unfused 640 MiB, fused 512 MiB (saves 20%)</div>
<p>The row lives in dynamic shared memory (<code>extern __shared__ float row[]</code>), sized <code>cols * sizeof(float)</code> at launch. The reference allows up to 12,288 columns (48 KB). The saving is modest here (20%) but this exact fusion is a hot path in serving engines; in P5.10 you will swap your kernel into vLLM.</p>
<p>A numerical detail: if the residual is stored in bf16, normalize the <i>rounded</i> stored value, not the fp32 sum, so the fused result equals the two-kernel version.</p>
<p>The same thinking applies to <b>SwiGLU</b>: the gate and up projections of a Llama MLP produce two tensors that are immediately combined as <code>silu(gate) · up</code>. Fusing that into the matrix multiplication that produces them (L5) means the two tensors never reach DRAM at all.</p>`,
        check: { q: "Fused residual-add + RMSNorm reads x and residual, writes the new residual r, and writes y. Compared with the unfused add kernel followed by a norm kernel (5 passes), how many row-passes does it use?",
          options: ["4: it never re-reads r", "3: the residual write is skipped", "5: fusion only saves launches"], answer: 0,
          why: "Reads of x and residual (2) plus writes of r and y (2) make 4. The unfused version writes r and then reads it back for the norm, so the extra pass is that re-read. The write of r cannot be skipped because the next layer needs it as its residual." },
        scene(G) {
          G.text(24, 32, "row-passes, drawn to scale", { size: 13 });
          const seg = (x, y, lab, c) => G.box(x, y, 108, 34, lab, { fill: c, size: 11, boxOpacity: 0.9 });
          G.label(24, 66, "unfused: add kernel, then norm kernel", { size: 12, color: "hot" });
          [["read x", "k"], ["read res", "k"], ["write r", "v"], ["read r", "k"], ["write y", "v"]].forEach(([l, c], i) => seg(24 + i * 114, 76 + 0, l, c));
          G.rect(24, 118, 336, 6, { fill: "hot", rx: 3 }); G.label(24, 138, "add kernel: 3 passes", { size: 10 });
          G.rect(366, 118, 222, 6, { fill: "q", rx: 3 }); G.label(366, 138, "norm kernel: 2 passes", { size: 10 });
          G.label(24, 214, "fused: r kept in shared memory", { size: 12, color: "ok" });
          [["read x", "k"], ["read res", "k"], ["write r", "v"], ["write y", "v"]].forEach(([l, c], i) => seg(24 + i * 114, 224, l, c));
          G.rect(24, 268, 450, 6, { fill: "ok", rx: 3 });
          G.text(24, 330, "5 passes → 4 passes, and one launch fewer", { size: 14, color: "ok" });
          G.text(24, 358, "8192 × 4096 fp32 = 128 MiB per pass", { size: 13, color: "ink" });
          G.text(24, 384, "640 MiB → 512 MiB", { size: 13, color: "v" });
          G.caption("the pass that re-reads r disappears");
        } },

      { rail: "cross entropy", title: "Cross entropy: never take the log of a softmax",
        body: `<p>To train (or to score) a language model, the <b>cross-entropy loss</b> for one token is <code>−log p[label]</code>, where p is the softmax of the logits and <code>label</code> is the correct token. Take logits <code>[2.0, 1.0, 0.1]</code>, label 0: p<sub>0</sub> = e<sup>2</sup> / (e<sup>2</sup> + e<sup>1</sup> + e<sup>0.1</sup>) = 7.389 / 11.212 = 0.659, loss = −ln 0.659 = 0.417.</p>
<p>Computing it as log(softmax(x)[label]) is dangerous. For logits <code>[−200, 0]</code> and label 0, p<sub>0</sub> = e<sup>−200</sup> ≈ 10<sup>−87</sup>, far below the smallest float32 (about 10<sup>−38</sup>, or 10<sup>−45</sup> subnormal). It rounds to 0 and log(0) = −inf. The loss should be about 200.</p>
<p>Use the identity <code>−log p[label] = logsumexp(x) − x[label]</code>, with <code>logsumexp(x) = M + log(Σ e^{x−M})</code>. That is exactly the online softmax state: <code>M + log D</code>.</p>
<div class="eq">x = [2.0, 1.0, 0.1], label 0:
  logsumexp = ln(7.389 + 2.718 + 1.105)
            = ln 11.212 = 2.417
  loss = 2.417 - 2.0 = 0.417
x = [-200, 0], label 0:
  M = 0, D = 1 + e^-200 = 1, logsumexp = 0
  loss = 0 - (-200) = 200</div>
<p>One block per row, one read of the logits (the thread that owns column <code>label</code> saves its value in shared memory; the block reduction's barriers make it visible), then one <code>atomicAdd(loss/N)</code> per row. The logits are the biggest tensor in a language model's forward pass. For Llama-3's 128,256-token vocabulary, one fp32 row is 513,024 bytes (about 501 KiB), and 4,096 tokens make 2.1 GB of logits. Reading them once instead of three times matters. See <code>${LV}/solutions/25-categorical-cross-entropy/kernel.cu</code>.</p>`,
        scene(G) {
          G.text(24, 30, "logits [2.0, 1.0, 0.1], label 0", { size: 13 });
          [2.0, 1.0, 0.1].forEach((v, i) => { G.rect(24 + i * 70, 150 - v * 40, 56, v * 40, { fill: i === 0 ? "v" : "k", rx: 4, opacity: 0.9 }); G.label(30 + i * 70, 168, v.toFixed(1), { size: 11 }); });
          G.label(24, 186, "x_label = 2.0 (amber)", { size: 11, color: "v" });
          G.text(260, 70, "logsumexp = ln(e² + e¹ + e⁰·¹)", { size: 13, color: "ink" });
          G.text(260, 96, "= ln 11.212 = 2.417", { size: 13, color: "ink" });
          G.text(260, 130, "loss = 2.417 − 2.0 = 0.417", { size: 15, color: "ok" });
          G.line(24, 214, 616, 214, { color: "line", w: 1 });
          G.text(24, 244, "logits [−200, 0], label 0", { size: 13, color: "hot" });
          G.box(24, 260, 280, 34, "softmax[0] = e^−200 → 0 in float32", { fill: "hot", size: 11 });
          G.box(320, 260, 296, 34, "log(0) = −inf  ✗", { fill: "hot", size: 12 });
          G.box(24, 310, 280, 34, "logsumexp = 0 + ln(1 + e^−200) = 0", { fill: "ok", size: 11 });
          G.box(320, 310, 296, 34, "loss = 0 − (−200) = 200  ✓", { fill: "ok", size: 12 });
          G.text(24, 388, "loss = (M + log D) − x[label]: online (M, D), one read", { size: 13, color: "v" });
          G.caption("compute the loss in log space, never through a probability that can underflow");
        } },

      { rail: "formats & vectors", title: "fp16, bf16 and vector loads: fewer bytes, careful sums",
        body: `<p>Memory-bound kernels speed up when the numbers are smaller. A float32 has 1 sign bit, 8 exponent bits and 23 mantissa bits. The 16-bit formats trade differently:</p>
<ul><li><b>fp16</b>: 1 + 5 + 10 bits. Precision ≈ 2<sup>−11</sup> ≈ 4.9 × 10<sup>−4</sup> relative, but the largest value is only 65,504.</li>
<li><b>bf16</b>: 1 + 8 + 7 bits. The same range as fp32 (max about 3.4 × 10<sup>38</sup>), but precision only ≈ 2<sup>−8</sup> ≈ 3.9 × 10<sup>−3</sup>.</li>
<li><b>fp32</b>: precision 2<sup>−24</sup> ≈ 6.0 × 10<sup>−8</sup>.</li></ul>
<p>The rule: <b>store in 16 bits, accumulate in fp32</b>. A fp16 running sum stops growing at 2,048: the next float above is 2,050 (spacing 2), so 2,048 + 1 rounds back to 2,048. Summing 100,000 ones in fp16 would give 2,048 instead of 100,000 (the FP16 Dot Product test checks exactly that). Also, an fp16 × fp16 product is exact in fp32 (11 + 11 bits fit in 24), so the only error left is the summation order, and the test tolerance is the same as for fp32 dot product.</p>
<p><b>Tolerance by derivation.</b> With unit roundoff u, a sum of n products carried in fp32 has worst-case relative error about n · u<sub>32</sub> and a typical error about √n · u<sub>32</sub> (relative to Σ|aᵢbᵢ|). For n = 10<sup>6</sup>: worst ≈ 0.06, typical ≈ 6 × 10<sup>−5</sup>. Choose test tolerances from that, not by trial and error.</p>
<p><b>Vector loads.</b> One <code>float4</code> load moves 16 bytes in one instruction; a warp then reads 32 × 16 = 512 bytes at once. For fp16, <code>__half2</code> is 4 bytes per load. Fewer instructions per byte keeps the memory system busy. Requirement: the address must be aligned to the vector size. <code>cudaMalloc</code> returns 256-byte-aligned pointers, but <code>in + n</code> is 16-byte aligned only if <code>n % 4 == 0</code>, so the reference kernels check that and fall back to scalar loads. The FP16 Dot Product reference (<code>${LV}/solutions/58-fp16-dot-product/kernel.cu</code>) shows <code>__half22float2</code>, <code>fmaf</code> into an fp32 accumulator, and the odd-<code>n</code> tail.</p>`,
        check: { q: "Why does a pure fp16 running sum of 100,000 ones stall near 2,048?",
          options: ["fp16 overflows at 2,048", "Above 2,048 the spacing between fp16 values is 2, so adding 1 rounds back to the same value", "Ones cannot be represented in fp16"], answer: 1,
          why: "fp16 has an 11-bit significand. At 2,048 = 2^11 the gap to the next representable number is 2, so 2,048 + 1 = 2,049 rounds to even, 2,048, and the sum stops growing. Overflow happens at 65,504, and 1 is exactly representable. An fp32 accumulator (24-bit significand) stays exact up to 16,777,216." },
        scene(G) {
          const fm = [["fp32", [1, 8, 23], 8], ["fp16", [1, 5, 10], 8], ["bf16", [1, 8, 7], 8]], u = 15, cols = ["hot", "v", "k"];
          G.text(24, 30, "bit layout: sign | exponent | mantissa", { size: 13 });
          fm.forEach(([nm, b], r) => {
            const y = 56 + r * 56; G.label(24, y + 21, nm, { size: 12, color: "ink" });
            let x = 70; b.forEach((n, j) => { G.rect(x, y, n * u, 30, { fill: cols[j], rx: 4, opacity: 0.85 }); G.label(x + 4, y + 20, String(n), { size: 11, color: "bg" }); x += n * u + 3; });
          });
          G.label(70, 222, "red: sign · amber: exponent (range) · cyan: mantissa (precision)", { size: 11 });
          G.text(24, 262, "fp16 range ±65,504 · bf16 and fp32 range ±3.4e38", { size: 12, color: "ink" });
          G.text(24, 290, "sum of 100,000 ones:", { size: 13 });
          G.box(24, 304, 270, 34, "fp16 accumulator → stalls at 2,048", { fill: "hot", size: 11 });
          G.box(310, 304, 300, 34, "fp32 accumulator → 100,000", { fill: "ok", size: 11 });
          G.text(24, 372, "float4: 16 B/lane, 512 B per warp · half2: 4 B/lane", { size: 12, color: "v" });
          G.label(24, 398, "store small, accumulate in fp32; vector loads need alignment", { size: 11, color: "ink" });
          G.caption("fewer bytes per element, but never a 16-bit running sum");
        } },

      { rail: "gated activations", title: "SwiGLU and GEGLU: an activation times a second projection",
        body: `<p>A Llama MLP does not simply apply an activation. It computes two projections of the same input, a <b>gate</b> a = x·W<sub>gate</sub> and an <b>up</b> value b = x·W<sub>up</sub>, and combines them: <code>out = act(a) · b</code>. With <code>act = silu</code> this is <b>SwiGLU</b>; with GELU it is <b>GEGLU</b>.</p>
<div class="eq">silu(x) = x * sigmoid(x) = x / (1 + e^-x)
silu(1)  =  1 / (1 + 0.3679) =  0.7311
silu(-1) = -1 / (1 + 2.7183) = -0.2689
a = 1, b = 3:  silu(1) * 3 = 2.193</div>
<p>The kernel is purely elementwise: 2 reads and 1 write per output. In fp32 that is 12 bytes per element, memory-bound again. For Llama-3-8B the intermediate size is 14,336 (<code>platform/capacity/presets/llama3-8b.json</code>), so a token reads 2 × 14,336 values and writes 14,336. Implementation points from <code>${LV}/solutions/54-swiglu/</code>: grid-stride loop, <code>__expf</code> is accurate enough (a few ulp), and <code>float4</code> loads for both halves when <code>n % 4 == 0</code>.</p>
<p><b>GELU</b> is <code>x · Φ(x) = ½ x (1 + erf(x / √2))</code>, a smooth version of ReLU. Many libraries use the faster <b>tanh approximation</b> <code>½ x (1 + tanh(√(2/π) (x + 0.044715 x³)))</code>. They differ by up to about 10<sup>−3</sup> (per the reference README: measure it), which fails a tight test tolerance if the problem expects the other one. The challenge set has no plain GELU, so GEGLU stands in for it. Read the statement for which half is gated and which GELU is meant.</p>
<p>Like the norm fusions, the best home for these is the epilogue of the matrix multiplication that produces a and b (L5), so the 2 × 14,336 intermediate values per token never touch DRAM.</p>`,
        scene(G) {
          const x0 = 50, y0 = 270, sx = 62, sy = 40;
          G.text(24, 30, "SiLU and exact GELU, x from −4 to 4", { size: 13 });
          G.line(x0 - 20, y0, x0 + 8 * sx + 10, y0, { color: "muted", w: 1 }); G.line(x0 + 4 * sx, y0 - 140, x0 + 4 * sx, y0 + 50, { color: "muted", w: 1 });
          const f1 = (x) => x / (1 + Math.exp(-x)), f2 = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));
          [[f1, "k"], [f2, "v"]].forEach(([f, c]) => { let d = ""; for (let i = 0; i <= 80; i++) { const x = -4 + i * 0.1; d += (i ? " L " : "M ") + (x0 + (x + 4) * sx).toFixed(1) + " " + (y0 - f(x) * sy).toFixed(1); } G.path(d, { color: c, w: 2.5 }); });
          G.label(x0 + 6 * sx, y0 - 3 * sy - 6, "SiLU", { size: 12, color: "k" }); G.label(x0 + 6 * sx, y0 - 3 * sy + 18, "GELU", { size: 12, color: "v" });
          for (let x = -4; x <= 4; x += 2) G.label(x0 + (x + 4) * sx - 6, y0 + 18, String(x), { size: 10 });
          G.box(24, 336, 130, 36, "gate a", { fill: "k", size: 12 }); G.box(24, 380, 130, 36, "up b", { fill: "q", size: 12 });
          G.text(170, 360, "out = act(a) · b", { size: 15, color: "ok" });
          G.label(380, 354, "2 reads + 1 write = 12 B/elem (fp32)", { size: 11, color: "ink" });
          G.label(380, 374, "silu(1) = 0.731 → ×3 = 2.193", { size: 11, color: "v" });
          G.caption("a smooth gate times a linear value; elementwise and memory-bound");
        } },

      { rail: "RoPE", title: "RoPE: rotate each pair of dimensions by a position-dependent angle",
        body: `<p>Attention needs to know <i>where</i> a token is. <b>Rotary position embedding (RoPE)</b> encodes position by rotating the query and key vectors. Treat the head vector (D = 128 numbers in Llama-3-8B) as D/2 pairs. Pair i is rotated in its own 2D plane by an angle that grows with the token position p:</p>
<div class="eq">theta_i(p) = p * base^(-2i / D)
x'_a = x_a * cos(theta) - x_b * sin(theta)
x'_b = x_b * cos(theta) + x_a * sin(theta)

D = 128, base = 10,000, position p = 1:
i = 0:   theta = 1 * 10000^0     = 1 rad
         pair (1, 0) → (0.5403, 0.8415)
i = 1:   theta = 10000^(-2/128)  = 0.866 rad
i = 63:  theta = 10000^(-126/128) = 1.16e-4 rad</div>
<p>Low pairs rotate fast, high pairs slowly, like the hands of a clock. Rotating the query by p<sub>q</sub> and the key by p<sub>k</sub> makes their dot product depend only on p<sub>q</sub> − p<sub>k</sub>: relative position, for free. Llama-3 uses a base (<code>rope_theta</code>) of 500,000; the repo's NumPy engine takes it from the model config (<code>rope_tables</code> and <code>apply_rope</code> in <code>platform/engine/v0/reference/llama_numpy.py</code>).</p>
<p><b>Which two elements form a pair?</b> Two conventions exist and they are not interchangeable. <b>rotate_half</b> (Llama, Hugging Face) pairs <code>(i, i + D/2)</code>. <b>Interleaved</b> (GPT-NeoX/GPT-J style) pairs <code>(2i, 2i + 1)</code>. A wrong choice passes every shape check and gives wrong numbers.</p>
<p>Kernel (<code>${LV}/solutions/61-rotary-positional-embedding/</code>): one block per (token, head), one thread per pair, in place, because each pair is read and written by one thread. The angle comes from <code>sincosf</code> (or a precomputed cos/sin table: memory versus compute, measure both). Large angles reduce <code>sincosf</code> accuracy, so compute <code>p · inv_freq</code> in fp32 or better and scale the test tolerance. RoPE runs on q and k in every layer for every token, so it is usually fused into the QKV projection epilogue (L5).</p>`,
        check: { q: "With D = 8 and the rotate_half convention, which element is paired with element 1?",
          options: ["Element 0", "Element 5", "Element 2"], answer: 1,
          why: "rotate_half pairs i with i + D/2 = i + 4, so (0,4), (1,5), (2,6), (3,7). The interleaved convention would pair 1 with 0 as (0,1), (2,3), … The two layouts are different permutations of the same math, so the kernel must match the checkpoint's convention." },
        scene(G) {
          const cx = 150, cy = 150, R = 90;
          G.text(24, 30, "one pair = a point in a plane", { size: 13 });
          G.circle(cx, cy, R, { stroke: "line" });
          G.line(cx - R - 10, cy, cx + R + 10, cy, { color: "muted", w: 1 }); G.line(cx, cy - R - 10, cx, cy + R + 10, { color: "muted", w: 1 });
          G.arrow(cx, cy, cx + R, cy, { color: "k", w: 2.5 });
          const th = 1, ex = cx + R * Math.cos(th), ey = cy - R * Math.sin(th);
          G.arrow(cx, cy, ex, ey, { color: "v", w: 2.5 });
          G.path(`M ${cx + 34} ${cy} A 34 34 0 0 0 ${cx + 34 * Math.cos(th)} ${cy - 34 * Math.sin(th)}`, { color: "ok", w: 2 });
          G.label(cx + 40, cy - 12, "θ = 1 rad", { size: 11, color: "ok" });
          G.label(cx + R + 10, cy + 14, "(1, 0)", { size: 11, color: "k" }); G.label(ex + 8, ey - 4, "(0.540, 0.841)", { size: 11, color: "v" });
          G.label(310, 70, "θ_i = p · base^(−2i/D)", { size: 13, color: "ink" });
          G.label(310, 94, "i = 0: 1 rad (fast)", { size: 12 }); G.label(310, 114, "i = 1: 0.866 rad", { size: 12 }); G.label(310, 134, "i = 63: 0.000116 rad (slow)", { size: 12 });
          G.label(310, 160, "(D = 128, base 10,000, p = 1)", { size: 11 });
          G.text(24, 268, "rotate_half: pairs (i, i + D/2)", { size: 12, color: "k" });
          const cell = (x, y, i, c) => G.box(x + i * 44, y, 38, 30, String(i), { fill: c, size: 12, boxOpacity: 0.85 });
          for (let i = 0; i < 8; i++) cell(24, 304, i, i % 4 === 1 ? "v" : "k");
          [[0, 4], [1, 5], [2, 6], [3, 7]].forEach(([a, b], j) => G.path(`M ${43 + a * 44} 302 C ${43 + a * 44} 284, ${43 + b * 44} 284, ${43 + b * 44} 302`, { color: ["k", "v", "q", "ok"][j], w: 1.5 }));
          G.text(24, 352, "interleaved: pairs (2i, 2i + 1)", { size: 12, color: "q" });
          for (let i = 0; i < 8; i++) cell(24, 380, i, "q");
          [[0, 1], [2, 3], [4, 5], [6, 7]].forEach(([a, b]) => G.path(`M ${43 + a * 44} 378 C ${43 + a * 44} 364, ${43 + b * 44} 364, ${43 + b * 44} 378`, { color: "pink", w: 1.5 }));
          G.caption("same rotation, two ways to choose the pairs: match your model");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Memory-bound kernels are costed in row-passes. Fusing removes passes by keeping intermediates in registers or shared memory.</li>
<li>Softmax subtracts the row max. Online softmax keeps (m, d), rescales d when the max moves, and merges associatively: 2 reads + 1 write instead of 3 + 1.</li>
<li>RMSNorm and LayerNorm are one block per row: strided loop, block reduce, scale. Variance needs Welford (with Chan's merge), not E[x²] − E[x]².</li>
<li>Residual-add + norm drops 5 passes to 4; cross entropy is logsumexp − x[label], never log(softmax).</li>
<li>Store in fp16/bf16, accumulate in fp32; derive tolerances from unit roundoff; vector loads need aligned addresses.</li>
<li>Gated activations are elementwise (2 reads, 1 write). RoPE rotates pairs: know the pairing convention.</li></ul>
<p>Next, L5 takes the other half of the layer: matrix multiplication, tiles, tensor cores, and the epilogues where these fused ops live.</p>`,
    sim: {
      title: "Count the row-passes",
      intro: "Pick an operation, a row width and how many rows (tokens). Each bar is one implementation, drawn as the bytes it must move if intermediates are not cached. The chosen variant shows how many row-passes it needs, the total traffic, and the lower-bound time at the bandwidth you enter. The widths are real Llama-3-8B and vocabulary sizes.",
      height: 300,
      controls: [
        { id: "fam", label: "operation", type: "select", value: "softmax", options: [["softmax", "softmax (rows)"], ["resnorm", "residual + RMSNorm"], ["swiglu", "SwiGLU (gate, up)"]] },
        { id: "var", label: "variant (0 = unfused)", type: "select", value: 1, options: [[0, "first / naive"], [1, "second"], [2, "third (softmax only)"]] },
        { id: "w", label: "row width", type: "select", value: 4096, options: [[1024, "1,024"], [4096, "4,096 (hidden)"], [14336, "14,336 (MLP)"], [32768, "32,768"], [128256, "128,256 (vocab)"]] },
        { id: "rows", label: "rows (tokens)", min: 1, max: 32768, step: 1, value: 8192 },
        { id: "bytes", label: "bytes per element", type: "select", value: 4, options: [[2, "2 (fp16 / bf16)"], [4, "4 (fp32)"]] },
        { id: "bw", label: "memory bandwidth GB/s (example value)", min: 50, max: 1000, step: 10, value: 300 },
      ],
      draw(G, v) {
        const FAM = {
          softmax: [["naive: max, sum, write (3 reads + 1 write)", 4], ["online: 2 reads + 1 write", 3], ["row kept on-chip: 1 read + 1 write", 2]],
          resnorm: [["unfused: add kernel + norm kernel", 5], ["fused: read x, res; write r, y", 4]],
          swiglu: [["unfused: silu kernel then multiply kernel", 5], ["fused: read gate, up; write out", 3]],
        }[v.fam];
        const idx = Math.min(v.var, FAM.length - 1), n = v.rows * v.w, passBytes = n * v.bytes;
        const maxP = Math.max(...FAM.map((f) => f[1])), x0 = 20, wmax = 600;
        FAM.forEach(([name, p], i) => {
          const y = 24 + i * 78, sel = i === idx;
          G.label(x0, y - 4, name, { size: 11, color: sel ? "ink" : "muted" });
          for (let k = 0; k < p; k++) G.rect(x0 + k * (wmax / maxP), y + 4, wmax / maxP - 4, 30, { fill: sel ? "k" : "line", rx: 4, opacity: sel ? 0.9 : 0.8 });
          G.label(x0 + 6, y + 54, `${p} passes · ${F.bytes(p * passBytes)}`, { size: 11, color: sel ? "ok" : "muted" });
        });
        const p = FAM[idx][1], tot = p * passBytes, ms = tot / (v.bw * 1e9) * 1e3, base = FAM[0][1], rowB = v.w * v.bytes;
        const fits = rowB <= 48 * 1024;
        const rowsPanel = [["row bytes", F.bytes(rowB)], ["one pass over all rows", F.bytes(passBytes)], ["row fits in 48 KB shared memory", fits ? "yes" : "no"]];
        return [
          { title: "Selected variant", rows: [["row-passes", String(p)], ["bytes moved", F.bytes(tot)], ["time ≥ bytes / BW", `${ms.toFixed(3)} ms`], ["vs first variant", `${(100 * p / base).toFixed(0)}% of the bytes`]],
            gauge: [[p / base, "k"], [1 - p / base, "line"]], gaugeText: "bytes relative to the unfused/naive version",
            chip: [p < base, p < base ? `${(base / p).toFixed(2)}× fewer bytes` : "baseline"] },
          { title: "Row size", rows: rowsPanel,
            html: `<p class="note">Counts treat every pass as DRAM traffic. GPU caches can serve re-reads of small rows, so measured gains are smaller; measure with --bench. “On-chip” softmax needs the whole row in registers or shared memory (the course kernels use up to 48 KB). Bandwidth is an example value.</p>` },
        ];
      },
      note: "Pass counts come from the module's hint ladders: softmax 4/3/2 (#5), residual+RMSNorm 5/4 (#83), SwiGLU 5/3 (#54, silu kernel reads a and writes s; multiply reads s and b and writes out).",
    },
    practice: {
      intro: `<p>Solve each problem on <a href="https://leetgpu.com/challenges">LeetGPU</a> first (T1, in the browser). We don't reproduce the statements; read them there, especially for epsilon, which half is gated, the GELU variant, and the RoPE convention. Then run the local reference and test (T2, needs an NVIDIA GPU and CUDA ≥ 12.4). Build once from the repository root:</p>
<p><code>cmake -S laneB-cuda -B build/laneB -DCMAKE_BUILD_TYPE=Release &amp;&amp; cmake --build build/laneB -j</code></p>
<p><b>Exit check:</b> run <code>./build/laneB/L4_5-softmax --bench</code> (rows of 4k to 32k elements) and report GB/s and the percentage of the copy bandwidth the harness measures (<code>TODO(run-on: g4dn.xlarge)</code> until run). State your byte count per element.</p>`,
      items: [
        P("5-softmax", "Softmax", "medium", "core · exit check", "Single-pass online (max, sum) per row, block merge, normalizing pass."),
        P("50-rms-normalization", "RMS Normalization", "medium", "core", "One block per row, block-reduced sum of squares, <code>float4</code> loads, one <code>rsqrtf</code>."),
        P("113-layer-normalization", "Layer Normalization", "medium", "core", "Welford per thread, Chan merge across warps, population variance; the large-mean test."),
        P("83-fused-residual-rmsnorm", "Fused Residual Add and RMS Norm", "medium", "core", "Keep the summed row in shared memory between the reduction and the scale."),
        P("25-categorical-cross-entropy", "Categorical Cross Entropy Loss", "medium", "core", "logsumexp minus the label logit from the online state; one atomic per row."),
        P("54-swiglu", "Swish-Gated Linear Unit", "easy", "core", "silu(gate) times up, <code>float4</code> when n is a multiple of 4."),
        P("65-geglu", "Gaussian Error Gated Linear Unit", "easy", "core", "Exact GELU with <code>erff</code>; know the tanh approximation's error."),
        P("58-fp16-dot-product", "FP16 Dot Product", "medium", "core", "<code>half2</code> loads, fp32 accumulation, odd-n tail."),
        P("61-rotary-positional-embedding", "Rotary Positional Embedding", "medium", "core", "rotate_half pairs, <code>sincosf</code> on the fly versus a table."),
        P("40-batch-normalization", "Batch Normalization", "medium", "practice", "Per-channel statistics across the batch."),
        P("105-group-normalization", "Group Normalization", "medium", "practice", "A reduction over a group of channels."),
        P("106-token-embedding", "Token Embedding Layer", "medium", "practice", "A row gather: memory-bound."),
        P("64-weight-dequantization", "Weight Dequantization", "medium", "practice", "Dequantize in the read path instead of a separate pass."),
        P("60-top-p-sampling", "Top-p Sampling", "medium", "stretch", "Sort-free threshold search over the probabilities (outline in the hints)."),
        P("104-min-p-sampling", "Min-P Sampling", "medium", "stretch", "Keep tokens above a fraction of the top probability (outline in the hints)."),
        P("67-moe-top-k-gating", "MoE Top-K Gating", "medium", "stretch", "The router: softmax plus top-k per token (outline in the hints)."),
      ],
      labs: [
        { label: "Level syllabus, objectives and exit check", path: `${LV}/syllabus.md` },
        { label: "Problem map with links and priorities", path: `${LV}/leetgpu-map.md` },
        { label: "The harness: CUDA_CHECK, time_gpu, measure_copy_gbs, report", path: "laneB-cuda/harness/" },
        { label: "Online softmax animation (Lane A P5.5)", path: "animations/p5-online-softmax.html" },
        { label: "NumPy reference: rmsnorm, rope_tables, apply_rope, silu, softmax", path: "platform/engine/v0/reference/llama_numpy.py" },
        { label: "AWS T2 box for L1–L4 (T4, sm_75): instance_type = \"g4dn.xlarge\", then make down", path: "infra/aws/single-node" },
      ],
    },
  });
})();
