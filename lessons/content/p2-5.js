/* P2.5 — Quantization. Fewer bytes per number is faster decode and more KV room; this lesson builds the number
   formats from bits, then scales, groups, GPTQ, AWQ, the KV cache, and how to measure the quality cost honestly. */
(function () {
  const F = S2S.fmt;
  const P8 = 8.030261248e9;           // Llama-3-8B parameters (P1.1)

  /* ---------- number helpers shared by scenes and the simulator ---------- */
  // all non-negative FP8 values for a format, sorted
  function fp8Table(fmt) {
    const [E, M, bias] = fmt === "e5m2" ? [5, 2, 15] : [4, 3, 7], out = [];
    for (let e = 0; e < 2 ** E; e++) for (let m = 0; m < 2 ** M; m++) {
      if (fmt === "e4m3" && e === 15 && m === 7) continue;          // NaN in the "fn" variant
      if (fmt === "e5m2" && e === 31) continue;                      // inf / NaN
      out.push(e === 0 ? m * 2 ** (1 - bias - M) : (1 + m / 2 ** M) * 2 ** (e - bias));
    }
    return out.sort((a, b) => a - b);
  }
  const T43 = fp8Table("e4m3");
  function roundTo(table, x) {
    const a = Math.abs(x), mx = table[table.length - 1];
    if (a >= mx) return Math.sign(x) * mx;
    let lo = 0, hi = table.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (table[mid] <= a) lo = mid; else hi = mid; }
    return Math.sign(x) * (a - table[lo] <= table[hi] - a ? table[lo] : table[hi]);
  }
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  function bf16(x) { f32[0] = x; const b = u32[0]; u32[0] = ((b + 0x7fff + ((b >>> 16) & 1)) >>> 0) & 0xffff0000; return f32[0]; }
  function rng(seed) { let a = seed * 2654435761 + 11; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  // synthetic weights: rows × cols Gaussian (std 0.02) with a fraction of outliers at `mag`× (like examples/01_formats.py)
  function weights(seed, rows, cols, frac, mag) {
    const R = rng(seed), w = new Float64Array(rows * cols);
    for (let i = 0; i < w.length; i++) {
      const g = Math.sqrt(-2 * Math.log(1 - R())) * Math.cos(2 * Math.PI * R());
      w[i] = 0.02 * g * (R() < frac ? mag : 1);
    }
    return w;
  }
  // int quantize + dequantize with per-group scales along each row
  function intQ(w, cols, bits, group, sym) {
    const out = new Float64Array(w.length), qmax = 2 ** (bits - 1) - 1, levels = 2 ** bits - 1;
    for (let s = 0; s < w.length; s += group) {
      let mx = 0, mn = Infinity, MX = -Infinity;
      for (let i = s; i < s + group; i++) { mx = Math.max(mx, Math.abs(w[i])); mn = Math.min(mn, w[i]); MX = Math.max(MX, w[i]); }
      if (sym) { const sc = mx / qmax || 1; for (let i = s; i < s + group; i++) out[i] = Math.max(-qmax - 1, Math.min(qmax, Math.round(w[i] / sc))) * sc; }
      else { const sc = (MX - mn) / levels || 1, z = Math.round(-mn / sc); for (let i = s; i < s + group; i++) out[i] = (Math.max(0, Math.min(levels, Math.round(w[i] / sc) + z)) - z) * sc; }
    }
    return out;
  }
  function fp8Q(w, cols) {
    const out = new Float64Array(w.length);
    for (let r = 0; r < w.length; r += cols) {
      let mx = 0; for (let i = r; i < r + cols; i++) mx = Math.max(mx, Math.abs(w[i]));
      const sc = mx / 448 || 1; for (let i = r; i < r + cols; i++) out[i] = roundTo(T43, w[i] / sc) * sc;
    }
    return out;
  }
  const relRMS = (w, q) => { let e = 0, s = 0; for (let i = 0; i < w.length; i++) { e += (w[i] - q[i]) ** 2; s += w[i] ** 2; } return Math.sqrt(e / s); };

  S2S.lesson({
    id: "p2-5", n: "P2.5", title: "Quantization",
    subtitle: "Serving engines · first principles · T0 maths, T2 bakeoff (L4 for FP8)",
    kicker: "Lesson · ≈ 45 min",
    headline: "Fewer bits, same answers?",
    intro: `<p>Decode reads every weight once per step (P1.2), so the number of <b>bytes per weight</b> sets its speed limit, and the bytes per KV number set how many users fit (P2.1). Quantization stores those numbers in fewer bits. This lesson builds the formats from the bits up, shows exactly where the error comes from and how scales, groups, GPTQ and AWQ shrink it, and ends with how to measure the quality cost without fooling yourself.</p>`,
    facts: ["10 steps", "4 checkpoints", "1 simulator", "4 exercises"],
    legend: [["w", "bf16 / 16-bit"], ["k", "8-bit"], ["v", "4-bit"], ["hot", "error / outlier"], ["ok", "kept / recovered"]],
    prev: "p2-4", next: "p2-6",
    steps: [
      { rail: "bytes are speed", title: "For decode, bytes per weight is speed",
        body: `<p>Batch-1 decode is memory-bound: each step reads all the weights once, so the fastest possible step is <code>weight bytes ÷ memory bandwidth</code> (P1.4, P2.1). Halve the bytes and you halve that floor.</p>
<div class="eq">Llama-3-8B: 8,030,261,248 weights
L4 bandwidth ≈ 300 GB/s (gpu_specs.yaml, UNVERIFIED)

bf16   16 bits  16.06 GB → 18.7 tokens/s ceiling
fp8     8 bits   8.03 GB → 37.4 tokens/s
int4  4.125 b    4.14 GB → 72.5 tokens/s (on paper)</div>
<p>The same arithmetic decides what fits. Llama-3-70B (70.6 B weights) needs 141 GB in bf16, but about 36 GB at 4.125 bits per weight.</p>
<p>Two warnings before the details. The int4 ceiling is "on paper": weights must be converted back to 16-bit inside the matrix multiply, which costs compute, so real speedups land below it. And every bit removed costs some accuracy. The rest of the lesson is about how much, and how to keep it small.</p>`,
        scene(G) {
          const rows = [["bf16", 16, "w"], ["fp8", 8, "k"], ["int4 (g128)", 4.125, "v"]];
          G.text(24, 36, "Llama-3-8B: weight bytes and the batch-1 decode ceiling", { size: 13 });
          const bars = [];
          rows.forEach(([n, b, c], i) => {
            const y = 70 + i * 92, gb = (P8 * b) / 8 / 1e9, tps = 300 / gb;
            G.text(24, y + 26, n, { size: 14 });
            bars.push(G.rect(150, y, (440 * gb) / 16.06, 40, { fill: c, rx: 5 }));
            G.label(158 + (440 * gb) / 16.06, y + 26, `${gb.toFixed(2)} GB`, { size: 13, color: "ink" });
            G.label(150, y + 62, `ceiling ≈ ${tps.toFixed(1)} tokens/s at 300 GB/s`, { size: 12, color: c });
          });
          G.text(24, 370, "half the bytes → twice the ceiling", { size: 15, color: "ok" });
          G.label(24, 394, "L4 bandwidth: course gpu_specs.yaml value, UNVERIFIED", { size: 11 });
          G.from(bars, { attr: { width: 0 }, stagger: 0.2, duration: 0.6 });
          G.caption("memory-bound decode: fewer bytes, faster steps");
        } },

      { rail: "floating point", title: "How a float spends its bits",
        body: `<p>A floating-point number is stored as three fields: a <b>sign</b>, an <b>exponent</b> that picks a power of two, and a <b>mantissa</b> that picks a point between that power and the next:</p>
<div class="eq">value = (−1)^sign × 2^(exponent − bias) × 1.mantissa</div>
<p>Exponent bits buy <b>range</b> (how big and how small). Mantissa bits buy <b>precision</b>: with <i>m</i> mantissa bits there are 2<sup>m</sup> evenly spaced values between each pair of powers of two, so the step is always about 2<sup>−m</sup> <i>relative</i> to the number.</p>
<div class="eq">FP16  1 / 5 / 10   max 65,504     step near 1: 2⁻¹⁰
BF16  1 / 8 / 7    max ≈ 3.4e38   step near 1: 2⁻⁷
      next BF16 after 1.0 is 1.0078125</div>
<p>BF16 keeps FP32's 8 exponent bits, so anything FP32 can hold, BF16 can too, only less precisely. That is why models train and serve in BF16 without overflow trouble. GPUs before Ampere (the T4 included) have no BF16 tensor cores.</p>`,
        scene(G) {
          const fmts = [["FP32", 1, 8, 23], ["FP16", 1, 5, 10], ["BF16", 1, 8, 7], ["FP8 E5M2", 1, 5, 2], ["FP8 E4M3", 1, 4, 3]];
          const u = 16;
          G.text(24, 36, "bit layout: sign · exponent (range) · mantissa (precision)", { size: 13 });
          const all = [];
          fmts.forEach(([n, s, e, m], i) => {
            const y = 64 + i * 62;
            G.text(24, y + 20, n, { size: 13 });
            let x = 120; const cell = (c, k) => { for (let j = 0; j < k; j++) { all.push(G.rect(x, y, u - 2, 28, { fill: c, rx: 2 })); x += u; } };
            cell("hot", s); cell("k", e); cell("v", m);
            G.label(120, y + 44, `${s + e + m} bits: ${e} exponent, ${m} mantissa`, { size: 11 });
          });
          G.label(24, 392, "red sign · cyan exponent · amber mantissa (one square per bit)", { size: 12, color: "ink" });
          G.from(all, { opacity: 0, stagger: 0.004, duration: 0.15 });
          G.caption("exponent bits set the range, mantissa bits the precision");
        } },

      { rail: "fp8", title: "FP8: eight values per power of two",
        body: `<p>FP8 E4M3 has 4 exponent bits (bias 7) and 3 mantissa bits, so 8 values per power of two. The largest is <code>1.75 × 2⁸ = 448</code> (the all-ones pattern is NaN in the "fn" variant serving uses), and the smallest positive is <code>2⁻⁹</code>.</p>
<p>Round 0.3 to E4M3. It lies between 0.25 and 0.5, where the step is 0.25 ÷ 8 = 0.03125:</p>
<div class="eq">0.25, 0.28125, 0.3125, 0.34375, ...
0.3 → 0.3125    (error 0.0125, about 4%)</div>
<p>So an FP8 number is accurate to a few percent wherever it lands in range. Range is the bigger problem: real weights are around 0.02 and activations can be in the hundreds. Each tensor is therefore stored with a <b>scale</b>: divide by <code>max|x| ÷ 448</code> so the largest value lands on 448, store in FP8, multiply back when used.</p>
<p>E5M2 trades a mantissa bit for range (max 57,344) and is used for gradients in training. Inference uses E4M3. FP8 tensor cores exist on Ada (the L4) and Hopper and newer, not on Ampere or the T4 (README; check your GPU's spec sheet).</p>`,
        check: { q: "Serving uses FP8 E4M3 rather than E5M2 for weights and activations. Why?",
          options: ["E4M3 has more range", "E4M3 has one more mantissa bit, so it is more precise, and scales make its ±448 range sufficient", "E5M2 isn't supported by GPUs"], answer: 1,
          why: "With per-channel or per-token scales, the values are moved into range anyway, so precision matters more than range. E4M3's 3 mantissa bits give 8 steps per power of two instead of 4." },
        scene(G) {
          const x0 = 40, W = 560, lo = 0.125, hi = 1;
          const X = (v) => x0 + (W * Math.log2(v / lo)) / Math.log2(hi / lo);
          G.text(24, 36, "E4M3 values between 0.125 and 1 (log scale)", { size: 13 });
          G.line(x0, 120, x0 + W, 120, { color: "line" });
          const vals = T43.filter((v) => v >= lo && v <= hi);
          const t = vals.map((v) => G.line(X(v), 100, X(v), 140, { color: "k", w: 2 }));
          [0.125, 0.25, 0.5, 1].forEach((v) => G.label(X(v), 160, String(v), { anchor: "middle", size: 12, color: "ink" }));
          G.label(x0, 186, "8 values per power of two: the step doubles each time", { size: 12 });
          G.text(24, 236, "zoom: rounding 0.3", { size: 13 });
          const z0 = 60, zw = 520, a = 0.25, b = 0.375, Z = (v) => z0 + (zw * (v - a)) / (b - a);
          G.line(z0, 290, z0 + zw, 290, { color: "line" });
          [0.25, 0.28125, 0.3125, 0.34375, 0.375].forEach((v) => { G.line(Z(v), 274, Z(v), 306, { color: "k", w: 2 }); G.label(Z(v), 326, String(v), { anchor: "middle", size: 11 }); });
          const p = G.circle(Z(0.3), 290, 7, { fill: "hot" });
          G.label(Z(0.3), 262, "0.3", { anchor: "middle", color: "hot", size: 12 });
          G.arrow(Z(0.3) + 8, 284, Z(0.3125) - 4, 284, { color: "ok", w: 2 });
          G.text(24, 372, "0.3 → 0.3125: about 4% error; max 448, scale moves data into range", { size: 13, color: "ok" });
          G.from(t, { opacity: 0, stagger: 0.03, duration: 0.2 }); G.pulse(p, { repeat: 5 });
          G.caption("relative precision is constant; absolute spacing grows");
        } },

      { rail: "integers + scale", title: "Integers with a scale",
        body: `<p>The other family stores integers and one floating-point <b>scale</b> per block of numbers. INT4 has 16 levels, −8 … 7. The grid is <b>uniform</b>: the same absolute step everywhere.</p>
<p>Symmetric quantization of a small group:</p>
<div class="eq">w     = [0.12, −0.50, 0.03, 0.31]
scale = max|w| / 7 = 0.50 / 7 = 0.0714
q     = round(w / scale) = [2, −7, 0, 4]
ŵ     = q × scale = [0.143, −0.500, 0.000, 0.286]
error = [0.023, 0.000, 0.030, 0.024]   all ≤ scale/2</div>
<p>Rounding error is at most half a step, <code>scale ÷ 2</code>. <b>Asymmetric</b> quantization adds a <b>zero point</b> so the 16 levels cover <code>[min, max]</code> instead of <code>[−max, max]</code>, which helps when a group is lopsided (all positive, say).</p>
<p>INT8 works the same with 256 levels. Exercise 1 implements both versions, per group, in <code>exercises/int4.py</code>, and its test checks the <code>scale/2</code> bound.</p>`,
        scene(G) {
          const sc = 0.5 / 7, x0 = 40, W = 560, X = (v) => x0 + (W * (v + 0.57)) / 1.14;
          G.text(24, 36, "INT4 symmetric grid for this group: step = 0.0714", { size: 13 });
          G.line(x0, 140, x0 + W, 140, { color: "line" });
          for (let q = -8; q <= 7; q++) { G.line(X(q * sc), 128, X(q * sc), 152, { color: "k", w: 1.5 }); if (Math.abs(q) % 2 === 1) G.label(X(q * sc), 172, String(q), { anchor: "middle", size: 10 }); }
          const w = [0.12, -0.5, 0.03, 0.31], qq = w.map((v) => Math.round(v / sc));
          const dots = [];
          w.forEach((v, i) => {
            dots.push(G.circle(X(v), 100, 6, { fill: "hot" }));
            G.label(X(v), 84, String(v), { anchor: "middle", size: 11, color: "hot" });
            G.arrow(X(v), 106, X(qq[i] * sc), 126, { color: "ok", w: 1.5 });
          });
          G.label(x0, 196, "q = round(w / scale): each dot snaps to the nearest tick", { size: 12 });
          const tbl = [["w", w.map(String)], ["q", qq.map(String)], ["ŵ", qq.map((q) => (q * sc).toFixed(3))], ["|err|", w.map((v, i) => Math.abs(v - qq[i] * sc).toFixed(3))]];
          tbl.forEach(([h, r], j) => { G.text(40, 240 + j * 30, h, { size: 13, color: j === 3 ? "hot" : "ink" }); r.forEach((c, i) => G.text(150 + i * 110, 240 + j * 30, c, { size: 13, anchor: "middle", color: j === 3 ? "hot" : "ink" })); });
          G.text(24, 380, "worst error ≤ scale / 2 = 0.036", { size: 14, color: "ok" });
          G.from(dots, { opacity: 0, stagger: 0.15, duration: 0.3 });
          G.caption("integers are a uniform grid; the scale sets its spacing");
        } },

      { rail: "outliers & groups", title: "One outlier ruins a scale: use groups",
        body: `<p>LLM weights are mostly tiny (around ±0.02) with a few large <b>outliers</b>. The scale is set by the largest value, so one outlier stretches the grid for every number sharing its scale.</p>
<div class="eq">typical weights ≈ ±0.02, one outlier 0.5
scale = 0.5 / 7 = 0.071 → step 0.071
every |w| &lt; 0.036 rounds to 0</div>
<p>With one scale for a whole matrix (<b>per-tensor</b>), most weights collapse to zero. The fix is more scales: one per row (<b>per-channel</b>) or one per <b>group</b> of, say, 128 consecutive weights. An outlier then only coarsens its own group.</p>
<p>Scales are not free. A 16-bit scale per 128 weights adds <code>16 / 128 = 0.125</code> bits per weight, so "4-bit" with <code>group_size=128</code> is really 4.125 bits; asymmetric adds a zero point on top. Smaller groups: less error, more overhead. Run the simulator to see both curves.</p>`,
        check: { q: "INT4 weights with one 16-bit scale per group of 128 (symmetric). How many bits per weight is that, including the scales?",
          options: ["4", "4.125", "4.5", "20"], answer: 1,
          why: "Each weight costs 4 bits, and each group of 128 shares one 16-bit scale: 16/128 = 0.125 extra bits per weight. That's the 4.125 used for the size and speed estimates." },
        scene(G) {
          const R = rng(5), n = 32, vals = Array.from({ length: n }, () => 0.02 * (R() * 2 - 1) * 1.6);
          vals[9] = 0.5;
          const y0 = 176, sx = 17, x0 = 40, H = 180;
          G.text(24, 36, "32 weights, one outlier", { size: 13 });
          G.text(24, 58, "per-tensor: one scale for all 32", { size: 12, color: "hot" });
          const sc1 = 0.5 / 7;
          vals.forEach((v, i) => { const q = Math.round(v / sc1) * sc1; G.rect(x0 + i * sx, y0 - Math.max(0, v) * H, 12, Math.abs(v) * H, { fill: i === 9 ? "hot" : "muted", rx: 1 }); if (q !== 0 && i !== 9) G.circle(x0 + i * sx + 6, y0 - q * H, 3, { fill: "ok" }); });
          G.line(x0, y0, x0 + n * sx, y0, { color: "line" });
          G.label(x0 + 10 * sx + 4, 104, "← outlier sets the step to 0.071:", { size: 11, color: "hot" }); G.label(x0 + 10 * sx + 4, 120, "   the other 31 all round to 0", { size: 11, color: "hot" });
          G.text(24, 214, "groups of 8: each group gets its own scale", { size: 12, color: "ok" });
          const y1 = 330, H2 = 1800;
          for (let g = 0; g < 4; g++) {
            const grp = vals.slice(g * 8, g * 8 + 8), mx = Math.max(...grp.map(Math.abs)), sc = mx / 7;
            G.rect(x0 + g * 8 * sx - 2, 236, 8 * sx, 120, { stroke: g === 1 ? "hot" : "ok", dash: "4 3", rx: 4 });
            grp.forEach((v, j) => {
              const i = g * 8 + j, vv = Math.max(-0.045, Math.min(0.045, v)), q = Math.max(-0.045, Math.min(0.045, Math.round(v / sc) * sc));
              G.rect(x0 + i * sx, y1 - 36 - Math.max(0, vv) * H2 / 2, 12, (Math.abs(vv) * H2) / 2, { fill: i === 9 ? "hot" : "muted", rx: 1 });
              G.circle(x0 + i * sx + 6, y1 - 36 - (q * H2) / 2, 3, { fill: "ok" });
            });
          }
          G.label(x0, 380, "green dots = dequantized values (lower panel zoomed 8×; outlier clipped)", { size: 11 });
          G.label(x0, 400, "only the outlier's own group loses its small weights", { size: 12, color: "ink" });
          G.caption("more scales: each outlier only coarsens its own group");
        } },

      { rail: "what gets quantized", title: "Weights, activations, or both",
        body: `<p>A layer computes <code>y = x · W</code>: activations <code>x</code> times weights <code>W</code>. You can quantize either side, and the choice decides what gets faster.</p>
<ul><li><b>W4A16</b> (AWQ, GPTQ): 4-bit weights, 16-bit activations. The kernel reads int4 weights, converts them to 16-bit on chip, and multiplies in 16-bit. Decode gets faster (≈4× fewer weight bytes). Prefill does not: it is compute-bound and the math is still 16-bit, plus conversion work.</li><li><b>W8A8</b> (INT8 or FP8): both sides 8-bit, so the multiply itself runs on 8-bit tensor cores. Both decode (half the bytes) and prefill (more FLOPs per second on Ada/Hopper) speed up. Activation scales are computed per token at runtime ("dynamic").</li><li><b>KV cache</b> (<code>--kv-cache-dtype fp8</code>): halves the cache, so twice the tokens fit (step 9).</li></ul>
<p>FP8 "dynamic" needs no calibration data: weight scales come from the weights, activation scales are measured on the fly. AWQ and GPTQ need a few hundred calibration samples, because they choose scales and rounding by looking at real activations (next two steps).</p>`,
        check: { q: "You switch a model from bf16 to AWQ W4A16. Long prompts' TTFT barely changes. Why?",
          options: ["AWQ is broken for long prompts", "Prefill is compute-bound and W4A16 still multiplies in 16-bit, so fewer weight bytes don't help it", "TTFT only depends on the network"], answer: 1,
          why: "W4A16 cuts the bytes read, which is what limits decode. Prefill reuses each weight across many tokens, so it is limited by arithmetic, which W4A16 doesn't reduce (it even adds dequantization)." },
        scene(G) {
          G.text(24, 36, "y = x · W : what is stored low-precision, where the math runs", { size: 13 });
          const rows = [["W4A16", "int4", "16-bit", "16-bit math", "decode ✓  prefill ✗", "v"], ["W8A8 (FP8 / INT8)", "8-bit", "8-bit", "8-bit tensor cores", "decode ✓  prefill ✓", "k"], ["FP8 KV cache", "—", "—", "KV stored in fp8", "2× KV tokens", "ok"]];
          G.label(200, 70, "weights", { size: 12 }); G.label(290, 70, "activations", { size: 12 }); G.label(400, 70, "matmul", { size: 12 });
          const bx = [];
          rows.forEach(([n, w, a, m, gain, c], i) => {
            const y = 86 + i * 96;
            G.text(24, y + 26, n, { size: 13 });
            bx.push(G.box(196, y, 80, 40, w, { fill: w === "—" ? undefined : c, stroke: w === "—" ? "line" : undefined, size: 13 }));
            bx.push(G.box(290, y, 90, 40, a, { stroke: "line", size: 13 }));
            bx.push(G.box(396, y, 210, 40, m, { stroke: c, color: c, size: 12 }));
            G.label(196, y + 62, gain, { size: 13, color: "ink" });
          });
          G.label(24, 396, "W8A8 activations: scales measured per token at runtime", { size: 12 });
          G.from(bx, { opacity: 0, stagger: 0.05, duration: 0.25 });
          G.caption("weight-only helps memory-bound decode; W8A8 helps both phases");
        } },

      { rail: "GPTQ", title: "GPTQ: fix the next weights to cancel the error",
        body: `<p>Rounding each weight to its nearest grid point is not the best you can do. What matters is the layer's <b>output</b> <code>x · W</code>, not each weight on its own. When inputs are correlated, one weight's rounding error can be cancelled by nudging another.</p>
<p>A tiny example: two weights whose inputs are almost always equal (x₁ ≈ x₂ = x), and a coarse grid of step 0.25:</p>
<div class="eq">w = [0.62, 0.36]       output ≈ (0.62 + 0.36)·x = 0.98x
round each:  [0.50, 0.25] → 0.75x   error −0.23x
GPTQ: round w₁ → 0.50 (error −0.12)
      add 0.12 to w₂ → 0.48 → round → 0.50
      [0.50, 0.50] → 1.00x          error +0.02x</div>
<p>GPTQ (Frantar et al., arXiv 2210.17323) does this for a whole layer: it quantizes one column at a time and spreads each column's rounding error over the columns not yet quantized. How much to move each one comes from second-order statistics of the calibration activations (the "Hessian" <code>XᵀX</code>), which measure exactly those input correlations. Errors then partly cancel instead of adding up.</p>`,
        scene(G) {
          const x0 = 70, W = 500, X = (v) => x0 + (W * v) / 1.0;
          G.text(24, 36, "grid step 0.25 · inputs x₁ ≈ x₂", { size: 13 });
          const grid = (y) => { G.line(x0, y, x0 + W, y, { color: "line" }); [0, 0.25, 0.5, 0.75, 1].forEach((v) => { G.line(X(v), y - 10, X(v), y + 10, { color: "k", w: 1.5 }); G.label(X(v), y + 26, String(v), { anchor: "middle", size: 11 }); }); };
          G.text(24, 80, "round to nearest", { size: 13, color: "hot" });
          grid(120); grid(180);
          G.label(40, 124, "w₁", { size: 12, color: "ink" }); G.label(40, 184, "w₂", { size: 12, color: "ink" });
          G.circle(X(0.62), 120, 6, { fill: "hot" }); G.arrow(X(0.62), 112, X(0.5) + 4, 112, { color: "hot" });
          G.circle(X(0.36), 180, 6, { fill: "hot" }); G.arrow(X(0.36), 172, X(0.25) + 4, 172, { color: "hot" });
          G.label(X(0.75) + 10, 160, "sum 0.75 (want 0.98)", { size: 12, color: "hot" });
          G.text(24, 250, "GPTQ: carry w₁'s error into w₂", { size: 13, color: "ok" });
          grid(290); grid(350);
          G.label(40, 294, "w₁", { size: 12, color: "ink" }); G.label(40, 354, "w₂", { size: 12, color: "ink" });
          G.circle(X(0.62), 290, 6, { fill: "ok" }); G.arrow(X(0.62), 282, X(0.5) + 4, 282, { color: "ok" });
          const m = G.circle(X(0.48), 350, 6, { fill: "ok" });
          G.circle(X(0.36), 350, 5, { stroke: "muted" }); G.arrow(X(0.36) + 6, 342, X(0.48) - 6, 342, { color: "muted", dash: "3 3" });
          G.label(X(0.36), 334, "+0.12", { anchor: "middle", size: 11 });
          G.label(X(0.75) + 10, 330, "sum 1.00 (want 0.98)", { size: 12, color: "ok" });
          G.from(m, { attr: { cx: X(0.36) }, duration: 0.7, delay: 0.3 });
          G.caption("errors that cancel through the layer's output are cheap");
        } },

      { rail: "AWQ", title: "AWQ: protect the weights that matter most",
        body: `<p>Not all weights matter equally. A weight multiplied by a large activation contributes a large term to the output, so its rounding error is amplified. AWQ (Lin et al., arXiv 2306.00978) found that a small fraction of input channels, those with consistently large activations, carry most of the damage.</p>
<p>Its trick: <b>scale those channels' weights up by s</b> before quantizing, and divide the matching activations by s (folded into the previous operation, so it costs nothing at run time). The product is unchanged, but the weight now spans more grid steps, so its <i>relative</i> rounding error shrinks.</p>
<div class="eq">weight 0.03, activation ≈ 10, step 0.0714
plain:   round(0.03/0.0714) = 0  → ŵ = 0
         output error ≈ 0.03 × 10 = 0.30
s = 2:   round(0.06/0.0714) = 1  → 0.0714 / 2 = 0.0357
         output error ≈ 0.0057 × 10 = 0.057</div>
<p>Scaling can also enlarge the group's maximum and coarsen everyone else's step, so AWQ <b>searches</b> the scales per layer on calibration data. In this course AWQ and GPTQ checkpoints come from LLM Compressor <code>0.14.0</code> (<code>platform/bakeoff/quantize.py</code>); AutoAWQ is deprecated.</p>`,
        scene(G) {
          G.text(24, 36, "salient channel: large activation × small weight", { size: 13 });
          const x0 = 70, W = 500, X = (v) => x0 + (W * v) / 0.2143;
          const grid = (y) => { G.line(x0, y, x0 + W, y, { color: "line" }); [0, 1, 2, 3].forEach((q) => { G.line(X(q * 0.0714), y - 10, X(q * 0.0714), y + 10, { color: "k", w: 1.5 }); G.label(X(q * 0.0714), y + 26, (q * 0.0714).toFixed(3), { anchor: "middle", size: 11 }); }); };
          G.text(24, 84, "plain: 0.03 rounds to 0", { size: 13, color: "hot" });
          grid(124); G.circle(X(0.03), 124, 6, { fill: "hot" }); G.arrow(X(0.03) - 6, 116, X(0) + 4, 116, { color: "hot" });
          G.label(x0 + 260, 104, "output error 0.03 × 10 = 0.30", { size: 12, color: "hot" });
          G.text(24, 200, "AWQ, s = 2: quantize 0.06, then divide by 2", { size: 13, color: "ok" });
          grid(240); const d = G.circle(X(0.06), 240, 6, { fill: "ok" }); G.arrow(X(0.06) + 6, 232, X(0.0714) - 4, 232, { color: "ok" });
          G.label(x0 + 260, 220, "ŵ = 0.0714 / 2 = 0.0357", { size: 12, color: "ok" });
          G.label(x0 + 260, 290, "output error 0.0057 × 10 = 0.057", { size: 12, color: "ok" });
          G.box(70, 320, 220, 44, "x / s   (folded into the", { stroke: "muted", size: 12 });
          G.label(80, 380, "previous norm/linear: free at run time)", { size: 11 });
          G.box(330, 320, 160, 44, "W · s", { stroke: "ok", color: "ok", size: 13 });
          G.text(510, 348, "= x · W", { size: 14 });
          G.from(d, { attr: { cx: X(0.03) }, duration: 0.6 });
          G.caption("scale up what matters, so rounding hurts it less");
        } },

      { rail: "KV cache", title: "Quantizing the KV cache: twice the users",
        body: `<p>Everything so far shrank the weights. The KV cache can shrink too. With <code>--kv-cache-dtype fp8</code>, vLLM stores keys and values in FP8 (with scales), so each number costs 1 byte instead of 2:</p>
<div class="eq">Llama-3-8B KV per token (P1.2):
  bf16: 2 × 32 × 8 × 128 × 2 = 128 KiB
  fp8:  2 × 32 × 8 × 128 × 1 =  64 KiB

P2.1's L4 budget of 5.64 GiB:
  bf16 ≈ 46,200 tokens   fp8 ≈ 92,400 tokens</div>
<p>Twice the tokens means twice the concurrent sequences at the same length, or twice the context. It also halves the attention's KV reads in each decode step, which matters at long context where those reads rival the weights.</p>
<p>The cost is precision in attention scores. Short prompts rarely show it; long contexts are where to look. Exercise 4 measures both sides: KV tokens and knee with and without fp8 KV, and quality. If the token ratio comes out below 2, explain why (per-block scales, the activation reserve).</p>`,
        scene(G) {
          G.text(24, 36, "the same 5.64 GiB KV budget (P2.1, L4)", { size: 13 });
          const row = (y, label, per, c, n) => {
            G.text(24, y - 10, label, { size: 13 });
            G.rect(24, y, 590, 50, { stroke: "ink", rx: 6 });
            const cells = [];
            for (let i = 0; i < n; i++) cells.push(G.rect(28 + (i * 582) / n, y + 4, 582 / n - 3, 42, { fill: c, rx: 2 }));
            G.label(24, y + 70, per, { size: 12, color: "ink" });
            return cells;
          };
          row(80, "bf16 KV: 128 KiB per token", "≈ 46,200 tokens · each block = 16 tokens (2,888 blocks; 1 square ≈ 120 blocks)", "w", 24);
          const b = row(220, "fp8 KV: 64 KiB per token", "≈ 92,400 tokens · twice the blocks in the same memory", "ok", 48);
          G.text(24, 370, "2× users at the same length, or 2× context", { size: 15, color: "ok" });
          G.label(24, 394, "cost: precision in attention, mostly visible at long context", { size: 12 });
          G.from(b, { opacity: 0, stagger: 0.02, duration: 0.15 });
          G.caption("one byte per KV number instead of two");
        } },

      { rail: "measuring quality", title: "Measuring the quality cost without fooling yourself",
        body: `<p>#7, the bakeoff (<code>platform/bakeoff</code>), serves each variant (bf16, fp8-dynamic, AWQ, GPTQ, bf16 with fp8 KV) on the same GPU and records VRAM, KV tokens, latency and throughput through #4, and accuracy through lm-evaluation-harness tasks.</p>
<p>Accuracy needs the same care as latency (P2.4). A score measured on n questions has a <b>standard error</b>:</p>
<div class="eq">SE = √( p (1 − p) / n )

p = 0.80, n = 250:
  SE = √(0.16 / 250) = √0.00064 ≈ 0.025 = 2.5 points</div>
<p>With 250 samples per task (<code>limit: 250</code> in <code>variants.yaml</code>), a variant scoring 80.0 and one scoring 79.2 are indistinguishable: the 0.8-point gap is a third of one standard error. 250 samples catch large regressions (a broken quantization, a 10-point drop), not small ones. Say so in the report, and keep everything else fixed: same prompts, same sampling settings, same seed.</p>
<p>Two practical rules from the README: keep <code>lm_head</code> and the embeddings in bf16 (sensitive, and cheap), and calibrate on data that looks like your traffic.</p>`,
        check: { q: "AWQ scores 79.2 and bf16 scores 80.0 on a task, with 250 samples each. What should the report say?",
          options: ["AWQ loses 0.8 points", "No measurable difference: the gap is well inside the ±2.5-point standard error at n = 250", "AWQ is broken"], answer: 1,
          why: "The standard error at p ≈ 0.8 and n = 250 is about 2.5 points. A 0.8-point gap is far smaller than the noise, so the honest statement is that this test can't tell them apart." },
        scene(G) {
          G.text(24, 36, "accuracy ± 1 standard error, n = 250 per task", { size: 13 });
          const x0 = 140, W = 460, X = (v) => x0 + (W * (v - 70)) / 20;
          const rows = [["bf16", 80.0, "w"], ["fp8", 79.6, "k"], ["AWQ", 79.2, "v"], ["broken", 71.0, "hot"]];
          [70, 75, 80, 85, 90].forEach((v) => { G.line(X(v), 60, X(v), 330, { color: "line", dash: "2 4" }); G.label(X(v), 350, String(v), { anchor: "middle", size: 11 }); });
          const ds = [];
          rows.forEach(([n, p, c], i) => {
            const y = 90 + i * 64, se = 100 * Math.sqrt((p / 100) * (1 - p / 100) / 250);
            G.text(24, y + 5, n, { size: 13 });
            G.line(X(p - se), y, X(p + se), y, { color: c, w: 3 });
            G.line(X(p - 2 * se), y, X(p + 2 * se), y, { color: c, w: 1, opacity: 0.5 });
            ds.push(G.circle(X(p), y, 7, { fill: c }));
            G.label(X(p + 2 * se) + 8, y + 4, p.toFixed(1), { size: 11, color: "ink" });
          });
          G.label(24, 380, "made-up scores to show the error bars; thick = ±1 SE, thin = ±2 SE", { size: 11 });
          G.label(24, 400, "overlapping bars: can't rank · a 9-point drop: clearly broken", { size: 12, color: "ink" });
          G.from(ds, { opacity: 0, stagger: 0.12, duration: 0.3 });
          G.caption("250 samples find big regressions, not half-point ones");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Decode speed scales with bytes per weight; the KV cache's capacity with bytes per KV number.</li>
<li>Floats spend bits on range (exponent) and precision (mantissa); FP8 E4M3 has 8 values per power of two and needs a scale.</li>
<li>Integer formats are a uniform grid set by a scale; error ≤ scale/2. Groups confine outliers at a cost of 16/group bits per weight.</li>
<li>W4A16 speeds up decode only; W8A8 and FP8 speed up both phases. FP8 KV doubles the tokens that fit.</li>
<li>GPTQ cancels rounding errors through correlated inputs; AWQ scales salient channels before rounding.</li>
<li>Quality differences smaller than √(p(1−p)/n) are noise.</li></ul>`,
    sim: {
      title: "Quantization error vs bits",
      intro: "Synthetic weights like examples/01_formats.py: Gaussian with std 0.02 and a fraction of outliers. Choose the integer bit width, group size and scheme. The left chart shows error against bits for your group size (dots), with FP8 E4M3 and BF16 for reference; the right chart shows one group's weights and the grid they snap to.",
      controls: [
        { id: "bits", label: "integer bits", min: 2, max: 8, value: 4 },
        { id: "group", label: "group size (weights per scale)", type: "select", value: 128, options: [[16, "16"], [32, "32"], [64, "64"], [128, "128"], [256, "256"], [1024, "1024 (per row)"]] },
        { id: "sym", label: "scheme", type: "select", value: 1, options: [[1, "symmetric"], [0, "asymmetric (zero point)"]] },
        { id: "frac", label: "outlier fraction, %", min: 0, max: 2, step: 0.1, value: 0.1, format: (x) => x.toFixed(1) },
        { id: "mag", label: "outlier size, × typical", min: 2, max: 40, value: 10 },
        { id: "bw", label: "memory bandwidth, GB/s (example value: check your GPU's spec sheet)", min: 100, max: 4000, step: 50, value: 300 },
      ],
      height: 290,
      draw(G, v) {
        const cols = 1024, rows = 32, w = weights(3, rows, cols, v.frac / 100, v.mag), sym = !!v.sym;
        const errAt = (b) => relRMS(w, intQ(w, cols, b, v.group, sym));
        const curve = [2, 3, 4, 5, 6, 7, 8].map((b) => [b, errAt(b)]);
        const eFp8 = relRMS(w, fp8Q(w, cols));
        let eBf = 0, sBf = 0; for (let i = 0; i < w.length; i++) { eBf += (w[i] - bf16(w[i])) ** 2; sBf += w[i] ** 2; } eBf = Math.sqrt(eBf / sBf);
        const eCur = curve.find((c) => c[0] === v.bits)[1];
        // left: error vs bits (log y)
        const x0 = 50, y0 = 240, Wd = 270, Hd = 200, lmin = -4, lmax = 0.5;
        const Y = (e) => y0 - (Hd * (Math.log10(Math.max(e, 1e-4)) - lmin)) / (lmax - lmin), Xb = (b) => x0 + ((b - 2) * Wd) / 14;
        G.axes(x0, y0 - Hd, Wd, Hd, {});
        G.label(x0, 22, "relative RMS error (log) vs bits per weight", { size: 11 });
        [-4, -3, -2, -1, 0].forEach((l) => { G.line(x0, Y(10 ** l), x0 + Wd, Y(10 ** l), { color: "line", dash: "2 4" }); G.label(x0 - 6, Y(10 ** l) + 4, l === 0 ? "1" : "1e" + l, { anchor: "end", size: 10 }); });
        [2, 4, 8, 16].forEach((b) => G.label(Xb(b), y0 + 16, String(b), { anchor: "middle", size: 10 }));
        const ov = (sym ? 16 : 32) / v.group;
        let d = ""; curve.forEach(([b, e], i) => { d += `${i ? " L" : "M"} ${Xb(b + ov)} ${Y(e)}`; });
        G.path(d, { color: "v", w: 2 });
        curve.forEach(([b, e]) => G.circle(Xb(b + ov), Y(e), b === v.bits ? 6 : 3.5, { fill: b === v.bits ? "hot" : "v" }));
        G.circle(Xb(8 + 16 / cols), Y(eFp8), 5, { fill: "k" }); G.label(Xb(8) + 8, Y(eFp8) - 6, "fp8 e4m3", { size: 10, color: "k" });
        G.circle(Xb(16), Y(eBf), 5, { fill: "ok" }); G.label(Xb(16) - 6, Y(eBf) - 8, "bf16", { size: 10, color: "ok", anchor: "end" });
        // right: one group around an outlier, and its grid
        let gi = 0; for (let i = 0; i < w.length; i++) if (Math.abs(w[i]) > Math.abs(w[gi])) gi = i;
        const g0 = Math.floor(gi / v.group) * v.group, show = Math.min(v.group, 64), s0 = Math.max(g0, Math.min(gi - show / 2, g0 + v.group - show));
        const grp = Array.from(w.subarray(g0, g0 + v.group)), mx = Math.max(...grp.map(Math.abs)), mn = Math.min(...grp), MX = Math.max(...grp);
        const rx = 370, rw = 260, ry = 40, rh = 200;
        const lo = sym ? -mx : mn, hi = sym ? mx : MX, Yr = (val) => ry + rh - (rh * (val - lo)) / (hi - lo || 1);
        G.label(rx, 22, `group with the largest |w| (${show} of ${v.group} shown)`, { size: 11 });
        G.rect(rx, ry, rw, rh, { stroke: "line", rx: 3 });
        const levels = 2 ** v.bits, sc = sym ? mx / (2 ** (v.bits - 1) - 1) : (MX - mn) / (levels - 1);
        if (levels <= 64) for (let q = 0; q < levels; q++) { const val = sym ? (q - (2 ** (v.bits - 1) - 1)) * sc : mn + q * sc; if (val >= lo - 1e-12 && val <= hi + 1e-12) G.line(rx, Yr(val), rx + rw, Yr(val), { color: "line", w: 0.8 }); }
        const qv = intQ(w.subarray(g0, g0 + v.group), cols, v.bits, v.group, sym);
        for (let j = 0; j < show; j++) {
          const i = s0 + j, x = rx + 6 + (j * (rw - 12)) / Math.max(1, show - 1);
          G.line(x, Yr(w[i]), x, Yr(qv[i - g0]), { color: "hot", w: 1 });
          G.circle(x, Yr(w[i]), 2.4, { fill: i === gi ? "hot" : "ink" });
          G.circle(x, Yr(qv[i - g0]), 2, { fill: "v" });
        }
        G.label(rx, ry + rh + 16, "dark = true · amber = quantized", { size: 10 });
        const bpw = v.bits + ov, gb = (P8 * bpw) / 8 / 1e9;
        let zeros = 0; for (let i = g0; i < g0 + v.group; i++) if (qv[i - g0] === 0 && w[i] !== 0) zeros++;
        return [
          { title: `INT${v.bits}, group ${v.group}, ${sym ? "symmetric" : "asymmetric"}`, rows: [["relative RMS error", (100 * eCur).toFixed(2) + " %"], ["FP8 E4M3 (per-row scale)", (100 * eFp8).toFixed(2) + " %"], ["BF16", (100 * eBf).toFixed(3) + " %"], ["weights rounded to 0 in the shown group", `${zeros} of ${v.group}`]] },
          { title: "Bits and bytes", rows: [["bits per weight incl. scales", bpw.toFixed(3)], ["Llama-3-8B weights", gb.toFixed(2) + " GB"], ["vs bf16", (16 / bpw).toFixed(2) + "× smaller"]] },
          { title: "Batch-1 decode ceiling", rows: [["bf16", (v.bw / 16.06).toFixed(1) + " tok/s"], ["this format", (v.bw / gb).toFixed(1) + " tok/s"]],
            html: `<p class="note">Ceiling = bandwidth ÷ weight bytes, ignoring KV reads and the dequantization cost inside the GEMM, so real W4A16 speedups land well below it. Error is on synthetic Gaussian weights, not a real model's accuracy.</p>` },
        ];
      },
    },
    practice: {
      intro: "Exercises 1–3 run on a laptop (exercise 2 needs the torch extra for its comparison). Exercise 4 and the real bakeoff need an FP8-capable GPU: aws.md uses g6.xlarge (L4), bound to 127.0.0.1, torn down with make down. T4 has no FP8 or bf16.",
      items: [
        { title: "INT4 group quantization in NumPy", tier: "T0 · medium", goal: "Write quantize()/dequantize() with per-group scales, symmetric and asymmetric. The test checks error ≤ scale/2, asymmetric wins on skewed groups, and smaller groups win with outliers.",
          cmd: "uv run pytest course/P2-serving-engines/P2.5-quantization/exercises/test_int4.py" },
        { title: "FP8 E4M3 emulation", tier: "T0 · medium", goal: "Round float32 to the nearest E4M3 (and E5M2) value, ties to even; must match torch.float8_e4m3fn bit for bit on every in-range value and midpoint.",
          cmd: "uv run pytest course/P2-serving-engines/P2.5-quantization/exercises/test_fp8.py" },
        { title: "The #7 table from results", tier: "T0 (+ T2 run)", goal: "Build the bakeoff table from fixtures, add decode-speedup-vs-bf16 and bits/weight columns, then run the real bakeoff and write three conclusions plus a caveat about n.",
          cmd: "uv run pytest course/P2-serving-engines/P2.5-quantization/exercises/test_table.py" },
        { title: "FP8 KV cache: capacity vs quality", tier: "T2 · hard", goal: "Compare bf16 with bf16+fp8kv: KV-token ratio, knee, and quality (add a long-context task if you can), recorded in results/p25_fp8kv.json.",
          cmd: "uv run python course/P2-serving-engines/P2.5-quantization/exercises/check_fp8kv.py results/p25_fp8kv.json" },
      ],
      labs: [
        { label: "Round-trip error of one tensor in every format", path: "course/P2-serving-engines/P2.5-quantization/examples/01_formats.py" },
        { label: "#7, the bakeoff: quantize, run, table", path: "platform/bakeoff/ (quantize.py, run.py, table.py, variants.yaml)" },
        { label: "AWS guide: L4 for FP8, cost, teardown", path: "course/P2-serving-engines/P2.5-quantization/aws.md" },
        { label: "How vLLM dispatches each format to a kernel", path: "vllm/model_executor/layers/quantization/ (pinned v0.31.0)" },
        { label: "Papers: GPTQ 2210.17323 · AWQ 2306.00978 · FP8 formats 2209.05433", path: "SOURCES.md" },
      ],
    },
  });
})();
