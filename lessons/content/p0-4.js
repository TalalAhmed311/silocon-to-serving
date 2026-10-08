/* P0.4 — SIMD and the roofline. One instruction, many lanes; then the model that says whether a kernel is limited by arithmetic or by memory. */
(function () {
  const F = S2S.fmt;
  // log-scale helper for the roofline scenes: value -> pixel
  const logX = (x0, w, lo, hi) => (v) => x0 + ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * w;
  const logY = (y0, h, lo, hi) => (v) => y0 - ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * h;
  // kernels for the simulator: FLOPs and DRAM bytes (ideal, each operand crosses DRAM once)
  const KERNELS = {
    add: { name: "vector add c = a + b", W: (n) => n, Q: (n) => 12 * n },
    dot: { name: "dot product", W: (n) => 2 * n, Q: (n) => 8 * n },
    gemv: { name: "matvec, n × n", W: (n) => 2 * n * n, Q: (n) => 4 * n * n + 8 * n },
    gemm: { name: "matmul, n × n × n", W: (n) => 2 * n ** 3, Q: (n) => 12 * n * n },
  };

  S2S.lesson({
    id: "p0-4", n: "P0.4", title: "SIMD and the roofline",
    subtitle: "Systems primer · first principles · T0, any laptop",
    kicker: "Lesson · ≈ 50 min",
    headline: "Is your code waiting for arithmetic, or for memory?",
    intro: `<p>A modern CPU core can add eight pairs of numbers with one instruction, and two such multiply-adds can start every cycle. Yet many programs run nowhere near that speed, because the numbers can't arrive from memory fast enough.</p>
<p>This lesson builds both halves of that story. First <b>SIMD</b>: how one instruction works on many numbers, and what it takes to keep the hardware busy. Then the <b>roofline model</b>: a two-line picture that tells you, for any loop, whether arithmetic or memory is the limit, and so what is worth optimising. You will use the same picture on GPUs in P1.4, and it explains why generating text with an LLM is limited by memory.</p>`,
    facts: ["11 steps", "5 checkpoints", "1 simulator", "5 exercises"],
    legend: [["k", "vector lanes"], ["v", "data"], ["q", "instruction"], ["ok", "busy / fast"], ["hot", "idle / slow"]],
    prev: "p0-3", next: "p0-5",
    steps: [
      { rail: "lanes", title: "One instruction, eight additions",
        body: `<p>A plain loop <code>c[i] = a[i] + b[i]</code> issues one add instruction per element. But CPUs have wide <b>vector registers</b> that hold several numbers side by side, and instructions that work on every slot (every <b>lane</b>) at once. This is <b>SIMD</b>: single instruction, multiple data.</p>
<table><tr><th>instruction set</th><th>register</th><th>fp32 lanes</th></tr>
<tr><td>x86 SSE / ARM NEON</td><td>128 bit</td><td>4</td></tr>
<tr><td>x86 AVX2</td><td>256 bit</td><td>8</td></tr>
<tr><td>x86 AVX-512</td><td>512 bit</td><td>16</td></tr></table>
<p>In C++ you reach these instructions through <b>intrinsics</b>: functions that map one-to-one onto instructions.</p>
<div class="eq">#include &lt;immintrin.h&gt;   // x86; build with -mavx2 -mfma
__m256 va = _mm256_loadu_ps(a);   // 8 floats in
__m256 vb = _mm256_loadu_ps(b);
_mm256_storeu_ps(c, _mm256_add_ps(va, vb));</div>
<p>One <code>vaddps</code> costs about the same as one scalar add, so 8 results for the price of 1. ARM's equivalents are <code>vld1q_f32</code>, <code>vaddq_f32</code> (4 lanes, header <code>&lt;arm_neon.h&gt;</code>). The module's <code>examples/simd.hpp</code> wraps both, so the exercises compile on either. A GPU warp (P5) is the same idea taken further: 32 lanes per instruction.</p>`,
        scene(G) {
          G.text(24, 36, "scalar: 8 add instructions", { size: 13, color: "hot" });
          const sc = [];
          for (let i = 0; i < 8; i++) {
            G.box(24 + i * 74, 48, 64, 26, `a${i}`, { stroke: "v", color: "v", size: 11 });
            G.box(24 + i * 74, 80, 64, 26, `b${i}`, { stroke: "v", color: "v", size: 11 });
            sc.push(G.box(24 + i * 74, 120, 64, 26, `c${i}`, { fill: "hot", size: 11 }));
          }
          G.label(24, 166, "addss, addss, addss, … one result per instruction", { size: 11 });
          G.text(24, 210, "AVX2: 1 add instruction on 8 lanes", { size: 13, color: "ok" });
          G.rect(20, 222, 596, 40, { stroke: "k", rx: 8 }); G.rect(20, 268, 596, 40, { stroke: "k", rx: 8 });
          for (let i = 0; i < 8; i++) { G.box(24 + i * 74, 226, 70, 32, `a${i}`, { fill: "k", size: 11 }); G.box(24 + i * 74, 272, 70, 32, `b${i}`, { fill: "k", size: 11 }); }
          G.label(20, 220, "", {});
          const op = G.box(200, 318, 240, 30, "vaddps ymm (one instruction)", { fill: "q", size: 12 });
          const res = G.group();
          for (let i = 0; i < 8; i++) G.box(24 + i * 74, 360, 70, 32, `c${i}`, { fill: "ok", size: 11, parent: res });
          G.from(sc, { opacity: 0, stagger: 0.15, duration: 0.2 }); G.from(res, { opacity: 0, delay: 1.4, duration: 0.3 });
          G.caption("a 256-bit register holds 8 floats; one instruction adds all lanes");
        } },

      { rail: "tails · alignment", title: "Leftovers and line-crossing loads",
        body: `<p>Real arrays are not a multiple of 8 long. The vector loop handles groups of 8 while they fit; a scalar loop finishes the <b>tail</b>:</p>
<div class="eq">size_t i = 0;
for (; i + 8 &lt;= n; i += 8)        // full vectors
  store(y + i, fma(va, load(x + i), load(y + i)));
for (; i &lt; n; ++i)               // tail: n % 8 left
  y[i] = a * x[i] + y[i];</div>
<p>For n = 67 that is 8 vector iterations (64 elements) and a 3-element tail. Exercise 1 (SAXPY, <code>y = a·x + y</code>) is tested for every n from 0 to 67, exactly to catch tail bugs.</p>
<p><b>Alignment.</b> <code>_mm256_load_ps</code> requires a 32-byte-aligned address and crashes otherwise; <code>_mm256_loadu_ps</code> accepts any address and, on modern cores, is just as fast when the data happens to be aligned. The real cost is a load that <b>straddles two cache lines</b> (P0.1): the core must fetch from both. If an array starts 4 bytes past a line boundary, the 32-byte loads cover bytes 4–35, 36–67, 68–99, …: every second load crosses a 64-byte boundary. <code>examples/01_scalar_vs_simd_add.cpp</code> measures what that costs on your CPU; aligning buffers to 64 bytes (P0.1's aligned owner) removes it.</p>`,
        check: { q: "AVX2 (8 lanes), n = 67. How many vector iterations and how many scalar tail elements?",
          options: ["8 iterations, 3 tail elements", "9 iterations, 0 tail elements", "8 iterations, 0 tail elements: the last vector reads past the end"], answer: 0,
          why: "8 × 8 = 64 elements fit in full vectors; 67 − 64 = 3 are left for the scalar loop. A 9th full vector load would read 5 floats past the end of the array: a buffer overflow, not an optimisation." },
        scene(G) {
          G.text(24, 36, "n = 67 floats with 8-wide vectors", { size: 13 });
          const cells = [];
          for (let i = 0; i < 67; i++) cells.push(G.rect(24 + i * 8.8, 50, 7.4, 34, { fill: i >= 64 ? "hot" : Math.floor(i / 8) % 2 ? "v" : "k", rx: 1.5 }));
          for (let g = 0; g < 8; g++) G.label(24 + g * 70.4 + 35, 102, String(g + 1), { anchor: "middle", size: 10 });
          G.label(24 + 64 * 8.8, 102, "tail", { size: 10, color: "hot" });
          G.label(24, 124, "8 vector iterations (64 elements) + 3 scalar tail elements", { size: 12, color: "ink" });
          G.text(24, 176, "array starting 4 bytes past a 64-byte line boundary", { size: 13 });
          const px = 7.4;
          G.rect(24, 192, 64 * px - 2, 34, { stroke: "muted", rx: 3 }); G.rect(24 + 64 * px, 192, 16 * px, 34, { stroke: "muted", rx: 3 });
          G.label(28, 244, "cache line: bytes 0–63", { size: 10 }); G.label(28 + 64 * px, 244, "64…", { size: 10 });
          const loads = [];
          for (let l = 0; l < 2; l++) {
            const s = 4 + l * 32, split = (s % 64) + 32 > 64;
            loads.push(G.box(24 + s * px, 196, 32 * px - 3, 26, split ? "load 2: splits" : "load 1: one line", { fill: split ? "hot" : "ok", size: 11 }));
          }
          G.line(24 + 64 * px - 1, 186, 24 + 64 * px - 1, 232, { color: "ink", dash: "4 3", w: 2 });
          G.label(24, 272, "bytes 4–35: inside line 0   ·   bytes 36–67: crosses into line 1", { size: 11, color: "ink" });
          G.text(24, 316, "load  (aligned):   needs a 32-byte-aligned address", { size: 12 });
          G.text(24, 340, "loadu (unaligned): any address, same speed when aligned", { size: 12 });
          G.text(24, 372, "fix the split: allocate 64-byte-aligned buffers", { size: 13, color: "ok" });
          G.from(loads, { opacity: 0, x: -20, stagger: 0.4, duration: 0.4 });
          G.caption("vector body + scalar tail; misaligned starts make loads straddle lines");
        } },

      { rail: "FMA · peak", title: "Fused multiply-add and the peak FLOP rate",
        body: `<p>Almost every inner loop in ML is "multiply, then add to a running total". CPUs have one instruction for it: <b>FMA</b> (fused multiply-add) computes <code>a·b + c</code> with a single rounding. One FMA on 8 lanes does 8 multiplies and 8 adds: <b>16 floating-point operations</b> (FLOPs).</p>
<div class="eq">_mm256_fmadd_ps(a, b, c)  // x86: a*b + c
vfmaq_f32(c, a, b)        // ARM: c + a*b
                          // (note the argument order)</div>
<p>Many recent cores have two FMA units, each able to start one FMA per cycle. That gives the core's <b>peak</b>:</p>
<div class="eq">peak = FMA units × lanes × 2 FLOP × clock

2 × 8 × 2 × 3.0 GHz = 96 GFLOP/s per core
× 8 cores           = 768 GFLOP/s</div>
<p>These are <b>example values</b> for the arithmetic, not your machine. The number of FMA units and the sustained clock under vector load are in your CPU's optimisation manual, and the clock often drops when wide vector units are busy. <code>examples/02_fma_peak.cpp</code> <i>measures</i> your peak; that measured number is the one you use from now on.</p>`,
        scene(G) {
          G.text(24, 40, "one FMA instruction, 8 lanes", { size: 13 });
          const lanes = [];
          for (let i = 0; i < 8; i++) {
            const x = 24 + i * 74;
            lanes.push(G.box(x, 54, 66, 54, "", { fill: "k", boxOpacity: 0.85 }));
            G.text(x + 33, 76, `a·b`, { anchor: "middle", size: 12, color: "bg" }); G.text(x + 33, 96, `+ c`, { anchor: "middle", size: 12, color: "bg" });
          }
          G.label(24, 130, "8 multiplies + 8 adds = 16 FLOPs, one rounding each", { size: 12, color: "ink" });
          const f = [["2", "FMA units"], ["8", "lanes"], ["2", "FLOP / lane"], ["3.0 GHz", "clock"]];
          f.forEach(([v, l], i) => {
            G.box(24 + i * 150, 176, 120, 64, "", { stroke: "line" });
            G.text(84 + i * 150, 206, v, { anchor: "middle", size: 20 }); G.label(84 + i * 150, 228, l, { anchor: "middle", size: 11 });
            if (i < 3) G.text(159 + i * 150, 214, "×", { anchor: "middle", size: 18, color: "muted" });
          });
          G.text(24, 286, "= 96 GFLOP/s per core", { size: 18, color: "ok" });
          G.text(24, 318, "× 8 cores = 768 GFLOP/s", { size: 15, color: "ok" });
          G.label(24, 356, "example values for the arithmetic: measure yours with 02_fma_peak", { size: 12, color: "hot" });
          G.from(lanes, { opacity: 0, y: -10, stagger: 0.06, duration: 0.25 });
          G.caption("peak = units × lanes × 2 × clock, per core");
        } },

      { rail: "latency", title: "Why one accumulator reaches only 1/8 of peak",
        body: `<p>Peak assumes a new FMA starts on each unit every cycle. But an FMA's result is not ready immediately: its <b>latency</b> is several cycles (about 4 on many recent x86 cores; Agner Fog's instruction tables list yours). An FMA that needs the previous one's result has to wait.</p>
<p>A dot product written the obvious way is one long chain:</p>
<div class="eq">acc = fma(a[i], b[i], acc);   // needs the last acc</div>
<p>Each FMA waits 4 cycles for the one before, so the core starts 1 FMA every 4 cycles while it could start 2 per cycle: <b>1/8 of peak</b>. The fix is to keep <b>latency × units = 4 × 2 = 8</b> independent chains in flight, each with its own accumulator, and add them together at the end:</p>
<div class="eq">acc0 = fma(a0, b0, acc0);
acc1 = fma(a1, b1, acc1);    // independent of acc0
...
acc7 = fma(a7, b7, acc7);
sum = acc0 + acc1 + ... + acc7;   // once, at the end</div>
<p><code>02_fma_peak.cpp</code> sweeps from 1 to 12 accumulators: GFLOP/s climbs, then flattens at your latency × throughput product. Exercise 3's SGEMM micro-kernel uses exactly 8 accumulators for this reason.</p>`,
        check: { q: "An FMA has 4-cycle latency and the core has 2 FMA units. How many independent accumulators keep both units busy?",
          options: ["2", "4", "8", "16"], answer: 2,
          why: "Each unit can start one FMA per cycle, and each chain can only start a new FMA every 4 cycles. To start 2 per cycle you need 4 × 2 = 8 chains in flight. Fewer leaves issue slots empty; more doesn't help." },
        legend: [["k", "FMA started"], ["hot", "empty issue slot"]],
        scene(G) {
          const cw = 34, x0 = 100;
          const gantt = (y, title, filled) => {
            G.text(24, y - 12, title, { size: 13 });
            const cells = [];
            for (let u = 0; u < 2; u++) {
              G.label(24, y + u * 36 + 21, `unit ${u}`, { size: 11 });
              for (let c = 0; c < 15; c++) {
                const f = filled(u, c);
                const r = f !== null ? G.box(x0 + c * cw, y + u * 36, cw - 4, 30, String(f), { fill: "k", size: 10 }) : G.rect(x0 + c * cw, y + u * 36, cw - 4, 30, { fill: "hot", rx: 4, opacity: 0.25 });
                if (f !== null) cells.push(r);
              }
            }
            return cells;
          };
          const one = gantt(62, "1 accumulator: each FMA waits 4 cycles for the last", (u, c) => (u === 0 && c % 4 === 0 ? 0 : null));
          G.label(x0, 150, "4 of 30 slots used ≈ 1/8 of peak", { size: 12, color: "hot" });
          const eight = gantt(212, "8 accumulators: always 8 FMAs in flight", (u, c) => ((c * 2 + u) % 8));
          G.label(x0, 300, "every slot used: acc 0–7 take turns, each waits its 4 cycles", { size: 12, color: "ok" });
          for (let c = 0; c < 15; c += 2) G.label(x0 + c * cw + 15, 326, String(c), { anchor: "middle", size: 10 });
          G.label(24, 326, "cycle →", { size: 10 });
          G.text(24, 372, "needed chains = latency × units = 4 × 2 = 8", { size: 14, color: "ink" });
          G.from(eight, { opacity: 0, stagger: 0.02, duration: 0.15 });
          G.caption("independent accumulators hide latency: same work, 8× the rate");
        } },

      { rail: "reductions", title: "Folding eight lanes into one number",
        body: `<p>A dot product ends with 8 partial sums in one register (or 8 registers of 8). The answer is their total: a <b>horizontal sum</b>. Done through a temporary array it costs more than the whole loop for short vectors; done with shuffles it is a few instructions, each halving the live lanes:</p>
<div class="eq">8 lanes → add high 128 bits to low  → 4
4 lanes → movehdup (copy odd lanes) + add → 2
2 lanes → movehl (move high pair) + add  → 1</div>
<p>Exercise 2 asks for this in at most 6 intrinsics. Every reduction in an LLM ends this way: dot products in matmuls, the softmax denominator, the sum of squares in RMSNorm.</p>
<p>Why not let the compiler do it? Compilers do <b>auto-vectorise</b> simple loops at <code>-O3 -march=native</code> (ask with <code>-fopt-info-vec-optimized</code> on GCC or <code>-Rpass=loop-vectorize</code> on Clang). But a vectorised sum adds the numbers in a different order, and floating-point addition is not associative: <code>(a + b) + c</code> can differ from <code>a + (b + c)</code> in the last bits. So without <code>-ffast-math</code> the compiler must keep your order and won't vectorise the reduction. Writing the intrinsics yourself is how you accept the reordering explicitly. That is also why the course compares results with a tolerance (<code>rtol</code>), never bit for bit.</p>`,
        scene(G) {
          const vals = [1, 2, 3, 4, 5, 6, 7, 8];
          G.text(24, 36, "horizontal sum of [1, 2, 3, 4, 5, 6, 7, 8]", { size: 13 });
          const lvl = [vals, [6, 8, 10, 12], [14, 22], [36]];
          const names = ["8 lanes", "extract high + add", "movehdup + add", "movehl + add"];
          const out = [];
          lvl.forEach((row, d) => {
            const y = 60 + d * 86, w = 50, gap = 6, tot = row.length * (w + gap) - gap, x0 = 24 + (456 - tot) / 2;
            const g = G.group();
            row.forEach((v, i) => G.box(x0 + i * (w + gap), y, w, 40, String(v), { fill: d === 3 ? "ok" : "k", size: 14, parent: g }));
            G.label(616, y + 25, names[d], { anchor: "end", size: 11, color: d ? "q" : "muted", parent: g });
            if (d) out.push(g);
          });
          G.label(24, 404, "1+5, 2+6, 3+7, 4+8 → (6+8), (10+12) → 36", { size: 12, color: "ink" });
          G.from(out, { opacity: 0, y: -14, stagger: 0.35, duration: 0.3 });
          G.caption("each shuffle-and-add halves the live lanes: 8 → 4 → 2 → 1");
        } },

      { rail: "intensity", title: "Count FLOPs per byte: arithmetic intensity",
        body: `<p><code>examples/01</code> shows something surprising: on arrays that fit in cache, the SIMD add is several times faster than scalar; on arrays much bigger than the last-level cache, they run at about the same speed. Vector width stopped mattering. Why?</p>
<p>Count the work. <code>c[i] = a[i] + b[i]</code> does 1 FLOP and moves 12 bytes from and to memory (read 4 + 4, write 4). The core can do the FLOP far faster than memory can deliver the bytes, so it waits. The ratio is the kernel's <b>arithmetic intensity</b>:</p>
<div class="eq">I = W / Q    FLOPs per byte of DRAM traffic</div>
<p>Count Q as bytes that actually travel to or from main memory (DRAM), not bytes served by caches.</p>
<table><tr><th>kernel (fp32)</th><th>FLOPs W</th><th>bytes Q</th><th>I</th></tr>
<tr><td>vector add, n</td><td>n</td><td>12n</td><td>0.083</td></tr>
<tr><td>dot product, n</td><td>2n</td><td>8n</td><td>0.25</td></tr>
<tr><td>matvec, n × n</td><td>2n²</td><td>≈ 4n²</td><td>≈ 0.5</td></tr>
<tr><td>matmul, n × n × n</td><td>2n³</td><td>12n² (ideal)</td><td>n / 6</td></tr></table>
<p>Check the last row for n = 4,096: W = 2 × 4096³ ≈ 137.4 G FLOPs; Q = 12 × 4096² ≈ 201 MB; I ≈ 683. Matmul reuses every number it loads n times, so its intensity <b>grows with size</b>. Matvec uses each matrix element once, so it is stuck near 0.5 at any size. "Ideal" assumes each matrix crosses DRAM once, which takes the tiling of step 10.</p>`,
        scene(G) {
          const X = logX(40, 560, 0.05, 2000);
          G.text(24, 40, "arithmetic intensity (FLOP per byte, log scale)", { size: 13 });
          G.line(40, 220, 600, 220, { color: "muted", w: 2 });
          [0.1, 1, 10, 100, 1000].forEach((v) => { G.line(X(v), 214, X(v), 226, { color: "muted" }); G.label(X(v), 244, String(v), { anchor: "middle", size: 11 }); });
          const pts = [[1 / 12, "add", "hot", 0], [0.25, "dot", "hot", 1], [0.5, "matvec", "hot", 2], [64 / 6, "matmul n=64", "v", 0], [4096 / 6, "matmul n=4096", "ok", 1]];
          const dots = pts.map(([v, n, col, row]) => {
            const g = G.group();
            G.circle(X(v), 220, 8, { fill: col, parent: g });
            G.line(X(v), 210, X(v), 172 - row * 36, { color: col, w: 1, dash: "3 3", parent: g });
            G.text(X(v), 166 - row * 36, n, { anchor: "middle", size: 12, color: col, parent: g });
            G.label(X(v), 268 + row * 18, v < 1 ? v.toFixed(2) : v.toFixed(0), { anchor: "middle", size: 11, color: col, parent: g });
            return g;
          });
          G.text(24, 340, "add, dot, matvec: each number is used once → I stays < 1", { size: 13, color: "hot" });
          G.text(24, 366, "matmul: each number is reused n times → I = n / 6", { size: 13, color: "ok" });
          G.from(dots, { opacity: 0, y: -10, stagger: 0.2, duration: 0.3 });
          G.caption("intensity = how much arithmetic each byte from memory pays for");
        } },

      { rail: "roofline", title: "The roofline: min(compute, intensity × bandwidth)",
        body: `<p>A machine has two speed limits: its peak compute <b>P</b> (FLOP/s, step 3) and its memory bandwidth <b>B</b> (bytes/s). A kernel with intensity I gets I FLOPs for every byte, so memory alone allows I × B FLOP/s. It can't beat either limit:</p>
<div class="eq">attainable FLOP/s = min(P, I × B)
ridge point      I* = P / B</div>
<p>On log-log axes this is a slanted line (memory-bound) meeting a flat roof (compute-bound) at the <b>ridge point</b>. Left of the ridge, faster arithmetic is useless: only fewer bytes or more reuse help. Right of it, memory is no longer the problem.</p>
<p>Worked example with <b>example values</b> P = 400 GFLOP/s and B = 50 GB/s (the placeholder in <code>animations/roofline.html</code>; use your measured numbers):</p>
<div class="eq">ridge = 400 / 50 = 8 FLOP/byte
matvec (I = 0.5):  0.5 × 50 = 25 GFLOP/s  (6% of peak)
matmul n=4096 (I ≈ 683): min(400, 34,133) = 400</div>
<p>No amount of SIMD tuning lifts that matvec above 25 GFLOP/s on this machine. That is the most useful thing the roofline tells you: <b>which optimisation can possibly help</b>.</p>`,
        check: { q: "Peak 400 GFLOP/s, bandwidth 50 GB/s (example values). A 4096 × 4096 fp32 matvec has I ≈ 0.5. What limits it, and what is its ceiling?",
          options: ["Compute: 400 GFLOP/s", "Memory: 0.5 × 50 = 25 GFLOP/s", "Memory: 50 GFLOP/s"], answer: 1,
          why: "0.5 is far left of the ridge (8), so the slanted memory line is the lower limit: I × B = 0.5 FLOP/byte × 50 GB/s = 25 GFLOP/s. Wider SIMD or more FMA units can't change that; fewer bytes per FLOP can." },
        legend: [["k", "memory roof (I × B)"], ["v", "compute roof (P)"], ["hot", "memory-bound"], ["ok", "compute-bound"]],
        scene(G) {
          const P = 400, B = 50, X = logX(70, 520, 0.05, 2000), Y = logY(380, 320, 1, 1000);
          G.line(70, 380, 600, 380, { color: "muted" }); G.line(70, 60, 70, 380, { color: "muted" });
          [0.1, 1, 10, 100, 1000].forEach((v) => G.label(X(v), 398, String(v), { anchor: "middle", size: 10 }));
          [1, 10, 100, 1000].forEach((v) => G.label(64, Y(v) + 4, String(v), { anchor: "end", size: 10 }));
          G.label(590, 416, "intensity, FLOP/byte (log) →", { anchor: "end", size: 11 });
          G.label(74, 48, "attainable GFLOP/s (log)", { size: 11 });
          const ridge = P / B;
          const slope = G.line(X(0.05), Y(Math.max(1, 0.05 * B)), X(ridge), Y(P), { color: "k", w: 3 });
          G.line(X(ridge), Y(P), X(2000), Y(P), { color: "v", w: 3 });
          G.line(X(ridge), Y(P), X(ridge), 380, { color: "muted", dash: "4 4" });
          G.label(X(ridge) + 6, 370, "ridge = P / B = 8", { size: 11, color: "ink" });
          G.label(X(2000) - 4, Y(P) + 20, "P = 400 GFLOP/s", { anchor: "end", size: 11, color: "v" });
          G.label(90, Y(300), "memory roof: I × 50 GB/s", { size: 11, color: "k" });
          const pts = [[1 / 12, "add 4.2"], [0.5, "matvec 25"], [64 / 6, "matmul 64"], [4096 / 6, "matmul 4096"]];
          const dots = pts.map(([I, n]) => { const a = Math.min(P, I * B), g = G.group(); G.circle(X(I), Y(a), 7, { fill: a < P ? "hot" : "ok", parent: g }); G.label(X(I) + (I > 100 ? -10 : 10), Y(a) + (a < P ? 18 : -12), n, { size: 11, color: "ink", anchor: I > 100 ? "end" : "start", parent: g }); return g; });
          G.from(dots, { opacity: 0, scale: 0.3, transformOrigin: "center", stagger: 0.2, duration: 0.3 });
          G.caption("left of the ridge: memory-bound · right: compute-bound");
        } },

      { rail: "decode", title: "Why generating text is memory-bound",
        body: `<p>Now the reason this lesson sits at the start of an LLM course. When a model generates text, each new token needs every weight of the model once, multiplied by one vector: a chain of <b>matvecs</b> (P1.2 shows why). Take Llama-3-8B with about 8.0 billion weights stored in bf16 (2 bytes each):</p>
<div class="eq">bytes per token ≈ 8.0e9 × 2 B   = 16 GB
FLOPs per token ≈ 8.0e9 × 2     = 16 GFLOP
intensity       ≈ 16 GFLOP / 16 GB = 1 FLOP/byte</div>
<p>With the example machine (B = 50 GB/s, P = 400 GFLOP/s): reading 16 GB takes at least 16 / 50 = 0.32 s, so at most about <b>3 tokens/s</b>; the arithmetic alone would allow 25 tokens/s. Memory wins: decode speed ≈ bandwidth ÷ model bytes. P0.5 tests this prediction on a real engine.</p>
<p>Two levers follow directly from the roofline:</p>
<ul><li><b>Fewer bytes</b>: store weights in 8 or 4 bits (step 11). Half the bytes, nearly twice the tokens/s.</li>
<li><b>More reuse</b>: generate for B users at once. Each weight read now serves B vectors, so I ≈ B. At B = 8 this example machine reaches its ridge. That is why serving engines batch (P2).</li></ul>`,
        scene(G) {
          G.text(24, 36, "one generated token, Llama-3-8B in bf16", { size: 13 });
          G.rect(24, 52, 420, 150, { fill: "w", opacity: 0.6, rx: 8 });
          for (let r = 0; r < 6; r++) G.rect(36, 64 + r * 22, 396, 16, { fill: "v", opacity: 0.55, rx: 3 });
          G.text(234, 220, "16 GB of weights, read once per token", { anchor: "middle", size: 12 });
          G.box(470, 100, 60, 52, "x", { fill: "k", size: 16 });
          G.label(500, 172, "1 vector", { anchor: "middle", size: 11 });
          const flow = G.arrow(446, 126, 466, 126, { color: "hot", w: 3 });
          G.box(548, 100, 68, 52, "core", { stroke: "ink", size: 12 });
          G.text(24, 264, "memory:  16 GB ÷ 50 GB/s   = 0.32 s → ≤ 3 tokens/s", { size: 13, color: "hot" });
          G.text(24, 290, "compute: 16 GFLOP ÷ 400 GFLOP/s = 0.04 s → ≤ 25 tokens/s", { size: 13, color: "muted" });
          G.text(24, 334, "the slower limit wins: memory", { size: 14, color: "ink" });
          G.label(24, 366, "batch B users: same 16 GB read, B tokens out → I ≈ B", { size: 12, color: "ok" });
          G.label(24, 386, "example P and B: use your measured numbers", { size: 11, color: "muted" });
          G.pulse(flow, { repeat: 6 });
          G.caption("decode streams every weight for very little arithmetic");
        } },

      { rail: "loop order", title: "The SGEMM ladder, rungs 0–1: walk memory in order",
        body: `<p>Exercise 3 (D1) builds a fast matrix multiply, <code>C += A·B</code>, one rung at a time, all row-major fp32. Rung 0 is the textbook triple loop:</p>
<div class="eq">for i: for j: for k:
  C[i][j] += A[i][k] * B[k][j];</div>
<p>The inner loop walks <code>A</code> along a row (good) but <code>B</code> down a column: consecutive <code>k</code> values are N floats apart. That is P0.1's naive transpose again: every access lands on a different cache line and uses 4 of its 64 bytes.</p>
<p>Rung 1 only swaps the two inner loops:</p>
<div class="eq">for i: for k:
  a = A[i][k];                 // one number, reused
  for j: C[i][j] += a * B[k][j];</div>
<p>Now the inner loop streams along a row of <code>B</code> and a row of <code>C</code> with unit stride: every byte of every fetched line is used, and the loop is <code>y += a·x</code> (exercise 1's SAXPY), which the compiler vectorises by itself. Same arithmetic, same result up to rounding, and on most machines this one change gives the largest single speed-up of the ladder.</p>`,
        scene(G) {
          const n = 6, c = 14;
          const panel = (x0, title, col, mark) => {
            G.text(x0, 40, title, { size: 13, color: col });
            [["A", 0], ["B", 1], ["C", 2]].forEach(([nm, m]) => {
              const x = x0 + m * 100;
              G.label(x, 70, nm, { size: 12, color: "ink" });
              G.grid(x, 80, n, n, c, c, (r, k) => mark(m, r, k) || "line", { gap: 2, rx: 2 });
            });
          };
          panel(24, "rung 0: i, j, k", "hot", (m, r, k) => (m === 0 && r === 2 ? "k" : m === 1 && k === 3 ? "hot" : m === 2 && r === 2 && k === 3 ? "v" : null));
          panel(340, "rung 1: i, k, j", "ok", (m, r, k) => (m === 0 && r === 2 && k === 1 ? "q" : m === 1 && r === 1 ? "ok" : m === 2 && r === 2 ? "v" : null));
          G.label(24, 196, "B read down a column: stride N", { size: 11, color: "hot" });
          G.label(24, 214, "one useful float per cache line", { size: 11, color: "hot" });
          G.label(340, 196, "A[i][k] broadcast; B and C rows", { size: 11, color: "ok" });
          G.label(340, 214, "unit stride: whole lines used", { size: 11, color: "ok" });
          G.text(24, 270, "inner loop of rung 1:  C[i][0..N) += A[i][k] · B[k][0..N)", { size: 13 });
          G.text(24, 296, "= SAXPY on two rows: auto-vectorised", { size: 13, color: "ok" });
          G.label(24, 340, "same FLOPs: 2 · M · N · K in both orders", { size: 12, color: "ink" });
          G.label(24, 360, "only the order of memory accesses changed", { size: 12, color: "ink" });
          G.caption("loop order decides whether B streams by rows or jumps by columns");
        } },

      { rail: "tiles · registers", title: "Rungs 2–4: reuse from cache, then from registers",
        body: `<p>Rung 1 still streams all of <code>B</code> from memory once per row of <code>A</code>. <b>Rung 2, tiling</b>, works on blocks (for example 64 × 256 × 256) small enough to stay in L1/L2, so a loaded tile of <code>B</code> is reused by many rows of <code>A</code> before it is evicted. Fewer DRAM bytes for the same FLOPs: higher effective intensity.</p>
<p><b>Rung 3, the register micro-kernel</b>, is the heart of every BLAS library. Keep a 4 × 16 block of <code>C</code> in 8 vector registers (4 rows × 2 vectors of 8) for the whole <code>k</code> loop. For each <code>k</code>:</p>
<div class="eq">4 broadcasts : A[i..i+3][k]       (set1)
2 loads      : B[k][j..j+15]      (16 floats)
8 FMAs       : acc[r][h] += a_r · b_h
= 4 × 16 × 2 = 128 FLOPs per 20 floats loaded</div>
<p>That is 6.4 FLOPs per float pulled into registers, and the 8 accumulators are exactly the 8 independent chains step 4 asked for. <code>C</code> is written back once, at the end. (NEON uses a 4 × 8 tile with 4-lane vectors.)</p>
<p><b>Rung 4</b> (exercise 4) adds P0.3's thread pool: <code>parallel_for</code> over contiguous blocks of rows of <code>C</code>. Blocks write disjoint rows, so no locks and no false sharing; all threads read <code>B</code>, which is cheap because nobody writes it. <code>bench/sgemm_bench.py</code> prints every rung's GFLOP/s as a percentage of your measured peak. The GPU ladder in P5.6 repeats this progression with shared memory and warps.</p>`,
        check: { q: "In the 4 × 16 AVX2 micro-kernel, how many FLOPs does one k-step perform?",
          options: ["16", "64", "128", "256"], answer: 2,
          why: "Each of the 8 FMAs works on 8 lanes and does a multiply and an add: 8 × 8 × 2 = 128. Equivalently, 4 rows × 16 columns of C each get one multiply-add: 64 × 2 = 128." },
        legend: [["q", "A: broadcast"], ["v", "B: loaded"], ["k", "C: accumulators in registers"], ["ok", "threads"]],
        scene(G) {
          G.text(24, 36, "one k-step of the 4 × 16 register tile", { size: 13 });
          G.label(140, 62, "B[k][j … j+15]: 2 vector loads", { size: 11, color: "v" });
          for (let h = 0; h < 2; h++) G.box(140 + h * 240, 70, 232, 30, `b${h} (8 floats)`, { fill: "v", size: 12 });
          G.label(24, 128, "A[i..i+3][k]", { size: 11, color: "q" });
          const acc = [];
          for (let r = 0; r < 4; r++) {
            G.box(24, 138 + r * 46, 96, 38, `a${r} → set1`, { fill: "q", size: 11 });
            for (let h = 0; h < 2; h++) acc.push(G.box(140 + h * 240, 138 + r * 46, 232, 38, `acc${r * 2 + h} += a${r} · b${h}`, { fill: "k", size: 12 }));
          }
          G.text(24, 352, "4 broadcasts + 2 loads → 8 FMAs = 128 FLOPs", { size: 13, color: "ok" });
          G.label(24, 378, "8 accumulators = the 8 chains that hide FMA latency", { size: 12, color: "ink" });
          G.label(24, 400, "rung 4: threads take contiguous row blocks of C", { size: 12, color: "ok" });
          G.from(acc, { opacity: 0, stagger: 0.08, duration: 0.25 });
          G.caption("C stays in registers for the whole k loop; only A and B stream");
        } },

      { rail: "int8", title: "Int8 dot products: fewer bytes per weight",
        body: `<p>Decode is memory-bound, so the fastest way to speed it up is to read fewer bytes. <b>Quantisation</b> stores weights as small integers plus a scale. A <code>q8</code> block holds 32 values:</p>
<div class="eq">struct BlockQ8 { float scale; int8_t q[32]; };
x[i] ≈ scale × q[i],  scale = max|x| / 127
q = round(x / scale), in [−127, 127]</div>
<p>The dot product of two blocks factors nicely: <code>Σ (sa·qa)(sb·qb) = sa·sb·Σ qa·qb</code>. The inner sum is pure <b>integer</b> arithmetic, and the two float scales are applied once per block. Integer products are accumulated in <b>int32</b>: the largest possible sum is 32 × 127 × 127 = 516,128, far too big for int16 (max 32,767) but tiny for int32.</p>
<p>On AVX2 the trick is <code>_mm256_maddubs_epi16</code>, which multiplies unsigned × signed bytes and adds neighbouring pairs into int16. Making one operand unsigned with <code>abs(a)</code> and moving a's sign onto b keeps each pair within 2 × 127 × 127 = 32,258, just under the int16 limit; <code>_mm256_madd_epi16</code> then widens to int32. ARM's <code>vdotq_s32</code> does 4-way int8 dot products directly.</p>
<p>Bytes per 32 weights: fp32 128, bf16 64, this format 36 (9 bits per weight), llama.cpp's <code>block_q8_0</code> 34 (it stores the scale in fp16; <code>ggml/src/ggml-common.h</code>). <code>examples/05_int8_dot.cpp</code> maps each step to llama.cpp's <code>ggml_vec_dot_q8_0_q8_0</code>; exercise 5 builds an int8 matvec and checks it against fp32 within an error bound you derive.</p>`,
        check: { q: "Why accumulate the int8 × int8 products of a 32-element block in int32 rather than int16?",
          options: ["int32 instructions are faster", "The sum can reach 32 × 127 × 127 = 516,128, which overflows int16 (max 32,767)", "int16 can't hold negative numbers"], answer: 1,
          why: "One product is at most 127 × 127 = 16,129, and 32 of them can reach 516,128. int16 overflows after just two maximal products; int32 holds the whole block with plenty of room." },
        legend: [["k", "int8 values"], ["v", "scale"], ["q", "integer math"], ["ok", "result"]],
        scene(G) {
          G.text(24, 36, "two q8 blocks: 32 int8 values + one scale each", { size: 13 });
          [["a", 52], ["b", 104]].forEach(([n, y]) => {
            G.box(24, y, 64, 40, `s${n}`, { fill: "v", size: 13 });
            for (let i = 0; i < 32; i++) G.rect(96 + i * 16, y, 14, 40, { fill: "k", rx: 2, opacity: 0.6 + 0.4 * ((i * 7) % 5) / 5 });
            G.label(96, y + 54, "", {});
          });
          const sum = G.box(96, 172, 300, 40, "Σ qa · qb  →  int32", { fill: "q", size: 13 });
          G.text(410, 198, "× sa · sb → float", { size: 13, color: "ok" });
          G.label(96, 232, "max |Σ| = 32 × 127 × 127 = 516,128", { size: 11, color: "ink" });
          G.text(24, 278, "bytes per 32 weights", { size: 13 });
          const rows = [["fp32", 128, "hot"], ["bf16", 64, "v"], ["q8, float scale", 36, "ok"], ["q8_0 (llama.cpp)", 34, "ok"]];
          const bars = rows.map(([n, b, col], i) => { const y = 292 + i * 30; G.label(24, y + 17, n, { size: 11, color: "ink" }); G.label(170 + b * 3.4 + 8, y + 17, `${b} B`, { size: 11, color: "ink" }); return G.rect(170, y, b * 3.4, 22, { fill: col, rx: 3 }); });
          G.from(bars, { attr: { width: 0 }, stagger: 0.15, duration: 0.4 }); G.from(sum, { opacity: 0, duration: 0.4, delay: 0.3 });
          G.caption("integer sums inside the block, one float multiply per block");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>SIMD registers hold 4, 8 or 16 floats; one intrinsic works on every lane. Vector loops need a scalar tail, and aligned buffers avoid loads that straddle cache lines.</li>
<li>Peak = FMA units × lanes × 2 × clock. Reaching it needs latency × units independent accumulators (8 in the example).</li>
<li>Arithmetic intensity I = FLOPs ÷ DRAM bytes. The roofline says attainable = min(P, I × B), with the ridge at P ÷ B.</li>
<li>Vector add, dot and matvec stay below 1 FLOP/byte and are memory-bound everywhere; matmul's intensity grows with n. LLM decode at batch 1 is matvecs, so its speed ≈ bandwidth ÷ model bytes.</li>
<li>The SGEMM ladder: loop order (stride), tiling (cache reuse), register micro-kernel (register reuse, hidden latency), threads. Int8 blocks cut bytes per weight about 4× versus fp32 with integer inner sums.</li></ul>
<p>Next, P0.5 puts it all together: an mmap loader, a thread pool and SIMD matvecs in a working CPU inference engine, and a roofline prediction of its speed.</p>`,
    sim: {
      title: "Build your own roofline",
      intro: "Describe a CPU (cores, vector width, FMA units, clock) and its memory bandwidth. The chart draws the roofline and places a kernel on it; the panels give its intensity, the limit that binds, and the fastest it could possibly run. Replace the example values with your measured peak (02_fma_peak) and bandwidth (03_stream).",
      height: 300,
      controls: [
        { id: "cores", label: "cores", min: 1, max: 64, value: 8 },
        { id: "lanes", label: "fp32 lanes", type: "select", value: 8, options: [[4, "4 (NEON / SSE)"], [8, "8 (AVX2)"], [16, "16 (AVX-512)"]] },
        { id: "units", label: "FMA units per core", type: "select", value: 2, options: [[1, "1"], [2, "2"]] },
        { id: "ghz", label: "clock, GHz (example value)", min: 1, max: 5, step: 0.1, value: 3, format: (v) => v.toFixed(1) },
        { id: "bw", label: "memory bandwidth, GB/s (example value)", min: 10, max: 800, step: 5, value: 50 },
        { id: "k", label: "kernel", type: "select", value: "gemm", options: [["add", "vector add"], ["dot", "dot product"], ["gemv", "matvec"], ["gemm", "matmul"], ["decode", "LLM decode"]] },
        { id: "lg", label: "size n = 2^x (add, dot, matvec, matmul)", min: 4, max: 14, value: 10, format: (v) => F.num(2 ** v) },
        { id: "b", label: "decode batch (Llama-3-8B)", min: 1, max: 256, value: 1 },
        { id: "wb", label: "decode weight bytes", type: "select", value: 2, options: [[4, "fp32 (4 B)"], [2, "bf16 (2 B)"], [1.0625, "q8_0 (34 B / 32)"]] },
      ],
      draw(G, v) {
        const P = v.cores * v.units * v.lanes * 2 * v.ghz * 1e9, B = v.bw * 1e9, n = 2 ** v.lg;
        let W, Q, name;
        if (v.k === "decode") { const params = 8.0e9; W = 2 * params * v.b; Q = params * v.wb; name = `decode, batch ${v.b}`; }
        else { const k = KERNELS[v.k]; W = k.W(n); Q = k.Q(n); name = `${k.name}, n = ${F.num(n)}`; }
        const I = W / Q, att = Math.min(P, I * B), ridge = P / B, t = Math.max(W / P, Q / B);
        const lo = 0.01, hi = 10000, yhi = P * 3, ylo = yhi / 1e5;
        const X = logX(70, 540, lo, hi), Y = logY(260, 230, ylo, yhi);
        G.line(70, 260, 610, 260, { color: "line" }); G.line(70, 30, 70, 260, { color: "line" });
        [0.01, 0.1, 1, 10, 100, 1000, 10000].forEach((x) => G.label(X(x), 276, String(x), { anchor: "middle", size: 10 }));
        for (let e = Math.ceil(Math.log10(ylo)); e <= Math.log10(yhi); e++) G.label(64, Y(10 ** e) + 4, F.si(10 ** e, 0).trim(), { anchor: "end", size: 10 });
        G.label(610, 294, "intensity, FLOP/byte →", { anchor: "end", size: 11 });
        G.label(74, 22, "attainable FLOP/s (log)", { size: 11 });
        const x0 = Math.max(lo, ylo / B);
        G.line(X(x0), Y(x0 * B), X(Math.min(hi, ridge)), Y(Math.min(P, hi * B)), { color: "k", w: 3 });
        if (ridge < hi) G.line(X(ridge), Y(P), X(hi), Y(P), { color: "v", w: 3 });
        if (ridge > lo && ridge < hi) { G.line(X(ridge), Y(P), X(ridge), 260, { color: "muted", dash: "4 4" }); G.label(X(ridge) + 5, 252, `ridge ${F.num(ridge, 1)}`, { size: 10 }); }
        // reference kernels
        [[1 / 12, "add"], [0.25, "dot"], [0.5, "matvec"]].forEach(([i, nm]) => { G.circle(X(i), Y(Math.min(P, i * B)), 3.5, { fill: "muted" }); G.label(X(i), Y(Math.min(P, i * B)) - 8, nm, { anchor: "middle", size: 9 }); });
        const Ic = Math.min(hi, Math.max(lo, I));
        G.circle(X(Ic), Y(att), 7, { fill: att < P ? "hot" : "ok" });
        G.label(X(Ic) + (X(Ic) > 420 ? 6 : -6), Y(att) - 14, name, { size: 11, color: "ink", anchor: X(Ic) > 420 ? "end" : "start" });
        const memBound = I * B < P;
        const rows = [["intensity I", F.num(I, I < 10 ? 3 : 0) + " FLOP/B"], ["ridge P / B", F.num(ridge, 1) + " FLOP/B"], ["attainable", F.si(att, 1) + "FLOP/s"], ["share of peak", (100 * att / P).toFixed(1) + "%"]];
        const panels = [
          { title: "Machine", rows: [["peak P", F.si(P, 1) + "FLOP/s"], ["per core", F.si(P / v.cores, 1) + "FLOP/s"], ["bandwidth B", F.si(B, 0) + "B/s"]], html: `<p class="note">P = cores × FMA units × lanes × 2 × clock. Example values: measure your own.</p>` },
          { title: "Kernel on this machine", rows, chip: [!memBound, memBound ? "memory-bound: cut bytes or add reuse" : "compute-bound: SIMD and threads matter"] },
          { title: "Work and time", rows: [["FLOPs", F.si(W, 2)], ["DRAM bytes", F.bytes(Q)], ["time ≥", F.ms(t)]].concat(v.k === "decode" ? [["tokens/s ≤ (all users)", F.num(v.b / t, 1)]] : []) },
          { title: "Model", html: `<p class="note">Ideal traffic: each operand crosses DRAM once (matmul assumes good tiling; decode counts weights only, not the KV cache). Real kernels move more bytes and reach a fraction of either roof.</p>` },
        ];
        return panels;
      },
    },
    practice: {
      intro: `All five exercises build as one CMake project with <code>-O3 -march=native</code>. Run the commands from the module folder <code>course/P0-systems-primer/P0.4-simd-and-roofline</code>. <code>-DS2S_USE_SOLUTIONS=ON</code> builds the reference solutions instead.`,
      items: [
        { title: "Vectorised SAXPY with a tail", tier: "T0 · easy", goal: "y = a·x + y with simd::fma, correct for every n from 0 to 67 and for misaligned starts.",
          cmd: "cmake -S exercises -B build/ex && cmake --build build/ex -j && ctest --test-dir build/ex -R 01-saxpy --output-on-failure" },
        { title: "Horizontal sum in ≤ 6 instructions", tier: "T0 · easy", goal: "Fold 8 lanes (AVX2) or 4 lanes (NEON) into one float, halving the live lanes each step.",
          cmd: "ctest --test-dir build/ex -R 02-hsum --output-on-failure" },
        { title: "D1: the SGEMM ladder", tier: "T0 · medium", goal: "Loop reorder, tiling and the 4 × 16 register micro-kernel, each matching a float64 reference on odd shapes.",
          cmd: "ctest --test-dir build/ex -R 03-sgemm-ladder --output-on-failure" },
        { title: "Multithread D1", tier: "T0 · medium", goal: "parallel_for over row blocks of C with the P0.3 pool; identical results and a scaling table.",
          cmd: "ctest --test-dir build/ex -R 04-sgemm-threads --output-on-failure" },
        { title: "Int8 GEMV with per-block scales", tier: "T0 · hard", goal: "Quantise to 32-value blocks, multiply with integer sums and two scales, and stay within the error bound you derive.",
          cmd: "ctest --test-dir build/ex -R 05-int8-gemv --output-on-failure" },
        { title: "Your roofline", tier: "T0 · lab", goal: "Measure peak and bandwidth, time every SGEMM rung, and plot them all on your own roofline.",
          cmd: "cmake -S examples -B build/examples -DCMAKE_BUILD_TYPE=Release && cmake --build build/examples -j && ./build/examples/02_fma_peak && ./build/examples/03_stream && cmake -S bench -B build/bench -DCMAKE_BUILD_TYPE=Release && cmake --build build/bench -j && uv run python bench/sgemm_bench.py && uv run python examples/04_roofline.py" },
      ],
      labs: [
        { label: "Examples: scalar vs SIMD add, FMA peak sweep, STREAM, roofline plot, int8 dot", path: "course/P0-systems-primer/P0.4-simd-and-roofline/examples/" },
        { label: "The 30-line AVX2 / NEON portability layer", path: "course/P0-systems-primer/P0.4-simd-and-roofline/examples/simd.hpp" },
        { label: "Benchmark: every SGEMM rung as % of measured peak", path: "course/P0-systems-primer/P0.4-simd-and-roofline/bench/sgemm_bench.py" },
        { label: "Animations: SIMD lanes and misaligned loads; the interactive roofline", path: "animations/p0-simd-lanes.html · animations/roofline.html" },
        { label: "Self-check questions", path: "course/P0-systems-primer/P0.4-simd-and-roofline/quiz.md" },
      ],
    },
  });
})();
