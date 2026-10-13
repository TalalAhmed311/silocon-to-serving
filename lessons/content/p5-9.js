/* P5.9 — Triton. A program owns a tile and you write tensor operations on it; the compiler picks threads, shared memory and tensor-core layouts. */
(function () {
  const F = S2S.fmt;
  // the four autotune configs of platform/kernels/triton_kernels/matmul.py: [BM, BN, BK, num_stages]
  const CONFIGS = [[128, 128, 32, 3], [128, 64, 32, 4], [64, 128, 32, 4], [64, 64, 32, 5]];
  // shared memory a software pipeline needs for fp16 operands: S stages of an A tile (BM×BK) and a B tile (BK×BN), 2 B each.
  // The compiler may add layout padding on top: this is the minimum.
  const pipeBytes = (BM, BN, BK, S, eb) => S * (BM * BK + BK * BN) * eb;
  const GPUS = {
    t4: { name: "T4 (sm_75)", smem: 64, note: "Turing: tensor cores for fp16, no cp.async pipelines" },
    l4: { name: "L4 (sm_89)", smem: 100, note: "Ada: cp.async, bf16/fp16 tensor cores" },
    a100: { name: "A100 (sm_80)", smem: 164, note: "Ampere" },
  };

  S2S.lesson({
    id: "p5-9", n: "P5.9", title: "Triton",
    subtitle: "CUDA deep dive · first principles · T0 correctness on a laptop, T2 for performance",
    kicker: "Lesson · ≈ 45 min",
    headline: "Write the tile, let the compiler write the threads",
    intro: `<p>P5.1 to P5.8 were hand-written CUDA: thread indices, shared memory, <code>__syncthreads</code>, fragment layouts, <code>cp.async</code> bookkeeping. <b>Triton</b> is a Python-embedded language and compiler that raises the level of abstraction by one step: instead of one thread, you program one <b>tile</b> of data, and the compiler decides how threads, shared memory and tensor-core instructions implement it. This lesson builds the model from a vector add, explains masks and the one-block-per-row softmax, shows what <code>tl.dot</code>, grouped ordering and <code>num_stages</code> hide from you, and rewrites the module's kernels (softmax, RMSNorm, matmul, FlashAttention-2). You can check correctness on a laptop with Triton's interpreter.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "5 exercises"],
    legend: [["k", "program / tile"], ["q", "A / query"], ["v", "memory / stage"], ["ok", "result"], ["hot", "masked / wrong"]],
    prev: "p5-8", next: "p5-10",
    steps: [
      { rail: "the idea", title: "Program a tile, not a thread",
        body: `<p>Think back to what a CUDA GEMM needed: map <code>threadIdx</code> to elements, stage tiles in shared memory, <code>__syncthreads</code> at the right moments, load fragments in the exact layout the tensor core wants, issue <code>cp.async</code> groups, guard every access against the array bound. Most of those lines say <i>how</i> to implement a tile operation, not <i>what</i> the tile operation is.</p>
<p>Triton's bet: you write the <b>what</b>. A Triton <b>program</b> (launched like a CUDA block, <code>tl.program_id</code>) owns a block of data and operates on <b>whole tensors</b>: <code>x = tl.load(ptrs)</code>, <code>tl.sum(x, axis=0)</code>, <code>tl.dot(a, b)</code>. There is no per-thread code at all. The compiler chooses how many threads and warps share the block, where to stage data, and how to lay values out for the tensor cores.</p>
<table><tr><th>CUDA (P5.1 to P5.8)</th><th>Triton</th></tr>
<tr><td><code>threadIdx</code>, warps</td><td>programs (<code>tl.program_id</code>) on whole blocks</td></tr>
<tr><td>shared memory, <code>__syncthreads</code></td><td>implicit: staging and barriers are generated</td></tr>
<tr><td>fragments, <code>ldmatrix</code>, <code>mma.sync</code></td><td><code>tl.dot(a, b)</code> on fp16 / bf16 blocks</td></tr>
<tr><td><code>cp.async</code> stages, by hand</td><td><code>num_stages</code> in a config</td></tr>
<tr><td>per-thread bounds checks</td><td><code>mask=</code> on loads and stores</td></tr>
<tr><td>templates for tile sizes</td><td><code>tl.constexpr</code> + <code>@triton.autotune</code></td></tr></table>
<p>The price is control: you cannot place a value in a particular register, choose the exact shared-memory layout, or specialise warps for different jobs. When a kernel needs that (Hopper-style FlashAttention-3), people go back to CUDA or CUTLASS.</p>`,
        scene(G) {
          const rows = [["threadIdx → element", "program_id → tile"], ["shared memory + sync", "implicit staging"], ["fragments, ldmatrix, mma", "tl.dot(a, b)"], ["cp.async stages", "num_stages"], ["bounds check per thread", "mask= on load / store"], ["tile-size templates", "constexpr + autotune"]];
          G.text(24, 32, "what you write", { size: 13 });
          G.text(24, 54, "CUDA", { size: 13, color: "hot" }); G.text(344, 54, "Triton", { size: 13, color: "ok" });
          const all = [];
          rows.forEach(([a, b], i) => {
            const y = 66 + i * 52;
            all.push(G.box(24, y, 250, 38, a, { stroke: "hot", color: "hot", size: 12 }));
            G.arrow(278, y + 19, 336, y + 19, { color: "muted" });
            all.push(G.box(344, y, 270, 38, b, { fill: "ok", size: 12 }));
          });
          G.label(24, 396, "the compiler writes the left column from the right one", { size: 13, color: "ink" });
          G.from(all, { opacity: 0, x: -14, stagger: 0.05, duration: 0.25 });
          G.caption("a tile-level program; the thread-level details are generated");
        } },

      { rail: "first kernel", title: "A first kernel: vector add",
        body: `<p>Here is the smallest useful Triton kernel (the course's tutorial 01):</p>
<div class="eq">@triton.jit
def add(x_ptr, y_ptr, out_ptr, n,
        BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs &lt; n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

grid = (triton.cdiv(n, 256),)
add[grid](x, y, out, n, BLOCK=256)</div>
<p>Read it as tensor code. <code>tl.arange(0, BLOCK)</code> is a vector 0 to 255; <code>offs</code> is the 256 element indices this program owns; <code>x_ptr + offs</code> is a vector of 256 pointers; <code>tl.load</code> gathers all of them at once. The launch grid says how many programs: <code>triton.cdiv(n, 256)</code>, a ceiling division.</p>
<div class="eq">n = 1000, BLOCK = 256
programs = cdiv(1000, 256) = 4
program 3: offs = 768 .. 1023
  mask = offs &lt; 1000
  → 232 lanes active, 24 masked off</div>
<p>The last program overshoots the array, so the <b>mask</b> turns the overshooting lanes off: they do not load and do not store. This is the same job as <code>if (i &lt; n)</code> in CUDA, but for a whole block at once. <code>BLOCK</code> is a <code>tl.constexpr</code>: a compile-time constant, so Triton compiles one specialised kernel per value. <code>tl.arange</code> sizes must be powers of two, which is the reason masks matter so much.</p>`,
        check: { q: "n = 1000, BLOCK = 256. How many lanes of the last program are masked off?",
          options: ["0", "24", "232"], answer: 1,
          why: "cdiv(1000, 256) = 4 programs cover 4 × 256 = 1024 positions. The last one owns offsets 768 to 1023 and only offsets below 1000 are valid: 232 active lanes, so 1024 - 1000 = 24 are masked." },
        scene(G) {
          G.text(24, 32, "n = 1000, BLOCK = 256 → 4 programs", { size: 13 });
          const x0 = 40, sc = 0.52, y0 = 80;
          const parts = [];
          for (let p = 0; p < 4; p++) {
            const w = 256 * sc, x = x0 + p * w;
            const active = p < 3 ? 256 : 232;
            parts.push(G.rect(x + 1, y0, active * sc - 2, 44, { fill: ["k", "q", "ok", "v"][p], rx: 3 }));
            if (p === 3) parts.push(G.rect(x + 1 + active * sc, y0, (256 - active) * sc - 2, 44, { fill: "hot", rx: 3, opacity: 0.8 }));
            G.text(x + w / 2, y0 + 28, "program " + p, { anchor: "middle", size: 12, color: "bg" });
            G.label(x + 2, y0 + 64, p * 256 + " .. " + (p * 256 + 255), { size: 11 });
          }
          G.label(x0 + 3 * 256 * sc + 120, y0 - 8, "24 masked", { size: 12, color: "hot", anchor: "middle" });
          G.rect(24, 200, 592, 190, { stroke: "line", rx: 8 });
          G.label(40, 228, "offs = pid · BLOCK + tl.arange(0, BLOCK)       a vector of indices", { size: 12, color: "ink" });
          G.label(40, 252, "mask = offs < n                                  a vector of booleans", { size: 12, color: "ink" });
          G.label(40, 276, "tl.load(x_ptr + offs, mask=mask)                a vector of pointers", { size: 12, color: "ink" });
          G.label(40, 308, "BLOCK is tl.constexpr: one compiled kernel per value", { size: 12, color: "v" });
          G.label(40, 332, "tl.arange sizes must be powers of two → pad and mask", { size: 12, color: "v" });
          G.label(40, 364, "grid = (cdiv(n, BLOCK),): the number of programs, like CUDA blocks", { size: 12 });
          G.from(parts, { opacity: 0, scaleY: 0, transformOrigin: "bottom", stagger: 0.1, duration: 0.3 });
          G.caption("the mask switches off lanes past the end of the array");
        } },

      { rail: "masks & other", title: "Masked lanes need a harmless value",
        body: `<p>A masked <code>tl.load</code> does not read memory, but the lane still holds <i>some</i> value, and later operations such as <code>tl.max</code> and <code>tl.sum</code> see all lanes. The <code>other=</code> argument sets that value. Choose it so the lane is invisible to the operation that follows.</p>
<p>Softmax over a row of 3 values <code>[-3, -1, -2]</code> must run in a block of 4 (a power of two). One lane is padding:</p>
<div class="eq">correct: e^x = 0.050, 0.368, 0.135
         sum = 0.553
         p = 0.090, 0.665, 0.245   (sum 1)

other = 0 (wrong): pad lane x = 0
  max = 0 (not -1!)
  e^(x-0) = 0.050, 0.368, 0.135, 1.000
  sum = 1.553
  p = 0.032, 0.237, 0.087   (sum 0.356)

other = -inf: e^-inf = 0 → pad lane vanishes ✓</div>
<p>With <code>other=0</code> the padding lane wins the max and adds <code>e⁰ = 1</code> to the sum, so every stored probability is wrong and they no longer add up to 1. With <code>other=-inf</code> the lane cannot raise the max and contributes exactly 0 to the sum. The same reasoning picks <code>other=0.0</code> for a matmul: a zero adds nothing to a dot product. The rule: <b>use the identity element of the operation that follows</b> (-inf for max, 0 for sum, 0 for a dot product). It is the first item in the module's common-mistakes list.</p>`,
        check: { q: "A softmax block is padded with masked lanes. Which tl.load argument keeps them from affecting the result?",
          options: ["other=0.0", "other=float('-inf')", "No argument: masked lanes are removed from the block"], answer: 1,
          why: "Masked lanes still occupy positions in the block. With -inf they cannot become the max and exp(-inf) = 0 adds nothing to the sum. With 0 the pad lane can win the max and adds exp(0) = 1 to the sum, making all probabilities wrong." },
        scene(G) {
          const bars = (x0, vals, cols, title, tcol) => {
            G.text(x0, 40, title, { size: 13, color: tcol });
            const out = vals.map((v, i) => {
              const h = v * 220;
              const r = G.rect(x0 + i * 62, 300 - h, 48, Math.max(h, 2), { fill: cols[i], rx: 3 });
              G.text(x0 + i * 62 + 24, 292 - h, v.toFixed(3), { anchor: "middle", size: 12 });
              G.label(x0 + i * 62 + 24, 320, i < 3 ? "x = " + [-3, -1, -2][i] : "pad", { anchor: "middle", size: 11 });
              return r;
            });
            return out;
          };
          const a = bars(30, [0.09, 0.665, 0.245, 0], ["ok", "ok", "ok", "line"], "other = -inf  (right)", "ok");
          const b = bars(340, [0.032, 0.237, 0.087, 0.644], ["hot", "hot", "hot", "hot"], "other = 0  (wrong)", "hot");
          G.label(30, 346, "pad p = 0; real lanes sum to 1", { size: 12, color: "ok" });
          G.label(340, 346, "pad steals 0.644; real sum 0.356", { size: 12, color: "hot" });
          G.label(30, 384, "use the identity of what follows: -inf for max, 0 for sum and dot", { size: 13, color: "ink" });
          G.from(a.concat(b), { attr: { height: 0 }, duration: 0.5, stagger: 0.05 });
          G.caption("a padding lane must not be able to win the max or add to the sum");
        } },

      { rail: "softmax", title: "Softmax: one program per row",
        body: `<p>Now the real softmax (<code>platform/kernels/triton_kernels/softmax.py</code>). One program handles one row, and if the row fits in one block, the whole row is one tensor:</p>
<div class="eq">row  = tl.program_id(0)
offs = tl.arange(0, BLOCK)
mask = offs &lt; n_cols
x = tl.load(x_ptr + row*stride + offs,
            mask=mask, other=-inf)
x = x - tl.max(x, axis=0)
num = tl.exp(x)
y = num / tl.sum(num, axis=0)
tl.store(y_ptr + ..., y, mask=mask)</div>
<p>Compare with P5.5's CUDA version: there, a warp per row, 32 values per lane in registers, <code>__shfl_xor_sync</code> trees for max and sum. Here the same thing is <code>tl.max(x, axis=0)</code> and <code>tl.sum(...)</code>: Triton generates the cross-lane reduction. The kernel reads the row once and writes once: the 1-read, 1-write floor from P5.5, with <code>BLOCK = next_power_of_2(cols)</code>.</p>
<div class="eq">cols = 1024  → BLOCK 1024, one block (4 KiB fp32)
cols = 20000 → next_pow2 = 32768 &gt; 16384
  → online loop, BLOCK = 4096
  → 5 iterations (4 full + 3,616 active)
  → 2 reads + 1 write (P5.5 online softmax)</div>
<p>Rows too long for one block use the online <code>(m, d)</code> loop of P5.5, written as an ordinary Python-looking <code>for</code> over blocks: <code>d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))</code>. The exercise test includes a 20,000-column row for exactly this path. Softmax is bandwidth-bound, so expect Triton to land near your CUDA GB/s: there is no tensor-core or pipelining advantage to win.</p>`,
        scene(G) {
          G.text(24, 32, "cols = 1024: the whole row is one tensor", { size: 13 });
          const cells = G.grid(40, 48, 1, 64, 8.6, 26, () => "k", { gap: 1, rx: 1 });
          G.arrow(320, 78, 320, 112, { color: "muted" });
          const r = [G.box(150, 116, 150, 34, "m = tl.max(x)", { fill: "q", size: 13 }), G.box(330, 116, 150, 34, "d = tl.sum(e^(x-m))", { fill: "ok", size: 12 })];
          G.label(150, 170, "no shuffles, no __syncthreads: the compiler writes them", { size: 12, color: "ink" });
          G.text(24, 214, "cols = 20000: online loop, BLOCK = 4096", { size: 13 });
          const blocks = [0, 1, 2, 3, 4].map((i) => G.box(40 + i * 114, 230, i < 4 ? 106 : 88, 34, i < 4 ? "4096" : "3616", { fill: "k", size: 12 }));
          G.label(40, 286, "pass 1: (m, d) merged block by block · pass 2: write e^(x-m)/d", { size: 12, color: "ink" });
          G.rect(24, 312, 592, 78, { stroke: "line", rx: 8 });
          G.label(40, 338, "one block per row (≤ 16384 cols): 1 read + 1 write", { size: 13, color: "ok" });
          G.label(40, 360, "longer rows: 2 reads + 1 write (P5.5 online softmax)", { size: 13, color: "ink" });
          G.label(40, 380, "bandwidth-bound: expect GB/s near your CUDA kernel", { size: 12 });
          G.from(cells, { opacity: 0, stagger: 0.004, duration: 0.2 });
          G.from(r.concat(blocks), { opacity: 0, duration: 0.3, stagger: 0.05, delay: 0.3 });
          G.caption("the reductions of P5.4 are single calls");
        } },

      { rail: "tl.dot", title: "tl.dot: tiles in, tensor cores out",
        body: `<p>The matmul kernel shows what the abstraction buys. A program owns a <code>BM × BN</code> tile of C and walks along K:</p>
<div class="eq">rm = pid_m*BM + tl.arange(0, BM)
rn = pid_n*BN + tl.arange(0, BN)
rk = tl.arange(0, BK)
a_ptrs = a + rm[:, None]*sam
           + rk[None, :]*sak
b_ptrs = b + rk[:, None]*sbk
           + rn[None, :]*sbn
acc = tl.zeros((BM, BN), tl.float32)
for k in range(cdiv(K, BK)):
    a = tl.load(a_ptrs, mask=..., other=0.0)
    b = tl.load(b_ptrs, mask=..., other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += BK*sak;  b_ptrs += BK*sbk</div>
<p><code>rm[:, None]</code> turns a vector of BM row indices into a column, <code>rk[None, :]</code> turns BK indices into a row, and adding them <b>broadcasts</b> to a BM × BK grid of pointers: one expression addresses the whole tile. <code>tl.dot(a, b)</code> multiplies a BM × BK tile by a BK × BN tile. With fp16 or bf16 inputs the compiler lowers it to tensor-core instructions: all of P5.7's fragment layouts, <code>ldmatrix</code> and the shared-memory staging are generated for you.</p>
<div class="eq">per loop iteration, BM = BN = 128, BK = 32:
  FLOPs = 2 · 128 · 128 · 32 = 1,048,576
  loads: A tile 128×32, B tile 32×128 = 8,192
         elements  (16 KiB in fp16)</div>
<p>Same FLOP count per k-tile as the WMMA kernel of P5.7. One trap: for <b>fp32 inputs</b>, <code>tl.dot</code> on Ampere and newer may use TF32 (tensor-core fp32 with 10 mantissa bits, about 1e-3 relative error per input). To compare fairly with your IEEE-fp32 SGEMM of P5.6, the kernel passes <code>input_precision="ieee"</code>. Likewise cuBLAS can use TF32 unless told not to (P5.6). Always check which precision each side of a benchmark actually used.</p>`,
        check: { q: "You benchmark Triton's fp32 tl.dot against your P5.6 SGEMM and Triton wins by a huge margin. What should you check first?",
          options: ["That the grid is large enough", "Whether tl.dot used TF32 tensor cores (input_precision) while your SGEMM used full IEEE fp32", "Whether num_warps is a power of two"], answer: 1,
          why: "On Ampere and newer, fp32 tl.dot defaults to TF32 on tensor cores, which is not the same arithmetic as IEEE fp32 on CUDA cores. The kernel in the repo sets input_precision=\"ieee\" for fp32 so the comparison is fair. A suspiciously large win usually means unequal precision." },
        scene(G) {
          const cs = 12;
          G.text(24, 32, "one program: a BM × BN tile of C", { size: 13 });
          G.label(250, 56, "B strip  BK × BN", { color: "k", size: 12 });
          G.grid(250, 62, 4, 14, cs, cs, () => "k", { gap: 1, rx: 1 });
          G.label(24, 150, "A strip", { color: "q", size: 12 }); G.label(24, 166, "BM × BK", { color: "q", size: 12 });
          G.grid(100, 112, 14, 4, cs, cs, () => "q", { gap: 1, rx: 1 });
          G.grid(250, 112, 14, 14, cs, cs, () => "ok", { gap: 1, rx: 1 });
          G.label(250, 292, "acc  BM × BN  (fp32, kept across K)", { color: "ok", size: 12 });
          G.arrow(430, 130, 430, 200, { color: "v", w: 2 });
          G.label(440, 150, "slide k by BK", { size: 12 });
          G.label(440, 168, "acc += tl.dot(a, b)", { size: 12, color: "ink" });
          G.rect(24, 316, 592, 76, { stroke: "line", rx: 8 });
          G.label(40, 340, "rm[:, None] + rk[None, :] broadcasts to a BM × BK grid of pointers", { size: 12, color: "ink" });
          G.label(40, 362, "tl.dot → tensor cores; fragments, ldmatrix, staging: generated", { size: 12, color: "ok" });
          G.label(40, 382, "fp32: set input_precision=\"ieee\" to match SGEMM", { size: 12, color: "hot" });
          G.caption("the same tiling as P5.6 and P5.7, written once at tile level");
        } },

      { rail: "grouped order", title: "Grouped ordering: make neighbours share panels",
        body: `<p>Programs run in the order of their <code>pid</code>. If <code>pid</code> simply walks the C tiles row by row, the programs running at the same moment all want different data. Take 4096³ with 128 × 128 tiles: a 32 × 32 grid of 1,024 programs. Suppose about 58 programs run concurrently (one per SM on a 58-SM L4, an example value). Row by row:</p>
<div class="eq">row-major pid → (pid // 32, pid % 32)
programs 0..57: C-tile rows 0 and 1, all 32 cols
  → 2 A row-panels + 32 B column-panels
  = 34 panels
  each panel 128 × 4096 × 4 B = 2 MiB (fp32)

grouped (GROUP_M = 8): pid_m = pid % 8,
                       pid_n = pid // 8
programs 0..57: tile rows 0..7, cols 0..7
  → 8 A panels + 8 B panels = 16 panels</div>
<p>Same work, <b>16 distinct panels instead of 34</b>: less data pulled in from DRAM, and what is shared is read by several programs at about the same time, so the <b>L2 cache</b> serves the repeats. This is the L2-reuse effect from P5.7 step 6 (and the reason tensor-core GEMMs can run at all), implemented by changing how a program computes its tile coordinates:</p>
<div class="eq">per_group = GROUP_M * num_n
group     = pid // per_group
first_m   = group * GROUP_M
size_m    = min(num_m - first_m, GROUP_M)
pid_m = first_m + (pid % per_group) % size_m
pid_n = (pid % per_group) // size_m</div>
<p>Inside a group of GROUP_M tile-rows, consecutive pids go <i>down</i> a column of tiles first, then across. The other thing to notice in the kernel is the ragged edge: loads use <code>mask=(rm[:, None] &lt; M) &amp; (rk[None, :] &lt; K - k*BK)</code> with <code>other=0.0</code>, so any M, N, K work: out-of-range lanes contribute zeros to the dot product.</p>`,
        check: { q: "What does grouped program ordering in the matmul kernel improve?",
          options: ["The number of FLOPs", "L2 cache hits: programs running together reuse the same A row-panels and B column-panels", "The accuracy of the accumulation"], answer: 1,
          why: "It does not change the arithmetic. It changes which tiles run at the same time: a square-ish block of tiles needs far fewer distinct panels of A and B than a thin row of tiles (16 versus 34 in the example), so more loads hit in L2." },
        scene(G) {
          const cs = 7, n = 32;
          const grid = (x0, y0, hit, title, tcol, note) => {
            G.text(x0, y0 - 12, title, { size: 13, color: tcol });
            const out = [];
            for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) out.push(G.rect(x0 + c * cs, y0 + r * cs, cs - 1, cs - 1, { fill: hit(r, c) ? tcol : "line", rx: 1, opacity: hit(r, c) ? 1 : 0.35 }));
            G.label(x0, y0 + n * cs + 18, note, { size: 12, color: "ink" });
            return out;
          };
          const pidRow = (r, c) => r * 32 + c, pidGrp = (r, c) => (c * 8 + (r % 8)) + 0;
          const a = grid(40, 70, (r, c) => pidRow(r, c) < 58, "row-major", "hot", "58 programs: 2 A + 32 B = 34 panels");
          const b = grid(340, 70, (r, c) => r < 8 && pidGrp(r, c) < 58, "grouped, GROUP_M = 8", "ok", "58 programs: 8 A + 8 B = 16 panels");
          G.label(40, 336, "each square = one program (a 128 × 128 tile of C); highlighted = running now", { size: 12 });
          G.rect(24, 350, 592, 42, { stroke: "line", rx: 8 });
          G.label(40, 376, "same work, fewer distinct A and B panels → more L2 hits (example: 58 SMs)", { size: 12, color: "ink" });
          G.from(a.concat(b), { opacity: 0, stagger: 0.0008, duration: 0.2 });
          G.caption("a square block of tiles shares panels; a thin row does not");
        } },

      { rail: "stages & autotune", title: "num_stages and autotune: the knobs left to you",
        body: `<p>P5.7's multi-stage <code>cp.async</code> pipeline, with its commit groups and <code>wait_group</code> counts, collapses to one number: <code>num_stages</code> in a <code>triton.Config</code>. The compiler generates the async copies and the buffer rotation. <code>num_warps</code> sets how many warps share a program (CUDA's block size ÷ 32). Tile shapes are <code>tl.constexpr</code> parameters. The kernel tries a list of configs and keeps the fastest:</p>
<div class="eq">CONFIGS = [
 Config({"BM":128,"BN":128,"BK":32,..},
        num_warps=4, num_stages=3),
 Config({"BM":128,"BN": 64,"BK":32,..},
        num_warps=4, num_stages=4),
 Config({"BM": 64,"BN":128,"BK":32,..}, ...4),
 Config({"BM": 64,"BN": 64,"BK":32,..}, ...5)]
@triton.autotune(configs=CONFIGS,
                 key=["M", "N", "K"])</div>
<p>Look at the shared memory those pipelines need (fp16 operands: stages × (BM·BK + BK·BN) × 2 bytes, before any padding the compiler adds):</p>
<div class="eq">128×128×32, 3 stages: 3 × 8,192 × 2 = 49,152 B
128× 64×32, 4 stages: 4 × 6,144 × 2 = 49,152 B
 64×128×32, 4 stages: 4 × 6,144 × 2 = 49,152 B
 64× 64×32, 5 stages: 5 × 4,096 × 2 = 40,960 B</div>
<p>The pattern is the P5.7 trade in numbers: smaller tiles get more stages for about the same shared-memory budget. <code>key=["M","N","K"]</code> means the tuning result is cached per problem shape; the <b>first call</b> compiles and times every config, so warm up before you benchmark (<code>triton.testing.do_bench</code> does). Configs that need more shared memory than the GPU has fail to compile and are skipped. With fp32 operands the byte counts double.</p>`,
        check: { q: "Where did the cp.async commit-group / wait_group machinery of P5.7 go in Triton?",
          options: ["It is gone: Triton cannot pipeline loads", "Into num_stages in the Config; the compiler generates the async copies and buffer rotation", "Into tl.constexpr parameters you write by hand"], answer: 1,
          why: "num_stages tells the compiler how many tiles to keep in flight; it emits the cp.async instructions, the commits and waits, and the shared-memory buffer rotation. You choose the depth, the compiler does the bookkeeping that was hand-written in P5.7." },
        scene(G) {
          G.text(24, 32, "shared memory of each autotune config (fp16, minimum)", { size: 13 });
          const x0 = 200, sc = 320 / 49152;
          const bs = CONFIGS.map(([BM, BN, BK, S], i) => {
            const y = 60 + i * 62, b = pipeBytes(BM, BN, BK, S, 2);
            G.label(24, y + 16, `${BM}×${BN}×${BK}`, { size: 12, color: "ink" });
            G.label(24, y + 34, `${S} stages`, { size: 11 });
            const stagesW = (b * sc) / S;
            const out = [];
            for (let s = 0; s < S; s++) out.push(G.rect(x0 + s * stagesW, y, stagesW - 3, 30, { fill: ["k", "q", "v", "ok", "blue"][s], rx: 3 }));
            G.text(x0 + 49152 * sc + 12, y + 20, F.num(b) + " B", { size: 12 });
            return out;
          });
          G.line(x0 + 49152 * sc, 50, x0 + 49152 * sc, 300, { color: "hot", dash: "4 4" });
          G.label(x0 + 49152 * sc - 6, 316, "48 KiB", { size: 11, color: "hot", anchor: "end" });
          G.rect(24, 330, 592, 62, { stroke: "line", rx: 8 });
          G.label(40, 354, "smaller tile → more stages for the same budget", { size: 13, color: "ink" });
          G.label(40, 376, "key = (M, N, K): tuned per shape; first call compiles: warm up", { size: 12, color: "v" });
          G.from([].concat(...bs), { opacity: 0, x: -10, stagger: 0.04, duration: 0.2 });
          G.caption("each coloured slice is one pipeline stage's pair of tiles");
        } },

      { rail: "FA-2", title: "FlashAttention-2 in Triton: the same algorithm as tensors",
        body: `<p>P5.8's FA-2 in CUDA gave each thread one query row (<code>q[64]</code>, <code>o[64]</code> in registers). In Triton the program owns a <b>tile</b> of BM query rows and the state is tensors: <code>m_i</code> and <code>l_i</code> of shape <code>[BM]</code> and <code>acc</code> of shape <code>[BM, D]</code>. Both matrix products are <code>tl.dot</code>, so they run on tensor cores:</p>
<div class="eq">q = tl.load(Q tile)              # [BM, D]
m_i = -inf; l_i = 0; acc = 0
for start_n in range(0, hi, BN):
    k = tl.load(K tile)          # [D, BN]
    s = tl.dot(q, k) * scale     # [BM, BN]
    s = tl.where(valid, s, -inf)
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(s - m_new[:, None])
    l_i = l_i*alpha + tl.sum(p, 1)
    v = tl.load(V tile)          # [BN, D]
    acc = acc*alpha[:, None]
          + tl.dot(p.to(v.dtype), v)
    m_i = m_new
acc = acc / l_i[:, None]</div>
<p>This is line for line the update of P5.8 step 4: <code>alpha</code> is the rescale factor <code>f</code>, applied to both <code>l_i</code> and <code>acc</code>. The causal case sets <code>hi = (start_m + 1) * BM</code> so later K/V tiles are never visited (the tile skipping of P5.8 step 8), and masks the diagonal tile with <code>offs_m &gt;= cols</code> (the diagonal stays visible).</p>
<div class="eq">acc: [64, 64] fp32 = 4,096 values
4 warps = 128 threads → 32 values per thread
(CUDA FA-2, P5.8: o[64] per thread, 64 threads)</div>
<p>The accumulator is spread across the program's threads by the compiler instead of being one thread's private row. In the module's exercise, the Triton version typically beats the CUDA-core FA-2 of P5.8 by a wide margin, and the reason is the tensor cores, not a cleverer algorithm. (<code>06-fused-attention.py</code> in the Triton tutorials adds the backward pass and deeper pipelining.)</p>`,
        scene(G) {
          G.text(24, 32, "per K/V tile: shapes of the tensors a program holds", { size: 13 });
          const b = [
            G.box(24, 56, 96, 40, "q [BM, D]", { fill: "q", size: 12 }),
            G.box(24, 112, 96, 40, "k [D, BN]", { fill: "k", size: 12 }),
            G.box(160, 82, 150, 44, "s = q·k  [BM, BN]", { fill: "v", size: 12 }),
            G.box(344, 82, 150, 44, "m_new, alpha, p", { fill: "w", size: 12 }),
            G.box(24, 168, 96, 40, "v [BN, D]", { fill: "k", size: 12 }),
            G.box(160, 168, 296, 44, "acc = acc·alpha + p·v  [BM, D]", { fill: "ok", size: 12 }),
          ];
          G.arrow(122, 76, 158, 98, { color: "muted" }); G.arrow(122, 132, 158, 112, { color: "muted" });
          G.arrow(312, 104, 342, 104, { color: "muted" }); G.arrow(419, 128, 419, 166, { color: "muted" });
          G.arrow(122, 188, 158, 190, { color: "muted" });
          G.label(504, 98, "tl.dot #1", { size: 12, color: "v" }); G.label(504, 114, "tensor cores", { size: 11, color: "v" });
          G.label(466, 180, "tl.dot #2", { size: 12, color: "ok" }); G.label(466, 196, "tensor cores", { size: 11, color: "ok" });
          G.rect(24, 250, 592, 142, { stroke: "line", rx: 8 });
          G.label(40, 276, "state: m_i [BM], l_i [BM], acc [BM, D] across the whole K/V sweep", { size: 12, color: "ink" });
          G.label(40, 298, "alpha = e^(m_i - m_new) rescales both l_i and acc (P5.8 step 4)", { size: 12, color: "ink" });
          G.label(40, 320, "causal: hi = (start_m + 1)·BM skips later tiles; diagonal: offs_m ≥ cols", { size: 12, color: "ink" });
          G.label(40, 346, "acc 64 × 64 over 128 threads = 32 values each (CUDA FA-2: 64, one row)", { size: 12, color: "k" });
          G.label(40, 372, "the gap to CUDA-core FA-2 is the tensor cores, not the algorithm", { size: 12, color: "ok" });
          G.from(b, { opacity: 0, y: 8, stagger: 0.06, duration: 0.25 });
          G.caption("the algorithm of P5.8 with tl.dot for both matmuls");
        } },

      { rail: "interpreter", title: "Test on a laptop: the interpreter",
        body: `<p>Triton's <b>interpreter</b> (<code>TRITON_INTERPRET=1</code>) executes the block operations in Python on CPU tensors: no GPU, no compilation. It checks <b>correctness</b> (masks, indexing, the online-softmax rescale, grouped ordering) at a speed of seconds per test. It says nothing about performance: for that you need a real GPU. The repo's tests set the flag automatically when no GPU is present, so the same file serves both tiers:</p>
<div class="eq">T0 (laptop, interpreter):
uv run --extra torch pytest \\
  course/P5-cuda-deep-dive/P5.9-triton/exercises \\
  -m torch

T2 (GPU box): the same command
reference kernels: S2S_SOLUTIONS=1 ...</div>
<p>The tests use shapes chosen to break the common mistakes: a 1 × 1 softmax, a 33-column row (not a power of two), a 20,000-column row (online path), a 1,000-column RMSNorm (masks), odd matmul shapes like 65 × 47 × 70 (ragged K), and attention with N = 100 and 130 (not multiples of the tile) in full and causal modes. Tolerances follow the earlier lessons: <code>rtol 1e-4</code> for fp32 softmax and matmul, <code>1e-2</code> for bf16 RMSNorm (the bf16 derivation from P5.5), and <code>2e-3</code> for fp16 matmul and attention (P5.7 and P5.8). The residual in the fused RMSNorm is compared <b>exactly</b>, since the same fp32 add and rounding are applied.</p>
<p>The module pins Triton <code>v3.8.0</code>. The installed torch brings its own matching Triton, so check <code>triton.__version__</code> and record it beside any numbers: details such as the <code>input_precision</code> argument can differ between versions. Exercise 5 (hard) is the <b>Triton Puzzles</b> (srush), 8 puzzles at least, which also run in the interpreter.</p>`,
        scene(G) {
          G.text(24, 32, "two tiers, one test file", { size: 13 });
          G.box(24, 50, 270, 190, "", { stroke: "ok", dash: "5 4" });
          G.text(40, 76, "T0 · laptop", { size: 14, color: "ok" });
          G.label(40, 100, "TRITON_INTERPRET=1 (automatic)", { size: 12, color: "ink" });
          G.label(40, 122, "block ops run in Python, on CPU", { size: 12 });
          G.label(40, 144, "checks correctness: masks, indexing,", { size: 12 });
          G.label(40, 162, "online rescale, ragged shapes", { size: 12 });
          G.label(40, 190, "seconds per test; no performance", { size: 12, color: "hot" });
          G.box(326, 50, 290, 190, "", { stroke: "v", dash: "5 4" });
          G.text(342, 76, "T2 · GPU (g6.xlarge)", { size: 14, color: "v" });
          G.label(342, 100, "same tests on real hardware", { size: 12, color: "ink" });
          G.label(342, 122, "tl.dot on tensor cores", { size: 12 });
          G.label(342, 144, "bench_triton_vs_cuda.py:", { size: 12 });
          G.label(342, 162, "Triton vs PyTorch vs your CUDA", { size: 12 });
          G.label(342, 190, "record triton.__version__", { size: 12, color: "v" });
          G.rect(24, 264, 592, 128, { stroke: "line", rx: 8 });
          G.label(40, 290, "test shapes that break the usual bugs:", { size: 13, color: "ink" });
          G.label(40, 314, "softmax 1×1, 33 cols, 20,000 cols · RMSNorm 1,000 cols (masks)", { size: 12, color: "ink" });
          G.label(40, 336, "matmul 65×47×70 (ragged K) · attention N = 100, 130, full and causal", { size: 12, color: "ink" });
          G.label(40, 360, "tolerances: 1e-4 fp32 · 1e-2 bf16 norm · 2e-3 fp16 matmul, attention", { size: 12, color: "ink" });
          G.caption("T0 proves it is right; only a GPU tells you it is fast");
        } },

      { rail: "when to use", title: "What you gain, what you give up",
        body: `<p>Rewrite all four kernels, then compare. What to expect from each (the repo has not recorded numbers yet; predict first, then measure with <code>bench_triton_vs_cuda.py</code>):</p>
<table><tr><th>kernel</th><th>why Triton lands where it does</th></tr>
<tr><td>softmax</td><td>bandwidth-bound: near your CUDA GB/s. Nothing to gain beyond 1 read + 1 write.</td></tr>
<tr><td>RMSNorm (+ residual)</td><td>same: two loops over the row, fp32 accumulation, the shape of vLLM's <code>fused_add_rms_norm</code>.</td></tr>
<tr><td>matmul fp16</td><td><code>tl.dot</code> + <code>num_stages</code> give tensor cores and pipelining: near cuBLAS is plausible.</td></tr>
<tr><td>matmul fp32</td><td>CUDA cores with <code>ieee</code>: comparable to your rung-7 SGEMM at best.</td></tr>
<tr><td>FA-2</td><td>both matmuls on tensor cores: well ahead of the CUDA-core FA-2 of P5.8.</td></tr></table>
<p>The pattern: where the CUDA kernel was already at the bandwidth floor (softmax, norms), Triton ties; where the win came from hardware (tensor cores) or hand-built pipelining, Triton gets it with a fraction of the code. That is why serving stacks use it for custom ops (P5.10's alternatives).</p>
<p>What you give up: control over register allocation, exact shared-memory layouts, warp specialisation, and anything that does not fit block-level tensor operations. Debugging is also different: errors are in compiler output, and performance changes with the Triton version. When you need the last 10% or a Hopper-specific design, you drop to CUDA or CUTLASS (P5.6, P5.7). <b>Deliverable of the module:</b> the CUDA-vs-Triton table, plus 3 to 5 sentences per kernel on what Triton did for you (tiling, shared memory, tensor-core layouts, pipelining) and what you could not control.</p>
<p>Common mistakes to avoid: forgetting <code>other=</code> on a masked load; a non-power-of-two <code>tl.arange</code>; comparing TF32 with IEEE; timing the first (tuning) call.</p>`,
        scene(G) {
          const rows = [["softmax", "bandwidth floor", "≈ CUDA", "k"], ["RMSNorm", "bandwidth floor", "≈ CUDA", "k"], ["matmul fp16", "tensor cores + stages", "near cuBLAS", "ok"], ["matmul fp32", "CUDA cores, ieee", "≈ rung 7", "w"], ["FA-2", "2 × tl.dot", "≫ CUDA-core FA-2", "ok"]];
          G.text(24, 32, "predict, then measure (no numbers recorded yet)", { size: 13 });
          [24, 170, 360].forEach((x, i) => G.label(x, 58, ["kernel", "what decides it", "expect"][i], { size: 12 }));
          const bs = [];
          rows.forEach(([a, b, c, col], i) => {
            const y = 70 + i * 46;
            bs.push(G.box(24, y, 132, 34, a, { fill: col, size: 12 }));
            G.label(170, y + 22, b, { size: 12, color: "ink" });
            G.text(360, y + 22, c, { size: 13, color: col === "w" ? "muted" : col });
          });
          G.rect(24, 312, 592, 80, { stroke: "line", rx: 8 });
          G.label(40, 338, "gives up: register control, exact smem layout, warp specialisation", { size: 12, color: "hot" });
          G.label(40, 360, "gains: tensor cores and pipelining for a fraction of the code", { size: 12, color: "ok" });
          G.label(40, 380, "need the last 10% or Hopper tricks → CUDA / CUTLASS (P5.6, P5.7)", { size: 12 });
          G.from(bs, { opacity: 0, x: -14, stagger: 0.07, duration: 0.25 });
          G.caption("Triton ties where memory was the limit and wins where hardware was");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A Triton program owns a tile and writes tensor operations: <code>tl.load</code>, <code>tl.max</code>, <code>tl.dot</code>. Threads, shared memory, barriers and tensor-core layouts are generated.</li>
<li>Masks switch off lanes past the end of the data; <code>other=</code> must be the identity of what follows (-inf for max, 0 for sums and dot products). <code>tl.arange</code> sizes are powers of two.</li>
<li>Softmax is one program per row (1 read, 1 write) with an online loop for long rows; matmul is <code>acc += tl.dot(a, b)</code> over K with ragged edges masked.</li>
<li>Grouped ordering changes which tiles run together (16 distinct panels instead of 34 in the example) to raise L2 hits; <code>num_stages</code> replaces hand-written <code>cp.async</code> pipelines; autotune picks per-shape configs, so warm up before timing.</li>
<li>FA-2 in Triton is P5.8's algorithm with state as tensors and two <code>tl.dot</code> calls; its edge over CUDA-core FA-2 is the tensor cores.</li>
<li>The interpreter (<code>TRITON_INTERPRET=1</code>) checks correctness on a laptop; only a GPU measures speed. Compare precisions honestly (TF32 vs IEEE) and record the Triton version.</li></ul>`,
    sim: {
      title: "Tile, stages and shared memory",
      intro: "Pick a matmul tile (BM × BN × BK), the pipeline depth (num_stages) and the operand type. The panels show the shared memory the software pipeline needs at minimum, the FLOPs per k-tile, and whether it fits the SM; the lower bar shows how grouped ordering changes the number of distinct A and B panels that a given number of concurrent programs touches.",
      height: 280,
      controls: [
        { id: "bm", label: "BM", type: "select", value: 128, options: [[32, "32"], [64, "64"], [128, "128"], [256, "256"]] },
        { id: "bn", label: "BN", type: "select", value: 128, options: [[32, "32"], [64, "64"], [128, "128"], [256, "256"]] },
        { id: "bk", label: "BK", type: "select", value: 32, options: [[16, "16"], [32, "32"], [64, "64"]] },
        { id: "st", label: "num_stages", type: "select", value: 3, options: [[1, "1"], [2, "2"], [3, "3"], [4, "4"], [5, "5"], [6, "6"]] },
        { id: "eb", label: "operand type", type: "select", value: 2, options: [[2, "fp16 / bf16 (2 B)"], [4, "fp32 (4 B)"]] },
        { id: "gpu", label: "GPU (example values)", type: "select", value: "l4", options: [["t4", "T4 · 64 KiB per SM"], ["l4", "L4 · 100 KiB per SM"], ["a100", "A100 · 164 KiB per SM"]] },
        { id: "gm", label: "GROUP_M", type: "select", value: 8, options: [[1, "1 (row-major)"], [2, "2"], [4, "4"], [8, "8"], [16, "16"]] },
        { id: "conc", label: "programs running at once", min: 10, max: 256, step: 2, value: 58 },
      ],
      draw(G, v) {
        const BM = v.bm, BN = v.bn, BK = v.bk, S = v.st, eb = v.eb, g = GPUS[v.gpu];
        const stage = (BM * BK + BK * BN) * eb, total = stage * S, cap = g.smem * 1024;
        const flops = 2 * BM * BN * BK;
        const x0 = 40, W = 560, y0 = 70;
        G.text(x0, 28, "shared memory of the pipeline: " + S + " × " + F.num(stage) + " B", { size: 13 });
        const colors = ["k", "q", "v", "ok", "blue", "pink"];
        const sc = Math.min(1, W / Math.max(total, cap));
        for (let s = 0; s < S; s++) G.rect(x0 + s * stage * sc, y0, Math.max(2, stage * sc - 2), 44, { fill: colors[s], rx: 4 });
        G.line(x0 + cap * sc, y0 - 14, x0 + cap * sc, y0 + 62, { color: "hot", dash: "5 4", w: 2 });
        G.label(x0 + cap * sc, y0 + 80, `SM capacity ${g.smem} KiB`, { size: 12, color: "hot", anchor: cap * sc > W - 60 ? "end" : "middle" });
        G.label(x0, y0 + 80, F.num(total) + " B = " + (total / 1024).toFixed(1) + " KiB", { size: 12, color: "ink" });
        // distinct panels for `conc` programs on a 4096-wide problem tiled BM × BN
        const numM = Math.ceil(4096 / BM), numN = Math.ceil(4096 / BN), GM = Math.min(v.gm, numM), conc = v.conc;
        const rowsSeen = new Set(), colsSeen = new Set();
        const perGroup = GM * numN;
        for (let pid = 0; pid < Math.min(conc, numM * numN); pid++) {
          const group = Math.floor(pid / perGroup), firstM = group * GM, sizeM = Math.min(numM - firstM, GM);
          rowsSeen.add(firstM + ((pid % perGroup) % sizeM)); colsSeen.add(Math.floor((pid % perGroup) / sizeM));
        }
        const panels = rowsSeen.size + colsSeen.size, panelB = 4096 * Math.max(BM, BN) * eb;
        G.text(x0, y0 + 130, `grouped order, N = 4096: ${conc} concurrent programs touch ${panels} panels`, { size: 13 });
        const maxP = numM + numN, pw = (W * panels) / maxP;
        G.rect(x0, y0 + 144, W, 22, { fill: "line", rx: 3, opacity: 0.5 });
        G.rect(x0, y0 + 144, pw, 22, { fill: "ok", rx: 3 });
        G.label(x0, y0 + 186, `${rowsSeen.size} A row-panels + ${colsSeen.size} B column-panels (of ${numM} + ${numN})`, { size: 12, color: "ink" });
        const fits = total <= cap;
        return [
          { title: "Pipeline memory", rows: [["per stage (A + B tiles)", F.bytes(stage)], ["stages", String(S)], ["total, minimum", F.bytes(total)], ["SM capacity (example)", g.smem + " KiB"], ["share of SM", ((total / cap) * 100).toFixed(0) + "%"], ["blocks per SM by smem", String(Math.floor(cap / total))]],
            gauge: [[Math.min(1, total / cap), fits ? "v" : "hot"]], gaugeText: `${(total / 1024).toFixed(1)} of ${g.smem} KiB`,
            chip: [fits, fits ? "fits one SM" : "too big: the config would be skipped"] },
          { title: "Per k-tile and reuse", rows: [["FLOPs per k-tile (2·BM·BN·BK)", F.num(flops)], ["bytes loaded per k-tile", F.bytes(stage)], ["FLOP per byte (this tile)", (flops / stage).toFixed(1)], ["distinct panels, " + conc + " programs", String(panels)], ["panel size (N = 4096)", F.bytes(panelB)], ["data in those panels", F.bytes(panels * panelB)]],
            html: `<p class="note">Minimum shared memory only: the compiler may add layout padding, and real autotuning also weighs registers and occupancy. Panel counts follow the repo's grouped-ordering formula for a 4096 × 4096 output; "programs running at once" is your assumption (for example, one block per SM). SM sizes are example values recalled from the CUDA Programming Guide table, UNVERIFIED.</p>` },
        ];
      },
    },
    practice: {
      intro: `Correctness runs on a laptop with Triton's interpreter (T0); performance needs a GPU (T2, an L4 <code>g6.xlarge</code> recommended). Implement the four files in <code>course/P5-cuda-deep-dive/P5.9-triton/exercises/ours/</code>; each raises <code>NotImplementedError</code> until you do. Use <code>S2S_SOLUTIONS=1</code> to run the same tests against the reference kernels in <code>platform/kernels/triton_kernels/</code>. Measured numbers are not yet recorded in the repo (<code>TODO(run-on: g6.xlarge)</code>).`,
      items: [
        { title: "Softmax: one program per row", tier: "T0 / T2", goal: "A row in one block with mask and other=-inf, plus an online loop for rows longer than one block. The test includes a 20,000-column row. Then compare GB/s with your P5.5 CUDA softmax.",
          cmd: "uv run --extra torch pytest course/P5-cuda-deep-dive/P5.9-triton/exercises -m torch -k softmax" },
        { title: "RMSNorm with fused residual", tier: "T0 / T2", goal: "fp32 accumulation, any storage dtype, optional residual updated in place and compared exactly. Compare with P5.5's fused_add_rmsnorm.",
          cmd: "uv run --extra torch pytest course/P5-cuda-deep-dive/P5.9-triton/exercises -m torch -k rmsnorm" },
        { title: "Matmul: tiles, tl.dot, grouped order, autotune", tier: "T0 / T2", goal: "Block tiles, masks for any shape, grouped program ordering and a config list; fp32 inputs use input_precision=\"ieee\". Compare with your P5.6 rung 7 (fp32) and P5.7 HGEMM (fp16).",
          cmd: "uv run --extra torch pytest course/P5-cuda-deep-dive/P5.9-triton/exercises -m torch -k matmul" },
        { title: "FlashAttention-2 forward + causal", tier: "T0 / T2", goal: "One program per (query block, batch x head), K/V tiles streaming, (m, l, acc) as tensors, two tl.dot calls, causal tile skipping. Compare with your P5.8 CUDA FA-2.",
          cmd: "uv run --extra torch pytest course/P5-cuda-deep-dive/P5.9-triton/exercises -m torch -k flash" },
        { title: "Triton Puzzles (hard)", tier: "T0", goal: "Puzzles 1 to 8 at least, in the interpreter, from srush/Triton-Puzzles (linked, not copied). They train the tile-level reading of indexing and masks.",
          cmd: "git clone https://github.com/srush/Triton-Puzzles" },
      ],
      labs: [
        { label: "Reference Triton kernels: softmax, rmsnorm, matmul, flash_attention", path: "platform/kernels/triton_kernels/__init__.py" },
        { label: "Triton vs PyTorch on the D4 shapes, next to your CUDA numbers (PYTHONPATH=platform)", path: "course/P5-cuda-deep-dive/P5.9-triton/examples/bench_triton_vs_cuda.py" },
        { label: "The exercise tests (interpreter on CPU or a real GPU)", path: "course/P5-cuda-deep-dive/P5.9-triton/exercises/test_triton.py" },
        { label: "Your four starter files", path: "course/P5-cuda-deep-dive/P5.9-triton/exercises/ours/matmul.py" },
        { label: "AWS guide (g6.xlarge, about 2 instance-hours; record triton.__version__)", path: "course/P5-cuda-deep-dive/P5.9-triton/aws.md" },
        { label: "Triton tutorials 01, 02, 03, 05, 06 (python/tutorials at the pinned tag v3.8.0)", path: "https://github.com/triton-lang/triton/tree/v3.8.0/python/tutorials" },
        { label: "Triton Puzzles (srush)", path: "https://github.com/srush/Triton-Puzzles" },
      ],
    },
  });
})();
