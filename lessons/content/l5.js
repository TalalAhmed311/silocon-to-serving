/* L5 — GEMM and tensor cores (Lane B). Concepts behind the L5 LeetGPU problems: arithmetic intensity, tiling, register
   blocking, feeding the tile (k-major A, predication, alpha/beta), double buffering, tensor cores, batched GEMM,
   int8/int4 quantization, epilogue fusion, and how to measure a GEMM (TFLOP/s, waves). */
(function () {
  const F = S2S.fmt;
  const LV = "laneB-cuda/L5-gemm-and-tensor-cores";
  const mono = (h) => h.replace(/<code>(.*?)<\/code>/g, '<span style="font-family:var(--f-mono);font-size:13px">$1</span>');
  const P = (slug, title, diff, prio, goal) => ({
    title: `#${slug.split("-")[0]} ${title}`, tier: `${diff} · ${prio}`,
    goal: mono(`${goal} Hints and reference solution: <code>${LV}/solutions/${slug}/</code>`),
    cmd: `ctest --test-dir build/laneB -R '^L5_${slug}$' --output-on-failure`,
  });
  // log-scale helpers for the roofline drawings
  const lg = (v, lo, hi, a, b) => a + ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (b - a);

  S2S.lesson({
    id: "l5", n: "L5", title: "GEMM and tensor cores",
    subtitle: "Lane B · CUDA on LeetGPU · T2 harness (cuBLAS comparison); tensor-core rungs sm_80+",
    kicker: "Lesson · ≈ 80 min",
    headline: "The same arithmetic, a hundred times faster",
    intro: `<p>A <b>GEMM</b> (general matrix multiply) computes <code>C = A · B</code>. It is where an LLM spends most of its arithmetic: the attention projections and the MLP are all GEMMs. Every correct GEMM does the same 2·M·N·K floating-point operations. What separates a slow kernel from one near cuBLAS is how many times each number fetched from memory is <i>reused</i> before it is thrown away.</p>
<p>The dark panel climbs that ladder: arithmetic intensity, shared-memory tiles, register blocking, feeding the tile, double buffering, tensor cores. Then it covers the variants the LeetGPU problems add (batched, int8, int4) and the epilogue fusions that connect GEMM to L4. The Simulate tab is a roofline calculator for your tile choices.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 core problems"],
    legend: [["k", "operand / load"], ["v", "accumulator / result"], ["hot", "bottleneck"], ["ok", "good"]],
    prev: "l4", next: "l6",
    steps: [
      { rail: "intensity", title: "Count FLOPs per byte before writing a line of code",
        body: `<p>Multiplying an M × K matrix by a K × N matrix produces M·N outputs, each a dot product of K pairs: K multiplies and K adds. One fused multiply-add (FMA) is 2 floating-point operations (FLOPs), so:</p>
<div class="eq">FLOPs = 2 * M * N * K
M = N = K = 4096:
  2 * 4096^3 = 137,438,953,472 (137 GFLOP)</div>
<p>The least memory traffic possible is reading A and B once and writing C once. In fp32 that is 3 · 4096² · 4 bytes = 201,326,592 bytes (201 MB). Dividing gives the <b>arithmetic intensity</b>, FLOPs per byte moved:</p>
<div class="eq">ideal: 137.4e9 / 201.3e6 = 683 FLOP/byte
naive: each FMA loads 2 floats = 8 B for 2 FLOP
       = 0.25 FLOP/byte</div>
<p>Is a kernel limited by memory or by arithmetic? Compare its intensity with the GPU's <b>ridge point</b>, peak FLOP/s ÷ peak bytes/s. As an example, an NVIDIA T4 lists about 8.1 TFLOP/s fp32 and 320 GB/s (example values from the datasheet, check the spec sheet for your GPU): ridge = 8.1e12 / 320e9 ≈ 25 FLOP/byte. Below 25 the kernel waits on memory; above, on the FMA units. The ideal GEMM at 683 is far above it, so a perfect kernel is compute-bound (about 17 ms at 8.1 TFLOP/s). The naive kernel at 0.25 is a hundred times below the ridge and runs at roughly 1% of peak (0.25 × 320 GB/s = 80 GFLOP/s).</p>
<p>Everything in this lesson is a way to raise the intensity of the kernel you actually run: reuse each loaded value more times.</p>`,
        scene(G) {
          const x0 = 50, x1 = 600, yb = 300, yt = 70, lo = 0.1, hi = 1000, peak = 8.1, bw = 320;
          G.text(24, 32, "roofline, example GPU: 8.1 TFLOP/s, 320 GB/s", { size: 13 });
          const X = (i) => lg(i, lo, hi, x0, x1), Y = (t) => lg(t, 0.01, 20, yb, yt);
          G.line(x0, yb, x1, yb, { color: "muted", w: 1 }); G.line(x0, yb, x0, yt - 10, { color: "muted", w: 1 });
          let d = ""; [0.1, 25.3, 1000].forEach((i, k) => { const t = Math.min(peak, (i * bw) / 1000); d += (k ? " L " : "M ") + X(i).toFixed(1) + " " + Y(t).toFixed(1); });
          G.path(d, { color: "ink", w: 2.5 });
          [[0.25, "naive 0.25", "hot", 22], [8, "32×32 tile: 8", "v", 24], [32, "128×128: 32", "q", -14], [683, "ideal: 683", "ok", -14]].forEach(([i, nm, c, dy]) => {
            const t = Math.min(peak, (i * bw) / 1000); G.circle(X(i), Y(t), 6, { fill: c }); G.label(X(i) - 20, Y(t) + dy, nm, { size: 10, color: c });
          });
          [0.1, 1, 10, 100, 1000].forEach((i) => G.label(X(i) - 8, yb + 16, String(i), { size: 10 }));
          G.label(x0 + 150, yb + 34, "FLOP per byte (log scale)", { size: 11 }); G.label(x0 + 8, yt - 16, "attainable TFLOP/s", { size: 11 });
          G.label(X(25.3) - 190, Y(peak) - 18, "ridge ≈ 25 FLOP/B", { size: 11, color: "ink" });
          G.text(24, 372, "4096³ GEMM: 137 GFLOP, 201 MB minimum → 683 FLOP/B", { size: 13, color: "v" });
          G.text(24, 398, "left of the ridge: memory-bound; right: compute-bound", { size: 13, color: "ink" });
          G.caption("raise the intensity: reuse each loaded number");
        } },

      { rail: "tiling", title: "Tiles: load once into shared memory, reuse many times",
        body: `<p>L2 introduced the idea: a block loads a tile of A and a tile of B into shared memory, synchronizes, and all its threads reuse them. Count the benefit. A block that owns a BM × BN tile of C and walks K in steps of BK loads (BM + BN) · BK values per step and performs BM · BN · BK multiply-adds:</p>
<div class="eq">global loads per FMA = (BM + BN) / (BM * BN)
32 x 32 tile:   64 / 1024    = 1/16   (naive: 2)
128 x 128 tile: 256 / 16384  = 1/64
intensity = 2 * BM * BN / ((BM + BN) * 4 B)
32 x 32:   2048 / 256  =  8 FLOP/B
128 x 128: 32768 / 1024 = 32 FLOP/B</div>
<p>Bigger tiles reuse more, and bytes drop in proportion to the tile's side. The limit is on-chip capacity: a tile pair costs (BM + BN) · BK · 4 bytes of shared memory per block (two 16-deep strips of 128 columns each, 16 KiB). And a 128 × 128 tile has 16,384 outputs while a block has at most 1,024 threads, so each thread must compute several outputs. That is the next rung.</p>
<p>The ladder in <code>platform/kernels/include/d4/gemm.cuh</code> (P5.6) names these steps: rung 1 (threadIdx.x walks rows: uncoalesced), rung 2 (walks columns: coalesced, L2), rung 3 (shared tiles). The rungs differ in code by a few lines and by a factor of 10 or more in speed, so measure each one.</p>`,
        check: { q: "A block computes a 64 × 64 tile of C in fp32 (BK = 16). Roughly how many FLOPs does it do per byte it loads from global memory?",
          options: ["8 (like a 32 × 32 tile)", "16", "64"], answer: 1,
          why: "Per k-step the block loads (64 + 64) · 4 B = 512 B of operands per 64 · 64 = 4096 FMAs = 8192 FLOPs, so intensity = 8192 / 512 = 16 FLOP/byte. Doubling the tile side doubles the intensity (32 × 32 gives 8, 128 × 128 gives 32). BK cancels out of the ratio." },
        scene(G) {
          const T = [8, 16, 32, 64, 128], base = 330, sc = 360;
          G.text(24, 32, "global loads per FMA = 2 / tile side (square tiles)", { size: 13 });
          G.rect(24, 62, 150, 26, { fill: "hot", rx: 4 }); G.label(182, 80, "naive (rung 1, 2): 2 loads per FMA", { size: 11, color: "hot" });
          const bars = T.map((t, i) => {
            const v = 2 / t, h = Math.max(4, v * 360), x = 40 + i * 110;
            G.rect(x, base - h * 0.6, 70, h * 0.6, { fill: i === 2 ? "v" : "k", rx: 4, opacity: 0.9 });
            G.label(x + 4, base + 16, `${t}×${t}`, { size: 11 }); G.label(x + 4, base - h * 0.6 - 6, v.toFixed(4).replace(/0+$/, ""), { size: 11, color: "ink" });
            return 0;
          });
          G.label(24, 372, "intensity = tile side / 2 FLOP per byte (fp32): 8 → 16 → 32 → …", { size: 12, color: "ok" });
          G.label(24, 394, "bigger tile = more reuse, but more shared memory and registers", { size: 12 });
          G.caption("loads per FMA fall in proportion to the tile side");
        } },

      { rail: "register blocking", title: "Inside the tile, the next wall is shared memory itself",
        body: `<p>With 32 × 32 tiles the inner loop is <code>acc += As[ty][k] * Bs[k][tx]</code>: <b>2 shared-memory loads for every FMA</b>. Shared memory is fast but an SM can serve only so many requests per clock, while its FMA units want fresh operands every clock. Now the kernel waits on shared memory.</p>
<p>The same idea applies one level down. <b>Registers</b> are the fastest storage and private to a thread. Let each thread compute a small tile of TM × TN outputs. For each k it loads TM values of A and TN values of B into registers and does all TM · TN products (an <b>outer product</b>):</p>
<div class="eq">per k: loads = TM + TN, FMAs = TM * TN
1 x 1 (rung 3):  2 loads / 1 FMA   = 2.0
8 x 1 (rung 4): (1 + 8) / 8        = 1.125
4 x 4 (L5 #22):  8 / 16            = 0.5
8 x 8:          16 / 64            = 0.25</div>
<p>A 64 × 64 block tile with 4 × 4 per thread needs 64 · 64 / 16 = 256 threads, and each thread holds 16 accumulators plus 8 operand registers. The L5 reference (<code>${LV}/solutions/22-gemm/kernel.cu</code>) uses exactly this: <code>GBM = GBN = 64, GBK = 16, GTM = GTN = 4</code>. Bigger thread tiles cut shared-memory traffic further but use more registers per thread, which lowers occupancy (L2) and, past the register limit, spills to slow local memory. The reference's README says "4 FMAs per shared-memory load"; by the count above 4 × 4 gives 2 FMAs per load, and 8 × 8 gives 4.</p>`,
        check: { q: "Each thread computes a 4 × 4 block of outputs. For one k it loads 4 values of A and 4 values of B from shared memory. How many shared loads per FMA is that?",
          options: ["1 (8 loads for 8 FMAs)", "0.5 (8 loads for 16 FMAs)", "2 (same as no blocking)"], answer: 1,
          why: "The 4 A values and 4 B values form an outer product of 16 products, so 8 loads feed 16 FMAs: 0.5 loads per FMA, a quarter of the 2.0 of an unblocked thread. An 8 × 8 thread tile would give 16 loads for 64 FMAs = 0.25." },
        scene(G) {
          const cw = 40;
          G.text(24, 32, "one thread, one k: outer product of a (4) and b (4)", { size: 13 });
          const a = [0, 1, 2, 3].map((i) => G.box(24, 100 + i * 46, 56, 40, `a${i}`, { fill: "k", size: 13 }));
          const b = [0, 1, 2, 3].map((j) => G.box(130 + j * 70, 50, 64, 36, `b${j}`, { fill: "q", size: 13 }));
          const acc = [];
          for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) acc.push(G.box(130 + j * 70, 100 + i * 46, 64, 40, `c${i}${j}`, { fill: "v", size: 12, boxOpacity: 0.85 }));
          G.label(24, 300, "8 shared-memory loads (a0 … a3, b0 … b3)", { size: 12, color: "k" });
          G.label(24, 320, "16 FMAs: c[i][j] += a[i] · b[j]", { size: 12, color: "v" });
          const bars = [["1×1", 2], ["8×1", 1.125], ["4×4", 0.5], ["8×8", 0.25]];
          G.label(430, 330, "smem loads per FMA", { size: 11, color: "ink" });
          bars.forEach(([n, v], i) => { G.rect(430, 342 + i * 18, v * 90, 12, { fill: i === 2 ? "ok" : "k", rx: 2 }); G.label(390, 352 + i * 18, n, { size: 10 }); G.label(430 + v * 90 + 6, 352 + i * 18, String(v), { size: 10 }); });
          G.text(24, 372, "registers hold the accumulators", { size: 13, color: "ok" });
          G.text(24, 398, "16 accumulators + 8 operands per thread", { size: 12, color: "ink" });
          G.from(acc, { opacity: 0, stagger: 0.03, duration: 0.2 });
          G.caption("each loaded value is used TM or TN times from registers");
        } },

      { rail: "feeding the tile", title: "Feeding the tile: k-major A, vector loads, edges, alpha and beta",
        body: `<p>The inner loop needs the 4 A-values of a thread's rows for one k. If the A tile is stored row-by-row in shared memory (<code>As[row][k]</code>), those 4 values are 16 floats apart. Store it <b>k-major</b> (transposed): <code>As[k][row]</code>, so the 4 values are adjacent and can be read as one <code>float4</code>. The reference declares <code>As[GBK][GBM + 4]</code>; the extra 4 pads rows to avoid bank conflicts and keeps 16-byte alignment (L2).</p>
<p>Global loads should be vectorized too: a <code>float4</code> load moves 16 bytes per lane (L4), and B's rows (contiguous in N) are the natural candidate.</p>
<p><b>Any shape.</b> LeetGPU does not promise M, N, K are multiples of 64. The tile loop handles the ragged edge with <b>predication</b>: a load that would read outside the matrix stores 0 instead (a zero contributes nothing to a dot product), and the final store is guarded by <code>if (m &lt; M &amp;&amp; n &lt; N)</code>. No branches inside the inner loop.</p>
<p><b>The epilogue</b> is what happens after the K loop, while the result is still in registers. GEMM is specified as <code>C = α·A·B + β·C</code>:</p>
<div class="eq">c = C + m * N + n;
*c = alpha * acc + (beta != 0 ? beta * *c : 0);</div>
<p>The <code>beta != 0</code> test matters. If β = 0 the old C may hold garbage or NaN, and 0 · NaN = NaN. So with β = 0 the kernel must not read C at all (the test checks it). This tiny epilogue is also where the fusions of step 9 live.</p>`,
        scene(G) {
          G.text(24, 30, "A tile in shared memory: 4 rows × 4 k", { size: 13 });
          G.label(24, 52, "row-major As[row][k]: a thread's 4 rows are 4 apart", { size: 11, color: "hot" });
          const r = []; for (let i = 0; i < 16; i++) r.push(G.box(24 + i * 34, 60, 30, 28, String(i), { fill: [0, 4, 8, 12].includes(i) ? "hot" : "line", size: 11, boxOpacity: [0, 4, 8, 12].includes(i) ? 1 : 0.5 }));
          G.label(24, 108, "k-major As[k][row]: for k = 0 the 4 rows are adjacent → one float4", { size: 11, color: "ok" });
          for (let i = 0; i < 16; i++) G.box(24 + i * 34, 116, 30, 28, String((i % 4) * 4 + Math.floor(i / 4)), { fill: i < 4 ? "ok" : "line", size: 11, boxOpacity: i < 4 ? 1 : 0.5 });
          G.text(24, 188, "ragged edge: matrix 10 × 10, tiles 4 × 4", { size: 13 });
          for (let rr = 0; rr < 12; rr++) for (let c = 0; c < 12; c++) G.rect(24 + c * 20, 200 + rr * 14, 18, 12, { fill: rr < 10 && c < 10 ? "k" : "hot", rx: 2, opacity: rr < 10 && c < 10 ? 0.7 : 0.9 });
          G.label(280, 214, "cyan: real elements", { size: 11, color: "k" }); G.label(280, 232, "red: zero-filled loads,", { size: 11, color: "hot" });
          G.label(280, 248, "stores are guarded", { size: 11, color: "hot" });
          G.label(280, 280, "no branches in the inner loop", { size: 11, color: "ink" });
          G.text(24, 392, "C = alpha·acc + (beta ≠ 0 ? beta·C : 0)", { size: 14, color: "v" });
          G.label(24, 414, "β = 0 must not read C: 0 · NaN = NaN", { size: 12, color: "hot" });
          G.caption("shape the data for the inner loop, pad the edge, finish in registers");
        } },

      { rail: "double buffering", title: "Hide the load latency: double buffering and warp tiles",
        body: `<p>The tile loop so far is: <i>load tile → barrier → compute → barrier → next tile</i>. While the block computes, no loads are in flight; while it loads, no FMAs run. The GPU hides some of this by switching to other blocks on the SM, but with large tiles only one or two blocks fit.</p>
<p><b>Double buffering</b> keeps two copies of the shared tiles. While threads compute on buffer 0 they issue the loads for the next k-step into buffer 1; then the roles swap. Compute and memory overlap. On sm_80 and later, <code>cp.async</code> copies global to shared memory <i>without passing through registers</i> and without stalling the issuing thread, which makes the overlap cheap.</p>
<div class="eq">stage = (BM + BN) * BK * 4 B
128 x 128, BK = 8: 256 * 8 * 4 = 8192 B (8 KiB)
double buffered: 2 * 8 KiB = 16 KiB per block

n tiles, load = compute = 1 unit:
  single buffer: n * (1 + 1) = 2n
  double buffer: 1 + n * 1   = n + 1
  n = 4: 8 versus 5</div>
<p>At best the loop runs at the larger of the two costs instead of their sum. This is rung 7 of P5.6, together with <b>warp tiling</b>: the block tile is split into warp tiles (for example 64 × 32 outputs per warp), and each warp reads its own slice of shared memory. It shortens the path between the shared tile and the registers, and keeps the 32 lanes of a warp reading consecutive banks. Autotuning (searching BM, BN, BK, TM, TN for each GPU and shape) usually finishes the job, because the best tile depends on registers, shared memory and SM count.</p>
<p>The exit check of this lesson is a ratio, not a rule: SGEMM at N = 4096 reaches ≥ 70% of cuBLAS SGEMM, as the median TFLOP/s of repeated runs on the same GPU. The harness measures it on the P5.6 ladder (<code>p5.6_01-gemm-rungs --bench</code>).</p>`,
        check: { q: "Each tile load takes 1 time unit and each tile's compute takes 1 unit. For 4 tiles, how long do a single-buffered and a double-buffered loop take (ideal overlap)?",
          options: ["8 and 5", "8 and 4", "4 and 4"], answer: 0,
          why: "Single buffer: load and compute alternate, 4 × (1 + 1) = 8. Double buffer: the first load cannot be hidden (1), then each of 4 compute steps overlaps the next load, 4 × 1 = 4, so 1 + 4 = 5. The gain is at most 2× and only when load and compute times are close." },
        scene(G) {
          G.text(24, 32, "4 tiles, load = compute = 1 unit", { size: 13 });
          const u = 62, x0 = 70;
          G.label(24, 66, "single buffer: 8 units", { size: 12, color: "hot" });
          for (let t = 0; t < 4; t++) { G.rect(x0 + t * 2 * u, 76, u - 3, 30, { fill: "k", rx: 4 }); G.label(x0 + t * 2 * u + 8, 96, `load ${t}`, { size: 10, color: "bg" }); G.rect(x0 + (t * 2 + 1) * u, 76, u - 3, 30, { fill: "v", rx: 4 }); G.label(x0 + (t * 2 + 1) * u + 6, 96, `FMA ${t}`, { size: 10, color: "bg" }); }
          G.label(24, 166, "double buffer: 5 units", { size: 12, color: "ok" });
          G.rect(x0, 176, u - 3, 30, { fill: "k", rx: 4 }); G.label(x0 + 8, 196, "load 0", { size: 10, color: "bg" });
          for (let t = 0; t < 4; t++) {
            G.rect(x0 + (t + 1) * u, 176, u - 3, 30, { fill: "v", rx: 4 }); G.label(x0 + (t + 1) * u + 6, 196, `FMA ${t}`, { size: 10, color: "bg" });
            if (t < 3) { G.rect(x0 + (t + 1) * u, 212, u - 3, 22, { fill: "k", rx: 4, opacity: 0.8 }); G.label(x0 + (t + 1) * u + 6, 227, `load ${t + 1}`, { size: 10, color: "bg" }); }
          }
          G.label(x0, 256, "loads for tile t+1 run while the FMAs of tile t compute", { size: 11, color: "ink" });
          G.line(x0 + 8 * u - 3, 60, x0 + 8 * u - 3, 240, { color: "hot", w: 1, dash: "4 3" }); G.label(x0 + 8 * u - 70, 56, "8 units", { size: 10, color: "hot" });
          G.line(x0 + 5 * u - 3, 160, x0 + 5 * u - 3, 240, { color: "ok", w: 1, dash: "4 3" });
          G.text(24, 316, "buffers: 2 × (BM + BN) · BK · 4 B", { size: 13, color: "v" });
          G.text(24, 342, "128 × 128, BK = 8: 16 KiB per block", { size: 13, color: "ink" });
          G.text(24, 372, "cp.async (sm_80+): global → shared without registers", { size: 13, color: "k" });
          G.text(24, 398, "warp tiles shorten the shared → register path", { size: 13, color: "ink" });
          G.caption("overlap memory with arithmetic: the loop costs max, not sum");
        } },

      { rail: "tensor cores", title: "Tensor cores: a whole small matrix multiply per instruction",
        body: `<p>An ordinary FMA instruction does 32 multiply-adds per warp (one per lane): 64 FLOPs. NVIDIA's <b>tensor cores</b> are units that multiply two small matrices and add a third in one warp-wide operation: <code>D = A · B + C</code> on a tile. With the WMMA interface a 16 × 16 × 16 tile is 16 · 16 · 16 = 4,096 multiply-adds; the underlying <code>mma.sync</code> instruction on current GPUs does 16 × 8 × 16 = 2,048 multiply-adds (4,096 FLOPs) per warp, 64 times the work of an FMA instruction.</p>
<p>The inputs are low precision (fp16 or bf16, also int8) and the accumulator is fp32 (or int32). The datasheet peak is therefore much higher than fp32 CUDA-core peak. Example values for a T4 (check the datasheet): 65 TFLOP/s fp16 tensor against 8.1 TFLOP/s fp32, about 8 times. A higher roof moves the ridge to the right, so a kernel needs <i>more</i> reuse to stay compute-bound, which is why the memory techniques above matter even more.</p>
<p>How a tensor-core kernel is built (P5.7, <code>${LV}/solutions/57-fp16-batched-matmul/kernel.cu</code>):</p>
<ol><li>Stage fp16 A and B tiles in shared memory, zero-filled at the edges, because a fragment must be full.</li>
<li>Each warp loads <b>fragments</b> from shared memory (WMMA <code>load_matrix_sync</code>, or <code>ldmatrix</code> for the lower-level <code>mma.sync</code>) and issues <code>mma</code> into fp32 accumulator fragments that stay in registers over the whole K loop.</li>
<li>Epilogue: fragments cannot be stored partially, so write accumulators to shared memory, then do a guarded copy to global, converting to fp16.</li></ol>
<p><b>Precision.</b> An fp16 × fp16 product is exact in fp32 (L4), and the sum is in fp32, so the main error is the single fp16 rounding of the output: relative 2<sup>−11</sup> ≈ 4.9 × 10<sup>−4</sup>. The reference test uses a tolerance of 2 × 10<sup>−3</sup>. The syllabus marks the tensor-core rungs as needing sm_80 or newer (A10G, L4) for <code>cp.async</code> and the faster paths; the exit check is HGEMM ≥ 50% of cuBLAS HGEMM at N = 4096 (<code>TODO(run-on: g6.xlarge)</code> until run).</p>`,
        check: { q: "A tensor-core kernel multiplies fp16 inputs and accumulates in fp32. Why is the dominant error at the end just the final rounding of the output to fp16?",
          options: ["fp16 products are exact in fp32 and sums are carried in fp32, so only the output conversion rounds", "tensor cores skip rounding entirely", "fp16 sums are exact up to 65,504"], answer: 0,
          why: "An 11-bit by 11-bit significand product needs at most 22 bits, which fits in fp32's 24, so every product is exact; accumulating in fp32 keeps the sum accurate; the only large rounding left is converting the result to fp16, a relative error of about 2^-11. fp16 sums do not stay exact (they stall near 2,048, L4)." },
        scene(G) {
          G.text(24, 30, "one warp-wide tensor-core operation", { size: 13 });
          const g = (x, y, n, c, lab) => { G.rect(x, y, n * 10, n * 10, { fill: c, rx: 3, opacity: 0.85 }); G.label(x + 4, y + n * 10 + 14, lab, { size: 11, color: c }); };
          g(24, 60, 10, "k", "A 16×16 fp16"); G.text(134, 120, "×", { size: 22, color: "ink" });
          g(160, 60, 10, "q", "B 16×16 fp16"); G.text(270, 120, "+", { size: 22, color: "ink" });
          g(296, 60, 10, "w", "C fp32"); G.text(406, 120, "=", { size: 22, color: "ink" });
          g(432, 60, 10, "v", "D fp32");
          G.text(24, 232, "16 · 16 · 16 = 4,096 multiply-adds", { size: 14, color: "ok" });
          G.text(24, 258, "regular FMA instruction: 32 per warp", { size: 13, color: "ink" });
          G.text(24, 284, "mma.sync m16n8k16: 2,048 MACs = 64× an FMA", { size: 13, color: "ink" });
          G.label(24, 322, "peak FLOP/s, example T4 datasheet values (verify):", { size: 11, color: "ink" });
          G.rect(24, 334, 8.1 * 5, 18, { fill: "k", rx: 3 }); G.label(24 + 8.1 * 5 + 8, 348, "fp32 CUDA cores: 8.1 TFLOP/s", { size: 11 });
          G.rect(24, 360, 65 * 5, 18, { fill: "v", rx: 3 }); G.label(24 + 65 * 5 + 8, 374, "fp16 tensor: 65", { size: 11 });
          G.label(24, 404, "flow: smem tiles → fragments → mma → fp32 accumulators → smem → global", { size: 11, color: "ink" });
          G.caption("low-precision inputs, fp32 accumulation, much higher peak");
        } },

      { rail: "batched", title: "Batched GEMM: one launch, a third grid dimension",
        body: `<p>Attention does many independent small matrix products: one <code>Q·Kᵀ</code> and one <code>P·V</code> per head and per sequence. A <b>batched GEMM</b> runs a whole batch of same-shaped products. The wrong way is a host loop with one launch per matrix. The right way is to put the batch index in the grid's third dimension, <code>gridDim.z</code>, and offset the pointers at the top of the kernel:</p>
<div class="eq">const int b = blockIdx.z;
A += b * (size_t)M * K;
B += b * (size_t)K * N;
C += b * (size_t)M * N;
// ... the same tile core as the single GEMM</div>
<p>Example from Llama-3-8B: 32 query heads with head dimension 128. For a 2,048-token sequence, <code>Q·Kᵀ</code> is, per head, [2048 × 128] times [128 × 2048]:</p>
<div class="eq">per head: 2 * 2048 * 2048 * 128
        = 1,073,741,824 FLOP
32 heads: 34,359,738,368 FLOP (34.4 GFLOP)
64 x 64 tiles: 32 * 32 = 1,024 blocks per head
grid = (32, 32, 32) = 32,768 blocks</div>
<p>One launch fills every SM, even if each individual GEMM is small. For tiny matrices (a 64 × 64 matrix per head) a 64 × 64 tile has no reuse to exploit, so smaller tiles, or one warp per matrix, can win. Measure. cuBLAS exposes the same layout as <code>gemmStridedBatched</code>, where the batch stride is a parameter. The next generalization is <b>grouped GEMM</b>, where each problem has its own M (the experts of a mixture-of-experts layer receive different numbers of tokens). The FP16 version (#57) is this plus the tensor-core core from step 6.</p>`,
        scene(G) {
          G.text(24, 30, "grid.z selects the matrix in the batch", { size: 13 });
          [0, 1, 2].forEach((z) => {
            const ox = 24 + z * 36, oy = 50 + z * 22;
            G.rect(ox, oy, 150, 84, { fill: ["k", "v", "q"][z], rx: 6, opacity: 0.25 + 0.15 * z, stroke: ["k", "v", "q"][z] });
            for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) G.rect(ox + 6 + j * 47, oy + 6 + i * 25, 42, 20, { stroke: ["k", "v", "q"][z], rx: 3, opacity: 0.8 });
            if (z === 2) G.label(ox + 8, oy + 102, "matrix b = blockIdx.z", { size: 11, color: "q" });
          });
          G.label(280, 76, "gridDim = (N/64, M/64, batch)", { size: 12, color: "ink" });
          G.label(280, 98, "each cell = one 64×64 block", { size: 11 });
          G.label(280, 128, "A += b · M·K,  B += b · K·N,  C += b · M·N", { size: 11, color: "k" });
          G.text(24, 250, "Llama-3-8B, 2048 tokens, Q·Kᵀ", { size: 13, color: "v" });
          G.label(24, 274, "32 heads · [2048×128]·[128×2048]", { size: 12 });
          G.label(24, 294, "34.4 GFLOP in one launch", { size: 12, color: "ok" });
          G.label(24, 314, "grid (32, 32, 32) = 32,768 blocks of 64×64", { size: 12, color: "ok" });
          G.text(24, 372, "no host loop · every SM busy · pointer offset per z", { size: 13, color: "ink" });
          G.label(24, 398, "grouped GEMM: each problem has its own M (MoE experts)", { size: 12 });
          G.caption("batch index = blockIdx.z");
        } },

      { rail: "int8 and int4", title: "Quantized GEMM: fewer bytes per weight",
        body: `<p>At decode time a model processes only 1 to a few dozen tokens per step, so each GEMM is skinny: M is tiny. With bf16 weights its intensity is about M FLOP per weight byte (each 2-byte weight does 2·M FLOPs), far below the ridge, so the GEMM is <b>memory-bound on the weights</b> and the time is roughly weight bytes ÷ bandwidth. Fewer bytes per weight is a direct speed-up. For Llama-3-8B (≈ 8.0 billion parameters):</p>
<div class="eq">at 300 GB/s (example value):
bf16: 8.0e9 * 2 B   = 16.0 GB → 53 ms per token
int8: 8.0e9 * 1 B   =  8.0 GB → 26.7 ms
int4: 8.0e9 * 0.5 B =  4.0 GB → 13.3 ms (+ scales)</div>
<p>(Lower bounds for reading all weights once per decoded token; bandwidth is an example value. Real engines also read the KV cache.)</p>
<p><b>INT8 GEMM</b> (#32). Inputs are int8, products accumulate in int32 (exact), and one fused requantization at the end returns to int8: <code>q = clamp(round(acc · sA · sB / sC), −128, 127)</code> where sA, sB, sC are the scales. The hardware instruction <code>__dp4a(a, b, c)</code> (sm_61 and later) multiplies four packed int8 pairs and adds them to an int32 in one instruction. Worked example:</p>
<div class="eq">a = [10, -20, 30, 40]   b = [1, 2, -3, 4]
acc = 10 - 40 - 90 + 160 = 40
sA = 0.1, sB = 0.05, sC = 0.2
q = round(40 * 0.1 * 0.05 / 0.2) = round(1.0) = 1</div>
<p>dp4a needs both operands <i>contiguous in k</i>. A is (row-major), but B (K × N row-major) is not, so the kernel packs B's columns while loading the tile into shared memory (a transpose on load). Tensor-core int8 (<code>mma.sync … .s8.s8.s32</code>, sm_80+) is the next 2 to 4 times faster (P5.7).</p>
<p><b>INT4 weight-only</b> (#81, stretch). Two weights share one byte. A kernel loads packed bytes, unpacks in registers, scales by a per-group scale, and FMAs. It never writes the dequantized weight matrix. For a symmetric format: <code>low = (b &amp; 0xF) − 8</code>, <code>high = (b &gt;&gt; 4) − 8</code>. For b = 0xA3: low = 3 − 8 = −5, high = 10 − 8 = 2. This is W4A16: 4-bit weights, 16-bit activations.</p>`,
        scene(G) {
          G.text(24, 30, "Llama-3-8B weights per token (300 GB/s, example)", { size: 13 });
          [["bf16", 16, "w", "53 ms"], ["int8", 8, "k", "26.7 ms"], ["int4", 4, "ok", "13.3 ms"]].forEach(([n, gb, c, t], i) => {
            const y = 56 + i * 50; G.label(24, y + 22, n, { size: 12, color: "ink" });
            G.rect(70, y, gb * 28, 34, { fill: c, rx: 5, opacity: 0.9 }); G.label(70 + gb * 28 + 8, y + 22, `${gb} GB · ≥ ${t}`, { size: 11 });
          });
          G.text(24, 232, "int8: dp4a = 4 multiply-adds, int32 accumulate", { size: 13, color: "k" });
          nums4(G, 24, 246, [10, -20, 30, 40], "a", "k"); nums4(G, 300, 246, [1, 2, -3, 4], "b", "q");
          G.label(24, 322, "acc = 40 → q = round(40 · 0.1 · 0.05 / 0.2) = 1", { size: 12, color: "v" });
          G.text(24, 346, "int4: two weights per byte, unpack in registers", { size: 13, color: "ok" });
          G.box(24, 358, 90, 34, "0xA3", { fill: "w", size: 14 });
          G.arrow(118, 375, 160, 366, { color: "ok" }); G.arrow(118, 375, 160, 388, { color: "ok" });
          G.box(164, 352, 150, 24, "low: 3 − 8 = −5", { fill: "ok", size: 11 }); G.box(164, 382, 150, 24, "high: 10 − 8 = 2", { fill: "ok", size: 11 });
          G.label(330, 376, "× group scale → FMA", { size: 11 });
          G.caption("decode is memory-bound: fewer weight bytes is a direct speed-up");
        } },

      { rail: "epilogue fusion", title: "Epilogue fusion: finish the next op while the result is in registers",
        body: `<p>After the K loop the output tile sits in registers. Writing it to DRAM, then launching another kernel that reads it back, is exactly the waste L4 counted. Instead, apply the next operation right there:</p>
<ul><li><b>SwiGLU MLP</b> (#84, practice). A Llama MLP is <code>(silu(x·W_gate) ⊙ (x·W_up)) · W_down</code>. Unfused, the two M × f intermediates (f = 14,336 in Llama-3-8B) go to DRAM and come back. Compute the gate and up tiles for the <i>same</i> output block in one kernel and apply <code>silu(g) · u</code> in the epilogue; only the result <code>h</code> is written.</li>
<li><b>LoRA</b> (#85). A low-rank adapter adds <code>(α/r) · x · A · B</code> to the output. Associativity decides the cost: multiply by A first.</li>
<li><b>QKV + RoPE + KV cache</b> (#115, stretch). The epilogue applies RoPE (L4) to the q and k columns and scatters k and v straight into the KV cache.</li></ul>
<div class="eq">SwiGLU, 8192 tokens, f = 14336, fp16:
8192 * 14336 * 2 B = 224 MiB per intermediate
unfused: write g, u (448 MiB)
         + read g, u (448) + write h (224)
         = 1,120 MiB
fused:   write h only = 224 MiB

LoRA, M = 1, K = N = 4096, r = 16 (MACs):
(x*A)*B: 4096*16 + 16*4096 = 131,072
x*(A*B): 4096*16*4096 + 4096*4096
       = 285,212,672  (2,176x more)</div>
<p>The MLP holds most of a dense model's parameters: per layer, W<sub>gate</sub>, W<sub>up</sub> and W<sub>down</sub> total 3 · 4096 · 14,336 = 176 million, so about 5.6 billion of Llama-3-8B's 8.0 billion across 32 layers. At decode this GEMM is memory-bound on those weights, so fusing the intermediates matters less than quantizing the weights; at prefill (large M) fusion matters a lot. The down projection can't join, because it needs the whole <code>h</code> row, not a tile. A tidy trick from serving engines: concatenate <code>[W_gate | W_up]</code> into one d × 2f matrix, so one GEMM shares the x tile loads, and the epilogue pairs column n with column n + f.</p>`,
        check: { q: "LoRA adds (x·A)·B to a layer's output, with x of size 1 × 4096, A of 4096 × 16, B of 16 × 4096. Why must you compute (x·A) first, not (A·B) first?",
          options: ["It does 131,072 multiply-adds instead of building a 4096 × 4096 matrix (about 268 million more multiply-adds)", "The result is different", "A·B is not defined"], answer: 0,
          why: "Matrix multiplication is associative, so the results agree (up to rounding), but the cost does not. (x·A)·B is a 1 × 16 vector times a 16 × 4096 matrix: 131,072 multiply-adds in total. Forming A·B first costs 4096 · 16 · 4096 = 268 million multiply-adds and materializes a full-size matrix. Always keep r as the small dimension in the middle." },
        scene(G) {
          G.text(24, 30, "GEMM tile → registers → epilogue ops", { size: 13 });
          G.box(24, 54, 150, 44, "accumulators (regs)", { fill: "v", size: 12 });
          const ops = [["α·acc + β·C", "k"], ["+ bias", "q"], ["silu(g)·u", "ok"], ["+ LoRA r FMAs", "pink"], ["RoPE + KV scatter", "blue"]];
          ops.forEach(([t, c], i) => { G.arrow(176 + 0, 76, 206, 70 + i * 10, { color: "muted", w: 1 }); G.box(206, 50 + i * 38, 190, 32, t, { fill: c, size: 12 }); });
          G.label(410, 70, "one store at the end", { size: 11, color: "ok" }); G.arrow(398, 80, 430, 130, { color: "ok" });
          G.box(430, 130, 180, 36, "global memory", { fill: "w", size: 12 });
          G.text(24, 270, "SwiGLU MLP, 8192 tokens, fp16 (intermediates):", { size: 13, color: "ink" });
          G.rect(24, 286, 1120 / 4, 24, { fill: "hot", rx: 4 }); G.label(24 + 280 + 8, 303, "unfused: 1,120 MiB", { size: 11, color: "hot" });
          G.rect(24, 318, 224 / 4, 24, { fill: "ok", rx: 4 }); G.label(24 + 56 + 8, 335, "fused: 224 MiB (write h only)", { size: 11, color: "ok" });
          G.text(24, 380, "LoRA: (x·A)·B = 131,072 MACs vs 285 M", { size: 13, color: "v" });
          G.label(24, 404, "never materialize A·B", { size: 12 });
          G.caption("the intermediate never reaches DRAM");
        } },

      { rail: "measuring", title: "Measuring a GEMM: TFLOP/s, waves and the exit checks",
        body: `<p>Report GEMM speed as <b>TFLOP/s</b> = 2·M·N·K ÷ time, and as a percentage of cuBLAS on the same GPU. Example: a 4096³ product that takes 20 ms runs at 137.4e9 / 0.020 = 6.87 TFLOP/s (a made-up time to show the arithmetic, not a measurement). Always use the median of many runs after warm-up, as the harness does (<code>s2s::time_gpu</code>).</p>
<p>Beyond the kernel, the <b>shape</b> decides utilization through <b>wave quantization</b>. The GPU runs a fixed number of blocks at once: SMs × blocks per SM. If the grid has more blocks than that, they execute in waves, and the last wave may be partly empty. Example numbers (assumed, check your kernel's occupancy): 40 SMs × 2 resident blocks = 80 slots.</p>
<div class="eq">4096 x 4096, 128 x 128 tiles: 32 * 32 = 1024 blocks
1024 / 80 = 12.8 waves → 13 launches of the slots
efficiency ≈ 12.8 / 13 = 98.5%

1024 x 1024, 128 x 128 tiles: 8 * 8 = 64 blocks
64 / 80 = 0.8 wave → 20% of the slots idle</div>
<p>Small or skinny problems (the decode case) cannot fill the GPU at all, so measure the shapes you actually serve, not only 4096³.</p>
<p>The two exit checks of L5, from the syllabus:</p>
<ul><li>SGEMM at N = 4096 reaches ≥ 70% of cuBLAS SGEMM (the P5.6 ladder).</li>
<li>Tensor-core HGEMM at N = 4096 reaches ≥ 50% of cuBLAS HGEMM (P5.7).</li></ul>
<p>Both are reported as median TFLOP/s on the same GPU, and both are <code>TODO(run-on: g6.xlarge)</code> until someone runs them. The safe habit: compare against cuBLAS first, then against the theoretical peak from the spec sheet (an upper bound that real kernels rarely approach), and say which you did.</p>`,
        scene(G) {
          G.text(24, 30, "80 resident-block slots (example), 1024 blocks", { size: 13 });
          const cw = 12, gap = 2;
          for (let w = 0; w < 13; w++) {
            const full = w < 12 ? 80 : 64, y = 52 + w * 17;
            G.label(4, y + 11, w + 1 < 10 ? " " + (w + 1) : String(w + 1), { size: 9 });
            for (let i = 0; i < 40; i++) {
              const idx = i * 2; // draw 40 cells, each = 2 slots
              const on = idx < full;
              G.rect(24 + i * (cw + gap), y, cw, 13, { fill: on ? (w === 12 ? "v" : "k") : "line", rx: 2, opacity: on ? 0.9 : 0.4 });
            }
          }
          G.label(24 + 40 * (cw + gap) - 160, 52 + 13 * 17 + 14, "each cell = 2 slots", { size: 10 });
          G.label(24, 52 + 13 * 17 + 14, "wave 13 is 64 / 80 = 80% full", { size: 11, color: "v" });
          G.text(24, 364, "efficiency ≈ 12.8 / 13 = 98.5%", { size: 14, color: "ok" });
          G.text(24, 392, "1024² with 128² tiles: 64 blocks → 20% idle", { size: 13, color: "hot" });
          G.caption("shape and tile size decide how full the GPU is");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A GEMM is 2·M·N·K FLOPs. Compare its FLOP-per-byte intensity with the GPU's ridge point to know whether memory or arithmetic limits it.</li>
<li>Tiling raises intensity in proportion to the tile side (8 for 32 × 32, 16 for 64 × 64, 32 for 128 × 128 in fp32). Register blocking does the same one level down: a 4 × 4 outer product needs 0.5 shared loads per FMA.</li>
<li>Feed the tile well: k-major A, vector loads, zero-filled loads and guarded stores for any shape, and an epilogue that never reads C when β = 0.</li>
<li>Double buffering (cp.async) overlaps loads with FMAs; warp tiles shorten the shared-to-register path.</li>
<li>Tensor cores do 16 × 16 × 16 per warp-wide op with fp32 accumulation: much higher peak, so reuse matters more.</li>
<li>Batch goes in gridDim.z. At decode, fewer weight bytes (int8, int4) is the speed-up. Epilogues fuse the next op while the tile is in registers.</li>
<li>Report TFLOP/s as a percentage of cuBLAS, and check wave quantization for your shapes.</li></ul>
<p>Next, L6 puts GEMM, softmax and reductions together into attention, and shows the same tiling ideas in Triton.</p>`,
    sim: {
      title: "GEMM roofline and tile calculator",
      intro: "Choose the GEMM size, the block tile, the element size and the GPU's peak FLOP/s and bandwidth (defaults are example T4-class values: check your GPU's spec sheet). The chart is a roofline: the point is your tile's arithmetic intensity, the line is what the GPU can deliver at that intensity. The panels give the time lower bound, shared memory per block and the wave count.",
      height: 320,
      controls: [
        { id: "n", label: "M = N = K", type: "select", value: 4096, options: [[512, "512"], [1024, "1,024"], [2048, "2,048"], [4096, "4,096"], [8192, "8,192"]] },
        { id: "bm", label: "block tile BM = BN", type: "select", value: 64, options: [[16, "16"], [32, "32"], [64, "64"], [128, "128"], [256, "256"]] },
        { id: "bk", label: "BK", type: "select", value: 16, options: [[8, "8"], [16, "16"], [32, "32"]] },
        { id: "bytes", label: "element size", type: "select", value: 4, options: [[2, "2 B (fp16 / bf16)"], [4, "4 B (fp32)"]] },
        { id: "peak", label: "peak TFLOP/s (example)", min: 1, max: 400, step: 0.1, value: 8.1 },
        { id: "bw", label: "bandwidth GB/s (example)", min: 50, max: 3000, step: 10, value: 320 },
        { id: "slots", label: "resident blocks (SMs × blocks per SM)", min: 1, max: 600, step: 1, value: 80 },
      ],
      draw(G, v) {
        const n = v.n, bm = v.bm, B = v.bytes, peak = v.peak, bw = v.bw;
        const I = (2 * bm * bm) / ((bm + bm) * B), idealI = (2 * n * n * n) / (3 * n * n * B);
        const ridge = (peak * 1e12) / (bw * 1e9);
        const att = Math.min(peak, (I * bw) / 1000), flops = 2 * n * n * n, ms = (flops / (att * 1e12)) * 1e3;
        const x0 = 50, x1 = 610, yb = 270, yt = 20, lo = 0.1, hi = 2000, tlo = peak / 3000, thi = peak * 1.5;
        const X = (i) => lg(i, lo, hi, x0, x1), Y = (t) => lg(Math.max(t, tlo), tlo, thi, yb, yt);
        G.line(x0, yb, x1, yb, { color: "muted", w: 1 }); G.line(x0, yb, x0, yt - 6, { color: "muted", w: 1 });
        let d = ""; [lo, ridge, hi].forEach((i, k) => { const t = Math.min(peak, (i * bw) / 1000); d += (k ? " L " : "M ") + X(i).toFixed(1) + " " + Y(t).toFixed(1); });
        G.path(d, { color: "ink", w: 2.5 });
        G.line(X(ridge), yb, X(ridge), Y(peak), { color: "line", w: 1, dash: "4 3" });
        G.circle(X(0.25 * 4 / B), Y(Math.min(peak, (0.25 * 4 / B * bw) / 1000)), 5, { fill: "hot" }); G.label(X(0.25 * 4 / B) + 8, Y(Math.min(peak, (0.25 * 4 / B * bw) / 1000)) + 20, "naive", { size: 11, color: "hot" });
        G.circle(X(I), Y(att), 7, { fill: "v" }); G.label(X(I) - 20, Y(att) - 14, `your tile ${I.toFixed(0)}`, { size: 11, color: "v" });
        G.circle(X(idealI), Y(Math.min(peak, (idealI * bw) / 1000)), 5, { fill: "ok" }); G.label(X(idealI) - 70, Y(Math.min(peak, (idealI * bw) / 1000)) + 18, `ideal ${idealI.toFixed(0)}`, { size: 11, color: "ok" });
        [0.1, 1, 10, 100, 1000].forEach((i) => G.label(X(i) - 8, yb + 14, String(i), { size: 10 }));
        G.label(x0 + 200, yb + 30, "FLOP per byte (log)", { size: 11 }); G.label(x0 + 4, yt - 8, "TFLOP/s (log)", { size: 10 });
        G.label(X(ridge) + 4, yb - 6, `ridge ${ridge.toFixed(0)}`, { size: 10 });
        const smem = (2 * bm * v.bk * B), blocks = Math.ceil(n / bm) ** 2, waves = blocks / v.slots, wq = waves / Math.ceil(waves);
        return [
          { title: "Your tile", rows: [["intensity", `${I.toFixed(1)} FLOP/B`], ["ridge point", `${ridge.toFixed(1)} FLOP/B`], ["bound by", I < ridge ? "memory" : "compute"], ["attainable", `${att.toFixed(2)} TFLOP/s`]],
            gauge: [[att / peak, I < ridge ? "hot" : "ok"], [1 - att / peak, "line"]], gaugeText: "fraction of peak at this intensity",
            chip: [I >= ridge, I >= ridge ? "compute-bound" : `${(ridge / I).toFixed(1)}× below the ridge`] },
          { title: "Time and size", rows: [["FLOPs", `${(flops / 1e9).toFixed(1)} GFLOP`], ["time ≥ FLOPs / attainable", `${ms.toFixed(2)} ms`], ["min traffic (A, B, C once)", F.bytes(3 * n * n * B)], ["ideal intensity", `${idealI.toFixed(0)} FLOP/B`]] },
          { title: "Resources", rows: [["shared memory / block", F.bytes(smem)], ["double buffered", F.bytes(2 * smem)], ["blocks in grid", F.num(blocks)], ["waves", waves.toFixed(2)], ["last-wave filling", `${(wq * 100).toFixed(1)}% of the slots`]],
            html: `<p class="note">Model: intensity = 2·BM·BN / ((BM + BN)·bytes), global traffic only, large K. It ignores the C write and caches. Peak, bandwidth and resident blocks are example inputs: use your GPU's spec sheet and the occupancy API. An upper bound, not a prediction.</p>` },
        ];
      },
      note: "Roofline: attainable = min(peak, intensity × bandwidth). Wave efficiency = waves / ceil(waves).",
    },
    practice: {
      intro: `<p>Solve each problem on <a href="https://leetgpu.com/challenges">LeetGPU</a> first (T1, in the browser). We don't reproduce the statements; read them there. Then run the local reference and test (T2, needs an NVIDIA GPU and CUDA ≥ 12.4; the cuBLAS comparisons link CUDA::cublas). Build once from the repository root:</p>
<p><code>cmake -S laneB-cuda -B build/laneB -DCMAKE_BUILD_TYPE=Release &amp;&amp; cmake --build build/laneB -j</code></p>
<p>Problem #22 <i>is</i> the P5.6 ladder: climb rungs 1 to 7 with <code>platform/kernels/include/d4/gemm.cuh</code> as the reference. The tensor-core rungs need sm_80 or newer (A10G, L4). <b>Exit checks</b> (median TFLOP/s on the same GPU, <code>TODO(run-on: g6.xlarge)</code> until run): SGEMM ≥ 70% of cuBLAS SGEMM at N = 4096; tensor-core HGEMM ≥ 50% of cuBLAS HGEMM at N = 4096. <b>Cloud GPU (T2):</b> use <code>infra/aws/single-node</code> (g6.xlarge, an L4, is its default). It bills per hour while running plus disk while the disk exists (look up current prices on the EC2 pricing page), has an idle-stop alarm and an AWS Budget, no inbound ports (SSM only), and <code>make down</code> deletes everything: tear it down when you finish.</p>`,
      items: [
        P("22-gemm", "General Matrix Multiplication (GEMM)", "medium", "core · exit check", "64 × 64 × 16 tile, 4 × 4 outputs per thread, predicated loads and stores, and the α/β epilogue. β = 0 must not read C."),
        P("30-batched-matmul", "Batched Matrix Multiplication", "medium", "core", "The #22 tile core with the batch in <code>blockIdx.z</code> and a pointer offset."),
        P("57-fp16-batched-matmul", "FP16 Batched Matrix Multiplication", "medium", "core · exit check", "WMMA 16 × 16 × 16 fragments with fp32 accumulators, shared-memory staging, guarded copy-out."),
        P("32-int8-quantized-matmul", "INT8 Quantized MatMul", "medium", "core", "<code>__dp4a</code> with B packed on load, int32 accumulate, fused requantize."),
        P("85-lora-linear", "LoRA Linear", "medium", "practice", "Multiply by A first; add the low-rank term in the GEMM epilogue."),
        P("84-swiglu-mlp-block", "SwiGLU MLP Block", "medium", "practice", "Gate and up tiles in one kernel with <code>silu(g) · u</code> in the epilogue, then the down projection."),
        P("81-int4-weight-only-matmul", "INT4 Weight-Only Quantized MatMul", "medium", "stretch", "Hints and outline only: unpack nibbles in registers, group scales, one warp per output column."),
        P("115-fused-qkv-rope-kv-cache", "Fused QKV Projection with RoPE and KV Cache Update", "medium", "stretch", "Hints and outline only: a GEMM epilogue that applies RoPE and scatters into the cache."),
      ],
      labs: [
        { label: "Level syllabus, objectives and exit checks", path: `${LV}/syllabus.md` },
        { label: "Problem map with links and priorities", path: `${LV}/leetgpu-map.md` },
        { label: "The SGEMM ladder, rungs 1 to 7, and the cuBLAS baseline", path: "platform/kernels/include/d4/gemm.cuh" },
        { label: "The tensor-core HGEMM reference", path: "platform/kernels/include/d4/hgemm.cuh" },
        { label: "The harness: CUDA_CHECK, time_gpu, report", path: "laneB-cuda/harness/" },
        { label: "Shared-memory tiling animation (Lane A P5.2)", path: "animations/p5-smem-tiling.html" },
        { label: "Roofline animation", path: "animations/roofline.html" },
      ],
    },
  });
  function nums4(G, x, y, vals, nm, c) { vals.forEach((v, i) => G.box(x + i * 60, y, 56, 36, String(v), { fill: c, size: 13, boxOpacity: 0.8 })); G.label(x, y + 52, nm, { size: 11 }); }
})();
