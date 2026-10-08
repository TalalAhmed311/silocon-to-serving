/* P5.6 — The GEMM ladder (rungs 1–7). Every rung raises the reuse of a loaded number; the lesson counts it. */
(function () {
  const F = S2S.fmt;
  // Example GPU values from course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/gpu_specs.yaml (UNVERIFIED there).
  const GPUS = {
    t4: { name: "T4", fp32: 8.1e12, bw: 320e9, sms: 40 },
    l4: { name: "L4", fp32: 30.3e12, bw: 300e9, sms: 58 },
    a100: { name: "A100-80GB", fp32: 19.5e12, bw: 2039e9, sms: 108 },
  };
  // DRAM intensity (FLOP per byte, fp32) of a BM×BN block tile when every tile load comes from DRAM
  const tileAI = (BM, BN) => (BM * BN) / (2 * (BM + BN));
  // a small log-log roofline on the stage: x = FLOP/byte 0.1..1000, y = TFLOP/s 0.01..10
  function roofline(G, x0, y0, w, h, gpu, pts) {
    const lx = (ai) => x0 + ((Math.log10(ai) + 1) / 4) * w;
    const ly = (fl) => y0 + h - ((Math.log10(fl / 1e12) + 2) / 3) * h;
    G.axes(x0, y0, w, h, {});
    [0.1, 1, 10, 100, 1000].forEach((a) => G.label(lx(a), y0 + h + 16, String(a), { anchor: "middle", size: 11 }));
    [0.01, 0.1, 1, 10].forEach((f) => G.label(x0 - 6, ly(f * 1e12) + 4, String(f), { anchor: "end", size: 11 }));
    G.label(x0 + w, y0 + h + 32, "arithmetic intensity, FLOP per DRAM byte (log)", { anchor: "end", size: 11 });
    G.label(x0 - 40, y0 - 10, "TFLOP/s (log)", { size: 11 });
    const ridge = gpu.fp32 / gpu.bw;
    const aMin = 0.1, aMax = 1000;
    const roof = G.path(`M ${lx(aMin)} ${ly(aMin * gpu.bw)} L ${lx(ridge)} ${ly(gpu.fp32)} L ${lx(aMax)} ${ly(gpu.fp32)}`, { color: "ink", w: 2.5 });
    G.line(lx(ridge), ly(gpu.fp32), lx(ridge), y0 + h, { color: "muted", dash: "3 4" });
    G.label(lx(ridge) + 6, y0 + h - 8, `ridge ≈ ${ridge.toFixed(0)}`, { size: 11 });
    const dots = pts.map(([ai, lbl, col, dx, dy, anc]) => {
      const at = Math.min(gpu.fp32, ai * gpu.bw);
      const d = G.circle(lx(ai), ly(at), 6, { fill: col });
      G.label(lx(ai) + dx, ly(at) + dy, lbl, { size: 11, color: col, anchor: anc });
      return d;
    });
    return { roof, dots, lx, ly };
  }
  // draw a matrix as a grid of cells with a highlight function
  const mat = (G, x, y, n, cs, fill) => G.grid(x, y, n, n, cs, cs, (r, c) => fill(r, c) || "line", { gap: 2, rx: 2 });

  S2S.lesson({
    id: "p5-6", n: "P5.6", title: "The GEMM ladder",
    subtitle: "CUDA deep dive · first principles · T2, one NVIDIA GPU (sm_75+)",
    kicker: "Lesson · ≈ 50 min",
    headline: "Seven kernels, one multiplication",
    intro: `<p>Almost all of an LLM's arithmetic is matrix multiplication (GEMM: <b>GE</b>neral <b>M</b>atrix <b>M</b>ultiply). This lesson climbs from the obvious one-thread-per-output kernel to a kernel within reach of cuBLAS. The arithmetic never changes: every rung computes the same 2N³ FLOPs. What changes is how many times each number fetched from memory gets used. Count that, and you can predict each rung's speed before you run it.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "2 exercises"],
    legend: [["q", "row of A"], ["k", "column of B"], ["ok", "output C"], ["v", "shared memory"], ["hot", "wasted / stalled"]],
    prev: "p5-5", next: "p5-7",
    steps: [
      { rail: "the job", title: "GEMM does a lot of maths on a little data",
        body: `<p>C = A · B. With A of size M×K and B of size K×N, every output <code>C[m][n]</code> is a dot product of row <i>m</i> of A with column <i>n</i> of B: K multiplies and K adds. One multiply plus one add is a <b>fused multiply-add (FMA)</b>, counted as 2 FLOPs.</p>
<div class="eq">FLOPs  = 2 × M × N × K
values = M×K + K×N + M×N    (read A, B; write C)

square, N = 4096, fp32 (4 bytes):
  FLOPs  = 2 × 4096³      = 137.4 GFLOP
  bytes  = 3 × 4096² × 4  = 201 MB
  ratio  = 137.4e9 / 201e6 ≈ 683 FLOP per byte</div>
<p>The work grows as N³ but the data only as N². Every number in A is used N times (once per column of B). That reuse is the whole game: if a kernel could load each number once and use it all N times, GEMM would do 683 FLOPs for every byte it reads.</p>`,
        scene(G) {
          const cs = 17, n = 8;
          G.label(250, 34, "B  (K × N)", { color: "k" });
          const b = mat(G, 250, 42, n, cs, (r, c) => (c === 5 ? "k" : null));
          G.label(24, 196, "A  (M × K)", { color: "q" });
          const a = mat(G, 70, 204, n, cs, (r) => (r === 2 ? "q" : null));
          G.label(250, 196, "C  (M × N)", { color: "ok" });
          mat(G, 250, 204, n, cs, (r, c) => (r === 2 && c === 5 ? "ok" : null));
          G.arrow(208, 246, 330, 246, { color: "q", dash: "4 3" });
          G.arrow(343, 182, 343, 236, { color: "k", dash: "4 3" });
          G.text(420, 80, "one output", { size: 14 });
          G.label(420, 100, "= row of A · column of B", { size: 12 });
          G.label(420, 118, "= K FMAs = 2K FLOPs", { size: 12 });
          G.text(420, 220, "N = 4096, fp32", { size: 14 });
          G.label(420, 242, "work   2N³ = 137.4 GFLOP", { color: "ink" });
          G.label(420, 262, "data   3N² × 4 B = 201 MB", { color: "ink" });
          G.text(420, 292, "≈ 683 FLOP / byte", { size: 16, color: "ok" });
          G.label(420, 312, "if each number is loaded once", { size: 12 });
          G.from(a.concat(b), { opacity: 0.2, stagger: 0.004, duration: 0.25 });
          G.caption("work grows as N³, data as N²: each number can be reused N times");
        } },

      { rail: "the target", title: "A good GEMM is compute-bound",
        body: `<p>Recall the <b>roofline</b> from P1.4. A kernel's speed is capped by whichever is slower: doing its FLOPs at the GPU's peak, or moving its bytes at the memory bandwidth. The crossover, peak FLOP/s ÷ bandwidth, is the <b>ridge point</b>: a kernel that does more FLOPs per byte than this is compute-bound.</p>
<div class="eq">T4 (example values from gpu_specs.yaml, UNVERIFIED):
  fp32 peak   8.1 TFLOP/s
  bandwidth   320 GB/s
  ridge       8.1e12 / 320e9 ≈ 25 FLOP per byte

N = 4096, at peak:  137.4e9 / 8.1e12 ≈ 17 ms</div>
<p>683 is far to the right of 25, so a perfect kernel would spend its time computing, about 17 ms. But a kernel only gets that intensity if it actually reuses data. The dots on the stage are the rungs of this lesson, placed by how many FLOPs each does per byte it pulls from DRAM. Each rung moves the dot right until it crosses the ridge.</p>`,
        check: { q: "A kernel does 8 FLOPs per DRAM byte on a GPU whose ridge point is 25. What limits it?",
          options: ["Compute: the FMA units are saturated", "Memory bandwidth: it can reach at most 8 × bandwidth FLOP/s, about a third of peak", "Neither: it will run at peak"], answer: 1,
          why: "Left of the ridge the roof is the slanted line: attainable FLOP/s = intensity × bandwidth = 8 × 320 GB/s ≈ 2.6 TFLOP/s, about 32% of the 8.1 TFLOP/s peak. To go faster it must reuse data more, not compute faster." },
        scene(G) {
          const r = roofline(G, 80, 50, 520, 300, GPUS.t4, [[0.25, "rung 2: 0.25", "hot", 10, 4, "start"], [8, "rung 3: 8", "v", -10, 16, "end"], [16, "rung 4: 16", "q", -10, -8, "end"], [32, "rung 5: 32", "ok", 4, 22, "start"], [683, "ideal: 683", "k", 0, -14, "middle"]]);
          G.from(r.dots, { opacity: 0, scale: 0, transformOrigin: "center", stagger: 0.25, duration: 0.35 });
          G.caption("T4 example values · every rung moves the dot right by reusing data");
        } },

      { rail: "rungs 1–2", title: "Rungs 1 and 2: one thread per output",
        body: `<p>The obvious kernel gives each thread one <code>C[m][n]</code> and loops over k. It loads 2 floats (8 bytes) for every FMA (2 FLOPs): <b>0.25 FLOP per byte</b> if every load went to DRAM.</p>
<p><b>Rung 1</b> maps <code>threadIdx.x</code> to rows. The 32 lanes of a warp then read A at 32 different rows, each K × 4 bytes apart. From P5.2: that is 32 separate 32-byte sectors to deliver 128 useful bytes, an 8× waste, and the C stores scatter the same way.</p>
<p><b>Rung 2</b> swaps the mapping: <code>threadIdx.x</code> walks columns. Now all 32 lanes read the <i>same</i> A element (one broadcast) and 32 neighbouring B elements (one 128-byte line, 4 sectors). The code is identical except for which index is <code>threadIdx.x</code>.</p>
<div class="eq">// rung 1                    // rung 2
m = blockIdx.x*32 + tx;      n = blockIdx.x*32 + tx;
n = blockIdx.y*32 + ty;      m = blockIdx.y*32 + ty;
for k: acc += A[m*K+k] * B[k*N+n];</div>
<p>Coalescing fixes the waste, not the reuse. If every load came from DRAM, N = 4096 would need 2 × 4096³ × 4 B ≈ 550 GB of traffic: 1.7 s at 320 GB/s, about 100× the 17 ms compute time. Caches catch some of it, but the kernel is still far left of the ridge.</p>`,
        scene(G) {
          G.text(24, 36, "rung 1: lanes → rows of A", { size: 14, color: "hot" });
          for (let l = 0; l < 8; l++) G.box(24 + l * 34, 50, 28, 26, String(l), { stroke: "muted", size: 11, color: "muted", rx: 4 });
          G.label(310, 68, "8 of 32 lanes", { size: 11 });
          const sec1 = [];
          for (let r = 0; r < 8; r++) {
            G.rect(24, 100 + r * 14, 280, 10, { fill: "line", rx: 2 });
            sec1.push(G.rect(24 + 40, 100 + r * 14, 32, 10, { fill: "hot", rx: 2 }));
            G.line(38 + r * 34, 76, 80, 100 + r * 14 + 5, { color: "hot", w: 1, opacity: 0.6 });
          }
          G.label(320, 130, "one 32-byte sector", { size: 12, color: "hot" });
          G.label(320, 146, "per lane, 4 bytes used", { size: 12, color: "hot" });
          G.text(24, 250, "rung 2: lanes → columns of B", { size: 14, color: "ok" });
          for (let l = 0; l < 8; l++) G.box(24 + l * 34, 264, 28, 26, String(l), { stroke: "muted", size: 11, color: "muted", rx: 4 });
          G.rect(24, 320, 280, 14, { fill: "line", rx: 2 });
          const line2 = G.rect(24, 320, 272, 14, { fill: "ok", rx: 2 });
          for (let l = 0; l < 8; l++) G.line(38 + l * 34, 290, 38 + l * 34, 320, { color: "ok", w: 1 });
          G.label(320, 330, "one 128-byte line, all bytes used", { size: 12, color: "ok" });
          G.label(24, 370, "A[m][k] is the same for the whole warp → 1 broadcast", { size: 12 });
          G.label(24, 392, "still 2 loads per FMA = 0.25 FLOP per byte from DRAM", { size: 12, color: "ink" });
          G.from(sec1, { opacity: 0, stagger: 0.08, duration: 0.2 });
          G.from(line2, { attr: { width: 0 }, duration: 0.6, delay: 0.6 });
          G.caption("coalescing stops wasting bytes; it doesn't reuse any");
        } },

      { rail: "rung 3: tiles", title: "Rung 3: load a tile once, use it 32 times",
        body: `<p>The neighbours of <code>C[m][n]</code> need the same row of A and the same column of B. So let a block of 32×32 threads cooperate: it computes a 32×32 tile of C, and walks along k in steps of 32.</p>
<ol><li>Each thread loads <b>one</b> element of the A tile and one of the B tile into <b>shared memory</b> (the on-chip scratchpad from P5.2).</li>
<li><code>__syncthreads()</code>: wait until the whole tile is there.</li>
<li>Each thread does 32 FMAs reading only shared memory.</li>
<li><code>__syncthreads()</code> again before the tile is overwritten.</li></ol>
<div class="eq">per tile step, one block:
  loads    2 × 32 × 32 × 4 B = 8 KiB
  FMAs     32 × 32 × 32      = 32,768  (65,536 FLOPs)
  ratio    65,536 / 8,192    = 8 FLOP per byte</div>
<p>Each value fetched from global memory now feeds 32 FMAs instead of 1: DRAM traffic drops 32×, from ≈550 GB to ≈17 GB at N = 4096 (≈54 ms at 320 GB/s). This is <code>sgemm_r3</code> in <code>platform/kernels/include/d4/gemm.cuh</code>, and the load → sync → compute → sync loop of <code>animations/p5-smem-tiling.html</code>.</p>`,
        check: { q: "Rung 3 uses 32×32 tiles. How much does it cut global-memory loads compared with rung 2?",
          options: ["About 2×", "About 32×", "About 1,024×"], answer: 1,
          why: "Each element of an A or B tile is loaded from global memory once and then used by the 32 threads of its row or column from shared memory. Every global load now feeds 32 FMAs instead of 1." },
        scene(G) {
          const cs = 14, n = 8;
          G.label(250, 26, "B: tile column, k →", { color: "k", size: 11 });
          mat(G, 250, 34, n, cs, (r, c) => (c >= 4 && c < 6 ? (r < 2 ? "k" : "w") : null));
          G.label(16, 176, "A: tile row", { color: "q", size: 11 });
          mat(G, 120, 156, n, cs, (r, c) => (r >= 2 && r < 4 ? (c < 2 ? "q" : "w") : null));
          mat(G, 250, 156, n, cs, (r, c) => (r >= 2 && r < 4 && c >= 4 && c < 6 ? "ok" : null));
          G.label(250, 290, "C block tile (32×32)", { color: "ok", size: 11 });
          G.rect(390, 40, 230, 126, { stroke: "v", rx: 10, dash: "5 4" });
          G.label(400, 58, "shared memory, one block", { color: "v", size: 12 });
          const sa = G.box(404, 72, 96, 40, "A tile 32×32", { fill: "q", size: 11 });
          const sb = G.box(510, 72, 96, 40, "B tile 32×32", { fill: "k", size: 11 });
          G.label(404, 134, "4 KiB + 4 KiB = 8 KiB", { size: 12, color: "ink" });
          G.label(404, 152, "each value read by 32 threads", { size: 12 });
          const steps = ["load tiles", "sync", "32 FMAs", "sync"];
          const sx = steps.map((s, i) => G.box(390 + (i % 2) * 118, 200 + Math.floor(i / 2) * 50, 108, 36, s, { fill: i % 2 ? "w" : "v", size: 12 }));
          G.arrow(560, 286, 560, 330, { color: "muted", dash: "3 3" });
          G.label(400, 350, "slide 32 along k, repeat", { size: 12 });
          G.label(16, 374, "grey = tiles still to come along k", { size: 12 });
          G.label(16, 396, "global loads per FMA: 2 → 2/32", { size: 13, color: "ink" });
          G.from([sa, sb], { opacity: 0, x: -60, duration: 0.5, stagger: 0.2 });
          G.from(sx, { opacity: 0, stagger: 0.2, duration: 0.25, delay: 0.5 });
          G.caption("the coloured tiles meet in shared memory; each is reused 32 times");
        } },

      { rail: "tile shape", title: "Bigger tiles, thinner strips",
        body: `<p>Why stop at 32? Generalise. A block computes a BM×BN tile of C and steps along k by BK. Per step it loads (BM + BN) × BK floats and does BM × BN × BK FMAs:</p>
<div class="eq">intensity = 2·BM·BN·BK / ((BM + BN)·BK·4)
          = BM·BN / (2·(BM + BN))   FLOP per byte

 BM=BN=32  →  1024/128  = 8
 BM=BN=64  →  4096/256  = 16
 BM=BN=128 → 16384/512  = 32   (T4 ridge ≈ 25)</div>
<p>Two things to notice. <b>BK cancels</b>: the strip depth doesn't change intensity. And the shared memory per block is <code>(BM + BN) × BK × 4</code> bytes. Rung 3's square 32×32×32 tile costs 8 KiB; a 128×128 tile with the same BK = 32 would cost 32 KiB, enough to cut how many blocks fit on an SM (occupancy, P5.1). With a thin <b>BK = 8</b> it costs only 8 KiB.</p>
<p>So the higher rungs use big, flat tiles: 128×128 wide, 8 deep. But a 128×128 tile has 16,384 outputs, and a block has at most 1,024 threads. Each thread must compute several outputs. That is the next two rungs.</p>`,
        scene(G) {
          const T = [[32, 32], [64, 8], [128, 8], [128, 32]];
          G.text(24, 36, "intensity (FLOP/byte)", { size: 13 });
          G.text(340, 36, "shared memory per block", { size: 13 });
          const ai = T.map(([b]) => tileAI(b, b)), sm = T.map(([b, k]) => (2 * b * k * 4) / 1024);
          const ridgeY = 300 - (220 * 25) / 40;
          const b1 = G.bars(40, 300, ai, { w: 50, gap: 22, h: 220, max: 40, fill: (i, v) => (v >= 25 ? "ok" : "q") });
          G.line(30, ridgeY, 310, ridgeY, { color: "hot", dash: "5 4" });
          ai.forEach((v, i) => G.text(65 + i * 72, 292 - (220 * v) / 40, String(v), { anchor: "middle", size: 13 }));
          const b2 = G.bars(356, 300, sm, { w: 50, gap: 22, h: 220, max: 32, fill: (i, v) => (v > 16 ? "hot" : "v") });
          sm.forEach((v, i) => G.text(381 + i * 72, 292 - (220 * v) / 32, v + " KiB", { anchor: "middle", size: 12 }));
          T.forEach(([b, k], i) => {
            G.label(65 + i * 72, 320, `${b}×${b}`, { anchor: "middle", size: 11 }); G.label(65 + i * 72, 336, `BK ${k}`, { anchor: "middle", size: 11 });
            G.label(381 + i * 72, 320, `${b}×${b}`, { anchor: "middle", size: 11 }); G.label(381 + i * 72, 336, `BK ${k}`, { anchor: "middle", size: 11 });
          });
          G.label(24, 356, "dashed red = T4 ridge ≈ 25 FLOP/byte (example value)", { size: 12, color: "hot" });
          G.label(24, 376, "intensity depends on BM and BN only", { size: 12, color: "ink" });
          G.label(24, 396, "shared memory grows with BK too → keep BK small", { size: 12, color: "ink" });
          G.from(b1.concat(b2), { attr: { height: 0, y: 300 }, stagger: 0.08, duration: 0.4 });
          G.caption("128×128×8: past the ridge, still only 8 KiB of shared memory");
        } },

      { rail: "rung 4: 1D", title: "Rung 4: shared memory is the next wall",
        body: `<p>After rung 3, DRAM is no longer the main problem. Look inside the inner loop: <code>acc += As[ty][k] * Bs[k][tx]</code> issues <b>2 shared-memory loads for every FMA</b>. Shared memory is fast but not free: an SM can serve a limited number of shared-memory requests per clock (32 banks of 4 bytes, P5.2), while its FMA units want operands every clock. The kernel now waits on shared memory.</p>
<p>The fix is the same idea one level down: <b>registers</b> are the fastest storage and are private to a thread. Rung 4 uses a 64×64 block tile with BK = 8 and gives each of its 512 threads a column of <b>TM = 8</b> outputs. Per k, the thread reads one B value into a register and reuses it for 8 FMAs:</p>
<div class="eq">for k in 0..7:
    b = Bs[k][col]              // 1 smem load
    for i in 0..7:
        acc[i] += As[row+i][k] * b
        // 8 smem loads, 8 FMAs

smem loads per FMA: (1 + 8) / 8 = 1.125   (rung 3: 2)
DRAM intensity: 64·64 / (2·128) = 16 FLOP per byte</div>
<p>The ncu metric that moves is shared-memory wavefronts (<code>l1tex__data_pipe_lsu_wavefronts_mem_shared.sum</code>).</p>`,
        scene(G) {
          G.label(24, 40, "Bs strip", { color: "k", size: 12 }); G.label(24, 56, "8 × 64", { color: "k", size: 12 });
          const bs = G.grid(150, 20, 8, 16, 26, 7, (r, c) => (c === 5 ? "k" : "line"), { gap: 1, rx: 1 });
          G.label(16, 220, "As strip", { color: "q", size: 12 }); G.label(16, 236, "64 × 8", { color: "q", size: 12 });
          G.grid(80, 96, 16, 8, 7, 18, (r, c) => (r >= 4 && r < 6 ? "q" : "line"), { gap: 1, rx: 1 });
          G.grid(150, 96, 16, 16, 26, 18, (r, c) => (c === 5 && r >= 4 && r < 6 ? "ok" : null), { gap: 1, rx: 1 });
          G.rect(150, 96, 416, 288, { stroke: "line", rx: 4 });
          G.label(576, 120, "C tile", { color: "ok", size: 11 });
          G.label(576, 136, "64×64", { color: "ok", size: 11 });
          G.rect(286, 168, 26, 36, { stroke: "ink", sw: 2, rx: 2 });
          G.label(320, 186, "← one thread: 8 outputs", { size: 12, color: "ink" });
          G.label(320, 202, "(drawn as 2 cells, really 8 rows)", { size: 11 });
          const reg = G.box(330, 240, 120, 34, "b in a register", { fill: "k", size: 12 });
          G.arrow(299, 82, 299, 160, { color: "k", w: 2 });
          G.label(330, 296, "1 load of b → reused by 8 FMAs", { size: 13, color: "ink" });
          G.label(330, 316, "9 smem loads per 8 FMAs", { size: 13, color: "ink" });
          G.pulse(reg, { repeat: 5 });
          G.caption("one shared value in a register feeds a column of 8 outputs");
        } },

      { rail: "rung 5: 2D", title: "Rung 5: an outer product per thread",
        body: `<p>Rung 4 reuses B values but still loads one A value per FMA. Give each thread an 8×8 square of outputs instead (<b>TM = TN = 8</b>). At each k it loads 8 values of A's column and 8 values of B's row into registers, then multiplies <i>every</i> pair: an <b>outer product</b>.</p>
<div class="eq">for k in 0..7:
    rm[0..7] = As[rows][k]        // 8 smem loads
    rn[0..7] = Bs[k][cols]        // 8 smem loads
    for i, j: acc[i][j] += rm[i] * rn[j]   // 64 FMAs

smem loads per FMA: 16 / 64 = 0.25
block 128×128, 256 threads × 64 outputs
DRAM intensity: 128·128 / (2·256) = 32 FLOP per byte</div>
<p>Loads grow like TM + TN, FMAs like TM × TN: the same "perimeter vs area" argument as the block tile, now in registers. The price is registers: 64 accumulators + 16 operands = 80 floats per thread, before addresses and indices. Too many and the compiler <b>spills</b> to slow local memory; check <code>-Xptxas -v</code> for spill stores.</p>
<p>At 32 FLOP per DRAM byte the kernel is past the T4 ridge (≈25): DRAM is no longer the limit, the FMA units are.</p>`,
        check: { q: "Rung 5 gives each thread an 8×8 output tile. How many FMAs does each shared-memory load feed?",
          options: ["1", "4", "64"], answer: 1,
          why: "Per k the thread loads 8 A values and 8 B values (16 loads) and combines every pair: 8 × 8 = 64 FMAs. 64 / 16 = 4 FMAs per load, versus 0.5 in rung 3." },
        scene(G) {
          const x0 = 200, y0 = 110, c = 30;
          const rn = []; for (let j = 0; j < 8; j++) rn.push(G.box(x0 + j * c, y0 - 46, c - 4, 30, "", { fill: "k", rx: 4 }));
          const rm = []; for (let i = 0; i < 8; i++) rm.push(G.box(x0 - 46, y0 + i * c, 30, c - 4, "", { fill: "q", rx: 4 }));
          G.label(x0, y0 - 56, "rn[8]: one row of the B strip, in registers", { color: "k", size: 12 });
          G.label(x0 - 48, y0 + 8 * c + 18, "rm[8]", { color: "q", size: 12 });
          const cells = G.grid(x0, y0, 8, 8, c, c, () => "ok", { gap: 4, rx: 3 });
          G.text(470, 150, "per k step", { size: 14 });
          G.label(470, 174, "16 shared loads", { size: 13, color: "ink" });
          G.label(470, 194, "64 FMAs", { size: 13, color: "ink" });
          G.text(470, 228, "4 FMAs per load", { size: 15, color: "ok" });
          G.label(470, 270, "registers ≈ 64 acc", { size: 12 });
          G.label(470, 288, "+ 8 rm + 8 rn = 80", { size: 12 });
          G.label(470, 306, "watch for spills", { size: 12, color: "hot" });
          G.from(cells, { opacity: 0, stagger: { each: 0.012, from: "start" }, duration: 0.2, delay: 0.3 });
          G.from(rm.concat(rn), { opacity: 0, duration: 0.3 });
          G.caption("8 + 8 loads, 8 × 8 FMAs: area beats perimeter");
        } },

      { rail: "rung 6: vectors", title: "Rung 6: fewer, wider instructions",
        body: `<p>Rung 5 moves the right number of bytes but issues one load <i>instruction</i> per float. Every instruction occupies an issue slot that an FMA could have used. A <code>float4</code> load moves 16 bytes in one instruction: same bytes, a quarter of the load instructions.</p>
<p>B is easy: a thread's 8 values <code>Bs[k][cols]</code> are already contiguous, so two <code>float4</code> reads. A is not. In <code>As[m][k]</code> (row-major, BK = 8 wide) a thread's 8 values <code>As[rows][k]</code> sit in one column, 8 floats apart. The fix: store the A tile <b>transposed</b>, <code>As[k][m]</code>, while copying it in. Now the column is contiguous, and also two <code>float4</code> reads.</p>
<div class="eq">rung 5: rm[i] = As[(row+i)*BK + k]
        stride 8 floats → 8 scalar loads
rung 6: rm[i] = As[k*BM + row + i]
        stride 1 float  → 2 float4 loads</div>
<p>The global loads become <code>float4</code> too, so the kernel requires 16-byte alignment: K and N multiples of 4. <code>sgemm_supported()</code> asserts the contract; an unaligned <code>float4</code> crashes or silently reads the wrong data. The ncu metric to watch: load instructions executed, <code>smsp__inst_executed_op_ld.sum</code>.</p>`,
        scene(G) {
          G.text(24, 36, "As[m][k] (row-major)", { size: 13, color: "q" });
          const cs = 22;
          G.grid(24, 50, 8, 8, cs, cs, (r, c) => (c === 3 ? "q" : "line"), { gap: 2 });
          G.label(24, 242, "a thread wants column k = 3", { size: 12 });
          G.label(24, 262, "memory order:", { size: 12 });
          const mem1 = []; for (let i = 0; i < 24; i++) mem1.push(G.rect(24 + i * 12, 272, 10, 18, { fill: i % 8 === 3 ? "hot" : "line", rx: 1 }));
          G.label(24, 310, "8 scattered floats → 8 load instructions", { size: 12, color: "hot" });
          G.text(340, 36, "As[k][m] (stored transposed)", { size: 13, color: "q" });
          G.grid(340, 50, 8, 8, cs, cs, (r, c) => (r === 3 ? "q" : "line"), { gap: 2 });
          G.label(340, 242, "the same 8 values are now a row", { size: 12 });
          G.label(340, 262, "memory order:", { size: 12 });
          for (let i = 0; i < 24; i++) G.rect(340 + i * 12, 272, 10, 18, { fill: "line", rx: 1 });
          const v4 = [G.rect(340 + 0 * 12, 272, 46, 18, { fill: "ok", rx: 2 }), G.rect(340 + 4 * 12, 272, 46, 18, { fill: "ok", rx: 2 })];
          G.label(340, 310, "2 float4 loads, 16 bytes each", { size: 12, color: "ok" });
          G.label(24, 360, "same bytes moved; 4× fewer load instructions leave issue slots for FMAs", { size: 13, color: "ink" });
          G.label(24, 384, "(rows of the 8-deep strip shown 3 deep to fit)", { size: 11 });
          G.from(v4, { attr: { width: 0 }, duration: 0.5, stagger: 0.25 });
          G.caption("transposing the A tile turns a strided column into contiguous float4s");
        } },

      { rail: "rung 7: overlap", title: "Rung 7: never wait for the next tile",
        body: `<p>Rungs 3–6 follow a strict rhythm: load tile → sync → compute → sync → load. During the load, the FMA units idle; a global load takes hundreds of cycles to come back. ncu reports this as warps stalled on <b>long scoreboard</b> (waiting for a memory result).</p>
<p><b>Double buffering</b> gives shared memory two tile slots. While the block computes from slot 0, each thread has already issued the global loads for the next tile into registers; it then writes them into slot 1. One <code>__syncthreads()</code> per tile, and the memory latency hides behind compute.</p>
<div class="eq">issue loads for tile t+1 → registers
compute tile t from As[cur], Bs[cur]
store registers → As[cur^1], Bs[cur^1]
__syncthreads(); cur ^= 1</div>
<p>The cost: twice the shared memory (2 × 8 KiB = 16 KiB for 128×128×8) and extra registers for the in-flight loads, both of which can lower occupancy.</p>
<p>Rung 7 also adds <b>warp tiling</b>: each warp of 32 threads owns a compact 64×32 sub-tile, so its lanes read nearby shared-memory addresses and share fragments. The tile sizes become template parameters that <code>examples/autotune.py</code> sweeps on your GPU.</p>`,
        check: { q: "What does double buffering trade for hiding global-load latency?",
          options: ["Twice the shared memory per block, plus registers for loads in flight", "Twice the FLOPs", "It needs tensor cores"], answer: 0,
          why: "The next tile has to land somewhere while the current one is still being read, so the block holds two tiles in shared memory and the loads sit in registers in between. More resources per block can mean fewer blocks per SM." },
        scene(G) {
          const u = 40, x0 = 120;
          G.text(24, 40, "one buffer (rungs 3–6)", { size: 13 });
          G.label(24, 74, "memory", { size: 12 }); G.label(24, 108, "compute", { size: 12 });
          const a = [];
          for (let t = 0; t < 4; t++) {
            a.push(G.rect(x0 + t * 3 * u, 60, 1.5 * u - 4, 22, { fill: "v", rx: 3 }));
            a.push(G.rect(x0 + t * 3 * u + 1.5 * u, 94, 1.5 * u - 4, 22, { fill: "ok", rx: 3 }));
            G.rect(x0 + t * 3 * u, 94, 1.5 * u - 4, 22, { fill: "hot", rx: 3, opacity: 0.35 });
          }
          G.label(x0, 140, "red: FMA units idle while the tile loads", { size: 12, color: "hot" });
          G.text(24, 200, "two buffers (rung 7)", { size: 13 });
          G.label(24, 234, "memory", { size: 12 }); G.label(24, 268, "compute", { size: 12 });
          const b = [];
          b.push(G.rect(x0, 220, 1.5 * u - 4, 22, { fill: "v", rx: 3 }));
          for (let t = 0; t < 7; t++) {
            if (t < 6) b.push(G.rect(x0 + 1.5 * u + t * 1.5 * u, 220, 1.5 * u - 4, 22, { fill: "v", rx: 3, opacity: 0.8 }));
            b.push(G.rect(x0 + 1.5 * u + t * 1.5 * u, 254, 1.5 * u - 4, 22, { fill: "ok", rx: 3 }));
          }
          G.label(x0, 300, "loads for tile t+1 run while tile t computes", { size: 12, color: "ok" });
          G.line(x0, 160, x0 + 480, 160, { color: "line" }); G.line(x0, 316, x0 + 480, 316, { color: "line" });
          G.label(x0 + 480, 176, "time →", { anchor: "end", size: 11 });
          G.box(24, 340, 130, 40, "As[0] Bs[0]", { fill: "v", size: 12 });
          G.box(164, 340, 130, 40, "As[1] Bs[1]", { stroke: "v", dash: "4 3", color: "v", size: 12 });
          G.label(310, 358, "2 × 8 KiB = 16 KiB shared memory", { size: 12, color: "ink" });
          G.label(310, 376, "one __syncthreads per tile", { size: 12 });
          G.from(b, { opacity: 0, stagger: 0.08, duration: 0.2 });
          G.caption("same work, shorter timeline: memory latency hides behind math");
        } },

      { rail: "vs cuBLAS", title: "Measuring against cuBLAS honestly",
        body: `<p>The exit check (L5) is your best rung at ≥ 70% of cuBLAS SGEMM for N = 1024 to 8192. Three things make the comparison fair:</p>
<ul><li><b>Layout.</b> cuBLAS is column-major. A row-major matrix read as column-major is its transpose, so row-major C = A·B is column-major Cᵀ = Bᵀ·Aᵀ: pass B first. That is <code>d4::gemm_cublas</code>.</li>
<li><b>Precision.</b> On sm_80+ cuBLAS may use TF32 tensor cores (P5.7) for "fp32" unless told not to. Run with <code>NVIDIA_TF32_OVERRIDE=0</code>.</li>
<li><b>Method.</b> Same sizes, warm-up, median of many runs (the harness does it).</li></ul>
<p><b>Shape matters.</b> A 128×128 tile at N = 1024 makes (1024/128)² = 64 blocks. With one block per SM on a 40-SM T4 (example value), that is 2 <b>waves</b>, and the second wave keeps only 24 of 40 SMs busy. cuBLAS picks a different kernel per shape; your one tile size can't. And the decode GEMMs of P1.2 have M = batch size, often 1–64: with M = 1 a 128-row tile wastes 127/128 of its work. That is a GEMV, memory-bound, and exercise 2's subject.</p>
<p>Before you run each rung: <b>predict</b> its speed-up from the counts above, <b>measure</b> it with <code>--bench</code>, <b>explain</b> it with one ncu metric.</p>`,
        check: { q: "Your kernel multiplies row-major matrices. How do you get the same C from column-major cuBLAS without copying anything?",
          options: ["Transpose A and B on the GPU first", "Ask cuBLAS for Cᵀ = Bᵀ·Aᵀ by passing B as the first operand and A as the second", "It can't be done"], answer: 1,
          why: "A row-major array read as column-major is the transpose. Column-major cuBLAS computing Bᵀ·Aᵀ writes Cᵀ in column-major order, which is exactly C in row-major order. No data moves." },
        scene(G) {
          G.text(24, 36, "N = 1024 · 128×128 tiles → 64 blocks · 40 SMs (T4 example)", { size: 13 });
          G.label(24, 66, "wave 1", { size: 12 });
          const w1 = G.grid(90, 52, 2, 20, 26, 22, () => "ok", { gap: 3 });
          G.label(24, 128, "wave 2", { size: 12 });
          const w2 = G.grid(90, 114, 2, 20, 26, 22, (r, c) => (r * 20 + c < 24 ? "ok" : "hot"), { gap: 3 });
          G.label(90, 180, "red: SMs idle in the last wave · 64 / 80 slots = 80% used", { size: 12, color: "hot" });
          G.text(24, 228, "column-major trick", { size: 13 });
          G.box(24, 244, 160, 40, "row-major C = A·B", { stroke: "ink", size: 12 });
          G.text(198, 270, "≡", { size: 20 });
          G.box(226, 244, 190, 40, "col-major Cᵀ = Bᵀ·Aᵀ", { stroke: "k", color: "k", size: 12 });
          G.label(430, 262, "pass B first,", { size: 12 }); G.label(430, 278, "then A", { size: 12 });
          G.text(24, 330, "fair-fight checklist", { size: 13 });
          G.label(24, 352, "✓ NVIDIA_TF32_OVERRIDE=0     ✓ same N, warm-up, median", { size: 12, color: "ink" });
          G.label(24, 372, "✓ M = 1 (decode) is a GEMV: memory-bound, tile mostly wasted", { size: 12, color: "ink" });
          G.from(w2, { opacity: 0, stagger: 0.03, duration: 0.15, delay: 0.4 });
          G.from(w1, { opacity: 0, stagger: 0.01, duration: 0.15 });
          G.caption("one tile size can't fit every shape; cuBLAS chooses per shape");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>GEMM does 2N³ FLOPs on 3N² numbers, so it can be compute-bound, but only if each loaded number is reused many times.</li>
<li>Coalescing (rung 2) stops wasting bytes. Shared-memory tiles (rung 3) reuse each global load BM or BN times: intensity = BM·BN / (2(BM+BN)) FLOP per byte.</li>
<li>Register blocking (rungs 4–5) applies the same perimeter-vs-area argument to shared memory: 8×8 per thread gives 4 FMAs per shared load.</li>
<li>Vector loads, a transposed A tile, warp tiles and double buffering (rungs 6–7) cut instructions and hide latency.</li>
<li>Compare with cuBLAS fairly: swap operands, disable TF32, and remember shape effects (waves, M = 1).</li></ul>`,
    sim: {
      title: "Design a block tile",
      intro: "Choose a block tile (BM × BN × BK) and a per-thread tile (TM × TN). The chart is a roofline for the GPU you pick: the dot is your tile's DRAM intensity and the speed it allows. The panels show what the tile costs in threads, shared memory and registers, and how the blocks fill the GPU at size N.",
      height: 300,
      controls: [
        { id: "gpu", label: "GPU (example values)", type: "select", value: "t4", options: [["t4", "T4"], ["l4", "L4"], ["a100", "A100-80GB"]] },
        { id: "bm", label: "BM (block tile rows)", type: "select", value: 128, options: [[16, "16"], [32, "32"], [64, "64"], [128, "128"], [256, "256"]] },
        { id: "bn", label: "BN (block tile columns)", type: "select", value: 128, options: [[16, "16"], [32, "32"], [64, "64"], [128, "128"], [256, "256"]] },
        { id: "bk", label: "BK (strip depth)", type: "select", value: 8, options: [[8, "8"], [16, "16"], [32, "32"]] },
        { id: "tm", label: "TM (outputs per thread, rows)", type: "select", value: 8, options: [[1, "1"], [2, "2"], [4, "4"], [8, "8"], [16, "16"]] },
        { id: "tn", label: "TN (outputs per thread, cols)", type: "select", value: 8, options: [[1, "1"], [2, "2"], [4, "4"], [8, "8"], [16, "16"]] },
        { id: "db", label: "double buffering", type: "select", value: 0, options: [[0, "off"], [1, "on (2 tile slots)"]] },
        { id: "n", label: "matrix size N (square)", min: 256, max: 8192, step: 256, value: 4096 },
      ],
      draw(G, v) {
        const g = GPUS[v.gpu], BM = v.bm, BN = v.bn, BK = v.bk, TM = Math.min(v.tm, BM), TN = Math.min(v.tn, BN), N = v.n;
        const ai = tileAI(BM, BN), ridge = g.fp32 / g.bw;
        const W = 560, H = 230, x0 = 60, y0 = 26;
        const lx = (a) => x0 + ((Math.log10(a) + 1) / 4) * W, ly = (f) => y0 + H - ((Math.log10(f / 1e12) + 2) / 3) * H;
        G.axes(x0, y0, W, H, {});
        [0.1, 1, 10, 100, 1000].forEach((a) => G.label(lx(a), y0 + H + 16, String(a), { anchor: "middle", size: 11 }));
        [0.01, 0.1, 1, 10].forEach((f) => G.label(x0 - 6, ly(f * 1e12) + 4, String(f), { anchor: "end", size: 11 }));
        G.label(x0 + W, y0 + H + 34, "FLOP per DRAM byte (log)", { anchor: "end", size: 11 });
        G.label(x0 - 50, 14, "TFLOP/s (log)", { size: 11 });
        G.path(`M ${lx(0.1)} ${ly(0.1 * g.bw)} L ${lx(ridge)} ${ly(g.fp32)} L ${lx(1000)} ${ly(g.fp32)}`, { color: "ink", w: 2.5 });
        G.line(lx(ridge), ly(g.fp32), lx(ridge), y0 + H, { color: "muted", dash: "3 4" });
        G.label(lx(ridge) + 6, y0 + H - 8, `ridge ${ridge.toFixed(0)}`, { size: 11 });
        [[0.25, "r2"], [8, "r3"], [16, "r4"], [32, "r5–7"]].forEach(([a, l]) => { G.circle(lx(a), ly(Math.min(g.fp32, a * g.bw)), 4, { fill: "muted" }); G.label(lx(a), ly(Math.min(g.fp32, a * g.bw)) + 18, l, { anchor: "middle", size: 10 }); });
        const att = Math.min(g.fp32, ai * g.bw);
        G.circle(lx(ai), ly(att), 8, { fill: ai >= ridge ? "ok" : "q" });
        G.label(lx(ai) + 12, ly(att) - 10, `your tile: ${ai.toFixed(1)} FLOP/B`, { color: ai >= ridge ? "ok" : "q", size: 12 });

        const threads = (BM * BN) / (TM * TN);
        const smem = (BM + BN) * BK * 4 * (v.db ? 2 : 1);
        const regs = TM * TN + TM + TN;
        const smemPerFma = (TM + TN) / (TM * TN);
        const blocks = Math.ceil(N / BM) * Math.ceil(N / BN);
        const waves = blocks / g.sms, lastFill = blocks % g.sms === 0 ? 1 : (blocks % g.sms) / g.sms;
        const flops = 2 * N * N * N, bytes = flops / ai;
        const tC = flops / g.fp32, tM = bytes / g.bw;
        const okThreads = Number.isInteger(threads) && threads >= 32 && threads <= 1024;
        const okSmem = smem <= 48 * 1024, okRegs = regs + 20 <= 255;
        const valid = okThreads && okSmem && okRegs;
        return [
          { title: "Tile cost", rows: [["threads per block", Number.isInteger(threads) ? F.num(threads) : "not whole"], ["outputs per thread", F.num(TM * TN)], ["shared memory per block", F.bytes(smem)], ["registers per thread (≈, data only)", F.num(regs)], ["smem loads per FMA", smemPerFma.toFixed(3)]],
            chip: [valid, valid ? "launchable (by these checks)" : !okThreads ? "threads must be 32–1024" : !okSmem ? "over 48 KiB static smem" : "too many registers: spills"] },
          { title: "Intensity", rows: [["GPU fp32 peak · bandwidth", `${(g.fp32 / 1e12).toFixed(1)} TFLOP/s · ${F.num(g.bw / 1e9)} GB/s`], ["DRAM FLOP per byte", ai.toFixed(1)], ["GPU ridge point", ridge.toFixed(1)], ["bound", ai >= ridge ? "compute" : "memory"], ["attainable", F.si(att, 2) + "FLOP/s"]] },
          { title: `N = ${N} on ${g.name}`, rows: [["blocks", F.num(blocks)], ["waves (1 block per SM)", waves.toFixed(2)], ["last wave fill", (lastFill * 100).toFixed(0) + "%"], ["time if compute-bound", F.ms(tC)], ["time if DRAM-bound", F.ms(tM)], ["lower bound", F.ms(Math.max(tC, tM))]],
            html: `<p class="note">A model: every tile load counted as a DRAM read (L2 reuse ignored), one block per SM, peaks from gpu_specs.yaml (UNVERIFIED). Real kernels land below the lower bound's speed; the shape of the trade-offs is what to take away.</p>` },
        ];
      },
    },
    practice: {
      intro: `Build once per GPU session (see <code>course/P5-cuda-deep-dive/aws-common.md</code>): <code>cmake -S course/P5-cuda-deep-dive -B build/p5 -DCMAKE_CUDA_ARCHITECTURES=native &amp;&amp; cmake --build build/p5 -j</code>. Add <code>-DS2S_USE_SOLUTIONS=ON</code> (in a separate build dir) to run the tests against the reference kernels. For every rung: predict, measure, explain with one ncu metric.`,
      items: [
        { title: "GEMM rungs 3–7", tier: "T2 · hard", goal: "Implement rungs 3–7 behind sgemm(rung, …) in the starter kernel.cuh (rungs 1–2 are given); each is tested against cuBLAS with rtol = atol = 1e-3. Then run ./build/p5/p5.6_01-gemm-rungs --bench for the L5 exit check (best ≥ 70% of cuBLAS).",
          cmd: "ctest --test-dir build/p5 -R p5.6_01-gemm-rungs --output-on-failure" },
        { title: "Any shape, with predication", tier: "T2 · hard", goal: "sgemm_any: the rung-5 structure with zero-filled out-of-range loads and guarded stores, tested on 1×1×1, M = 1 and non-multiples. Bench M = 1, 8, 64 at N = K = 4096 against cuBLAS.",
          cmd: "ctest --test-dir build/p5 -R p5.6_02-predication --output-on-failure" },
      ],
      labs: [
        { label: "The ladder runner (rungs 1–7 + cuBLAS): run with NVIDIA_TF32_OVERRIDE=0 ./build/p5/p5.6_gemm_ladder", path: "course/P5-cuda-deep-dive/P5.6-gemm-ladder/examples/gemm_ladder.cu" },
        { label: "Plot TFLOP/s vs N per rung", path: "course/P5-cuda-deep-dive/P5.6-gemm-ladder/examples/gemm_plot.py" },
        { label: "Autotune rung 7's tile parameters (≈15 min of builds)", path: "course/P5-cuda-deep-dive/P5.6-gemm-ladder/examples/autotune.py" },
        { label: "The D4 kernels, one function per rung", path: "platform/kernels/include/d4/gemm.cuh" },
        { label: "AWS guide (g4dn.xlarge, ≈3 instance-hours, ncu per rung)", path: "course/P5-cuda-deep-dive/P5.6-gemm-ladder/aws.md" },
        { label: "Tiling animation from P5.2", path: "animations/p5-smem-tiling.html" },
        { label: "Reading: siboehm, How to Optimize a CUDA Matmul Kernel (kernels 1–12; read, don't vendor)", path: "https://siboehm.com/articles/22/CUDA-MMM" },
      ],
    },
  });
})();
