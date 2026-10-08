/* P1.4 — GPU architecture and the roofline. */
(function () {
  const F = S2S.fmt;
  // Copied from course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/gpu_specs.yaml.
  // EVERY value is UNVERIFIED there (author's recollection of NVIDIA datasheets); exercise 1 verifies them.
  // bw = memory bandwidth GB/s, fp16 / fp8 = dense tensor TFLOP/s (null = not supported), mem = GB, l2 = MB.
  const GPUS = {
    T4: { name: "T4", arch: "Turing", sm: 40, mem: 16, type: "GDDR6", bw: 320, l2: 4, fp32: 8.1, fp16: 65, fp8: null, aws: "g4dn.*" },
    L4: { name: "L4", arch: "Ada Lovelace", sm: 58, mem: 24, type: "GDDR6", bw: 300, l2: 48, fp32: 30.3, fp16: 121, fp8: 242, aws: "g6.*" },
    L40S: { name: "L40S", arch: "Ada Lovelace", sm: 142, mem: 48, type: "GDDR6", bw: 864, l2: 96, fp32: 91.6, fp16: 362, fp8: 733, aws: "g6e.*" },
    A100: { name: "A100-80GB-SXM", arch: "Ampere", sm: 108, mem: 80, type: "HBM2e", bw: 2039, l2: 40, fp32: 19.5, fp16: 312, fp8: null, aws: "p4de.24xlarge" },
    H100: { name: "H100-SXM", arch: "Hopper", sm: 132, mem: 80, type: "HBM3", bw: 3350, l2: 50, fp32: 67, fp16: 989, fp8: 1979, aws: "p5.48xlarge" },
  };
  const P8B = 8.03e9; // Llama-3-8B parameters (P1.1)

  /* log-log roofline frame. Returns mappers X(intensity), Y(TFLOP/s) and a roof(gpu, peak) path drawer. */
  function frame(G, x0, y0, w, h, o = {}) {
    const i0 = o.i0 ?? 0.125, i1 = o.i1 ?? 8192, t0 = o.t0 ?? 0.01, t1 = o.t1 ?? 3000;
    const X = (i) => x0 + (w * (Math.log2(i) - Math.log2(i0))) / (Math.log2(i1) - Math.log2(i0));
    const Y = (t) => y0 + h - (h * (Math.log10(t) - Math.log10(t0))) / (Math.log10(t1) - Math.log10(t0));
    G.axes(x0, y0, w, h, {});
    (o.xt ?? [0.125, 1, 8, 64, 512, 4096]).forEach((i) => { if (i >= i0 && i <= i1) G.label(X(i), y0 + h + 15, i < 1 ? String(i) : F.num(i), { anchor: "middle", size: o.fs ?? 10 }); });
    (o.yt ?? [0.01, 0.1, 1, 10, 100, 1000]).forEach((t) => { if (t >= t0 && t <= t1) G.label(x0 - 5, Y(t) + 3, String(t), { anchor: "end", size: o.fs ?? 10 }); });
    if (o.xlabel !== false) G.label(x0 + w, y0 + h + 30, o.xlabel ?? "arithmetic intensity (FLOP per byte, log)", { anchor: "end", size: o.fs ?? 10 });
    if (o.ylabel !== false) G.label(x0 + 4, y0 - 6, o.ylabel ?? "TFLOP/s (log)", { size: o.fs ?? 10 });
    const roof = (bw, peak, so = {}) => {
      const ridge = (peak * 1e12) / (bw * 1e9);
      const iStart = Math.max(i0, (t0 * 1e12) / (bw * 1e9));
      const d = `M ${X(iStart)} ${Y((iStart * bw) / 1000)} L ${X(ridge)} ${Y(peak)} L ${X(i1)} ${Y(peak)}`;
      return { path: G.path(d, Object.assign({ color: "k", w: 3 }, so)), ridge };
    };
    return { X, Y, roof };
  }
  const unverified = (G, x, y) => G.box(x, y, 96, 20, "UNVERIFIED", { stroke: "v", color: "v", size: 10, rx: 4 });

  S2S.lesson({
    id: "p1-4", n: "P1.4", title: "GPU architecture and the roofline",
    subtitle: "Inference fundamentals · first principles · T0 (exercise 4 needs one GPU)",
    kicker: "Lesson · ≈ 45 min",
    headline: "Two numbers predict your speed before you run anything",
    intro: `<p>P1.1 and P1.2 counted the work of a forward pass: FLOPs and bytes. This lesson turns those counts into time. It opens up a GPU, shows where its arithmetic and its memory live, and builds the <b>roofline</b>: a one-line model that tells you, from two datasheet numbers, whether a workload is waiting on arithmetic or on memory, and how many tokens per second it can reach at best.</p>
<p><b>Every GPU number here is UNVERIFIED.</b> They come from <code>gpu_specs.yaml</code>, which records the course author's recollection of NVIDIA's datasheets and has not yet been checked against them. Exercise 1 is to check them. Until you do, read each one as "approximately".</p>`,
    facts: ["11 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "memory / bandwidth"], ["q", "compute"], ["ok", "achievable"], ["hot", "the limit you hit"], ["v", "UNVERIFIED spec"]],
    prev: "p1-3", next: "p1-5",
    steps: [
      { rail: "two speeds", title: "A processor has two speeds",
        body: `<p>Any chip that runs a model has two separate limits. It can do at most so many <b>floating-point operations per second</b> (FLOP/s), and it can move at most so many <b>bytes per second</b> between its memory and its arithmetic units (the <b>memory bandwidth</b>). An operation needs both: its numbers must arrive, then be computed on.</p>
<p>If the chip can load data while it computes (GPUs can), the time is set by whichever takes longer:</p>
<div class="eq">time ≥ max( FLOPs ÷ peak FLOP/s ,
            bytes ÷ bandwidth )</div>
<p>Apply it to one decode token of Llama-3-8B in bf16 on an NVIDIA <b>L4</b> (121 TFLOP/s dense fp16, 300 GB/s, both UNVERIFIED):</p>
<div class="eq">compute: 15.0 GFLOP ÷ 121 TFLOP/s =  0.12 ms
memory:  16.06 GB  ÷ 300 GB/s     = 53.5  ms</div>
<p>The arithmetic would take a tenth of a millisecond. Reading the weights takes 430 times longer. The rest of the lesson explains where those two numbers come from and how to use them for any workload.</p>`,
        scene(G) {
          G.text(24, 44, "one decode token · Llama-3-8B bf16 · L4", { size: 14 });
          unverified(G, 518, 30);
          G.box(24, 90, 150, 70, "memory", { fill: "k", size: 14 });
          G.label(24, 180, "24 GB · 300 GB/s", { size: 12, color: "k" });
          G.box(454, 90, 160, 70, "arithmetic", { fill: "q", size: 14 });
          G.label(454, 180, "121 TFLOP/s", { size: 12, color: "q" });
          const flow = [];
          for (let i = 0; i < 6; i++) flow.push(G.rect(190 + i * 42, 115, 26, 20, { fill: "k", rx: 3, opacity: 0.8 }));
          G.arrow(186, 125, 446, 125, { color: "muted", dash: "4 4" });
          G.label(250, 106, "16.06 GB of weights", { size: 11 });
          const sc = 520 / 53.5;
          G.label(24, 248, "time to move the bytes", { size: 12, color: "ink" });
          const mb = G.rect(24, 258, 53.5 * sc, 30, { fill: "k", rx: 4 });
          G.text(34, 278, "53.5 ms", { size: 13, color: "bg" });
          G.label(24, 320, "time to do the arithmetic", { size: 12, color: "ink" });
          G.rect(24, 330, Math.max(3, 0.124 * sc), 30, { fill: "q", rx: 2 });
          G.text(36, 351, "0.12 ms", { size: 13, color: "q" });
          G.text(24, 400, "the slower of the two sets the pace: here, memory", { size: 13, color: "hot" });
          G.from(flow, { x: -30, opacity: 0, stagger: 0.1, duration: 0.4 });
          G.from(mb, { attr: { width: 0 }, duration: 1 });
          G.caption("time ≥ max(FLOPs ÷ FLOP/s, bytes ÷ bandwidth)");
        } },

      { rail: "the chip", title: "Inside a GPU: many small processors",
        body: `<p>A CPU has a handful of large cores. A GPU has dozens of smaller processors called <b>streaming multiprocessors (SMs)</b>: 58 on the L4, 132 on the H100 (UNVERIFIED). SMs are grouped into clusters (GPCs), but for performance you mostly think in SMs.</p>
<p>Each SM is split into <b>4 sub-partitions</b>. Each sub-partition has its own <b>warp scheduler</b>, a bank of simple arithmetic units ("CUDA cores", for FP32 and integer maths), a <b>tensor core</b> for matrix maths, and special-function units for <code>exp</code> or <code>sin</code>. The SM also owns a large <b>register file</b> and a block of fast on-chip memory used as L1 cache and <b>shared memory</b>.</p>
<p>Outside the SMs sit two shared resources: the <b>L2 cache</b> (48 MB on the L4) and the main memory, <b>HBM</b> or <b>GDDR</b> (24 GB of GDDR6 on the L4). The bandwidth number on a datasheet is the speed of that last link.</p>`,
        scene(G) {
          G.text(24, 36, "L4 (Ada Lovelace) · 58 SMs", { size: 13 });
          unverified(G, 236, 22);
          G.rect(20, 50, 312, 330, { stroke: "line", rx: 10 });
          const sms = [];
          for (let i = 0; i < 58; i++) {
            const r = Math.floor(i / 10), c = i % 10, y = r < 3 ? 62 + r * 30 : 250 + (r - 3) * 30;
            sms.push(G.rect(30 + c * 29.5, y, 25, 24, { fill: "q", rx: 3, opacity: i === 9 ? 1 : 0.6 }));
          }
          G.rect(30, 158, 292, 82, { fill: "w", opacity: 0.6, rx: 6 });
          G.text(176, 204, "L2 cache · 48 MB", { anchor: "middle", size: 13 });
          G.rect(20, 392, 312, 26, { fill: "k", rx: 6 });
          G.text(176, 410, "GDDR6 · 24 GB · 300 GB/s", { anchor: "middle", size: 12, color: "bg" });
          // zoom on one SM
          G.line(321, 62, 356, 50, { color: "q", dash: "3 3" }); G.line(321, 86, 356, 330, { color: "q", dash: "3 3" });
          G.rect(356, 50, 268, 280, { stroke: "q", rx: 10 });
          G.label(366, 68, "one SM", { size: 12, color: "q" });
          for (let p = 0; p < 4; p++) {
            const x = 366 + (p % 2) * 128, y = 78 + Math.floor(p / 2) * 92;
            G.rect(x, y, 120, 84, { stroke: "line", rx: 6 });
            G.label(x + 6, y + 15, "warp scheduler", { size: 10, color: "ink" });
            G.box(x + 6, y + 22, 52, 26, "FP32", { fill: "blue", size: 10, rx: 4 });
            G.box(x + 62, y + 22, 52, 26, "tensor", { fill: "q", size: 10, rx: 4 });
            G.box(x + 6, y + 52, 108, 24, "registers", { fill: "w", size: 10, rx: 4 });
          }
          G.box(366, 266, 248, 50, "L1 / shared memory", { fill: "k", size: 12 });
          G.from(sms, { opacity: 0, stagger: 0.01, duration: 0.2 });
          G.caption("many SMs share one L2 and one main memory");
        } },

      { rail: "warps", title: "Warps: hiding the wait instead of avoiding it",
        body: `<p>GPU threads run in groups of 32 called <b>warps</b>. All 32 execute the same instruction at the same time on different data, like the CPU's SIMD lanes from P0.4, but 32 wide.</p>
<p>A load from main memory takes hundreds of clock cycles to come back. A CPU hides that with big caches and by running instructions out of order. A GPU does something simpler: each SM keeps <b>many warps resident at once</b>, and every cycle the scheduler issues an instruction from any warp that is ready. While one warp waits for memory, others compute.</p>
<div class="eq">warp A: compute 2 cycles, then wait 6
4 warps, staggered: some warp is ready
every cycle → the scheduler never idles</div>
<p>This is why GPU code wants <b>lots</b> of threads: enough resident warps (<b>occupancy</b>) to cover the latency. P5.1 measures it. The catch for this lesson: switching hides <i>latency</i> but cannot create <i>bandwidth</i>. If every warp is waiting for bytes, nothing is ready.</p>`,
        check: { q: "How does an SM keep busy during a memory load that takes hundreds of cycles?",
          options: ["A large cache makes every load fast", "It runs instructions from other resident warps while that warp waits", "It speculates on the loaded value and continues"], answer: 1,
          why: "SMs keep many warps resident and issue from whichever is ready each cycle, so one warp's wait is filled with other warps' work. GPUs rely on this rather than on big caches or out-of-order execution." },
        scene(G) {
          const N = 24, cw = 22, x0 = 90;
          const busy = (w, s) => { const k = (s - w * 2 + 64) % 8; return k < 2; };
          G.text(24, 40, "four resident warps · one cell = one cycle", { size: 13 });
          const cells = [], cols = ["q", "k", "v", "ok"];
          for (let w = 0; w < 4; w++) {
            const y = 70 + w * 52;
            G.label(24, y + 22, `warp ${"ABCD"[w]}`, { size: 12, color: "ink" });
            for (let s = 0; s < N; s++) cells.push(G.rect(x0 + s * cw, y, cw - 3, 32, { fill: busy(w, s) ? cols[w] : "line", rx: 3, opacity: busy(w, s) ? 1 : 0.5 }));
          }
          G.label(x0, 290, "colour = issuing an instruction · grey = waiting for memory", { size: 11 });
          G.label(24, 336, "issued", { size: 12, color: "ink" });
          for (let s = 0; s < N; s++) { let w = 0; while (!busy(w, s)) w++; G.box(x0 + s * cw, 314, cw - 3, 32, "ABCD"[w], { fill: cols[w], size: 10, rx: 3 }); }
          G.label(x0, 372, "the scheduler issues every cycle, although each warp waits 75% of the time", { size: 11, color: "ok" });
          G.from(cells, { opacity: 0, stagger: 0.004, duration: 0.15 });
          G.caption("latency is hidden by switching warps, not by caches");
        } },

      { rail: "tensor cores", title: "Tensor cores supply almost all the FLOPs",
        body: `<p>Plain arithmetic units do one multiply-add per thread per instruction. A <b>tensor core</b> does a whole small matrix product in one instruction, for example a 16 × 16 tile times a 16 × 8 tile, accumulated into a 16 × 8 result:</p>
<div class="eq">C[16×8] += A[16×16] · B[16×8]
multiply-adds = 16 × 8 × 16 = 2,048
FLOPs         = 2 × 2,048   = 4,096</div>
<p>Matrix multiplication is exactly what the weight matrices of P1.1 need, so the headline "TFLOP/s" of an AI GPU is tensor-core throughput at a low precision (fp16, bf16, fp8). The FP32 number on plain cores is far lower:</p>
<div class="eq">        FP32 cores   fp16 tensor (dense)
L4        30.3          121     ≈ 4×
H100      67            989     ≈ 15×</div>
<p>Two consequences. Use the tensor-core figure for matrix multiplies and the FP32 figure for other code; mixing them up mis-states your efficiency by 4–15×. And lower precision is faster as well as smaller: on GPUs that support fp8 the table lists twice the fp16 rate (L4: 242, H100: 1,979). The T4 (Turing) and A100 (Ampere) have no fp8 tensor cores.</p>`,
        scene(G) {
          G.text(24, 40, "one tensor-core instruction", { size: 13 });
          const A = G.grid(24, 60, 16, 16, 8, 8, () => "q", { gap: 1, rx: 1 });
          G.label(24, 200, "A 16 × 16", { size: 11 });
          G.text(166, 130, "·", { size: 24, color: "muted" });
          const B = G.grid(184, 60, 16, 8, 8, 8, () => "k", { gap: 1, rx: 1 });
          G.label(184, 200, "B 16 × 8", { size: 11 });
          G.text(264, 130, "→", { size: 22, color: "muted" });
          const C = G.grid(292, 60, 16, 8, 8, 8, () => "ok", { gap: 1, rx: 1 });
          G.label(292, 200, "C 16 × 8", { size: 11 });
          G.label(380, 110, "4,096 FLOPs", { size: 14, color: "ink" });
          G.label(380, 130, "in one instruction", { size: 12 });
          G.text(24, 250, "dense TFLOP/s (gpu_specs.yaml)", { size: 13 });
          unverified(G, 300, 236);
          const rows = [["L4  FP32 cores", 30.3, "blue"], ["L4  fp16 tensor", 121, "q"], ["H100 FP32 cores", 67, "blue"], ["H100 fp16 tensor", 989, "q"]];
          const bars = rows.map(([n, v, c], i) => {
            const y = 270 + i * 34;
            G.label(24, y + 17, n, { size: 11, color: "ink" });
            const r = G.rect(150, y + 4, (440 * v) / 989, 20, { fill: c, rx: 3 });
            G.label(156 + (440 * v) / 989, y + 18, String(v), { size: 11 });
            return r;
          });
          G.from([A, B], { opacity: 0, stagger: 0.002, duration: 0.2 });
          G.from(C, { opacity: 0, stagger: 0.003, duration: 0.2, delay: 0.5 });
          G.from(bars, { attr: { width: 0 }, stagger: 0.15, duration: 0.5 });
          G.caption("matrix maths on tensor cores is 4–15× faster than plain FP32");
        } },

      { rail: "memory levels", title: "A ladder of memories: small and fast to big and slow",
        body: `<p>Data reaches the arithmetic units through a ladder. Each rung is bigger, farther away and slower than the one above:</p>
<table><tr><th>level</th><th>size (L4, UNVERIFIED)</th><th>shared by</th></tr>
<tr><td>registers</td><td>≈ 64K × 4 B = 256 KB per SM</td><td>one thread each</td></tr>
<tr><td>L1 / shared memory</td><td>≈ 100–256 KB per SM (varies by architecture)</td><td>the threads of one block</td></tr>
<tr><td>L2 cache</td><td>48 MB</td><td>all SMs</td></tr>
<tr><td>GDDR6 / HBM</td><td>24 GB at 300 GB/s</td><td>all SMs</td></tr></table>
<p>For inference the bottom rung decides almost everything. A Llama-3-8B layer's weights (≈ 436 MB in bf16) are far bigger than the 48 MB L2, so every decode step streams all 16 GB of weights from GDDR again. Caches help only when the same bytes are reused soon, which is what batching and tiling arrange (P5.2).</p>
<p>Two datasheet terms: <b>HBM</b> (high-bandwidth memory, stacked next to the chip: A100, H100) is much faster than <b>GDDR</b> (separate chips on the board: T4, L4, L40S). That is most of why an H100 has about 11× the L4's bandwidth.</p>`,
        scene(G) {
          const lv = [["registers", "256 KB / SM", 120, "q"], ["L1 / shared", "~100–256 KB / SM", 220, "blue"], ["L2 cache", "48 MB", 360, "w"], ["GDDR6 main memory", "24 GB · 300 GB/s", 540, "k"]];
          const bars = lv.map(([n, s, w, c], i) => {
            const y = 60 + i * 78, x = 320 - w / 2;
            const b = G.box(x, y, w, 52, n, { fill: c, size: 13 });
            G.label(320, y + 68, s, { anchor: "middle", size: 11 });
            return b;
          });
          G.arrow(612, 64, 612, 330, { color: "muted" });
          G.label(616, 356, "bigger, slower", { anchor: "end", size: 11 });
          G.label(24, 50, "closer, faster", { size: 11 });
          unverified(G, 24, 380);
          G.label(130, 394, "sizes for the L4 from gpu_specs.yaml and the P1.4 README", { size: 11 });
          G.from(bars, { opacity: 0, y: 20, stagger: 0.15, duration: 0.35 });
          G.caption("decode streams the weights from the bottom rung every step");
        } },

      { rail: "datasheet traps", title: "Reading a datasheet without being fooled",
        body: `<p>To use the two-speed model you need two numbers per GPU: <b>dense</b> peak FLOP/s at your precision, and memory bandwidth. Datasheets print both, but with traps:</p>
<ul><li><b>Sparsity.</b> "FP16 Tensor: 242 TFLOPS*", where the asterisk says "with sparsity". That figure assumes 2:4 structured sparsity (two of every four weights are zero and skipped). Normal models are dense, so <b>divide by 2</b>: the L4 does 121 dense.</li>
<li><b>Accumulate precision.</b> Some consumer cards halve fp16 throughput when accumulating in FP32. Read the footnotes.</li>
<li><b>Boost clock.</b> Peak assumes the boost clock; under a power cap the sustained clock can be lower.</li>
<li><b>Bandwidth</b> is a theoretical maximum. A good copy kernel reaches roughly 80–90% of it (the README's expectation; exercise 4 measures it).</li>
<li><b>Same name, different product.</b> A100 40 GB vs 80 GB, PCIe vs SXM: different bandwidth and power. Record the exact model (SKU).</li></ul>
<p>That is why <code>gpu_specs.yaml</code> stores only dense figures, names a <code>source:</code> document for each GPU, and stays <code>status: UNVERIFIED</code> until someone records the page they checked.</p>`,
        check: { q: "A datasheet lists \"FP8 Tensor: 1,979 TFLOPS*\" with the footnote \"with sparsity\". What peak do you use for a normal dense model?",
          options: ["1,979 TFLOP/s", "About 990 TFLOP/s", "About 495 TFLOP/s"], answer: 1,
          why: "The sparse figure assumes half the weights are structurally zero and skipped, which doubles the headline. A dense model gets half: about 990. Using 1,979 makes every efficiency number look half as good as it is." },
        scene(G) {
          G.rect(24, 40, 330, 250, { stroke: "line", rx: 10 });
          G.text(40, 70, "datasheet style, from gpu_specs.yaml", { size: 12, color: "muted" });
          const rows = [["FP32", "30.3 TFLOPS"], ["FP16 Tensor Core", "242 TFLOPS*"], ["FP8 Tensor Core", "484 TFLOPS*"], ["Memory bandwidth", "300 GB/s"]];
          rows.forEach(([a, b], i) => { const y = 108 + i * 38; G.text(40, y, a, { size: 13 }); G.text(338, y, b, { size: 13, anchor: "end", color: i === 1 || i === 2 ? "hot" : "ink" }); });
          G.label(40, 272, "* with sparsity", { size: 12, color: "hot" });
          const st = [G.line(250, 142, 340, 142, { color: "hot", w: 2 }), G.line(250, 180, 340, 180, { color: "hot", w: 2 })];
          G.arrow(360, 140, 420, 140, { color: "ok" }); G.arrow(360, 178, 420, 178, { color: "ok" });
          const d = [G.box(426, 124, 190, 30, "÷ 2 → 121 dense", { fill: "ok", size: 12 }), G.box(426, 162, 190, 30, "÷ 2 → 242 dense", { fill: "ok", size: 12 })];
          unverified(G, 426, 210);
          G.label(24, 330, "also check: accumulate precision · boost vs sustained clock ·", { size: 12, color: "ink" });
          G.label(24, 350, "bandwidth is a theoretical peak · exact SKU (PCIe vs SXM, 40 vs 80 GB)", { size: 12, color: "ink" });
          G.from(st, { attr: { x2: 250 }, duration: 0.4, stagger: 0.2 });
          G.from(d, { opacity: 0, x: -20, duration: 0.4, delay: 0.5, stagger: 0.2 });
          G.caption("headline tensor numbers often include a 2× sparsity bonus");
        } },

      { rail: "intensity", title: "Arithmetic intensity: FLOPs per byte",
        body: `<p>Divide an operation's FLOPs by the bytes it must move and you get its <b>arithmetic intensity</b> <code>I</code>, in FLOP per byte. It depends only on the operation, not on the GPU. P1.1 and P1.2 already computed the key ones:</p>
<div class="eq">decode, batch B, weights in fp16
  FLOPs = 2 × P × B,  bytes ≈ 2 × P
  I ≈ B                 (fp8: I ≈ 2B)

prefill, T prompt tokens:  I ≈ T

residual add, bf16: 1 FLOP per
  2 + 2 bytes read + 2 written: I ≈ 0.17</div>
<p>Attention during decode is the odd one. Each request reads its own KV cache, and every cached number is used by only the query heads that share it: in Llama-3-8B, 4 query heads per KV head gives <code>I ≈ 4</code> (1 without GQA). Batching does not help, because no two requests share a cache.</p>
<p>So the same model spans four orders of magnitude of intensity, from 0.17 to thousands. The question is where on that scale a given GPU changes from waiting on memory to waiting on arithmetic.</p>`,
        scene(G) {
          const X = (i) => 40 + (560 * (Math.log2(i) - Math.log2(0.125))) / (Math.log2(8192) - Math.log2(0.125));
          G.text(24, 40, "arithmetic intensity (FLOP per byte, log scale)", { size: 13 });
          G.line(40, 220, 600, 220, { color: "line", w: 2 });
          [0.125, 1, 8, 64, 512, 4096].forEach((i) => { G.line(X(i), 214, X(i), 226, { color: "muted" }); if (i >= 1) G.label(X(i) + 4, 210, F.num(i), { size: 11 }); });
          const ops = [[0.17, "residual add", "hot", 1], [1, "decode B=1", "k", 0], [4, "attention (decode)", "v", 1], [64, "decode B=64", "k", 0], [2048, "prefill T=2048", "q", 1]];
          const dots = ops.map(([i, n, c, up]) => {
            const y = up ? 150 : 290;
            G.line(X(i), 220, X(i), y + (up ? 14 : -14), { color: c, dash: "3 3" });
            G.text(X(i), up ? y : y + 4, n, { anchor: "middle", size: 12, color: c });
            return G.circle(X(i), 220, 8, { fill: c });
          });
          G.label(24, 350, "decode: I ≈ batch size · prefill: I ≈ prompt length", { size: 12, color: "ink" });
          G.label(24, 372, "elementwise ops and decode attention stay low whatever you do", { size: 12 });
          G.from(dots, { opacity: 0, scale: 0, transformOrigin: "center", stagger: 0.15, duration: 0.3 });
          G.caption("intensity belongs to the operation, not the GPU");
        } },

      { rail: "the roofline", title: "The roofline: min of two lines",
        body: `<p>Rewrite the two-speed rule as a rate. An operation of intensity <code>I</code> needs <code>1 / I</code> bytes per FLOP, so memory alone can feed it at most <code>I × bandwidth</code> FLOP/s. The arithmetic units cap it at the peak. The best achievable rate is the smaller:</p>
<div class="eq">attainable = min( peak , I × bandwidth )
ridge      = peak ÷ bandwidth

L4: 121e12 ÷ 300e9 ≈ 403 FLOP/byte</div>
<p>On log-log axes this is a slanted line (memory-bound) that hits a flat roof (compute-bound) at the <b>ridge point</b>. Left of the ridge, more FLOPs per byte means proportionally more speed. Right of it, only a faster GPU helps.</p>
<div class="eq">L4, I = 1:   1 × 300 GB/s = 0.3 TFLOP/s
             = 0.25% of the 121 peak
L4, I = 64:  19.2 TFLOP/s = 16% of peak</div>
<p>Batch-1 decode uses a quarter of one percent of the L4's tensor cores. Not because anything is broken: the bytes simply cannot arrive faster.</p>`,
        scene(G) {
          const L = GPUS.L4, f = frame(G, 70, 40, 530, 320);
          const bwLine = G.path(`M ${f.X(0.125)} ${f.Y(0.0375)} L ${f.X(8192)} ${f.Y(2457.6)}`, { color: "k", w: 1.5, dash: "5 5", opacity: 0.6 });
          G.line(f.X(0.125), f.Y(121), f.X(8192), f.Y(121), { color: "q", w: 1.5, dash: "5 5", opacity: 0.6 });
          const r = f.roof(L.bw, L.fp16, { color: "ok", w: 4 });
          G.line(f.X(r.ridge), f.Y(121), f.X(r.ridge), f.Y(0.01), { color: "muted", dash: "3 4" });
          G.label(f.X(r.ridge) + 6, f.Y(0.02), "ridge ≈ 403", { size: 12, color: "ink" });
          G.label(f.X(1500), f.Y(121) + 18, "peak 121", { size: 11, color: "q" });
          G.label(f.X(0.3), f.Y(0.3) - 18, "I × 300 GB/s", { size: 11, color: "k" });
          G.text(120, 76, "L4 · fp16 dense", { size: 13 });
          unverified(G, 240, 62);
          G.circle(f.X(1), f.Y(0.3), 6, { fill: "hot" }); G.label(f.X(1) + 10, f.Y(0.3) + 16, "decode B=1: 0.3", { size: 11, color: "hot" });
          G.circle(f.X(64), f.Y(19.2), 6, { fill: "hot" }); G.label(f.X(64) + 10, f.Y(19.2) + 16, "B=64: 19.2", { size: 11, color: "hot" });
          G.label(140, 200, "memory-bound", { size: 12, color: "k" }); G.label(470, 200, "compute-bound", { size: 12, color: "q" });
          G.from(r.path, { opacity: 0, duration: 0.8, delay: 0.4 });
          G.from(bwLine, { opacity: 0, duration: 0.5 });
          G.caption("attainable FLOP/s = min(peak, intensity × bandwidth)");
        } },

      { rail: "ridges compared", title: "Every GPU's ridge is in the hundreds",
        body: `<p>Compute the ridge for each GPU in the course's AWS table (dense fp16, from <code>gpu_specs.yaml</code>, UNVERIFIED):</p>
<div class="eq">GPU      TFLOP/s   GB/s   ridge
T4          65      320    ≈ 203
L4         121      300    ≈ 403
L40S       362      864    ≈ 419
A100 80GB  312    2,039    ≈ 153
H100 SXM   989    3,350    ≈ 295</div>
<p>A CPU's ridge, from P0.4, is in single digits. A GPU's is in the hundreds: it has far more arithmetic per byte of bandwidth. So the LLM workloads fall into place:</p>
<ul><li><b>Batch-1 decode</b> (I ≈ 1) is deeply memory-bound on all of them.</li>
<li><b>Batch-64 decode</b> (I ≈ 64) is still memory-bound, even on the A100.</li>
<li><b>Prefill</b> of a long prompt (I in the thousands) is compute-bound everywhere.</li>
<li><b>Decode attention</b> and <b>elementwise ops</b> are memory-bound at any batch size. Fusing elementwise ops into their neighbours removes their memory round trips (P5.5).</li></ul>
<p>For decode, compare GPUs by <b>GB/s</b> (and GB of capacity), not TFLOP/s. The T4 has more bandwidth than the L4 despite half its compute.</p>`,
        check: { q: "On the L4 (ridge ≈ 403 for fp16), is decode at batch 64 compute-bound?",
          options: ["Yes, 64 requests is plenty", "No: its intensity is about 64, well left of the ridge, so it is memory-bound", "It depends on the context length only"], answer: 1,
          why: "Decode intensity in fp16 is about the batch size, 64, and the ridge is about 403. Left of the ridge the rate is I × bandwidth, so the L4 is still waiting on memory. You would need a batch of roughly 400 (and the memory to hold it) to reach the roof." },
        scene(G) {
          const f = frame(G, 70, 40, 530, 320, { ylabel: false });
          const cols = { T4: "blue", L4: "ok", L40S: "v", A100: "pink", H100: "q" };
          const paths = [];
          Object.keys(GPUS).forEach((k, i) => {
            const g = GPUS[k], r = f.roof(g.bw, g.fp16, { color: cols[k], w: 2.5 });
            paths.push(r.path);
            G.circle(f.X(r.ridge), f.Y(g.fp16), 4, { fill: cols[k] });
            G.rect(180, 52 + i * 18, 10, 10, { fill: cols[k], rx: 2 });
            G.label(196, 61 + i * 18, `${k}  ridge ${Math.round(r.ridge)}`, { size: 11, color: "ink" });
          });
          [[1, "decode B=1"], [64, "B=64"], [2048, "prefill 2048"]].forEach(([i, n]) => { G.line(f.X(i), f.Y(0.01), f.X(i), f.Y(3000), { color: "hot", dash: "3 4", opacity: 0.7 }); G.label(f.X(i) + 4, f.Y(0.015), n, { size: 10, color: "hot" }); });
          G.text(84, 30, "dense fp16 rooflines · TFLOP/s vs FLOP/byte", { size: 12 });
          unverified(G, 420, 16);
          G.from(paths, { opacity: 0, stagger: 0.15, duration: 0.4 });
          G.caption("ridge points: 150–420 FLOP/byte; decode sits far to the left");
        } },

      { rail: "predict", title: "From a roofline to tokens per second",
        body: `<p>Now the payoff: predict speed before running anything. One decode step reads every weight once (P1.1), so for batch-1 decode:</p>
<div class="eq">ceiling = bandwidth ÷ bytes per token
Llama-3-8B bf16 on L4:
  300e9 ÷ 16.06e9 ≈ 18.7 tokens/s  (53.5 ms)
same model in fp8 (8.03 GB):
  300e9 ÷ 8.03e9  ≈ 37.4 tokens/s</div>
<p>Batch 64 reads the same 16.06 GB per step but produces 64 tokens: still 53.5 ms per step (the compute, 64 × 15.0 GFLOP ÷ 121 TFLOP/s = 7.9 ms, hides underneath), so about <b>1,196 tokens/s</b> in total while each user still sees 18.7. Prefill flips to the other limit:</p>
<div class="eq">prefill, 2,048 tokens:
  FLOPs ≈ 2 × 8.03e9 × 2048 ≈ 3.3e13
  at 121 TFLOP/s     ≈ 0.27 s
  at 60% of peak     ≈ 0.45 s  (TTFT)</div>
<p>These are <b>ceilings</b>. They ignore the KV-cache reads, kernel launch gaps and sampling, and assume the full datasheet bandwidth. P2.1 measures the real number with vLLM on an L4 and explains the gap term by term; the README expects 75–90% of the ceiling.</p>`,
        check: { q: "A 7B-parameter model in fp16 decodes at batch 1 on a GPU with 2 TB/s. What is the ceiling?",
          options: ["About 14 tokens/s", "About 143 tokens/s", "About 1,430 tokens/s"], answer: 1,
          why: "Each token reads all 7B × 2 bytes = 14 GB of weights. 2e12 B/s ÷ 14e9 B per token ≈ 143 tokens/s. Compute does not enter: at batch 1 the GPU is memory-bound." },
        scene(G) {
          const sc = 440 / 280, x0 = 160;
          const rows = [["decode B=1 bf16", 53.5, 0.124, "18.7 tok/s"], ["decode B=1 fp8", 26.8, 0.124, "37.4 tok/s"], ["decode B=64 bf16", 53.5, 7.9, "1,196 tok/s"], ["prefill 2,048", 53.5, 272, "TTFT ≥ 0.27 s"]];
          G.text(24, 40, "Llama-3-8B on an L4: time per pass (ms)", { size: 13 });
          unverified(G, 380, 26);
          const bars = [];
          rows.forEach(([n, mem, cmp, out], i) => {
            const y = 76 + i * 78;
            G.label(24, y + 22, n, { size: 12, color: "ink" });
            const memHot = mem >= cmp;
            bars.push(G.rect(x0, y, mem * sc, 20, { fill: "k", rx: 3, opacity: memHot ? 1 : 0.45 }));
            bars.push(G.rect(x0, y + 24, Math.max(2, cmp * sc), 20, { fill: "q", rx: 3, opacity: memHot ? 0.45 : 1 }));
            G.label(x0 + mem * sc + 6, y + 15, `bytes ${mem} ms`, { size: 10, color: "k" });
            if (cmp * sc > 360) G.label(x0 + 8, y + 38, `FLOPs ${cmp} ms`, { size: 11, color: "bg" }); else G.label(x0 + Math.max(2, cmp * sc) + 6, y + 39, `FLOPs ${cmp} ms`, { size: 10, color: "q" });
            G.text(616, y + 64, out, { anchor: "end", size: 13, color: "hot" });
          });
          G.from(bars, { attr: { width: 0 }, stagger: 0.08, duration: 0.4 });
          G.caption("the longer bar sets the time; decode waits on bytes, prefill on FLOPs");
        } },

      { rail: "measure it", title: "Measure the real bandwidth, and report % of peak",
        body: `<p>The roofline uses datasheet peaks. Your benchmarks should say how close they came. Exercise 4 measures achieved bandwidth with a large copy (count both the read and the write bytes), then compares it with <code>gpu_specs.yaml</code>.</p>
<p>The classic mistake is a buffer that fits in L2. A 16 MiB copy on an L4 lives inside its 48 MB L2 after the first pass, so you would be timing the cache, not GDDR. Use 1 GiB.</p>
<p>Every later bench in the course prints a "% of peak" column through one helper, <code>pct_of_peak</code> (exercise 3). It refuses to invent a peak: a <code>null</code> in the YAML (the A10G's tensor throughput is still unknown) returns <code>None</code>, and an unchecked spec is labelled as such:</p>
<div class="eq">annotate(123.4, "L4", "bandwidth")
→ "123.4 (41% of UNVERIFIED peak)"</div>
<p>Once you verify the L4's entry (exercise 1), the same call prints "41% of peak". That is the course's "verify before you cite" rule, enforced by code.</p>`,
        check: { q: "Why measure the L4's bandwidth with a 1 GiB buffer rather than 16 MiB?",
          options: ["Small copies are not allowed by CUDA", "16 MiB fits in the 48 MB L2, so you'd measure the cache instead of GDDR", "Larger buffers make the GPU raise its clock"], answer: 1,
          why: "After the first pass a 16 MiB buffer stays in L2, which is much faster than main memory, so the result would overstate the bandwidth that decode actually gets when streaming 16 GB of weights." },
        scene(G) {
          G.text(24, 40, "which memory does the copy hit?", { size: 13 });
          G.rect(24, 60, 250, 150, { stroke: "w", rx: 8, sw: 2 });
          G.label(34, 80, "L2 · 48 MB", { size: 12, color: "ink" });
          const small = G.box(44, 96, 120, 60, "16 MiB", { fill: "hot", size: 13 });
          G.label(34, 196, "fits: you time the cache", { size: 11, color: "hot" });
          G.rect(320, 60, 296, 150, { stroke: "k", rx: 8, sw: 2 });
          G.label(330, 80, "GDDR6 · 24 GB", { size: 12, color: "ink" });
          const big = G.box(340, 96, 250, 80, "1 GiB", { fill: "ok", size: 14 });
          G.label(330, 196, "≫ L2: you time main memory", { size: 11, color: "ok" });
          G.text(24, 264, "pct_of_peak / annotate (exercise 3)", { size: 13 });
          G.rect(24, 280, 592, 44, { stroke: "line", rx: 6 });
          G.text(36, 307, "123.4  (41% of UNVERIFIED peak)", { size: 14, color: "v" });
          G.label(24, 350, "A10G: tensor spec is null → pct_of_peak returns None", { size: 12 });
          G.label(24, 372, "after exercise 1: \"123.4 (41% of peak)\"", { size: 12, color: "ok" });
          G.from([small, big], { opacity: 0, scale: 0.6, transformOrigin: "center", stagger: 0.3, duration: 0.4 });
          G.caption("buffers bigger than L2, and peaks that say where they came from");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A GPU is many SMs, each with 4 warp schedulers, FP32 units and tensor cores, sharing an L2 cache and one main memory whose bandwidth is the datasheet's GB/s.</li>
<li>SMs hide memory latency by switching between resident warps; that hides waiting but cannot add bandwidth.</li>
<li>Use dense tensor-core FLOP/s (halve "with sparsity" figures) and record the exact SKU and source.</li>
<li>Attainable FLOP/s = min(peak, intensity × bandwidth); the ridge is peak ÷ bandwidth, in the hundreds for every GPU in the course.</li>
<li>Decode intensity ≈ batch size, so decode is memory-bound and its ceiling is bandwidth ÷ bytes per token (≈ 18.7 tokens/s for Llama-3-8B bf16 on an L4, UNVERIFIED spec). Prefill is compute-bound.</li></ul>`,
    sim: {
      title: "Place a workload on a roofline",
      intro: "Choose a GPU, a precision and a workload. The chart shows that GPU's roofline (log-log), the other GPUs faintly, and your workload as a dot. The panels predict the best-case time per pass and tokens per second for Llama-3-8B or Llama-3-70B. All specs are copied from <code>gpu_specs.yaml</code> and are UNVERIFIED until exercise 1.",
      height: 300,
      controls: [
        { id: "g", label: "GPU", type: "select", value: "L4", options: Object.keys(GPUS).map((k) => [k, GPUS[k].name]) },
        { id: "p", label: "weight precision", type: "select", value: "fp16", options: [["fp16", "bf16 · 2 B"], ["fp8", "fp8 · 1 B"]] },
        { id: "op", label: "workload", type: "select", value: "decode", options: [["decode", "decode"], ["prefill", "prefill"]] },
        { id: "n", label: "n = batch (decode) or prompt tokens (prefill)", min: 1, max: 4096, step: 1, value: 1 },
        { id: "m", label: "model", type: "select", value: "8", options: [["8", "Llama-3-8B"], ["70", "Llama-3-70B"]] },
        { id: "eff", label: "achievable % of peak", min: 30, max: 100, step: 5, value: 100 },
      ],
      draw(G, v) {
        const g = GPUS[v.g], b = v.p === "fp8" ? 1 : 2, peakRaw = v.p === "fp8" ? g.fp8 : g.fp16, e = v.eff / 100;
        const P = v.m === "70" ? 70.6e9 : P8B, n = v.n;
        const f = frame(G, 60, 26, 560, 228, { i0: 0.25, i1: 16384, t0: 0.03, t1: 5000, xt: [1, 8, 64, 512, 4096], yt: [0.1, 1, 10, 100, 1000] });
        Object.keys(GPUS).forEach((k) => { const o = GPUS[k], pk = v.p === "fp8" ? o.fp8 : o.fp16; if (k !== v.g && pk) f.roof(o.bw, pk, { color: "muted", w: 1.2, opacity: 0.45 }); });
        const I = (2 * n) / b;
        if (!peakRaw) {
          G.text(330, 120, `${g.name} has no ${v.p} tensor cores (null in gpu_specs.yaml)`, { anchor: "middle", size: 13, color: "hot" });
          return [{ title: g.name, chip: [false, "not supported at this precision"], rows: [["fp16 dense TFLOP/s", String(g.fp16)], ["fp8 dense TFLOP/s", "null"]] }];
        }
        const r = f.roof(g.bw, peakRaw, { color: "k", w: 3 });
        if (e < 1) f.roof(g.bw * e, peakRaw * e, { color: "ok", w: 2, dash: "5 4" });
        G.line(f.X(r.ridge), f.Y(peakRaw), f.X(r.ridge), f.Y(0.03), { color: "muted", dash: "3 4" });
        G.label(f.X(r.ridge) + 4, f.Y(0.05), "ridge " + Math.round(r.ridge), { size: 10, color: "ink" });
        const att = Math.min(peakRaw, (I * g.bw) / 1000) * e;
        const mem = att < peakRaw * e;
        G.circle(f.X(Math.min(I, 16384)), f.Y(Math.max(att, 0.031)), 7, { fill: "hot" });
        G.label(70, 44, `${g.name} · ${v.p} · I = ${F.num(I, I < 10 ? 1 : 0)} FLOP/byte`, { size: 11, color: "ink" });
        G.label(70, 60, "UNVERIFIED specs", { size: 10, color: "v" });
        // time per pass
        const bytes = P * b, flops = 2 * P * n;
        const tMem = bytes / (g.bw * 1e9 * e), tCmp = flops / (peakRaw * 1e12 * e), t = Math.max(tMem, tCmp);
        const fits = bytes / 1e9 <= g.mem;
        const out = v.op === "decode"
          ? { title: `Decode, batch ${n}`, rows: [["time per step", F.ms(t)], ["tokens/s per user", F.num(1 / t, 1)], ["tokens/s, all users", F.num(n / t, 0)], ["limited by", mem ? "memory bandwidth" : "compute"]] }
          : { title: `Prefill, ${n} prompt tokens`, rows: [["TTFT lower bound", F.ms(t)], ["prompt tokens/s", F.num(n / t, 0)], ["limited by", mem ? "memory bandwidth" : "compute"]] };
        out.html = `<p class="note">Weights only: KV-cache reads, attention FLOPs, launch gaps and sampling are ignored, so real runs are slower.</p>`;
        return [
          { title: `${g.name} (${g.arch}) · UNVERIFIED`, rows: [["dense " + v.p + " TFLOP/s", String(peakRaw)], ["memory bandwidth", g.bw + " GB/s"], ["ridge", F.num(r.ridge) + " FLOP/byte"], ["memory · L2", `${g.mem} GB ${g.type} · ${g.l2} MB`]] },
          { title: "Your workload on this roofline", rows: [["intensity 2n ÷ bytes/param", F.num(I, 1)], ["attainable", F.num(att, att < 10 ? 2 : 0) + " TFLOP/s"], ["% of datasheet peak", ((100 * att) / peakRaw).toFixed(att / peakRaw < 0.01 ? 2 : 1) + "%"]],
            gauge: [[att / peakRaw, mem ? "k" : "q"], [1 - att / peakRaw, "line"]], gaugeText: mem ? "memory-bound: left of the ridge" : "compute-bound: on the roof" },
          out,
          { title: "Does the model fit?", rows: [["weights", (bytes / 1e9).toFixed(1) + " GB"], ["GPU memory", g.mem + " GB"]], chip: [fits, fits ? "weights fit (KV cache needs the rest)" : "does not fit on one GPU: see P1.5"] },
        ];
      },
    },
    practice: {
      intro: "Run these from the repository root. <code>S2S_SOLUTIONS=1</code> runs a test against the reference solution. Run all four with <code>uv run pytest course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/exercises</code>; exercise 4 is skipped without CUDA.",
      items: [
        { title: "Verify gpu_specs.yaml (mandatory)", tier: "T0 · easy", goal: "Open each GPU's source document, check every number (halve sparse figures), fix and fill what you can, and mark the entry VERIFIED with document, page and date.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/exercises/01-verify-specs" },
        { title: "Ridge points and bound classification", tier: "T0 · easy", goal: "Write ridge, attainable_tflops, bound, decode_intensity and prefill_intensity; show fp8 doubles decode intensity and batch-1 decode is memory-bound on every GPU.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/exercises/02-classify" },
        { title: "The pct_of_peak helper", tier: "T0 · easy", goal: "Report measured bandwidth or FLOP/s as a % of the YAML peak, return None for a null spec, and label UNVERIFIED peaks.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/exercises/03-pct-of-peak" },
        { title: "Measure achieved bandwidth", tier: "T2 · hard", goal: "Time 20 large device-to-device copies with CUDA events, count read + write bytes, and reach at least 60% of the spec bandwidth on a 1 GiB copy.",
          cmd: "uv run --extra torch pytest course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/exercises/04-measure-bw" },
      ],
      labs: [
        { label: "The spec file every roofline reads (and its validator)", path: "course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/gpu_specs.yaml" },
        { label: "Example 01: load and validate the specs, print the ridge table (<code>uv run python …/01_gpu_specs.py</code>)", path: "course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/examples/01_gpu_specs.py" },
        { label: "Example 02: GPU rooflines with decode, prefill and attention placed (writes results/gpu_roofline.png)", path: "course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/examples/02_gpu_roofline.py" },
        { label: "Example 03 (T2, e.g. g6.xlarge): achieved copy bandwidth vs spec (<code>uv run --extra torch python …/03_measure_bw.py</code>)", path: "course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/examples/03_measure_bw.py" },
        { label: "Interactive roofline animation", path: "animations/roofline.html" },
      ],
    },
  });
})();
