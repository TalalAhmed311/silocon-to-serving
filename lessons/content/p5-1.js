/* P5.1 — The CUDA execution model: grid → blocks → warps → SMs, occupancy, divergence, async launches and streams. */
(function () {
  const F = S2S.fmt;
  // Per-SM limits by compute capability. Values recalled from the CUDA C++ Programming Guide table
  // "Technical Specifications per Compute Capability": UNVERIFIED, check the table or deviceQuery for your GPU.
  const ARCH = {
    75: { name: "sm_75 · T4", thr: 1024, blk: 16, regs: 65536, smem: 64 },
    80: { name: "sm_80 · A100", thr: 2048, blk: 32, regs: 65536, smem: 164 },
    89: { name: "sm_89 · L4", thr: 1536, blk: 24, regs: 65536, smem: 100 },
    90: { name: "sm_90 · H100", thr: 2048, blk: 32, regs: 65536, smem: 228 },
  };
  // a horizontal strip of n cells starting at x, each cw wide
  const strip = (G, x, y, n, cw, ch, fill, o = {}) => G.grid(x, y, 1, n, cw, ch, (r, c) => fill(c), o);

  S2S.lesson({
    id: "p5-1", n: "P5.1", title: "The CUDA execution model",
    subtitle: "CUDA deep dive · first principles · T2, one NVIDIA GPU (sm_75+)",
    kicker: "Lesson · ≈ 45 min",
    headline: "Ten thousand threads, one instruction at a time",
    intro: `<p>A GPU runs your function on thousands of threads at once. This lesson builds the machinery from the bottom: how you name each thread, how threads are grouped into blocks and warps, how blocks land on the GPU's processors (SMs), why an SM wants many warps at once (occupancy), what an <code>if</code> costs inside a warp, and how launches, errors and copies run asynchronously. Every kernel in the rest of P5 depends on these rules.</p>`,
    facts: ["11 steps", "4 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "thread / active lane"], ["v", "highlighted"], ["line", "idle / masked"], ["ok", "fits"], ["hot", "limit"]],
    prev: "p4-4", next: "p5-2",
    steps: [
      { rail: "a kernel", title: "One function, run by every thread",
        body: `<p>A CPU core runs one stream of instructions very fast. A GPU has many simpler cores and wins by running the <b>same function on many data items at once</b>. That function is a <b>kernel</b>. You write it for <i>one</i> element, and the GPU starts one copy, a <b>thread</b>, per element.</p>
<div class="eq">// CPU: one core walks the array
for (int i = 0; i &lt; n; ++i) c[i] = a[i] + b[i];

// GPU: each thread does ONE i
__global__ void vadd(const float* a, const float* b,
                     float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i &lt; n) c[i] = a[i] + b[i];
}
vadd&lt;&lt;&lt;blocks, threads&gt;&gt;&gt;(a, b, c, n);</div>
<p><code>__global__</code> marks a function the CPU can launch on the GPU. The <code>&lt;&lt;&lt;blocks, threads&gt;&gt;&gt;</code> part says how many threads to start, in two levels: a number of <b>blocks</b>, each with a number of threads. The next step explains why there are two levels and what <code>blockIdx</code>, <code>blockDim</code> and <code>threadIdx</code> are.</p>
<div class="analogy"><b>Picture it</b>One clerk stamping 10,000 forms in order, versus 10,000 clerks each stamping one form at the same moment. The GPU is the second office; the kernel is the instruction sheet every clerk follows.</div>`,
        scene(G) {
          G.text(24, 40, "CPU: one thread, n steps", { size: 14 });
          strip(G, 24, 56, 16, 36, 30, (c) => (c < 5 ? "w" : "line"), { gap: 3 });
          G.arrow(24 + 4 * 36 + 16, 106, 24 + 4 * 36 + 16, 90, { color: "v", w: 2 });
          G.label(24 + 4 * 36 + 28, 112, "i = 4 … still 11 to go", { color: "v" });
          G.text(24, 180, "GPU: 16 threads, 1 step each", { size: 14 });
          const th = []; for (let i = 0; i < 16; i++) th.push(G.box(24 + i * 36, 196, 33, 26, "t" + i, { fill: "k", size: 9, rx: 4 }));
          const ar = []; for (let i = 0; i < 16; i++) ar.push(G.arrow(40 + i * 36, 224, 40 + i * 36, 244, { color: "k" }));
          strip(G, 24, 250, 16, 36, 30, () => "ok", { gap: 3 });
          G.label(24, 300, "thread i computes c[i] = a[i] + b[i]; all 16 at once", { size: 13, color: "ink" });
          G.label(24, 340, "same code in every thread; only the index differs", { size: 13 });
          G.label(24, 366, "launch:  vadd<<<blocks, threads>>>(a, b, c, n)", { size: 13, color: "k" });
          G.from(th, { opacity: 0, y: -10, stagger: 0.03, duration: 0.3 });
          G.caption("the kernel is written for one element; the launch creates the threads");
        } },

      { rail: "grid & blocks", title: "Threads come in blocks, blocks form a grid",
        body: `<p>A launch creates a <b>grid</b> of <b>blocks</b>; each block holds the same number of threads (at most 1,024). Inside the kernel, every thread can read three built-in values:</p>
<ul><li><code>threadIdx.x</code>: my position inside my block (0 … blockDim.x − 1)</li>
<li><code>blockIdx.x</code>: which block I belong to</li>
<li><code>blockDim.x</code>: threads per block (the same for all blocks)</li></ul>
<p>Lay the blocks end to end and each thread gets a unique <b>global index</b>:</p>
<div class="eq">i = blockIdx.x * blockDim.x + threadIdx.x

blockDim = 256, block 2, thread 5:
  i = 2 * 256 + 5 = 517</div>
<p>Why two levels? Threads <i>in the same block</i> run on the same processor, so they can share a fast scratchpad (<b>shared memory</b>) and wait for each other (<code>__syncthreads()</code>). Threads in <i>different</i> blocks can't do either during a kernel. Blocks are independent, which lets the GPU run them in any order on any processor. (Indices also come in <code>.y</code> and <code>.z</code> for 2-D and 3-D problems.)</p>`,
        scene(G) {
          const bd = 8, x0 = 24, cw = 18;
          for (let b = 0; b < 4; b++) {
            const x = x0 + b * (bd * cw + 12);
            G.rect(x - 4, 60, bd * cw + 6, 66, { stroke: b === 2 ? "v" : "line", rx: 6, sw: b === 2 ? 2 : 1 });
            G.label(x, 52, "block " + b, { color: b === 2 ? "v" : "muted" });
            strip(G, x, 70, bd, cw, 22, (c) => (b === 2 && c === 5 ? "v" : "k"), { gap: 3 });
            for (let t = 0; t < bd; t++) G.label(x + t * cw + 7, 108, String(t), { size: 9, anchor: "middle" });
          }
          G.label(24, 150, "threadIdx.x restarts at 0 in every block · blockDim.x = 8 here", { size: 12 });
          G.text(24, 196, "global index i, blocks laid end to end:", { size: 13 });
          const cells = strip(G, 24, 210, 32, 18.5, 26, (c) => (c === 21 ? "v" : c >= 16 && c < 24 ? "q" : "line"), { gap: 3 });
          for (let i = 0; i < 32; i += 4) G.label(24 + i * 18.5 + 7, 252, String(i), { size: 9, anchor: "middle" });
          G.arrow(24 + 2 * (bd * cw + 12) + 5 * cw + 8, 96, 24 + 21 * 18.5 + 8, 206, { color: "v", w: 2 });
          G.text(24, 300, "i = blockIdx.x · blockDim.x + threadIdx.x", { size: 15, color: "ink" });
          G.text(24, 330, "  = 2 · 8 + 5 = 21", { size: 15, color: "v" });
          G.label(24, 370, "violet = block 2's slice of the array (elements 16–23)", { size: 12 });
          G.from(cells, { opacity: 0, stagger: 0.015, duration: 0.2 });
          G.caption("a unique index per thread, built from block and thread ids");
        } },

      { rail: "grid-stride", title: "A fixed grid for any n: the grid-stride loop",
        body: `<p>One thread per element means the grid must grow with <code>n</code>. That breaks in two ways: a huge <code>n</code> needs millions of blocks that mostly just wait in a queue, and an <code>int</code> index overflows past 2³¹ ≈ 2.1 billion elements.</p>
<p>Instead, launch a grid sized to the <b>GPU</b> (for example 8 blocks per processor × 256 threads) and let each thread loop, jumping by the total number of threads:</p>
<div class="eq">long long stride = (long long)gridDim.x * blockDim.x;
for (long long i = blockIdx.x * (long long)blockDim.x
                   + threadIdx.x;
     i &lt; n; i += stride)
  c[i] = a[i] + b[i];</div>
<p>Worked example: 2 blocks × 8 threads = 16 threads, <code>n = 40</code>. Thread 5 handles elements 5, 21 and 37; threads 8–15 handle only two elements each, because 40 isn't a multiple of 16. Neighbouring threads still touch neighbouring elements on every pass, which matters for memory speed (P5.2).</p>
<p>This is exactly <code>d4::vadd_grid_stride</code> in <code>platform/kernels/include/d4/elementwise.cuh</code>, launched with <code>sm_count() * 8</code> blocks of 256.</p>`,
        scene(G) {
          const cw = 14.8, x0 = 24;
          G.text(24, 40, "16 threads (2 blocks × 8) sweep n = 40 elements", { size: 14 });
          const passCol = (c) => (c < 16 ? "k" : c < 32 ? "blue" : "q");
          const cells = strip(G, x0, 70, 40, cw, 30, passCol, { gap: 2 });
          [5, 21, 37].forEach((i) => G.rect(x0 + i * cw - 1, 68, cw, 34, { stroke: "v", sw: 2.5, rx: 3 }));
          [0, 16, 32].forEach((i, p) => G.label(x0 + i * cw, 124, "pass " + p, { color: p === 0 ? "k" : p === 1 ? "blue" : "q" }));
          G.path(`M ${x0 + 5 * cw + 6} 66 C ${x0 + 5 * cw + 6} 36, ${x0 + 21 * cw + 6} 36, ${x0 + 21 * cw + 6} 66`, { color: "v", w: 2, arrow: true });
          G.path(`M ${x0 + 21 * cw + 6} 66 C ${x0 + 21 * cw + 6} 46, ${x0 + 37 * cw + 6} 46, ${x0 + 37 * cw + 6} 66`, { color: "v", w: 2, arrow: true });
          G.label(x0 + 13 * cw, 152, "thread 5: i = 5 → 21 → 37 (stride 16)", { color: "v", size: 13, anchor: "middle" });
          G.text(24, 210, "grid sized to the data:", { size: 13 });
          G.label(24, 232, "n = 2³⁶ elements ÷ 256 threads → 268 million blocks, int index overflows", { color: "hot" });
          G.text(24, 280, "grid sized to the GPU:", { size: 13 });
          G.label(24, 302, "40 SMs × 8 blocks × 256 threads = 81,920 threads, any n, long long i", { color: "ok" });
          G.label(24, 350, "(40 SMs: T4 entry in P1.4 gpu_specs.yaml, UNVERIFIED)", { size: 11 });
          G.from(cells, { opacity: 0, stagger: 0.02, duration: 0.15 });
          G.caption("each pass covers 16 consecutive elements; the loop repeats until i ≥ n");
        } },

      { rail: "warps", title: "The hardware runs threads 32 at a time: warps",
        body: `<p>The GPU doesn't schedule threads one by one. It groups each block's threads into <b>warps</b> of <b>32 consecutive threads</b>, and a warp issues <b>one instruction for all 32 lanes at once</b>. This is called SIMT: single instruction, multiple threads. A warp is the real unit of execution.</p>
<p>"Consecutive" is defined by the <b>linear thread id</b> inside the block, with <code>x</code> varying fastest:</p>
<div class="eq">tid  = threadIdx.z * (bx * by) + threadIdx.y * bx
     + threadIdx.x
warp = tid / 32        lane = tid % 32

blockDim = (16, 8), thread (x=5, y=3):
  tid = 3 * 16 + 5 = 53  →  warp 1, lane 21</div>
<p>Consequences you will use in every later lesson: a block of 256 threads is 8 warps; a block size that isn't a multiple of 32 wastes lanes (100 threads still occupy 4 warps, 128 lanes); and with a 16-wide 2-D block, each warp covers <b>two rows</b> of the block.</p>`,
        check: { q: "A block has blockDim = (8, 8). Which warp and lane is thread (x=3, y=5)?",
          options: ["warp 0, lane 43", "warp 1, lane 11", "warp 5, lane 3"], answer: 1,
          why: "Linear id = y·8 + x = 5·8 + 3 = 43. Warp = 43 / 32 = 1 (integer division), lane = 43 mod 32 = 11. Warps are cut from the linear id, not from rows." },
        scene(G) {
          const cw = 20, x0 = 24, y0 = 56;
          G.text(24, 36, "blockDim = (16, 8): 128 threads = 4 warps", { size: 14 });
          const wc = ["k", "blue", "q", "pink"];
          const cells = G.grid(x0, y0, 8, 16, cw, cw, (r, c) => (r === 3 && c === 5 ? "v" : wc[Math.floor((r * 16 + c) / 32)]), { gap: 3 });
          for (let r = 0; r < 8; r++) G.label(x0 + 16 * cw + 6, y0 + r * cw + 14, "y=" + r, { size: 10 });
          for (let w = 0; w < 4; w++) G.label(x0 + 16 * cw + 50, y0 + w * 2 * cw + 24, "warp " + w, { color: wc[w], size: 13 });
          G.label(x0, y0 + 8 * cw + 16, "x = 0 … 15 →", { size: 11 });
          G.text(472, 80, "thread (5, 3)", { size: 13, color: "v" });
          G.text(472, 104, "tid = 3·16 + 5", { size: 13 });
          G.text(472, 128, "    = 53", { size: 13 });
          G.text(472, 160, "warp = 53 / 32 = 1", { size: 13 });
          G.text(472, 184, "lane = 53 % 32 = 21", { size: 13 });
          G.text(24, 270, "one warp = 32 lanes = one instruction issued together", { size: 13 });
          const lanes = strip(G, 24, 284, 32, 18, 24, (c) => (c === 21 ? "v" : "blue"), { gap: 2 });
          for (let l = 0; l < 32; l += 8) G.label(24 + l * 18 + 4, 324, String(l), { size: 9 });
          G.label(24 + 21 * 18 - 4, 324, "21", { size: 9, color: "v" });
          G.label(24, 360, "warp 1 = rows y=2 and y=3 of this block", { size: 12 });
          G.from(lanes, { opacity: 0, stagger: 0.02, duration: 0.15 });
          G.caption("warps are cut every 32 linear ids, x fastest");
        } },

      { rail: "SMs", title: "Blocks are dealt out to SMs",
        body: `<p>A GPU is built from many <b>streaming multiprocessors (SMs)</b>. Each SM has its own warp schedulers, arithmetic units, register file and shared memory. A T4 has 40 SMs (P1.4's <code>gpu_specs.yaml</code>, UNVERIFIED until you run <code>deviceQuery</code>).</p>
<p>When you launch, a hardware scheduler hands whole blocks to SMs. A block <b>stays on one SM</b> until all its threads finish: that is why its threads can share memory. An SM can hold several blocks at once (how many is the next two steps). Blocks that don't fit wait, and start as earlier blocks finish.</p>
<p>A batch of blocks that runs together is called a <b>wave</b>. Example: 10 blocks, 4 SMs, 2 resident blocks per SM → 8 run in wave 1, and the last 2 run in wave 2 while 6 block slots sit empty. That half-empty last wave is the <b>tail effect</b>: with few, long blocks it wastes a large fraction of the GPU.</p>
<div class="eq">waves = ceil(blocks / (SMs × blocks per SM))
      = ceil(10 / (4 × 2)) = 2</div>
<p>Because blocks may run in any order, a kernel must never assume block 3 runs before block 7. <code>examples/01_hello_indices.cu</code> prints which SM ran each block (<code>%smid</code>): the order changes from run to run.</p>`,
        scene(G) {
          G.text(24, 34, "grid: 10 blocks", { size: 13 });
          const q = []; for (let b = 0; b < 10; b++) q.push(G.box(24 + b * 58, 46, 50, 30, "B" + b, { fill: b < 8 ? "k" : "v", size: 12, rx: 5 }));
          G.label(24, 96, "cyan = wave 1 · amber = wave 2", { size: 11 });
          for (let s = 0; s < 4; s++) {
            const x = 24 + s * 150;
            G.rect(x, 130, 138, 200, { stroke: "ink", rx: 10 });
            G.text(x + 10, 152, "SM " + s, { size: 13 });
            G.rect(x + 10, 162, 118, 54, { stroke: "line", dash: "4 3", rx: 6 });
            G.rect(x + 10, 226, 118, 54, { stroke: "line", dash: "4 3", rx: 6 });
            G.box(x + 16, 168, 106, 42, "B" + s, { fill: "k", size: 13, rx: 5 });
            G.box(x + 16, 232, 106, 42, "B" + (s + 4), { fill: "k", size: 13, rx: 5 });
            if (s < 2) G.box(x + 16, 290, 106, 30, "then B" + (s + 8), { fill: "v", size: 12, rx: 5 });
            else G.box(x + 16, 290, 106, 30, "idle", { stroke: "line", color: "muted", dash: "3 3", size: 12, rx: 5 });
          }
          G.label(24, 360, "2 resident blocks per SM · 8 slots · wave 2 fills 2 of 8", { size: 13, color: "ink" });
          G.label(24, 384, "waves = ceil(10 / 8) = 2 → tail wave 25% busy", { size: 13, color: "hot" });
          G.from(q, { opacity: 0, x: -20, stagger: 0.06, duration: 0.3 });
          G.caption("whole blocks go to SMs; leftovers wait for a free slot");
        } },

      { rail: "hiding latency", title: "Why an SM wants many warps",
        body: `<p>A load from GPU main memory (DRAM) takes hundreds of clock cycles to come back. A warp that needs that value can't issue its next dependent instruction, so it <b>stalls</b>.</p>
<p>The SM's answer is cheap switching. Every resident warp keeps its registers on the SM all the time, so a warp scheduler can pick a <i>different</i> ready warp on the very next cycle, at no cost. While warp 0 waits, warps 1, 2 and 3 compute. This is <b>latency hiding</b>: the waiting is still there, but the arithmetic units stay busy.</p>
<p>A toy model (made-up units, to show the shape): each warp computes for 1 unit, then waits 3 units for memory. With one warp the SM is busy 1 unit in 4 = 25%. With 4 warps taking turns, some warp is always ready: 100%.</p>
<div class="eq">warps needed ≈ (compute + wait) / compute
             = (1 + 3) / 1 = 4</div>
<p>That is why resident warps per SM matter, and why the next step counts them.</p>`,
        scene(G) {
          const u = 26, x0 = 110;
          G.text(24, 36, "toy model: compute 1 unit, then wait 3 units for memory", { size: 13 });
          G.text(24, 74, "1 warp", { size: 13 });
          for (let k = 0; k < 4; k++) { G.rect(x0 + k * 4 * u, 58, u - 2, 24, { fill: "k", rx: 3 }); G.rect(x0 + k * 4 * u + u, 58, 3 * u - 2, 24, { stroke: "line", dash: "3 3", rx: 3 }); }
          G.label(x0, 102, "issue slots busy 4 of 16 → 25%", { color: "hot" });
          G.text(24, 140, "4 warps", { size: 13 });
          const wc = ["k", "blue", "q", "pink"], blocks = [];
          for (let w = 0; w < 4; w++) {
            G.label(70, 162 + w * 30, "w" + w, { color: wc[w] });
            for (let k = 0; k < 4; k++) {
              const s = w + k * 4;
              blocks.push(G.rect(x0 + s * u, 146 + w * 30, u - 2, 24, { fill: wc[w], rx: 3 }));
              if (s + 1 < 16) G.rect(x0 + (s + 1) * u, 146 + w * 30, Math.min(3, 15 - s) * u - 2, 24, { stroke: "line", dash: "3 3", rx: 3 });
            }
          }
          G.text(24, 296, "issue", { size: 12 });
          const iss = strip(G, x0, 280, 16, u, 24, (c) => wc[c % 4], { gap: 2 });
          G.label(x0, 324, "scheduler picks a ready warp every slot → 100% busy", { color: "ok" });
          G.label(24, 370, "dashed = waiting on memory · solid = issuing instructions", { size: 12 });
          G.from(iss, { opacity: 0, stagger: 0.08, duration: 0.2 });
          G.caption("the wait doesn't shrink; other warps fill it");
        } },

      { rail: "occupancy", title: "Occupancy: how many warps fit on one SM",
        body: `<p><b>Occupancy</b> = resident warps per SM ÷ the maximum the SM supports. An SM runs out of one of four resources, and the tightest one decides how many blocks fit:</p>
<div class="eq">blocks/SM = min(
  threads_per_SM / threads_per_block,
  max_blocks_per_SM,
  regs_per_SM / (regs_per_thread × threads),
  smem_per_SM / smem_per_block )</div>
<p>Example values for sm_75 (CUDA C++ Programming Guide, compute-capability table; UNVERIFIED, check it or <code>deviceQuery</code>): 1,024 threads, 16 blocks, 65,536 registers and 64 KB shared memory per SM, so at most 32 warps.</p>
<p>Worked example from <code>examples/02_occupancy.cu</code>: 256-thread blocks, 64 registers per thread, 32 KB of shared memory per block (like <code>heavy_smem</code>):</p>
<div class="eq">threads: 1024 / 256          = 4
blocks:                         16
regs:    65536 / (64 × 256)   = 4
smem:    64 KB / 32 KB        = 2   ← tightest
→ 2 blocks × 8 warps = 16 of 32 warps = 50%</div>
<p>Drop the shared memory to zero and registers and threads tie at 4 blocks: 32 warps, 100%. Real hardware rounds register and shared-memory allocations up to fixed chunks, so <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code> (what the example calls) is the number to trust; <code>nvcc --resource-usage</code> shows each kernel's registers.</p>`,
        check: { q: "sm_75 example limits (65,536 registers, 1,024 threads per SM). A kernel uses 128 registers per thread in 256-thread blocks and no shared memory. Occupancy?",
          options: ["100%", "50%", "25%"], answer: 1,
          why: "One block needs 128 × 256 = 32,768 registers, so 65,536 / 32,768 = 2 blocks fit. That is 512 threads = 16 warps out of 32: 50%. The thread limit (4 blocks) isn't the tight one; registers are." },
        scene(G) {
          G.text(24, 34, "256 threads · 64 regs · 32 KB smem per block (sm_75 example limits)", { size: 13 });
          const lim = [["threads 1024 / 256", 4], ["block slots", 16], ["registers 65536 / 16384", 4], ["shared mem 64 KB / 32 KB", 2]];
          const bars = [];
          lim.forEach(([l, v], i) => {
            const y = 60 + i * 46;
            G.label(24, y + 18, l, { size: 12, color: v === 2 ? "hot" : "muted" });
            bars.push(G.rect(230, y + 4, v * 18, 22, { fill: v === 2 ? "hot" : "k", rx: 4 }));
            G.text(236 + v * 18, y + 20, v + " blocks", { size: 12, color: v === 2 ? "hot" : "ink" });
          });
          G.text(24, 270, "min = 2 blocks/SM → 2 × 8 warps = 16 warps", { size: 14 });
          G.text(24, 302, "warp slots on the SM (max 32):", { size: 12, color: "muted" });
          const slots = strip(G, 24, 314, 32, 18, 30, (c) => (c < 16 ? "ok" : "line"), { gap: 3 });
          G.text(24, 380, "occupancy = 16 / 32 = 50%", { size: 15, color: "ok" });
          G.from(bars, { attr: { width: 0 }, stagger: 0.12, duration: 0.4 });
          G.from(slots, { opacity: 0, stagger: 0.01, duration: 0.1 });
          G.caption("the scarcest resource sets blocks per SM");
        } },

      { rail: "enough is enough", title: "Occupancy is a means, not the goal",
        body: `<p>Occupancy only matters as a way to keep enough memory requests <b>in flight</b> to cover the latency. There is a second way to get them: give each thread several <b>independent</b> loads (instruction-level parallelism, ILP). A thread that issues four loads before using any of them keeps four requests in flight on its own.</p>
<div class="eq">// one load in flight per thread
float x = a[i];  use(x);
// four in flight per thread (ILP)
float x0 = a[i], x1 = a[i + s],
      x2 = a[i + 2*s], x3 = a[i + 3*s];
use(x0 + x1 + x2 + x3);</div>
<p>Both rows in the picture keep 16 requests in flight; the second needs a quarter of the warps. This is why fast kernels often run at <i>low</i> occupancy on purpose: the GEMM kernels in P5.6 spend many registers per thread holding a tile of results, accept 25–50% occupancy, and get far more reuse per byte loaded.</p>
<p>Rule: once latency is covered, more occupancy buys nothing, and chasing it (by capping registers with <code>__launch_bounds__</code>) can force <b>register spills</b> to slow local memory. Measure with Nsight Compute (P5.3) instead of maximizing a number.</p>`,
        scene(G) {
          G.text(24, 40, "goal: 16 memory requests in flight per SM", { size: 14 });
          G.text(24, 86, "A · 16 warps × 1 load each", { size: 13 });
          const a = []; for (let w = 0; w < 16; w++) { a.push(G.rect(24 + w * 36, 100, 30, 30, { fill: "k", rx: 4 })); G.circle(39 + w * 36, 146, 5, { fill: "v" }); }
          G.label(24, 180, "high occupancy · few registers per thread", { size: 12 });
          G.text(24, 236, "B · 4 warps × 4 independent loads each", { size: 13 });
          for (let w = 0; w < 4; w++) { a.push(G.rect(24 + w * 144, 250, 120, 30, { fill: "q", rx: 4 })); for (let k = 0; k < 4; k++) G.circle(42 + w * 144 + k * 28, 296, 5, { fill: "v" }); }
          G.label(24, 330, "low occupancy · more registers · same 16 requests in flight", { size: 12 });
          G.label(24, 370, "amber dot = one outstanding memory request", { size: 12, color: "v" });
          G.from(a, { opacity: 0, stagger: 0.03, duration: 0.2 });
          G.caption("latency is hidden by requests in flight, from warps or from ILP");
        } },

      { rail: "divergence", title: "An if inside a warp can run both sides",
        body: `<p>A warp issues one instruction for all 32 lanes. So what happens when lanes disagree on an <code>if</code>?</p>
<div class="eq">// odd vs even lanes
if (threadIdx.x &amp; 1) a(); else b();</div>
<p>The warp runs <code>a()</code> with the even lanes switched off (<b>masked</b>), then runs <code>b()</code> with the odd lanes masked. Both paths execute, one after the other, each using half the lanes: about <b>2× the time</b>. This is <b>warp divergence</b>. (Since Volta each lane has its own program counter, but a divergent warp still executes the paths separately.)</p>
<p>The fix is to make the condition the same for all 32 lanes of a warp:</p>
<div class="eq">// same answer for all 32 lanes of a warp
if ((threadIdx.x &gt;&gt; 5) &amp; 1) a(); else b();</div>
<p>Now warp 0 runs only <code>b()</code>, warp 1 only <code>a()</code>: no lane is ever idle. Short branches such as <code>x = c ? v*2 : v+1</code> cost nothing either: the compiler turns them into a <b>predicated select</b> that computes both cheap values and keeps one. <code>examples/03_divergence.cu</code> measures all three; expect divergent ≈ 2× uniform.</p>`,
        check: { q: "In a 256-thread block, the condition is (threadIdx.x < 128). How many warps diverge?",
          options: ["All 8", "4 of them", "None"], answer: 2,
          why: "Warps are threads 0–31, 32–63, … The boundary at 128 falls exactly between warp 3 and warp 4, so every warp sees the same answer for all its lanes. Divergence happens only when the split falls inside a warp." },
        scene(G) {
          const cw = 15.5, x0 = 110;
          G.text(24, 34, "divergent: if (lane & 1) a(); else b();", { size: 13 });
          G.label(24, 66, "pass 1: a()", { size: 12 });
          const p1 = strip(G, x0, 50, 32, cw, 24, (c) => (c & 1 ? "k" : "line"), { gap: 2 });
          G.label(24, 102, "pass 2: b()", { size: 12 });
          const p2 = strip(G, x0, 86, 32, cw, 24, (c) => (c & 1 ? "line" : "v"), { gap: 2 });
          G.label(x0, 134, "each pass: 16 of 32 lanes useful → 2 passes for one warp", { color: "hot" });
          G.text(24, 186, "warp-uniform: if ((tid >> 5) & 1) a(); else b();", { size: 13 });
          G.label(24, 218, "warp 0: b()", { size: 12 });
          strip(G, x0, 202, 32, cw, 24, () => "v", { gap: 2 });
          G.label(24, 254, "warp 1: a()", { size: 12 });
          strip(G, x0, 238, 32, cw, 24, () => "k", { gap: 2 });
          G.label(x0, 286, "one pass per warp, all 32 lanes useful", { color: "ok" });
          G.text(24, 340, "time per warp", { size: 12, color: "muted" });
          G.rect(140, 326, 300, 20, { fill: "hot", rx: 4 }); G.label(450, 341, "divergent ≈ 2×");
          G.rect(140, 354, 150, 20, { fill: "ok", rx: 4 }); G.label(300, 369, "uniform");
          G.from([p1, p2], { opacity: 0, stagger: 0.4, duration: 0.3 });
          G.caption("grey = lane masked off: it waits while the other path runs");
        } },

      { rail: "async launch", title: "Launches return before the kernel runs",
        body: `<p>A kernel launch only <b>queues</b> work for the GPU and returns to the CPU immediately. That's good (the CPU can prepare the next launch) but it changes where errors appear:</p>
<ul><li>A bad <b>launch configuration</b> (say 2,048 threads per block) is known at once: <code>cudaGetLastError()</code> right after the launch reports it.</li>
<li>A <b>fault inside the kernel</b> (an out-of-bounds write) happens later, on the GPU. It is reported by the <i>next</i> call that waits for the GPU, which may be a <code>cudaMemcpy</code> in a different file.</li></ul>
<div class="eq">vadd&lt;&lt;&lt;blocks, 256&gt;&gt;&gt;(a, b, c, n);
CUDA_CHECK(cudaGetLastError());      // config
CUDA_CHECK(cudaDeviceSynchronize()); // faults</div>
<p>The harness macro <code>CUDA_CHECK_LAUNCH()</code> (in <code>laneB-cuda/harness/include/s2s_cuda.cuh</code>) does both. For debugging, <code>CUDA_LAUNCH_BLOCKING=1</code> makes every launch synchronous so errors point at the right line.</p>
<p>Timing has the same trap: a host clock around a launch measures only the queuing. Time with CUDA events (<code>s2s::time_gpu</code>), which are recorded on the GPU's own timeline.</p>`,
        scene(G) {
          G.text(24, 40, "CPU", { size: 14 }); G.text(24, 150, "GPU", { size: 14 });
          G.line(80, 60, 620, 60, { color: "line" }); G.line(80, 170, 620, 170, { color: "line" });
          const c1 = G.box(90, 44, 90, 30, "launch", { fill: "k", size: 12, rx: 5 });
          G.box(190, 44, 140, 30, "other CPU work", { stroke: "line", size: 12, rx: 5 });
          G.box(340, 44, 120, 30, "cudaMemcpy", { fill: "v", size: 12, rx: 5 });
          G.arrow(135, 78, 200, 152, { color: "k", dash: "4 3" });
          const k = G.box(200, 154, 220, 32, "kernel runs · faults here", { fill: "hot", size: 12, rx: 5 });
          G.arrow(420, 154, 410, 80, { color: "hot", w: 2 });
          G.label(470, 110, "error reported", { color: "hot", size: 12 });
          G.label(470, 126, "at the memcpy", { color: "hot", size: 12 });
          G.text(24, 250, "launch returns at once → host timers measure ~nothing", { size: 13 });
          G.text(24, 280, "config errors: cudaGetLastError() right away", { size: 13, color: "k" });
          G.text(24, 306, "kernel faults: the next synchronizing call", { size: 13, color: "hot" });
          G.text(24, 332, "timing: CUDA events on the GPU timeline", { size: 13, color: "ok" });
          G.from(k, { opacity: 0, x: -40, duration: 0.6, delay: 0.3 }); G.from(c1, { opacity: 0, duration: 0.3 });
          G.caption("the CPU runs ahead; the GPU catches up later");
        } },

      { rail: "streams", title: "Streams overlap copies with compute",
        body: `<p>A <b>stream</b> is an in-order queue of GPU work. Operations in one stream run one after another; operations in <i>different</i> streams may run at the same time. The GPU has separate <b>copy engines</b> for host↔device transfers, so a copy in one stream can overlap a kernel in another.</p>
<p>Split the data into chunks and give each chunk its own copy-in, kernel, copy-out. Toy numbers: copy in, compute and copy out each take 3 units for the whole array. Serially that's 9 units. In 3 chunks of 1 unit each, chunk 2 copies in while chunk 1 computes, and so on: 5 units.</p>
<div class="eq">for (int c = 0; c &lt; chunks; ++c) {
  cudaStream_t st = s[c % 3];
  cudaMemcpyAsync(d + off, h + off, bytes, H2D, st);
  busy&lt;&lt;&lt;g, 256, 0, st&gt;&gt;&gt;(d + off, cn);
  cudaMemcpyAsync(h + off, d + off, bytes, D2H, st);
}</div>
<p>Two conditions. The host buffer must be <b>pinned</b> (<code>cudaMallocHost</code>): the copy engine reads it directly, while pageable memory has to be staged through a driver buffer and doesn't overlap (P0.2). And H2D and D2H run at the same time only if the GPU has two copy engines (<code>asyncEngineCount</code> in <code>deviceQuery</code>). Calling <code>cudaDeviceSynchronize()</code> inside that loop would serialize everything again. This is <code>examples/05_streams_overlap.cu</code>.</p>`,
        check: { q: "You switch the host buffer in 05_streams_overlap from cudaMallocHost to malloc. What happens to the overlap?",
          options: ["Nothing changes", "It largely disappears: pageable copies are staged and don't run concurrently with kernels", "The program crashes"], answer: 1,
          why: "The copy engine needs page-locked memory to DMA from. Pageable memory goes through a pinned staging buffer under driver control, and those copies don't overlap the way pinned async copies do. The code still runs, just serially." },
        scene(G) {
          const u = 40, x0 = 120;
          G.text(24, 36, "serial: one stream", { size: 13 });
          G.box(x0, 50, 3 * u - 2, 28, "H2D", { fill: "k", size: 12, rx: 4 });
          G.box(x0 + 3 * u, 50, 3 * u - 2, 28, "kernel", { fill: "q", size: 12, rx: 4 });
          G.box(x0 + 6 * u, 50, 3 * u - 2, 28, "D2H", { fill: "v", size: 12, rx: 4 });
          G.label(x0 + 9 * u + 6, 70, "9 units", { color: "hot" });
          G.text(24, 128, "3 chunks on 3 streams", { size: 13 });
          const bl = [];
          for (let s = 0; s < 3; s++) {
            const y = 146 + s * 40;
            G.label(24, y + 19, "stream " + s);
            bl.push(G.box(x0 + s * u, y, u - 2, 28, "H" + s, { fill: "k", size: 11, rx: 4 }));
            bl.push(G.box(x0 + (s + 1) * u, y, u - 2, 28, "K" + s, { fill: "q", size: 11, rx: 4 }));
            bl.push(G.box(x0 + (s + 2) * u, y, u - 2, 28, "D" + s, { fill: "v", size: 11, rx: 4 }));
          }
          G.line(x0 + 5 * u, 140, x0 + 5 * u, 270, { color: "ok", dash: "4 3" });
          G.label(x0 + 5 * u + 6, 266, "5 units", { color: "ok" });
          G.line(x0, 290, x0 + 9 * u, 290, { color: "line" });
          for (let t = 0; t <= 9; t++) G.label(x0 + t * u, 306, String(t), { size: 10, anchor: "middle" });
          G.label(24, 344, "needs pinned host memory · 2 copy engines for H2D ‖ D2H", { size: 12, color: "ink" });
          G.label(24, 368, "limit: approaches the slowest of the three stages", { size: 12 });
          G.from(bl, { opacity: 0, x: -10, stagger: 0.08, duration: 0.25 });
          G.caption("chunk 2 copies in while chunk 1 computes");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A kernel is written for one element; a launch creates a grid of blocks of threads, and <code>blockIdx·blockDim + threadIdx</code> gives each a unique index. A grid-stride loop with <code>long long</code> indices handles any n with a GPU-sized grid.</li>
<li>Threads execute in warps of 32 consecutive linear ids, one instruction per warp. Blocks are independent and are dealt to SMs in waves.</li>
<li>Resident warps hide memory latency. Occupancy is set by the tightest of threads, block slots, registers and shared memory, and is a means to latency hiding, not a goal.</li>
<li>Lanes that disagree on a long branch serialize both paths; make conditions warp-uniform or branch-free.</li>
<li>Launches are asynchronous: check errors after the launch and at a sync, time with events, and overlap copies with compute using streams and pinned memory.</li></ul>`,
    sim: {
      title: "Occupancy and waves calculator",
      intro: "Per-SM limits come from memory of the CUDA C++ Programming Guide's compute-capability table and are UNVERIFIED: check the table or deviceQuery. SM count is an example (T4 = 40 in gpu_specs.yaml, UNVERIFIED). Choose a compute capability and a kernel's resource use. The bars show how many blocks each SM resource allows; the tightest one wins. The strip shows resident warps out of the SM's maximum, and the panels count waves for a grid on a GPU with the SM count you set.",
      height: 300,
      controls: [
        { id: "a", label: "architecture", type: "select", value: 75, options: Object.keys(ARCH).map((k) => [k, ARCH[k].name]) },
        { id: "t", label: "threads per block", min: 32, max: 1024, step: 32, value: 256 },
        { id: "r", label: "registers per thread", min: 16, max: 255, value: 64 },
        { id: "s", label: "shared memory per block, KB", min: 0, max: 228, value: 0 },
        { id: "sm", label: "SMs on the GPU", min: 1, max: 160, value: 40 },
        { id: "g", label: "blocks in the grid", min: 1, max: 4000, step: 1, value: 400 },
      ],
      draw(G, v) {
        const A = ARCH[v.a], T = v.t, R = v.r, S = v.s;
        const lim = [
          ["threads", Math.floor(A.thr / T)],
          ["block slots", A.blk],
          ["registers", Math.floor(A.regs / (R * T))],
          ["shared mem", S > 0 ? Math.floor(A.smem / S) : Infinity],
        ];
        const bps = Math.max(0, Math.min(...lim.map((l) => l[1])));
        const maxW = A.thr / 32, warps = (bps * T) / 32, occ = warps / maxW;
        const scale = 380 / 32;
        G.label(10, 16, "blocks per SM allowed by each resource (bars capped at 32)", { size: 11 });
        lim.forEach(([n, b], i) => {
          const y = 28 + i * 34, tight = b === bps;
          G.text(10, y + 17, n, { size: 12, color: tight ? "hot" : "muted" });
          const w = Math.min(32, b) * scale;
          G.rect(110, y + 4, Math.max(2, w), 20, { fill: tight ? "hot" : "k", rx: 3, opacity: tight ? 1 : 0.6 });
          G.text(116 + Math.max(2, w), y + 19, b === Infinity ? "no limit" : String(b), { size: 12, color: tight ? "hot" : "ink" });
        });
        G.label(10, 182, `resident warps per SM: ${warps} of ${maxW}`, { size: 11 });
        const cw = Math.min(18, 600 / maxW);
        G.grid(10, 192, 1, maxW, cw, 26, (r, c) => (c < warps ? "ok" : "line"), { gap: 2 });
        const perWave = bps * v.sm, waves = bps ? Math.ceil(v.g / perWave) : 0, last = bps ? v.g - (waves - 1) * perWave : 0;
        G.label(10, 246, `waves: ${waves}  ·  last wave fills ${last} of ${perWave} block slots`, { size: 11 });
        if (waves) G.grid(10, 254, 1, Math.min(waves, 60), Math.min(40, 600 / Math.min(waves, 60)), 20, (r, c) => (c === waves - 1 && last < perWave ? "v" : "k"), { gap: 2 });
        return [
          { title: "Occupancy", rows: [["blocks per SM", String(bps)], ["warps per block", String(T / 32)], ["resident warps", `${warps} / ${maxW}`], ["occupancy", F.num(occ * 100, 0) + "%"]],
            gauge: [[occ, occ >= 0.5 ? "ok" : "v"]], gaugeText: `limited by ${lim.find((l) => l[1] === bps)[0]}`,
            chip: [bps > 0, bps > 0 ? "launches" : "does not launch: one block exceeds an SM resource"] },
          { title: "Per-block cost", rows: [["registers per block", F.num(R * T)], ["SM register file", F.num(A.regs)], ["shared memory per block", S + " KB"], ["SM shared memory", A.smem + " KB"]] },
          { title: "Waves on the GPU", rows: [["blocks per wave", F.num(perWave)], ["waves", String(waves)], ["last-wave utilization", perWave ? F.num((100 * last) / perWave, 0) + "%" : "–"]],
            html: `<p class="note">Model: ignores register and shared-memory allocation granularity, the per-block shared-memory reservation and the per-block shared-memory cap, so real occupancy can be lower. Trust <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code> (examples/02_occupancy.cu). Per-SM limits are from memory of the CUDA C++ Programming Guide table: check them.</p>` },
        ];
      },
      note: "Occupancy is a means to hide latency. Higher is not automatically faster: measure (P5.3).",
    },
    practice: {
      intro: "Build once from the repository root (needs CUDA 12.4+, CMake 3.24+ and an NVIDIA GPU; see the module's aws.md for a g4dn.xlarge): <code>cmake -S course/P5-cuda-deep-dive -B build/p5 -DCMAKE_CUDA_ARCHITECTURES=native &amp;&amp; cmake --build build/p5 -j</code>. Starter tests fail until you implement them; add <code>-DS2S_USE_SOLUTIONS=ON</code> (in a separate build dir) to build against the references.",
      items: [
        { title: "Grid-stride vector add for any N", tier: "T2 · easy", goal: "Write <b>vadd</b> with a grid-stride loop and a GPU-sized grid; the test checks n = 1 … 2²⁴ + 3 exactly. Then compare GB/s with the <b>float4</b> version.",
          cmd: "ctest --test-dir build/p5 -R p5.1_01-vector-add --output-on-failure && ./build/p5/p5.1_01-vector-add --bench" },
        { title: "Pick a block size by occupancy", tier: "T2 · medium", goal: "Run the occupancy example, explain each number by hand from the per-SM limits (cite the table), time <b>heavy_regs</b> at 64–512 threads, and try <b>__launch_bounds__(256, 4)</b>. Written answer: <b>exercises/02-occupancy.md</b>.",
          cmd: "./build/p5/p5.1_02_occupancy" },
        { title: "Remove divergence from a planted kernel", tier: "T2 · medium", goal: "Rewrite <b>clamp_scale_kernel</b> branch-free with identical results; confirm with the ncu active-threads-per-instruction ratio.",
          cmd: "ctest --test-dir build/p5 -R p5.1_03-divergence --output-on-failure && sudo $(which ncu) --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./build/p5/p5.1_03-divergence --bench" },
        { title: "Overlap H2D, compute and D2H with 3 streams", tier: "T2 · hard", goal: "Record serial vs pipelined time, capture the Nsight Systems timeline, vary the chunk count, and explain what pageable memory does. Brief: <b>exercises/04-streams.md</b>.",
          cmd: "nsys profile -o results/overlap ./build/p5/p5.1_05_streams_overlap" },
      ],
      labs: [
        { label: "Examples: indices and %smid, occupancy API, divergence, pinned vs pageable, streams", path: "course/P5-cuda-deep-dive/P5.1-execution-model/examples/" },
        { label: "AWS walkthrough (g4dn.xlarge, ≈ 2 instance-hours; cost, auto-stop and teardown on the shared page)", path: "course/P5-cuda-deep-dive/P5.1-execution-model/aws.md" },
        { label: "Shared P5 setup: build, Nsight counter access, teardown", path: "course/P5-cuda-deep-dive/aws-common.md" },
        { label: "D4 grid-stride and float4 vector add", path: "platform/kernels/include/d4/elementwise.cuh" },
        { label: "Harness: CUDA_CHECK, CUDA_CHECK_LAUNCH, time_gpu", path: "laneB-cuda/harness/include/s2s_cuda.cuh" },
        { label: "Standalone animation", path: "animations/p5-grid-warps-sm.html" },
      ],
    },
  });
})();
