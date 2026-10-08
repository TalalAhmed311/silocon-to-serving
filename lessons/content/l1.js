/* L1 — Launches and indexing (Lane B). Concepts behind the L1 LeetGPU problems: threads, blocks, grids,
   global indices, ceil-div and bounds, warps, grid-stride loops, row-major 2D/3D flattening, data layouts, host side. */
(function () {
  const F = S2S.fmt;
  const LV = "laneB-cuda/L1-launch-and-indexing";
  // one Practice item: LeetGPU number, slug folder, title, difficulty, priority, goal (our own words)
  // inline code inside a Practice goal (the shell styles <code> there as a block)
  const mono = (h) => h.replace(/<code>(.*?)<\/code>/g, '<span style="font-family:var(--f-mono);font-size:13px">$1</span>');
  const P = (slug, title, diff, prio, goal) => ({
    title: `#${slug.split("-")[0]} ${title}`, tier: `${diff} · ${prio}`,
    goal: mono(`${goal} Hints and reference solution: <code>${LV}/solutions/${slug}/</code>`),
    cmd: `ctest --test-dir build/laneB -R '^L1_${slug}$' --output-on-failure`,
  });
  // a row of n small cells, returns rects
  const cells = (G, x, y, n, w, h, fill, gap = 3) => Array.from({ length: n }, (_, i) => G.rect(x + i * (w + gap), y, w, h, { fill: typeof fill === "function" ? fill(i) : fill, rx: 3 }));

  S2S.lesson({
    id: "l1", n: "L1", title: "Launches and indexing",
    subtitle: "Lane B · CUDA on LeetGPU · T1 browser or T2 harness",
    kicker: "Lesson · ≈ 45 min",
    headline: "A million workers, each asking “which one is mine?”",
    intro: `<p>A GPU kernel is a function that thousands of threads run at the same time. Every thread runs the <i>same</i> code, so the first thing each one must do is work out which piece of the data belongs to it. That one calculation, the <b>index</b>, is what every L1 problem tests.</p>
<p>The dark panel builds it up: the thread hierarchy, the global index, the guard for the last partial block, warps, the grid-stride loop, then 2D and 3D data flattened into memory. It ends with the host code around a launch and the bytes each kernel moves.</p>`,
    facts: ["10 steps", "4 checkpoints", "1 simulator", "15 problems"],
    legend: [["k", "thread / element"], ["v", "highlighted"], ["hot", "idle or wrong"], ["ok", "valid"]],
    prev: "c2", next: "l2",
    steps: [
      { rail: "loop to kernel", title: "A kernel is the body of a loop",
        body: `<p>Adding two arrays on a CPU is a loop: one worker visits element 0, then 1, then 2, up to N − 1. Each iteration is independent of every other one. Nothing in iteration 7 needs the result of iteration 6.</p>
<p>A GPU exploits that independence. Instead of one worker doing N iterations, it starts N <b>threads</b> (lightweight workers) and each does <i>one</i> iteration. The code each thread runs is the loop body, written as a <b>kernel</b>: a function marked <code>__global__</code> that the CPU launches on the GPU.</p>
<div class="eq">// CPU: one worker, N iterations
for (int i = 0; i &lt; N; ++i) C[i] = A[i] + B[i];

// GPU: N workers, one iteration each
__global__ void add(const float* A, const float* B,
                    float* C, int N) {
  int i = /* which element is mine? */;
  C[i] = A[i] + B[i];
}</div>
<p>The loop variable <code>i</code> is gone. Each thread has to compute it from its own position among the threads. The rest of this lesson is about that computation.</p>
<div class="analogy"><b>Picture it</b>A teacher marking 1,000 exams alone, versus 1,000 assistants who each mark one. The assistants only need to know which exam is theirs.</div>`,
        scene(G) {
          G.text(24, 40, "CPU: one worker walks the loop", { size: 14 });
          const a = cells(G, 24, 60, 12, 40, 34, (i) => (i < 5 ? "k" : "line"));
          for (let i = 0; i < 4; i++) G.arrow(64 + i * 43, 77, 67 + i * 43, 77, { color: "ink" });
          G.circle(44, 118, 12, { fill: "v" }); G.label(64, 122, "1 worker · element 0, then 1, then 2 …", { color: "ink" });
          G.text(24, 190, "GPU: one thread per element, all at once", { size: 14 });
          const th = Array.from({ length: 12 }, (_, i) => G.circle(44 + i * 43, 222, 10, { fill: "v" }));
          for (let i = 0; i < 12; i++) G.arrow(44 + i * 43, 234, 44 + i * 43, 254, { color: "muted" });
          const b = cells(G, 24, 260, 12, 40, 34, "k");
          for (let i = 0; i < 12; i++) G.label(44 + i * 43, 312, String(i), { anchor: "middle", size: 11 });
          G.label(24, 350, "every thread runs the same kernel body", { color: "ink", size: 13 });
          G.label(24, 372, "it must compute its own i from its position", { size: 13 });
          G.from(a, { opacity: 0.2, stagger: 0.25, duration: 0.2 }); G.from(b, { opacity: 0, duration: 0.5, delay: 0.4 });
          G.caption("same code, many threads: the only difference is the index");
        } },

      { rail: "grid · block · thread", title: "Threads come in blocks, blocks in a grid",
        body: `<p>Threads are not launched one by one. They are grouped into <b>blocks</b> of up to 1,024 threads (CUDA Programming Guide, compute capability table), and the blocks form a <b>grid</b>. You choose both sizes at launch:</p>
<div class="eq">add&lt;&lt;&lt;blocks, threads_per_block&gt;&gt;&gt;(A, B, C, N);</div>
<p>Inside the kernel, four built-in variables tell a thread where it is:</p>
<ul><li><code>threadIdx.x</code>: my position inside my block (0 … blockDim.x − 1)</li>
<li><code>blockIdx.x</code>: which block I am in (0 … gridDim.x − 1)</li>
<li><code>blockDim.x</code>: threads per block; <code>gridDim.x</code>: blocks in the grid</li></ul>
<p>Lay the blocks end to end and the thread's position along the whole line is the <b>global index</b>:</p>
<div class="eq">i = blockIdx.x * blockDim.x + threadIdx.x

blockDim.x = 256, blockIdx.x = 2, threadIdx.x = 5
i = 2 * 256 + 5 = 517</div>
<p>Blocks 0 and 1 hold threads 0–511, so thread 5 of block 2 is number 517. Why blocks at all? A block runs on one streaming multiprocessor, its threads can share fast on-chip memory and wait for each other (L2). Different blocks cannot, which lets the hardware run them in any order.</p>`,
        scene(G) {
          const bd = 8;
          G.rect(16, 40, 608, 150, { stroke: "muted", dash: "5 4", rx: 10 }); G.label(26, 34, "grid · gridDim.x = 4", { color: "muted" });
          const hl = [];
          for (let b = 0; b < 4; b++) {
            const x0 = 28 + b * 150;
            G.rect(x0 - 4, 58, 142, 112, { stroke: "line", rx: 8 });
            G.label(x0, 76, `block ${b}`, { color: "ink" });
            for (let t = 0; t < bd; t++) {
              const on = b === 2 && t === 5;
              const r = G.rect(x0 + t * 17, 90, 14, 40, { fill: on ? "v" : "k", rx: 3, opacity: on ? 1 : 0.8 });
              if (on) hl.push(r);
              G.label(x0 + t * 17 + 7, 148, String(t), { anchor: "middle", size: 9 });
            }
          }
          G.label(26, 214, "threadIdx.x (under each thread)", { size: 11 });
          G.text(24, 260, "blockDim.x = 8 in this picture", { size: 14 });
          G.text(24, 292, "i = blockIdx.x · blockDim.x + threadIdx.x", { size: 15, color: "k" });
          G.text(24, 324, "  = 2 · 8 + 5 = 21", { size: 15, color: "v" });
          const strip = cells(G, 24, 352, 32, 15, 22, (i) => (i === 21 ? "v" : i < 16 ? "w" : "k"), 3);
          G.label(24, 396, "global index 0 … 31: blocks 0–1 cover 0–15, so block 2 starts at 16", { size: 11 });
          G.pulse(hl, { repeat: 6 }); G.from(strip, { opacity: 0, stagger: 0.02, duration: 0.2 });
          G.caption("the global index counts threads across all earlier blocks");
        } },

      { rail: "ceil-div + guard", title: "Round the grid up, then guard",
        body: `<p>N is rarely a multiple of the block size. With N = 1,000 and 256 threads per block, 3 blocks give 768 threads (too few) and 4 give 1,024 (24 too many). We must round <b>up</b>. Integer division rounds down, so add <code>threads − 1</code> first: the <b>ceil-div</b> idiom.</p>
<div class="eq">blocks = (N + threads - 1) / threads
       = (1000 + 255) / 256 = 1255 / 256 = 4</div>
<p>Now threads 1,000 … 1,023 exist but have no element. Without a check, thread 1,023 writes <code>C[1023]</code>: past the end of the array. That is an out-of-bounds write. It may crash, or silently corrupt another buffer. Every kernel therefore starts with a <b>bounds guard</b>:</p>
<div class="eq">int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i &lt; N) C[i] = A[i] + B[i];</div>
<p>This is exactly the LeetGPU #1 reference in <code>laneB-cuda/L1-launch-and-indexing/solutions/1-vector-addition/kernel.cu</code>. Its test runs N = 1, 7, 256, 257, 1000 and 2<sup>20</sup> + 3 to catch a missing guard.</p>`,
        check: { q: "N = 1,000 and 256 threads per block. How many blocks, and how many threads must the guard switch off?",
          options: ["3 blocks, none switched off", "4 blocks, 24 switched off", "4 blocks, 256 switched off"], answer: 1,
          why: "3 blocks cover only 768 elements, so we need 4 (ceil(1000/256) = 4). 4 × 256 = 1,024 threads, and the last 24 (indices 1,000 to 1,023) have no element." },
        scene(G) {
          G.text(24, 40, "N = 20 elements · 8 threads per block", { size: 14 });
          G.text(24, 64, "blocks = (20 + 7) / 8 = 3  →  24 threads", { size: 14, color: "k" });
          const th = [];
          for (let b = 0; b < 3; b++) {
            G.rect(20 + b * 200, 96, 192, 70, { stroke: "line", rx: 8 }); G.label(28 + b * 200, 114, `block ${b}`, { color: "ink" });
            for (let t = 0; t < 8; t++) { const i = b * 8 + t; th.push(G.rect(28 + b * 200 + t * 22, 124, 18, 32, { fill: i < 20 ? "k" : "hot", rx: 3 })); }
          }
          cells(G, 28, 210, 20, 18, 26, "ok", 4);
          for (let i = 0; i < 20; i += 4) G.label(37 + i * 22, 254, String(i), { anchor: "middle", size: 10 });
          G.label(28, 202, "the array: 20 elements", { size: 11 });
          G.rect(468, 210, 84, 26, { stroke: "hot", dash: "4 3", rx: 4 }); G.label(474, 254, "past the end", { color: "hot", size: 11 });
          G.arrow(530, 160, 510, 206, { color: "hot", dash: "3 3" });
          G.text(24, 310, "if (i < N)  // threads 20–23 do nothing", { size: 15, color: "ok" });
          G.label(24, 340, "without the guard they read and write memory that isn't yours", { size: 12 });
          G.from(th.slice(20), { opacity: 0, duration: 0.4, stagger: 0.1, delay: 0.3 });
          G.caption("ceil-div gives enough threads; the guard silences the extras");
        } },

      { rail: "warps", title: "Threads really run 32 at a time",
        body: `<p>The hardware does not schedule threads one at a time. It groups each block's threads into <b>warps</b> of 32 consecutive threads (threads 0–31, 32–63, …). A warp issues one instruction for all 32 lanes together. This is called SIMT: single instruction, multiple threads.</p>
<p>Two consequences for L1:</p>
<ul><li><b>Pick a block size that is a multiple of 32.</b> A 100-thread block becomes 4 warps (32 + 32 + 32 + 4), and the last warp runs with 28 of its 32 lanes idle. 128 or 256 waste nothing.</li>
<li><b>Branches can split a warp.</b> If some lanes take <code>if</code> and others take <code>else</code>, the warp runs both paths one after the other, switching lanes off for each. That is <b>divergence</b>.</li></ul>
<p>Short conditionals usually avoid it. Leaky ReLU's <code>x &gt; 0 ? x : alpha * x</code> compiles to a <b>select</b>: every lane computes both values and keeps one, with no branch. The bounds guard does diverge, but only in the last warp of the grid.</p>`,
        check: { q: "A block of 100 threads runs as how many warps, and how many lanes are idle?",
          options: ["3 warps, 4 lanes idle", "4 warps, 28 lanes idle", "4 warps, no lanes idle"], answer: 1,
          why: "100 = 3 × 32 + 4. The fourth warp holds only 4 threads, but it still occupies a full 32-lane warp, so 28 lanes do nothing. Block sizes such as 128 or 256 divide evenly into warps." },
        scene(G) {
          G.text(24, 36, "a block of 64 threads = 2 warps of 32 lanes", { size: 14 });
          const w1 = cells(G, 24, 52, 32, 15, 24, "k", 3), w2 = cells(G, 24, 84, 32, 15, 24, "blue", 3);
          G.label(24, 128, "warp 0: threads 0–31", { color: "k" }); G.label(330, 128, "warp 1: threads 32–63", { color: "blue" });
          G.text(24, 176, "a branch that splits one warp", { size: 14 });
          const lane = (i) => (i % 3 === 0 ? "v" : "q");
          cells(G, 24, 190, 32, 15, 22, lane, 3);
          G.rect(24, 236, 10, 10, { fill: "v", rx: 2 }); G.label(40, 245, "if path", { size: 11 });
          G.rect(110, 236, 10, 10, { fill: "q", rx: 2 }); G.label(126, 245, "else path", { size: 11 });
          G.label(24, 286, "time →", { size: 11 });
          G.rect(80, 272, 180, 22, { fill: "v", rx: 4 }); G.label(90, 288, "if path, others masked", { color: "bg", size: 11 });
          G.rect(266, 272, 180, 22, { fill: "q", rx: 4 }); G.label(276, 288, "else path, others masked", { color: "bg", size: 11 });
          G.label(456, 288, "= both paths, serially", { color: "hot", size: 11 });
          G.rect(80, 312, 180, 22, { fill: "ok", rx: 4 }); G.label(90, 328, "select: one pass", { color: "bg", size: 11 });
          G.label(270, 328, "x > 0 ? x : alpha * x", { color: "ok", size: 12 });
          G.text(24, 380, "100-thread block → 4 warps; the last runs 4 of 32 lanes", { size: 13, color: "hot" });
          G.from(w2, { opacity: 0, stagger: 0.02, duration: 0.2 }); G.from(w1, { opacity: 0, stagger: 0.02, duration: 0.2 });
          G.caption("a warp issues one instruction for 32 lanes");
        } },

      { rail: "grid-stride", title: "A grid-stride loop covers any N",
        body: `<p>One thread per element ties the grid size to N. Sometimes you want a fixed, smaller grid: enough blocks to fill the GPU, no more. Then each thread walks the array in jumps of the <b>whole grid's width</b>:</p>
<div class="eq">int stride = gridDim.x * blockDim.x;
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i &lt; N; i += stride)
    out[i] = f(in[i]);</div>
<p>Work an example. 4 blocks × 256 threads = 1,024 threads, N = 5,000. Thread 0 handles 0, 1,024, 2,048, 3,072 and 4,096: five elements. Thread 904 handles 904, 1,928, 2,952 and 3,976; its next would be 5,000, which fails <code>i &lt; N</code>, so it handles four.</p>
<p>The loop condition <i>is</i> the bounds guard. On every pass, consecutive threads still touch consecutive elements, so the memory access pattern stays good. The L1 elementwise references (ReLU, Leaky ReLU, Sigmoid) use this pattern.</p>`,
        scene(G) {
          const T = 8, N = 30, colr = ["k", "v", "q", "ok", "blue", "pink", "hot", "w"];
          G.text(24, 36, "8 threads (2 blocks × 4) · N = 30 · stride = 8", { size: 14 });
          const th = Array.from({ length: T }, (_, t) => G.box(24 + t * 74, 52, 64, 30, `t${t}`, { fill: colr[t], size: 12 }));
          const rows = [];
          for (let p = 0; p < 4; p++) {
            G.label(24, 128 + p * 54, `pass ${p}`, { size: 11 });
            for (let t = 0; t < T; t++) {
              const i = p * T + t;
              if (i >= N) { G.rect(24 + t * 74, 136 + p * 54, 64, 30, { stroke: "hot", dash: "4 3", rx: 4 }); continue; }
              rows.push(G.box(24 + t * 74, 136 + p * 54, 64, 30, String(i), { fill: colr[t], size: 12, boxOpacity: 0.85 }));
            }
          }
          G.label(24, 372, "pass 3: elements 30 and 31 don't exist → loop exits for t6, t7", { color: "hot", size: 12 });
          G.label(24, 394, "thread t handles t, t + 8, t + 16, …", { color: "ink", size: 12 });
          G.from(rows, { opacity: 0, stagger: 0.04, duration: 0.2 });
          G.caption("each pass, the whole grid steps forward by its own width");
        } },

      { rail: "row-major", title: "A matrix is one long row in memory",
        body: `<p>GPU memory is a flat list of bytes. A matrix with M rows and N columns has to be stored as one line of M·N numbers. The usual choice (C, C++, NumPy, PyTorch, LeetGPU) is <b>row-major</b>: row 0 first, then row 1, and so on.</p>
<div class="eq">element (row, col)  →  index row * N + col

M = 1000 rows, N = 37 columns
(2, 5)     → 2 * 37 + 5     = 79
(999, 36)  → 999 * 37 + 36  = 36,999   (the last)</div>
<p>Moving one step right adds 1 to the index. Moving one step down adds N: the <b>row stride</b>. Getting rows and columns swapped (<code>col * M + row</code>) is the classic L1 bug: for a square matrix it silently transposes, for a non-square one it reads garbage.</p>
<p>Use <code>size_t</code> for the multiply when M·N can exceed 2<sup>31</sup> − 1 (about 2.1 billion), as the Matrix Copy reference does: <code>B[size_t(row) * N + col]</code>.</p>`,
        scene(G) {
          const R = 4, C = 6, cw = 50, x0 = 140, y0 = 50;
          G.label(x0, y0 - 12, "4 × 6 matrix (M = 4, N = 6)", { color: "ink" });
          const hl = (r, c) => r === 2 && c === 3;
          for (let r = 0; r < R; r++) {
            G.label(x0 - 50, y0 + r * 40 + 24, `row ${r}`, { size: 11 });
            for (let c = 0; c < C; c++) G.box(x0 + c * cw, y0 + r * 40, cw - 4, 36, String(r * C + c), { fill: hl(r, c) ? "v" : ["k", "blue", "q", "ok"][r], size: 12, boxOpacity: hl(r, c) ? 1 : 0.75 });
          }
          G.text(24, 252, "in memory: row 0, then row 1, then row 2 …", { size: 13 });
          const flat = [];
          for (let i = 0; i < 24; i++) { const r = Math.floor(i / C); flat.push(G.rect(24 + i * 25, 268, 22, 30, { fill: i === 15 ? "v" : ["k", "blue", "q", "ok"][r], rx: 3, opacity: i === 15 ? 1 : 0.75 })); }
          for (let i = 0; i < 24; i += 6) G.label(24 + i * 25, 316, String(i), { size: 10 });
          G.text(24, 356, "(row 2, col 3) → 2 · 6 + 3 = 15", { size: 15, color: "v" });
          G.label(24, 384, "right: +1 · down: +N (the row stride)", { size: 12, color: "ink" });
          G.from(flat, { opacity: 0, x: -10, stagger: 0.03, duration: 0.2 });
          G.caption("row-major: index = row × N + col");
        } },

      { rail: "2D launch", title: "A 2D grid for a 2D problem",
        body: `<p>For matrices, give blocks and grids two dimensions with <code>dim3</code>. Each thread gets an x and a y coordinate, and each maps to one axis of the matrix:</p>
<div class="eq">dim3 block(16, 16);                  // 256 threads
dim3 grid((N + 15) / 16,             // x: columns
          (M + 15) / 16);            // y: rows
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row &lt; M &amp;&amp; col &lt; N) out[row * N + col] = …;</div>
<p><b>x must map to columns.</b> Within a warp, threadIdx.x changes fastest. With x → column, a warp's 32 lanes (two rows of 16 here) read consecutive floats, which the memory system merges into a few wide transactions. That merging is <b>coalescing</b>, and L2 is built on it. With x → row, each lane hits a different row, N·4 bytes apart.</p>
<p>The exit check uses a 1000 × 37 matrix. 37 columns need ceil(37/16) = 3 blocks in x (48 columns). 1,000 rows need ceil(1000/16) = 63 in y (1,008 rows). That is 48 × 1,008 = 48,384 threads for 37,000 elements: 11,384 (23.5%) fail the guard. Both halves of the guard are needed.</p>`,
        check: { q: "For a 1000-row × 37-column matrix with dim3 block(16, 16), x mapped to columns, what grid do you launch?",
          options: ["dim3 grid(63, 3)", "dim3 grid(3, 63)", "dim3 grid(1000, 37)"], answer: 1,
          why: "grid.x counts blocks along the columns: ceil(37/16) = 3. grid.y counts blocks down the rows: ceil(1000/16) = 63. dim3(63, 3) is the x → row mapping with the axes swapped." },
        scene(G) {
          const cw = 16, x0 = 40, y0 = 56, M = 13, N = 22, B = 8;
          G.text(24, 34, "M = 13 rows × N = 22 cols · block 8 × 8 → grid (3, 2)", { size: 14 });
          const gx = Math.ceil(N / B), gy = Math.ceil(M / B);
          const all = G.grid(x0, y0, gy * B, gx * B, cw, cw, (r, c) => (r < M && c < N ? "k" : "hot"), { gap: 2, rx: 2 });
          for (let bx = 0; bx <= gx; bx++) G.line(x0 + bx * B * cw - 1, y0 - 4, x0 + bx * B * cw - 1, y0 + gy * B * cw + 2, { color: "ink", w: 2 });
          for (let by = 0; by <= gy; by++) G.line(x0 - 4, y0 + by * B * cw - 1, x0 + gx * B * cw + 2, y0 + by * B * cw - 1, { color: "ink", w: 2 });
          // warp 0 of block (1,0): rows 0-3, cols 8-15
          const w = G.rect(x0 + 8 * cw - 1, y0 - 1, 8 * cw, 4 * cw, { stroke: "v", sw: 3, rx: 3 });
          G.label(440, 220, "amber box: one warp", { color: "v", size: 11 }); G.label(440, 236, "= 4 rows × 8 cols", { color: "v", size: 11 });
          G.arrow(436, 216, x0 + 16 * cw + 4, y0 + 2 * cw, { color: "v", dash: "3 3" });
          G.text(36, 340, "col = blockIdx.x · 8 + threadIdx.x", { size: 13, color: "ink" });
          G.text(36, 362, "row = blockIdx.y · 8 + threadIdx.y", { size: 13, color: "ink" });
          G.text(36, 390, "red cells: threads outside the matrix, stopped by the guard", { size: 13, color: "hot" });
          G.label(440, 80, "blockIdx.x →", { color: "ink" }); G.label(440, 100, "0 · 1 · 2", { size: 11 });
          G.label(440, 140, "blockIdx.y ↓", { color: "ink" }); G.label(440, 160, "0 · 1", { size: 11 });
          G.from(all, { opacity: 0, stagger: 0.002, duration: 0.15 }); G.pulse(w, { repeat: 6 });
          G.caption("thick lines are block edges; the grid overhangs the matrix");
        } },

      { rail: "3D", title: "Three dimensions: one more stride",
        body: `<p>Volumes (3D convolution, counting in a 3D array in L2 and L3) add a depth axis. Store slice 0 first, then slice 1, each slice being a row-major 2D matrix. With depth D, height H and width W:</p>
<div class="eq">(z, y, x)  →  (z * H + y) * W + x
            =  z * (H * W) + y * W + x

D = 4, H = 5, W = 6:
(2, 3, 4)  →  (2 * 5 + 3) * 6 + 4  = 82</div>
<p>Each axis has a stride: x moves by 1, y by W, z by H·W (one whole slice, 30 here). The launch uses <code>dim3 block(8, 8, 4)</code> and a 3D grid, with x still on the fastest-moving axis.</p>
<p>Know the limits: a block has at most 1,024 threads in total, and <code>gridDim.y</code> and <code>gridDim.z</code> are capped at 65,535 while <code>gridDim.x</code> can reach 2<sup>31</sup> − 1 (CUDA Programming Guide, technical specifications per compute capability). A tall 1D-shaped problem therefore belongs on x, or in a grid-stride loop.</p>`,
        scene(G) {
          const H = 3, W = 4, cw = 34;
          G.text(24, 34, "D = 3 slices · H = 3 · W = 4  → strides 12, 4, 1", { size: 14 });
          const sl = [];
          for (let z = 0; z < 3; z++) {
            const ox = 24 + z * 160, oy = 70;
            const g = G.group({});
            G.label(ox, oy - 8, `slice z = ${z}`, { size: 11, color: "ink", parent: g });
            for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
              const on = z === 1 && y === 2 && x === 1;
              G.box(ox + x * cw, oy + y * cw, cw - 4, cw - 4, String(z * 12 + y * 4 + x), { fill: on ? "v" : ["k", "blue", "q"][z], size: 11, parent: g, boxOpacity: on ? 1 : 0.85 });
            }
            sl.push(g);
          }
          G.text(24, 200, "(z, y, x) = (1, 2, 1)  →  1·12 + 2·4 + 1 = 21", { size: 14, color: "v" });
          G.text(24, 262, "in memory: slice 0, slice 1, slice 2", { size: 13 });
          const flat = [];
          for (let i = 0; i < 36; i++) flat.push(G.rect(24 + i * 16.5, 276, 14, 28, { fill: i === 21 ? "v" : ["k", "blue", "q"][Math.floor(i / 12)], rx: 2, opacity: i === 21 ? 1 : 0.85 }));
          [0, 12, 24].forEach((i) => G.label(24 + i * 16.5, 322, String(i), { size: 10 }));
          G.label(24, 360, "x: +1 · y: +W · z: +H·W", { color: "ink", size: 13 });
          G.from(sl, { opacity: 0, y: -14, stagger: 0.2, duration: 0.35 });
          G.caption("each extra dimension multiplies in one more stride");
        } },

      { rail: "layouts", title: "Pixels, pairs and mirrors: map the index to the layout",
        body: `<p>Several L1 problems keep the "one thread per item" rule but change what an item is and where it lives:</p>
<ul><li><b>RGBA pixels</b> (Color Inversion): 4 bytes per pixel, pixel p starts at byte 4p. One thread per <i>pixel</i>, read as one <code>uchar4</code>, flips R, G, B and leaves alpha alone.</li>
<li><b>Interleaved RGB floats</b> (RGB to Grayscale): pixel p's channels sit at 3p, 3p+1, 3p+2. This "array of structs" layout gives a stride of 3, but a warp's three loads still cover one contiguous span of 384 bytes.</li>
<li><b>Scatter</b> (Interleave Arrays): thread i reads A[i] and B[i] and writes out[2i] and out[2i+1]. The output index differs from the input index.</li>
<li><b>In-place mirror</b> (Reverse Array): if all N threads swapped a[i] with a[N−1−i], each pair would be swapped twice (by i and by N−1−i) and the two threads would race. Launch only N/2 threads: each pair has exactly one owner. For odd N the middle element stays put.</li></ul>
<p>The question each time is the same: what does one thread own, and what are the input and output addresses of that thing?</p>`,
        check: { q: "Reverse an array of N = 7 in place. How many threads should do swaps?",
          options: ["7: one per element", "3: one per pair, the middle stays", "4: round up"], answer: 1,
          why: "Each swap touches two elements, so one thread must own each pair. 7 / 2 = 3 in integer division: pairs (0,6), (1,5), (2,4). Element 3 is already in place. A fourth thread would swap 3 with itself, and 7 threads would swap every pair twice and race." },
        scene(G) {
          G.text(24, 34, "RGBA: thread p owns bytes 4p … 4p+3", { size: 13 });
          for (let p = 0; p < 4; p++) ["R", "G", "B", "A"].forEach((ch, j) => G.box(24 + p * 150 + j * 34, 46, 30, 26, ch, { fill: j === 3 ? "w" : ["hot", "ok", "blue"][j], size: 11 }));
          for (let p = 0; p < 4; p++) G.label(24 + p * 150, 90, `pixel ${p}`, { size: 10 });
          G.text(24, 130, "interleave: out[2i] = A[i], out[2i+1] = B[i]", { size: 13 });
          cells(G, 24, 142, 4, 40, 24, "k", 6); cells(G, 220, 142, 4, 40, 24, "v", 6);
          G.label(410, 160, "A (cyan) and B (amber)", { size: 11 }); G.label(410, 218, "out", { size: 11 });
          const out = cells(G, 24, 200, 8, 40, 24, (i) => (i % 2 ? "v" : "k"), 6);
          for (let i = 0; i < 4; i++) { G.line(44 + i * 46, 166, 44 + 2 * i * 46, 200, { color: "k", w: 1 }); G.line(240 + i * 46, 166, 44 + (2 * i + 1) * 46, 200, { color: "v", w: 1 }); }
          G.text(24, 268, "reverse in place, N = 7: three swappers", { size: 13 });
          const arr = cells(G, 24, 326, 7, 54, 30, (i) => (i === 3 ? "w" : "k"), 8);
          for (let i = 0; i < 7; i++) G.label(51 + i * 62, 374, String(i), { anchor: "middle", size: 11 });
          const arcs = [0, 1, 2].map((i) => G.path(`M ${51 + i * 62} 324 C ${51 + i * 62} ${296 - (3 - i) * 4}, ${51 + (6 - i) * 62} ${296 - (3 - i) * 4}, ${51 + (6 - i) * 62} 324`, { color: ["v", "q", "ok"][i], w: 2 }));
          G.label(24, 404, "thread i swaps a[i] ↔ a[N−1−i] for i < N/2 = 3; a[3] stays", { color: "ink", size: 12 });
          G.from(out, { opacity: 0, stagger: 0.08, duration: 0.2 }); G.from(arcs, { opacity: 0, stagger: 0.2, duration: 0.3 });
          G.caption("decide what one thread owns, then compute its addresses");
        } },

      { rail: "host side", title: "Around the kernel: memory, launches, errors, bytes",
        body: `<p>On LeetGPU, <code>solve()</code> receives pointers that already live in GPU memory. In the local harness, and in real code, the CPU (the <b>host</b>) does the setup:</p>
<div class="eq">cudaMalloc(&amp;dA, bytes);                 // GPU memory
cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
add&lt;&lt;&lt;blocks, 256&gt;&gt;&gt;(dA, dB, dC, N);  // async
CUDA_CHECK(cudaGetLastError());     // bad config?
CUDA_CHECK(cudaDeviceSynchronize()); // kernel fault?
cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);</div>
<p>A launch is <b>asynchronous</b>: the CPU queues it and moves on. A wrong configuration (say 2,048 threads per block, over the 1,024 limit) is reported by <code>cudaGetLastError</code>. A fault inside the kernel (an out-of-bounds write) only shows up at the next synchronizing call. The harness macro <code>CUDA_CHECK_LAUNCH()</code> in <code>laneB-cuda/harness/include/s2s_cuda.cuh</code> does both checks. To find which thread wrote out of bounds, run the test under <code>compute-sanitizer</code>.</p>
<p><b>Count the bytes.</b> Vector add reads 2 floats and writes 1: 12 bytes for 1 addition. No GPU can do that little arithmetic per byte fast enough to matter, so every L1 elementwise kernel is <b>memory-bound</b>: its time is about bytes ÷ bandwidth. For N = 2<sup>26</sup>: 12 × 67,108,864 = 805 MB. At 300 GB/s (example value: check your GPU's spec sheet) that is 805 MB ÷ 300 GB/s ≈ 2.7 ms. The harness's <code>--bench</code> reports your kernel's GB/s next to a measured copy kernel, which is the realistic ceiling.</p>`,
        scene(G) {
          G.box(24, 40, 220, 120, "", { stroke: "muted" }); G.text(36, 64, "host (CPU) memory", { size: 13 });
          G.box(396, 40, 220, 120, "", { stroke: "k" }); G.text(408, 64, "device (GPU) memory", { size: 13, color: "k" });
          ["hA", "hB", "hC"].forEach((n, i) => G.box(36 + i * 68, 90, 58, 50, n, { fill: "w", size: 12 }));
          ["dA", "dB", "dC"].forEach((n, i) => G.box(408 + i * 68, 90, 58, 50, n, { fill: i < 2 ? "k" : "ok", size: 12 }));
          const a1 = G.arrow(250, 92, 390, 92, { color: "k", w: 2 }); G.label(262, 84, "cudaMemcpy H→D", { size: 11, color: "k" });
          const a2 = G.arrow(390, 136, 250, 136, { color: "ok", w: 2 }); G.label(262, 156, "cudaMemcpy D→H", { size: 11, color: "ok" });
          G.text(24, 210, "timeline", { size: 13 });
          G.label(24, 236, "CPU", { size: 11 }); G.label(24, 270, "GPU", { size: 11 });
          G.rect(70, 224, 70, 18, { fill: "k", rx: 3 }); G.rect(146, 224, 26, 18, { fill: "v", rx: 3 }); G.label(176, 238, "launch returns", { size: 10, color: "v" });
          G.rect(276, 224, 124, 18, { stroke: "muted", dash: "3 3", rx: 3 }); G.label(282, 238, "synchronize: wait", { size: 10 });
          const k = G.rect(160, 258, 240, 18, { fill: "ok", rx: 3 }); G.label(166, 272, "kernel runs", { size: 10, color: "bg" });
          G.rect(406, 224, 70, 18, { fill: "ok", rx: 3 }); G.label(482, 238, "copy back", { size: 10 });
          G.text(24, 330, "vector add: 2 reads + 1 write = 12 B per element", { size: 13 });
          G.text(24, 354, "for 1 add  →  memory-bound", { size: 13, color: "hot" });
          G.text(24, 386, "time ≈ 12 · N bytes ÷ memory bandwidth", { size: 14, color: "ink" });
          G.from(k, { attr: { width: 0 }, duration: 0.8, delay: 0.2 }); G.from([a1, a2], { opacity: 0, stagger: 0.3, duration: 0.3 });
          G.caption("launches are asynchronous; errors surface at the next sync");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A kernel is a loop body; each thread finds its element with <code>blockIdx.x * blockDim.x + threadIdx.x</code>.</li>
<li>Round the grid up with <code>(N + T − 1) / T</code>, then guard with <code>if (i &lt; N)</code>. A grid-stride loop makes the guard the loop condition.</li>
<li>Threads run as warps of 32: use block sizes that are multiples of 32 and prefer selects to divergent branches.</li>
<li>Row-major means <code>row * N + col</code> (2D) and <code>(z * H + y) * W + x</code> (3D). Map threadIdx.x to the fastest-moving axis.</li>
<li>Launches are asynchronous: check <code>cudaGetLastError</code> and synchronize. L1 kernels are memory-bound, so count bytes per element.</li></ul>
<p>Next, L2 asks what happens when threads read the <i>same</i> data many times, and how shared memory lets them read it once.</p>`,
    sim: {
      title: "2D launch and index calculator",
      intro: "Choose a matrix shape, a block shape and which matrix axis threadIdx.x walks. The chart shows the grid of blocks laid over the matrix (red = threads the guard switches off) and the 32 elements touched by the warp that owns the element you pick. The panels give that thread's coordinates and the launch's waste.",
      height: 360,
      controls: [
        { id: "M", label: "rows M", min: 1, max: 96, value: 37 },
        { id: "N", label: "columns N", min: 1, max: 96, value: 50 },
        { id: "bx", label: "blockDim.x", type: "select", value: 16, options: [[4, "4"], [8, "8"], [16, "16"], [32, "32"]] },
        { id: "by", label: "blockDim.y", type: "select", value: 16, options: [[1, "1"], [2, "2"], [4, "4"], [8, "8"], [16, "16"], [32, "32"]] },
        { id: "map", label: "threadIdx.x walks", type: "select", value: "col", options: [["col", "columns (good)"], ["row", "rows (bad)"]] },
        { id: "r", label: "pick element: row (clamped to M − 1)", min: 0, max: 95, value: 20 },
        { id: "c", label: "pick element: col (clamped to N − 1)", min: 0, max: 95, value: 21 },
      ],
      draw(G, v) {
        const M = v.M, N = v.N, bx = v.bx, by = v.by, xcol = v.map === "col";
        if (bx * by > 1024) {
          G.text(20, 60, `blockDim ${bx} × ${by} = ${bx * by} threads: over the 1,024 limit.`, { color: "hot", size: 15 });
          G.text(20, 90, "The launch would fail with an invalid-configuration error.", { size: 13 });
          return [{ title: "Launch", rows: [["threads per block", F.num(bx * by)]], chip: [false, "invalid configuration"] }];
        }
        const r = Math.min(v.r, M - 1), c = Math.min(v.c, N - 1);
        const bR = xcol ? by : bx, bC = xcol ? bx : by;          // block extent in rows / cols
        const gR = Math.ceil(M / bR), gC = Math.ceil(N / bC);     // blocks along rows / cols
        const covR = gR * bR, covC = gC * bC;
        const cs = Math.max(2, Math.min(560 / covC, 320 / covR));
        const x0 = 40, y0 = 20;
        G.rect(x0 + N * cs, y0, (covC - N) * cs, covR * cs, { fill: "hot", opacity: 0.3, rx: 0 });
        G.rect(x0, y0 + M * cs, N * cs, (covR - M) * cs, { fill: "hot", opacity: 0.3, rx: 0 });
        G.rect(x0, y0, N * cs, M * cs, { fill: "k", opacity: 0.3, rx: 0 });
        if (cs >= 5) {
          for (let i = 0; i <= covC; i++) if (i % bC) G.line(x0 + i * cs, y0, x0 + i * cs, y0 + covR * cs, { color: "line", w: 0.5 });
          for (let i = 0; i <= covR; i++) if (i % bR) G.line(x0, y0 + i * cs, x0 + covC * cs, y0 + i * cs, { color: "line", w: 0.5 });
        }
        for (let i = 0; i <= gC; i++) G.line(x0 + i * bC * cs, y0, x0 + i * bC * cs, y0 + covR * cs, { color: "ink", w: 1.2 });
        for (let i = 0; i <= gR; i++) G.line(x0, y0 + i * bR * cs, x0 + covC * cs, y0 + i * bR * cs, { color: "ink", w: 1.2 });
        // the selected thread
        const xc = xcol ? c : r, yc = xcol ? r : c;
        const bix = Math.floor(xc / bx), tx = xc % bx, biy = Math.floor(yc / by), ty = yc % by;
        const lin = ty * bx + tx, warp = Math.floor(lin / 32);
        const rowsTouched = new Set(); let inb = 0, minI = Infinity, maxI = -1;
        for (let l = warp * 32; l < Math.min(warp * 32 + 32, bx * by); l++) {
          const ltx = l % bx, lty = Math.floor(l / bx);
          const ex = bix * bx + ltx, ey = biy * by + lty;
          const er = xcol ? ey : ex, ec = xcol ? ex : ey;
          const ok = er < M && ec < N;
          G.rect(x0 + ec * cs, y0 + er * cs, cs, cs, { fill: ok ? "v" : "hot", rx: 0, opacity: 0.95 });
          if (ok) { inb++; rowsTouched.add(er); const f = er * N + ec; minI = Math.min(minI, f); maxI = Math.max(maxI, f); }
        }
        G.rect(x0 + c * cs - 1, y0 + r * cs - 1, cs + 2, cs + 2, { stroke: "ink", sw: 2, rx: 0 });
        G.label(x0, y0 + covR * cs + 16, `${M} × ${N} matrix · ${bx} × ${by} blocks · amber = the picked thread's warp`, { size: 11 });
        const total = covR * covC, idle = total - M * N;
        const gx = xcol ? gC : gR, gy = xcol ? gR : gC;
        return [
          { title: "Launch", rows: [["dim3 block", `(${bx}, ${by})`], ["dim3 grid", `(${gx}, ${gy})`], ["threads launched", F.num(total)], ["elements", F.num(M * N)], ["guarded off", `${F.num(idle)} (${((100 * idle) / total).toFixed(1)}%)`]],
            gauge: [[(M * N) / total, "k"], [idle / total, "hot"]], gaugeText: "useful vs idle threads" },
          { title: `Element (${r}, ${c})`, rows: [["blockIdx", `(${bix}, ${biy})`], ["threadIdx", `(${tx}, ${ty})`], ["row-major index", `${r}·${N} + ${c} = ${F.num(r * N + c)}`], ["warp in block", `${warp} (lanes ${warp * 32}–${warp * 32 + 31})`]] },
          { title: "That warp's memory access", rows: [["lanes in bounds", `${inb} / 32`], ["distinct rows touched", String(rowsTouched.size)], ["address span (floats)", inb ? F.num(maxI - minI + 1) : "–"]],
            chip: [rowsTouched.size <= 2 && inb > 0, rowsTouched.size <= 2 ? "contiguous: coalesces well" : "scattered across rows"],
            html: `<p class="note">Rows touched approximates separate memory segments. Set threadIdx.x to walk rows, or blockDim.x to 4, and watch it grow.</p>` },
        ];
      },
      note: "Limits used: 1,024 threads per block. The picture caps the matrix at 96 × 96 so cells stay visible.",
    },
    practice: {
      intro: `<p>Solve each problem on <a href="https://leetgpu.com/challenges">LeetGPU</a> first (T1, in the browser). We don't reproduce the statements; read them there. Then run the local reference and test (T2, needs an NVIDIA GPU and CUDA ≥ 12.4). Build once from the repository root:</p>
<p><code>cmake -S laneB-cuda -B build/laneB -DCMAKE_BUILD_TYPE=Release &amp;&amp; cmake --build build/laneB -j</code></p>
<p>Each command below runs one test; add <code>--bench</code> by running the binary directly, for example <code>./build/laneB/L1_1-vector-addition --bench</code>. Order: the nine core problems top to bottom, then practice and stretch.</p>`,
      items: [
        P("1-vector-addition", "Vector Addition", "easy", "core", "Global index, ceil-div grid and the <code>i &lt; N</code> guard; error-check every launch."),
        P("21-relu", "ReLU", "easy", "core", "An elementwise map in a grid-stride loop; <code>fmaxf</code>."),
        P("23-leaky-relu", "Leaky ReLU", "easy", "core", "A select instead of a divergent branch; a scalar kernel argument."),
        P("7-color-inversion", "Color Inversion", "easy", "core", "One thread per RGBA pixel through a <code>uchar4</code> view; leave alpha alone."),
        P("19-reverse-array", "Reverse Array", "easy", "core", "In-place mirror with N/2 threads so each pair has one owner."),
        P("31-matrix-copy", "Matrix Copy", "easy", "core", "2D launch with x on columns; compare the coalesced and swapped mappings in <code>--bench</code>."),
        P("8-matrix-addition", "Matrix Addition", "easy", "core", "2D elementwise with both bounds checked."),
        P("66-rgb-to-grayscale", "RGB to Grayscale", "easy", "core", "Interleaved 3-channel layout: pixel p reads 3p, 3p+1, 3p+2."),
        P("62-value-clipping", "Value Clipping", "easy", "practice", "A branchless clamp."),
        P("63-interleave-arrays", "Interleave Arrays", "easy", "practice", "A scatter with stride 2, then one <code>float2</code> store per thread."),
        P("68-sigmoid", "Sigmoid Activation", "easy", "practice", "<code>expf</code> versus <code>__expf</code> accuracy; still memory-bound."),
        P("52-silu", "Sigmoid Linear Unit", "easy", "practice", "A fused activation, <code>x · sigmoid(x)</code>, in one pass."),
        P("2-matrix-multiplication", "Matrix Multiplication (naive)", "easy", "core · exit check", "One thread per output with a 2D grid for any M × N × K. The exit check: write it unaided for a non-square shape."),
        P("41-simple-inference", "Simple Inference", "easy", "stretch", "A linear layer, <code>x · Wᵀ + b</code>: the naive matmul plus a bias."),
        P("24-rainbow-table", "Rainbow Table", "easy", "stretch", "A per-thread loop of integer hashing: the first compute-bound problem."),
      ],
      labs: [
        { label: "Level syllabus, objectives and exit check", path: `${LV}/syllabus.md` },
        { label: "Problem map with links and priorities", path: `${LV}/leetgpu-map.md` },
        { label: "The harness: CUDA_CHECK, DeviceBuffer, time_gpu, measure_copy_gbs", path: "laneB-cuda/harness/" },
        { label: "Grid → blocks → warps animation (Lane A P5.1)", path: "animations/p5-grid-warps-sm.html" },
        { label: "AWS T2 box for L1–L4 (T4, sm_75): instance_type = \"g4dn.xlarge\", then make down", path: "infra/aws/single-node" },
        { label: "Optional problems off the inference path", path: "laneB-cuda/electives.md" },
      ],
    },
  });
})();
