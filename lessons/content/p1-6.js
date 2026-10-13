/* P1.6 — Your first CUDA program: host and device, toolchain, a kernel, the five-step program, ceil-div,
   asynchronous launches and errors, debugging, honest timing, and the transfer tax. */
(function () {
  const F = S2S.fmt;
  const MOD = "course/P1-inference-fundamentals/P1.6-first-cuda-program";
  // a horizontal strip of n cells starting at x
  const strip = (G, x, y, n, cw, ch, fill, o = {}) => G.grid(x, y, 1, n, cw, ch, (r, c) => fill(c), o);
  // inline code inside a Practice goal (the shell styles <code> there as a block)
  const mono = (h) => h.replace(/<code>(.*?)<\/code>/g, '<span style="font-family:var(--f-mono);font-size:13px">$1</span>');

  S2S.lesson({
    id: "p1-6", n: "P1.6", title: "Your first CUDA program",
    subtitle: "Inference fundamentals · first principles · T1 browser or Colab for the kernel, T2 for the full program",
    kicker: "Lesson · ≈ 45 min",
    headline: "Two computers, one slow wire, five steps",
    intro: `<p>P1.4 showed you what a GPU is. Now you write for one. A CUDA program is really <b>two programs in one file</b>: code that runs on the CPU (the <b>host</b>) and code that runs on the GPU (the <b>device</b>). The CPU is in charge. It sets up memory on the GPU, copies data over, tells the GPU what to run, and copies the answer back.</p>
<p>This lesson builds your first complete program for the sum of two arrays. It covers where to get a GPU, how <code>nvcc</code> compiles the file, what a kernel is, the five host steps, the grid-size arithmetic, why errors arrive late, how to debug, how to time it honestly, and why the copies can cost far more than the kernel. It assumes P1.4 and the pointer and heap lessons R.1 and R.2. After it, Lane B L1 drills the indexing, and P5 goes deep on speed.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["w", "host (CPU, RAM)"], ["k", "device (GPU)"], ["v", "highlighted"], ["hot", "cost / error"], ["ok", "fine"]],
    prev: "p1-5", next: "p2-1",
    steps: [
      { rail: "host & device", title: "Two processors, two memories, one slow link",
        body: `<p>Your machine has a CPU with its RAM. A GPU card is a second computer inside it, with its own processor and its own memory. In CUDA words the CPU side is the <b>host</b> and the GPU side is the <b>device</b>. The two are joined by a bus called <b>PCIe</b>.</p>
<p>The CPU runs <code>main()</code>. It decides what the GPU does. The GPU only runs what it is handed. Two facts follow:</p>
<ul><li><b>A GPU pointer is not a CPU pointer.</b> An address from <code>cudaMalloc</code> points into <i>GPU</i> memory. The CPU must never read or write through it. It can only pass it to <code>cudaMemcpy</code> and to kernels.</li>
<li><b>PCIe is slow compared with GPU memory.</b> Example values (check your platform): PCIe 3.0 x16 is about 15.75 GB/s in theory (8 GT/s × 16 lanes × 128/130 ÷ 8 bits), and 10 to 13 GB/s is common in practice. A T4's memory is about 320 GB/s in P1.4's <code>gpu_specs.yaml</code> (UNVERIFIED). Every byte that crosses PCIe costs about 20 to 27 times a byte inside the GPU.</li></ul>
<p>So the rule for fast GPU code is: send data over once, do a lot of work on it there, and bring back only the result. Step 9 puts numbers on it.</p>`,
        check: { q: "After <code>float* d_x; cudaMalloc(&amp;d_x, bytes);</code> a colleague writes <code>d_x[0] = 1.0f;</code> in host code. What is wrong?",
          options: ["Nothing: cudaMalloc memory is shared by CPU and GPU", "d_x points into GPU memory, which the CPU cannot access through a normal pointer; copy with cudaMemcpy instead", "It is fine, but slower than cudaMemcpy"], answer: 1,
          why: "cudaMalloc allocates in the device's own memory. The address is only meaningful to the GPU. Host code dereferencing it crashes or reads garbage. Data crosses the link only through cudaMemcpy (or a kernel that writes it)." },
        scene(G) {
          G.rect(16, 44, 196, 188, { stroke: "w", rx: 10 }); G.label(28, 62, "HOST", { color: "w" });
          G.box(30, 76, 168, 46, "CPU", { fill: "w", size: 14 });
          G.box(30, 140, 168, 70, "RAM (host memory)", { fill: "w", size: 13, opacity: 0.7 });
          G.rect(428, 44, 196, 188, { stroke: "k", rx: 10 }); G.label(440, 62, "DEVICE", { color: "k" });
          G.box(442, 76, 168, 46, "GPU", { fill: "k", size: 14 });
          G.box(442, 140, 168, 70, "GPU memory", { fill: "k", size: 13, opacity: 0.7 });
          const a1 = G.arrow(218, 138, 422, 138, { color: "hot", w: 2.5 });
          const a2 = G.arrow(422, 162, 218, 162, { color: "hot", w: 2.5 });
          G.text(320, 120, "PCIe", { size: 14, color: "hot", anchor: "middle" });
          G.label(320, 186, "the slow link", { size: 12, anchor: "middle" });
          G.text(24, 270, "bandwidth, GB/s (example values, UNVERIFIED)", { size: 13 });
          const b1 = G.rect(24, 284, 480, 22, { fill: "k", rx: 4 });
          G.label(32, 300, "GPU memory ≈ 320 (T4, gpu_specs.yaml)", { size: 12, color: "bg" });
          const b2 = G.rect(24, 316, 23.6, 22, { fill: "hot", rx: 4 });
          G.label(54, 332, "PCIe 3.0 x16 ≈ 15.75 in theory", { size: 12, color: "hot" });
          G.label(24, 374, "one byte over PCIe costs about 20× a byte in GPU memory", { size: 13, color: "ink" });
          G.label(24, 396, "(320 ÷ 15.75 ≈ 20 in theory; 320 ÷ 12 ≈ 27 at 12 GB/s)", { size: 12 });
          G.from([b1, b2], { attr: { width: 0 }, stagger: 0.25, duration: 0.5 });
          G.from([a1, a2], { opacity: 0, duration: 0.4, delay: 0.2 });
          G.caption("the CPU drives; data crosses a link far slower than the GPU's own memory");
        } },

      { rail: "get a GPU", title: "Three ways to get a GPU, and one command to check it",
        body: `<p>You need an NVIDIA GPU to run CUDA. You have three options. Pick the cheapest one that does what you need today.</p>
<table><tr><th></th><th>What you can do</th><th>Cost</th></tr>
<tr><td><b>LeetGPU</b> (browser, T1)</td><td>Write and run a kernel. The site supplies the host side. Good for the first kernel and for Lane B.</td><td>Free tier: check the site</td></tr>
<tr><td><b>Google Colab</b> (notebook, T1)</td><td>A GPU runtime. Write a <code>.cu</code> file in a cell and run <code>!nvcc …</code>. The GPU you get varies, so run <code>!nvidia-smi</code> first.</td><td>Free tier: check current limits</td></tr>
<tr><td><b>AWS g4dn.xlarge</b> (T2)</td><td>A real machine with one T4 (sm_75). Everything in this course. Set up as in P5's <code>aws.md</code>.</td><td>hours × hourly price (example value: check the price page)</td></tr></table>
<p>Whichever you use, the first check is the same:</p>
<div class="eq">nvidia-smi         # GPU name, driver, memory in use
nvcc --version     # the CUDA compiler (the toolkit)</div>
<p><code>nvidia-smi</code> talks to the <i>driver</i>. <code>nvcc</code> is part of the <i>toolkit</i>. They are separate installs, so one can work while the other is missing. If <code>nvidia-smi</code> fails, there is no GPU or no driver, and nothing else will work. On AWS, the cost is hours × price, so use the auto-stop and teardown steps in the shared guide.</p>`,
        scene(G) {
          const cards = [
            { x: 16, t: "LeetGPU", tier: "T1", c: "w", rows: [["write a kernel", true], ["write host code", false], ["run nvcc yourself", false]], foot: "browser · free tier" },
            { x: 224, t: "Colab", tier: "T1", c: "blue", rows: [["write a kernel", true], ["write host code", true], ["run nvcc in a cell", true]], foot: "GPU type varies" },
            { x: 432, t: "AWS g4dn.xlarge", tier: "T2", c: "k", rows: [["write a kernel", true], ["write host code", true], ["sanitizer + Nsight", true]], foot: "hours × hourly price" },
          ];
          const all = [];
          cards.forEach((c) => {
            all.push(G.rect(c.x, 44, 192, 190, { stroke: c.c, rx: 10, sw: 2 }));
            G.text(c.x + 12, 70, c.t, { size: 14, color: c.c });
            G.box(c.x + 148, 52, 34, 22, c.tier, { fill: c.c, size: 11, rx: 5 });
            c.rows.forEach(([l, y], i) => {
              G.circle(c.x + 20, 108 + i * 34, 7, { fill: y ? "ok" : "line" });
              G.label(c.x + 36, 112 + i * 34, l, { size: 12, color: y ? "ink" : "muted" });
            });
            G.label(c.x + 12, 222, c.foot, { size: 11 });
          });
          G.text(24, 276, "first check, on any of them:", { size: 13 });
          G.rect(24, 288, 592, 70, { stroke: "line", rx: 8 });
          G.text(38, 312, "$ nvidia-smi", { size: 14, color: "ok" });
          G.label(250, 312, "→ GPU name, driver version, memory", { size: 12 });
          G.text(38, 342, "$ nvcc --version", { size: 14, color: "ok" });
          G.label(250, 342, "→ the CUDA compiler, from the toolkit", { size: 12 });
          G.label(24, 392, "green dot = you can do it there", { size: 12 });
          G.from(all, { opacity: 0, y: 14, stagger: 0.15, duration: 0.4 });
          G.caption("start in the browser; use a real GPU box for the full program");
        } },

      { rail: "toolchain", title: "nvcc splits your file: host code and device code",
        body: `<p>A CUDA source file ends in <code>.cu</code>. It holds ordinary C++ (the host part) and <code>__global__</code> functions (the device part). A normal C++ compiler cannot read the GPU code, so you compile with <b><code>nvcc</code></b>, which does two jobs:</p>
<ol><li>It sends the <b>host code</b> to your usual C++ compiler (g++ or clang).</li>
<li>It compiles the <b>device code</b> to <b>PTX</b>, a virtual GPU instruction set, and then to <b>SASS</b>, the real machine code of one GPU generation.</li></ol>
<p>Both parts are packed into one executable. At run time the host part runs on the CPU and loads the device part onto the GPU.</p>
<div class="eq">nvcc -arch=sm_75 -lineinfo -o add add.cu</div>
<p><code>-arch=sm_75</code> picks the GPU generation. <b>Compute capability</b> is the version number of a GPU generation's features: 7.5 means <code>sm_75</code>, and a T4 is 7.5. Build for the GPU you will run on. The course's CMake files use <code>CMAKE_CUDA_ARCHITECTURES=native</code>, which asks the machine. <code>-lineinfo</code> keeps source line numbers for the debugging tools in step 8. <code>examples/01_device_info.cu</code> prints your GPU's capability.</p>`,
        scene(G) {
          G.box(16, 168, 112, 64, ".cu file", { stroke: "ink", size: 14, rx: 8 });
          G.box(160, 168, 112, 64, "nvcc", { fill: "v", size: 15, rx: 8 });
          G.arrow(130, 200, 158, 200, { color: "ink", w: 2 });
          const top = G.box(310, 70, 204, 74, "host code", { fill: "w", size: 14, rx: 8 });
          const bot = G.box(310, 256, 204, 74, "device code", { fill: "k", size: 14, rx: 8 });
          G.arrow(272, 190, 306, 120, { color: "w", w: 2 });
          G.arrow(272, 212, 306, 270, { color: "k", w: 2 });
          G.label(310, 62, "sent to g++ / clang", { size: 12, color: "w" });
          G.label(310, 160, "→ CPU machine code", { size: 12 });
          G.label(310, 248, "→ PTX (virtual) → SASS for sm_75", { size: 12, color: "k" });
          G.box(534, 168, 90, 64, "one program", { fill: "ok", size: 12, rx: 8 });
          G.arrow(516, 112, 560, 164, { color: "ink", w: 1.5 });
          G.arrow(516, 292, 560, 236, { color: "ink", w: 1.5 });
          G.text(24, 372, "-arch=sm_75 : build for compute capability 7.5 (T4)", { size: 13, color: "v" });
          G.label(24, 396, "CMake: CMAKE_CUDA_ARCHITECTURES=native asks the machine", { size: 12 });
          G.from([top, bot], { opacity: 0, x: -20, stagger: 0.25, duration: 0.4 });
          G.caption("one file in, host and device machine code out");
        } },

      { rail: "a kernel", title: "A kernel is a function that every thread runs",
        body: `<p>A <b>kernel</b> is a function marked <code>__global__</code>. The host launches it, and the GPU runs one copy per <b>thread</b>. It returns <code>void</code>. You write it for <i>one</i> element, and the launch creates the threads.</p>
<div class="eq">__global__ void hello(int* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = i;
}
hello&lt;&lt;&lt;2, 4&gt;&gt;&gt;(d_out);   // 2 blocks of 4 threads</div>
<p>The <code>&lt;&lt;&lt;blocks, threads&gt;&gt;&gt;</code> part is the launch. It says how many <b>blocks</b>, and how many threads per block. Threads are grouped in blocks (up to 1,024 threads each, from the CUDA Programming Guide). Inside the kernel, three built-in values say where this thread is (1D only here):</p>
<ul><li><code>threadIdx.x</code>: position inside my block, from 0.</li>
<li><code>blockIdx.x</code>: which block I am in, from 0.</li>
<li><code>blockDim.x</code>: threads per block (4 here).</li></ul>
<p>Put the blocks end to end and every thread gets a unique <b>global index</b>: <code>i = blockIdx.x * blockDim.x + threadIdx.x</code>. For block 1, thread 2: 1 × 4 + 2 = 6. This is exercise 1. The same code runs in all 8 threads, and only <code>i</code> differs. L1 covers 2D and 3D.</p>`,
        scene(G) {
          const cw = 34, x0 = 40, hl = [];
          G.text(24, 36, "hello<<<2, 4>>>(d_out): 2 blocks × 4 threads", { size: 14 });
          for (let b = 0; b < 2; b++) {
            const x = x0 + b * 290;
            G.rect(x - 8, 56, 4 * (cw + 4) + 8, 92, { stroke: b === 1 ? "v" : "line", rx: 8, sw: b === 1 ? 2 : 1 });
            G.label(x, 74, "block " + b, { size: 12, color: b === 1 ? "v" : "muted" });
            for (let t = 0; t < 4; t++) {
              const on = b === 1 && t === 2;
              const r = G.box(x + t * (cw + 4), 86, cw, 36, "t" + t, { fill: on ? "v" : "k", size: 12, rx: 5 });
              if (on) hl.push(r);
            }
            G.label(x, 140, "threadIdx.x = 0 … 3", { size: 10 });
          }
          G.label(24, 176, "the array out[0 … 7], one cell per global index:", { size: 12 });
          const cells = strip(G, 40, 190, 8, 50, 34, (c) => (c === 6 ? "v" : c < 4 ? "w" : "q"), { gap: 6 });
          for (let i = 0; i < 8; i++) G.label(40 + i * 56 + 25, 242, String(i), { size: 12, anchor: "middle" });
          G.arrow(x0 + 290 + 2 * (cw + 4) + cw / 2, 124, 40 + 6 * 56 + 25, 186, { color: "v", w: 2 });
          G.text(24, 290, "i = blockIdx.x · blockDim.x + threadIdx.x", { size: 14 });
          G.text(24, 320, "block 1, thread 2:  i = 1 · 4 + 2 = 6", { size: 15, color: "v" });
          G.label(24, 358, "same code in all 8 threads; only i differs", { size: 13, color: "ink" });
          G.label(24, 382, "out[i] = i  →  the array becomes 0 1 2 3 4 5 6 7", { size: 13 });
          G.pulse(hl, { repeat: 6 }); G.from(cells, { opacity: 0, stagger: 0.04, duration: 0.2 });
          G.caption("the launch makes the threads; each finds its own element from its ids");
        } },

      { rail: "five steps", title: "The host program: allocate, copy in, launch, copy out, free",
        body: `<p>Every CUDA program does the same five things, in this order. The example adds two arrays, <code>c = a + b</code>, with <code>n</code> floats each:</p>
<div class="eq">size_t bytes = n * sizeof(float);
float *d_a, *d_b, *d_c;           // device pointers
// 1 allocate
cudaMalloc(&amp;d_a, bytes);
cudaMalloc(&amp;d_b, bytes);
cudaMalloc(&amp;d_c, bytes);
// 2 copy in
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
// 3 launch
add_kernel&lt;&lt;&lt;blocks, 256&gt;&gt;&gt;(d_a, d_b, d_c, n);
// 4 copy out
cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
// 5 free
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);</div>
<p>Three details trip everyone once. <code>cudaMalloc</code> takes <code>&amp;d_a</code> (a pointer to your pointer) because it must write the new address <i>into</i> <code>d_a</code>. <code>cudaMemcpy(dst, src, bytes, kind)</code> puts the <b>destination first</b> and counts <b>bytes</b>, not elements. And every <code>cudaMalloc</code> needs a <code>cudaFree</code>: R.2's ownership rule applies to the device too.</p>
<p>Steps 2 and 4 cross PCIe, so they cost time. This is the full program of exercise 2. The error checks are left out here; step 7 adds them.</p>`,
        check: { q: "Which order is right?",
          options: ["cudaMalloc, launch, cudaMemcpy to device, cudaMemcpy to host, cudaFree", "cudaMalloc, cudaMemcpy to device, launch, cudaMemcpy to host, cudaFree", "cudaMemcpy to device, cudaMalloc, launch, cudaFree, cudaMemcpy to host"], answer: 1,
          why: "You need device memory before you can copy into it, the inputs must be there before the kernel reads them, the result must exist before you copy it back, and you free only when you are done with the buffers." },
        scene(G) {
          G.rect(16, 36, 196, 214, { stroke: "w", rx: 10 }); G.label(28, 54, "HOST", { color: "w" });
          ["h_a", "h_b", "h_c"].forEach((n, i) => G.box(30 + i * 60, 76, 54, 40, n, { fill: "w", size: 12, rx: 5 }));
          G.rect(428, 36, 196, 214, { stroke: "k", rx: 10 }); G.label(440, 54, "DEVICE", { color: "k" });
          ["d_a", "d_b", "d_c"].forEach((n, i) => G.box(442 + i * 60, 76, 54, 40, n, { fill: "k", size: 12, rx: 5 }));
          const k = G.box(442, 150, 168, 40, "add_kernel<<<…>>>", { fill: "q", size: 12, rx: 5 });
          G.text(440, 222, "5 cudaFree", { size: 12, color: "muted" });
          const st = [];
          st.push(G.arrow(218, 96, 422, 96, { color: "v", w: 2 }));
          st.push(G.arrow(218, 118, 422, 118, { color: "v", w: 2 }));
          st.push(G.arrow(218, 168, 422, 168, { color: "q", w: 2, dash: "4 3" }));
          G.label(226, 160, "3 launch (no data)", { size: 11, color: "q" });
          st.push(G.arrow(422, 216, 218, 216, { color: "ok", w: 2.5 }));
          G.label(226, 88, "2 copy in (a, b)", { size: 11, color: "v" });
          G.label(232, 238, "4 copy out (c)", { size: 11, color: "ok" });
          const L = [["1", "cudaMalloc  d_a d_b d_c", "k"], ["2", "cudaMemcpy host → device  (a, b)", "v"], ["3", "kernel<<<blocks, 256>>>", "q"], ["4", "cudaMemcpy device → host  (c)", "ok"], ["5", "cudaFree  d_a d_b d_c", "w"]];
          L.forEach(([n, t, c], i) => { G.circle(34, 288 + i * 26, 9, { fill: c }); G.text(34, 292 + i * 26, n, { size: 11, color: "bg", anchor: "middle" }); G.text(54, 293 + i * 26, t, { size: 13 }); });
          G.from(st, { opacity: 0, stagger: 0.25, duration: 0.35 });
          G.from(k, { opacity: 0, duration: 0.4, delay: 0.5 });
          G.caption("steps 2 and 4 cross PCIe; step 3 is the only one that runs on the GPU");
        } },

      { rail: "ceil-div", title: "Launch a few extra threads, then switch them off",
        body: `<p>How many blocks? You need at least <code>n</code> threads. <code>n</code> is rarely a multiple of the block size, so round <b>up</b>. Integer division rounds down, so add <code>threads − 1</code> first. This is <b>ceil-div</b>:</p>
<div class="eq">blocks = (n + threads - 1) / threads

n = 10, threads = 4:
  blocks = (10 + 3) / 4 = 13 / 4 = 3
  3 x 4 = 12 threads, 2 too many</div>
<p>The two extra threads (indices 10 and 11) have no element. If they ran <code>c[i] = a[i] + b[i]</code> they would read and write <b>past the end</b> of the arrays. That is an out-of-bounds access. It may crash, or silently corrupt a neighbour. So every kernel starts with a <b>bounds check</b>:</p>
<div class="eq">__global__ void add_kernel(
    const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i &lt; n) c[i] = a[i] + b[i];
}</div>
<p>Real size from exercise 2: n = 2<sup>20</sup> + 3 = 1,048,579 with 256 threads. 1,048,579 ÷ 256 = 4,096.01, so 4,097 blocks. That is 4,097 × 256 = 1,048,832 threads, and the last <b>253</b> do nothing. The extra threads are cheap. A missing check is not.</p>`,
        check: { q: "n = 1,000,000 and 256 threads per block. How many blocks, and how many threads does the bounds check switch off?",
          options: ["3,906 blocks, none switched off", "3,907 blocks, 192 switched off", "3,907 blocks, 256 switched off"], answer: 1,
          why: "(1,000,000 + 255) / 256 = 3,907 in integer maths (1,000,000 ÷ 256 = 3,906.25 rounds up). 3,907 × 256 = 1,000,192 threads, so 1,000,192 − 1,000,000 = 192 have no element. 3,906 blocks would cover only 999,936." },
        scene(G) {
          G.text(24, 36, "n = 10 elements · 4 threads per block", { size: 14 });
          G.text(24, 62, "blocks = (10 + 3) / 4 = 3  →  12 threads", { size: 14, color: "k" });
          const th = [];
          for (let b = 0; b < 3; b++) {
            G.rect(24 + b * 200, 84, 188, 74, { stroke: "line", rx: 8 });
            G.label(32 + b * 200, 102, "block " + b, { color: "ink", size: 12 });
            for (let t = 0; t < 4; t++) { const i = b * 4 + t; const r = G.box(34 + b * 200 + t * 44, 112, 38, 36, "i=" + i, { fill: i < 10 ? "k" : "hot", size: 11, rx: 5 }); if (i >= 10) th.push(r); }
          }
          G.label(24, 190, "the arrays have 10 elements (0 … 9):", { size: 12 });
          strip(G, 24, 200, 10, 36, 28, () => "ok", { gap: 4 });
          G.rect(24 + 10 * 40, 200, 80, 28, { stroke: "hot", dash: "4 3", rx: 4 });
          G.label(24 + 10 * 40 + 4, 248, "past the end", { size: 11, color: "hot" });
          G.text(24, 292, "if (i < n)  → threads 10 and 11 do nothing", { size: 15, color: "ok" });
          G.label(24, 322, "no check: they read and write memory that isn't yours", { size: 12, color: "hot" });
          G.text(24, 368, "2^20 + 3 elements, 256 threads: 4,097 blocks", { size: 13 });
          G.label(24, 392, "4,097 × 256 = 1,048,832 threads · 253 switched off", { size: 12 });
          G.pulse(th, { repeat: 5 }); G.from(th, { opacity: 0, duration: 0.4, delay: 0.3 });
          G.caption("round the grid up, then guard: the extra threads are harmless");
        } },

      { rail: "async & errors", title: "Launches return early, so errors arrive late",
        body: `<p>A kernel launch does not wait. It puts the work in a queue for the GPU and returns to the CPU at once. That lets the CPU get ahead, but it changes <i>when you hear about a mistake</i>:</p>
<ul><li>A bad <b>launch configuration</b>, such as 2,048 threads per block (the limit is 1,024), is spotted at launch. <code>cudaGetLastError()</code> right after returns the error. Without the check, <b>no thread runs and nothing tells you</b>.</li>
<li>A <b>fault inside the kernel</b>, such as an out-of-bounds write, happens later, on the GPU. It is reported by the <i>next</i> call that waits for the GPU. That may be a <code>cudaMemcpy</code> in a different function from the cause.</li></ul>
<p>So check twice after a launch. The course harness, <code>laneB-cuda/harness/include/s2s_cuda.cuh</code>, gives you both macros:</p>
<div class="eq">CUDA_CHECK(cudaMalloc(&amp;d_a, bytes));  // any call
add_kernel&lt;&lt;&lt;blocks, 256&gt;&gt;&gt;(d_a, d_b, d_c, n);
CUDA_CHECK(cudaGetLastError());       // valid?
CUDA_CHECK(cudaDeviceSynchronize());  // finished?
// those last two lines in one: CUDA_CHECK_LAUNCH();</div>
<p><code>CUDA_CHECK</code> prints the error name with the file and line and exits. Synchronizing after every launch is right for tests and debugging. It is wrong for a tuned pipeline, where you want the CPU to run ahead (P5.1 covers streams). If you cannot find a fault, set <code>CUDA_LAUNCH_BLOCKING=1</code> in the environment. Every launch then waits, so the error points at its true line. Exercise 3 plants both bugs.</p>`,
        check: { q: "A kernel writes out of bounds. <code>cudaGetLastError()</code> right after the launch returns success. Why?",
          options: ["The error check is broken", "The launch was valid and only queued; the bad write happens later on the GPU and shows at the next synchronizing call", "Out-of-bounds writes are never reported by CUDA"], answer: 1,
          why: "cudaGetLastError reports problems found while launching. The kernel has not run yet. A fault inside it is detected when the GPU executes it, and surfaces at a sync such as cudaDeviceSynchronize or a cudaMemcpy. That is why CUDA_CHECK_LAUNCH does both." },
        scene(G) {
          G.text(24, 40, "CPU", { size: 14 }); G.text(24, 150, "GPU", { size: 14 });
          G.line(70, 60, 620, 60, { color: "line" }); G.line(70, 170, 620, 170, { color: "line" });
          G.box(76, 44, 70, 30, "launch", { fill: "k", size: 12, rx: 5 });
          G.box(152, 44, 112, 30, "GetLastError", { fill: "ok", size: 11, rx: 5 });
          G.box(270, 44, 84, 30, "other work", { stroke: "line", size: 11, rx: 5 });
          G.box(360, 44, 130, 30, "DeviceSynchronize", { fill: "hot", size: 11, rx: 5 });
          G.arrow(112, 78, 190, 156, { color: "k", dash: "4 3" });
          const kn = G.box(190, 156, 260, 30, "kernel runs", { fill: "q", size: 12, rx: 5 });
          G.circle(420, 171, 11, { fill: "hot" }); G.text(420, 176, "×", { size: 15, color: "bg", anchor: "middle" });
          G.arrow(420, 156, 425, 80, { color: "hot", w: 2 });
          G.label(436, 120, "fault reported here", { size: 12, color: "hot" });
          G.label(214, 104, "← bad config? seen here", { size: 12, color: "ok" });
          G.text(24, 252, "1  launch returns at once; the kernel runs later", { size: 13 });
          G.text(24, 282, "2  bad configuration → cudaGetLastError()", { size: 13, color: "ok" });
          G.text(24, 312, "3  fault inside the kernel → the next sync", { size: 13, color: "hot" });
          G.text(24, 342, "4  CUDA_CHECK_LAUNCH() does 2 and 3", { size: 13, color: "k" });
          G.label(24, 380, "CUDA_LAUNCH_BLOCKING=1 makes every launch wait (debug only)", { size: 12 });
          G.from(kn, { opacity: 0, x: -40, duration: 0.6, delay: 0.3 });
          G.caption("the CPU runs ahead; a kernel's fault is reported when the CPU next waits");
        } },

      { rail: "debugging", title: "When it is wrong: printf and compute-sanitizer",
        body: `<p>There is no debugger window pointing at thread 5,000. You have two simple tools.</p>
<p><b><code>printf</code> inside a kernel.</b> It works, with caveats. The text appears only when the GPU synchronizes. Lines from different threads arrive in <i>any order</i>. And the output buffer has a fixed size, so printing from millions of threads loses lines. Print from a few threads:</p>
<div class="eq">if (i &lt; 4) printf("i=%d a=%f\\n", i, a[i]);</div>
<p><b><code>compute-sanitizer</code>.</b> Run your program under it and it checks every memory access against the exact bounds of each allocation:</p>
<div class="eq">nvcc -lineinfo -o add add.cu
compute-sanitizer ./add</div>
<p>It catches the case that fooled you: a kernel without <code>if (i &lt; n)</code> often "works", because the stray writes land in padding the allocator added. The program prints the right answer and still has a bug. The sanitizer reports the bad write, which thread and block did it, how far past the end, and (with <code>-lineinfo</code>) the source line. It makes the program much slower, so use small inputs. Exercise 4 gives you a program to run it on.</p>
<p>The report in the picture is the <i>shape</i> of a real one, written by hand. Your addresses and wording will differ.</p>`,
        scene(G) {
          G.rect(16, 40, 608, 196, { stroke: "line", rx: 8 });
          G.label(28, 58, "compute-sanitizer ./oob      (shape of a report, not real output)", { size: 11, color: "v" });
          const L = [
            ["COMPUTE-SANITIZER", "ink"],
            ["Invalid __global__ write of size 4 bytes", "hot"],
            ["  at scale_kernel(float *, int, float)", "ink"],
            ["  by thread (232,0,0) in block (3,0,0)", "v"],
            ["  Address 0x… is out of bounds", "ink"],
            ["  … 4 bytes after the nearest allocation", "ink"],
            ["  … of size 4000 bytes", "ink"],
          ];
          const ls = L.map(([t, c], i) => G.text(30, 84 + i * 21, "=== " + t, { size: 12, color: c }));
          G.text(24, 270, "read it:", { size: 13 });
          G.label(24, 294, "which thread: block 3, thread 232 → i = 3·256 + 232 = 1000", { size: 12, color: "v" });
          G.label(24, 316, "n = 1000, so index 1000 is the first element that doesn't exist", { size: 12 });
          G.label(24, 346, "fix: if (i < n)    ·    then run the sanitizer again: no errors", { size: 12, color: "ok" });
          G.label(24, 384, "printf: few threads, any order, shown at the next sync", { size: 12 });
          G.from(ls, { opacity: 0, x: -10, stagger: 0.12, duration: 0.25 });
          G.caption("a correct-looking answer is not proof; the sanitizer checks the bounds");
        } },

      { rail: "timing", title: "Time it with events, and expect the copies to dominate",
        body: `<p>A host clock around a launch measures only how long it took to <i>queue</i> the work. To time the GPU, use <b>CUDA events</b>, markers the GPU records on its own timeline:</p>
<div class="eq">cudaEventRecord(e0);
add_kernel&lt;&lt;&lt;blocks, 256&gt;&gt;&gt;(...);
cudaEventRecord(e1);
cudaEventSynchronize(e1);       // wait for the marker
float ms;
cudaEventElapsedTime(&amp;ms, e0, e1);
GB/s = bytes_moved / (ms * 1e6)</div>
<p>Two honesty rules. The <b>first</b> CUDA call creates the GPU context, which takes visible time, so warm up and time the later runs. And time the three parts <b>separately</b>: copy in, kernel, copy out. <code>examples/02_transfer_tax.cu</code> does this.</p>
<p>Now compare with the roofline from P1.4. Vector add does 1 add for 12 bytes (two 4-byte reads and one 4-byte write), so its intensity is 1 ÷ 12 ≈ 0.083 FLOP/byte, far left of any ridge. It is memory-bound: its best time is <b>bytes ÷ memory bandwidth</b>. For n = 2<sup>26</sup> = 67,108,864:</p>
<div class="eq">bytes  = 12 x 67,108,864 = 805,306,368 (805 MB)
kernel = 805.3e6 / 320e9 = 2.52 ms  (T4, UNVERIFIED)
copies = 805.3e6 / 12e9  = 67.1 ms  (example PCIe)
copies / kernel = 320 / 12 = 26.7</div>
<p>The copies cost about 27 times the kernel. Both rates are example values, so treat it as a model and measure on your machine. Made-up reading to practise the formula: if the event timer said 3.1 ms, you moved 805.3 MB ÷ 3.1 ms ≈ 260 GB/s, which is 81% of 320.</p>`,
        check: { q: "Vector add on 805 MB: kernel about 2.5 ms, copies about 67 ms (example rates). What do you conclude?",
          options: ["Make the kernel faster: that is where the time is", "Almost all the time is PCIe copies, so keep data on the GPU and do more work per copy", "The GPU is slower than the CPU for this"], answer: 1,
          why: "The kernel is about 4% of the total. Speeding it up barely helps. The lever is to cross PCIe less: copy once and run many kernels on the resident data, as inference engines do with weights and the KV cache. (The rates are examples; measure yours.)" },
        scene(G) {
          G.text(24, 34, "event timers sit on the GPU's timeline", { size: 13 });
          G.label(24, 64, "host", { size: 11 }); G.label(24, 108, "GPU", { size: 11 });
          G.rect(70, 52, 24, 20, { fill: "w", rx: 3 }); G.label(100, 66, "host clock stops here", { size: 11, color: "hot" });
          const kn = G.rect(100, 96, 300, 22, { fill: "q", rx: 3 }); G.label(108, 112, "kernel", { size: 11, color: "bg" });
          G.line(100, 134, 400, 134, { color: "ok", w: 2 }); G.label(100, 150, "event start → event stop covers the whole kernel", { size: 11, color: "ok" });
          G.text(24, 190, "first call vs later calls (shape only, not measured)", { size: 12, color: "muted" });
          G.rect(24, 202, 380, 20, { fill: "hot", rx: 3, opacity: 0.55 }); G.label(30, 216, "context creation + module load", { size: 11 });
          G.rect(404, 202, 26, 20, { fill: "ok", rx: 3 });
          G.rect(24, 230, 26, 20, { fill: "ok", rx: 3 }); G.label(58, 244, "later calls: just the work", { size: 11 });
          G.text(24, 290, "vector add, n = 2^26, 805 MB moved (example rates)", { size: 12, color: "muted" });
          const b1 = G.rect(24, 302, 420, 22, { fill: "hot", rx: 3 }); G.label(32, 318, "copies over PCIe at 12 GB/s ≈ 67 ms", { size: 11, color: "bg" });
          const b2 = G.rect(24, 332, 15.8, 22, { fill: "k", rx: 3 }); G.label(46, 348, "kernel at 320 GB/s ≈ 2.5 ms", { size: 11, color: "k" });
          G.label(24, 384, "GB/s = bytes ÷ (ms × 10⁶)       12 B per add: 320 ÷ 12 ≈ 26.7 G adds/s", { size: 12, color: "ink" });
          G.from([b1, b2], { attr: { width: 0 }, stagger: 0.3, duration: 0.5 }); G.from(kn, { attr: { width: 0 }, duration: 0.5 });
          G.caption("time the GPU with events; compare each part with its bandwidth");
        } },

      { rail: "what next", title: "Where this leads: Lane B, then the deep dive",
        body: `<p>You can now write and run a complete CUDA program. The course builds on it in three places:</p>
<ul><li><b>Lane B, L1 (weeks 6 to 10).</b> Many small problems on indexing: 1D, 2D and 3D, grid-stride loops, bounds. It runs alongside P2 to P4. <b>Do P1.6 before L1.</b> L2 (weeks 11 to 16) adds shared memory and coalescing. L3 (weeks 17 to 24) adds reductions and scans.</li>
<li><b>P5, the CUDA performance deep dive (weeks 25 to 34).</b> It assumes this lesson and L1 to L3. Its first lesson, P5.1, recaps launches and indexing quickly and then goes deeper: warps, occupancy, divergence, streams and pinned memory. The other P5 lessons climb the speed ladders with Nsight.</li>
<li><b>Before L1, if you want it:</b> an optional no-GPU warm-up. The srush GPU-Puzzles notebook (linked in the module README) teaches thread indexing with no hardware.</li></ul>
<p>The habit to carry forward: <b>know the five steps, check every call, and count the bytes</b>. Most serving-side CUDA work is deciding what stays on the GPU and what crosses the link.</p>`,
        scene(G) {
          const nodes = [
            { x: 16, t: "P1.6", s: "this lesson", c: "v" },
            { x: 138, t: "L1", s: "weeks 6–10", c: "k" },
            { x: 260, t: "L2", s: "weeks 11–16", c: "k" },
            { x: 382, t: "L3", s: "weeks 17–24", c: "k" },
            { x: 504, t: "P5", s: "weeks 25–34", c: "q" },
          ];
          G.text(24, 40, "CUDA path", { size: 14 });
          nodes.forEach((n, i) => {
            G.box(n.x, 70, 104, 64, n.t, { fill: n.c, size: 17, rx: 8 });
            G.label(n.x + 52, 154, n.s, { size: 11, anchor: "middle" });
            if (i) G.arrow(n.x - 14, 102, n.x - 2, 102, { color: "ink", w: 2 });
          });
          G.label(138, 176, "L1–L3 run alongside P2–P4", { size: 12, color: "k" });
          G.label(624, 176, "assumes P1.6 + L1–L3", { size: 12, color: "q", anchor: "end" });
          G.text(24, 240, "optional before L1:", { size: 13 });
          G.rect(24, 252, 296, 46, { stroke: "line", dash: "4 3", rx: 8 });
          G.label(36, 280, "L0 warm-up: GPU-Puzzles, no GPU needed", { size: 12 });
          G.text(24, 336, "the habit: five steps · check every call · count the bytes", { size: 14, color: "ok" });
          G.label(24, 366, "P5.1 recaps this quickly, then adds warps, occupancy and streams", { size: 12 });
          G.caption("do P1.6 before L1; P5 assumes both");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A CUDA program is host code plus device code. The GPU has its own memory, joined to the CPU by a PCIe link that is far slower than GPU memory. A <code>cudaMalloc</code> pointer is only valid on the device.</li>
<li><code>nvcc</code> sends host code to the C++ compiler and device code to PTX and SASS for a chosen compute capability (<code>-arch=sm_75</code> for a T4). Check any machine with <code>nvidia-smi</code> and <code>nvcc --version</code>.</li>
<li>A <code>__global__</code> kernel runs once per thread. <code>blockIdx.x * blockDim.x + threadIdx.x</code> is the global index; round the grid up with <code>(n + t − 1) / t</code> and guard with <code>if (i &lt; n)</code>.</li>
<li>The host program is allocate, copy in, launch, copy out, free. Launches are asynchronous: <code>cudaGetLastError()</code> catches a bad configuration, a synchronize catches a kernel fault, and <code>CUDA_CHECK</code> wraps every call.</li>
<li>Debug with <code>printf</code> from a few threads and with <code>compute-sanitizer</code>. Time with CUDA events, warm up first, and compare the copies and the kernel with their bandwidths: for vector add the copies dominate.</li></ul>
<p>Next, P2.1 puts the GPU to work serving a real model. Start Lane B L1 alongside it, and come back to this lesson's five steps whenever a CUDA program surprises you.</p>`,
    sim: {
      title: "The transfer tax",
      intro: "Choose how much data goes to the GPU, how fast the link and the GPU memory are, and how many times the kernel passes over the data. The bar splits the time between copying in, the kernel and copying out. The curve shows how the copy share falls as the kernel does more passes. A model, not a measurement: PCIe and GPU memory rates are example values (check your platform; the T4's 320 GB/s is UNVERIFIED in P1.4's gpu_specs.yaml). Launch overhead and copy latency are ignored.",
      height: 300,
      controls: [
        { id: "n", label: "elements N", type: "select", value: 67108864, options: [[1048576, "2^20 = 1.05 M"], [16777216, "2^24 = 16.8 M"], [67108864, "2^26 = 67.1 M"], [268435456, "2^28 = 268 M"]] },
        { id: "b", label: "bytes per element", type: "select", value: 4, options: [[1, "1 (int8 / fp8)"], [2, "2 (fp16 / bf16)"], [4, "4 (fp32)"]] },
        { id: "ai", label: "arrays copied in", min: 1, max: 4, step: 1, value: 2 },
        { id: "ao", label: "arrays copied out", min: 1, max: 2, step: 1, value: 1 },
        { id: "pcie", label: "PCIe bandwidth, GB/s (example value, check your platform)", min: 4, max: 64, step: 0.5, value: 12 },
        { id: "bw", label: "GPU memory bandwidth, GB/s (example value, check the spec)", min: 100, max: 3500, step: 10, value: 320 },
        { id: "p", label: "kernel passes over the data", min: 1, max: 100, step: 1, value: 1 },
      ],
      draw(G, v) {
        const N = v.n, B = v.b, bytesIn = v.ai * N * B, bytesOut = v.ao * N * B;
        const tIn = (bytesIn / (v.pcie * 1e9)) * 1e3, tOut = (bytesOut / (v.pcie * 1e9)) * 1e3;
        const tK = (v.p * (bytesIn + bytesOut) / (v.bw * 1e9)) * 1e3;
        const total = tIn + tK + tOut, copies = tIn + tOut, share = copies / total;
        const crossover = v.bw / v.pcie;
        // stacked bar
        const x0 = 20, W = 600, y0 = 36, H = 52;
        G.label(x0, 24, "time split (ms)", { size: 11 });
        const parts = [["copy in", tIn, "v"], ["kernel", tK, "k"], ["copy out", tOut, "hot"]];
        let x = x0;
        parts.forEach(([nm, t, c]) => {
          const w = Math.max(1, (W * t) / total);
          G.rect(x, y0, w, H, { fill: c, rx: 0 });
          if (w > 70) G.text(x + w / 2, y0 + 22, nm, { size: 12, color: "bg", anchor: "middle" });
          if (w > 70) G.text(x + w / 2, y0 + 40, F.num(t, t < 10 ? 2 : 1) + " ms", { size: 11, color: "bg", anchor: "middle" });
          x += w;
        });
        if (tK / total < 0.12) G.label(x0, y0 + H + 16, `kernel: ${F.num(tK, 2)} ms (${F.num((100 * tK) / total, 1)}% of the total)`, { size: 11, color: "k" });
        // copy share vs passes
        const cx = 70, cy = 136, cw = 540, ch = 124;
        G.axes(cx, cy, cw, ch, {});
        G.label(cx, cy - 6, "copy share of total time", { size: 11 });
        G.label(cx + cw, cy + ch + 28, "kernel passes over the data (1 … 100)", { size: 11, anchor: "end" });
        [0, 0.5, 1].forEach((f) => G.label(cx - 6, cy + ch - f * ch + 4, `${f * 100}%`, { size: 10, anchor: "end" }));
        [1, 25, 50, 75, 100].forEach((p) => G.label(cx + ((p - 1) / 99) * cw, cy + ch + 14, String(p), { size: 10, anchor: "middle" }));
        const kPer = (bytesIn + bytesOut) / (v.bw * 1e9) * 1e3;
        let d = "";
        for (let p = 1; p <= 100; p++) {
          const sh = copies / (copies + p * kPer);
          d += (p === 1 ? "M " : "L ") + (cx + ((p - 1) / 99) * cw) + " " + (cy + ch - sh * ch) + " ";
        }
        G.line(cx, cy + ch / 2, cx + cw, cy + ch / 2, { color: "line", dash: "3 4" });
        G.path(d, { color: "hot", w: 2.5 });
        if (crossover >= 1 && crossover <= 100) {
          const xc = cx + ((crossover - 1) / 99) * cw;
          G.line(xc, cy, xc, cy + ch, { color: "ok", dash: "4 3" });
          G.label(xc + 4, cy + 14, `break-even ≈ ${F.num(crossover, 1)} passes`, { size: 11, color: "ok" });
        }
        G.circle(cx + ((v.p - 1) / 99) * cw, cy + ch - share * ch, 6, { fill: "v" });
        return [
          { title: "Where the time goes", rows: [["copy in (H2D)", F.num(tIn, 2) + " ms"], ["kernel", F.num(tK, 3) + " ms"], ["copy out (D2H)", F.num(tOut, 2) + " ms"], ["total", F.num(total, 2) + " ms"]],
            gauge: [[share, "hot"], [1 - share, "k"]], gaugeText: `copies are ${F.num(100 * share, 1)}% of the time`,
            chip: [share < 0.5, share < 0.5 ? "the kernel dominates: the copies are worth it" : "the copies dominate: keep data on the GPU"] },
          { title: "Break-even", rows: [["copy time = kernel time at", F.num(crossover, 1) + " passes"], ["= GPU bandwidth ÷ PCIe", `${v.bw} ÷ ${v.pcie}`], ["end-to-end rate", F.num(((bytesIn + bytesOut) / total) / 1e6, 2) + " GB/s"]],
            html: `<p class="note">Model: copy time = bytes ÷ PCIe rate; one kernel pass reads every input array and writes every output array once, at the GPU's memory rate. Ignores launch overhead, copy latency, pageable-memory staging (P0.2) and overlap with streams (P5.1). Measure with <code>examples/02_transfer_tax.cu</code>.</p>` },
        ];
      },
      note: "Defaults reproduce the lesson's vector add: 2^26 floats, 2 arrays in, 1 out, 12 GB/s PCIe, 320 GB/s GPU memory, one pass: copies ≈ 96% of the time.",
    },
    practice: {
      intro: `<p>Kernel-only work (T1) runs on LeetGPU or Colab. The full host program and the tools need a CUDA GPU (T2): the course's AWS <code>g4dn.xlarge</code> as in <code>aws.md</code>, which points at the P5 guide. Build once from the repository root. Every run below is <code>TODO(run-on: T2)</code>; no result has been recorded in the repo yet.</p>
<p><code>cmake -S ${MOD}/exercises -B build/p1.6 -DCMAKE_CUDA_ARCHITECTURES=native &amp;&amp; cmake --build build/p1.6 -j</code></p>
<p>Starter tests fail until you implement them. Add <code>-DS2S_USE_SOLUTIONS=ON</code> in a separate build directory to build against the references.</p>`,
      items: [
        { title: "Hello, device", tier: "T2 · easy", goal: mono("Launch 2 blocks of 4 threads. Each prints its global index with <code>printf</code> and stores it in the output array. The test checks the launch with <code>CUDA_CHECK_LAUNCH()</code> and the stored indices."),
          cmd: "ctest --test-dir build/p1.6 -R p1.6_01-hello-device --output-on-failure" },
        { title: "Vector add, the full program", tier: "T2 · medium", goal: mono("Write the kernel and all five host steps for <code>c = a + b</code>, with ceil-div and the bounds check. The test compares with a CPU loop for N = 1, 1000, 2²⁰ and 2²⁰ + 3."),
          cmd: "ctest --test-dir build/p1.6 -R p1.6_02-vector-add-full --output-on-failure" },
        { title: "Make the errors speak", tier: "T2 · medium", goal: mono("Add <code>cudaGetLastError()</code> and <code>cudaDeviceSynchronize()</code> so a 2,048-thread launch and a kernel fault are returned as errors instead of silent garbage."),
          cmd: "ctest --test-dir build/p1.6 -R p1.6_03-error-checks --output-on-failure" },
        { title: "Read a compute-sanitizer report", tier: "T2 · easy (no test)", goal: mono("Run <code>compute-sanitizer</code> on a program with a missing bounds check that still prints the right answer. Explain the first error, then fix the kernel."),
          cmd: "cmake --build build/p1.6 --target p1.6_04-sanitizer-oob && compute-sanitizer ./build/p1.6/p1.6_04-sanitizer-oob" },
      ],
      labs: [
        { label: "Module README: all ten steps in text, with the commands", path: `${MOD}/README.md` },
        { label: "Example 01: print your GPU's compute capability, SM count and memory", path: `${MOD}/examples/01_device_info.cu` },
        { label: "Example 02: time H2D, kernel and D2H with CUDA events, and the first-call cost", path: `${MOD}/examples/02_transfer_tax.cu` },
        { label: "AWS g4dn.xlarge flow (points at the shared P5 guide; cost, auto-stop and teardown there)", path: `${MOD}/aws.md` },
        { label: "Harness: CUDA_CHECK, CUDA_CHECK_LAUNCH, DeviceBuffer, time_gpu", path: "laneB-cuda/harness/include/s2s_cuda.cuh" },
        { label: "Next: Lane B L1, indexing in 1D, 2D and 3D", path: "laneB-cuda/L1-launch-and-indexing/syllabus.md" },
      ],
    },
  });
})();
