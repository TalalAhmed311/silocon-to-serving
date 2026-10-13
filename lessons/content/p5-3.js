/* P5.3 — Profiling with Nsight Systems and Nsight Compute: measure first, then change one thing. */
(function () {
  const F = S2S.fmt;
  const amdahl = (s, x) => 1 / (1 - s + s / x);

  S2S.lesson({
    id: "p5-3", n: "P5.3", title: "Profiling with Nsight",
    subtitle: "CUDA deep dive · first principles · T2, GPU with counter access",
    kicker: "Lesson · ≈ 40 min",
    headline: "Don't guess which line is slow",
    intro: `<p>You can now write kernels (P5.1) and reason about their memory traffic (P5.2). The next skill is <b>finding out what a real program is doing</b>. Programmers are bad at guessing where time goes, and GPU code is worse: a kernel can be slow because of memory, arithmetic, occupancy, or the CPU not feeding it. NVIDIA ships two profilers that answer different questions: <b>Nsight Systems</b> for the whole program's timeline and <b>Nsight Compute</b> for one kernel's internals. This lesson teaches a fixed order to use them in, how to read the key sections, and how to get them running on a cloud GPU.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "3 exercises"],
    legend: [["k", "GPU kernel"], ["v", "CPU / API"], ["q", "NVTX range"], ["line", "idle gap"], ["hot", "problem"], ["ok", "good"]],
    prev: "p5-2", next: "p5-4",
    steps: [
      { rail: "why profile", title: "Optimizing the wrong thing is free, and useless",
        body: `<p>Suppose a run takes 100 ms and one kernel accounts for 3 ms of it. You spend a week and make that kernel <b>twice as fast</b>. How much faster is the program? Amdahl's law answers it. If a fraction <i>s</i> of the time gets sped up by a factor <i>x</i>:</p>
<div class="eq">speedup = 1 / ((1 - s) + s / x)

s = 0.03, x = 2:
  1 / (0.97 + 0.015) = 1.015  (1.5% faster)
s = 0.03, x = infinity:
  1 / 0.97           = 1.031  (best possible)</div>
<p>One week for 1.5%. The number to find first is <i>s</i>: <b>what share of the whole run does this kernel take</b>? That is a question about the program's timeline, not about the kernel.</p>
<p>The same law caps the gain from the other direction. If your kernel is 90% of the run (s = 0.9), then making it 10x faster gives 1 / (0.1 + 0.09) = 5.26x overall. After that, the rest dominates, and you profile again.</p>
<p>So the order is always: <b>measure the whole program, pick the biggest slice, measure that one kernel, change one thing, measure again.</b> The rest of the lesson is the tools for each of those steps.</p>`,
        scene(G) {
          G.text(24, 36, "a 100 ms run; one kernel is 3 ms", { size: 14 });
          G.rect(24, 60, 6, 30, { fill: "k", rx: 1 });
          G.rect(30, 60, 586, 30, { fill: "line", rx: 1 });
          G.label(40, 80, "everything else: 97 ms", { size: 12 }); G.label(34, 54, "kernel 3 ms", { size: 11, color: "k" });
          G.text(24, 140, "make the kernel 2x faster:", { size: 13 });
          G.rect(24, 160, 3, 30, { fill: "ok", rx: 1 });
          G.rect(27, 160, 586, 30, { fill: "line", rx: 1 });
          G.line(24 + 6, 150, 24 + 6, 160, { color: "ok" });
          G.label(40, 180, "total 98.5 ms: 1.5% faster", { size: 12, color: "hot" });
          G.text(24, 250, "Amdahl:  speedup = 1 / ((1 - s) + s / x)", { size: 14 });
          G.label(24, 280, "s = 0.03, x = 2   ->  1.015", { size: 13, color: "hot" });
          G.label(24, 304, "s = 0.90, x = 10  ->  5.26", { size: 13, color: "ok" });
          G.label(24, 350, "find the big slice first: that is a timeline question", { size: 13, color: "v" });
          G.caption("the payoff of a speed-up is capped by the share it touches");
        } },

      { rail: "two tools", title: "Two tools, two questions",
        body: `<p>NVIDIA's profilers split the job by zoom level:</p>
<table><tr><th>question</th><th>tool</th></tr>
<tr><td>Where does wall-clock time go? Is the GPU idle between kernels? Do copies overlap with compute?</td><td><b>Nsight Systems</b> (<code>nsys</code>)</td></tr>
<tr><td>Why is <i>this one kernel</i> slow? Memory- or compute-bound? What are its warps waiting for?</td><td><b>Nsight Compute</b> (<code>ncu</code>)</td></tr></table>
<p><b>nsys</b> records a <i>timeline</i>: when every API call, kernel, copy and user-marked range started and ended. It adds little overhead, so the timing it shows is close to the real run. <b>ncu</b> records <i>hardware counters</i> for individual kernels. It is far more detailed and far more intrusive (step 5).</p>
<div class="eq">nsys profile -o out ./bin   # whole program
ncu --set full -o out ./bin # every kernel
ncu -k regex:name --set full ...
                            # matching kernels</div>
<p>Always start with <code>nsys</code>. The slow part is often not a kernel at all: a decode loop that launches many tiny kernels can be bound by <b>launch overhead and CPU gaps</b>, which only the timeline shows (CUDA graphs, P6.6, remove them).</p>`,
        scene(G) {
          G.box(24, 50, 270, 130, "", { stroke: "k", rx: 8 });
          G.text(40, 78, "Nsight Systems", { size: 15, color: "k" });
          G.label(40, 100, "the whole program, a timeline", { size: 12 });
          G.label(40, 122, "gaps, launches, copies, streams", { size: 12 });
          G.label(40, 144, "low overhead", { size: 12 });
          G.label(40, 166, "tool: nsys", { size: 12, color: "k" });
          G.box(346, 50, 270, 130, "", { stroke: "q", rx: 8 });
          G.text(362, 78, "Nsight Compute", { size: 15, color: "q" });
          G.label(362, 100, "one kernel, hardware counters", { size: 12 });
          G.label(362, 122, "memory, compute, stalls, source", { size: 12 });
          G.label(362, 144, "replays the kernel: slow", { size: 12 });
          G.label(362, 166, "tool: ncu", { size: 12, color: "q" });
          G.arrow(160, 190, 160, 230, { color: "ok", w: 2 }); G.arrow(480, 190, 480, 230, { color: "ok", w: 2 });
          G.box(24, 236, 270, 44, "which kernel? or is it a gap?", { stroke: "line", size: 13, rx: 6 });
          G.box(346, 236, 270, 44, "why is that kernel slow?", { stroke: "line", size: 13, rx: 6 });
          G.arrow(300, 258, 340, 258, { color: "v", w: 2 });
          G.label(24, 330, "order: nsys first, then ncu on the biggest kernel", { size: 13, color: "v" });
          G.caption("zoom out with nsys, zoom in with ncu");
        } },

      { rail: "timeline", title: "Reading an nsys timeline: kernels, gaps, ranges",
        body: `<p>An nsys report is a set of horizontal <b>rows</b> sharing one time axis. The ones you read first:</p>
<ul><li><b>CPU thread / CUDA API:</b> each <code>cudaLaunchKernel</code>, <code>cudaMemcpy</code>, <code>cudaDeviceSynchronize</code> call, as a bar. A long bar means the CPU waited.</li>
<li><b>GPU kernels:</b> one bar per kernel execution. <b>White space between bars is idle GPU</b>.</li>
<li><b>Memcpy rows:</b> host-to-device and device-to-host copies (P5.1 streams: do they overlap kernels?).</li>
<li><b>NVTX ranges:</b> named spans <i>you</i> put in the code, such as "prefill" or "decode step 7" (next step).</li></ul>
<p>What you look for: the share of a range covered by kernels. A toy example (round numbers, not a measurement): a decode step range lasts 10 ms, and the kernels inside add up to 6 ms.</p>
<div class="eq">GPU busy = 6 ms / 10 ms = 60%
GPU idle = 4 ms         = 40%
           (launch gaps + syncs)</div>
<p>No kernel optimization can touch that 40%. The fix is on the CPU side: fewer, bigger launches, or a CUDA graph (P6.6). This is exactly what the module's second example, <code>02_nvtx_ranges.cu</code>, lets you see.</p>`,
        scene(G) {
          const x0 = 110, W = 500;
          const rows = [["NVTX", 62], ["CPU API", 112], ["GPU kernels", 162]];
          rows.forEach(([n, y]) => { G.label(24, y + 18, n, { size: 12 }); G.line(x0, y + 28, x0 + W, y + 28, { color: "line" }); });
          G.box(x0, 62, W, 26, "decode step 7 (10 ms)", { stroke: "q", color: "q", size: 12, rx: 4 });
          const api = [], ker = [];
          const ks = [[0, 70], [85, 55], [150, 80], [245, 40], [300, 60]];
          ks.forEach(([o, w], i) => {
            api.push(G.box(x0 + o - 20 + (i ? 0 : 20), 112, 28, 24, "L", { fill: "v", size: 11, rx: 3 }));
            ker.push(G.box(x0 + o + 10, 162, w, 26, "k" + i, { fill: "k", size: 11, rx: 3 }));
          });
          G.rect(x0 + 80, 162, 15, 26, { stroke: "hot", rx: 2, dash: "3 3" });
          G.rect(x0 + 140, 162, 20, 26, { stroke: "hot", rx: 2, dash: "3 3" });
          G.rect(x0 + 330, 162, 170, 26, { stroke: "hot", rx: 2, dash: "3 3" });
          G.label(x0 + 290, 206, "idle: next launch not ready", { size: 12, color: "hot" });
          G.text(24, 270, "toy numbers: step = 10 ms, kernels = 6 ms", { size: 13 });
          G.label(24, 296, "GPU busy 60%  ·  idle 40%", { size: 13, color: "hot" });
          G.label(24, 334, "no kernel tuning removes the gaps: fix the launch pattern", { size: 12, color: "v" });
          G.from(ker, { opacity: 0, x: -20, stagger: 0.1, duration: 0.3 });
          G.caption("white space on the GPU row is time no kernel can win back");
        } },

      { rail: "NVTX", title: "Put names on the timeline with NVTX",
        body: `<p>An nsys timeline shows kernel names like <code>ampere_sgemm_128x64_nn</code>, but not <i>why</i> you launched them. <b>NVTX</b> (NVIDIA Tools Extension) lets you push and pop <b>named ranges</b> around your own code. They appear as labelled bars on the timeline, nested like function calls.</p>
<div class="eq">#include &lt;nvtx3/nvToolsExt.h&gt;  // header-only

nvtxRangePushA("prefill");
run_prefill();
nvtxRangePop();
for (int t = 0; t &lt; 32; ++t) {
  nvtxRangePushA("decode step");
  run_decode_step();
  nvtxRangePop();
}</div>
<p>Push and pop must match, like braces. Ranges nest, so you can wrap a layer's <code>attention</code> and <code>mlp</code> inside a <code>decode step</code> range. Cost is a few tens of nanoseconds per call, small enough to leave in for profiling builds. Exercise 2 has you add them to the repo's #0 engine (<code>platform/engine/v0/src/engine.hpp</code>) behind <code>#ifdef S2S_NVTX</code>, so the plain CPU build does not need CUDA. Nsight Systems also traces CPU-only programs, so the ranges are useful even before the GPU version exists.</p>
<p>Then ask the questions from this lesson: which phase dominates? Does a decode step get longer as the position grows (attention reads a longer KV cache, P1.2)? Keep the ranges: P6.6 reuses this view to show what CUDA graphs remove.</p>`,
        check: { q: "\"The GPU is idle 40% of every decode step.\" Which tool answers that, and why?",
          options: ["ncu: it reads hardware counters", "nsys: it is a timeline question about gaps and launches", "Either: both give the same view"], answer: 1,
          why: "Idle time between kernels is invisible inside any single kernel. nsys places every launch, kernel and sync on one time axis, so gaps and CPU waits show up directly. ncu looks inside one kernel and says nothing about what happens between kernels." },
        scene(G) {
          const x0 = 40;
          G.text(24, 36, "ranges nest like calls", { size: 14 });
          G.box(x0, 56, 560, 30, "prefill", { stroke: "q", color: "q", size: 12, rx: 4 });
          G.box(x0, 100, 560, 30, "", { stroke: "q", rx: 4 });
          G.label(x0 + 6, 120, "decode loop", { size: 12, color: "q" });
          for (let i = 0; i < 3; i++) {
            const x = x0 + 100 + i * 150;
            G.box(x, 138, 140, 28, "decode step " + i, { stroke: "q", color: "q", size: 11, rx: 4 });
            G.box(x, 172, 66, 26, "attn", { fill: "k", size: 11, rx: 3 });
            G.box(x + 70, 172, 70, 26, "mlp", { fill: "blue", size: 11, rx: 3 });
          }
          G.text(24, 250, "nvtxRangePushA(\"decode step\");", { size: 13, color: "q" });
          G.text(24, 276, "   run_decode_step();", { size: 13 });
          G.text(24, 302, "nvtxRangePop();", { size: 13, color: "q" });
          G.label(24, 346, "guard with #ifdef S2S_NVTX so the CPU build stays CUDA-free", { size: 12 });
          G.caption("your own labels turn kernel names into a story");
        } },

      { rail: "ncu replay", title: "How ncu measures one kernel (and why its times differ)",
        body: `<p>The GPU has far more hardware counters than it can read at once: only a handful can be collected during one run of a kernel. To give you hundreds of metrics, <b>ncu runs the same kernel many times</b>, reading a different counter group each time. This is <b>kernel replay</b>. Between passes ncu saves and restores the memory the kernel touches and, by default, <b>flushes the caches</b> so each pass starts from the same state.</p>
<p>Consequences you must remember:</p>
<ul><li>ncu <b>serializes</b> kernels and adds overhead, so its <i>duration</i> is not your program's time. For timing use CUDA events (P5.1) or nsys.</li>
<li>Cold caches make ncu's memory numbers pessimistic for kernels that normally hit in cache.</li>
<li>Profile a <b>Release</b> build with <code>-lineinfo</code> (the P5 CMake does). A debug build (<code>-G</code>) disables optimizations, so every number is wrong.</li>
<li>The kernel must be deterministic across replays; kernels that accumulate into their own output with atomics need care.</li></ul>
<p>Treat ncu as a microscope: it tells you <i>why</i>, never <i>how long the whole thing took</i>.</p>`,
        check: { q: "ncu reports a kernel at 2.0 ms but your CUDA-event timing says 1.2 ms. What is the most likely reason?",
          options: ["Your event timing is wrong", "ncu serializes and replays the kernel with flushed caches and overhead", "ncu reports GPU time in different units"], answer: 1,
          why: "ncu replays the kernel in several passes, flushes caches between them by default, and serializes launches. Its duration measures a cold, instrumented run. Use ncu for metrics and ratios, and events or nsys for timing." },
        scene(G) {
          G.text(24, 36, "one kernel launch in your program", { size: 13 });
          G.box(24, 50, 120, 34, "kernel", { fill: "k", size: 13, rx: 5 });
          G.arrow(150, 67, 190, 67, { color: "muted" });
          G.text(196, 72, "ncu replays it:", { size: 13 });
          const ps = [];
          ["counters A", "counters B", "counters C", "counters D"].forEach((c, i) => {
            const y = 110 + i * 56;
            ps.push(G.box(24, y, 100, 36, "pass " + (i + 1), { fill: "q", size: 12, rx: 5 }));
            G.arrow(128, y + 18, 168, y + 18, { color: "muted" });
            ps.push(G.box(172, y, 130, 36, c, { stroke: "k", color: "k", size: 12, rx: 5 }));
            G.label(316, y + 22, "save/restore memory, flush caches", { size: 11 });
          });
          G.text(24, 354, "merged into one report (hundreds of metrics)", { size: 13, color: "ok" });
          G.label(24, 380, "slow, cold caches: use for WHY, not for how long", { size: 12, color: "hot" });
          G.from(ps, { opacity: 0, x: -12, stagger: 0.08, duration: 0.25 });
          G.caption("a few counters per pass, so the kernel runs again and again");
        } },

      { rail: "speed of light", title: "First look: Speed of Light",
        body: `<p>Open an ncu report and start at the <b>GPU Speed of Light</b> section. It reports two numbers, each as a percentage of the GPU's <b>peak</b> for that resource:</p>
<ul><li><b>Memory throughput %:</b> how close the busiest part of the memory system (L1, L2 or DRAM) is to its limit.</li>
<li><b>Compute (SM) throughput %:</b> how close the busiest arithmetic or issue pipeline is to its limit.</li></ul>
<p>These two numbers sort kernels into four cases:</p>
<table><tr><th>memory %</th><th>compute %</th><th>diagnosis</th><th>look at next</th></tr>
<tr><td>high</td><td>low</td><td>memory-bound</td><td>can you move fewer bytes? Memory Workload (P5.2: sectors, reuse)</td></tr>
<tr><td>low</td><td>high</td><td>compute-bound</td><td>instruction mix, tensor cores (P5.7)</td></tr>
<tr><td>high</td><td>high</td><td>well balanced</td><td>close to done: only a smarter algorithm helps</td></tr>
<tr><td>low</td><td>low</td><td><b>latency-bound</b></td><td>occupancy, stalls, grid too small</td></tr></table>
<p>Example: memory 15%, compute 10%. Neither resource is busy, so neither is the limit. The warps are mostly <i>waiting</i>: for memory they can't hide, because too few are resident (P5.1 occupancy), or for a dependency, or the grid is too small to fill the GPU. You check occupancy and the stall reasons next.</p>
<p>This is the same idea as the roofline from P1.4, measured instead of calculated, and it should be your first look every time, <b>before</b> optimizing the top stall reason.</p>`,
        check: { q: "Speed of Light shows memory 15% and compute 10%. What is the right next step?",
          options: ["Rewrite the arithmetic to use fewer instructions", "The kernel is latency-bound: check achieved occupancy, stall reasons and whether the grid fills the GPU", "Nothing: low numbers mean the kernel is efficient"], answer: 1,
          why: "Both resources are mostly idle, so neither is the bottleneck. That signature means warps are waiting (latency) or there is too little parallel work. Occupancy, warp-stall reasons and grid size explain it. Cutting instructions would not help because compute is not limiting." },
        scene(G) {
          const x0 = 70, y0 = 60, w = 400, h = 280;
          G.axes(x0, y0, w, h, {});
          G.label(x0 + w / 2 - 40, y0 + h + 22, "memory throughput %", { size: 12 });
          G.label(24, y0 - 8, "compute %", { size: 12 });
          [0, 50, 100].forEach((v) => { G.label(x0 + v / 100 * w, y0 + h + 12, String(v), { size: 10, anchor: "middle" }); G.label(x0 - 8, y0 + h - v / 100 * h + 4, String(v), { size: 10, anchor: "end" }); });
          G.line(x0 + w / 2, y0, x0 + w / 2, y0 + h, { color: "line", dash: "4 4" });
          G.line(x0, y0 + h / 2, x0 + w, y0 + h / 2, { color: "line", dash: "4 4" });
          G.label(x0 + 10, y0 + h - 12, "latency-bound", { size: 12, color: "hot" });
          G.label(x0 + w - 100, y0 + h - 12, "memory-bound", { size: 12, color: "v" });
          G.label(x0 + 10, y0 + 18, "compute-bound", { size: 12, color: "k" });
          G.label(x0 + w - 78, y0 + 18, "balanced", { size: 12, color: "ok" });
          const d = G.circle(x0 + 0.15 * w, y0 + h - 0.10 * h, 8, { fill: "hot" });
          G.label(x0 + 0.15 * w + 14, y0 + h - 0.10 * h - 10, "15 / 10", { size: 12, color: "hot" });
          G.pulse(d, { lo: 0.4, hi: 1, d: 0.6, repeat: -1 });
          G.text(490, 150, "first look", { size: 13 });
          G.label(490, 172, "at every report", { size: 12 });
          G.caption("two percentages place the kernel in one of four regions");
        } },

      { rail: "memory", title: "Memory Workload: confirm your sector counts",
        body: `<p>If Speed of Light says memory-bound, or you suspect a bad access pattern, open <b>Memory Workload Analysis</b>. It reports the ideas from P5.2 as measured numbers: <b>sectors per request</b>, L1 and L2 hit rates, shared-memory bank conflicts, and DRAM bytes.</p>
<p>The README's <i>predict first</i> exercise is a good drill. A kernel assigns one <b>row</b> to each thread; rows are 256 floats (1 KB) apart. In one warp request, lane <i>i</i> reads <code>m[(base + i) * 256 + c]</code>:</p>
<div class="eq">lane spacing = 256 floats x 4 B = 1024 B
  (a sector is 32 B, so every lane is in its own)
sectors / request = 32
useful / fetched  = 128 B / 1024 B = 12.5%
expected speed vs a coalesced version
  &lt;= 1/8  (upper-bound model, caches may soften it)</div>
<p>Then check with ncu: the metric <code>l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum</code> divided by <code>l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum</code> should read close to 32 (versus 4 for the coalesced case). If your hand count and the profiler agree, you understand the access pattern. If they disagree, one of your assumptions is wrong, and that is the interesting part.</p>
<p>The same section reports <b>bank conflicts</b> (<code>l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum</code>), which exercise 3 of P5.2 uses. Section <b>Occupancy</b> compares <i>theoretical</i> occupancy (what registers, shared memory and block size allow, P5.1) with <i>achieved</i> occupancy (what actually ran) and names the limiter.</p>`,
        scene(G) {
          G.text(24, 36, "one warp request, rows 1 KB apart", { size: 13 });
          const lanes = [];
          for (let i = 0; i < 16; i++) lanes.push(G.circle(46 + i * 36, 66, 6, { fill: "k" }));
          for (let i = 0; i < 16; i++) G.line(46 + i * 36, 74, 46 + i * 36, 110, { color: "k", w: 1 });
          const cells = G.grid(24, 110, 1, 16, 33, 24, () => "blue", { gap: 3 });
          G.label(24, 150, "16 of the 32 lanes shown: each lane pulls its own 32 B sector", { size: 12 });
          G.text(24, 200, "predicted: 32 sectors / request, 12.5%", { size: 14, color: "hot" });
          G.text(24, 250, "what to measure in ncu:", { size: 13 });
          G.label(24, 276, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", { size: 11, color: "k" });
          G.label(24, 294, "  / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", { size: 11, color: "k" });
          G.label(24, 330, "coalesced version: 4.  This one should read close to 32.", { size: 12, color: "ok" });
          G.label(24, 354, "hand count == profiler  ->  you understand the pattern", { size: 12 });
          G.from(cells, { opacity: 0, stagger: 0.02, duration: 0.2 });
          G.caption("predict first, then let the counter confirm");
        } },

      { rail: "stalls", title: "Warp stall reasons: what are the warps waiting for?",
        body: `<p>On every cycle each scheduler picks an <b>eligible</b> warp and issues one instruction. A warp that cannot issue is <b>stalled</b>, and ncu samples <i>why</i>. The section <b>Warp State Statistics</b> (and the source view) shows the stall reasons. The ones you will meet most:</p>
<table><tr><th>stall reason</th><th>the warp is waiting for</th><th>typical response</th></tr>
<tr><td><b>Long Scoreboard</b></td><td>a global (or local) memory load to return</td><td>better coalescing and reuse (P5.2), more resident warps or more independent loads per thread to hide latency</td></tr>
<tr><td><b>Short Scoreboard / MIO Throttle</b></td><td>shared memory, special-function units</td><td>bank conflicts, fewer shared accesses</td></tr>
<tr><td><b>LG Throttle</b></td><td>the load/store queue is full</td><td>fewer, wider memory instructions (<code>float4</code>)</td></tr>
<tr><td><b>Barrier</b></td><td>other threads at <code>__syncthreads()</code></td><td>balance work across threads of the block</td></tr>
<tr><td><b>Math Pipe Throttle</b></td><td>an arithmetic pipeline that is busy</td><td>the kernel really is compute-heavy</td></tr></table>
<p>Read the <i>top</i> reason as a hint about <i>which part of the machine is the bottleneck</i>, then fix the cause (the access pattern, the occupancy), not the symptom. If Speed of Light says the kernel is near peak, the stall list is not worth chasing.</p>
<p>The <b>source view</b> attaches stall samples to the CUDA or SASS line that incurred them. This is where <code>-lineinfo</code> pays off: you see which load or atomic is the hot spot.</p>`,
        check: { q: "The top stall reason of a kernel is Long Scoreboard. What does it mean?",
          options: ["Warps wait on global-memory loads", "Warps wait at __syncthreads()", "The instruction cache is cold"], answer: 0,
          why: "A scoreboard tracks outstanding results; the \"long\" one tracks loads from global or local memory (hundreds of cycles). Fixes: touch fewer bytes per useful result (coalescing, shared-memory reuse) or keep more work in flight (occupancy, independent loads per thread) so other warps cover the wait." },
        scene(G) {
          G.text(24, 36, "shape of a Warp State report (illustrative, not data)", { size: 13 });
          const items = [["Long Scoreboard", 62, "hot"], ["LG Throttle", 14, "v"], ["Barrier", 9, "q"], ["Short Scoreboard", 7, "blue"], ["Selected (issuing)", 8, "ok"]];
          const bars = [];
          items.forEach(([n, v, c], i) => {
            const y = 64 + i * 52;
            G.label(24, y + 18, n, { size: 12, color: i === 0 ? "hot" : "muted" });
            bars.push(G.rect(190, y, v * 6, 28, { fill: c, rx: 3 }));
            G.label(194 + v * 6, y + 19, v + "%", { size: 12 });
          });
          G.label(24, 340, "long scoreboard on top: waiting for global memory", { size: 12, color: "hot" });
          G.label(24, 362, "-> go back to sectors per request and occupancy", { size: 12 });
          G.from(bars, { scaleX: 0, transformOrigin: "0% 50%", stagger: 0.1, duration: 0.4 });
          G.caption("the biggest bar names the part of the machine to look at");
        } },

      { rail: "% of peak", title: "From counters to one number: % of peak",
        body: `<p>Every benchmark in this repo ends in a table with a <b>% of peak</b> column, because "12 ms" means nothing without a yardstick. For a memory-bound kernel the yardstick is bandwidth:</p>
<div class="eq">achieved GB/s = bytes moved / time
% of peak     = achieved / peak GB/s

worked example (made-up time):
  copy of 8192 x 8192 floats: 2 x 256 MiB
                              = 0.537 GB
  time 2.1 ms  -> 0.537 / 0.0021 = 256 GB/s
  vs example peak 320 GB/s (T4, UNVERIFIED)
  -> 256 / 320 = 80% of peak</div>
<p>For a compute-bound kernel use FLOP/s against the arithmetic peak (P5.6 does this for GEMM). Where does <i>peak</i> come from? Two honest sources: the cited spec (<code>P1.4 gpu_specs.yaml</code>, status UNVERIFIED until checked) and a <b>measured</b> copy or FMA micro-benchmark on your own GPU, which the harness prints (<code>s2s::measure_copy_gbs()</code>). The measured copy number is usually the fairer ceiling, because real DRAM cannot be driven at the spec figure.</p>
<p>Exercise 3 builds the glue: <code>platform/kernels/bench/ncu_to_table.py</code> turns <code>ncu --csv</code> output (time, <code>dram__bytes.sum</code>) into a <code>| kernel | time | GB/s | % of copy | TFLOP/s | % of peak |</code> row. Its test uses a <i>synthetic</i> CSV, so it runs on a laptop with no GPU. One thing to check on a GPU: do ncu's DRAM bytes match the bytes you assumed? If they differ, explain it (L2 hits, write-allocate behaviour).</p>
<p>The <b>roofline</b> chart in ncu plots the same thing on the P1.4 axes: your kernel's point against the memory slope and the compute ceiling.</p>`,
        scene(G) {
          G.text(24, 36, "bytes moved / time  vs  peak (example values)", { size: 13 });
          G.label(24, 62, "peak (example, T4: 320 GB/s, UNVERIFIED)", { size: 12 });
          G.rect(24, 70, 580, 30, { fill: "line", rx: 4 });
          G.rect(24, 70, 580 * 0.8, 30, { fill: "ok", rx: 4 });
          G.label(30, 90, "achieved 256 GB/s = 80%", { size: 12, color: "ink" });
          G.text(24, 150, "0.537 GB / 2.1 ms = 256 GB/s", { size: 15, color: "ink" });
          G.label(24, 176, "(2 x 8192 x 8192 x 4 B = 536,870,912 B; the 2.1 ms is made up)", { size: 11 });
          G.text(24, 236, "where does peak come from?", { size: 13 });
          G.box(24, 252, 270, 44, "cited spec sheet (UNVERIFIED)", { stroke: "v", color: "v", size: 12, rx: 6 });
          G.box(330, 252, 270, 44, "measured copy kernel (fairer)", { stroke: "ok", color: "ok", size: 12, rx: 6 });
          G.label(24, 340, "ncu --csv -> ncu_to_table.py -> | kernel | GB/s | % of peak |", { size: 12, color: "k" });
          G.label(24, 364, "commit CSV summaries; never commit .ncu-rep files", { size: 12 });
          G.caption("a number is only meaningful next to its yardstick");
        } },

      { rail: "on the cloud", title: "Getting it running on a cloud GPU",
        body: `<p>Nsight Compute reads GPU performance counters, and on Linux those are <b>admin-only by default</b>. Without permission ncu stops with <code>ERR_NVGPUCTRPERM</code>. There are two fixes:</p>
<ul><li><b>Quick:</b> run it as root, keeping your PATH: <code>sudo $(which ncu) ...</code>.</li>
<li><b>Persistent:</b> set the driver option <code>NVreg_RestrictProfilingToAdminUsers=0</code> in <code>/etc/modprobe.d/</code> and reload the driver or reboot (the shared P5 page has the exact lines). Acceptable on a single-user lab box, <b>never on a shared host</b>, because it lets any user read counters that can leak information about other users' work.</li></ul>
<p>Reports are large binary files (<code>.ncu-rep</code>, <code>.nsys-rep</code>). Workflow on the AWS box (see <code>aws.md</code>): write the report with <code>-o</code>, pack and print it (<code>tar czf - *.ncu-rep | base64</code>) through the SSM session since no inbound ports are open, decode on your laptop, and open it in the free desktop UI. They are git-ignored; commit only CSV summaries.</p>
<p>The common mistakes, all from this lesson: profiling a debug build; treating ncu durations as end-to-end time; chasing the top stall reason without reading Speed of Light; leaving the counter restriction off on a shared machine. And the AWS rules from the course apply: the module needs about 2 instance-hours on a <code>g4dn.xlarge</code>, the instance auto-stops after 30 idle minutes, and <code>make down</code> tears it down.</p>`,
        check: { q: "ncu fails with ERR_NVGPUCTRPERM on your single-user lab instance. What are the two fixes?",
          options: ["Rebuild with -G, or lower the clock", "Run ncu under sudo, or set the nvidia module option NVreg_RestrictProfilingToAdminUsers=0 and reload the driver", "Install a newer CUDA toolkit, or disable ECC"], answer: 1,
          why: "The error means your user is not allowed to read GPU performance counters. sudo lets ncu run as root; the module option lifts the restriction for everyone, which is fine on a single-user box and unsafe on a shared one." },
        scene(G) {
          G.box(24, 56, 250, 150, "", { stroke: "line", rx: 8 });
          G.text(36, 80, "cloud GPU box (SSM)", { size: 13 });
          G.box(40, 94, 218, 30, "sudo $(which ncu) -o rep ...", { fill: "q", size: 11, rx: 4 });
          G.box(40, 134, 218, 30, "rep.ncu-rep (large, binary)", { stroke: "v", color: "v", size: 11, rx: 4 });
          G.box(40, 170, 218, 28, "tar czf - *.ncu-rep | base64", { stroke: "line", size: 11, rx: 4 });
          G.arrow(280, 130, 346, 130, { color: "ok", w: 2 });
          G.label(284, 118, "copy text", { size: 11 });
          G.box(352, 56, 250, 150, "", { stroke: "line", rx: 8 });
          G.text(364, 80, "your laptop", { size: 13 });
          G.box(368, 94, 218, 30, "base64 -d | tar xz", { stroke: "line", size: 11, rx: 4 });
          G.box(368, 134, 218, 30, "Nsight Compute UI", { fill: "ok", size: 11, rx: 4 });
          G.text(24, 258, "ERR_NVGPUCTRPERM = counters are admin-only", { size: 13, color: "hot" });
          G.label(24, 284, "fix 1: sudo       fix 2: NVreg_RestrictProfilingToAdminUsers=0 + reload", { size: 11 });
          G.label(24, 306, "fix 2 only on a single-user box, never on a shared host", { size: 12, color: "hot" });
          G.label(24, 346, "~2 instance-hours on g4dn.xlarge · auto-stop 30 idle min · make down", { size: 12, color: "v" });
          G.caption("no inbound ports: reports travel back as text through the SSM session");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Amdahl's law: a speed-up helps only the share of time it touches, so find the biggest slice of the timeline first.</li>
<li>Nsight Systems answers whole-program questions (gaps, launches, copies, NVTX ranges); Nsight Compute answers why one kernel is slow, by replaying it with cold caches, which is why its times are not end-to-end times.</li>
<li>Read an ncu report top-down: Speed of Light (memory % vs compute %, low and low means latency-bound), Memory Workload (sectors per request, bank conflicts), Occupancy (theoretical vs achieved), warp stall reasons (Long Scoreboard = waiting on global memory), then the source view.</li>
<li>Report bytes moved over time against a yardstick: spec peak (UNVERIFIED until cited) or a measured copy kernel.</li>
<li>On a cloud box, counters need sudo or <code>NVreg_RestrictProfilingToAdminUsers=0</code> (single-user only); reports come back as text and are never committed.</li></ul>`,
    sim: {
      title: "Amdahl and % of peak",
      intro: "Two numbers decide whether a kernel is worth tuning. Top: the whole-program speed-up you get from making a kernel that takes share <i>s</i> of the run <i>x</i> times faster (Amdahl's law). Bottom: how close a kernel is to a bandwidth peak, from bytes moved and time. The default peak is an example value (T4: 320 GB/s, gpu_specs.yaml, UNVERIFIED): replace it with your GPU's measured copy bandwidth.",
      height: 330,
      controls: [
        { id: "s", label: "kernel share of the run, %", min: 1, max: 100, step: 1, value: 30 },
        { id: "x", label: "kernel speed-up factor x", min: 1, max: 50, step: 0.5, value: 4 },
        { id: "mb", label: "bytes moved, MiB", min: 1, max: 4096, step: 1, value: 512 },
        { id: "ms", label: "kernel time, ms", min: 0.05, max: 20, step: 0.05, value: 2.1 },
        { id: "peak", label: "peak bandwidth, GB/s (example)", min: 50, max: 1000, step: 10, value: 320 },
      ],
      draw(G, v) {
        const s = v.s / 100, x = v.x;
        const sp = amdahl(s, x), cap = 1 / (1 - s + 1e-9);
        const gx = 40, gy = 24, gw = 560, gh = 150;
        G.label(gx, 14, "overall speed-up vs kernel speed-up (this share s)", { size: 11 });
        G.axes(gx, gy, gw, gh, {});
        const maxY = Math.max(1.5, Math.min(cap, 12));
        const px = (xv) => gx + ((xv - 1) / 49) * gw, py = (yv) => gy + gh - ((yv - 1) / (maxY - 1)) * gh;
        let d = "";
        for (let k = 0; k <= 49; k += 0.5) d += (k ? " L " : "M ") + px(1 + k) + " " + py(Math.min(amdahl(s, 1 + k), maxY));
        G.path(d, { color: "k", w: 2.5 });
        if (cap <= maxY) { G.line(gx, py(cap), gx + gw, py(cap), { color: "hot", dash: "4 4" }); G.label(gx + gw - 4, py(cap) - 5, "ceiling " + F.num(cap, 2) + "x", { size: 11, color: "hot", anchor: "end" }); }
        G.circle(px(x), py(Math.min(sp, maxY)), 6, { fill: "v" });
        G.label(gx, gy + gh + 16, "kernel speed-up 1x ... 50x", { size: 10 });
        const bytes = v.mb * 1048576, gbs = bytes / (v.ms * 1e6), pct = gbs / v.peak;
        G.label(10, 224, "bandwidth: achieved vs peak", { size: 11 });
        G.rect(10, 234, 600, 26, { fill: "line", rx: 4 });
        G.rect(10, 234, Math.max(2, Math.min(1, pct) * 600), 26, { fill: pct > 1 ? "hot" : pct >= 0.8 ? "ok" : "v", rx: 4 });
        G.label(16, 252, `${F.num(gbs, 1)} GB/s of ${F.num(v.peak)} GB/s`, { size: 12, color: "ink" });
        if (pct > 1) G.label(10, 282, "above peak: wrong bytes, wrong time, or the data sits in L2", { size: 11, color: "hot" });
        return [
          { title: "Amdahl", rows: [["kernel share s", v.s + "%"], ["kernel speed-up", F.num(x, 1) + "x"], ["overall speed-up", F.num(sp, 3) + "x"], ["ceiling at x = inf", F.num(cap, 2) + "x"]],
            gauge: [[Math.min(1, (sp - 1) / Math.max(0.001, cap - 1)), "ok"]], gaugeText: "share of the possible gain reached",
            chip: [s >= 0.2, s >= 0.2 ? "worth tuning" : "small slice: look elsewhere first"] },
          { title: "% of peak", rows: [["bytes moved", F.bytes(bytes)], ["time", F.num(v.ms, 2) + " ms"], ["achieved", F.num(gbs, 1) + " GB/s"], ["% of peak", F.num(pct * 100, 1) + "%"]],
            gauge: [[Math.min(1, pct), pct >= 0.8 ? "ok" : "v"]], gaugeText: "of the example peak",
            html: `<p class="note">Memory-bound kernels only. A transpose or copy is judged against a measured copy kernel (exit check in P5.2: at least 80%).</p>` },
        ];
      },
      note: "Amdahl's law assumes the rest of the program is unchanged and the kernel really speeds up by x. Bandwidth uses bytes you assume the kernel moves; ncu's dram__bytes.sum may differ (cache hits).",
    },
    practice: {
      intro: "Needs a GPU with counter access for exercises 1 and 2 (see the shared P5 page: <code>ERR_NVGPUCTRPERM</code>). Exercise 3 is plain Python and runs anywhere. Build once from the repository root: <code>cmake -S course/P5-cuda-deep-dive -B build/p5 -DCMAKE_CUDA_ARCHITECTURES=native &amp;&amp; cmake --build build/p5 -j</code>. Do exercise 1 before reading the answers at the bottom of <b>01_planted_slow.cu</b>.",
      items: [
        { title: "Diagnose three planted slow kernels from ncu alone", tier: "T2 · medium", goal: "Profile <b>reference</b>, <b>mystery_a</b>, <b>mystery_b</b>, <b>mystery_c</b>. For each, fill the table: Speed-of-Light memory/compute %, the section that gave it away, the metric value, the diagnosis, the fix. Then implement the fixes and show the metric that moved. Predict first: a row-per-thread kernel with rows 1 KB apart.",
          cmd: "sudo $(which ncu) --set full -o results/planted ./build/p5/p5.3_01_planted_slow" },
        { title: "NVTX-annotate the #0 inference loop", tier: "T2 · medium", goal: "Run <b>02_nvtx_ranges</b> under nsys, then add <b>prefill</b>, <b>decode step</b>, <b>attention</b> and <b>mlp</b> ranges to platform/engine/v0/src/engine.hpp behind <b>#ifdef S2S_NVTX</b>. Profile a 32-token generation: which phase dominates, and does a decode step grow with position? Deliverable: a screenshot and 3-5 sentences.",
          cmd: "nsys profile -o results/nvtx ./build/p5/p5.3_02_nvtx_ranges" },
        { title: "ncu --csv to the \"% of peak\" column", tier: "T0 + T2 · easy", goal: "Read <b>platform/kernels/bench/ncu_to_table.py</b> and make its test pass (units, thousands separators). On T2 run it on the transpose ladder's CSV with the measured copy peak, check ncu's DRAM bytes against 8 x rows x cols, and add an achieved-occupancy column with a test.",
          cmd: "uv run pytest course/P5-cuda-deep-dive/P5.3-profiling-with-nsight/exercises" },
      ],
      labs: [
        { label: "Examples: planted slow kernels, NVTX ranges, ncu CSV to table", path: "course/P5-cuda-deep-dive/P5.3-profiling-with-nsight/examples/" },
        { label: "AWS walkthrough (g4dn.xlarge, about 2 instance-hours; cost, auto-stop and teardown on the shared page)", path: "course/P5-cuda-deep-dive/P5.3-profiling-with-nsight/aws.md" },
        { label: "Shared P5 setup: build, Nsight counter access (ERR_NVGPUCTRPERM), teardown", path: "course/P5-cuda-deep-dive/aws-common.md" },
        { label: "ncu CSV to bench table script", path: "platform/kernels/bench/ncu_to_table.py" },
        { label: "Engine #0 loop to annotate", path: "platform/engine/v0/src/engine.hpp" },
      ],
    },
  });
})();
