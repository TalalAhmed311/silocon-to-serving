/* P6.6 — CUDA graph capture: launch-bound decode, what a graph records, the capture rules, static shapes in
   TorchRunner.decode, buckets and padding, what stays eager, and a launch-vs-GPU time model.
   Mirrors platform/engine/v1/s2s_engine/cuda_graph.py and model_runner.py. No GPU in this environment: nothing here is
   measured; every time is arithmetic on stated example values (see the simulator note). */
(function () {
  const F = S2S.fmt;
  const cell = (G, x, y, w, h, t, color, o = {}) => G.box(x, y, w, h, t, Object.assign({ fill: color, size: 11, rx: 3 }, o));
  const WEIGHTS_8B = 16e9; // Llama-3-8B bf16: ~8.0B params x 2 bytes (P1.2)
  const KV_TOKEN = 131072; // 128 KiB of KV per token (P1.2)
  const bucketFor = (list, B) => { const s = list.slice().sort((a, b) => a - b); const x = s.find((v) => v >= B); return x === undefined ? null : x; };
  const meanPad = (list, n = 64) => { let t = 0; for (let B = 1; B <= n; B++) { const b = bucketFor(list, B); t += (b === null ? B : b) - B; } return t / n; };

  S2S.lesson({
    id: "p6-6", n: "P6.6", title: "CUDA graph capture",
    subtitle: "Engine internals · first principles · T2 one GPU for timings; the rules and the model are T0",
    kicker: "Lesson · ≈ 40 min",
    headline: "Record the launches once, replay them with one call",
    intro: `<p>At batch 1 a decode step does very little arithmetic per kernel, and the GPU can finish a kernel faster than the CPU can launch the next one. Then the step is limited by the CPU, not the GPU. A <b>CUDA graph</b> records the whole sequence of kernel launches once and replays it with a single call. This lesson explains why that helps, what the recording forbids and how the engine's decode step is written to obey, how batch sizes are padded to a few captured <i>buckets</i>, what has to stay outside the graph, and when graphs stop mattering. The reference is <code>platform/engine/v1/s2s_engine/cuda_graph.py</code>. <b>This environment has no GPU:</b> every timing in the lesson is arithmetic on stated example values, and the bench numbers are left as <code>TODO(run-on: g6.xlarge)</code> for you to measure.</p>`,
    facts: ["8 steps", "5 checkpoints", "1 simulator", "3 exercises"],
    legend: [["k", "GPU kernel"], ["v", "CPU launch"], ["hot", "idle gap / padding / wasted"], ["ok", "graph / valid"], ["w", "static buffer"]],
    prev: "p6-5", next: "p6-7",
    steps: [
      { rail: "launch-bound", title: "When the GPU waits for the CPU",
        body: `<p>A GPU runs <b>kernels</b>: small programs launched by the CPU. The CPU enqueues a launch into a queue; the GPU executes the queue in order, asynchronously. As long as the queue is never empty, the GPU is busy and the CPU's launch work is hidden.</p>
<p>A Llama decode step at batch 1 is dozens of kernels per layer (norms, projections, RoPE, attention, MLP, residuals) times 32 layers: a few thousand kernels (the module README's estimate; count yours with <code>nsys</code>). Each kernel is memory-bound and short. And each <b>eager</b> launch costs CPU time: the Python call, PyTorch's dispatcher, the driver. If launching a kernel takes longer than running it, the queue drains and the GPU idles between kernels. The step is <b>launch-bound</b>: its time is the CPU's time to enqueue, not the GPU's time to compute.</p>
<div class="eq">example values, not measured:
  N = 2,000 kernels per step
  L = 5 us CPU time per eager launch
  CPU enqueue time = N * L = 10 ms
  GPU compute time (small batch) = 3 ms
  step = max(10, 3) = 10 ms   (GPU idle 70 %)</div>
<p>In an <code>nsys</code> timeline this shows as gaps between kernels on the GPU row. The inter-token latency of every sequence in the batch includes this overhead, and nothing you do to the kernels themselves fixes it. A second problem is the other direction: if the GPU is the slower one (big model, big batch), the launches hide behind it and graphs gain nothing. Step 8 gives the rule.</p>`,
        check: { q: "Eager decode at batch 1: 2,000 kernels, 5 us of CPU time per launch, 3 ms of GPU compute in total. About how long is the step, and why?",
          options: ["3 ms: the GPU time is all that counts", "About 10 ms: the CPU needs 2,000 × 5 us = 10 ms to enqueue the kernels, and the GPU waits for them", "13 ms: the two times add", "5 us"], answer: 1,
          why: "Launches and execution overlap, so the step takes about the longer of the two, not the sum. CPU enqueue 10 ms beats GPU 3 ms, so the GPU spends most of the step idle between kernels. The numbers are example values to show the regime; the real ones come from your nsys trace." },
        scene(G) {
          G.text(24, 30, "eager decode, small batch (illustrative, not measured)", { size: 13 });
          G.label(24, 62, "CPU: launch, launch, launch…", { size: 11, color: "v" });
          const cp = [], gp = [];
          const n = 12, dx = 46;
          for (let i = 0; i < n; i++) cp.push(cell(G, 24 + i * dx, 70, 40, 24, "L" + (i + 1), "v", { size: 10 }));
          G.label(24, 138, "GPU: each kernel finishes before the next arrives", { size: 11, color: "k" });
          for (let i = 0; i < n; i++) {
            const x = 24 + i * dx + 24;
            gp.push(cell(G, x, 146, 18, 24, "", "k"));
            if (i < n - 1) G.rect(x + 18, 146, dx - 18, 24, { stroke: "hot", dash: "2 2", rx: 2, opacity: 0.9 });
          }
          G.label(24, 196, "red dashed = GPU idle, waiting for the next launch", { size: 12, color: "hot" });
          G.text(24, 244, "step time = max(CPU enqueue, GPU compute)", { size: 14, color: "ink" });
          G.label(24, 276, "CPU: N × L = 2,000 × 5 us = 10 ms   (example values)", { size: 12 });
          G.label(24, 298, "GPU: 3 ms of actual work", { size: 12 });
          G.text(24, 338, "→ about 10 ms, GPU idle 70 %", { size: 15, color: "hot", weight: 700 });
          G.label(24, 372, "fix: don't launch 2,000 times. Launch once.", { size: 13, color: "ok" });
          G.from([...cp, ...gp], { opacity: 0, x: -8, stagger: 0.03, duration: 0.2 });
          G.caption("short kernels + slow launches = a GPU that spends its time waiting");
        } },

      { rail: "the idea", title: "A graph records the launches; replay is one call",
        body: `<p>A <b>CUDA graph</b> is a recorded list of GPU operations (kernels, memory copies) with their arguments and their dependencies. You <b>capture</b> it once by running the work while the driver records instead of executing; then you <b>replay</b> it as often as you like. Replay submits the whole recording to the GPU in one call. The thousands of Python-dispatcher-driver round trips become one.</p>
<p>The module's first example makes the idea small enough to see (<code>examples/01_capture_replay.py</code>): a toy step of 50 repetitions of <code>silu(x @ w) * 0.5 + x</code>. That is a matmul, a <code>silu</code>, a multiply and an add per repetition: 4 kernels × 50 = <b>200 tiny kernels</b>, at a matrix size where each kernel is trivial. The script captures it, replays it, checks <code>torch.allclose(replay, eager)</code>, then prints µs per call for both. Replay is much cheaper at this size: the 200 launches collapse to one, while the GPU does the same work. (I have no GPU here, so the numbers are yours to measure: exercise 1.)</p>
<p>What changes, and what does not:</p>
<ul><li><b>Same:</b> the kernels, their order, their results. Replay computes exactly what eager would.</li>
<li><b>Different:</b> the CPU's work. Per-kernel Python and dispatcher cost disappears; the GPU still runs the same kernels, now back to back.</li>
<li><b>Frozen:</b> everything the recording baked in: kernel launch shapes, tensor addresses, the path through the code. The next two steps are about living with that.</li></ul>`,
        scene(G) {
          G.text(24, 30, "eager vs graph: the same 200 kernels", { size: 13 });
          G.label(24, 54, "eager: 200 launches", { size: 12, color: "v" });
          const a = [];
          for (let i = 0; i < 24; i++) a.push(G.line(24 + i * 6, 64, 24 + i * 6, 110, { color: "v", w: 1.5, opacity: 0.9 }));
          G.label(24 + 24 * 6 + 6, 92, "… ×200 CPU calls", { size: 11 });
          G.rect(24, 112, 24 * 6 + 20, 30, { fill: "k", rx: 4 });
          G.label(30, 131, "GPU: 200 short kernels", { size: 11, color: "bg" });
          G.label(330, 54, "graph: 1 launch", { size: 12, color: "ok" });
          G.arrow(380, 64, 380, 110, { color: "ok", w: 3 });
          G.label(392, 92, "replay()", { size: 12, color: "ok" });
          G.rect(330, 112, 285, 30, { fill: "k", rx: 4 });
          G.label(336, 131, "GPU: the same 200 kernels, back to back", { size: 11, color: "bg" });
          G.text(24, 188, "capture once: record", { size: 13, weight: 700 });
          const nodes = [];
          for (let r = 0; r < 4; r++) for (let c = 0; c < 10; c++) nodes.push(G.rect(24 + c * 20, 200 + r * 18, 16, 14, { fill: ["k", "q", "v", "ok"][r], rx: 2 }));
          G.label(240, 218, "each square: one kernel node", { size: 11 });
          G.label(240, 238, "matmul, silu, mul, add, × 50 = 200 nodes", { size: 11 });
          G.label(240, 258, "with their arguments and addresses baked in", { size: 11 });
          G.text(24, 306, "replay: the whole recording in one submit", { size: 13, weight: 700, color: "ok" });
          G.label(24, 334, "same kernels, same order, same results; CPU work → one call", { size: 12, color: "ink" });
          G.label(24, 358, "frozen inside: launch shapes, tensor addresses, the code path", { size: 12, color: "hot" });
          G.label(24, 392, "toy step from examples/01_capture_replay.py; timings TODO(run-on: g6.xlarge)", { size: 10 });
          G.from(nodes, { opacity: 0, stagger: 0.01, duration: 0.15 });
          G.caption("a graph moves the launch cost from every step to once");
        } },

      { rail: "capture recipe", title: "The capture recipe: warm up, capture, copy in, replay",
        body: `<p>The recipe in <code>cuda_graph.py</code> has four moves, and the order matters:</p>
<div class="eq">s = torch.cuda.Stream()            # side stream
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):          # 1. warm-up
    for _ in range(2): decode(buf...)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, pool=pool): # 2. capture
    buf["out"] = decode(buf["tok"], ...)

buf["tok"][:n].copy_(tok)   # 3. copy INTO
buf["pos"][:n].copy_(pos)   #    the buffers
g.replay()                  # 4. one call
return buf["out"][:n]       #    read OUT</div>
<p><b>1. Warm-up.</b> The first call of many operations does one-off work: cuBLAS creates a handle, the allocator grows, kernels autotune. Do that inside a capture and it either fails or bakes junk into the graph. Run the step a couple of times <i>before</i> capturing, on a side stream.</p>
<p><b>2. Capture.</b> Inside <code>torch.cuda.graph(g)</code>, kernels are recorded, not run. Every tensor the step reads or writes is a <b>static buffer</b>: <code>buf["tok"]</code>, <code>buf["pos"]</code>, <code>buf["tables"]</code> are created once, and <code>buf["out"]</code> is whatever tensor the captured step returned. Those buffers' device addresses are now part of the graph.</p>
<p><b>3. Copy in.</b> A new batch arrives as new tensors. The graph does not know about them; it reads the addresses it recorded. So you <code>copy_</code> the new values <i>into</i> the static buffers.</p>
<p><b>4. Replay</b>, then read the result out of the static output buffer. That buffer is the same memory on every replay, so copy anything you need to keep (the result you hand back is a slice of it, which the next replay overwrites).</p>`,
        scene(G) {
          const st = (x, y, n, t, sub, col) => { G.box(x, y, 136, 76, "", { stroke: col, rx: 10 }); G.text(x + 10, y + 22, n + "  " + t, { size: 13, weight: 700, color: col }); sub.forEach((s, i) => G.label(x + 10, y + 42 + i * 14, s, { size: 10 })); };
          st(24, 50, 1, "warm-up", ["side stream", "2 eager runs", "cuBLAS, allocator"], "w");
          st(176, 50, 2, "capture", ["torch.cuda.graph", "record, don't run", "bake in addresses"], "v");
          st(328, 50, 3, "copy in", ["buf.copy_(new)", "same addresses", "new values"], "k");
          st(480, 50, 4, "replay", ["g.replay()", "one launch", "read buf['out']"], "ok");
          [160, 312, 464].forEach((x) => G.arrow(x, 88, x + 14, 88, { color: "ink", w: 2 }));
          G.text(24, 176, "static buffers (allocated once, never rebound)", { size: 13, weight: 700 });
          const b = [["tok", "w"], ["pos", "w"], ["tables", "w"], ["out (result)", "ok"]];
          b.forEach(([t, c], i) => cell(G, 24 + i * 150, 190, 142, 34, t, c, { size: 11 }));
          G.label(24, 242, "shapes: [B]  ·  [B]  ·  [B, max_blocks]  ·  [B, vocab]", { size: 11 });
          G.label(24, 262, "inputs are copied INTO them; the output is read OUT of the last one", { size: 12, color: "ink" });
          G.text(24, 296, "frozen by capture:", { size: 13, weight: 700 });
          ["the kernels and their launch shapes", "the device addresses of every tensor", "the single code path that ran during capture"].forEach((t, i) => G.label(34, 320 + i * 22, "· " + t, { size: 12, color: "ink" }));
          G.caption("order matters: warm up first, copy into the captured buffers, then replay");
        } },

      { rail: "the rules", title: "What capture forbids, and why",
        body: `<p>Replay re-issues a recording. Anything that would have to be <i>different</i> at replay time breaks it. Four rules follow, each tied to a failure you can reproduce (exercise 1 has you break each one on purpose):</p>
<table><tr><th>rule</th><th>because</th><th>how decode obeys</th></tr>
<tr><td>fixed shapes</td><td>launch configurations are frozen</td><td>pad the batch to a captured size; pad block tables to <code>max_blocks</code></td></tr>
<tr><td>fixed addresses</td><td>pointers are baked in</td><td>copy into persistent buffers; read the output from one</td></tr>
<tr><td>no host sync</td><td>the CPU is not part of the replay</td><td>no <code>.item()</code>, <code>.cpu()</code>, printing a tensor</td></tr>
<tr><td>no data-dependent control flow</td><td>only the recorded path replays</td><td>masks (<code>masked_fill</code>) instead of Python <code>if</code></td></tr></table>
<p>Two failure modes deserve a name. <b>Rebinding vs copying:</b> <code>static_in = torch.randn(...)</code> creates a <i>new</i> tensor at a new address and points the Python name at it; the graph still reads the old address and silently computes on stale data. <code>static_in.copy_(x)</code> writes into the old address. Nothing crashes in the first case. <b>A hidden sync:</b> a <code>max(positions)</code> used to size a buffer calls <code>.item()</code> under the hood; at best capture errors out, at worst a value is frozen into the graph and replay uses it for every batch.</p>
<p>Also: don't hold a reference to a graph's output across replays and expect it to stay. It is the same memory, and the next replay overwrites it.</p>`,
        check: { q: "You capture a graph with static_in, then for each new batch you write static_in = torch.randn(...) and call g.replay(). What happens?",
          options: ["Replay uses the new tensor", "Capture fails with an error", "Replay silently computes on the old data in the captured address: the new tensor is never seen. Use static_in.copy_(new)", "Replay raises an out-of-memory error"], answer: 2,
          why: "The graph recorded the device address of the original tensor. Rebinding the Python name creates a new tensor elsewhere and leaves the captured address untouched, so every replay reads the same stale values. There is no error, which is what makes the bug dangerous. copy_ writes into the captured address." },
        scene(G) {
          G.text(24, 30, "the graph reads an ADDRESS, not a Python name", { size: 13 });
          G.rect(40, 70, 250, 130, { stroke: "line", rx: 8 });
          G.label(50, 88, "device memory", { size: 11 });
          G.box(56, 100, 100, 40, "B: new data", { fill: "v", size: 12, rx: 5 });
          G.box(170, 100, 100, 40, "A: old data", { fill: "w", size: 12, rx: 5 });
          G.box(330, 94, 130, 50, "graph", { fill: "k", size: 14, rx: 8 });
          G.arrow(330, 120, 274, 120, { color: "k", w: 2.5 });
          G.label(480, 112, "reads A (recorded)", { size: 12, color: "k" });
          G.label(40, 220, "static_in = torch.randn(...)", { size: 12, color: "hot" });
          G.label(40, 240, "→ the name now points at B. The graph still reads A.", { size: 12, color: "hot" });
          G.label(40, 262, "→ no error, stale results", { size: 12, color: "hot" });
          G.label(40, 296, "static_in.copy_(x)", { size: 12, color: "ok" });
          G.label(40, 316, "→ writes the new values into A. The graph reads them.", { size: 12, color: "ok" });
          G.text(24, 372, "four rules: shapes, addresses, no host sync, no data-dependent if", { size: 12, color: "ink" });
          G.caption("copy into the captured buffer, never rebind it");
        } },

      { rail: "static decode", title: "Writing decode so every shape is fixed",
        body: `<p>Look at what <code>TorchRunner.decode</code> must satisfy. Eager decode in P6.3's paged runner reads <i>each sequence's own</i> slots: a different number per sequence, a different shape every step. That cannot be captured. The static version:</p>
<ul><li>The block table is a padded matrix <code>[B, max_blocks]</code>, zeros where a sequence has no block (<code>execute()</code> builds it).</li>
<li>Attention reads <b>all</b> <code>S = max_blocks × block_size</code> slots of every row: <code>read = slots(block_tables, all_pos)</code>, a gather of shape <code>[B, S, kv_heads, head_dim]</code>.</li>
<li>A <b>mask</b> hides the slots that do not belong to the sequence: <code>mask = (all_pos &lt;= positions[:, None])</code>. Masked scores become <code>-inf</code> before the softmax (<code>masked_fill</code>), so they get weight 0.</li>
<li>The new token's K/V is written at <code>write = slots(block_tables, positions)</code>, all by tensor indexing, with no Python branch on values.</li></ul>
<p>Every tensor now has a shape that depends only on <code>B</code> and <code>max_blocks</code>, never on the data. That is "static shapes" in practice.</p>
<div class="eq">02_decode_itl.py: block_size 16, max_blocks 64
S = 64 * 16 = 1,024 slots per sequence per layer
position 100 uses 101 of them (10 %)
the other 923 are gathered, then masked out</div>
<p>That is the <b>price of static shapes</b> in this reference runner: the gather and the attention math always cover the maximum context, whatever the real one. It is acceptable for a teaching engine on a tiny model. Production engines avoid it with an attention kernel that takes <code>context_lens</code> and a block table directly (P5.8, the paged decode kernel; exercise 4 of P6.3) and capture graphs around the parts that really are static. Either way the rule is the same: whatever varies must be <i>data inside fixed shapes</i>, not the shapes themselves.</p>`,
        scene(G) {
          G.text(24, 30, "one row of the batch, one layer: S = 64 × 16 = 1,024 slots", { size: 13 });
          const x0 = 24, W = 592;
          G.label(x0, 58, "block table row: 7 real blocks, 57 zero-padded (max_blocks = 64)", { size: 11, color: "ink" });
          const used = 101 / 1024;
          G.rect(x0, 66, W, 36, { stroke: "line", rx: 4 });
          G.rect(x0, 66, W * used, 36, { fill: "k", rx: 4 });
          G.label(x0 + W * used + 6, 88, "101 slots hold this sequence (position 100)", { size: 11, color: "k" });
          G.label(x0, 124, "gathered every step: all 1,024 slots, [B, S, kv_heads, head_dim]", { size: 11 });
          G.text(x0, 166, "mask = (slot position ≤ current position)", { size: 13, color: "ink" });
          G.rect(x0, 180, W, 30, { stroke: "line", rx: 4 });
          G.rect(x0, 180, W * used, 30, { fill: "ok", rx: 4, opacity: 0.9 });
          G.rect(x0 + W * used, 180, W * (1 - used), 30, { fill: "hot", rx: 4, opacity: 0.35 });
          G.label(x0, 230, "green: attended · red: gathered, then masked to −inf (weight 0)", { size: 11 });
          G.text(x0, 276, "923 of 1,024 gathered slots (90 %) do no useful work", { size: 14, color: "hot" });
          G.label(x0, 306, "that is the cost of fixed shapes in this reference runner", { size: 12, color: "ink" });
          G.label(x0, 330, "a paged kernel reads context_lens + the block table instead (P5.8)", { size: 12 });
          G.label(x0, 354, "either way: what varies must be DATA inside fixed shapes", { size: 12, color: "ok" });
          G.caption("static shapes trade wasted work for a graph that can replay");
        } },

      { rail: "buckets", title: "Buckets: pad the batch up to a captured size",
        body: `<p>Shape depends on <code>B</code>, the number of sequences decoding this step, and <code>B</code> changes constantly (P6.2). You cannot capture every value cheaply, so you capture a <b>few</b> sizes, the <b>buckets</b>, and round the real batch up to the next one. <code>DecodeGraphs.run</code> does it with <code>bisect_left</code> and copies the real rows in, leaving the extra rows at the zeros it just wrote:</p>
<div class="eq">sizes = (1, 2, 4, 8, 16, 32, 64)
n = 5 sequences -> bucket 8 (3 padded rows)
buf["tok"].zero_(); buf["pos"].zero_()
buf["tables"].zero_()
buf["tok"][:5].copy_(tok)   # rows 5..7 stay 0
graphs[8].replay()
return buf["out"][:5]       # drop the padded rows</div>
<p>A padded row decodes token 0 at position 0 through a block table of zeros, so it writes a K/V into <b>slot 0 of block 0</b>. That is harmless only if no real sequence uses block 0, which is why the server reserves it when graphs are on (<code>server.py</code>: <code>eng.blocks._take()</code>). Forget it and a padded row quietly corrupts some sequence's first token.</p>
<p>Above the largest bucket (64), <code>run</code> falls back to eager: at that batch the step is long enough that launch overhead no longer matters (step 8).</p>
<p>How many rows of padding does a bucket list cost? If the running batch size is uniform in 1 to 64, the average padded rows per step for powers of two are 651 / 64 = <b>10.2</b> (I computed this with the same <code>bisect</code> rule). More buckets mean less padding but more capture time and memory, since every bucket owns a graph and its buffers; exercise 2 measures both.</p>`,
        check: { q: "The running batch has 5 sequences and the buckets are (1, 2, 4, 8). What does the engine do?",
          options: ["Replays the 4-graph twice", "Pads to 8, replays the 8-graph, and discards the 3 padded rows' outputs (their writes land in the reserved block 0)", "Runs eagerly because 5 was not captured", "Captures a new graph for 5"], answer: 1,
          why: "bisect_left finds the first bucket ≥ 5, which is 8. Rows 5 to 7 are zeros: token 0, position 0, block table of zeros, so they compute garbage that is ignored and write to slot 0 of block 0, the reserved block. Only a batch larger than the biggest bucket falls back to eager." },
        scene(G) {
          G.text(24, 30, "buckets (1, 2, 4, 8, 16, 32, 64) · batch of 5", { size: 13 });
          const bs = [1, 2, 4, 8, 16, 32, 64];
          bs.forEach((b, i) => G.box(24 + i * 84, 44, 76, 30, "graph " + b, { stroke: b === 8 ? "ok" : "line", color: b === 8 ? "ok" : "muted", size: 12, rx: 6, fill: b === 8 ? null : null }));
          G.label(24, 98, "5 sequences → first bucket ≥ 5 is 8", { size: 12, color: "ink" });
          const row = [];
          for (let i = 0; i < 8; i++) row.push(cell(G, 24 + i * 70, 112, 62, 44, i < 5 ? "seq " + (i + 1) : "pad", i < 5 ? "k" : "hot", { size: 12, rx: 5, opacity: i < 5 ? 1 : 0.6 }));
          G.label(24, 176, "padded rows: token 0, position 0, block table of zeros", { size: 12, color: "hot" });
          G.arrow(24 + 6.5 * 70, 158, 24 + 6.5 * 70, 196, { color: "hot", w: 2 });
          G.box(24 + 4 * 70, 198, 256, 34, "writes K/V to slot 0 of block 0", { stroke: "hot", color: "hot", size: 12, rx: 6 });
          G.label(24, 258, "→ block 0 is reserved when graphs are on (server.py)", { size: 12, color: "ink" });
          G.text(24, 306, "mean padded rows for B uniform in 1 … 64:", { size: 13 });
          G.text(24, 334, "powers of two: 651 / 64 = 10.2 rows", { size: 15, color: "v", weight: 700 });
          G.label(24, 362, "> 64: eager fallback (launch overhead no longer matters)", { size: 12 });
          G.label(24, 384, "more buckets: less padding, more capture time and memory (ex. 2)", { size: 12 });
          G.from(row, { opacity: 0, y: 8, stagger: 0.05, duration: 0.2 });
          G.caption("round up to the next captured size; extra rows are harmless garbage");
        } },

      { rail: "what stays eager", title: "What cannot go in the graph: prefill, and (for now) sampling",
        body: `<p>Only the <b>decode</b> step is captured. A real engine step mixes three kinds of work, and two of them stay outside:</p>
<ul><li><b>Prefill chunks</b> stay eager. Their shapes change with every chunk (how many tokens, how long the cached prefix is), so there is no small set of shapes to capture. They are also large kernels, where launch overhead is a small fraction of the time. In <code>TorchRunner.execute</code>, decodes are split from prefill chunks: prefills run eagerly, then the decodes go through <code>graphs.run</code>. (vLLM also captures <i>piecewise</i> graphs around attention so mixed batches get some of the benefit; see its <code>docs/design/cuda_graphs.md</code> at the pin, re-verify the path.)</li>
<li><b>Sampling</b> runs after the graph, eagerly, and then copies the token ids to the CPU. That is one more launch sequence and one device-to-host copy per step, which is worth moving inside the graph. <code>sample_torch</code> (P6.5) cannot be captured as written, for two reasons.</li></ul>
<p>First, <code>if (top_k &gt; 0).any():</code> is a <b>host sync</b>: a Python <code>if</code> on a tensor value. Rewrite it so every filter always runs and a disabled filter is a no-op by construction (<code>top_k = V</code> keeps everything). Second, <b>randomness</b>: a captured random generator replays the <i>same</i> random numbers on every replay, because the recording holds its state at capture time. PyTorch has graph-safe generator handling (see its CUDA graphs docs at your version), or you pass per-row Philox offsets as a static input tensor that you update before each replay. Exercise 3 asks for this.</p>
<p>The test that keeps all this honest is <code>test_torch_runner.py</code>: greedy outputs with graphs on (and block 0 reserved) and off must both equal the NumPy reference. A graph must change speed, never the answer.</p>`,
        check: { q: "Why is prefill usually left outside the captured graphs?",
          options: ["Prefill is too short to benefit", "Its shapes vary with every chunk (token count, context length), and its kernels are big enough that launch overhead is a small share of the step", "Prefill needs the CPU to run", "Graphs cannot run matrix multiplications"], answer: 1,
          why: "Capture needs a small set of fixed shapes. Prefill chunk sizes and context lengths vary with every request, so there is nothing to bucket, and large kernels already keep the GPU busy, so the saving would be small anyway. Decode at small batch is the opposite on both counts." },
        scene(G) {
          G.text(24, 30, "one engine step (TorchRunner.execute)", { size: 13 });
          const seg = [["copy_ops", 80, "w", ""], ["prefill chunks", 170, "v", "eager: shapes vary"], ["decode", 150, "ok", "graphs.run: one replay"], ["sample + copy", 130, "hot", "eager (ex. 3 fixes)"]];
          let x = 24; const out = [];
          seg.forEach(([t, w, c, sub]) => { out.push(cell(G, x, 66, w - 4, 52, t, c, { size: 12, rx: 6 })); G.label(x, 138, sub, { size: 10, color: c }); x += w; });
          G.label(24, 176, "stays eager", { size: 11, color: "v" });
          G.label(24, 200, "· prefill: a different shape for every chunk, big kernels", { size: 12, color: "ink" });
          G.label(24, 220, "· sampling: needs a host sync and a random stream", { size: 12, color: "ink" });
          G.text(24, 264, "to put sampling in the graph", { size: 13, weight: 700 });
          G.label(24, 288, "1. no `if tensor.any()`: every filter always runs (top_k = V is a no-op)", { size: 12, color: "ink" });
          G.label(24, 310, "2. random numbers must advance across replays: graph-safe generator", { size: 12, color: "ink" });
          G.label(24, 330, "   or per-row Philox offsets as a static input tensor", { size: 12, color: "ink" });
          G.text(24, 376, "graphs on or off, greedy output must equal the NumPy reference", { size: 13, color: "ok" });
          G.from(out, { opacity: 0, y: 8, stagger: 0.1, duration: 0.25 });
          G.caption("capture only what has fixed shapes and no host sync");
        } },

      { rail: "when it pays", title: "When does a graph pay? Compare CPU time with GPU time",
        body: `<p>Graphs remove the CPU launch cost. They pay only when that cost was the bottleneck. Using the model from step 1, with the graph adding a small replay overhead <i>R</i> and the GPU time <i>G</i> unchanged:</p>
<div class="eq">eager step ~ max(N * L, G)
graph step ~ R + G
saving = max(N*L, G) - G - R
  (positive only if N*L > G + R)</div>
<p>Two regimes, with <b>example values</b> (N = 2,000, L = 5 µs, R = 50 µs; check yours with <code>nsys</code>):</p>
<table><tr><th></th><th>GPU time G</th><th>eager</th><th>graph</th></tr>
<tr><td>small model, batch 1</td><td>1 ms</td><td>10 ms</td><td>1.05 ms</td></tr>
<tr><td>8B bf16 on a 300 GB/s GPU</td><td>16 GB / 300 GB/s = 53.3 ms</td><td>53.3 ms</td><td>53.4 ms</td></tr></table>
<p>For the 8B row, G is only the time to read the 16 GB of weights once (≈ 8.0B parameters × 2 bytes, P1.2) at 300 GB/s, which is an example value; check the spec sheet of your GPU. The CPU's 10 ms hides completely behind 53 ms of GPU work, so there is nothing to save. The small model is the other case: the GPU finishes in 1 ms, the CPU needs 10, and the graph removes nearly the whole 9 ms.</p>
<p>So graphs help most when the model is small, the GPU fast, the batch small, or the per-step kernel count high; they help least for big models and big batches, where each step is long and the saving, though still about N·L in absolute terms, is a small fraction of it. That is why the engine falls back to eager above the largest bucket, and why the module's measurements are at B = 1, 8 and 32. Exercise 1 asks you to find, on real hardware, the matrix size at which replay stops saving meaningful time.</p>
<p>Measure, don't guess: <code>02_decode_itl.py</code> prints ms per decode step for B = 1, 8, 32, eager and with graphs. In nsys the eager trace has gaps between short kernels and the graph trace does not. The bench table in the module is <code>TODO(run-on: g6.xlarge)</code>.</p>`,
        check: { q: "Why does the saving from CUDA graphs, as a fraction of step time, shrink at large batch or for a big model?",
          options: ["Graphs get slower at large batch", "The GPU time per step grows (more weight and KV bytes to read, more work), so the fixed CPU launch cost is hidden behind it and is a smaller share of the step", "Large batches cannot be padded", "The CPU gets faster"], answer: 1,
          why: "Eager time is about max(N·L, G). When G, the GPU time, exceeds the CPU enqueue time, launches overlap with execution and the CPU cost is invisible; the graph then saves only a little. As batch and model size raise G, the launch overhead shrinks relative to it." },
        scene(G) {
          G.text(24, 30, "two regimes (example values, not measured)", { size: 13 });
          const rows = [["small model, B = 1", 10, 1.05, 1], ["8B bf16, 300 GB/s", 53.3, 53.4, 53.3]];
          const sc = 440 / 56, out = [];
          rows.forEach(([t, e, g, gpu], i) => {
            const y = 66 + i * 120;
            G.text(24, y, t, { size: 12, weight: 700 });
            out.push(G.rect(24, y + 14, e * sc, 24, { fill: "v", rx: 3 })); G.label(24 + e * sc + 6, y + 31, "eager " + e + " ms", { size: 11, color: "ink" });
            out.push(G.rect(24, y + 44, g * sc, 24, { fill: "ok", rx: 3 })); G.label(24 + g * sc + 6, y + 61, "graph " + g + " ms", { size: 11, color: "ok" });
            G.rect(24, y + 74, gpu * sc, 6, { fill: "k", rx: 2 }); G.label(24 + gpu * sc + 6, y + 82, "GPU time " + gpu + " ms", { size: 10, color: "k" });
          });
          G.label(24, 62 + 100, "", { size: 1 });
          G.text(24, 316, "eager ≈ max(N·L, G)     graph ≈ R + G", { size: 14, color: "ink" });
          G.label(24, 342, "N·L = 2,000 × 5 us = 10 ms · R = 50 us   (all example values)", { size: 12 });
          G.label(24, 366, "launch-bound (N·L > G): graphs save almost N·L", { size: 12, color: "ok" });
          G.label(24, 388, "GPU-bound (G > N·L): the launches were hidden, nothing to save", { size: 12, color: "hot" });
          G.from(out, { attr: { width: 0 }, stagger: 0.1, duration: 0.5 });
          G.caption("graphs only help when the CPU was the slow one");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A decode step is thousands of short kernels. Eager launches cost CPU time each; when N·L exceeds the GPU time the step is launch-bound and the GPU idles between kernels.</li>
<li>A CUDA graph records the launches once; <code>replay()</code> submits them in one call. Same kernels, same results, far less CPU work.</li>
<li>Recipe: warm up on a side stream, capture inside <code>torch.cuda.graph</code>, <code>copy_</code> inputs into static buffers, replay, read the static output.</li>
<li>Capture freezes shapes, addresses and the code path: no host syncs, no data-dependent <code>if</code>, never rebind a captured tensor.</li>
<li><code>TorchRunner.decode</code> uses padded <code>[B, max_blocks]</code> block tables, attends over all <code>max_blocks × block_size</code> slots and masks, which trades wasted work for static shapes.</li>
<li>Buckets pad the batch up to a captured size (5 sequences → bucket 8, 3 padded rows writing to the reserved block 0); above the biggest bucket the engine runs eager. Prefill and, for now, sampling stay outside.</li>
<li>Graphs pay when N·L &gt; G + R: small models, small batches, fast GPUs. Measure at B = 1, 8, 32 on real hardware.</li></ul>`,
    sim: {
      title: "Eager vs graph decode: a launch-bound or GPU-bound step?",
      intro: "A model, not a measurement. Eager step time is max(CPU enqueue, GPU time); a graph step is the replay overhead plus GPU time, with the batch padded up to the next bucket. GPU time is the bytes read per step (weights once, plus every sequence's KV) divided by memory bandwidth, as in P6.1. All inputs are example values: check your GPU's spec sheet and count your kernels with nsys.",
      height: 280,
      controls: [
        { id: "model", label: "model weights", type: "select", value: 2, options: [[0.05, "tiny (50 MB)"], [2, "1B bf16 (2 GB)"], [16, "8B bf16 (16 GB)"]] },
        { id: "bw", label: "memory bandwidth GB/s (example, check the spec sheet)", min: 100, max: 4000, step: 50, value: 300 },
        { id: "N", label: "kernels per step N", min: 200, max: 6000, step: 100, value: 2000 },
        { id: "L", label: "CPU time per eager launch (µs, example)", min: 1, max: 20, step: 0.5, value: 5, format: (x) => x.toFixed(1) },
        { id: "R", label: "graph replay overhead (µs, example)", min: 5, max: 500, step: 5, value: 50 },
        { id: "ctx", label: "context per sequence (tokens)", min: 16, max: 4096, step: 16, value: 512 },
        { id: "B", label: "running batch size B", min: 1, max: 80, step: 1, value: 1 },
        { id: "buckets", label: "capture buckets", type: "select", value: "p2", options: [["p2", "1, 2, 4, 8, 16, 32, 64"], ["odd", "1, 3, 8, 24, 64"], ["big", "8, 16, 32, 64"], ["one", "64 only"]] },
      ],
      draw(G, v) {
        const lists = { p2: [1, 2, 4, 8, 16, 32, 64], odd: [1, 3, 8, 24, 64], big: [8, 16, 32, 64], one: [64] };
        const list = lists[v.buckets], cpu = (v.N * v.L) / 1000, R = v.R / 1000;
        const gpuMs = (B) => ((v.model * 1e9 + B * v.ctx * KV_TOKEN) / (v.bw * 1e9)) * 1000;
        const eager = (B) => Math.max(cpu, gpuMs(B));
        const graph = (B) => { const b = bucketFor(list, B); return b === null ? eager(B) : R + gpuMs(B); };
        const Bs = Array.from({ length: 80 }, (_, i) => i + 1);
        const mx = Math.max(...Bs.map((B) => Math.max(eager(B), graph(B)))) * 1.1;
        const x0 = 50, W = 570, H = 190, base = 230, X = (B) => x0 + (W * (B - 1)) / 79, Y = (t) => base - (H * t) / mx;
        G.axes(x0, base - H, W, H);
        G.label(10, 16, "ms per decode step, against running batch size B (1 to 80)", { size: 11 });
        G.path(Bs.map((B, i) => `${i ? "L" : "M"} ${X(B)} ${Y(eager(B))}`).join(" "), { color: "v", w: 2.5 });
        G.path(Bs.map((B, i) => `${i ? "L" : "M"} ${X(B)} ${Y(graph(B))}`).join(" "), { color: "ok", w: 2.5 });
        G.line(x0, Y(cpu), x0 + W, Y(cpu), { color: "hot", dash: "3 4" });
        G.label(x0 + W, Y(cpu) - 4, "CPU enqueue N·L = " + cpu.toFixed(1) + " ms", { size: 10, anchor: "end", color: "hot" });
        list.forEach((b) => { if (b <= 80) G.line(X(b), base, X(b), base + 5, { color: "ok", w: 2 }); });
        G.line(X(v.B), base, X(v.B), base - H, { color: "ink", dash: "2 3" });
        G.label(x0, base + 20, "B = 1", { size: 10 }); G.label(x0 + W, base + 20, "B = 80", { size: 10, anchor: "end" });
        G.label(10, 270, "amber: eager · green: graph (ticks = bucket sizes; above the largest it falls back to eager) · dotted: your B", { size: 10 });
        const e = eager(v.B), g = graph(v.B), b = bucketFor(list, v.B), gp = gpuMs(v.B);
        const launchBound = cpu > gp + R;
        const cross = Bs.find((B) => gpuMs(B) >= cpu);
        return [
          { title: `At B = ${v.B}`, rows: [["GPU time", F.ms(gp / 1000)], ["eager step", F.ms(e / 1000)], ["graph step", F.ms(g / 1000)], ["saving", (e - g >= 0 ? "" : "−") + F.ms(Math.abs(e - g) / 1000) + ` (${(100 * (e - g) / e).toFixed(0)} %)`], ["bucket used", b === null ? "none: eager" : `${b} (${b - v.B} padded rows)`]],
            chip: [launchBound, launchBound ? "launch-bound: the graph helps" : "GPU-bound: launches were already hidden"] },
          { title: "Bucket list", rows: [["mean padded rows, B uniform in 1 … 64", meanPad(list).toFixed(1)], ["eager stops being launch-bound at", cross === undefined ? "B > 80" : "B ≈ " + cross]],
            html: `<p class="note">With the 8B model at 300 GB/s the weights alone take 53 ms, so the CPU's 10 ms is hidden at every B; the 1B and tiny models show the launch-bound regime. Padding is shown as wasted rows only; this model charges no extra GPU time for them.</p>` },
        ];
      },
      note: "A model: eager ≈ max(N·L, G), graph ≈ R + G, G = (weights + B·context·128 KiB) / bandwidth. All of N, L, R and the bandwidth are example values, not measurements; this environment has no GPU. Padded rows are not charged GPU time, and the model ignores kernel tails and CPU-side Python outside the launch loop.",
    },
    practice: {
      intro: "The module is T2: capture needs a GPU. The reading above and the simulator are T0. Run the examples and exercises on one g6.xlarge using the P6.7 AWS guide (cost line, auto-stop and teardown are there; budget about 1 to 2 instance-hours for this module). Every timing cell in the module is TODO(run-on: g6.xlarge) until you measure it.",
      items: [
        { title: "Capture and replay, and break each rule on purpose", tier: "T2 · medium", goal: "Run the toy step, then rebind instead of copy_, add a .item() inside the step, skip the warm-up, and allocate by input value. For each: error, silent wrong result, or fine, and why. Then vary the matrix size and find where replay stops saving time.",
          cmd: "uv run --extra torch python course/P6-engine-internals/P6.6-cuda-graph-capture/examples/01_capture_replay.py" },
        { title: "Batch-size buckets and padding in the engine", tier: "T2 · medium", goal: "Check graphs on and off against the NumPy reference, change the bucket list, measure mean padding on a uniform 1 to 64 trace and the capture time and pool memory for 4, 8 and 16 buckets.",
          cmd: "uv run --extra torch pytest platform/engine/v1/tests/test_torch_runner.py -m gpu -q" },
        { title: "Graph-safe sampling (hard)", tier: "T2 · hard", goal: "Rewrite sample_torch without host syncs and give it a random stream that advances across replays; check greedy equality, seeded reproducibility and the P6.5 distribution test through replays.",
          cmd: "cat course/P6-engine-internals/P6.6-cuda-graph-capture/exercises/03-graph-sampling.md" },
        { title: "Decode ms per step, eager vs graphs", tier: "T2", goal: "Print ms per decode step at B = 1, 8, 32 with and without graphs; profile both with nsys and compare the gaps between kernels.",
          cmd: "nsys profile -o decode_graphs uv run --extra torch python course/P6-engine-internals/P6.6-cuda-graph-capture/examples/02_decode_itl.py" },
      ],
      labs: [
        { label: "The graph capture code (DecodeGraphs, capture_decode)", path: "platform/engine/v1/s2s_engine/cuda_graph.py" },
        { label: "TorchRunner.decode and execute: static shapes, the eager/graph split", path: "platform/engine/v1/s2s_engine/model_runner.py" },
        { label: "Where block 0 is reserved for padded rows", path: "platform/engine/v1/s2s_engine/server.py" },
        { label: "Eager (same command with --eager) for the nsys comparison", path: "course/P6-engine-internals/P6.6-cuda-graph-capture/examples/02_decode_itl.py" },
        { label: "AWS guide: shared with P6.7 (g6.xlarge, SSM only, make down)", path: "course/P6-engine-internals/P6.7-project-tiny-engine-gpu/aws.md" },
        { label: "Quiz", path: "course/P6-engine-internals/P6.6-cuda-graph-capture/quiz.md" },
        { label: "Read, don't copy: vLLM docs/design/cuda_graphs.md and vllm/compilation/cuda_graph.py at the pin; SGLang cuda_graph_runner.py; nano-vllm capture_cudagraph (re-verify paths)", path: "docs/design/cuda_graphs.md" },
        { label: "CUDA C++ Programming Guide, CUDA Graphs; PyTorch docs, CUDA Graphs (torch.cuda.graph)", path: "https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs" },
      ],
    },
  });
})();
