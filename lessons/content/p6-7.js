/* P6.7 — Project: a tiny GPU engine (#0 v1). How the P6.1-P6.6 pieces fit, the runner contract and the test ladder,
   bf16 parity, swapping in kernels, the HTTP server, a fair benchmark against vLLM, and profile-driven diagnosis.
   Mirrors platform/engine/v1 (engine.py, model_runner.py, server.py) and bench/compare.sh. No GPU here: the simulator is
   a model with example values, and the bench table stays TODO(run-on: g6.xlarge). */
(function () {
  const F = S2S.fmt;
  const cell = (G, x, y, w, h, t, color, o = {}) => G.box(x, y, w, h, t, Object.assign({ fill: color, size: 11, rx: 3 }, o));
  const MODELS = { tiny: { w: 0.05e9, kv: 8 * 1024 }, b1: { w: 2e9, kv: 32 * 1024 }, b8: { w: 16e9, kv: 128 * 1024 } }; // illustrative sizes; 8B: 16 GB, 128 KiB/token (P1.2)

  /* the step-time model used by the simulator */
  function stepModel(v, B) {
    const m = MODELS[v.model], S = v.S, ctx = v.ctx;
    const attnBytes = v.attn === "gather" ? 3 * S * m.kv : ctx * m.kv; // gather: read S slots, write the copy, read it again
    const gpu = (m.w + B * attnBytes) / (v.bw * 1e9) * 1e3; // ms
    const launch = v.graphs === "on" ? v.R / 1e3 : (v.N * v.L) / 1e3;
    const gpuPhase = Math.max(launch, gpu);
    const sched = (v.s0 + B * v.s) / 1e3;
    const step = v.async === "on" ? Math.max(sched, gpuPhase) : sched + gpuPhase;
    return { gpu, launchExtra: gpuPhase - gpu, sched: v.async === "on" ? Math.max(0, step - gpuPhase) : sched, step, tps: (B / step) * 1e3, idle: 1 - gpu / step };
  }

  S2S.lesson({
    id: "p6-7", n: "P6.7", title: "Project: a tiny GPU engine",
    subtitle: "Engine internals · project · T0 for the logic and the server, T2 for the GPU runner and the benchmark",
    kicker: "Project · ≈ 40 min reading, ≈ 14 h hands-on",
    headline: "Own the engine: test it, serve it, measure it, explain the gap",
    intro: `<p>Phase 6 took an inference engine apart: the loop (P6.1), the scheduler (P6.2), the block manager (P6.3), the radix cache (P6.4), sampling and verification (P6.5), CUDA graphs (P6.6). This project puts it back together as <b>#0 v1</b> in <code>platform/engine/v1</code>: a GPU serving engine behind an OpenAI-compatible API, benchmarked against vLLM on the same GPU and model. The reference implementation is complete; your job is to <i>own</i> it by rebuilding the pieces you wrote in P6.2 to P6.6 into it, swapping in your kernels, and then explaining, with a profile, where each gap to vLLM comes from. This lesson is the map and the method. Nothing here is a benchmark result: this environment has no GPU, and the bench table is <code>TODO(run-on: g6.xlarge)</code> for you to fill.</p>`,
    facts: ["9 steps", "5 checkpoints", "1 simulator", "7 build steps"],
    legend: [["k", "engine / scheduler"], ["ok", "memory / valid"], ["v", "runner / kernels"], ["q", "sampling"], ["hot", "gap / risk"], ["w", "GPU / reference"]],
    prev: "p6-6", next: "c1",
    steps: [
      { rail: "the map", title: "What you are assembling, and what the tests are for",
        body: `<p>The engine is ten small modules, each one a lesson of this phase. Together they are the three decisions from P6.1 (what runs now, where its KV goes, which token comes next) plus the machinery around them:</p>
<table><tr><th>module</th><th>job</th><th>from</th></tr>
<tr><td><code>sequence.py</code></td><td>request state: tokens, block table, <code>num_computed</code></td><td>P6.1</td></tr>
<tr><td><code>scheduler.py</code></td><td>continuous batching, chunked prefill, preemption, priorities + aging</td><td>P6.2</td></tr>
<tr><td><code>block_manager.py</code></td><td>paged KV, chained block hashes, copy-on-write</td><td>P6.3</td></tr>
<tr><td><code>radix_cache.py</code></td><td>token-level prefix cache (optional, not in the loop)</td><td>P6.4</td></tr>
<tr><td><code>sampling.py</code>, <code>spec_verify.py</code></td><td>filters, seeded RNG; batched rejection sampling</td><td>P6.5</td></tr>
<tr><td><code>model_runner.py</code></td><td>Fake, NumPy-paged and Torch runners</td><td>P6.1, P6.3</td></tr>
<tr><td><code>cuda_graph.py</code></td><td>decode graphs per batch bucket</td><td>P6.6</td></tr>
<tr><td><code>engine.py</code>, <code>server.py</code></td><td>the step loop; the HTTP API</td><td>P6.1, here</td></tr></table>
<p>The reference passes its T0 tests. A project like this lives or dies by one question: <b>how do you know you got it right?</b> A scheduler bug does not crash, it makes some request slow. A block-table bug does not crash, it makes a token wrong somewhere in a long output. So the project is organised around <b>tests as the contract</b>: replace a module with yours and the same tests must still pass. The strongest of them, <code>test_numpy_engine.py</code>, runs the full paged engine, with chunking and preemption, against the contiguous v0 model and demands the <b>same greedy tokens</b>. Any wrong slot, mask, position or recompute changes a token.</p>`,
        scene(G) {
          const b = (x, y, w, t, sub, col) => { G.box(x, y, w, 50, "", { stroke: col, rx: 9 }); G.text(x + 8, y + 20, t, { size: 12, weight: 700, color: col }); G.label(x + 8, y + 38, sub, { size: 10 }); };
          b(24, 44, 180, "server.py", "HTTP, SSE, /metrics", "hot");
          b(230, 44, 180, "engine.py", "the step loop", "k");
          b(436, 44, 180, "sequence.py", "request state · P6.1", "k");
          G.arrow(206, 69, 228, 69, { color: "ink", w: 2 }); G.arrow(412, 69, 434, 69, { color: "ink", w: 2 });
          b(24, 132, 180, "scheduler.py · P6.2", "what runs this step", "k");
          b(230, 132, 180, "block_manager.py · P6.3", "where the KV goes", "ok");
          b(436, 132, 180, "radix_cache.py · P6.4", "optional, standalone", "ok");
          G.arrow(320, 96, 114, 130, { color: "line" }); G.arrow(320, 96, 320, 130, { color: "line" });
          b(24, 220, 180, "model_runner.py", "Fake · NumPy · Torch", "v");
          b(230, 220, 180, "cuda_graph.py · P6.6", "decode graphs", "v");
          b(436, 220, 180, "sampling.py · P6.5", "filters, seeded RNG", "q");
          b(436, 296, 180, "spec_verify.py · P6.5", "batched rejection", "q");
          G.arrow(114, 184, 114, 218, { color: "line" }); G.arrow(206, 245, 228, 245, { color: "line" }); G.arrow(410, 245, 434, 245, { color: "line" });
          G.text(24, 376, "tests/ = the contract: swap a module, the tests stay", { size: 13, color: "ink" });
          G.label(24, 398, "strongest: paged engine == contiguous v0, greedy, token for token", { size: 12, color: "ok" });
          G.label(24, 416, "(under chunking and preemption)", { size: 11 });
          G.caption("ten modules, one contract");
        } },

      { rail: "one step", title: "One step, end to end",
        body: `<p>Everything in the engine happens inside <code>LLMEngine.step()</code>. It is four lines, and you have now built each one:</p>
<div class="eq">out    = scheduler.step()       # P6.2: which chunks
logits = runner.execute(out)    # forward pass, KV I/O
tokens = sample(logits)         # P6.5: per-seq RNG
done   = scheduler.update(out, tokens)</div>
<p>Trace one step from the real run printed by <code>P6.1 examples/01_trace_one_step.py</code> (token budget 8, 8 blocks of 4 slots). Three requests, step 2:</p>
<div class="eq">A: Chunk(start 8, n 2)   end of its prompt
B: Chunk(start 8, n 2)   8-token prefix hit (P6.3)
C: Chunk(start 0, n 3)   a whole 3-token prompt
total tokens = 2 + 2 + 3 = 7 &lt;= budget 8
sampled: A 29, B 29, C 30 (printed by the script)</div>
<p>Inside the step: the scheduler picks the chunks and the <b>block manager</b> allocates blocks for them (and queues copy-on-write copies). The <b>runner</b> executes those copies first, then the forward pass over all 7 tokens, and returns logits only for chunks that <i>produce a token</i> (<code>start + n</code> reaches the end of the known tokens, P6.2). Here all three chunks reach the end of their prompt, so all three yield a token; in step 1 of the same trace A's chunk (positions 0 to 7 of a longer prompt) yields none. The engine samples a token for each of those with that request's own RNG, and <code>update()</code> advances <code>num_computed</code>, appends tokens, registers new block hashes, and frees finished sequences.</p>
<p>Two design points to keep in mind through the project. The step is <b>synchronous</b> in the reference: the CPU scheduling for step t+1 waits for the GPU work of step t (async scheduling is a stretch goal, step 9). And <code>num_computed</code> advances in <code>update()</code>, not when chunks are scheduled, so a step that fails leaves state consistent.</p>`,
        scene(G) {
          const st = (x, y, w, t, sub, col) => { G.box(x, y, w, 62, "", { stroke: col, rx: 10 }); G.text(x + 10, y + 24, t, { size: 12, weight: 700, color: col }); G.label(x + 10, y + 44, sub, { size: 10 }); };
          st(24, 44, 138, "scheduler.step()", "chunks + blocks", "k");
          st(176, 44, 138, "runner.execute()", "copies, forward", "v");
          st(328, 44, 138, "sample()", "per-seq RNG", "q");
          st(480, 44, 136, "update()", "num_computed, free", "k");
          [162, 314, 466].forEach((x) => G.arrow(x, 75, x + 12, 75, { color: "ink", w: 2 }));
          G.text(24, 150, "step 2 of the P6.1 trace (budget 8)", { size: 13, weight: 700 });
          const cells = [["A", "k"], ["A", "k"], ["B", "v"], ["B", "v"], ["C", "q"], ["C", "q"], ["C", "q"], ["", null]];
          cells.forEach(([t, c], i) => c ? cell(G, 24 + i * 72, 168, 66, 36, t, c, { size: 13 }) : G.rect(24 + i * 72, 168, 66, 36, { stroke: "line", rx: 3, dash: "3 3" }));
          G.label(24, 224, "A: Chunk(8, 2)   B: Chunk(8, 2), prefix hit   C: Chunk(0, 3)   → 7 of 8 tokens", { size: 11, color: "ink" });
          G.text(24, 270, "logits come back only for chunks that reach the end of the known tokens", { size: 12, color: "ink" });
          G.label(24, 296, "all three here: sampled A 29, B 29, C 30 (step 1's A[0:8]: no token)", { size: 11 });
          G.label(24, 330, "then update(): num_computed += n · append sampled tokens · commit hashes · free finished", { size: 11 });
          G.label(24, 362, "synchronous in the reference: CPU waits for the GPU each step", { size: 12, color: "hot" });
          G.caption("schedule, run, sample, update: the whole engine in four calls");
        } },

      { rail: "runner contract", title: "The runner contract, and why there are three runners",
        body: `<p>The engine and the scheduler never touch tensors. They talk to the model through one narrow interface, <code>runner.execute(out)</code>:</p>
<ul><li><b>In:</b> the scheduled chunks (sequence, start, token count), their block tables, and the pending copy-on-write <code>copy_ops</code>.</li>
<li><b>Out:</b> a dict from sequence id to logits, for every chunk that produces a token.</li></ul>
<p>Behind that interface there are three implementations, and each answers a different question, so each catches a different kind of bug:</p>
<table><tr><th>runner</th><th>tier</th><th>what it proves</th></tr>
<tr><td><code>FakeRunner</code></td><td>T0</td><td>"logits" are a hash of the token history: scheduling, blocks, preemption and sampling are right, independent of any numerics (P6.2)</td></tr>
<tr><td><code>NumpyPagedRunner</code></td><td>T0</td><td>the paged slot lookups reproduce the contiguous v0 model's tokens exactly (P6.3)</td></tr>
<tr><td><code>TorchRunner</code></td><td>T2</td><td>the same on a GPU, with batched decode and optional CUDA graphs (P6.6)</td></tr></table>
<p>This is a <b>test ladder</b>: each rung changes one thing and has a reference to compare with. If the GPU output differs from the NumPy one, the scheduler, the block manager and sampling are already exonerated (they ran unchanged on the fake and NumPy runners), so the bug is in the Torch code. Building it any other way, say writing the GPU runner and the scheduler together, leaves you debugging two unknowns at once.</p>
<p>Every runner must execute <code>out.copy_ops</code> before computing anything (P6.3): a forked child reads its parent's KV, and the copy must precede the write.</p>`,
        check: { q: "The GPU engine's output differs from the NumPy engine's for one prompt. Which layer do you suspect first, and why?",
          options: ["The scheduler: it is the most complex module", "The TorchRunner: the scheduler, block manager and sampler ran unchanged under the T0 runners and passed their tests, so the new code is the prime suspect", "The tokenizer", "The HTTP server"], answer: 1,
          why: "The ladder isolates one change per rung. Scheduling, blocks and sampling are identical code on every runner and were validated at T0, so a GPU-only difference points at what only the GPU path runs: the Torch forward pass, gather, dtype, or graph capture." },
        scene(G) {
          G.text(24, 30, "engine ⇄ runner: one narrow interface", { size: 13 });
          G.box(24, 50, 250, 90, "", { stroke: "k", rx: 12 });
          G.text(36, 74, "in: SchedulerOutput", { size: 13, weight: 700, color: "k" });
          ["chunks (seq, start, n)", "block tables", "copy_ops [(src, dst)]"].forEach((t, i) => G.label(40, 96 + i * 16, "· " + t, { size: 11, color: "ink" }));
          G.arrow(278, 95, 352, 95, { color: "ink", w: 2 });
          G.box(356, 50, 260, 90, "", { stroke: "q", rx: 12 });
          G.text(368, 74, "out: logits per sequence", { size: 13, weight: 700, color: "q" });
          ["chunks that produce a token", "{seq_id: logits [V]}"].forEach((t, i) => G.label(372, 96 + i * 16, "· " + t, { size: 11, color: "ink" }));
          const rr = [["FakeRunner", "T0", "scheduling, blocks, sampling", "w"], ["NumpyPagedRunner", "T0", "paging == contiguous v0", "ok"], ["TorchRunner", "T2", "GPU, batched decode, graphs", "v"]];
          rr.forEach(([n, t, d, c], i) => {
            const y = 176 + i * 62;
            G.box(24, y, 590, 50, "", { stroke: c, rx: 10 });
            G.text(36, y + 22, n, { size: 13, weight: 700, color: c }); G.label(36, y + 40, d, { size: 11 });
            G.label(560, y + 22, t, { size: 12, anchor: "end", color: c });
          });
          G.label(24, 376, "each rung changes one thing and has a reference to compare with", { size: 12, color: "ink" });
          G.label(24, 396, "all runners run copy_ops before the forward pass", { size: 11 });
          G.caption("a test ladder: a GPU-only failure points at GPU-only code");
        } },

      { rail: "parity", title: "GPU parity: exact in fp32, a tolerance in bf16",
        body: `<p>Build step 2 asks the GPU runner to match the reference. The check depends on the number format:</p>
<ul><li><b>fp32:</b> greedy outputs must match the NumPy reference <b>token for token</b>. fp32 has enough precision that, for a tiny model, no argmax should flip.</li>
<li><b>bf16:</b> compare <b>logits within a stated tolerance</b> and report the first step at which the greedy tokens diverge. Do not demand identical tokens.</li></ul>
<p>Why the difference? bf16 keeps 8 significant bits: 7 stored mantissa bits plus the implicit one. Neighbouring values are 2<sup>−7</sup> of their magnitude apart. For logits between 16 and 32 the spacing is 16 × 2<sup>−7</sup> = <b>0.125</b>. Every operation in the forward pass rounds, and sums accumulate in a different order on a GPU than on a CPU, so tiny differences are unavoidable, and they are of the size of one bf16 step.</p>
<div class="eq">illustrative: top-2 logits at magnitude ~20
fp32:  A = 20.05   B = 20.09   greedy picks B
bf16 pipeline error of a few 0.01 to 0.1
       can flip them: A &gt; B, greedy picks A
after that token the two texts differ for good</div>
<p>That flip is not a bug; <i>identical tokens</i> would be a stronger result than the format can give. What you can and should demand is that the <b>logits</b> are close (state the tolerance and why), that the divergence step is reported, and that fp32 is exact. Two engines that diverge at step 3 of one prompt but have logits within tolerance everywhere are numerically equal; one that diverges with a large logit difference has a bug. This is also why the benchmark must compare engines in the same dtype (step 7).</p>`,
        check: { q: "Your bf16 GPU engine and the fp32 NumPy reference produce different greedy tokens after step 40 of one prompt. What is the right conclusion?",
          options: ["The GPU engine is wrong", "Not necessarily: bf16 rounding can flip a near-tied argmax. Check the logit difference at the divergence step against your stated tolerance; fp32 must match exactly", "The scheduler is wrong", "Greedy decoding is random on GPUs"], answer: 1,
          why: "bf16 has spacing 0.125 for logits near 20, and the accumulation order differs on a GPU. If the top two logits are closer than the accumulated error, the argmax can flip without any bug. The test is the size of the logit difference compared with the tolerance, and exact equality in fp32 where the error is far smaller." },
        scene(G) {
          G.text(24, 30, "bf16 spacing near 20 is 0.125", { size: 13 });
          const x0 = 40, W = 560, lo = 19.75, hi = 20.5, X = (v) => x0 + (W * (v - lo)) / (hi - lo);
          G.line(x0, 140, x0 + W, 140, { color: "ink", w: 1.5 });
          const tk = [];
          for (let v = 19.75; v <= 20.5001; v += 0.125) { const x = X(v); tk.push(G.line(x, 130, x, 150, { color: "w", w: 2 })); G.label(x, 168, v.toFixed(3), { size: 9, anchor: "middle" }); }
          G.label(x0, 196, "grey ticks: the only values bf16 can represent in this range", { size: 11 });
          const a = 20.05, b = 20.09;
          G.circle(X(a), 140, 6, { fill: "k" }); G.label(X(a) - 6, 118, "A 20.05", { size: 11, anchor: "end", color: "k" });
          G.circle(X(b), 140, 6, { fill: "v" }); G.label(X(b) + 6, 118, "B 20.09", { size: 11, anchor: "start", color: "v" });
          G.label(X(a) - 60, 96, "fp32: B wins by 0.04", { size: 11, color: "ink" });
          G.rect(X(a) - 4, 124, X(b) - X(a) + 8, 32, { stroke: "hot", rx: 6, dash: "3 3" });
          G.text(24, 244, "any rounding or summation-order error of that size", { size: 13, color: "ink" });
          G.text(24, 268, "can swap A and B → a different greedy token", { size: 13, color: "ink" });
          G.label(24, 304, "fp32: demand identical tokens", { size: 13, color: "ok" });
          G.label(24, 326, "bf16: state a logit tolerance, report the first diverging step", { size: 13, color: "v" });
          G.label(24, 364, "illustrative logits; bf16 = 7 stored mantissa bits → spacing 16 × 2⁻⁷ = 0.125 in [16, 32)", { size: 10 });
          G.from(tk, { opacity: 0, stagger: 0.03, duration: 0.15 });
          G.caption("a near-tie below the format's resolution can flip without any bug");
        } },

      { rail: "kernels", title: "Swap in your kernels, one at a time",
        body: `<p>The GPU runner starts with plain PyTorch ops, which is correct and slow in known places. Build step 3 replaces them with your kernels from earlier phases, <b>one at a time, with a parity check after each</b>:</p>
<ol><li><b>RMSNorm</b> → the fused add + RMSNorm op from P5.10 (<code>torch.ops.s2s.fused_add_rms_norm</code>): fewer kernels per layer and one pass over the activations instead of several.</li>
<li><b>Decode attention</b> → your paged decode kernel (P5.8 exercise 4, P6.3 exercise 4): it reads K/V through the block table and the real <code>context_lens</code>, so it drops the gather copy and the wasted slots (P6.6 step 5: 923 of 1,024 gathered slots masked out at position 100).</li>
<li><b>Projections</b> → your HGEMM, or cuBLAS through <code>torch.matmul</code>. <b>Keep cuBLAS where yours loses, and say so in the table.</b> A hand-written GEMM beating cuBLAS is unusual; knowing that is the lesson.</li></ol>
<p>One change per run keeps two failure modes separate. A parity failure after the paged kernel is the kernel's fault, not the norm's. A speed-up that appears only when two changes land together tells you they interact (for example, a graph-captured step makes a launch-saving fusion matter less). Record each row of the bench table after each swap:</p>
<table><tr><th>configuration</th><th>what changed</th></tr>
<tr><td>#0 v1, eager, gather attention</td><td>baseline</td></tr>
<tr><td>+ CUDA graphs, gather attention</td><td>launch overhead (P6.6)</td></tr>
<tr><td>+ graphs, D4 paged attention + RMSNorm</td><td>attention bytes and kernel count</td></tr></table>
<p>The expectation, from the module README: well below vLLM at first (Python step loop, gather attention, no fused kernels). The goal is not to catch vLLM; it is to know <b>where</b> each gap comes from. And do not quote a percentage you did not measure.</p>`,
        scene(G) {
          G.text(24, 30, "one change per run, parity after each", { size: 13 });
          const rungs = [["baseline", "eager, gather attention", "w", 0], ["+ CUDA graphs", "launch overhead (P6.6)", "v", 1], ["+ fused RMSNorm", "fewer kernels, one pass (P5.10)", "q", 2], ["+ paged decode kernel", "no gather copy, real context_lens", "ok", 3]];
          const out = [];
          rungs.forEach(([t, sub, c, i]) => {
            const x = 24 + i * 40, y = 60 + i * 66;
            out.push(G.box(x, y, 340, 52, "", { stroke: c, rx: 10 }));
            G.text(x + 12, y + 22, t, { size: 13, weight: 700, color: c }); G.label(x + 12, y + 40, sub, { size: 11 });
            G.text(x + 360, y + 28, "parity ✓", { size: 12, color: "ok" });
            G.label(x + 360, y + 44, "bench row " + (i + 1), { size: 10 });
          });
          G.label(24, 340, "projections: your HGEMM or cuBLAS (torch.matmul)", { size: 12, color: "ink" });
          G.label(24, 360, "keep cuBLAS where yours loses, and say so in the table", { size: 12, color: "hot" });
          G.label(24, 386, "expect to start well below vLLM; know WHERE each gap comes from", { size: 12 });
          G.from(out, { opacity: 0, x: -14, stagger: 0.12, duration: 0.3 });
          G.caption("each swap is a separate, checked, measured step");
        } },

      { rail: "the server", title: "Serving it: an OpenAI-compatible API on top of a loop",
        body: `<p><code>server.py</code> wraps the engine in the API that vLLM exposes, so your loadgen (#4) and gateway (#6) treat it like vLLM. The design is small, and each choice has a reason:</p>
<ul><li><b>One background task drives the engine.</b> <code>_drive()</code> calls <code>engine.step()</code> whenever there is work (and sleeps on an event when idle). The step runs in <code>asyncio.to_thread</code> so model work does not block the event loop that is handling HTTP.</li>
<li><b>One lock</b> guards the engine: a step or an <code>add_request</code>, never both at once. The engine is not thread-safe.</li>
<li><b>One queue per request.</b> After each step, new tokens go into that request's <code>asyncio.Queue</code>; the HTTP handler streams them out as server-sent events (<code>"stream": true</code>) or collects them into one JSON body.</li>
<li><b>Metrics use vLLM's names</b> (<code>vllm:num_requests_running</code>, <code>vllm:kv_cache_usage_perc</code>, <code>vllm:time_to_first_token_seconds</code>, <code>vllm:inter_token_latency_seconds</code>, <code>vllm:num_preemptions_total</code>, …), so the P3.5 dashboards and alerts work unchanged and preemptions show in <code>/metrics</code> (an acceptance test).</li></ul>
<p>Configuration is by environment variable: <code>S2S_RUNNER</code> (<code>fake</code>, <code>numpy</code>, <code>torch</code>), <code>S2S_MODEL</code>, <code>S2S_NUM_BLOCKS</code> (default 1024), <code>S2S_BLOCK_SIZE</code> (16), <code>S2S_MAX_NUM_SEQS</code> (64), <code>S2S_MAX_BATCHED_TOKENS</code> (2048), <code>S2S_CUDA_GRAPHS</code> (on). With graphs on, the server reserves block 0 (<code>eng.blocks._take()</code>) for padded rows (P6.6).</p>
<p><b>Safety rules from the course apply:</b> set <code>S2S_API_KEY</code> (requests then need <code>Authorization: Bearer …</code>), bind to <code>127.0.0.1</code>, and reach a cloud box through SSM port forwarding. Never run a server on <code>0.0.0.0</code> without the key. Prompts are token ids (or one stable hash-id per word, a stand-in so loadgen's text prompts keep identical prefixes identical); real tokenization is out of scope.</p>`,
        check: { q: "Why does the server name its metrics like vLLM's (vllm:num_requests_running, vllm:time_to_first_token_seconds, …)?",
          options: ["Prometheus requires the vllm: prefix", "So the dashboards and alerts you built for vLLM (P3.5) work on your engine unchanged, and the two can be compared on the same panels", "To hide that it is a different engine", "vLLM's client library depends on them"], answer: 1,
          why: "Observability tooling keys on metric names. Reusing vLLM's lets the P3.5 dashboards, alert rules and the load-test comparison treat both engines the same, so a difference on a panel is a difference in the engines." },
        scene(G) {
          G.text(24, 30, "request path through server.py", { size: 13 });
          const bx = (x, y, w, t, sub, col) => { G.box(x, y, w, 56, "", { stroke: col, rx: 10 }); G.text(x + 8, y + 22, t, { size: 12, weight: 700, color: col }); G.label(x + 8, y + 42, sub, { size: 10 }); };
          bx(24, 50, 138, "POST /v1/…", "auth · add_request", "hot");
          bx(190, 50, 138, "queue per request", "asyncio.Queue", "ok");
          bx(356, 50, 120, "_drive()", "loop · to_thread", "k");
          bx(498, 50, 118, "engine.step()", "under the lock", "v");
          [164, 330, 478].forEach((x) => G.arrow(x, 78, x + 24, 78, { color: "ink", w: 2 }));
          G.path("M 557 108 C 557 150 259 150 259 112", { color: "ok", w: 2, arrow: true });
          G.label(300, 170, "new tokens → each request's queue", { size: 11, color: "ok" });
          G.text(24, 214, "out to the client", { size: 13, weight: 700 });
          G.label(24, 238, "stream: true → SSE  data: {chunk} … data: [DONE]", { size: 12, color: "ink" });
          G.label(24, 258, "otherwise one JSON body with usage {prompt, completion, total}", { size: 12, color: "ink" });
          G.text(24, 304, "GET /metrics (vLLM names)", { size: 13, weight: 700 });
          ["vllm:num_requests_running / waiting", "vllm:kv_cache_usage_perc", "vllm:time_to_first_token_seconds, …inter_token_latency…", "vllm:num_preemptions_total"].forEach((t, i) => G.label(34, 326 + i * 16, "· " + t, { size: 11, color: "ink" }));
          G.label(24, 396, "S2S_API_KEY · bind 127.0.0.1 · reach it with SSM port forwarding", { size: 11, color: "hot" });
          G.caption("an async shell around a synchronous engine");
        } },

      { rail: "fair benchmark", title: "A fair benchmark: change one thing, match everything else",
        body: `<p>You will compare #0 v1 with vLLM <code>v0.31.0</code> using the loadgen from P2.3 at increasing request rates and find each engine's <b>knee</b>. The comparison is only meaningful if the engines are configured the same. <code>bench/compare.sh</code> fixes the shared knobs and prints both configs:</p>
<table><tr><th>knob</th><th>default in compare.sh</th><th>must match because</th></tr>
<tr><td>max_num_seqs</td><td>64</td><td>bounds the batch, so sets the throughput / latency trade-off</td></tr>
<tr><td>max_num_batched_tokens</td><td>2048</td><td>bounds prefill chunking (P6.2)</td></tr>
<tr><td>max model length</td><td>4096</td><td>a context limit changes admission and memory</td></tr>
<tr><td>dtype</td><td>bf16</td><td>fp32 vs bf16 is a 2× difference in bytes per weight</td></tr>
<tr><td>GPU, model, workload</td><td>same box, prompt mean 512, output mean 128</td><td>otherwise you compare hardware or traffic</td></tr></table>
<p>Without this, you compare <i>policies</i>, not implementations. vLLM's defaults may allow far larger batches than 64, and an fp32 engine against a bf16 one measures the format.</p>
<p><b>One knob that the script does not set for you: the KV pool.</b> <code>compare.sh</code> passes the sequence and token limits to <code>server.py</code>, but <code>S2S_NUM_BLOCKS</code> keeps its default of 1024 blocks:</p>
<div class="eq">1,024 blocks * 16 slots = 16,384 token slots
per request: 512 prompt + 128 output = 640
16,384 / 640 = 25.6 requests at most
(vs max_num_seqs = 64)</div>
<p>So at the default pool size #0 v1 can hold about 25 such requests at once, fewer than the 64 you allowed, and it will queue or preempt where vLLM, which sizes its pool from free GPU memory, keeps running. Size <code>S2S_NUM_BLOCKS</code> from the GPU's free memory (P2.1) and print it next to vLLM's number. This is arithmetic on the repo's defaults, not a measurement. Print both configurations next to the result, always.</p>`,
        check: { q: "You benchmark your fp32 engine against vLLM in bf16, both at the same request rate. Why is the comparison unfair?",
          options: ["fp32 engines cannot be benchmarked", "fp32 doubles the bytes read per weight and per KV entry compared with bf16, so you are measuring the number format rather than the engine", "vLLM ignores dtype", "Request rates must differ by 2×"], answer: 1,
          why: "Decode is memory-bound (P1.3, P6.1): time scales with bytes read. An fp32 model reads about twice what a bf16 one does, so the gap would be there even if the engines were identical. Same dtype, same limits, same hardware, same workload; only then does the difference belong to the engine." },
        scene(G) {
          G.text(24, 30, "same on both sides", { size: 13 });
          const rows = [["max_num_seqs", "64", "64"], ["max_num_batched_tokens", "2048", "2048"], ["max model length", "4096", "4096"], ["dtype", "bf16", "bf16"], ["GPU · model · workload", "same", "same"], ["KV pool", "1024 blocks = 16,384 slots", "from free GPU memory"]];
          G.label(300, 56, "#0 v1", { size: 12, anchor: "middle", color: "v" }); G.label(470, 56, "vLLM", { size: 12, anchor: "middle", color: "w" });
          rows.forEach(([k, a, b], i) => {
            const y = 66 + i * 38, bad = i === 5;
            G.rect(24, y, 592, 32, { stroke: bad ? "hot" : "line", rx: 5, dash: bad ? "4 3" : undefined });
            G.label(34, y + 21, k, { size: 12, color: "ink" });
            G.label(300, y + 21, a, { size: 11, anchor: "middle", color: bad ? "hot" : "v" }); G.label(470, y + 21, b, { size: 11, anchor: "middle", color: bad ? "hot" : "w" });
          });
          G.text(24, 322, "default pool: 16,384 / 640 tokens per request = 25.6 requests", { size: 13, color: "hot" });
          G.label(24, 348, "→ fewer than max_num_seqs = 64: set S2S_NUM_BLOCKS from free memory", { size: 12, color: "ink" });
          G.label(24, 372, "print both configs next to every result", { size: 12 });
          G.label(24, 394, "dashed red: the knob compare.sh does not set for you (check)", { size: 10 });
          G.caption("a fair comparison changes one thing: the engine");
        } },

      { rail: "diagnose the gap", title: "Explaining the gap with a profile",
        body: `<p>Your engine will be slower than vLLM. The deliverable is not the percentage, it is the <b>explanation</b>, and the profile tells you what to say. Quiz question: your engine is at 40 % of vLLM's tokens/s at batch 32; what should the profile tell you first? Three questions, in order:</p>
<ol><li><b>Is the GPU idle?</b> In <code>nsys</code> (one decode step at B = 8, eager and with graphs, build step 7) look for gaps between kernels. Large gaps mean the step is CPU- or launch-bound (P6.6): graphs, a faster Python loop, or async scheduling. No gaps means the GPU is the limit, and the next questions apply.</li>
<li><b>Which kernels take the time, and how close is each to its roofline?</b> <code>ncu</code> on the top 3 kernels by time. A GEMM near the memory roofline is fine; an attention kernel far below it is a target (P5.8).</li>
<li><b>Does attention or the GEMMs dominate?</b> If the gather attention dominates, its cost grows with <code>max_blocks × block_size</code> and with the batch (P6.6 step 5), and the paged kernel is the fix.</li></ol>
<p>Use the simulator on the next tab to build intuition for what each answer implies. It models a step as GPU time (bytes read ÷ bandwidth), plus launch cost when eager, plus the Python scheduling cost when it is not overlapped. Turn graphs, the paged kernel and async scheduling on and off and watch which segment of the step moves. Then <b>go and measure</b>: the model's numbers are examples, your trace is the truth.</p>
<p>Keep the profiler reports out of git; put a summary table in your README (kernel, time share, achieved vs peak). A claim in the write-up should be traceable to a row in it.</p>`,
        check: { q: "Your engine reaches 40 % of vLLM's throughput at batch 32. What should you check first, and why?",
          options: ["Rewrite the scheduler in C++", "Whether the GPU is idle between kernels (nsys): that separates a CPU/launch-bound step from a GPU-bound one, and the fix for each is different", "Switch to fp32", "Increase max_num_seqs"], answer: 1,
          why: "The first split is where the time goes: gaps on the GPU row mean the CPU cannot keep it fed (graphs, async scheduling, a leaner loop); no gaps mean the kernels are the limit (roofline analysis with ncu, then fusion or a better attention kernel). Optimising before you know which regime you are in wastes days." },
        scene(G) {
          G.text(24, 30, "profile → diagnosis → fix", { size: 13 });
          const q = (y, t, col) => { G.box(24, y, 290, 44, "", { stroke: col, rx: 10 }); G.text(34, y + 27, t, { size: 11, weight: 700, color: col }); };
          q(56, "1. GPU idle between kernels? (nsys)", "k");
          q(150, "2. top kernels near roofline? (ncu)", "v");
          q(244, "3. attention or GEMMs dominate?", "q");
          G.arrow(169, 102, 169, 148, { color: "ink" }); G.arrow(169, 196, 169, 242, { color: "ink" });
          const a = (y, t1, t2, col) => { G.label(332, y + 16, t1, { size: 11, color: col }); G.label(332, y + 34, t2, { size: 11, color: "ink" }); };
          a(56, "yes → CPU / launch-bound", "graphs · async · leaner loop", "hot");
          a(150, "no, far below → kernel", "fuse · better kernel · cuBLAS", "hot");
          a(244, "gather → paged kernel", "real context_lens, no copy", "hot");
          G.text(24, 336, "claims in your write-up trace to a row in the profile table", { size: 13, color: "ink" });
          G.label(24, 362, "reports stay out of git; the summary table goes in your README", { size: 12 });
          G.label(24, 384, "never quote a percentage you did not measure", { size: 12, color: "hot" });
          G.caption("find which regime you are in before you optimise");
        } },

      { rail: "next", title: "Done means: the checklist, and where it goes next",
        body: `<p>The acceptance checklist from the module is the definition of finished:</p>
<ul><li>The engine's T0 tests pass (<code>uv run pytest platform/engine/v1/tests -q</code>).</li>
<li>Greedy outputs token-identical to the NumPy/PyTorch reference for 8 prompts in fp32 on the GPU, with graphs on and off; bf16 within your stated logit tolerance.</li>
<li>The server survives loadgen at 2× its knee rate for 5 minutes with no errors. Preemptions are fine, but they must show in <code>/metrics</code>.</li>
<li>The knee report against vLLM, and the bench table filled in.</li>
<li>ncu/nsys summaries for the top 3 kernels.</li></ul>
<p>Four stretch goals extend it, each one a design question:</p>
<ol><li><b>Async scheduling.</b> Overlap <code>scheduler.step()</code> for step t+1 with the GPU work of step t, which vLLM v1 does. The question to answer: what must the scheduler know before step t's tokens exist? It knows every running decode will produce exactly one token, so lengths and block needs are predictable; what it does not know is whether a sequence hit a stop token. Design for that uncertainty.</li>
<li><b>Spec decode in the loop.</b> Decode chunks of 1 + k tokens with a small draft model; <code>update()</code> appends up to k + 1 tokens and returns the blocks of rejected positions (P6.5).</li>
<li><b>Radix cache as the KV owner</b> instead of block hashing (P6.4 exercise 2), compared on the shared-prefix workload (<code>--shared-prefix 512 --shared-fraction 0.8</code>).</li>
<li><b>Tensor-parallel decode</b> on a four-GPU box, with your P4.2 sharding and NCCL all-reduces.</li></ol>
<p>The capstone (C1, C2) takes the serving stack you now understand from the inside and operates it: failover across regions, and teardown of a public benchmark.</p>`,
        scene(G) {
          G.text(24, 30, "acceptance checklist", { size: 13 });
          ["T0 tests pass", "fp32 GPU greedy == reference (8 prompts), graphs on and off", "bf16: logits within your stated tolerance", "survives 2× knee rate for 5 min; preemptions in /metrics", "knee report vs vLLM + bench table filled in", "ncu/nsys summary for the top 3 kernels"].forEach((t, i) => {
            G.rect(24, 50 + i * 30, 16, 16, { stroke: "ok", rx: 3, sw: 2 });
            G.label(50, 63 + i * 30, t, { size: 12, color: "ink" });
          });
          G.text(24, 252, "stretch goals", { size: 13 });
          [["async scheduling", "overlap CPU step t+1 with GPU step t", "k"], ["spec decode in the loop", "1 + k tokens per chunk; free rejected blocks", "q"], ["radix cache as KV owner", "token-level reuse instead of block hashes", "ok"], ["tensor-parallel decode", "your P4.2 sharding, NCCL all-reduce", "v"]].forEach(([t, d, c], i) => {
            const x = 24 + (i % 2) * 300, y = 266 + Math.floor(i / 2) * 64;
            G.box(x, y, 290, 54, "", { stroke: c, rx: 9 }); G.text(x + 10, y + 22, t, { size: 12, weight: 700, color: c }); G.label(x + 10, y + 42, d, { size: 10 });
          });
          G.caption("finish the checklist, then pick one stretch goal");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>The engine is ten small modules around a four-call step: <code>scheduler.step</code>, <code>runner.execute</code>, <code>sample</code>, <code>scheduler.update</code>. Tests are the contract for swapping any of them.</li>
<li>The runner interface is narrow (chunks and block tables in, logits per producing chunk out), and three runners form a test ladder: fake, NumPy-paged, Torch. A GPU-only failure points at GPU-only code.</li>
<li>Parity is exact in fp32 and a stated logit tolerance in bf16, with the first diverging step reported; a near-tied argmax can flip without a bug.</li>
<li>Swap kernels one at a time with parity and a bench row after each; keep cuBLAS where yours loses.</li>
<li>The server is an async shell: one driver task, one lock, a queue per request, vLLM metric names; key, localhost bind, SSM forwarding.</li>
<li>A fair benchmark matches limits, dtype, hardware, workload and the KV pool (the default 1,024 blocks hold about 25 requests of 640 tokens). Explain gaps with nsys, then ncu, and never quote an unmeasured percentage.</li></ul>`,
    sim: {
      title: "Where does a decode step's time go?",
      intro: "A model, not a measurement: step time is GPU time (bytes read ÷ bandwidth) plus, when running eagerly, launch cost beyond what the GPU hides, plus Python scheduling time unless async scheduling overlaps it. The gather attention is modelled as reading the S slots, writing the copy and reading it again (3 × S × bytes per token); the paged kernel reads only the real context. Stacked bars: GPU (cyan), launch excess (amber), scheduling (violet). Every number is an example value; measure yours with nsys.",
      height: 280,
      controls: [
        { id: "model", label: "model", type: "select", value: "b1", options: [["tiny", "tiny (50 MB)"], ["b1", "1B bf16 (2 GB)"], ["b8", "8B bf16 (16 GB)"]] },
        { id: "bw", label: "memory bandwidth GB/s (example)", min: 100, max: 4000, step: 50, value: 300 },
        { id: "ctx", label: "real context per sequence", min: 16, max: 4096, step: 16, value: 512 },
        { id: "S", label: "S = max_blocks × block_size (slots)", min: 256, max: 4096, step: 256, value: 1024 },
        { id: "attn", label: "decode attention", type: "select", value: "gather", options: [["gather", "gather (TorchRunner)"], ["paged", "paged kernel"]] },
        { id: "graphs", label: "CUDA graphs", type: "select", value: "off", options: [["off", "off (eager)"], ["on", "on"]] },
        { id: "async", label: "async scheduling", type: "select", value: "off", options: [["off", "off"], ["on", "on"]] },
        { id: "N", label: "kernels per step N (example)", min: 200, max: 6000, step: 100, value: 2000 },
        { id: "L", label: "µs per eager launch (example)", min: 1, max: 20, step: 0.5, value: 5, format: (x) => x.toFixed(1) },
        { id: "R", label: "graph replay µs (example)", min: 5, max: 500, step: 5, value: 50 },
        { id: "s0", label: "Python step cost µs (example)", min: 0, max: 2000, step: 25, value: 100 },
        { id: "s", label: "per-sequence Python µs (example)", min: 0, max: 100, step: 1, value: 20 },
        { id: "B", label: "running batch B", min: 1, max: 64, step: 1, value: 8 },
      ],
      draw(G, v) {
        const Bs = [1, 2, 4, 8, 16, 32, 64], res = Bs.map((B) => stepModel(v, B));
        const mx = Math.max(...res.map((r) => r.step)) * 1.1;
        const x0 = 50, W = 570, H = 190, base = 230, bw = W / Bs.length;
        G.axes(x0, base - H, W, H);
        G.label(10, 16, "ms per decode step at batch size B (bars) · tokens/s noted above each bar", { size: 11 });
        res.forEach((r, i) => {
          const x = x0 + i * bw + 8, w = bw - 16, hg = (H * r.gpu) / mx, hl = (H * r.launchExtra) / mx, hs = (H * r.sched) / mx;
          G.rect(x, base - hg, w, hg, { fill: "k", rx: 2 });
          G.rect(x, base - hg - hl, w, hl, { fill: "v", rx: 0 });
          G.rect(x, base - hg - hl - hs, w, hs, { fill: "q", rx: 2 });
          G.text(x + w / 2, base + 16, "B=" + Bs[i], { size: 10, anchor: "middle", color: Bs[i] === v.B ? "ink" : "muted" });
          G.text(x + w / 2, base - hg - hl - hs - 14, F.num(r.tps, 0) + " t/s", { size: 9, anchor: "middle", color: "ink" });
          G.text(x + w / 2, base - hg - hl - hs - 4, r.step.toFixed(1) + " ms", { size: 9, anchor: "middle", color: "muted" });
        });
        G.label(10, 270, "cyan: GPU bytes ÷ bandwidth · amber: launches the GPU cannot hide · violet: Python scheduling, not overlapped", { size: 10 });
        const r = stepModel(v, v.B), parts = [["GPU time", r.gpu], ["launch excess", r.launchExtra], ["scheduling", r.sched]];
        const top = parts.slice().sort((a, b) => b[1] - a[1])[0];
        return [
          { title: `At B = ${v.B}`, rows: [["step time", F.ms(r.step / 1000)], ["tokens / s (all sequences)", F.num(r.tps, 0)], ["GPU busy", (100 * (1 - r.idle)).toFixed(0) + " %"], ["GPU time", F.ms(r.gpu / 1000)], ["launch excess", F.ms(r.launchExtra / 1000)], ["scheduling", F.ms(r.sched / 1000)]],
            gauge: [[Math.min(1, r.gpu / r.step), "k"], [Math.min(1, r.launchExtra / r.step), "v"], [Math.min(1, r.sched / r.step), "q"]], gaugeText: "cyan GPU · amber launch · violet scheduling",
            chip: [r.idle < 0.1, r.idle < 0.1 ? "GPU-bound: optimise kernels and bytes" : `GPU idle ${(100 * r.idle).toFixed(0)} % of the step: biggest cost is ${top[0]}`] },
          { title: "Try", rows: [["graphs on", "removes launch excess"], ["paged kernel", "cuts attention bytes"], ["async on", "hides scheduling"]],
            html: `<p class="note">Everything here is a model with example values (this environment has no GPU). The gather model charges 3 × S × KV bytes per sequence because it reads the slots, writes the dense copy and reads that again; real kernels differ. Use it to decide what to look for in your nsys and ncu traces, not as a result.</p>` },
        ];
      },
      note: "A model: GPU time = (weights + B · attention bytes) / bandwidth; step = scheduling + max(launch, GPU) without async, max(scheduling, max(launch, GPU)) with it. Launch is N·L eager or R with graphs. All inputs are example values, not measurements of any GPU, model or engine.",
    },
    practice: {
      intro: "The reference is complete; the project is to own it. Steps 1 (T0) run on a laptop; steps 2 to 7 need one g6.xlarge: follow aws.md for the cost line (look up your region's on-demand price; about 6 instance-hours for this module), spot use, no inbound ports and teardown with make down. Timing cells in the bench table are TODO(run-on: g6.xlarge) until measured.",
      items: [
        { title: "T0 green", tier: "T0 · medium", goal: "All T0 tests of #0 v1 pass, especially test_numpy_engine.py: the paged engine with chunking and preemption must produce exactly v0's greedy tokens.",
          cmd: "uv run pytest platform/engine/v1/tests -q" },
        { title: "GPU parity", tier: "T2 · medium", goal: "fp32 greedy must match the NumPy reference token for token; for bf16 compare logits within a stated tolerance and report the first diverging step.",
          cmd: "uv run --extra torch pytest platform/engine/v1/tests/test_torch_runner.py -m gpu" },
        { title: "Swap in D4 kernels", tier: "T2 · hard", goal: "Replace RMSNorm, the gather attention in TorchRunner.decode, and the projections one at a time with a parity check after each; keep cuBLAS where yours loses and say so.",
          cmd: "uv run --extra torch pytest platform/engine/v1/tests/test_torch_runner.py -m gpu -q" },
        { title: "Serve", tier: "T2 · medium", goal: "Run the OpenAI-compatible server on localhost with an API key; convert a Llama-architecture model of at most 1B parameters first with platform/engine/v0/tools/convert_hf_model.py.",
          cmd: "S2S_RUNNER=torch S2S_MODEL=<model dir> S2S_API_KEY=… \\\n  uv run uvicorn s2s_engine.server:app --app-dir platform/engine/v1 \\\n  --host 127.0.0.1 --port 8002" },
        { title: "Register in the gateway", tier: "T2 · easy", goal: "Add a backend to your gateway config (P2.7) and send a request through it. Config: name s2s-v1, url http://127.0.0.1:8002, models [\"s2s-engine-v1\"], api_key_env S2S_API_KEY.",
          cmd: "cat course/P6-engine-internals/P6.7-project-tiny-engine-gpu/README.md   # build step 5 has the YAML" },
        { title: "Benchmark against vLLM", tier: "T2 · hard", goal: "Same GPU, model, dtype and limits for both engines; also set S2S_NUM_BLOCKS from free GPU memory (the default pool holds about 25 requests of 640 tokens). Print both configs beside the table.",
          cmd: "MODEL_DIR=build/my-model HF_MODEL=<model id> bash course/P6-engine-internals/P6.7-project-tiny-engine-gpu/bench/compare.sh" },
        { title: "Profile", tier: "T2 · hard", goal: "nsys for one decode step at B = 8, eager and with graphs; ncu for your top 3 kernels. Keep the reports out of git and put the summary table in your README.",
          cmd: "nsys profile -o decode_graphs uv run --extra torch python course/P6-engine-internals/P6.6-cuda-graph-capture/examples/02_decode_itl.py" },
      ],
      labs: [
        { label: "The reference engine (read alongside your version)", path: "platform/engine/v1/s2s_engine/" },
        { label: "The step loop", path: "platform/engine/v1/s2s_engine/engine.py" },
        { label: "The HTTP server and metrics", path: "platform/engine/v1/s2s_engine/server.py" },
        { label: "Benchmark script vs vLLM (loadgen #4)", path: "course/P6-engine-internals/P6.7-project-tiny-engine-gpu/bench/compare.sh" },
        { label: "AWS guide (g6.xlarge, SSM only, make down): shared with P6.3, P6.5, P6.6", path: "course/P6-engine-internals/P6.7-project-tiny-engine-gpu/aws.md" },
        { label: "Quiz", path: "course/P6-engine-internals/P6.7-project-tiny-engine-gpu/quiz.md" },
        { label: "Read, don't copy: nano-vllm again, and vLLM docs/design/ at the pin (re-verify paths)", path: "docs/design/" },
      ],
    },
  });
})();
