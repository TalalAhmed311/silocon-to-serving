/* P6.1 — Reading nano-vllm: one request through an engine's five files, the step loop, and where a step's time goes. */
(function () {
  const F = S2S.fmt;
  // a file box: name on top, one-line role below
  const file = (G, x, y, w, name, role, color) => {
    const g = G.box(x, y, w, 54, "", { stroke: color || "ink", rx: 10 });
    G.text(x + 12, y + 22, name, { size: 14, color: color || "ink", weight: 700, parent: g });
    G.label(x + 12, y + 41, role, { size: 11, parent: g });
    return g;
  };
  const SEQ = { A: "k", B: "v", C: "q" };
  // the real trace printed by examples/01_trace_one_step.py (budget 8 tokens, 8 blocks of 4 slots)
  const TRACE = [
    { chunks: [["A", 0, 8]], free: 6, note: "A: chunk of 8" },
    { chunks: [["A", 8, 2], ["B", 8, 2], ["C", 0, 3]], free: 3, note: "B: 8-token prefix hit" },
    { chunks: [["A", 10, 1], ["B", 10, 1], ["C", 3, 1]], free: 3, note: "three decodes" },
    { chunks: [["A", 11, 1], ["B", 11, 1], ["C", 4, 1]], free: 3, note: "B finishes" },
    { chunks: [["A", 12, 1], ["C", 5, 1]], free: 6, note: "A finishes" },
    { chunks: [["C", 6, 1]], free: 6, note: "" },
    { chunks: [["C", 7, 1]], free: 8, note: "C finishes" },
  ];

  S2S.lesson({
    id: "p6-1", n: "P6.1", title: "Reading nano-vllm",
    subtitle: "Engine internals · first principles · T0 reading, T2 for timing",
    kicker: "Lesson · ≈ 35 min",
    headline: "A serving engine is a loop with three decisions",
    intro: `<p>You have used vLLM as a black box. This phase opens it. nano-vllm is a complete vLLM-style engine (continuous batching, paged KV with prefix caching, CUDA graphs) in about a thousand lines of Python. This lesson gives you the map before you read it: the loop every engine runs, the three decisions inside that loop, the files that own them, and the data the model actually receives. The rest of P6 rebuilds each file in this repository's engine, <b>#0 v1</b>.</p>`,
    facts: ["9 steps", "4 checkpoints", "1 simulator", "3 exercises"],
    legend: [["k", "what (scheduler)"], ["ok", "where (KV blocks)"], ["q", "which (sampler)"], ["w", "GPU / runner"], ["v", "CPU work"]],
    prev: "p5-10", next: "p6-2",
    steps: [
      { rail: "the problem", title: "A model is a function; serving needs a machine around it",
        body: `<p>From P1 you know what the model does: given a list of tokens, return a probability for every possible next token. That is a pure function. It knows nothing about users, memory or time.</p>
<p>A server has a different problem. Hundreds of requests arrive at random times, with different prompt lengths and different output lengths. One GPU must serve all of them, and it has a fixed amount of memory for the KV cache. Something must decide, many times per second, which requests run, which wait, where each request's keys and values live, and which token each one gets next.</p>
<p>That something is the <b>engine</b>. The model code is the easy part; the engine is where serving performance is won or lost.</p>
<div class="analogy"><b>Picture it</b>A restaurant kitchen. The stove (GPU) cooks whatever is put on it. The head chef (engine) decides which orders go on the stove now, which wait, and which pans hold whose food.</div>`,
        scene(G) {
          const req = [];
          for (let i = 0; i < 6; i++) req.push(G.box(24, 40 + i * 52, 120, 38, `request ${i + 1}`, { stroke: ["k", "v", "q", "pink", "blue", "ok"][i], color: ["k", "v", "q", "pink", "blue", "ok"][i], size: 12 }));
          G.label(24, 30, "arrive at random times", { size: 11 });
          const eng = G.box(220, 110, 200, 150, "", { stroke: "ink", rx: 14 });
          G.text(320, 140, "ENGINE", { anchor: "middle", size: 16, weight: 700 });
          ["what runs now?", "where does its KV go?", "which token next?"].forEach((t, i) => G.label(236, 172 + i * 26, "· " + t, { size: 12, color: "ink" }));
          for (let i = 0; i < 6; i++) G.arrow(148, 59 + i * 52, 216, 185, { color: "line" });
          const gpu = G.box(470, 110, 150, 150, "", { fill: "w", rx: 14, boxOpacity: 0.5 });
          G.text(545, 170, "model(tokens)", { anchor: "middle", size: 13 });
          G.text(545, 192, "→ logits", { anchor: "middle", size: 13 });
          G.label(545, 284, "the GPU: a pure function", { anchor: "middle", size: 11 });
          G.arrow(424, 185, 466, 185, { color: "ink", w: 2 });
          G.rect(220, 330, 400, 40, { stroke: "ok", rx: 8 });
          G.text(232, 355, "KV cache memory: fixed size, shared by all", { size: 12, color: "ok" });
          G.from(req, { opacity: 0, x: -30, stagger: 0.15, duration: 0.4 });
          G.from(eng, { opacity: 0, delay: 0.8, duration: 0.5 });
          G.caption("the engine turns many requests into one stream of forward passes");
        } },

      { rail: "the step loop", title: "Every engine runs the same four-line loop",
        body: `<p>Open any engine (nano-vllm, vLLM, SGLang, ours) and you find the same loop. Each turn of it is one <b>step</b>: one forward pass of the model over a batch of tokens drawn from many requests.</p>
<div class="eq">while engine.has_unfinished():
    out    = scheduler.step()        # what to compute
    logits = runner.execute(out)     # one forward pass
    tokens = sample(logits)          # which token next
    scheduler.update(out, tokens)    # append, finish, free</div>
<p>In this repository it is <code>LLMEngine.step</code> in <code>platform/engine/v1/s2s_engine/engine.py</code>. In nano-vllm it is <code>LLMEngine.step</code> in <code>engine/llm_engine.py</code>. In vLLM v1 it is <code>EngineCore.step</code>.</p>
<p>The loop is synchronous: the CPU decides, the GPU computes, the CPU reads the result and decides again. Keep that in mind for step 8.</p>`,
        scene(G) {
          const cx = 320, cy = 215, r = 135;
          const st = [["scheduler.step()", "what to compute", "k"], ["runner.execute()", "forward pass", "w"], ["sample()", "which token", "q"], ["scheduler.update()", "append · finish · free", "v"]];
          const pos = [[cx, cy - r], [cx + r + 40, cy], [cx, cy + r], [cx - r - 40, cy]];
          G.circle(cx, cy, r, { stroke: "line", sw: 2 });
          const boxes = st.map(([a, b, c], i) => {
            const [x, y] = pos[i];
            const g = G.box(x - 95, y - 30, 190, 60, "", { fill: "bg", stroke: c, rx: 10 });
            G.text(x, y - 5, a, { anchor: "middle", size: 13, color: c, weight: 700, parent: g });
            G.label(x, y + 15, b, { anchor: "middle", size: 11, parent: g });
            return g;
          });
          G.text(cx, cy - 4, "one step", { anchor: "middle", size: 16, weight: 700 });
          G.label(cx, cy + 18, "= one forward pass", { anchor: "middle", size: 12 });
          [[cx + 70, cy - r + 10, cx + r - 10, cy - 50], [cx + r - 10, cy + 50, cx + 70, cy + r - 10], [cx - 70, cy + r - 10, cx - r + 10, cy + 50], [cx - r + 10, cy - 50, cx - 70, cy - r + 10]]
            .forEach(([a, b, c, d]) => G.arrow(a, b, c, d, { color: "muted", w: 2 }));
          G.from(boxes, { opacity: 0, stagger: 0.25, duration: 0.4 });
          G.caption("CPU decides → GPU computes → CPU updates → repeat");
        } },

      { rail: "three decisions", title: "Three decisions, three owners",
        body: `<p>Strip the loop down and the engine makes exactly three decisions every step:</p>
<table><tr><th>decision</th><th>owner</th><th>lesson</th></tr>
<tr><td><b>What</b> to compute: which requests, how many tokens each</td><td>scheduler</td><td>P6.2</td></tr>
<tr><td><b>Where</b> each token's K/V lives in GPU memory</td><td>block manager (+ prefix cache)</td><td>P6.3, P6.4</td></tr>
<tr><td><b>Which</b> token comes next</td><td>sampler (+ spec-decode verifier)</td><td>P6.5</td></tr></table>
<p>The model runner does not decide anything. It executes the scheduler's decision as fast as possible (kernels from P5, CUDA graphs from P6.6). That is why the README warns: don't start reading at the model code. The engine logic lives in the scheduler and the block manager.</p>`,
        check: { q: "A request's tokens are computed, but the engine keeps its K/V in the wrong physical slots. Which component has the bug?",
          options: ["The scheduler", "The block manager", "The sampler", "The model weights"], answer: 1,
          why: "\"Where does each token's K/V live\" is the block manager's decision: it owns the block tables that map token positions to physical slots. The scheduler only decides which tokens to compute; the runner just writes where it is told." },
        scene(G) {
          const rows = [["WHAT", "scheduler.py", "which requests, how many tokens", "k"], ["WHERE", "block_manager.py", "block tables · ref counts · hashes", "ok"], ["WHICH", "sampling.py", "temperature · top-p · verify", "q"]];
          const g = rows.map(([w, f, d, c], i) => {
            const y = 40 + i * 92;
            const grp = G.group();
            G.box(24, y, 110, 66, w, { fill: c, size: 18, weight: 700, parent: grp });
            G.box(150, y, 466, 66, "", { stroke: c, parent: grp });
            G.text(166, y + 28, f, { size: 15, color: c, weight: 700, parent: grp });
            G.label(166, y + 50, d, { size: 12, parent: grp });
            return grp;
          });
          G.box(150, 330, 466, 56, "", { fill: "w", boxOpacity: 0.45 });
          G.text(166, 355, "model_runner.py: executes, decides nothing", { size: 13 });
          G.label(166, 374, "token ids + positions + slots → logits", { size: 11, color: "ink" });
          G.from(g, { opacity: 0, x: -24, stagger: 0.25, duration: 0.4 });
          G.caption("read the deciders first; the runner is just a fast function");
        } },

      { rail: "five files", title: "One request, five files",
        body: `<p>nano-vllm (pinned at <code>bb823b3e</code>, MIT licence) splits the engine into five core files. Each one maps onto a file in our engine, <code>platform/engine/v1/s2s_engine/</code>:</p>
<ul><li><code>sequence.py</code>: the request's state (tokens, status, block table).</li>
<li><code>llm_engine.py</code> (ours: <code>engine.py</code>): the loop and the public API.</li>
<li><code>scheduler.py</code>: the "what" decision.</li>
<li><code>block_manager.py</code>: the "where" decision.</li>
<li><code>model_runner.py</code>: builds the GPU inputs and runs the forward pass (and captures CUDA graphs).</li></ul>
<p>Read them in the order of the arrows: the data model first, then the deciders, then the runner. Our engine adds <code>radix_cache.py</code>, <code>spec_verify.py</code>, <code>cuda_graph.py</code> and an HTTP <code>server.py</code>, which nano-vllm leaves out.</p>
<p>Licence rule for your notes: nano-vllm is MIT, vLLM and SGLang are Apache-2.0. Quote at most 15 lines at a time with the file path and the pinned commit, and annotate rather than copy.</p>`,
        scene(G) {
          const e = file(G, 24, 40, 200, "llm_engine.py", "generate · step", "ink");
          const s = file(G, 24, 200, 200, "sequence.py", "tokens · status · table", "ink");
          const sc = file(G, 300, 40, 200, "scheduler.py", "schedule · postprocess", "k");
          const bm = file(G, 300, 140, 200, "block_manager.py", "allocate · hash · free", "ok");
          const mr = file(G, 300, 260, 200, "model_runner.py", "prepare · run · graphs", "w");
          G.arrow(228, 67, 296, 67, { color: "k" }); G.label(236, 60, "schedule", { size: 10 });
          G.arrow(124, 98, 124, 196, { color: "ink" }); G.label(130, 150, "add_request", { size: 10 });
          G.arrow(400, 98, 400, 136, { color: "ok" }); G.label(406, 122, "allocate", { size: 10 });
          G.path("M 228 77 C 260 120, 260 260, 296 287", { color: "w", arrow: true }); G.label(250, 200, "run", { size: 10 });
          G.box(520, 270, 104, 34, "sampler", { stroke: "q", color: "q", size: 12 });
          G.arrow(504, 287, 516, 287, { color: "q" });
          G.path("M 572 266 C 572 30, 540 30, 504 60", { color: "q", dash: "4 4", arrow: true }); G.label(578, 150, "token ids", { size: 10, color: "q" });
          G.text(24, 352, "nano-vllm bb823b3e  ↔  platform/engine/v1/s2s_engine/", { size: 13 });
          G.label(24, 376, "same five roles in both; ours adds radix_cache, spec_verify, cuda_graph, server", { size: 11 });
          G.from([s, e, sc, bm, mr], { opacity: 0, y: 10, stagger: 0.18, duration: 0.35 });
          G.caption("data model → deciders → runner, then tokens flow back");
        } },

      { rail: "num_computed", legend: [["k", "K/V in cache"], ["ok", "prefix-cache hit"], ["q", "new token"], ["hot", "recompute"]], title: "One counter drives the whole engine",
        body: `<p>A request's state is mostly one list of tokens (prompt + output so far) and one number: <code>num_computed</code>, the count of tokens whose keys and values are already in the cache. Everything the scheduler does is "compute the tokens after <code>num_computed</code>":</p>
<div class="eq">num_uncomputed = len(prompt + output) - num_computed

fresh prompt of 40     → 40 to compute (prefill)
prefill chunk of 16    → 40 → 24 → 8 → 0 over 3 steps
decode                 → exactly 1 (the newest token)
prefix-cache hit of 32 → starts at 32, not 0
preempted (recompute)  → back to 0: redo it all</div>
<p>This is why chunked prefill, prefix caching and preemption are not three special cases in the code: they all just move <code>num_computed</code>. In <code>sequence.py</code>, look at <code>num_uncomputed</code> and <code>in_prefill</code>. vLLM's equivalent is <code>Request.num_computed_tokens</code>.</p>
<p>A common mistake is reading <code>num_computed</code> as "the prompt length". After a prefix hit it starts above zero; after a recompute preemption it falls back to zero.</p>`,
        check: { q: "A 40-token prompt gets a prefix-cache hit on its first 32 tokens. What is num_computed, and what does the scheduler compute next?",
          options: ["0; the whole 40-token prompt", "32; the remaining 8 prompt tokens (or a chunk of them)", "40; the first output token directly"], answer: 1,
          why: "The 32 hit tokens already have K/V in the cache, so num_computed starts at 32. The last prompt tokens still have to run through the model: the engine needs logits at the final prompt position to sample the first output token." },
        scene(G) {
          const n = 20, cw = 28, x0 = 24;
          const rows = [["fresh prompt", 0, "k"], ["after prefix hit", 12, "ok"], ["mid prefill chunk", 16, "k"], ["decoding", 20, "k"], ["after preemption", 0, "hot"]];
          rows.forEach(([lab, c, col], r) => {
            const y = 50 + r * 66;
            G.label(x0, y - 6, lab + ` · num_computed = ${c}`, { size: 12, color: "ink" });
            const cells = G.grid(x0, y, 1, n + (r === 3 ? 1 : 0), cw, 30, (_, k) => (k < c ? col : k === n ? "q" : "line"), { gap: 3, rx: 3 });
            if (r === 3) G.label(x0 + n * cw + 4, y + 50, "new", { size: 10, color: "q" });
            G.line(x0 + c * cw - 2, y - 4, x0 + c * cw - 2, y + 34, { color: "ink", w: 2 });
            if (r === 4) G.from(cells, { opacity: 0, stagger: 0.02 });
          });
          G.label(x0, 400, "filled = K/V in cache · dark = still to compute", { size: 12 });
          G.caption("prefill, chunks, prefix hits and recompute all just move one marker");
        } },

      { rail: "a real trace", legend: [["k", "request A"], ["v", "request B"], ["q", "request C"], ["ok", "free KV blocks"]], title: "Watching the loop: three requests, seven steps",
        body: `<p>Run the example and the engine prints one line per step:</p>
<div class="eq">uv run python course/P6-engine-internals/
  P6.1-reading-nano-vllm/examples/01_trace_one_step.py</div>
<p>It uses the fake model (logits are a hash of the token history), a pool of <b>8 blocks of 4 token slots</b>, and a budget of <b>8 tokens per step</b>. Request A and B both have the same 10-token prompt; C has a 3-token prompt. The printed trace:</p>
<div class="eq">step 1 | 8 tok | A[0:8]                  free 6
step 2 | 7 tok | A[8:10] B[8:10] C[0:3]  free 3
step 3 | 3 tok | A[10:11] B[10:11] C[3:4]
step 4 | 3 tok | ...  B finishes
step 5 | 2 tok | A[12:13] C[5:6]  A finishes
step 6 | 1 tok | C[6:7]
step 7 | 1 tok | C[7:8]  C finishes   free 8</div>
<p>Read it closely. Step 1: A's prompt does not fit the budget, so it gets a chunk of 8. Step 2: B starts at position 8, not 0: its first two blocks (8 tokens) hashed the same as A's, so they were reused (a prefix hit of 8 tokens). Prefill chunks and decodes share one step. From step 3, each running request decodes one token per step.</p>`,
        scene(G) {
          const x0 = 70, cw = 40, y0 = 52, rh = 44;
          G.label(x0, 30, "token budget: 8 slots per step", { size: 12 });
          G.label(x0 + 8 * cw + 18, 30, "free KV blocks (of 8)", { size: 12, color: "ok" });
          const all = [];
          TRACE.forEach((s, r) => {
            const y = y0 + r * rh;
            G.label(20, y + 22, `step ${r + 1}`, { size: 11, color: "ink" });
            for (let k = 0; k < 8; k++) G.rect(x0 + k * cw, y, cw - 3, rh - 8, { stroke: "line", rx: 3 });
            let k = 0;
            s.chunks.forEach(([name, st, num]) => {
              const g = G.box(x0 + k * cw, y, num * cw - 3, rh - 8, num > 1 ? `${name}[${st}:${st + num}]` : name, { fill: SEQ[name], size: num > 1 ? 11 : 12, rx: 3 });
              all.push(g); k += num;
            });
            G.rect(x0 + 8 * cw + 18, y + 6, s.free * 7, rh - 20, { fill: "ok", rx: 2, opacity: 0.8 });
            G.label(x0 + 8 * cw + 80, y + 22, String(s.free), { size: 11, color: "ok" });
            if (s.note) G.label(x0 + 8 * cw + 96, y + 22, s.note, { size: 11, color: "ink" });
          });
          G.from(all, { opacity: 0, x: -10, stagger: 0.08, duration: 0.25 });
          G.caption("chunk, prefix hit, mixed batch, decode: the whole engine in 7 rows");
        } },

      { rail: "runner inputs", title: "What the model runner actually receives",
        body: `<p>The scheduler's output is a list of <b>chunks</b>: (sequence, start position, number of tokens). The runner flattens all chunks into one batch and, for every scheduled token, needs three things:</p>
<ul><li>its <b>token id</b>;</li><li>its <b>position</b> (for RoPE and for the causal mask);</li><li>its <b>slot</b>: the physical cache index where its K and V get written.</li></ul>
<p>The slot comes from the sequence's <b>block table</b>, which lists the physical block ids holding logical blocks 0, 1, 2, …:</p>
<div class="eq">slot = block_table[pos // bs] * bs + pos % bs

bs = 4, block_table = [5, 2, 7], pos = 9:
  9 // 4 = 2 → block 7;   9 % 4 = 1
  slot = 7 * 4 + 1 = 29</div>
<p>For attention, each sequence also needs its whole block table and its context length, so the kernel can gather every earlier key and value. nano-vllm builds these in <code>prepare_prefill</code> (<code>cu_seqlens</code>, <code>slot_mapping</code>) and <code>prepare_decode</code> (<code>context_lens</code>, <code>block_tables</code>). Ours is <code>BlockManager.slot</code> and <code>NumpyPagedRunner._slot</code>.</p>`,
        check: { q: "Block size 4, block_table = [5, 2, 7]. Where is the K/V of position 6 stored?",
          options: ["slot 6", "slot 10", "slot 22", "slot 30"], answer: 1,
          why: "6 // 4 = 1, so logical block 1, which is physical block 2. 6 % 4 = 2 is the offset inside it. slot = 2·4 + 2 = 10." },
        scene(G) {
          const bs = 4, cw = 30;
          G.text(24, 36, "logical positions of one sequence (bs = 4)", { size: 12, color: "muted" });
          const table = [5, 2, 7];
          for (let p = 0; p < 12; p++) {
            const lb = Math.floor(p / bs), x = 24 + p * 40 + lb * 12;
            G.box(x, 48, 34, 32, String(p), { fill: p === 9 ? "q" : "k", size: 12, rx: 4, boxOpacity: p === 9 ? 1 : 0.6 });
          }
          table.forEach((b, i) => G.box(24 + i * 172, 100, 154, 30, `logical ${i} → block ${b}`, { stroke: "ok", color: "ok", size: 12 }));
          G.text(24, 168, "physical KV pool: 8 blocks × 4 slots = 32 slots", { size: 12, color: "muted" });
          for (let b = 0; b < 8; b++) {
            const x = 24 + b * 74;
            G.label(x + 2, 190, `block ${b}`, { size: 10, color: table.includes(b) ? "ok" : "muted" });
            for (let o = 0; o < bs; o++) {
              const slot = b * bs + o, own = table.indexOf(b);
              G.box(x, 198 + o * 32, 64, 28, String(slot), { fill: slot === 29 ? "q" : own >= 0 ? "k" : null, stroke: own >= 0 ? undefined : "line", color: own >= 0 ? "bg" : "muted", size: 11, rx: 3, boxOpacity: slot === 29 ? 1 : 0.55 });
            }
          }
          const a = G.path("M 427 82 C 430 150, 560 150, 556 290", { color: "q", w: 2, arrow: true });
          G.text(24, 352, "pos 9 → block_table[9 // 4] = block 7, offset 9 % 4 = 1", { size: 13, color: "q" });
          G.text(24, 376, "slot = 7 × 4 + 1 = 29", { size: 14, color: "q", weight: 700 });
          G.from(a, { opacity: 0, duration: 0.6, delay: 0.3 });
          G.caption("the block table turns a logical position into a physical slot");
        } },

      { rail: "where time goes", legend: [["v", "CPU: Python + driver"], ["w", "GPU work"], ["pink", "kernel launch"]], title: "At small batch, the CPU can cost as much as the GPU",
        body: `<p>A decode step at batch 1 is a few milliseconds of memory-bound GPU work (P1.2: read every weight once). But that work is split into hundreds of small kernels: several per layer for norms, projections, RoPE, attention, the MLP. Each kernel launch costs a few microseconds of CPU and driver time.</p>
<p>Around the forward pass the CPU also runs Python: scheduling, building the input tensors (token ids, positions, slot mapping), sampling bookkeeping, appending tokens, checking stop conditions. None of that shrinks when the GPU gets faster.</p>
<div class="eq">step time ≈ CPU: schedule + prepare
                + launches + update
            + GPU: bytes read ÷ bandwidth</div>
<p>At batch 1 the GPU part is small, so the fixed CPU part is a large share. At batch 64 the GPU part grows (more KV to read, more tokens) while the per-step CPU part grows slowly, so its share falls. Two fixes come later: <b>CUDA graphs</b> (P6.6) replace hundreds of launches with one, and <b>async scheduling</b> (vLLM v1) prepares step <i>n+1</i> on the CPU while the GPU runs step <i>n</i>. Exercise 2 measures this on a real GPU; the simulator lets you play with a model of it.</p>`,
        check: { q: "Why can CPU-side work dominate a decode step at batch 1 but not at batch 64?",
          options: ["Python runs faster at large batch", "The GPU part is small at batch 1, while scheduling and kernel launches are a roughly fixed cost per step", "At batch 64 the CPU work is moved to the GPU automatically"], answer: 1,
          why: "Launch overhead and much of the Python per-step work don't depend on how much data each kernel processes. When the GPU work per step is only a few milliseconds, that fixed cost is a big fraction; when the GPU work grows with batch, the same fixed cost becomes a small fraction." },
        scene(G) {
          const rowsY = [70, 230];
          [["batch 1", 1], ["batch 64", 4.2]].forEach(([lab, gpuScale], r) => {
            const y = rowsY[r];
            G.text(24, y - 14, lab + " (illustrative proportions, not measured)", { size: 12, color: "ink" });
            let x = 24;
            const seg = (w, col, t) => { const g = G.box(x, y, w, 44, t, { fill: col, size: 11, rx: 4 }); x += w + 2; return g; };
            seg(58, "v", "sched"); seg(50, "v", "prep");
            const launches = []; for (let i = 0; i < 14; i++) launches.push(G.rect(x + i * 6, y + 50, 4, 10, { fill: "pink", rx: 1 }));
            seg(Math.round(80 * gpuScale), "w", "GPU forward");
            seg(58, "v", "sample"); seg(58, "v", "update");
            G.label(24, y + 78, "pink ticks: kernel launches issued by the CPU, one per kernel", { size: 10, color: "pink" });
          });
          G.rect(24, 360, 14, 14, { fill: "v", rx: 2 }); G.label(44, 372, "CPU: Python + driver", { size: 12 });
          G.rect(220, 360, 14, 14, { fill: "w", rx: 2 }); G.label(240, 372, "GPU: memory-bound compute", { size: 12 });
          G.caption("the CPU bars barely change with batch; the GPU bar grows");
        } },

      { rail: "the big map", legend: [["k", "decode token"], ["v", "prompt token"], ["hot", "stalled decode"]], title: "Same roles in vLLM and SGLang",
        body: `<p>Once you can name the role, you can find it in any engine. At the pinned versions (vLLM <code>v0.31.0</code>, SGLang <code>v0.5.21</code>; re-verify each path at the pin, files move between releases):</p>
<ul><li><b>request</b>: ours <code>sequence.py</code> · vLLM <code>v1/request.py</code> · SGLang <code>managers/schedule_batch.py</code></li>
<li><b>loop</b>: <code>engine.py</code> · <code>v1/engine/core.py</code> · <code>managers/scheduler.py</code> (event loop)</li>
<li><b>what</b>: <code>scheduler.py</code> · <code>v1/core/sched/scheduler.py</code> · <code>scheduler.py</code> + <code>schedule_policy.py</code></li>
<li><b>where</b>: <code>block_manager.py</code>, <code>radix_cache.py</code> · <code>v1/core/kv_cache_manager.py</code>, <code>block_pool.py</code> · <code>mem_cache/radix_cache.py</code></li>
<li><b>run</b>: <code>model_runner.py</code>, <code>cuda_graph.py</code> · <code>v1/worker/gpu_model_runner.py</code> · <code>model_executor/model_runner.py</code></li>
<li><b>which</b>: <code>sampling.py</code>, <code>spec_verify.py</code> · <code>v1/sample/</code> · <code>layers/sampler.py</code></li></ul>
<p>One real difference to look for (exercise 3): nano-vllm schedules prefill and decode in <b>separate</b> steps, while vLLM v1 and our engine mix them under one token budget, as in step 2 of the trace (verify both at the pins). The scene shows the trade-off on a toy trace: a new 12-token prompt P arrives while a and b decode. Prefill-only steps give P its first token sooner (better TTFT) but stall a and b for a step (worse ITL). The mixed budget keeps a and b flowing and delays P by one step.</p>`,
        scene(G) {
          // one long prompt P (12 tokens) arrives while two requests decode; budget 8
          const x0 = 150, cw = 38;
          const lanes = [
            ["nano-vllm", "prefill-only steps", [["P[0:12] whole prompt", 12, "v"]], [["a", 1, "k"], ["b", 1, "k"], ["p", 1, "v"]], [["a", 1, "k"], ["b", 1, "k"], ["p", 1, "v"]]],
            ["vLLM v1 / #0 v1", "one mixed budget of 8", [["a", 1, "k"], ["b", 1, "k"], ["P[0:6]", 6, "v"]], [["a", 1, "k"], ["b", 1, "k"], ["P[6:12]", 6, "v"]], [["a", 1, "k"], ["b", 1, "k"], ["p", 1, "v"]]],
          ];
          const all = [];
          lanes.forEach(([name, sub, ...steps], li) => {
            const y0 = 50 + li * 180;
            G.text(24, y0 + 4, name, { size: 14, weight: 700, color: li ? "k" : "ink" });
            G.label(24, y0 + 22, sub, { size: 11 });
            steps.forEach((chs, r) => {
              const y = y0 + 40 + r * 36 + (li === 0 && r > 0 ? 18 : 0);
              G.label(x0 - 10, y + 19, `step ${r + 1}`, { size: 10, anchor: "end" });
              let k = 0;
              chs.forEach(([lab, n, col]) => { all.push(G.box(x0 + k * cw, y, n * cw - 3, 28, lab, { fill: col, size: 11, rx: 3 })); k += n; });
              if (li === 0 && r === 0) G.text(x0, y + 50, "a and b get no token in step 1", { size: 11, color: "hot" });
              if (li === 1 && r === 1) G.text(x0 + 8 * cw + 10, y + 19, "← P's first token: step 2", { size: 11, color: "v" });
              if (li === 0 && r === 0) G.text(x0, y - 6, "P's first token: step 1", { size: 11, color: "v" });
            });
          });
          G.label(24, 412, "cyan = decodes of running requests a, b · amber = new 12-token prompt P", { size: 11 });
          G.from(all, { opacity: 0, x: -8, stagger: 0.04, duration: 0.25 });
          G.caption("separate: faster first token for P, a stall for a and b · mixed: the reverse");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>An engine is a loop: schedule, execute, sample, update. Each turn is one forward pass over tokens from many requests.</li>
<li>It makes three decisions per step: what to compute (scheduler), where K/V lives (block manager), which token next (sampler). The runner decides nothing.</li>
<li><code>num_computed</code> unifies prefill, chunking, prefix hits and recompute.</li>
<li>The runner needs token ids, positions and slots (<code>block_table[pos // bs]·bs + pos % bs</code>), plus block tables and context lengths for attention.</li>
<li>At small batch, CPU scheduling and launch overhead are a large share of a step; CUDA graphs and async scheduling attack it.</li></ul>`,
    sim: {
      title: "Where does a decode step's time go?",
      intro: "A simple model of one decode step: a CPU part (Python scheduling per sequence, a fixed per-step cost, and one launch per kernel) and a GPU part (read all weights plus every sequence's KV cache at the memory bandwidth). Every hardware number here is an example value, not a measurement: exercise 2 replaces them with your own timings.",
      height: 280,
      controls: [
        { id: "w", label: "model weights (GB)", type: "select", value: 1.2, options: [[1.2, "0.6B params, bf16 ≈ 1.2 GB"], [16, "8B params, bf16 ≈ 16 GB"]] },
        { id: "ctx", label: "context per sequence (tokens)", min: 128, max: 8192, step: 128, value: 1024 },
        { id: "kvt", label: "KV bytes per token (KiB; Llama-3-8B bf16 = 128)", min: 16, max: 256, step: 16, value: 112 },
        { id: "bw", label: "memory bandwidth, GB/s (example value: check the spec sheet)", min: 200, max: 3500, step: 50, value: 300 },
        { id: "k", label: "kernels per step", min: 50, max: 1500, step: 10, value: 400 },
        { id: "lo", label: "launch cost per kernel, µs (example)", min: 1, max: 20, value: 5 },
        { id: "py", label: "Python cost per sequence per step, µs (example)", min: 5, max: 200, step: 5, value: 40 },
        { id: "fx", label: "fixed Python cost per step, ms (example)", min: 0, max: 5, step: 0.1, value: 1 },
        { id: "g", label: "CUDA graphs", type: "select", value: "off", options: [["off", "off: one launch per kernel"], ["on", "on: one graph launch per step"]] },
      ],
      draw(G, v) {
        const Bs = [1, 2, 4, 8, 16, 32, 64, 128];
        const comp = (B) => {
          const launch = (v.g === "on" ? 1 : v.k) * v.lo * 1e-6;
          const cpu = v.fx * 1e-3 + B * v.py * 1e-6 + launch;
          const bytes = v.w * 1e9 + B * v.ctx * v.kvt * 1024;
          const gpu = bytes / (v.bw * 1e9);
          return { cpu, gpu, tot: cpu + gpu };
        };
        const res = Bs.map(comp);
        const mx = Math.max(...res.map((r) => r.tot)) * 1.08;
        G.label(10, 16, "step time, stacked: CPU (amber) + GPU (grey); line = CPU share", { size: 11 });
        G.axes(40, 28, 580, 210);
        G.label(620, 274, "batch size (sequences per step)", { size: 11, anchor: "end" });
        res.forEach((r, i) => {
          const x = 60 + i * 70, hc = (200 * r.cpu) / mx, hg = (200 * r.gpu) / mx;
          G.rect(x, 238 - hg, 40, hg, { fill: "w", rx: 2 });
          G.rect(x, 238 - hg - hc, 40, hc, { fill: "v", rx: 2 });
          G.text(x + 20, 256, String(Bs[i]), { anchor: "middle", size: 11, color: "muted" });
          G.text(x + 20, 232 - hg - hc, F.ms(r.tot), { anchor: "middle", size: 9, color: "ink" });
        });
        const pts = res.map((r, i) => `${i ? "L" : "M"} ${80 + i * 70} ${238 - 200 * (r.cpu / r.tot)}`).join(" ");
        G.path(pts, { color: "hot", w: 2 });
        G.line(40, 38, 620, 38, { color: "line", dash: "3 4" }); G.label(616, 34, "CPU share 100 %", { size: 10, anchor: "end", color: "hot" });
        res.forEach((r, i) => G.circle(80 + i * 70, 238 - 200 * (r.cpu / r.tot), 3, { fill: "hot" }));
        const b1 = res[0], b64 = res[6];
        const cross = Bs.find((B, i) => res[i].cpu / res[i].tot < 0.1);
        return [
          { title: "Batch 1", rows: [["CPU", F.ms(b1.cpu)], ["GPU", F.ms(b1.gpu)], ["CPU share", (100 * b1.cpu / b1.tot).toFixed(0) + " %"], ["tokens/s", F.num(1 / b1.tot, 0)]] },
          { title: "Batch 64", rows: [["CPU", F.ms(b64.cpu)], ["GPU", F.ms(b64.gpu)], ["CPU share", (100 * b64.cpu / b64.tot).toFixed(0) + " %"], ["tokens/s (all sequences)", F.num(64 / b64.tot, 0)]] },
          { title: "When CPU stops mattering", chip: [!!cross, cross ? `CPU share < 10 % from batch ${cross}` : "CPU share stays ≥ 10 % up to batch 128"],
            html: `<p class="note">Turn CUDA graphs on and watch the batch-1 CPU bar shrink: the launch cost collapses to one launch. The Python cost does not; that is what async scheduling hides.</p>` },
        ];
      },
      note: "A model with example costs, not a benchmark. It ignores compute time (fine for memory-bound decode), kernel tails, and overlap between CPU and GPU.",
    },
    practice: {
      intro: "Exercise 1 and 3 are T0 (reading and tracing with the fake model). Exercise 2 needs one GPU; see aws.md for cost and teardown. The engine's own tests run on a laptop.",
      items: [
        { title: "Concept-mapping table", tier: "T0 · medium", goal: "Extend the five-file map to function level for nano-vllm, #0 v1, vLLM v1 and SGLang; quiz questions 1–4 should be answerable from it.",
          cmd: "cat course/P6-engine-internals/P6.1-reading-nano-vllm/exercises/01-mapping.md" },
        { title: "Where does a step's time go?", tier: "T2 · medium", goal: "Add timers to nano-vllm's step (schedule, prepare, forward, sample, postprocess) at batch 1, 8, 64, with and without CUDA graphs.",
          cmd: "cat course/P6-engine-internals/P6.1-reading-nano-vllm/exercises/02-timers.md" },
        { title: "One scheduling difference, reproduced", tier: "T0/T2 · hard", goal: "Write a request trace that exposes a real difference between nano-vllm's and vLLM's scheduler, and show both per-step schedules.",
          cmd: "cat course/P6-engine-internals/P6.1-reading-nano-vllm/exercises/03-difference.md" },
        { title: "Trace the step loop", tier: "T0 · easy", goal: "Print what the scheduler decides each step for three requests with the fake model (the trace in step 6).",
          cmd: "uv run python course/P6-engine-internals/P6.1-reading-nano-vllm/examples/01_trace_one_step.py" },
        { title: "The engine's test suite", tier: "T0", goal: "Run every T0 test of #0 v1: scheduler, block manager, radix cache, sampling, verification, server.",
          cmd: "uv run pytest platform/engine/v1/tests -q" },
      ],
      labs: [
        { label: "Our engine (read alongside nano-vllm)", path: "platform/engine/v1/s2s_engine/" },
        { label: "The step loop", path: "platform/engine/v1/s2s_engine/engine.py" },
        { label: "AWS guide for exercise 2 (g6.xlarge, SSM only, make down)", path: "course/P6-engine-internals/P6.1-reading-nano-vllm/aws.md" },
        { label: "Quiz", path: "course/P6-engine-internals/P6.1-reading-nano-vllm/quiz.md" },
      ],
    },
  });
})();
