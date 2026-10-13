/* P6.2 — The scheduler: one step = one decision about which tokens go through the model. Token budget, chunked prefill,
   preemption by recompute, priorities with an anti-starvation bound. Mirrors platform/engine/v1/s2s_engine/scheduler.py. */
(function () {
  const F = S2S.fmt;
  const cdiv = (a, b) => Math.ceil(a / b);

  /* ---------- a JS port of Scheduler.step()/update() with block counting (no prefix cache, equal priorities) ---------- */
  function simulate(o) {
    const { budget, D, L, blocks, bs, chunked } = o;
    const maxSeqs = 64;
    let held = 0; // blocks in use
    const waiting = [], running = [], steps = [];
    let id = 0, preemptions = 0;
    const mk = (name, prompt, maxOut) => ({ id: id++, name, prompt, maxOut, out: 0, comp: 0, hw: 0, blocks: 0, pre: 0 });
    const fits = (prompt, maxOut) => cdiv(prompt + maxOut, bs) <= blocks;
    const free = () => blocks - held;
    const len = (s) => s.prompt + s.out;
    const can = (s, n) => Math.max(0, cdiv(s.comp + n, bs) - s.blocks) <= free();
    const alloc = (s, n) => { const need = Math.max(0, cdiv(s.comp + n, bs) - s.blocks); s.blocks += need; held += need; };
    const release = (s) => { held -= s.blocks; s.blocks = 0; };
    let rejected = 0, stuck = false;
    for (let i = 0; i < D; i++) { const s = mk("d" + i, 2, 60); if (fits(s.prompt, s.maxOut)) waiting.push(s); else rejected++; }
    let longSeq = null, longRejected = false, longArrivedStep = null;
    const preempt = (v, rec) => { running.splice(running.indexOf(v), 1); release(v); v.comp = 0; v.pre++; preemptions++; waiting.push(v); rec.pre.push(v.name); };
    for (let t = 0; t < 4000; t++) {
      if (t === 1 && L > 0) { const s = mk("LONG", L, 4); if (fits(L, 4)) { longSeq = s; waiting.push(s); longArrivedStep = t; } else longRejected = true; }
      if (!waiting.length && !running.length && t > 1) break;
      const rec = { dec: 0, pre: [], pf: 0, rc: 0, chunks: [], free: 0, noToken: 0 };
      let bud = budget; const sched = new Set();
      for (const s of running.slice()) {
        if (!running.includes(s)) continue;
        let n = len(s) - s.comp; if (n <= 0) continue;
        if (chunked) n = Math.min(n, bud); else if (n > bud) continue;
        if (n === 0) break;
        while (!can(s, n)) {
          const cands = running.filter((c) => c !== s && !sched.has(c.id));
          const v = cands.length ? cands[cands.length - 1] : s;
          preempt(v, rec); if (v === s) break;
        }
        if (!running.includes(s)) continue;
        alloc(s, n); rec.chunks.push([s, s.comp, n]); sched.add(s.id); bud -= n;
      }
      while (!rec.pre.length && bud > 0 && running.length < maxSeqs && waiting.length) {
        const s = waiting.slice().sort((a, b) => (b.pre > 0) - (a.pre > 0) || a.id - b.id)[0];
        let n = len(s) - s.comp;
        if (chunked) n = Math.min(n, bud); else if (n > bud) break;
        if (!can(s, n)) { release(s); break; }
        alloc(s, n); waiting.splice(waiting.indexOf(s), 1); running.push(s); rec.chunks.push([s, s.comp, n]); sched.add(s.id); bud -= n;
      }
      // classify tokens, then update
      let gotToken = new Set();
      rec.chunks.forEach(([s, st, n]) => {
        const rc = Math.max(0, Math.min(st + n, s.hw) - st);
        rec.rc += rc;
        const rest = n - rc;
        if (st >= s.prompt && n === 1 && rc === 0) rec.dec += rest; else rec.pf += rest;
      });
      rec.chunks.forEach(([s, st, n]) => {
        s.comp = st + n; s.hw = Math.max(s.hw, s.comp);
        if (st + n === len(s)) {
          s.out++; gotToken.add(s.id);
          if (s === longSeq && s.out === 1) longSeq.firstStep = t;
          if (s.out >= s.maxOut) { running.splice(running.indexOf(s), 1); release(s); }
        }
      });
      if (!rec.chunks.length) { if (waiting.length || running.length) { stuck = true; } break; }
      rec.free = free();
      // a decoder that is mid-flight but got no token this step is "stalled"
      rec.noToken = running.filter((s) => s !== longSeq && s.out > 0 && !gotToken.has(s.id)).length + rec.pre.filter((n) => n !== "LONG").length;
      rec.total = rec.chunks.reduce((a, c) => a + c[2], 0);
      steps.push(rec);
    }
    const ttft = longSeq && longSeq.firstStep !== undefined ? longSeq.firstStep - longArrivedStep + 1 : null;
    let stalled = 0;
    if (longSeq && longSeq.firstStep !== undefined) for (let t = longArrivedStep; t <= longSeq.firstStep; t++) if (steps[t] && steps[t].noToken > 0) stalled++;
    return { steps, ttft, preemptions, maxTok: Math.max(...steps.map((s) => s.total)), stalled, longRejected, rejected, recompute: steps.reduce((a, s) => a + s.rc, 0), peak: Math.max(...steps.map((s) => blocks - s.free)) };
  }

  // a labelled cell
  const cell = (G, x, y, w, h, t, color, o = {}) => G.box(x, y, w, h, t, Object.assign({ fill: color, size: 11, rx: 3 }, o));

  S2S.lesson({
    id: "p6-2", n: "P6.2", title: "The scheduler",
    subtitle: "Engine internals · first principles · T0, pure Python with a fake model",
    kicker: "Lesson · ≈ 45 min",
    headline: "Every step is a decision about which tokens run",
    intro: `<p>P2.3 showed continuous batching from the outside. Here you build the part that does it: the scheduler, a plain Python object that is called once per step and answers one question: <b>which tokens go through the model in the next forward pass?</b> You will see the limits the answer must respect, the order of the decisions, how a long prompt is cut into chunks, what happens when KV memory runs out, and how priorities can starve a request unless you bound the wait. The reference is <code>platform/engine/v1/s2s_engine/scheduler.py</code>, which you re-implement in the exercises.</p>`,
    facts: ["9 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "decode token"], ["v", "prefill token"], ["hot", "recompute / victim"], ["ok", "free / allowed"], ["w", "limit"]],
    prev: "p6-1", next: "p6-3",
    steps: [
      { rail: "the decision", title: "One step, one question",
        body: `<p>The engine loop from P6.1 calls <code>scheduler.step()</code> before every forward pass. The scheduler holds two lists. <b>Waiting</b> requests have not started (or were sent back). <b>Running</b> requests already hold KV cache blocks and are partway through their prompt or their output.</p>
<p>Its job is to turn those two lists into a <b>batch</b>: a list of <b>chunks</b>, each "sequence S, starting at position <i>p</i>, run <i>n</i> tokens". A decoding sequence contributes one token. A sequence in the middle of its prompt contributes a piece of the prompt. Everything is judged by one number, <b>tokens in this forward pass</b>, because that is what costs GPU time.</p>
<p>The scheduler knows nothing about the model. It never sees a weight or a logit. It only moves requests between lists and decides how many tokens each one gets. That is why it is pure Python and fully testable on a laptop.</p>`,
        scene(G) {
          G.label(24, 34, "waiting: not started yet", { size: 12 });
          const w = [["W1 · 20 tok", "v"], ["W2 · 5 tok", "v"], ["W3 · 9 tok", "v"]].map(([t, c], i) => G.box(24 + i * 130, 44, 120, 34, t, { stroke: c, color: c, size: 12 }));
          G.label(24, 112, "running: hold KV blocks", { size: 12 });
          const r = [["A · decode", "k"], ["B · decode", "k"], ["C · 20 left", "v"]].map(([t, c], i) => G.box(24 + i * 130, 122, 120, 34, t, { fill: c, size: 12 }));
          G.arrow(320, 164, 320, 196, { color: "ink", w: 2 });
          const s = G.box(124, 200, 392, 56, "", { stroke: "ink", rx: 12 });
          G.text(320, 224, "scheduler.step()", { anchor: "middle", size: 15, weight: 700, parent: s });
          G.label(320, 244, "who gets how many tokens in the NEXT forward pass?", { anchor: "middle", size: 11, parent: s });
          G.arrow(320, 260, 320, 290, { color: "ink", w: 2 });
          G.text(24, 312, "the batch (one forward pass)", { size: 13 });
          const cells = [["A", "k"], ["B", "k"], ["C", "v"], ["C", "v"], ["C", "v"], ["C", "v"], ["C", "v"], ["C", "v"]].map(([t, c], i) => cell(G, 24 + i * 72, 324, 66, 34, t, c, { size: 13 }));
          G.label(24, 384, "a list of chunks: (sequence, start, n)  ·  here 1 + 1 + 6 = 8 tokens", { size: 12, color: "ink" });
          G.from(cells, { opacity: 0, y: 10, stagger: 0.06, duration: 0.25 }); G.from([...w, ...r], { opacity: 0, stagger: 0.05, duration: 0.25 });
          G.caption("two lists in, one batch of chunks out");
        } },

      { rail: "three limits", title: "Three limits stop a step from growing",
        body: `<p>Why not put every waiting request in? Three resources run out, and the scheduler must respect all of them (<code>SchedulerConfig</code> and the block pool):</p>
<ul><li><b>Token budget</b> (<code>max_num_batched_tokens</code>, default 2048 here): the most tokens one forward pass may contain. A pass with more tokens takes longer, so this caps how long every running sequence waits for its next token (P2.3, chunked prefill).</li>
<li><b>Sequence cap</b> (<code>max_num_seqs</code>, default 64): the most sequences running at once. Each carries per-sequence work and, later, shapes captured in CUDA graphs (P6.6).</li>
<li><b>KV blocks</b>: the memory pool from P2.1/P6.3. A sequence needs blocks for every token it has computed. When none are free, nothing more can be admitted.</li></ul>
<div class="eq">example: 31 sequences decoding, budget 2048
  decode tokens        = 31 × 1       =   31
  left for prefill     = 2048 − 31    = 2017
  sequence slots left  = 64 − 31      =   33</div>
<p>A step stops adding work as soon as <b>any</b> limit is reached.</p>`,
        scene(G) {
          const gauge = (y, title, sub, parts, full) => {
            G.text(24, y, title, { size: 13, weight: 700 }); G.label(24, y + 18, sub, { size: 11 });
            G.rect(24, y + 28, 580, 26, { stroke: "w", rx: 5 });
            let x = 24; const out = [];
            parts.forEach(([frac, c, t]) => { const w = 580 * frac; out.push(G.rect(x, y + 28, Math.max(2, w), 26, { fill: c, rx: 5 })); if (t) G.label(x + 6, y + 46, t, { size: 11, color: "bg" }); x += w; });
            return out;
          };
          const a = gauge(40, "token budget · 2048", "31 decode tokens first, the rest is prefill", [[31 / 2048, "k"], [2017 / 2048, "v", "2017 prefill tokens"]]);
          const b = gauge(150, "sequences · max_num_seqs = 64", "31 running, 33 slots free", [[31 / 64, "k", "31 running"]]);
          const c = gauge(260, "KV blocks · pool size set by GPU memory (P2.1)", "illustrative fill, not a measurement", [[0.62, "ok", "in use"]]);
          G.text(24, 372, "the step ends when ANY gauge is full", { size: 15, color: "hot" });
          G.from([...a, ...b, ...c], { attr: { width: 0 }, stagger: 0.1, duration: 0.5 });
          G.caption("tokens, sequences and blocks: three independent ceilings");
        } },

      { rail: "the loop", title: "Running first, then admissions",
        body: `<p>The order of decisions matters. <code>Scheduler.step()</code> does two things, in this order:</p>
<div class="eq">budget = max_num_batched_tokens
1. for seq in running (admission order):
     n = min(seq.num_uncomputed, budget)
     make room (preempt if blocks run out)
     schedule (seq, seq.num_computed, n)
     budget -= n
2. if nothing was preempted:
     while budget &gt; 0 and room in max_num_seqs:
       seq = best waiting; n = min(uncomputed, budget)
       if blocks can't hold it: stop
       move to running; schedule; budget -= n</div>
<p>Worked example, budget 8. Running: A (decode, 1 token), B (decode, 1), C (20 prompt tokens left). Waiting: D (5), E (9).</p>
<div class="eq">A: n = min(1, 8) = 1     budget 7
B: n = min(1, 7) = 1     budget 6
C: n = min(20, 6) = 6    budget 0
admission: budget is 0 → D and E wait
batch = A1 + B1 + C6 = 8 tokens; C has 14 left</div>
<p>Running work goes first for two reasons. Those sequences already hold memory, and their users are watching a stream: delaying them raises inter-token latency. And admitting newcomers first would invite <b>thrash</b>: admit, run out of blocks, preempt, admit again.</p>`,
        check: { q: "Why does the scheduler serve running sequences before waiting ones?",
          options: ["Running sequences are cheaper to compute", "They already hold KV memory and a user is watching their stream; starving them raises latency, and admitting first invites admit-then-preempt thrash", "The GPU can only run one new request per step"], answer: 1,
          why: "A running sequence has paid for its blocks and is mid-stream. Skipping it stalls the user and keeps its memory locked for nothing. Putting newcomers first can also consume the blocks the running work needs, forcing preemption of work that was already in progress." },
        scene(G) {
          G.text(24, 34, "budget 8 · running first", { size: 13, weight: 700 });
          const rows = [["A", "decode", 1, "k"], ["B", "decode", 1, "k"], ["C", "20 left", 6, "v"]];
          let used = 0; const bars = [];
          rows.forEach(([n, d, t, c], i) => {
            const y = 56 + i * 52;
            G.box(24, y, 120, 38, `${n} · ${d}`, { stroke: c, color: c, size: 11 });
            G.arrow(148, y + 19, 186, y + 19, { color: c });
            G.label(190, y + 14, `n = min(${i === 2 ? 20 : 1}, ${8 - used}) = ${t}`, { size: 12, color: "ink" });
            G.label(190, y + 32, `budget left: ${8 - used - t}`, { size: 11 });
            for (let k = 0; k < t; k++) bars.push(cell(G, 380 + (used + k) * 28, y + 6, 26, 26, "", c));
            used += t;
          });
          for (let k = 0; k < 8; k++) G.rect(380 + k * 28, 218, 26, 26, { stroke: "line", rx: 3 });
          for (let k = 0; k < 8; k++) cell(G, 380 + k * 28, 218, 26, 26, k === 0 ? "A" : k === 1 ? "B" : "C", k < 2 ? "k" : "v", { size: 11 });
          G.label(380, 262, "the batch: 8 of 8 slots", { size: 11 });
          G.text(24, 292, "admission: budget = 0", { size: 13, weight: 700 });
          const wd = [["D · 5", "v"], ["E · 9", "v"]].map(([t, c], i) => G.box(24 + i * 120, 304, 108, 34, t, { stroke: "muted", color: "muted", dash: "4 3", size: 12 }));
          G.label(268, 326, "wait: not admitted this step", { size: 12, color: "hot" });
          G.label(24, 376, "C keeps 14 uncomputed tokens for the next step", { size: 12, color: "ink" });
          G.from(bars, { opacity: 0, x: -8, stagger: 0.06, duration: 0.2 });
          G.caption("budget is spent in running order; admission gets what's left");
        } },

      { rail: "num_computed", title: "Chunks and the token that does not exist yet",
        body: `<p>Each sequence carries <code>num_computed</code>: how many of its tokens already have K/V in the cache (P6.1). The tokens left are <code>num_uncomputed = len(seq) - num_computed</code>. A chunk is a window onto them:</p>
<div class="eq">Chunk(seq, start = seq.num_computed, num_tokens = n)
produces_token = (start + n == len(seq))</div>
<p>Only a chunk that reaches the <b>end of the known tokens</b> yields logits that matter, because the next token is predicted from the last known position. A chunk that stops inside the prompt only fills the cache; the engine samples nothing for it.</p>
<div class="eq">prompt 400, room for 150 tokens per step
step 1: start 0,   n 150: 150 != 400, no token
step 2: start 150, n 150: 300 != 400, no token
step 3: start 300, n 100: 400 == 400, TOKEN
step 4: len 401, num_computed 400
        -> Chunk(400, 1): a decode</div>
<p>This is the same rule for a whole prompt, a chunk, a decode and a recompute after preemption. Setting <code>chunked_prefill=False</code> only changes one thing: a prompt larger than the budget must wait for a step where it fits whole (it can wait forever if the budget is smaller than the prompt, so this mode is for comparison).</p>`,
        check: { q: "A prefill chunk covers positions 100-163 of a 400-token prompt. Does the engine sample a token after it?",
          options: ["Yes, every forward pass produces a token", "No: 100 + 64 = 164, which is not the end of the sequence (400)", "Only if the chunk is exactly 64 tokens"], answer: 1,
          why: "produces_token is start + n == len(seq). Here 164 ≠ 400, so the logits at position 163 aren't about a next token anyone needs: the next token of the sequence is already known (it is prompt text). The chunk only writes K/V." },
        scene(G) {
          const sx = 560 / 400, x0 = 40;
          G.text(24, 36, "a 400-token prompt, 150-token room per step", { size: 13 });
          const segs = [[0, 150, "step 1", false], [150, 150, "step 2", false], [300, 100, "step 3", true]];
          const bars = segs.map(([s, n, lab, tok], i) => {
            const g = G.box(x0 + s * sx, 60, n * sx - 3, 46, `[${s}:${s + n})`, { fill: tok ? "q" : "v", size: 13, rx: 4 });
            G.label(x0 + s * sx + 4, 124, lab, { size: 11, color: "ink" });
            G.text(x0 + s * sx + 4, 144, tok ? "produces a token" : "no token", { size: 12, color: tok ? "q" : "muted" });
            return g;
          });
          G.line(x0 + 400 * sx, 50, x0 + 400 * sx, 156, { color: "ink", w: 2 });
          G.label(x0 + 400 * sx - 4, 46, "len = 400", { size: 11, anchor: "end", color: "ink" });
          G.text(24, 214, "then the sequence has 401 tokens", { size: 13 });
          const dec = G.box(x0, 232, 560, 30, "", { stroke: "line", rx: 3 });
          G.rect(x0, 232, 400 * sx - 4, 30, { fill: "ok", rx: 3, opacity: 0.55 });
          cell(G, x0 + 400 * sx, 232, 6, 30, "", "q");
          G.label(x0, 284, "K/V in cache: 400 (green)   ·   the new token (violet) is uncomputed", { size: 11 });
          G.text(24, 322, "next step: Chunk(400, 1)", { size: 14, weight: 700, color: "k" });
          G.label(24, 346, "start + n = 401 = len → it produces the next token: a decode", { size: 12, color: "ink" });
          G.label(24, 376, "same rule for prompts, chunks, decodes and recomputes", { size: 12 });
          G.from(bars, { opacity: 0, x: -14, stagger: 0.2, duration: 0.3 });
          G.caption("a chunk speaks only if it reaches the end of the known tokens");
        } },

      { rail: "chunked prefill", title: "Chunked prefill: budget decides how many steps a prompt takes",
        body: `<p>With chunking, the scheduler gives a long prompt whatever the decodes leave: <code>n = min(num_uncomputed, budget)</code>. Decodes come first (running sequences go first), so the prompt gets <code>budget − D</code> tokens per step when <i>D</i> sequences are decoding. The number of steps to the prompt's <b>first token</b> is:</p>
<div class="eq">steps = ceil( L / (budget − D) )

L = 2000, D = 32:
  budget   64: 2000 / 32   = 62.5  → 63
  budget  128: 2000 / 96   = 20.8  → 21
  budget  256: 2000 / 224  =  8.9  →  9
  budget  512: 2000 / 480  =  4.2  →  5
  budget 2048: 2000 / 2016 = 0.99  →  1</div>
<p>These are the numbers <code>bench/budget_sweep.py</code> prints (fake model, so they are exact step counts, not times). In every row the 32 decodes stall for zero steps and no step exceeds its budget. The price is paid by the prompt: more steps to its first token. A small budget smooths ITL for everyone else; a big one gives the long prompt a fast first token. This is the knob <code>--max-num-batched-tokens</code> in vLLM.</p>`,
        check: { q: "Budget 1024, 64 sequences decoding, a new 2,000-token prompt. How many steps until its first token?",
          options: ["2", "3", "4", "1"], answer: 1,
          why: "Each step gives the prompt 1024 − 64 = 960 tokens. Two steps cover 1,920 of 2,000, leaving 80, so a third step reaches the end of the prompt and produces the token: ceil(2000 / 960) = 3." },
        scene(G) {
          const rows = [[64, 63, 64], [128, 21, 128], [256, 9, 256], [512, 5, 512], [2048, 1, 2032]];
          G.text(24, 36, "steps to the long prompt's first token (L = 2000, D = 32)", { size: 13 });
          const bars = [];
          rows.forEach(([b, st, mx], i) => {
            const y = 60 + i * 50;
            G.text(24, y + 20, "budget " + b, { size: 12 });
            bars.push(G.rect(112, y, st * 5.2, 30, { fill: "v", rx: 4 }));
            G.text(112 + st * 5.2 + 8, y + 16, `${st} step${st === 1 ? "" : "s"}`, { size: 13, weight: 700, color: "v" });
            G.label(112 + st * 5.2 + 8, y + 32, `max ${mx} tokens in a step`, { size: 10 });
          });
          G.text(24, 340, "steps = ceil( L / (budget − D) )", { size: 14, color: "ink" });
          G.label(24, 366, "decodes stalled: 0 steps in every row (running first)", { size: 12, color: "k" });
          G.label(24, 388, "source: bench/budget_sweep.py (fake model, step counts)", { size: 11 });
          G.from(bars, { attr: { width: 0 }, stagger: 0.12, duration: 0.5 });
          G.caption("a smaller budget protects decodes and slows the prompt");
        } },

      { rail: "memory", title: "When blocks run out: preemption by recompute",
        body: `<p>Running sequences grow by a token per step, and every <code>block_size</code> tokens each needs a new block. Sooner or later a running sequence needs a block and none is free. Something has to give. The scheduler <b>preempts</b>: it takes a running sequence away, frees all its blocks and puts it back in the waiting queue. Two ways to bring it back:</p>
<table><tr><th></th><th>recompute (ours, vLLM v1 default)</th><th>swap</th></tr>
<tr><td>on preempt</td><td>free blocks, <code>num_computed = 0</code></td><td>copy blocks to CPU memory</td></tr>
<tr><td>on resume</td><td>re-prefill prompt + generated tokens</td><td>copy back</td></tr>
<tr><td>cost</td><td>prefill compute (fast on a GPU)</td><td>PCIe traffic both ways + host memory</td></tr></table>
<p><b>Who is the victim?</b> The lowest effective priority, then the most recently admitted (it has the least work to lose), and <b>never a sequence already placed in this step's batch</b>. If nobody else qualifies, the sequence preempts itself.</p>
<p>The last rule exists because blocks are being handed out while the batch is built. If you free a block that a chunk already in the batch is about to write into, and give it to another sequence, two sequences write the same memory in one forward pass. Recompute loses time, never correctness: the engine's tests check outputs are identical with and without preemption (step 9).</p>`,
        check: { q: "Why can the preemption victim never be a sequence already placed in this step's batch?",
          options: ["Those sequences have the highest priority", "Its blocks would be freed, possibly handed to someone else, while the forward pass is about to write its K/V into them", "It would make the batch empty"], answer: 1,
          why: "Chunks already in the batch will write K/V into their blocks during this step's forward pass. Freeing those blocks and reallocating them would make two sequences write to the same slots. The victim must come from sequences whose blocks nobody is about to touch." },
        scene(G) {
          G.text(24, 34, "pool of 12 blocks, all in use", { size: 13 });
          const own = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"], col = { A: "k", B: "v", C: "q", D: "pink" };
          const bl = own.map((o, i) => cell(G, 24 + i * 50, 48, 44, 44, o, col[o], { size: 13, rx: 5 }));
          G.label(24, 112, "step in progress: A is already in the batch (protected); the loop is at B", { size: 11, color: "ink" });
          G.arrow(24 + 4 * 50 + 22, 126, 24 + 4 * 50 + 22, 156, { color: "v" });
          G.box(20 + 3 * 50 - 30, 160, 250, 34, "B needs a 4th block: none free", { stroke: "v", color: "v", size: 11 });
          G.text(24, 232, "victim = newest admitted, not protected → D", { size: 13, weight: 700, color: "hot" });
          const dv = G.rect(24 + 9 * 50 - 3, 44, 3 * 50 + 2, 52, { stroke: "hot", rx: 6, sw: 2, dash: "4 3" });
          G.label(24, 256, "free D's blocks → B gets one; D goes back to waiting", { size: 12, color: "ink" });
          G.box(24, 276, 110, 32, "D", { stroke: "hot", color: "hot", size: 13 });
          G.label(144, 290, "num_computed = 0, num_preemptions += 1", { size: 12 });
          G.label(144, 306, "later: re-prefill prompt + generated tokens", { size: 12 });
          G.text(24, 352, "never the victim: A (in this batch), B (itself, unless alone)", { size: 12 });
          G.text(24, 376, "no one else left? the sequence preempts itself", { size: 12, color: "muted" });
          G.pulse(dv, { repeat: 6 }); G.from(bl, { opacity: 0, stagger: 0.03, duration: 0.15 });
          G.caption("recompute: lose time, never correctness");
        } },

      { rail: "liveness", title: "Why it cannot thrash forever",
        body: `<p>Preemption invites a bad loop: preempt a sequence, admit another, run out of memory, preempt again, and never finish anything. Four rules in the reference scheduler prevent it:</p>
<ol><li><b>Admission never preempts.</b> Only running work needing memory can trigger a preemption; a newcomer that does not fit just waits.</li>
<li><b>Skip admission in a step that preempted.</b> The freed blocks go to the older work first, not to a new request that would take them straight back.</li>
<li><b>Reject at the door.</b> <code>add_request</code> refuses a request that can never fit: prompt + <code>max_tokens</code> larger than the whole pool (<code>RequestTooLarge</code>, tested with 4 blocks of 4 slots = 16 slots and a 10 + 10 token request).</li>
<li><b>Waiting requests never hold blocks.</b> A request that matched prefix-cache blocks (P6.3) but failed admission releases them, or it could pin memory that a running sequence needs.</li></ol>
<p>The argument: the victim is always the newest admitted, so the <b>oldest</b> running sequence is never evicted while anyone else is running. Each eviction frees memory, and because of rule 3 the oldest fits in the pool alone. So it eventually gets enough blocks, finishes, and frees them all; the next oldest is then the oldest. Exercise 3 asks you to write this in three sentences.</p>`,
        scene(G) {
          G.text(24, 34, "schematic: who gets preempted (A is oldest)", { size: 13 });
          const x0 = 70, sw = 48;
          for (let t = 0; t < 10; t++) G.label(x0 + t * sw + 18, 58, String(t + 1), { size: 10 });
          G.label(24, 58, "step", { size: 10 });
          const rowsD = [["A", "k", [[0, 10, 1]]], ["B", "v", [[0, 4, 1], [4, 7, 0], [7, 10, 1]]], ["C", "q", [[0, 2, 1], [2, 5, 0], [5, 8, 0], [8, 10, 1]]]];
          const cells = [];
          rowsD.forEach(([n, c, segs], i) => {
            const y = 70 + i * 50;
            G.text(24, y + 22, n, { size: 14, weight: 700, color: c });
            segs.forEach(([a, b, run]) => cells.push(run ? G.rect(x0 + a * sw, y, (b - a) * sw - 3, 34, { fill: c, rx: 4 }) : G.rect(x0 + a * sw, y, (b - a) * sw - 3, 34, { stroke: "hot", dash: "4 3", rx: 4 })));
          });
          G.label(x0 + 4 * sw, 116, "preempted, waiting", { size: 11, color: "hot" });
          G.label(x0 + 2 * sw, 166, "preempted, waiting", { size: 11, color: "hot" });
          G.text(24, 232, "four rules", { size: 13, weight: 700 });
          ["admission never preempts running work", "no admission in a step that preempted", "reject requests that can never fit (add_request)", "waiting requests hold no blocks"].forEach((t, i) => G.label(34, 256 + i * 22, `${i + 1} · ${t}`, { size: 12, color: "ink" }));
          G.text(24, 372, "victim = newest → the oldest always finishes", { size: 14, color: "ok" });
          G.from(cells, { opacity: 0, stagger: 0.08, duration: 0.25 });
          G.caption("only the newest are evicted; the oldest keeps its memory");
        } },

      { rail: "priorities", title: "Priorities can starve; aging bounds the wait",
        body: `<p>Some requests matter more (an interactive chat over a batch job). The scheduler admits the waiting request with the highest <b>effective priority</b>, ties broken by arrival. With plain priorities there is a trap: if high-priority requests keep arriving, a low-priority request <b>never</b> runs. <code>test_priorities_starve_without_aging</code> shows it: one priority-0 request, a new priority-1 request every step, <code>max_num_seqs = 1</code>, and the priority-0 request is still waiting after 40 steps.</p>
<p><b>Aging</b> fixes it. A request gains one point of priority per <code>aging_steps</code> steps spent waiting:</p>
<div class="eq">effective = priority + waited_steps // aging_steps

aging_steps = 3, low = 0, fresh arrivals = 1:
  waited 0,1,2 -> effective 0 (loses to a 1)
  waited 3     -> effective 1 (tie: older wins)
  -> the low request runs on step index 3</div>
<p>I ran this workload on the reference scheduler: the low request finishes at step index 3, 5 and 10 for <code>aging_steps</code> 3, 5 and 10. The test only asserts the looser bound <code>2·aging_steps + 1</code>. In general a request of priority <i>p</i> needs <code>aging_steps · (p_max − p)</code> steps of waiting to catch up with the top class; exercise 4 asks you to prove the exact bound. The wait counter resets on admission and on preemption. The cost: aging weakens priority isolation, because a high-priority request can occasionally wait behind an aged low one.</p>`,
        check: { q: "aging_steps = 10, a priority-0 request waits behind a stream of priority-1 arrivals. After roughly how many waiting steps does it win a slot?",
          options: ["Never: priority 1 always beats priority 0", "About 10: waited // 10 reaches 1, ties with a fresh arrival, and the earlier arrival wins", "Immediately, since it arrived first"], answer: 1,
          why: "Effective priority is 0 + waited // 10. At waited = 10 it equals the fresh arrivals' priority 1. The tie goes to the earlier arrival, which is the old request. Running the reference gives step index 10 for aging_steps = 10." },
        scene(G) {
          const x0 = 70, y0 = 80, w = 520, h = 220;
          G.axes(x0, y0, w, h);
          G.label(x0 - 6, y0 - 8, "effective priority", { size: 11 });
          G.label(x0 + w, y0 + h + 30, "steps waited by the low request", { size: 11, anchor: "end" });
          const X = (t) => x0 + (w * t) / 9, Y = (p) => y0 + h - (h * p) / 2.4;
          for (let t = 0; t <= 9; t += 1) G.label(X(t), y0 + h + 16, String(t), { size: 10, anchor: "middle" });
          [0, 1, 2].forEach((p) => G.label(x0 - 12, Y(p) + 4, String(p), { size: 11, anchor: "end" }));
          const hi = G.line(X(0), Y(1), X(9), Y(1), { color: "hot", w: 3 });
          G.label(X(0.2), Y(1) - 8, "fresh priority-1 arrivals: always 1", { size: 11, color: "hot" });
          let d = `M ${X(0)} ${Y(0)}`;
          [[3, 0], [3, 1], [6, 1], [6, 2], [9, 2]].forEach(([t, p]) => (d += ` L ${X(t)} ${Y(p)}`));
          const lo = G.path(d, { color: "k", w: 3 });
          G.label(X(0.2), Y(0) - 8, "low request, aging_steps = 3", { size: 11, color: "k" });
          G.circle(X(3), Y(1), 6, { fill: "ok" });
          G.text(X(3) + 10, Y(1) + 22, "tie: older wins → runs", { size: 12, color: "ok" });
          G.text(24, 372, "without aging the blue line stays at 0 forever", { size: 13, color: "ink" });
          G.from([hi, lo], { opacity: 0, duration: 0.5, stagger: 0.3 });
          G.caption("aging turns waiting time into priority");
        } },

      { rail: "testing", title: "Testing a scheduler with a fake model",
        body: `<p>How do you test something this stateful? The trick is in <code>FakeRunner</code> (<code>model_runner.py</code>): its "logits" are a hash of the token history. The next token therefore depends only on the tokens so far, never on how they were batched. A correct engine must give <b>the same greedy output however it schedules</b>: roomy or tight, chunked or not, with preemption or without.</p>
<p><code>test_outputs_independent_of_scheduling</code> runs the same workloads under six configurations and compares every output with an unbatched reference:</p>
<table><tr><th>config</th><th>what it exercises</th></tr>
<tr><td>512 blocks of 4, budget 4096</td><td>roomy: baseline</td></tr>
<tr><td>budget 7</td><td>chunked prefill</td></tr>
<tr><td>10 blocks of 4, budget 64</td><td>memory-bound: preemption</td></tr>
<tr><td>14 blocks of 2, 3 seqs, budget 5</td><td>everything tight</td></tr>
<tr><td>prefix caching off</td><td>no reuse path</td></tr>
<tr><td>chunked_prefill = False</td><td>whole prompts only</td></tr></table>
<p>It also asserts the invariants: no forward pass exceeds the token budget (<code>max(runner.calls) &lt;= budget</code>), all blocks are free at the end, and block bookkeeping is consistent. A separate test checks that memory pressure really preempts (5 requests of 9 + 20 tokens in 10 blocks of 4 gives <code>preemptions &gt; 0</code>). Without that one, the preemption path could silently never run.</p>
<p>Common mistakes the tests catch: advancing <code>num_computed</code> at schedule time (a failed step then corrupts state; it must advance in <code>update()</code>), sampling for a chunk that doesn't reach the end of the prompt, letting admission preempt, and picking a victim from this step's batch.</p>`,
        scene(G) {
          const cfgs = ["roomy (512 blocks, budget 4096)", "chunked (budget 7)", "memory-bound (10 blocks → preempt)", "tight (14 blocks of 2, 3 seqs)", "no prefix cache", "no chunking"];
          G.text(24, 34, "same prompts, six schedules", { size: 13 });
          const rows = cfgs.map((c, i) => {
            const y = 50 + i * 40;
            G.box(24, y, 360, 30, c, { stroke: "line", size: 12, rx: 6 });
            G.arrow(388, y + 15, 420, y + 15, { color: "muted" });
            return G.box(424, y, 190, 30, "same tokens ✓", { fill: "ok", size: 12, rx: 6 });
          });
          G.text(24, 316, "plus invariants after every run", { size: 13, weight: 700 });
          ["Σ tokens in any step ≤ budget", "every block free at the end", "outputs = unbatched reference"].forEach((t, i) => G.label(34, 340 + i * 22, "· " + t, { size: 12, color: "ink" }));
          G.from(rows, { opacity: 0, x: 14, stagger: 0.1, duration: 0.25 });
          G.caption("scheduling may change speed, never the answer");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A scheduler is called once per step and returns chunks: (sequence, start, n). It knows nothing about the model.</li>
<li>Three limits cap a step: token budget, sequence cap and free KV blocks. Running sequences go first, then admissions with what is left.</li>
<li>A chunk produces a token only if <code>start + n == len(seq)</code>. Chunked prefill gives a prompt <code>budget − D</code> tokens per step: <code>ceil(L / (budget − D))</code> steps to its first token.</li>
<li>When blocks run out, preempt by recompute: newest admitted, never one in this step's batch. Admission never preempts, and no admission in a step that preempted.</li>
<li>Strict priorities starve; aging adds <code>waited // aging_steps</code> to the priority and bounds the wait.</li>
<li>A fake model whose output depends only on the token history lets you prove scheduling never changes results.</li></ul>`,
    sim: {
      title: "Budget, memory and preemption, step by step",
      intro: "A JS port of the reference scheduler's loop (running first, then admission, chunking, preemption by recompute, no prefix cache). 32 decoding requests (prompt 2, 60 output tokens) start first; a long prompt arrives one step later. Each bar is one step: cyan decode tokens, amber new prefill tokens, red recomputed tokens. The green line is free KV blocks. Shrink the pool to see preemption.",
      height: 300,
      controls: [
        { id: "budget", label: "token budget per step", type: "select", value: 512, options: [[64, "64"], [128, "128"], [256, "256"], [512, "512"], [1024, "1024"], [2048, "2048"]] },
        { id: "D", label: "decoding requests (D)", min: 1, max: 32, step: 1, value: 32 },
        { id: "L", label: "long prompt length (tokens)", min: 0, max: 4000, step: 100, value: 2000 },
        { id: "blocks", label: "KV pool, blocks of 16 tokens", min: 16, max: 512, step: 8, value: 400 },
        { id: "chunked", label: "chunked prefill", type: "select", value: 1, options: [[1, "on"], [0, "off"]] },
      ],
      draw(G, v) {
        const r = simulate({ budget: +v.budget, D: v.D, L: v.L, blocks: v.blocks, bs: 16, chunked: !!+v.chunked });
        const n = r.steps.length, x0 = 50, W = 570, H = 170, base = 210;
        G.axes(x0, base - H, W, H);
        G.label(10, 16, "tokens per step (bars) · free blocks (line)", { size: 11 });
        const bw = W / Math.max(n, 1), top = +v.budget * 1.05;
        r.steps.forEach((s, i) => {
          const x = x0 + i * bw, w = Math.max(1, bw - 0.6);
          const hd = (H * s.dec) / top, hp = (H * s.pf) / top, hr = (H * s.rc) / top;
          G.rect(x, base - hd, w, hd, { fill: "k", rx: 0 });
          G.rect(x, base - hd - hp, w, hp, { fill: "v", rx: 0 });
          G.rect(x, base - hd - hp - hr, w, hr, { fill: "hot", rx: 0 });
          if (s.pre.length) G.rect(x, base + 4, w, 6, { fill: "hot", rx: 0 });
        });
        const fp = r.steps.map((s, i) => `${i ? "L" : "M"} ${x0 + i * bw + bw / 2} ${base - (H * s.free) / v.blocks}`).join(" ");
        G.path(fp, { color: "ok", w: 2 });
        G.label(x0, base + 28, "step 1", { size: 10 }); G.label(x0 + W, base + 28, "step " + n, { size: 10, anchor: "end" });
        G.line(x0, base - (H * Math.min(1, 1)), x0 + W, base - H, { color: "line", dash: "3 4" });
        G.label(x0 + W, base - H - 4, "budget " + v.budget, { size: 10, anchor: "end" });
        G.label(10, 270, "red tick under the axis: a step that preempted · green line: free blocks, full height = whole pool", { size: 10 });
        const chips = r.longRejected ? [false, "long prompt rejected at add_request: prompt + max_tokens exceeds the pool"] : [r.preemptions === 0, r.preemptions === 0 ? "no preemption" : r.preemptions + " preemptions"];
        const formula = v.L > 0 && v.budget > v.D ? Math.ceil(v.L / (v.budget - v.D)) : null;
        return [
          { title: "Long prompt", rows: [["steps to its first token", r.ttft === null ? "n/a" : String(r.ttft)], ["ceil(L / (budget − D))", formula === null ? "n/a" : String(formula)], ["decode-stall steps meanwhile", String(r.stalled)]] },
          { title: "Per-step work", rows: [["max tokens in a step", String(r.maxTok)], ["steps to finish everything", String(n)], ["budget respected", r.maxTok <= v.budget ? "yes" : "NO"]],
            chip: [r.maxTok <= v.budget, r.maxTok <= v.budget ? "no step exceeds the budget" : "budget exceeded"] },
          { title: "Memory", rows: [["peak blocks in use", `${r.peak} of ${v.blocks}`], ["preemptions", String(r.preemptions)], ["tokens recomputed", F.num(r.recompute)]], chip: chips,
            html: `<p class="note">With chunked prefill off, a prompt longer than the budget never fits a step and the run ends without it. A pool that cannot hold a request's prompt + output is refused up front.</p>` },
        ];
      },
      note: "A model of the reference scheduler: equal priorities, no prefix cache, one block table per sequence counted in blocks. With the default pool the first panel reproduces bench/budget_sweep.py (63, 21, 9, 5, 1 steps).",
    },
    practice: {
      intro: "Everything here is T0: pure Python with a fake model, no GPU. Implement the scheduler in exercises/my_scheduler.py; the tests fail until you do, and S2S_SOLUTIONS=1 runs them against the reference in platform/engine/v1.",
      items: [
        { title: "FCFS continuous batching", tier: "T0 · medium", goal: "Running sequences first, then admit in arrival order within the token budget, max_num_seqs and free blocks. Outputs must equal the unbatched reference and the budget must never be exceeded.",
          cmd: "uv run pytest course/P6-engine-internals/P6.2-scheduler/exercises -q -k fcfs" },
        { title: "Chunked prefill", tier: "T0 · medium", goal: "Schedule min(num_uncomputed, budget); sample only for chunks that reach the end. A budget of 1 is the extreme case, and running decodes keep getting a token every step.",
          cmd: "uv run pytest course/P6-engine-internals/P6.2-scheduler/exercises -q -k chunked" },
        { title: "Preemption by recompute", tier: "T0 · medium", goal: "Free the victim's blocks, reset num_computed, requeue it. Lowest priority, newest, never one already in this batch; skip admission in a step that preempted. Write the liveness argument.",
          cmd: "uv run pytest course/P6-engine-internals/P6.2-scheduler/exercises -q -k preempt" },
        { title: "Priorities with an anti-starvation bound", tier: "T0 · hard", goal: "effective = priority + waited_steps // aging_steps. Show strict priority starves and aging finishes within 2·aging_steps + 1 steps, then prove the general bound.",
          cmd: "uv run pytest course/P6-engine-internals/P6.2-scheduler/exercises -q -k priority" },
        { title: "The reference and the budget sweep", tier: "T0", goal: "Run the reference scheduler's tests, then the step-count sweep behind step 5.",
          cmd: "uv run pytest platform/engine/v1/tests/test_scheduler.py -q" },
      ],
      labs: [
        { label: "Budget sweep: steps to first token vs token budget (fake model)", path: "course/P6-engine-internals/P6.2-scheduler/bench/budget_sweep.py" },
        { label: "The reference scheduler you are re-implementing", path: "platform/engine/v1/s2s_engine/scheduler.py" },
        { label: "Sequence (num_computed, num_uncomputed)", path: "platform/engine/v1/s2s_engine/sequence.py" },
        { label: "Fake and real runners", path: "platform/engine/v1/s2s_engine/model_runner.py" },
        { label: "Animation: scheduler loop (waiting / running / preempted per step)", path: "animations/p6-scheduler-loop.html" },
        { label: "Quiz", path: "course/P6-engine-internals/P6.2-scheduler/quiz.md" },
        { label: "Read, don't copy: vLLM's scheduler at the pin (re-verify the path)", path: "vllm/v1/core/sched/scheduler.py" },
      ],
    },
  });
})();
