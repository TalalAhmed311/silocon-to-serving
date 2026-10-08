/* P2.3 — Batching and caching. How an engine keeps the GPU full without making anyone wait: continuous batching,
   chunked prefill, paged KV blocks, block-hash prefix caching, preemption, and the load test that finds the knee. */
(function () {
  const F = S2S.fmt;
  // the mock's cost model (platform/mockllm/server.py defaults; the constants are made up, the shape is the point)
  const BASE = 8, PER = 0.25, PF = 0.05;
  const stepMs = (dec, ptok) => BASE + PER * dec + PF * ptok;

  // a slot-by-step timeline grid: rows = batch slots, cells = (letter, color) or null
  function slotGrid(G, x0, y0, cols, cw, ch, rows, o = {}) {
    const cells = [];
    rows.forEach((row, r) => {
      G.label(x0 - 12, y0 + r * ch + ch / 2 + 4, "slot " + (r + 1), { anchor: "end", size: 11 });
      for (let c = 0; c < cols; c++) {
        const v = row[c];
        if (v) cells.push(G.box(x0 + c * cw, y0 + r * ch, cw - 4, ch - 6, v[0], { fill: v[1], size: 11, rx: 4 }));
        else cells.push(G.rect(x0 + c * cw, y0 + r * ch, cw - 4, ch - 6, { stroke: o.idle || "hot", dash: "3 3", rx: 4 }));
      }
    });
    for (let c = 0; c < cols; c++) if (c % 2 === 0) G.label(x0 + c * cw + (cw - 4) / 2, y0 + rows.length * ch + 10, String(c + 1), { anchor: "middle", size: 10 });
    return cells;
  }
  // run(letter, color, n) → n cells
  const run = (l, col, n) => Array.from({ length: n }, () => [l, col]);
  const idle = (n) => Array.from({ length: n }, () => null);

  /* ---------- the simulator: a JS port of exercises/solutions/batchsim.py ---------- */
  function rng(seed) { let a = seed * 2654435761 + 1013904223; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function trace(seed, n, rate, longEvery) {
    const R = rng(seed + 1), out = []; let t = 0;
    const gap = 1000 / rate;
    for (let i = 0; i < n; i++) {
      t += -Math.log(1 - R()) * gap;
      const p = longEvery && i % longEvery === 0 ? 4000 : [32, 128, 512][Math.floor(R() * 3)];
      out.push({ rid: i, arrival: t, prompt: p, output: [4, 16, 64, 256][Math.floor(R() * 4)] });
    }
    return out;
  }
  function newOut() { return { first: {}, done: {}, tt: {}, steps: [] }; }
  function staticSim(reqs, B) {
    const o = newOut(); let t = 0;
    for (let i = 0; i < reqs.length; i += B) {
      const g = reqs.slice(i, i + B); t = Math.max(t, g[g.length - 1].arrival);
      const pt = g.reduce((s, r) => s + r.prompt, 0);
      let st = BASE + PF * pt + PER * g.length; o.steps.push([t, t + st, 0, pt, B]); t += st;
      const left = {}; g.forEach((r) => { o.first[r.rid] = t; o.tt[r.rid] = [t]; left[r.rid] = r.output - 1; });
      for (;;) {
        const runn = g.filter((r) => left[r.rid] > 0); if (!runn.length) break;
        st = BASE + PER * runn.length; o.steps.push([t, t + st, runn.length, 0, B]); t += st;
        runn.forEach((r) => { o.tt[r.rid].push(t); left[r.rid]--; });
      }
      g.forEach((r) => { o.done[r.rid] = o.tt[r.rid][o.tt[r.rid].length - 1]; });
    }
    return o;
  }
  function contSim(reqs, B, chunk) {
    const o = newOut(), waiting = reqs.slice(), pre = {}, dec = {}; let running = [], t = 0, guard = 0;
    while ((waiting.length || running.length) && guard++ < 200000) {
      if (!running.length && waiting.length && waiting[0].arrival > t) t = waiting[0].arrival;
      while (waiting.length && waiting[0].arrival <= t && running.length < B) { const r = waiting.shift(); running.push(r); pre[r.rid] = r.prompt; }
      const decoding = running.filter((r) => pre[r.rid] === 0), prefilling = running.filter((r) => pre[r.rid] > 0);
      let budget = chunk ? Math.max(0, chunk - decoding.length) : Infinity; const work = [];
      for (const r of prefilling) { const n = Math.min(pre[r.rid], budget); if (n === 0) break; work.push([r, n]); budget -= n; }
      const pt = work.reduce((s, w) => s + w[1], 0), st = stepMs(decoding.length, pt);
      o.steps.push([t, t + st, decoding.length, pt, B]); t += st;
      decoding.forEach((r) => { o.tt[r.rid].push(t); dec[r.rid]--; });
      work.forEach(([r, n]) => { pre[r.rid] -= n; if (pre[r.rid] === 0) { o.first[r.rid] = t; o.tt[r.rid] = [t]; dec[r.rid] = r.output - 1; } });
      running = running.filter((r) => { const fin = pre[r.rid] === 0 && dec[r.rid] === 0; if (fin) o.done[r.rid] = o.tt[r.rid][o.tt[r.rid].length - 1]; return !fin; });
    }
    return o;
  }
  function stats(o, reqs) {
    const e2e = reqs.map((r) => o.done[r.rid] - r.arrival).sort((a, b) => a - b);
    const ttft = reqs.map((r) => o.first[r.rid] - r.arrival).sort((a, b) => a - b);
    let worst = 0; reqs.forEach((r) => { const a = o.tt[r.rid]; for (let i = 1; i < a.length; i++) worst = Math.max(worst, a[i] - a[i - 1]); });
    const p = (a, q) => a[Math.min(a.length - 1, Math.floor(q * a.length))];
    const end = Math.max(...reqs.map((r) => o.done[r.rid])), toks = reqs.reduce((s, r) => s + r.output, 0);
    return { e2eMean: e2e.reduce((s, v) => s + v, 0) / e2e.length, e2e90: p(e2e, 0.9), ttft90: p(ttft, 0.9), worst, end, tps: toks / ((end - reqs[0].arrival) / 1000) };
  }

  S2S.lesson({
    id: "p2-3", n: "P2.3", title: "Batching and caching",
    subtitle: "Serving engines · first principles · T0 simulator and load test, T2 for vLLM",
    kicker: "Lesson · ≈ 45 min",
    headline: "Keep the GPU full, keep nobody waiting",
    intro: `<p>P1.2 showed that one decode pass costs about the same whether it serves one request or thirty-two, so batching is the biggest throughput lever an engine has. This lesson is about <i>how</i> to batch. You will see why the obvious way wastes most of the GPU, how engines re-form the batch every step, why long prompts need to be cut into chunks, how KV memory is split into blocks so it can be shared, and how to find the load at which all of this stops keeping up.</p>`,
    facts: ["10 steps", "4 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "decoding"], ["v", "prefill"], ["hot", "wasted / stalled"], ["ok", "shared"], ["w", "memory"]],
    prev: "p2-2", next: "p2-4",
    steps: [
      { rail: "the lever", title: "Why batching is worth almost everything",
        body: `<p>Recall P1.2: a decode step reads every weight once, whatever the batch size. Adding a sequence to the batch adds only its own KV reads and a little arithmetic. So the cost of one engine step grows slowly with the number of sequences in it.</p>
<p>This course's mock server (<code>platform/mockllm/server.py</code>) uses exactly that shape, with made-up constants:</p>
<div class="eq">step_ms = 8 + 0.25 × decoding + 0.05 × prompt tokens

batch  1:  8.25 ms →   1 token  →   121 tokens/s
batch 16: 12.0 ms  →  16 tokens → 1,333 tokens/s
batch 64: 24.0 ms  →  64 tokens → 2,667 tokens/s</div>
<p>Going from 1 to 64 sequences makes each step 2.9× slower but produces 64× more tokens: 22× the throughput. Every policy in this lesson is a way to keep that batch as full as possible <b>without</b> making individual requests wait too long.</p>`,
        scene(G) {
          const Bs = [1, 2, 4, 8, 16, 32, 64];
          const st = Bs.map((b) => stepMs(b, 0)), tps = Bs.map((b, i) => (1000 * b) / st[i]);
          G.text(24, 36, "mock cost model · made-up constants, real shape", { size: 13 });
          G.label(24, 66, "step time (ms)", { size: 12 });
          const b1 = G.bars(30, 250, st, { w: 30, gap: 10, h: 160, max: 30, fill: "w" });
          st.forEach((v, i) => { G.label(45 + i * 40, 245 - (160 * v) / 30, v.toFixed(1), { anchor: "middle", size: 10, color: "ink" }); G.label(45 + i * 40, 268, String(Bs[i]), { anchor: "middle", size: 11 }); });
          G.label(30, 290, "batch size →", { size: 11 });
          G.label(340, 66, "tokens per second", { size: 12 });
          const b2 = G.bars(340, 250, tps, { w: 30, gap: 10, h: 160, max: 2800, fill: "k" });
          tps.forEach((v, i) => { G.label(355 + i * 40, 245 - (160 * v) / 2800, F.num(v), { anchor: "middle", size: 10, color: "ink" }); G.label(355 + i * 40, 268, String(Bs[i]), { anchor: "middle", size: 11 }); });
          G.label(340, 290, "batch size →", { size: 11 });
          G.text(24, 340, "64× the tokens for 2.9× the step time", { size: 15, color: "ok" });
          G.label(24, 366, "weights are read once per step and shared by every sequence in it", { size: 12 });
          G.from(b1, { attr: { height: 0, y: 250 }, stagger: 0.05, duration: 0.4 });
          G.from(b2, { attr: { height: 0, y: 250 }, stagger: 0.05, duration: 0.4, delay: 0.4 });
          G.caption("step time barely moves; throughput climbs almost linearly");
        } },

      { rail: "static", title: "Static batching: everyone waits for the slowest",
        body: `<p>The obvious design, borrowed from training and image models, is <b>static batching</b>: collect B requests, run them together until <b>all</b> of them finish, then take the next B.</p>
<p>LLM outputs have wildly different lengths. One request wants 3 tokens, another 12. In a static batch, a short request's slot sits <b>idle</b> after it finishes, while the step still runs for the others. Meanwhile new requests wait outside for the whole batch to end.</p>
<div class="eq">batch of 4, outputs 3, 12, 5 and 8 tokens
slot-steps paid for:  4 × 12 = 48
slot-steps used:      3 + 12 + 5 + 8 = 28   (58%)</div>
<p><b>Dynamic batching</b> adds a timeout: dispatch a batch when it is full <i>or</i> when the oldest request has waited too long. That helps at low load, but it is still request-level: once a batch starts, nobody joins and finished slots still idle.</p>
<div class="analogy"><b>Picture it</b>A tour bus that leaves only when every passenger is back on board, while new passengers queue at the stop for the next bus.</div>`,
        scene(G) {
          const rows = [run("A", "k", 3).concat(idle(9)), run("B", "v", 12), run("C", "q", 5).concat(idle(7)), run("D", "ok", 8).concat(idle(4))];
          G.text(24, 36, "static batch of 4 · one column = one engine step", { size: 13 });
          const cells = slotGrid(G, 80, 60, 12, 40, 44, rows);
          G.label(80, 260, "dashed red = slot paid for, nothing to do", { color: "hot", size: 12 });
          G.text(24, 300, "waiting outside:", { size: 13 });
          const q = ["E", "F", "G", "H"].map((l, i) => G.box(150 + i * 48, 284, 40, 28, l, { stroke: "muted", size: 12 }));
          G.label(350, 302, "← start only after step 12", { size: 12, color: "ink" });
          G.text(24, 350, "used 28 of 48 slot-steps (58%)", { size: 15, color: "hot" });
          G.label(24, 374, "dynamic batching changes when a batch starts, not this picture", { size: 12 });
          G.from(cells, { opacity: 0, stagger: 0.008, duration: 0.15 }); G.pulse(q, { repeat: 5 });
          G.caption("short requests leave holes; new requests can't fill them");
        } },

      { rail: "continuous", title: "Continuous batching: re-form the batch every step",
        body: `<p>The fix, introduced by the Orca paper (OSDI '22) as <b>iteration-level scheduling</b>, is to make scheduling decisions every <b>step</b> instead of every batch:</p>
<ol><li>a sequence that emitted its last token leaves <b>immediately</b>;</li><li>a waiting request takes the free slot at the next step (its prompt is prefilled in that step);</li><li>everyone else emits one more token.</li></ol>
<p>The batch is now a moving population rather than a fixed group. A slot only idles when nobody is waiting. vLLM, SGLang and TensorRT-LLM all work this way; it is what "continuous batching" means.</p>
<p>It only works because of the KV cache: each sequence carries its own cache, so sequences at different positions can share a step. Their attention reads different amounts of history, but every weight matrix is applied to all of them at once.</p>
<p>In the course simulator (<code>exercises/batchsim.py</code>), exercise 1 asks you to write <code>continuous()</code>. On the test's seeded traces it beats <code>static()</code> on mean end-to-end latency.</p>`,
        check: { q: "Four requests in a batch want 3, 12, 5 and 8 tokens, and more requests are waiting. Under continuous batching, when does the slot of the 3-token request get reused?",
          options: ["After 12 steps, when the batch ends", "At step 4, right after it emits its last token", "Never: slots are fixed per batch"], answer: 1,
          why: "Continuous batching decides membership every step. The 3-token request leaves after step 3, and a waiting request is admitted (and prefilled) in step 4." },
        scene(G) {
          const rows = [run("A", "k", 3).concat(run("E", "blue", 6), run("H", "pink", 3)), run("B", "v", 12), run("C", "q", 5).concat(run("F", "k", 4), idle(3)), run("D", "ok", 8).concat(run("G", "v", 4))];
          G.text(24, 36, "same 4 slots, same requests, continuous batching", { size: 13 });
          const cells = slotGrid(G, 80, 60, 12, 40, 44, rows);
          [[0, 3], [2, 5], [3, 8], [0, 9]].forEach(([r, c]) => G.rect(80 + c * 40 - 2, 60 + r * 44 - 2, 40, 42, { stroke: "ink", rx: 5, sw: 2 }));
          G.label(80, 260, "outlined = a waiting request joins the moment a slot frees", { color: "ink", size: 12 });
          G.text(24, 300, "queue: E F G H are all admitted inside the same 12 steps", { size: 13 });
          G.text(24, 350, "used 45 of 48 slot-steps (94%)", { size: 15, color: "ok" });
          G.label(24, 374, "the only idle cells: slot 3 after step 9, when nobody is waiting", { size: 12 });
          G.from(cells, { opacity: 0, stagger: 0.008, duration: 0.15 });
          G.caption("the batch is a moving population, re-formed every step");
        } },

      { rail: "prefill stall", title: "A long prompt freezes everyone else",
        body: `<p>Continuous batching admits a new request by prefilling its whole prompt inside one step. For a short prompt that is nothing. For a long one it is a disaster for everyone else in the batch.</p>
<p>Take 31 sequences decoding, and a new request arrives with a 4,000-token prompt. Using the mock's cost model:</p>
<div class="eq">normal step:  8 + 0.25×31            = 15.75 ms
with prefill: 8 + 0.25×31 + 0.05×4000 = 215.75 ms</div>
<p>Every one of the 31 running users waits 216 ms for their next token instead of 16 ms: a 14× spike in their <b>inter-token latency (ITL)</b>, the gap between streamed tokens from P1.3. On a real GPU the shape is the same: prefill is compute-bound and long, while decode steps are short.</p>
<p>Users see this as the stream freezing whenever somebody else pastes a long document.</p>`,
        scene(G) {
          const sx = 1.4, x0 = 30;
          G.text(24, 40, "one decoding user's tokens on a timeline (1.4 px per ms)", { size: 13 });
          const seq = [15.75, 15.75, 15.75, 215.75, 15.75, 15.75, 15.75];
          let x = x0; const blocks = [];
          seq.forEach((d, i) => {
            const big = d > 100;
            blocks.push(G.rect(x, 90, d * sx - 3, 40, { fill: big ? "v" : "k", rx: 4 }));
            if (big) G.text(x + (d * sx) / 2, 115, "4,000-token prefill: 215.75 ms", { anchor: "middle", size: 12, color: "bg" });
            G.line(x + d * sx - 1.5, 80, x + d * sx - 1.5, 140, { color: "ink", w: 1 });
            x += d * sx;
          });
          G.label(x0, 160, "vertical ticks = this user receives a token", { size: 12 });
          G.line(x0 + 3 * 15.75 * sx, 190, x0 + (3 * 15.75 + 215.75) * sx, 190, { color: "hot", w: 2 });
          G.text(x0 + (3 * 15.75 + 108) * sx, 212, "ITL spike: 216 ms instead of 16 ms", { anchor: "middle", color: "hot", size: 13 });
          G.text(24, 280, "the step cost is shared by everyone in the batch", { size: 14 });
          G.label(24, 306, "31 users × 200 ms of extra wait for one newcomer's prompt", { size: 12 });
          G.label(24, 330, "mock: platform/mockllm/server.py · step_ms formula from step 1", { size: 12 });
          G.from(blocks, { opacity: 0, stagger: 0.12, duration: 0.3 });
          G.caption("one long prefill step stalls every running stream");
        } },

      { rail: "chunked prefill", title: "Chunked prefill: cap the work in every step",
        body: `<p>The fix (SARATHI, then Sarathi-Serve) is to give every step a <b>token budget</b>. Each step first gives one token to every decoding sequence, then spends what remains of the budget on prefill. A long prompt is prefilled in <b>chunks</b> across several steps, and decodes ride along with each chunk.</p>
<div class="eq">budget 512 tokens, 31 decoding:
  prefill per step = 512 − 31 = 481 tokens
  step = 8 + 0.25×31 + 0.05×481   = 39.8 ms
  4,000 tokens = 8 chunks of 481 + one of 152
  last step  = 8 + 7.75 + 0.05×152 = 23.35 ms</div>
<p>The worst ITL drops from 215.75 ms to 39.8 ms. The price is paid by the long prompt itself: its first token now arrives after 8 × 39.8 + 23.35 = <b>341.8 ms</b> instead of 215.75 ms. Chunking trades the newcomer's <b>TTFT</b> for everyone else's ITL.</p>
<p>In vLLM V1 the per-step budget is <code>--max-num-batched-tokens</code>: the same knob from P2.1, now with a meaning. Exercise 2 adds <code>chunk_tokens</code> to your simulator, and its test asserts both halves of the trade-off.</p>`,
        check: { q: "You lower the chunk budget from 2,048 to 256 tokens. What happens?",
          options: ["Worst ITL falls and long prompts' TTFT rises", "Both ITL and TTFT fall", "Nothing changes: the total prefill work is the same"], answer: 0,
          why: "A smaller budget caps each step's cost lower, so running streams stall less. But the long prompt is spread over more steps, each paying the fixed step overhead and the decodes, so its first token comes later." },
        scene(G) {
          const sx = 1.0, x0 = 30;
          const row = (y, seq, label) => {
            G.label(x0, y - 10, label, { size: 12, color: "ink" });
            let x = x0; const out = [];
            seq.forEach(([d, kind]) => { out.push(G.rect(x, y, d * sx - 2, 34, { fill: kind, rx: 3 })); x += d * sx; });
            return [out, x];
          };
          const n = [15.75, "k"], plain = [n, n, [215.75, "v"], n, n, n];
          const ch = [n, n].concat(Array.from({ length: 8 }, () => [39.8, "v"]), [[23.35, "v"], n, n]);
          const [a, ax] = row(70, plain, "no chunking");
          const [b, bx] = row(170, ch, "budget 512 tokens per step");
          const t1 = x0 + (2 * 15.75 + 215.75) * sx, t2 = x0 + (2 * 15.75 + 8 * 39.8 + 23.35) * sx;
          G.line(t1, 60, t1, 116, { color: "ok", w: 2 }); G.label(t1 + 4, 128, "TTFT 216 ms", { color: "ok" });
          G.line(t2, 160, t2, 216, { color: "ok", w: 2 }); G.label(t2 - 4, 228, "TTFT 342 ms", { color: "ok", anchor: "end" });
          G.label(x0, 128, "worst ITL 216 ms", { color: "hot" });
          G.label(x0, 228, "worst ITL 40 ms", { color: "k" });
          G.text(24, 290, "amber = steps carrying prefill · cyan = decode-only steps", { size: 13 });
          G.text(24, 320, "every amber chunk still emits a token for all 31 decoders", { size: 13 });
          G.text(24, 362, "smaller chunks: smoother streams, slower first token for big prompts", { size: 13, color: "ok" });
          G.from(b, { opacity: 0, stagger: 0.07, duration: 0.2 });
          G.caption("the same 4,000 prompt tokens, spread across nine bounded steps");
        } },

      { rail: "fragmentation", title: "Reserving the maximum wastes most of the memory",
        body: `<p>Batching is capped by KV memory (P2.1): every running sequence needs its cache on the GPU. How that memory is handed out matters as much as how much there is.</p>
<p>The simple way is to reserve, for each request, one contiguous slab big enough for <code>max_model_len</code> tokens, because you don't know in advance how long it will run. Most requests are far shorter, and the unused part of each slab is locked away from everyone else.</p>
<div class="eq">Llama-3-8B bf16: 128 KiB per token (P1.2)
reserve 8,192 tokens   = 1 GiB per request
request uses 1,200     = 150 MiB
locked but unused      = 6,992 × 128 KiB ≈ 874 MiB</div>
<p>Pre-paging systems lost most of their KV memory to this kind of waste (reserved-but-unused space plus gaps between slabs); the PagedAttention paper (arXiv 2309.06180) measured it. The fix is the one operating systems use for RAM (P0.2): <b>pages</b>.</p>
<p>vLLM splits the KV budget into <b>blocks</b> of 16 tokens and allocates a new block only when a sequence fills its last one. A sequence then wastes at most 15 token slots, on average under 8: about 1 MiB instead of 874 MiB.</p>`,
        scene(G) {
          G.text(24, 36, "contiguous slabs: reserve max_model_len for every request", { size: 13 });
          const used = [1200, 2900, 400, 5300], W = 560;
          const a = [];
          used.forEach((u, i) => {
            const y = 56 + i * 38;
            G.rect(40, y, W, 26, { stroke: "hot", dash: "4 3", rx: 4 });
            a.push(G.rect(40, y, (W * u) / 8192, 26, { fill: "k", rx: 4 }));
            G.label(32, y + 18, "R" + (i + 1), { anchor: "end", size: 11 });
            G.label(46 + (W * u) / 8192, y + 18, F.num(u) + " used", { size: 11, color: "ink" });
          });
          G.label(40, 222, "dashed red: reserved for 8,192 tokens, never used · 9,800 of 32,768 slots used (30%)", { color: "hot", size: 11 });
          G.text(24, 262, "paged: 16-token blocks, allocated as the sequence grows", { size: 13 });
          const blocks = [];
          for (let i = 0; i < 4; i++) {
            const y = 282 + i * 30, nb = Math.ceil(used[i] / 16), shown = Math.min(30, Math.ceil(nb / 24));
            for (let b = 0; b < shown; b++) blocks.push(G.rect(40 + b * 18, y, 15, 20, { fill: "k", rx: 2, opacity: b === shown - 1 ? 0.55 : 0.95 }));
            G.label(46 + shown * 18, y + 15, `${nb} blocks (${nb * 16 - used[i]} slots spare)`, { size: 11, color: "ink" });
          }
          G.label(40, 406, "each square here stands for 24 blocks · spare slots only in the last block", { size: 11 });
          G.from(a, { attr: { width: 0 }, stagger: 0.1, duration: 0.5 });
          G.caption("reserve-the-max locks memory away; blocks waste at most one block");
        } },

      { rail: "block table", title: "The block table: logical blocks, physical blocks",
        body: `<p>Once a sequence's KV is in blocks, those blocks can sit <b>anywhere</b> in GPU memory. Each sequence keeps a <b>block table</b>: entry <i>i</i> says which physical block holds its tokens 16<i>i</i> … 16<i>i</i>+15.</p>
<div class="eq">token position 37 of sequence A
  logical block = 37 // 16 = 2,  offset = 37 % 16 = 5
  block_table[A][2] = 9           (physical block 9)
  KV slot = 9 × 16 + 5 = 149</div>
<p>The attention kernel is written to read keys and values <b>through</b> the block table instead of from one contiguous array. That is <b>PagedAttention</b>. It costs an extra lookup per block and a less regular memory pattern, and it buys three things:</p>
<ul><li>no reservation: a block is allocated only when needed;</li><li>no external fragmentation: any free block fits any sequence;</li><li>sharing: two sequences' tables may point at the <b>same</b> physical block (next step).</li></ul>
<p>This is the allocator you met in P0.3, applied to GPU memory. vLLM's version lives in <code>vllm/v1/core/kv_cache_manager.py</code>; P6.3 has you write one.</p>`,
        scene(G) {
          G.text(24, 36, "block tables (logical → physical)", { size: 13 });
          const A = [7, 2, 9, 4], B = [1, 5, 8];
          const la = A.map((p, i) => G.box(40 + i * 56, 70, 48, 34, "A" + i, { fill: "k", size: 12 }));
          const lb = B.map((p, i) => G.box(400 + i * 56, 70, 48, 34, "B" + i, { fill: "v", size: 12 }));
          G.label(40, 62, "sequence A · table [7, 2, 9, 4]", { size: 11, color: "k" }); G.label(400, 62, "sequence B · table [1, 5, 8]", { size: 11, color: "v" });
          G.text(24, 236, "physical KV blocks in GPU memory", { size: 13 });
          const phys = [];
          for (let j = 0; j < 12; j++) {
            const owner = A.indexOf(j) >= 0 ? "k" : B.indexOf(j) >= 0 ? "v" : null;
            phys.push(G.box(40 + j * 48, 250, 42, 42, String(j), owner ? { fill: owner, size: 12 } : { stroke: "line", size: 12, color: "muted" }));
          }
          const lines = [];
          A.forEach((p, i) => lines.push(G.line(64 + i * 56, 106, 61 + p * 48, 248, { color: "k", w: 1.5, opacity: 0.8 })));
          B.forEach((p, i) => lines.push(G.line(424 + i * 56, 106, 61 + p * 48, 248, { color: "v", w: 1.5, opacity: 0.8 })));
          G.label(40, 316, "grey = free: any sequence can take any free block", { size: 12 });
          G.text(24, 356, "token 37 of A → block 2 → physical 9 → slot 9×16+5 = 149", { size: 13, color: "ink" });
          G.from(lines, { opacity: 0, stagger: 0.08, duration: 0.3 });
          G.caption("contiguous to the sequence, scattered in memory");
        } },

      { rail: "prefix hashing", title: "Prefix caching with block hashes",
        body: `<p>P2.2 showed why a shared prompt prefix has identical KV (the causal mask) and how SGLang finds it with a radix tree. vLLM reaches the same goal with blocks.</p>
<p>Each <b>full</b> block gets a hash of its 16 token ids <b>chained</b> with the hash of the block before it, so the hash identifies the block's tokens <i>and</i> everything in front of them:</p>
<div class="eq">h0 = hash(None, tokens[0:16])
h1 = hash(h0,   tokens[16:32])
h2 = hash(h1,   tokens[32:48])  ...</div>
<p>Before allocating a block for a new request, the engine looks its hash up. On a hit it points the block table at the existing physical block, raises that block's <b>reference count</b>, and skips prefill for those 16 tokens. When a request ends, its blocks' counts drop; a block at count 0 stays cached and is evicted least-recently-used only when memory is needed.</p>
<div class="eq">two prompts share their first 1,000 tokens:
  1000 // 16 = 62 full blocks shared  (992 tokens)
  block 62 holds 8 shared + 8 different tokens → miss
  prefill skipped: 992 × 0.05 ms = 49.6 ms on the mock</div>
<p>When two sequences share a block and one needs to write into it (parallel sampling from one prompt, for example), the engine first copies it: <b>copy-on-write</b>.</p>`,
        check: { q: "Two requests share their first 40 tokens. With 16-token blocks, how many tokens of prefill does the second request skip?",
          options: ["40", "32", "16", "0"], answer: 1,
          why: "Only full blocks are hashed and shared: 40 // 16 = 2 blocks, so 32 tokens. The third block mixes shared and different tokens, so its hash differs and it is computed again." },
        scene(G) {
          G.text(24, 34, "two requests sharing a 1,000-token system prompt", { size: 13 });
          const lab = ["0", "1", "2", "…", "61", "62"];
          const xs = [40, 108, 176, 244, 312, 380];
          G.label(24, 74, "A", { size: 13, color: "ink" }); G.label(24, 244, "B", { size: 13, color: "ink" });
          const a = xs.map((x, i) => G.box(x, 54, 58, 32, lab[i], { fill: i === 5 ? "k" : "ok", size: 12 }));
          const b = xs.map((x, i) => G.box(x, 224, 58, 32, lab[i], { fill: i === 5 ? "v" : "ok", size: 12 }));
          const ph = xs.slice(0, 5).map((x, i) => G.box(x, 139, 58, 32, i === 3 ? "…" : "rc 2", { stroke: "ok", size: 11, color: "ok" }));
          const ln = [];
          xs.slice(0, 5).forEach((x) => { ln.push(G.line(x + 29, 88, x + 29, 137, { color: "ok" })); ln.push(G.line(x + 29, 222, x + 29, 173, { color: "ok" })); });
          G.label(452, 160, "62 physical blocks,", { color: "ok", size: 12 }); G.label(452, 176, "ref count 2 each", { color: "ok", size: 12 });
          G.label(452, 74, "A's own block 62", { color: "k", size: 12 }); G.label(452, 244, "B's own block 62", { color: "v", size: 12 });
          G.text(24, 300, "h_i = hash(h_(i-1), 16 token ids): a chain, so a hit", { size: 13 });
          G.text(24, 322, "means this block AND everything before it match", { size: 13 });
          G.text(24, 362, "992 tokens skip prefill · the partial block never matches", { size: 14, color: "ok" });
          G.from(ln, { opacity: 0, stagger: 0.05, duration: 0.3 });
          G.caption("a cache hit is a block-table pointer plus a ref count");
        } },

      { rail: "preemption", title: "When memory runs out: preemption",
        body: `<p>Continuous batching admits requests while blocks are free. But running sequences keep growing, one token per step, and every 16 tokens each one needs a new block. Eventually a sequence needs a block and none is free.</p>
<p>The scheduler must then <b>preempt</b>: pick a running sequence (vLLM V1 takes the most recently admitted), free all its blocks, and put it back at the front of the waiting queue. When memory frees up, it is re-admitted and its KV is <b>recomputed</b> with a prefill of its prompt plus the tokens it already produced. Nothing is lost except time, but that time is pure waste.</p>
<p>The server reports all of this on its Prometheus <code>/metrics</code> endpoint (P1.3):</p>
<ul><li><code>vllm:kv_cache_usage_perc</code>: fraction of blocks in use;</li><li><code>vllm:num_requests_waiting</code>: the queue outside the batch;</li><li><code>vllm:num_preemptions</code>: a counter of evictions.</li></ul>
<p>When usage sits near 1 and preemptions climb, the limit is KV capacity, not compute. The remedies come from P2.1: a smaller <code>--max-num-seqs</code>, an fp8 KV cache (P2.5), shorter <code>--max-model-len</code>, or more GPU memory.</p>`,
        scene(G) {
          G.text(24, 36, "KV block pool · every sequence grows one token per step", { size: 13 });
          const segs = [["A", "k", 7], ["B", "v", 5], ["C", "q", 6], ["D", "ok", 4], ["E", "blue", 5], ["F", "pink", 5]];
          let x = 40; const out = [];
          segs.forEach(([l, c, n]) => { for (let i = 0; i < n; i++) out.push(G.rect(x + i * 17, 60, 14, 34, { fill: c, rx: 2 })); G.label(x + 2, 110, l, { color: c, size: 12 }); x += n * 17 + 4; });
          G.label(40, 132, "32 of 32 blocks in use · kv_cache_usage_perc = 1.0", { color: "hot", size: 12 });
          G.text(24, 176, "A needs one more block →", { size: 13 });
          G.box(220, 160, 120, 26, "none free", { stroke: "hot", color: "hot", size: 12 });
          G.text(24, 226, "preempt F (newest): free its 5 blocks, requeue it", { size: 13, color: "hot" });
          const fx = 40 + (7 + 5 + 6 + 4 + 5) * 17 + 20;
          const pre = G.rect(fx - 2, 56, 5 * 17 + 2, 42, { stroke: "hot", rx: 4, sw: 2, dash: "4 3" });
          G.text(24, 266, "waiting queue:", { size: 13 });
          G.box(140, 250, 40, 26, "F", { fill: "pink", size: 12 }); G.box(186, 250, 40, 26, "G", { stroke: "muted", size: 12 });
          G.label(236, 268, "F returns later and recomputes its KV", { size: 12, color: "ink" });
          G.text(24, 318, "metrics to watch:", { size: 13 });
          G.label(24, 342, "vllm:kv_cache_usage_perc → 1", { color: "k", size: 12 });
          G.label(24, 362, "vllm:num_requests_waiting grows", { color: "v", size: 12 });
          G.label(24, 382, "vllm:num_preemptions counts up", { color: "hot", size: 12 });
          G.pulse(pre, { repeat: 6 }); G.from(out, { opacity: 0, stagger: 0.02, duration: 0.15 });
          G.caption("a full pool turns growth into evictions and recomputation");
        } },

      { rail: "the knee", title: "Finding the knee with an open-loop load test",
        body: `<p>Put it together. As the request rate rises, the batch fills, then KV or <code>max_num_seqs</code> caps it, then requests queue and TTFT explodes. The <b>knee</b> is the highest load at which your latency target still holds.</p>
<p>Predict it on the mock first (README numbers: 64 max sequences, 128-token mean outputs):</p>
<div class="eq">saturated step = 8 + 0.25 × 64 = 24 ms
tokens/s       = 64 / 0.024     ≈ 2,667
requests/s     = 2,667 / 128    ≈ 21</div>
<p>To find it you must drive the server <b>open-loop</b>: requests arrive on a timetable (Poisson arrivals at a chosen rate) whether or not earlier ones have returned. A closed-loop client ("N users, each sends again when answered") slows down when the server does, so the queue never builds and you never see the knee. P2.4 makes this precise.</p>
<p><code>platform/loadgen</code> is project #4. It sweeps rates, scrapes the metrics from step 9, and reports the knee using the definition in <code>loadgen/analysis.py</code>: the largest offered rate whose p90 TTFT meets the SLO <b>and</b> whose goodput (requests meeting the SLO per second) is at least 90% of the offered rate.</p>`,
        check: { q: "Your load test uses 50 simulated users who each send a new request as soon as the previous one returns. TTFT never explodes, however many users you add. Why?",
          options: ["The server has infinite capacity", "The client is closed-loop: when the server slows, the clients send less, so the queue never grows", "Prefix caching absorbs the load"], answer: 1,
          why: "In a closed loop the offered load is set by the server's own speed. Overload is impossible by construction, so the knee is invisible. Open-loop arrivals keep coming on schedule and expose it." },
        scene(G) {
          const x0 = 70, y0 = 330, w = 520, h = 250, rmax = 32;
          G.axes(x0, y0 - h, w, h, {});
          G.label(x0 + w, y0 + 36, "offered load, requests/s →", { anchor: "end", size: 11 });
          const X = (r) => x0 + (w * r) / rmax, cap = 21;
          const rates = [5, 10, 15, 20, 25, 30];
          rates.forEach((r) => G.label(X(r), y0 + 16, String(r), { anchor: "middle", size: 11 }));
          G.line(X(0), y0, X(rmax), y0 - h * (rmax / 32), { color: "muted", dash: "3 4" });
          G.label(X(26), y0 - h * (26 / 32) - 8, "offered", { size: 11 });
          let d = `M ${X(0)} ${y0}`;
          for (let r = 0; r <= rmax; r += 0.5) d += ` L ${X(r)} ${y0 - (h * Math.min(r, cap - Math.max(0, r - cap) * 0.08)) / 32}`;
          const gp = G.path(d, { color: "ok", w: 3 });
          G.label(X(25), y0 - (h * 20.7) / 32 + 22, "goodput flattens", { color: "ok", size: 12 });
          let d2 = ""; for (let r = 1; r <= 25; r += 0.25) { const v = Math.min(h - 10, 8 + 2.2 / Math.max(0.03, 1 - r / (cap + 1.2))); d2 += `${d2 ? " L" : "M"} ${X(r)} ${y0 - v}`; }
          const tt = G.path(d2, { color: "hot", w: 2.5 });
          G.label(X(13), y0 - 40, "p90 TTFT", { color: "hot", size: 12 });
          G.line(X(20), y0, X(20), y0 - h, { color: "ink", dash: "4 3" });
          G.text(X(20) - 6, y0 - h + 14, "knee ≈ 20–21 req/s", { anchor: "end", size: 13 });
          G.label(x0, 40, "shape predicted by the mock's cost model, not a measurement", { size: 12 });
          G.from([gp, tt], { opacity: 0, duration: 0.6, stagger: 0.3 });
          G.caption("past the knee, extra load only becomes queueing");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A step costs little more for 64 sequences than for one, so the engine's job is to keep the batch full.</li>
<li>Static batching idles finished slots and blocks newcomers. Continuous batching re-forms the batch every step.</li>
<li>Chunked prefill caps each step's work: worst ITL falls, long prompts' TTFT rises. In vLLM the cap is <code>--max-num-batched-tokens</code>.</li>
<li>Paged KV blocks remove reservation waste; block tables let sequences share physical blocks, and chained block hashes find shared prefixes.</li>
<li>When blocks run out, the engine preempts and recomputes. The knee is the last load where the SLO holds; only an open-loop test can find it.</li></ul>`,
    sim: {
      title: "Static vs continuous batching on one trace",
      intro: "A seeded stream of requests (prompts of 32, 128 or 512 tokens, outputs of 4, 16, 64 or 256 tokens, as in test_batchsim.py) runs through the mock cost model under three policies. Each lane shows how full the batch is over time. Raise the rate, add 4,000-token prompts, and change the chunk budget. Press Run to replay time.",
      controls: [
        { id: "rate", label: "arrival rate, requests/s", min: 2, max: 60, value: 12 },
        { id: "B", label: "max batch (slots)", min: 2, max: 64, value: 16 },
        { id: "chunk", label: "chunk budget, tokens per step", type: "select", value: 512, options: [[128, "128"], [256, "256"], [512, "512"], [1024, "1024"], [2048, "2048"]] },
        { id: "long", label: "long 4,000-token prompts", type: "select", value: 25, options: [[0, "none"], [50, "1 in 50 requests"], [25, "1 in 25"], [10, "1 in 10"]] },
        { id: "seed", label: "trace seed", min: 0, max: 9, value: 0 },
      ],
      height: 322,
      run: { label: "Replay time", frames: 50, ms: 50 },
      draw(G, v, t) {
        const reqs = trace(v.seed, 120, v.rate, v.long);
        const lanes = [["static", staticSim(reqs, v.B)], ["continuous", contSim(reqs, v.B, 0)], [`+ chunk ${v.chunk}`, contSim(reqs, v.B, v.chunk)]];
        const S = lanes.map(([n, o]) => stats(o, reqs));
        const T = Math.max(...S.map((s) => s.end)), x0 = 170, W = 460, NB = 230, now = T * t;
        lanes.forEach(([name, o], li) => {
          const y = 20 + li * 88, hh = 62;
          G.text(10, y + 26, name, { size: 12 });
          G.label(10, y + 44, `mean E2E ${F.ms(S[li].e2eMean / 1000)}`, { size: 11 });
          G.rect(x0, y, W, hh, { stroke: "line", rx: 3 });
          const occ = new Float64Array(NB), pf = new Float64Array(NB);
          o.steps.forEach(([a, b, n, pt, B]) => {
            if (a > now) return;
            const i0 = Math.floor((a / T) * NB), i1 = Math.min(NB - 1, Math.floor((Math.min(b, now) / T) * NB));
            for (let i = i0; i <= i1; i++) { const lo = (i * T) / NB, hi = ((i + 1) * T) / NB, ov = Math.max(0, Math.min(b, now, hi) - Math.max(a, lo)); occ[i] += (ov * n) / B; if (pt >= 1000) pf[i] += ov; }
          });
          const bw = W / NB, binT = T / NB;
          for (let i = 0; i < NB; i++) {
            const f = occ[i] / binT;
            if (f > 0) G.rect(x0 + i * bw, y + hh - hh * Math.min(1, f), bw + 0.3, hh * Math.min(1, f), { fill: "k", rx: 0 });
            if (pf[i] > 0) G.rect(x0 + i * bw, y - 6, bw + 0.3, 5, { fill: "v", rx: 0 });
          }
        });
        G.line(x0 + W * t, 12, x0 + W * t, 284, { color: "ink", dash: "2 3", w: 1 });
        G.label(x0, 294, "0", { size: 10 }); G.label(x0 + W, 294, F.ms(T / 1000), { size: 10, anchor: "end" }); G.label(x0 + W / 2, 294, "time →", { size: 10, anchor: "middle" });
        G.label(10, 314, "cyan height = share of slots decoding · amber tick = a step with ≥ 1,000 prefill tokens", { size: 10 });
        const row = (k, f) => S.map((s) => f(s[k]));
        const ms = (x) => F.ms(x / 1000);
        return [
          { title: "Mean end-to-end latency", rows: lanes.map(([n], i) => [n, ms(S[i].e2eMean)]),
            chip: [S[1].e2eMean < S[0].e2eMean, S[1].e2eMean < S[0].e2eMean ? "continuous beats static" : "static wins on this trace"] },
          { title: "p90 TTFT · p90 E2E", rows: lanes.map(([n], i) => [n, `${ms(S[i].ttft90)} · ${ms(S[i].e2e90)}`]) },
          { title: "Worst inter-token gap", rows: lanes.map(([n], i) => [n, ms(S[i].worst)]),
            html: `<p class="note">Chunking bounds this at about 8 + 0.25 × ${v.B} + 0.05 × ${v.chunk} = ${(stepMs(v.B, v.chunk)).toFixed(1)} ms.</p>` },
          { title: "Throughput", rows: lanes.map(([n], i) => [n, F.num(S[i].tps) + " tok/s"]),
            html: `<p class="note">Model: the mock's made-up cost constants (8 ms + 0.25 ms per decoding sequence + 0.05 ms per prefill token), 120 requests. A JS port of exercises/solutions/batchsim.py; real engines add memory limits and preemption.</p>` },
        ];
      },
    },
    practice: {
      intro: "Exercises 1–3 are T0 and run on a laptop. Exercise 4 needs one GPU (aws.md: g6.xlarge, server on 127.0.0.1 with an API key, reached through make forward, torn down with make down). The slow mock test runs only when selected with -m slow.",
      items: [
        { title: "Continuous batching in the simulator", tier: "T0 · medium", goal: "Write continuous() in batchsim.py: admit FIFO every step, finished sequences leave at once. It must beat static() on mean latency and never exceed max_batch.",
          cmd: "uv run pytest course/P2-serving-engines/P2.3-batching-and-caching/exercises/test_batchsim.py -k \"beats_static or invariants\"" },
        { title: "Chunked prefill in the simulator", tier: "T0 · medium", goal: "Add chunk_tokens: decodes first, the rest of the budget on prefill. The test checks the worst ITL is bounded and that long prompts' TTFT goes up.",
          cmd: "uv run pytest course/P2-serving-engines/P2.3-batching-and-caching/exercises/test_batchsim.py -k chunked" },
        { title: "#4 on the mock", tier: "T0 · medium", goal: "Sweep rates against mockllm with platform/loadgen; the knee must land within ±25% of max_num_seqs / (step × mean output). Then add one loadgen feature with a test.",
          cmd: "uv run pytest -m slow course/P2-serving-engines/P2.3-batching-and-caching/exercises/test_loadgen_mock.py" },
        { title: "Prefix caching on vLLM", tier: "T2 · hard", goal: "Find the knee with prefix caching on and off for a 1,000-token shared prefix, record results/p23.json, and explain the TTFT gap with cached tokens × prefill cost.",
          cmd: "uv run python course/P2-serving-engines/P2.3-batching-and-caching/exercises/check_prefix.py results/p23.json" },
      ],
      labs: [
        { label: "Static vs continuous vs chunked on one trace (writes results/batching.png)", path: "course/P2-serving-engines/P2.3-batching-and-caching/examples/01_batching_sim.py" },
        { label: "#4, the load generator (workload, runner, knee analysis, plot)", path: "platform/loadgen/" },
        { label: "The mock server and its cost model", path: "platform/mockllm/server.py" },
        { label: "AWS guide: prefix caching on/off sweep, cost, teardown", path: "course/P2-serving-engines/P2.3-batching-and-caching/aws.md" },
        { label: "Animations: continuous batching, paged attention", path: "animations/p2-continuous-batching.html, animations/p2-paged-attention.html" },
        { label: "vLLM's scheduler and KV manager (read-only; rebuilt in P6.2–P6.3)", path: "vllm/v1/core/sched/scheduler.py, vllm/v1/core/kv_cache_manager.py" },
      ],
    },
  });
})();
