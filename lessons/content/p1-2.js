/* P1.2 — Prefill, decode and the KV cache. The reference lesson: copy its shape for every other lesson. */
(function () {
  const TOK = ["The", "cache", "stores", "keys", "and", "values"];
  // token row helper: n boxes at y, 100px apart
  const toks = (G, n, y = 50, o = {}) => TOK.slice(0, n).map((t, i) => G.box(24 + i * 100, y, 88, 38, t, Object.assign({ stroke: "ink", sw: 1 }, o)));
  // k and v chips under token i
  const kv = (G, i, y, color) => {
    const x = 24 + i * 100;
    return [G.box(x + 6, y, 36, 24, "k", { fill: color || "k", size: 11, rx: 5 }), G.box(x + 46, y, 36, 24, "v", { fill: color || "v", size: 11, rx: 5 })];
  };
  const MODELS = { llama8: { name: "Llama-3-8B", L: 32, kv: 8, hd: 128, P: 8.0 }, llama70: { name: "Llama-3-70B", L: 80, kv: 8, hd: 128, P: 70.6 }, mistral: { name: "Mistral-7B", L: 32, kv: 8, hd: 128, P: 7.2 } };
  const F = S2S.fmt;

  S2S.lesson({
    id: "p1-2", n: "P1.2", title: "Prefill, decode and the KV cache",
    subtitle: "Inference fundamentals · first principles · T0, no GPU needed",
    kicker: "Lesson · ≈ 40 min",
    headline: "The model that refused to read twice",
    intro: `<p>Scroll. The dark panel is a tiny language model. Each step changes it: first how text is generated, then the attention mechanism that makes generation repeat work, then the cache that removes the repetition, and finally the memory bill that decides how many people one GPU can serve.</p>`,
    facts: ["10 steps", "4 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "key"], ["v", "value"], ["q", "query"], ["hot", "recomputed"], ["w", "weights"]],
    prev: "p1-1", next: "p1-3",
    steps: [
      { rail: "the loop", title: "Generation is a loop",
        body: `<p>A language model is a function. Its input is a list of <b>tokens</b>: pieces of words, each mapped to an integer id. Its output is a probability for every token in its vocabulary (often 32,000 to 128,000 of them): how likely each one is to come next.</p>
<p>To generate text, call the function, pick a token from those probabilities, append it, and call the function again on the longer list.</p>
<div class="eq">tokens = prompt
while not finished:
    probs = model(tokens)          # sees ALL tokens so far
    tokens.append(pick(probs))</div>
<div class="analogy"><b>Picture it</b>Writing a sentence one word at a time, where before every new word you re-read the whole sentence from the start.</div>`,
        scene(G) {
          const t = toks(G, 5);
          const q = G.box(524, 50, 88, 38, "?", { stroke: "k", dash: "5 4", color: "k" });
          G.path("M 274 104 C 274 175, 568 175, 568 98", { color: "muted", dash: "4 4", arrow: true });
          G.label(250, 192, "model( all 5 tokens ) → next token", { size: 13 });
          const calls = [0, 1, 2, 3, 4].map((c) => G.box(24 + c * 46, 250, 38, 38, String(c + 1), { fill: "k", size: 13 }));
          G.label(24, 312, "one model call per generated token", { size: 12 });
          G.from(t[4], { opacity: 0, x: -30, duration: 0.6 }); G.from(calls, { opacity: 0, y: 12, stagger: 0.12, duration: 0.3 }); G.pulse(q, { repeat: 7 });
          G.caption("call 5: the input is every token so far");
        } },

      { rail: "inside the model", title: "Inside: a stack of layers",
        body: `<p>Each token id is first turned into a vector of numbers (an <b>embedding</b>, 4,096 numbers for Llama-3-8B). That vector then passes through a stack of identical <b>layers</b>: 32 of them in Llama-3-8B.</p>
<p>Each layer has two parts. An <b>MLP</b> transforms each token's vector on its own. <b>Attention</b> is the only place where a token can use information from <i>other</i> tokens. Everything in this lesson comes from attention.</p>
<p>At the top, the last token's vector is multiplied by one more matrix to get a score for every vocabulary entry. Softmax turns those scores into the probabilities from step 1.</p>`,
        scene(G) {
          const cols = 5;
          for (let i = 0; i < cols; i++) G.box(24 + i * 100, 360, 88, 32, TOK[i], { stroke: "ink", sw: 1, size: 12 });
          const layers = [];
          for (let l = 0; l < 4; l++) {
            const y = 300 - l * 64;
            layers.push(G.rect(16, y - 8, 520, 50, { fill: "w", opacity: 0.18, rx: 8 }));
            G.label(548, y + 10, l === 3 ? "layer 32" : `layer ${l + 1}`, { color: "muted" });
            G.label(548, y + 26, "attn + MLP", { size: 10, color: "muted" });
            for (let i = 0; i < cols; i++) G.rect(44 + i * 100, y, 48, 34, { fill: "blue", opacity: 0.85, rx: 5 });
            if (l < 3) G.label(270, y - 12, "⋮", { color: "muted", size: 12 });
          }
          G.arrow(468, 60, 468, 24, { color: "ok", w: 2 });
          G.label(456, 30, "probabilities for the next token", { color: "ok", anchor: "end" });
          G.from(layers, { opacity: 0, stagger: 0.15, duration: 0.4 });
          G.caption("each column is one token's vector moving up the stack");
        } },

      { rail: "q, k, v", title: "Every token makes a key and a value",
        body: `<p>Inside attention, each token's vector <code>x</code> is multiplied by three learned weight matrices. That gives three new vectors per token:</p>
<div class="eq">k = x · W_K     what I contain          (key)
v = x · W_V     what I hand over        (value)
q = x · W_Q     what I'm looking for    (query)</div>
<p>In Llama-3-8B each key and value has 128 numbers per <b>head</b> (attention runs several small "heads" side by side; more on heads in step 10). The weights <code>W_K, W_V, W_Q</code> are the same for every token; only <code>x</code> changes.</p>
<div class="analogy"><b>Picture it</b>A library: each book has a label on its spine (key) and contents (value). A reader arrives with a question (query) and checks the labels to decide which books to read.</div>`,
        scene(G) {
          toks(G, 5);
          const chips = []; for (let i = 0; i < 5; i++) chips.push(...kv(G, i, 112));
          const q = G.box(450, 160, 36, 24, "q", { fill: "q", size: 11, rx: 5 });
          G.label(24, 250, "x · W_K → k      x · W_V → v      x_new · W_Q → q", { size: 13, color: "ink" });
          G.label(24, 280, "same W matrices for every token · one set per layer", { size: 12 });
          G.from(chips, { opacity: 0, y: -18, stagger: 0.05, duration: 0.35 });
          G.from(q, { opacity: 0, scale: 0.3, transformOrigin: "center", delay: 0.5, duration: 0.4 });
          G.caption("every token gets a key and a value; the newest one also needs a query");
        } },

      { rail: "attention", title: "The new token looks back",
        body: `<p>The newest token's query is compared with every key by a <b>dot product</b> (multiply matching numbers and add them up). A big result means "this token is relevant to me". The scores are divided by √128 to keep them in a sensible range, then <b>softmax</b> turns them into weights that are positive and sum to 1.</p>
<div class="eq">scores  = [2.0, 0.5, 1.0]
exp     = [7.39, 1.65, 2.72]       sum = 11.76
weights = [0.63, 0.14, 0.23]       sum = 1.00
output  = 0.63·v₁ + 0.14·v₂ + 0.23·v₃</div>
<p>The output is a blend of the values, weighted by relevance. It becomes part of the token's vector for the next layer.</p>`,
        scene(G) {
          const W = [0.08, 0.31, 0.12, 0.36, 0.13];
          toks(G, 5); for (let i = 0; i < 5; i++) kv(G, i, 112);
          const qx = 468, qy = 250;
          G.box(qx - 18, qy, 36, 24, "q", { fill: "q", size: 11, rx: 5 });
          const lines = W.map((w, i) => G.line(qx, qy, 48 + i * 100, 138, { color: "q", w: 2 + w * 26, opacity: 0.7 }));
          W.forEach((w, i) => G.text(48 + i * 100, 330, w.toFixed(2), { anchor: "middle", size: 13 }));
          G.label(24, 360, "attention weights (sum = 1) → output = Σ weight · value", { size: 12 });
          G.from(lines, { attr: { "stroke-width": 0 }, stagger: 0.08, duration: 0.5 });
          G.caption("thicker line = more attention");
        } },

      { rail: "causal mask", title: "Tokens only look backwards",
        body: `<p>When a model is trained to predict the next token, it must not peek at the answer. So attention uses a <b>causal mask</b>: token <i>i</i> may attend to tokens 1…<i>i</i>, never to later ones.</p>
<p>This has a consequence that the whole lesson rests on. Token 3's key and value depend only on tokens 1–3. Nothing generated afterwards can change them. <b>Once computed, a past token's keys and values are fixed forever.</b></p>`,
        check: { q: "Token 3's key was computed in an earlier step. Can it change after token 7 is generated?",
          options: ["Yes, attention updates earlier tokens", "No, tokens only look backwards, so it depends only on tokens 1–3"],
          answer: 1, why: "The causal mask means token 3's key and value are computed only from tokens 1–3, in every layer. Later tokens can't affect them." },
        scene(G) {
          const n = 6, x0 = 150, y0 = 70, c = 46;
          for (let i = 0; i < n; i++) { G.label(x0 + i * c + 14, y0 - 10, String(i + 1), { size: 12 }); G.label(x0 - 26, y0 + i * c + 28, String(i + 1), { size: 12 }); }
          const cells = G.grid(x0, y0, n, n, c, c, (r, k) => (k <= r ? "ok" : "line"), { gap: 4, rx: 4 });
          G.label(x0 + 40, y0 - 34, "can attend to (key position) →", { size: 12 });
          G.label(20, y0 + 150, "query", { size: 12 }); G.label(20, y0 + 166, "position", { size: 12 });
          G.from(cells, { opacity: 0, stagger: 0.015, duration: 0.2 });
          G.caption("green = allowed · row 3 only ever sees columns 1–3");
        } },

      { rail: "the waste", title: "Without memory, everything is recomputed",
        body: `<p>Go back to the loop in step 1. A naive implementation forgets everything between calls. Call <i>t</i> sends all <i>t</i> tokens through all 32 layers and rebuilds every key and value, even though the previous call computed exactly the same numbers for the first <i>t − 1</i> tokens.</p>
<p>Count the key/value computations per layer for <i>n</i> generated tokens: 1 + 2 + 3 + … + <i>n</i> ≈ <b>n²/2</b>. Only <b>n</b> of them are new.</p>`,
        check: { q: "With no cache, roughly how many key/value computations per layer does it take to generate 1,000 tokens?",
          options: ["About 1,000", "About 500,000", "About 1,000,000,000"], answer: 1,
          why: "1 + 2 + … + 1000 = 500,500. Only 1,000 of those are new; the rest repeat work." },
        scene(G) {
          toks(G, 5); for (let i = 0; i < 5; i++) kv(G, i, 112, i < 4 ? "hot" : null);
          G.text(24, 190, "call 5 recomputes 4 keys and 4 values it already had", { color: "hot", size: 14 });
          const b = G.bars(24, 400, [1, 2, 3, 4, 5, 6, 7, 8], { w: 30, gap: 14, h: 130, fill: "hot" });
          [1, 2, 3, 4, 5, 6, 7, 8].forEach((c, i) => G.label(39 + i * 44, 418, String(c), { anchor: "middle", size: 11 }));
          G.label(400, 300, "K,V computed per call:", {}); G.label(400, 320, "1, 2, 3 … n  →  ≈ n²/2 total", {});
          G.from(b, { attr: { height: 0, y: 400 }, stagger: 0.06, duration: 0.4 });
          G.caption("work per token grows with the length of the text");
        } },

      { rail: "the cache", title: "Keep them: the KV cache",
        body: `<p>Because past keys and values never change, store them the first time. Every later call then does three things, in every layer:</p>
<ol><li>compute <code>q, k, v</code> for the <b>new token only</b>;</li><li>append its <code>k</code> and <code>v</code> to the stored lists;</li><li>attend with <code>q</code> over everything stored.</li></ol>
<div class="eq">K_cache.append(x_new @ W_K)
V_cache.append(x_new @ W_V)
out = softmax(q @ K_cache.T / √d) @ V_cache</div>
<p>Those stored lists are the <b>KV cache</b>. Queries are never cached: each one is used once, by its own token. In this repository it is the line <code>self.k_cache[l, pos] = k</code> in <code>platform/engine/v0/reference/llama_numpy.py</code>.</p>`,
        scene(G) {
          toks(G, 5);
          G.rect(16, 196, 520, 76, { stroke: "k", dash: "6 4", rx: 10 });
          G.label(26, 190, "KV cache · written once, read every call", { color: "k" });
          const old = []; for (let i = 0; i < 4; i++) { const x = 24 + i * 100; old.push(G.rect(x + 6, 220, 36, 24, { fill: "k", rx: 5 }), G.rect(x + 46, 220, 36, 24, { fill: "v", rx: 5 })); }
          kv(G, 4, 112); G.arrow(468, 140, 468, 214, { color: "ink", dash: "3 3" });
          G.bars(24, 400, [1, 1, 1, 1, 1, 1, 1, 1], { w: 30, gap: 14, h: 16, fill: "k" });
          G.label(400, 372, "K,V computed per call: 1, 1, 1 …", {});
          G.from(old, { y: -100, stagger: 0.04, duration: 0.5 });
          G.caption("one new key/value pair per call; everything else is read");
        } },

      { rail: "prefill", title: "Prefill: the whole prompt at once",
        body: `<p>With a cache, a request runs in two phases. The prompt is known in advance, so all its tokens can go through the model <b>in one pass</b>. That pass computes and stores every prompt token's keys and values, and produces the first output token.</p>
<p>Inside, each weight matrix multiplies a whole block of token vectors at once: a <b>matrix × matrix</b> product. Each weight read from memory is reused for every prompt token, so the GPU spends its time computing. Prefill is <b>compute-bound</b>.</p>
<p>The time until this pass finishes is the <b>time to first token (TTFT)</b>. It grows with the prompt length.</p>`,
        scene(G) {
          toks(G, 6); const ch = []; for (let i = 0; i < 6; i++) ch.push(...kv(G, i, 112));
          G.rect(24, 200, 590, 56, { fill: "w", opacity: 0.55, rx: 8 });
          G.text(36, 233, "weights read once · used by all 6 prompt tokens", { size: 13 });
          G.text(24, 300, "one pass · matrix × matrix · compute-bound", { size: 14 });
          G.rect(24, 324, 300, 26, { fill: "k", rx: 4 });
          G.line(328, 316, 328, 360, { color: "ink", dash: "3 3" }); G.label(336, 342, "← first token: TTFT", { color: "ink" });
          G.from(ch, { opacity: 0, duration: 0.35 });
          G.caption("prefill fills the cache for the whole prompt in a single pass");
        } },

      { rail: "decode", title: "Decode: one token per pass",
        body: `<p>After prefill, every pass produces just one token per request. But it still has to read <b>all</b> of the model's weights from memory (16 GB for Llama-3-8B in bf16), plus that request's whole KV cache.</p>
<p>Count the work. Each weight is used for one multiply and one add: <b>2 FLOPs per weight</b>. In bf16 each weight is 2 bytes. So decode does about <b>1 FLOP per byte read</b>. Every modern GPU can do far more arithmetic per byte than that, so it sits waiting on memory: decode is <b>memory-bound</b>.</p>
<p>The time between tokens is the <b>inter-token latency (ITL)</b>. A lower bound is simply <code>bytes read ÷ memory bandwidth</code>.</p>
<p>Batching changes the ratio: if 32 requests decode in the same pass, each weight read serves 32 tokens.</p>`,
        check: { q: "Why does putting many users in the same decode pass raise throughput?",
          options: ["Each weight read from memory now serves several users' tokens", "It makes each user's KV cache smaller", "It skips the prefill phase"], answer: 0,
          why: "The weights are read once per pass whatever the batch size, so B users get B tokens for roughly the cost of reading the weights once. That raises FLOPs per byte." },
        scene(G) {
          toks(G, 6); for (let i = 0; i < 6; i++) kv(G, i, 112);
          const W = G.rect(24, 200, 590, 56, { fill: "w", opacity: 0.55, rx: 8 });
          G.text(36, 233, "ALL 16 GB of weights read again… for 1 token per request", { size: 13 });
          const tk = []; for (let i = 0; i < 10; i++) tk.push(G.rect(24 + i * 58, 300, 44, 26, { fill: "k", rx: 4 }));
          G.label(24, 350, "each block = one decode pass = one new token · gap = ITL", {});
          G.label(24, 380, "≈ 2 FLOPs per weight ÷ 2 bytes per weight = 1 FLOP per byte", { color: "ink" });
          G.pulse(W, { lo: 0.15, hi: 0.75, d: 0.35 }); G.from(tk, { opacity: 0, stagger: 0.12, duration: 0.2 });
          G.caption("decode moves many bytes for very little math");
        } },

      { rail: "the memory bill", title: "The memory bill",
        body: `<p>The cache stores one key and one value per token, in every layer, for every <b>KV head</b>, each <code>head_dim</code> numbers long:</p>
<div class="eq">bytes per token = 2 × layers × kv_heads × head_dim × bytes per number

Llama-3-8B, bf16:
  2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB per token
  × 8,192 tokens       = 1 GiB per conversation</div>
<p>Llama-3-8B has 32 query heads but only 8 KV heads: groups of 4 query heads share one key/value head. This is <b>grouped-query attention (GQA)</b>, and it makes the cache 4× smaller than giving every query head its own.</p>
<p>On a 24 GB GPU, 16 GB of weights leave room for about seven such conversations. <b>Memory, not compute, limits how many people one GPU can serve.</b> P2.3 and P6.3 show how engines page this memory to waste less of it.</p>`,
        check: { q: "Llama-3-8B uses 8 KV heads for 32 query heads. Compared with 32 KV heads, its cache is…",
          options: ["the same size", "4× smaller", "32× smaller"], answer: 1,
          why: "Cache size is proportional to the number of KV heads: 8 instead of 32 is 4× smaller. That's the point of grouped-query attention." },
        scene(G) {
          const GB = 24, Wt = 16, per = 1.074, sc = 590 / GB;
          G.text(24, 40, "24 GB GPU · Llama-3-8B bf16 · 8k-token conversations", { size: 14 });
          G.rect(24, 64, 590, 58, { stroke: "ink", rx: 8 });
          G.rect(24, 64, Wt * sc, 58, { fill: "w", opacity: 0.75, rx: 8 }); G.text(34, 98, "weights ≈ 16 GB", { size: 13 });
          const u = []; for (let i = 0; i < 8; i++) { const over = Wt + (i + 1) * per > GB; u.push(G.rect(24 + (Wt + i * per) * sc, 64, per * sc - 2, 58, { fill: over ? "hot" : "k", rx: 2, opacity: 0.9 })); }
          G.label(24, 150, "each slice = one conversation's 1 GiB cache (1.07 GB)", { size: 13 });
          G.text(24, 174, "red = no room: the 8th conversation doesn't fit", { color: "hot", size: 13 });
          const f = [["2", "K and V"], ["32", "layers"], ["8", "kv heads"], ["128", "head dim"], ["2", "bytes"]];
          f.forEach(([v, l], i) => { G.box(24 + i * 120, 220, 100, 58, "", { stroke: "line" }); G.text(74 + i * 120, 248, v, { anchor: "middle", size: 20 }); G.label(74 + i * 120, 268, l, { anchor: "middle", size: 11 }); if (i < 4) G.text(132 + i * 120, 254, "×", { anchor: "middle", color: "muted", size: 16 }); });
          G.text(24, 320, "= 131,072 B = 128 KiB per token", { color: "ok", size: 15 });
          G.from(u, { opacity: 0, stagger: 0.25, duration: 0.3 });
          G.caption("memory caps how many users fit on one GPU");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Generation calls the model once per new token, and every call needs every earlier token.</li>
<li>Attention needs each earlier token's key and value. Because tokens only look backwards, those never change, so they can be cached.</li>
<li>Prefill processes the prompt in one compute-bound pass (TTFT). Decode adds one token per memory-bound pass (ITL).</li>
<li>The cache costs <code>2 × layers × kv_heads × head_dim × bytes</code> per token, and that memory limits how many users fit.</li></ul>`,
    sim: {
      title: "Generate text and watch the bill",
      intro: "Pick a model and a workload. The chart compares key/value vectors computed in each decode step (one layer) with and without a cache. Press Run to watch generation unfold. The panels estimate memory and the fastest possible decode speed for the memory bandwidth you choose.",
      controls: [
        { id: "m", label: "model", type: "select", value: "llama8", options: [["llama8", "Llama-3-8B"], ["llama70", "Llama-3-70B"], ["mistral", "Mistral-7B"]] },
        { id: "d", label: "dtype", type: "select", value: 2, options: [[2, "bf16 (2 bytes)"], [1, "fp8 (1 byte)"]] },
        { id: "p", label: "prompt tokens", min: 64, max: 8192, step: 64, value: 1024 },
        { id: "o", label: "output tokens", min: 16, max: 2048, step: 16, value: 512 },
        { id: "u", label: "concurrent users", min: 1, max: 64, value: 8 },
        { id: "bw", label: "memory bandwidth, GB/s (example value: check your GPU's spec sheet)", min: 100, max: 4000, step: 50, value: 1000 },
      ],
      run: { label: "Run generation", frames: 40, ms: 60 },
      draw(G, v, t) {
        const m = MODELS[v.m], B = v.d, P = v.p, O = v.o, U = v.u;
        const N = 40, shown = Math.max(1, Math.round(N * t)), mx = P + O;
        G.label(10, 16, "K,V vectors computed in each decode step (one layer)", { size: 11 });
        G.rect(380, 7, 10, 10, { fill: "hot", rx: 2 }); G.label(394, 16, "no cache", { size: 11 });
        G.rect(470, 7, 10, 10, { fill: "k", rx: 2 }); G.label(484, 16, "with cache", { size: 11 });
        G.axes(16, 30, 610, 190, { xlabel: "decode step →" });
        for (let k = 0; k < shown; k++) {
          const tokensSoFar = P + (k * O) / N, h = (185 * tokensSoFar) / mx;
          G.rect(22 + k * 15, 220 - h, 6, h, { fill: "hot", rx: 1, opacity: 0.8 });
          G.rect(29 + k * 15, 220 - Math.max(2, 185 / mx), 6, Math.max(2, 185 / mx), { fill: "k", rx: 1 });
        }
        const tok = 2 * m.L * m.kv * m.hd * B, seq = tok * (P + O), all = seq * U, w = m.P * 1e9 * B, gpu = 24e9, tot = Math.max(gpu, w + all);
        const noCache = O * P + (O * (O + 1)) / 2;
        const stepBytes = w + all / 2;               // weights + (on average) half of every user's final cache
        const itl = stepBytes / (v.bw * 1e9);
        return [
          { title: "Machine · 24 GB GPU", gauge: [[Math.min(w, tot) / tot, "muted"], [Math.min(all, Math.max(0, gpu - w)) / tot, "k"], [Math.max(0, w + all - gpu) / tot, "hot"]],
            gaugeText: `${m.name}: weights ${(w / 1e9).toFixed(1)} GB · KV ${(all / 1e9).toFixed(1)} GB`, chip: [w + all <= gpu, w + all <= gpu ? "fits on one GPU" : "does not fit"] },
          { title: "KV cache math", rows: [["per token", F.bytes(tok)], ["per user (prompt + output)", F.bytes(seq)], ["all users", F.bytes(all)], ["max users that fit", w < gpu ? F.num(Math.floor((gpu - w) / seq)) : "0"]] },
          { title: "Decode speed limit", rows: [["bytes read per step", F.bytes(stepBytes)], ["ITL lower bound", F.ms(itl)], ["tokens/s per user", F.num(1 / itl, 1)], ["tokens/s, all users", F.num(U / itl, 0)]],
            html: `<p class="note">Assumes each step reads all weights once plus every user's cache (half full on average). Real engines are slower.</p>` },
          { title: "Work for one user (per layer)", rows: [["K,V computed, no cache", F.num(noCache)], ["K,V computed, with cache", F.num(O)], ["recomputation avoided", F.num(noCache / O, 0) + "×"]] },
        ];
      },
    },
    practice: {
      items: [
        { title: "KV size function", tier: "T0 · easy", goal: "Write <code>kv_bytes()</code> and check it against hand-computed cases including GQA, fp8 and many sequences.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/exercises/01-kv-size" },
        { title: "When does decode become compute-bound?", tier: "T0 · medium", goal: "Find the batch size where decode's FLOPs per byte reach the GPU's ridge point, at short and long context.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/exercises/02-decode-crossover" },
        { title: "KV cache correctness", tier: "T0 · medium", goal: "Make your cached forward pass produce exactly the same logits as full recomputation at every position.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/exercises/03-cache-correctness" },
        { title: "Sliding-window KV", tier: "T0 · hard", goal: "Keep only the last W tokens in a ring buffer; prove the memory bound and exactness inside the window.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/exercises/04-sliding-window" },
      ],
      labs: [
        { label: "Examples: no cache vs cache, KV sizer, arithmetic intensity vs batch", path: "course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/examples/" },
        { label: "Benchmark: cache speed-up on the tiny model", path: "course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/bench/cache_bench.py" },
        { label: "The reference implementation", path: "platform/engine/v0/reference/llama_numpy.py" },
      ],
    },
  });
})();
