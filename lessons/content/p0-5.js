/* P0.5 — Project: a tiny CPU inference engine (#0 v0). Every P0 idea in one loop: mmap'd weights, threaded SIMD matvecs, a KV cache, sampling, and a bandwidth prediction. */
(function () {
  const F = S2S.fmt;
  // model presets for the simulator. params/embed counts: tiny = platform/engine/v0/tools/make_tiny_llama.py defaults;
  // Llama-3-8B = public config (hidden 4096, 32 layers, 8 KV heads × 128, MLP 14336, vocab 128,256, untied).
  const MODELS = {
    tiny: { name: "tiny test model", params: 119104, embed: 256 * 64, d: 64, L: 2, kv: 32, tied: false },
    ex135: { name: "135M example (README round number)", params: 135e6, embed: 0, d: 0, L: 0, kv: 0, tied: true },
    llama8: { name: "Llama-3-8B", params: 8030261248, embed: 128256 * 4096, d: 4096, L: 32, kv: 1024, tied: false },
  };

  S2S.lesson({
    id: "p0-5", n: "P0.5", title: "Project: a tiny CPU inference engine",
    subtitle: "Systems primer · project #0 v0 · T0, any laptop",
    kicker: "Project lesson · ≈ 55 min",
    headline: "One loop, every idea from P0",
    intro: `<p>This is where the primer pays off. You will read and complete a working language-model engine in C++: <code>platform/engine/v0</code>. Its weights are mapped from a safetensors file without a copy (P0.2), every matrix product is a threaded (P0.3) SIMD (P0.4) matvec over flat row-major arrays (P0.1), and its speed is predicted by the roofline before you measure it.</p>
<p>The lesson walks through one decode step of a Llama-shaped model, piece by piece, at the level you need to implement it: what each operation computes, with small worked numbers, and where it lives in <code>engine.hpp</code> and <code>ops.hpp</code>. P1.1 and P1.2 revisit the same forward pass to count its costs on real models.</p>`,
    facts: ["11 steps", "5 checkpoints", "1 simulator", "5 exercises"],
    legend: [["k", "activations"], ["w", "weights"], ["v", "cache / memory"], ["q", "computation"], ["ok", "output"]],
    prev: "p0-4", next: "p1-1",
    steps: [
      { rail: "the loop", title: "An engine is a loop around one function",
        body: `<p>A language model reads text as <b>tokens</b>: small pieces of words, each an integer id (the test model's vocabulary has 256 ids; Llama-3-8B's has 128,256). The model is a function <code>forward(token, pos)</code> that takes one token at position <code>pos</code> and returns a score, a <b>logit</b>, for every id in the vocabulary: how likely each is to come next.</p>
<div class="eq"># feed the prompt
for pos, tok in enumerate(prompt):
    logits = engine.forward(tok, pos)
# generate
while generating:
    tok = sample(logits)        # pick the next id
    pos += 1
    logits = engine.forward(tok, pos)</div>
<p>That is the whole program (<code>platform/engine/v0/src/main.cpp</code>). Everything else is inside <code>forward</code>, about 40 lines in <code>engine.hpp</code>. The default test model from <code>tools/make_tiny_llama.py</code> is deliberately small:</p>
<table><tr><th></th><th>tiny test model</th><th>Llama-3-8B</th></tr>
<tr><td>hidden size d</td><td>64</td><td>4,096</td></tr>
<tr><td>layers</td><td>2</td><td>32</td></tr>
<tr><td>query / KV heads</td><td>4 / 2</td><td>32 / 8</td></tr>
<tr><td>MLP hidden</td><td>160</td><td>14,336</td></tr>
<tr><td>parameters</td><td>119,104 (0.45 MiB fp32)</td><td>≈ 8.03 billion</td></tr></table>
<p>The same code runs both shapes; only the numbers in <code>config.json</code> change.</p>`,
        scene(G) {
          const b = [[24, "weights file", "w", "safetensors, mmap"], [180, "forward()", "q", "2 layers (test model)"], [336, "logits [256]", "k", "one per id"], [492, "sample", "ok", "next id"]];
          const boxes = b.map(([x, t, c, s], i) => { const g = G.group(); G.box(x, 70, 124, 56, t, { fill: c, size: 12, parent: g }); G.label(x + 62, 146, s, { anchor: "middle", size: 11, parent: g }); return g; });
          for (let i = 0; i < 3; i++) G.arrow(150 + i * 156, 98, 176 + i * 156, 98, { color: "ink", w: 2 });
          const loop = G.path("M 554 160 C 554 220, 242 220, 242 164", { color: "ok", w: 2.5, arrow: true });
          G.label(398, 232, "append the token, pos + 1, call again", { anchor: "middle", size: 12, color: "ok" });
          G.text(24, 290, "platform/engine/v0/src/", { size: 13, color: "muted" });
          [["main.cpp", "the loop above"], ["engine.hpp", "forward(): one decode step"], ["ops.hpp", "rmsnorm, rope, attention, matvec, sampling"], ["safetensors_view.hpp", "the P0.2 loader"]].forEach(([f, d], i) => {
            G.text(40, 318 + i * 24, f, { size: 13, color: "q" }); G.label(250, 318 + i * 24, d, { size: 12 });
          });
          G.from(boxes, { opacity: 0, x: -12, stagger: 0.2, duration: 0.3 }); G.pulse(loop, { repeat: 4 });
          G.caption("one forward call per token; the output picks the next input");
        } },

      { rail: "embedding", title: "A token becomes a vector, and the vector flows up",
        body: `<p>Arithmetic needs numbers, not ids. The first tensor in the file, <code>embed_tokens</code>, is a table with one row of d numbers per vocabulary entry. Token 42's vector is row 42: with P0.1's row-major rule it starts at offset <code>42 × d</code>.</p>
<div class="eq">std::copy(embed_ + token * d,
          embed_ + (token + 1) * d, x_.begin());</div>
<p>That vector <code>x</code> (64 floats in the test model) is the <b>residual stream</b>. Each layer reads it, computes a correction, and <b>adds</b> the correction back:</p>
<div class="eq">x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))</div>
<p>Adding rather than replacing means each layer only has to learn a small change, which is what makes deep stacks trainable. After the last layer, one more RMSNorm and a final matvec with the <code>lm_head</code> matrix [vocab × d] turn x into the logits. Note what is read per token: one row of the embedding table, but <i>all</i> of every other matrix.</p>`,
        scene(G) {
          G.text(24, 36, "embed_tokens [256 × 64]", { size: 13, color: "w" });
          const rows = [];
          for (let r = 0; r < 8; r++) rows.push(G.box(24, 48 + r * 30, 180, 24, r === 4 ? "row 42 → x" : r === 7 ? "…" : `row ${[0, 1, 2, 41, 42, 43, 44][r]}`, { fill: r === 4 ? "k" : "w", size: 11, boxOpacity: r === 4 ? 1 : 0.6 }));
          G.label(24, 304, "offset = 42 × 64 floats", { size: 11, color: "ink" });
          G.arrow(206, 172, 262, 380, { color: "k", w: 2 });
          const stack = [["x (64 floats)", "k", 370], ["+ attention, layer 0", "q", 316], ["+ MLP, layer 0", "q", 262], ["+ attention, layer 1", "q", 208], ["+ MLP, layer 1", "q", 154], ["RMSNorm → lm_head", "w", 100], ["logits [256]", "ok", 46]];
          const st = stack.map(([t, c, y]) => G.box(270, y, 220, 40, t, { fill: c, size: 12 }));
          for (let i = 0; i < 6; i++) G.arrow(380, stack[i][2] - 2, 380, stack[i + 1][2] + 42, { color: "ink" });
          G.label(504, 270, "residual:", { size: 11 }); G.label(504, 286, "each layer", { size: 11 }); G.label(504, 302, "adds to x", { size: 11 });
          G.from(st, { opacity: 0, y: 14, stagger: 0.15, duration: 0.3 });
          G.caption("look up one row, then every layer adds its correction to x");
        } },

      { rail: "RMSNorm", title: "RMSNorm: keep the vector's size in check",
        body: `<p>Adding corrections layer after layer would let the numbers in x drift very large or very small. Before each block, the engine rescales x to a standard size. <b>RMSNorm</b> divides by the root-mean-square of the elements, then multiplies by a learned per-channel weight w:</p>
<div class="eq">y[i] = x[i] / sqrt(mean(x²) + ε) × w[i]

x = [1, 2, 3, 4], w = 1, ε ≈ 0:
mean(x²) = (1 + 4 + 9 + 16) / 4 = 7.5
rms = sqrt(7.5) = 2.739
y = [0.365, 0.730, 1.095, 1.461]</div>
<p>After the division the mean square is exactly 1. The tiny ε (1e-5 here) only prevents a division by zero. Unlike the older LayerNorm, RMSNorm does not subtract the mean: it is cheaper and works as well.</p>
<p>Two implementation details from <code>ops::rmsnorm</code>: compute <code>inv = 1 / sqrt(...)</code> once and multiply, rather than dividing d times; and accumulate Σx² in <code>double</code>, because summing 4,096 floats one by one loses digits (P0.4: float addition rounds at every step). It must also work in place (<code>out == x</code>), because the final norm is called that way. Exercise 1.</p>`,
        scene(G) {
          const x = [1, 2, 3, 4], y = [0.365, 0.730, 1.095, 1.461];
          G.text(24, 36, "x = [1, 2, 3, 4]", { size: 13 }); G.text(340, 36, "y = RMSNorm(x)", { size: 13, color: "ok" });
          G.line(24, 260, 290, 260, { color: "muted" }); G.line(340, 260, 606, 260, { color: "muted" });
          const bx = G.bars(40, 260, x, { w: 44, gap: 20, h: 190, max: 4, fill: "k" });
          const by = G.bars(356, 260, y, { w: 44, gap: 20, h: 190, max: 4, fill: "ok" });
          x.forEach((v, i) => G.label(62 + i * 64, 280, String(v), { anchor: "middle", size: 12, color: "ink" }));
          y.forEach((v, i) => G.label(378 + i * 64, 280, v.toFixed(3), { anchor: "middle", size: 11, color: "ink" }));
          G.text(24, 330, "mean(x²) = 7.5   rms = √7.5 = 2.739", { size: 13 });
          G.text(24, 356, "y = x / 2.739: same direction, mean square 1", { size: 13, color: "ok" });
          G.label(24, 392, "then × w (learned, one per channel)", { size: 12, color: "ink" });
          G.from(by, { attr: { height: 0, y: 260 }, stagger: 0.1, duration: 0.4, delay: 0.3 });
          G.caption("divide by the root mean square: the shape stays, the scale resets");
        } },

      { rail: "matvec", title: "Seven matvecs per layer: where the time goes",
        body: `<p>Almost every step of a layer is a matrix-vector product (P0.4): attention uses four weight matrices (<code>Wq, Wk, Wv, Wo</code>) and the MLP three (<code>Wgate, Wup, Wdown</code>). Add the <code>lm_head</code> at the end. Count Llama-3-8B's weights per layer:</p>
<table><tr><th>matrix</th><th>shape</th><th>weights</th></tr>
<tr><td>Wq, Wo</td><td>4096 × 4096</td><td>16.8 M each</td></tr>
<tr><td>Wk, Wv</td><td>1024 × 4096</td><td>4.2 M each</td></tr>
<tr><td>Wgate, Wup, Wdown</td><td>14336 × 4096</td><td>58.7 M each</td></tr>
<tr><td><b>layer total</b></td><td></td><td><b>218.1 M</b></td></tr></table>
<p>× 32 layers = 6.98 B, plus embedding and lm_head (525 M each) ≈ 8.03 B. The MLP holds 81% of each layer. Every one of those weights is read once per generated token and used in one multiply-add, so matvecs are essentially all of decode's time.</p>
<p>That is why <code>ops::matvec</code> is the only op that is threaded and vectorised:</p>
<div class="eq">pool.parallel_for(out, [&amp;](r0, r1) {   // P0.3
  for r in [r0, r1):                      // my rows
    4 SIMD accumulators over row r · x    // P0.4
    y[r] = hsum(acc) + scalar tail;
});</div>
<p>Rows are split into contiguous chunks across threads, each row is a SIMD dot product with 4 independent accumulators, and each thread writes its own part of y.</p>`,
        check: { q: "In one Llama-3-8B layer, which group of matrices holds most of the weights?",
          options: ["Wq and Wo (attention)", "Wk and Wv (keys and values)", "Wgate, Wup and Wdown (the MLP), about 81%"], answer: 2,
          why: "3 × 58.7 M = 176.2 M of the layer's 218.1 M weights are in the MLP: 81%. Wk and Wv are small because of grouped-query attention (1024 rows instead of 4096)." },
        scene(G) {
          G.text(24, 36, "Llama-3-8B, one layer: 218.1 M weights", { size: 13 });
          const parts = [["q", 16.8, "q"], ["k", 4.2, "v"], ["v", 4.2, "v"], ["o", 16.8, "q"], ["gate", 58.7, "w"], ["up", 58.7, "w"], ["down", 58.7, "w"]];
          let x = 24; const segs = [];
          parts.forEach(([n, m, c]) => { const w = (m / 218.1) * 592; segs.push(G.rect(x, 52, w - 2, 46, { fill: c, rx: 3, opacity: c === "w" ? 0.8 : 1 })); if (w > 30) G.text(x + w / 2, 80, n, { anchor: "middle", size: 12, color: c === "w" ? "ink" : "bg" }); x += w; });
          G.label(24, 118, "attention 42 M (19%)", { size: 11, color: "q" }); G.label(616, 118, "MLP 176 M (81%)", { anchor: "end", size: 11, color: "ink" });
          G.text(24, 168, "ops::matvec: rows split across the thread pool", { size: 13 });
          const cols = ["k", "v", "q", "ok"];
          const rows = [];
          for (let r = 0; r < 8; r++) rows.push(G.box(24, 184 + r * 26, 300, 22, `row ${r}: thread ${Math.floor(r / 2)}`, { fill: cols[Math.floor(r / 2)], size: 11 }));
          G.text(338, 274, "·", { size: 24, color: "ink" });
          G.box(356, 184, 40, 206, "x", { fill: "k", size: 14 });
          G.text(410, 292, "=", { size: 20, color: "ink" });
          for (let r = 0; r < 8; r++) G.rect(436, 184 + r * 26, 40, 22, { fill: cols[Math.floor(r / 2)], rx: 3 });
          G.label(492, 260, "each row:", { size: 11 }); G.label(492, 276, "SIMD dot,", { size: 11 }); G.label(492, 292, "4 accumulators", { size: 11 });
          G.from(rows, { opacity: 0, x: -10, stagger: 0.08, duration: 0.2 });
          G.caption("7 matvecs per layer + lm_head: every weight read once per token");
        } },

      { rail: "attention", title: "Attention: the one place tokens see each other",
        body: `<p>Everything so far treats the current token alone. <b>Attention</b> lets it use earlier tokens. The current position's vector is turned into a <b>query</b> q (what am I looking for?), and every position has a <b>key</b> k (what do I contain?) and a <b>value</b> v (what do I hand over?), all made by the matvecs <code>Wq, Wk, Wv</code>.</p>
<ol><li>Score each earlier position t: <code>s_t = q · k_t / √hd</code> (hd = 16 numbers per head in the test model, 128 in Llama-3-8B).</li>
<li><b>Softmax</b> turns scores into weights that are positive and sum to 1.</li>
<li>Output = Σ weight_t × v_t: a blend of the values, weighted by relevance.</li></ol>
<div class="eq">scores        = [1, 3, 2]          (positions 0, 1, 2)
minus max     = [-2, 0, -1]
exp           = [0.135, 1.000, 0.368]   sum 1.503
weights       = [0.090, 0.665, 0.245]   sum 1.000</div>
<p>Subtracting the maximum first changes nothing mathematically but keeps <code>exp()</code> from overflowing: a float overflows just above exp(88). The current token attends to positions <code>0 … pos</code> <b>inclusive</b>, so it sees itself, and never to later positions (they don't exist yet: this is called <b>causal</b> attention). The work is done per <b>head</b>: d is split into several independent slices of hd numbers (4 heads × 16 in the test model), each with its own scores. Exercise 2 implements <code>ops::attention</code>.</p>`,
        check: { q: "Your attention loop runs over positions 0 … pos − 1. What goes wrong?",
          options: ["Nothing: a token doesn't need to attend to itself", "The current token never sees its own key and value, and at pos = 0 there is nothing to attend to at all", "Softmax overflows"], answer: 1,
          why: "Causal attention covers 0 … pos inclusive. The engine writes the current K and V into the cache just before calling attention precisely so position pos is included; skipping it changes every output and leaves position 0 with an empty sum." },
        scene(G) {
          G.box(24, 60, 100, 44, "q (pos 2)", { fill: "q", size: 12 });
          const sc = [1, 3, 2], wt = [0.090, 0.665, 0.245];
          const lines = [];
          for (let t = 0; t < 3; t++) {
            const y = 40 + t * 70;
            G.box(220, y, 90, 34, `k${t}`, { fill: "v", size: 12 }); G.box(330, y, 90, 34, `v${t}`, { fill: "v", size: 12, boxOpacity: 0.7 });
            lines.push(G.line(126, 82, 218, y + 17, { color: "q", w: 1.5 + wt[t] * 10, opacity: 0.8 }));
            G.label(440, y + 14, `score ${sc[t]}`, { size: 11 }); G.text(440, y + 30, `weight ${wt[t].toFixed(3)}`, { size: 12, color: "ok" });
          }
          G.text(24, 278, "softmax(scores) → weights that sum to 1", { size: 13 });
          const out = G.box(24, 300, 592, 40, "out = 0.090·v0 + 0.665·v1 + 0.245·v2", { fill: "ok", size: 13 });
          G.label(24, 372, "positions 0 … pos inclusive: the token sees itself, never the future", { size: 12, color: "ink" });
          G.from(lines, { attr: { "stroke-width": 0 }, stagger: 0.15, duration: 0.4 }); G.from(out, { opacity: 0, delay: 0.6, duration: 0.3 });
          G.caption("thicker line = higher weight: position 1 matters most here");
        } },

      { rail: "KV cache", title: "The KV cache: compute each key once",
        body: `<p>A key or value depends only on its own position's input, and with causal attention nothing later can change it. So recomputing old keys and values every step would be pure waste: generating t tokens would cost 1 + 2 + … + t ≈ t²/2 key computations instead of t. The engine stores them in a <b>KV cache</b>, one per layer, laid out <code>[max_seq, kv_dim]</code> row-major, and writes the new row directly from the matvec:</p>
<div class="eq">ops::matvec(w.wk, xb, kc + pos * kv, kv, d, pool);
ops::matvec(w.wv, xb, vc + pos * kv, kv, d, pool);</div>
<p>Size per token, fp32: <code>2 (K and V) × layers × kv_dim × 4 B</code>. Test model: 2 × 2 × 32 × 4 = <b>512 B</b>. Llama-3-8B: 2 × 32 × 1024 × 4 = <b>256 KiB</b> per token, so 8,192 tokens need 2 GiB. P1.2 builds the full story of this cache, and P0.3's block allocator is how serving engines manage it.</p>
<p><b>Grouped-query attention (GQA)</b> is why kv_dim is smaller than d: several query heads share one key/value head. With <code>group = n_heads / n_kv_heads</code>, query head h reads KV head <code>h / group</code>. Test model: 4 query heads, 2 KV heads, so heads 0, 1 → KV 0 and heads 2, 3 → KV 1. The classic bug is <code>h % n_kv_heads</code>, which pairs them up differently.</p>`,
        check: { q: "Llama-3-8B: 32 query heads, 8 KV heads. Which KV head does query head 5 read?",
          options: ["KV head 5 (5 % 8)", "KV head 1 (5 / 4)", "KV head 0"], answer: 1,
          why: "group = 32 / 8 = 4 consecutive query heads share a KV head: heads 4–7 use KV head 1, and 5 / 4 = 1 in integer division. 5 % 8 = 5 would read the wrong head's keys and values." },
        scene(G) {
          G.text(24, 36, "K cache of one layer", { size: 13, color: "v" });
          const rows = [];
          for (let r = 0; r < 7; r++) {
            const filled = r < 4, cur = r === 4;
            rows.push(G.box(24, 50 + r * 30, 260, 26, cur ? "pos 4: written now" : filled ? `pos ${r}: stored` : `pos ${r}: empty`, { fill: cur ? "ok" : filled ? "v" : undefined, stroke: filled || cur ? undefined : "line", color: filled || cur ? "bg" : "muted", size: 11 }));
          }
          G.label(24, 278, "kc + pos × kv_dim: P0.1's row-major offset", { size: 11, color: "ink" });
          G.text(330, 36, "GQA: 4 query heads, 2 KV heads", { size: 13 });
          const qh = [0, 1, 2, 3].map((h) => G.box(330 + h * 70, 60, 60, 32, `q${h}`, { fill: "q", size: 12 }));
          [0, 1].forEach((k) => G.box(360 + k * 140, 180, 80, 32, `kv${k}`, { fill: "v", size: 12 }));
          [0, 1, 2, 3].forEach((h) => G.arrow(360 + h * 70, 94, 400 + Math.floor(h / 2) * 140, 176, { color: "q", w: 1.5 }));
          G.label(330, 236, "head h → KV head h / 2", { size: 12, color: "ok" });
          G.label(330, 256, "not h % 2 (that pairs 0 with 2)", { size: 11, color: "hot" });
          G.text(24, 330, "per token, fp32: 2 × layers × kv_dim × 4 B", { size: 13 });
          G.label(24, 356, "test model: 2 × 2 × 32 × 4 = 512 B", { size: 12, color: "ink" });
          G.label(24, 376, "Llama-3-8B: 2 × 32 × 1024 × 4 = 256 KiB", { size: 12, color: "ink" });
          G.from(rows[4], { opacity: 0, x: -30, duration: 0.5 });
          G.caption("each step appends one row and reads all stored rows");
        } },

      { rail: "RoPE", title: "RoPE: position as a rotation",
        body: `<p>Attention as described has no idea of order: shuffling the earlier tokens would give the same blend. The model needs to know <i>where</i> each token is. <b>RoPE</b> (rotary position embedding) encodes position by rotating q and k before the dot product.</p>
<p>Split a head's hd numbers into pairs. Pair i is treated as a point in a plane and rotated by an angle that grows with position:</p>
<div class="eq">angle_i = pos × θ^(−2i / hd),  θ = 10000
x[i]      ← x[i]·cos − x[i+hd/2]·sin
x[i+hd/2] ← x[i+hd/2]·cos + x[i]·sin</div>
<p>Pair 0 turns fast (1 radian per position); later pairs turn ever more slowly. With hd = 16, pair 7 turns by 10000^(−14/16) ≈ 0.00032 radians per position. Rotating both q (at its position) and k (at its position) means their dot product depends only on the <b>difference</b> of positions: "3 tokens ago" means the same thing anywhere in the text.</p>
<p>Worked example: the pair (1, 0) at pos = 1, pair 0: angle 1 rad gives (cos 1, sin 1) = (0.540, 0.841). At pos = 0 every angle is 0, so RoPE does nothing. That is why a RoPE bug shows up as "position 0 is right, everything after is garbage".</p>
<p>Two conventions pair the numbers differently: <b>rotate_half</b> pairs i with i + hd/2 (Hugging Face weights, and this engine); <b>interleaved</b> pairs (0, 1), (2, 3), … (the original paper, llama2.c). Rotate q and k only, never v. Exercise 1.</p>`,
        scene(G) {
          const dial = (cx, cy, r, step, title, sub) => {
            G.circle(cx, cy, r, { stroke: "line", sw: 1.5 });
            G.text(cx, cy - r - 30, title, { anchor: "middle", size: 13 }); G.label(cx, cy - r - 12, sub, { anchor: "middle", size: 11 });
            const hands = [];
            for (let p = 0; p < 4; p++) {
              const a = -p * step;
              hands.push(G.line(cx, cy, cx + r * Math.cos(a), cy + r * Math.sin(a), { color: ["k", "q", "v", "ok"][p], w: 3, arrow: true }));
              if (step > 0.1) G.label(cx + (r + 16) * Math.cos(a), cy + (r + 16) * Math.sin(a) + 4, `pos ${p}`, { anchor: "middle", size: 10, color: "ink" });
            }
            return hands;
          };
          const h1 = dial(160, 210, 100, 1, "pair 0: 1 rad per position", "fast");
          const h2 = dial(470, 210, 100, 0.00032, "pair 7 (hd 16): 0.00032 rad", "slow: pos 0–3 overlap");
          G.text(24, 376, "q·k after rotation depends only on (pos_q − pos_k)", { size: 13, color: "ok" });
          G.label(24, 400, "pos 0 = no rotation: a RoPE bug hides at position 0", { size: 12, color: "ink" });
          G.from(h1, { rotation: 0, opacity: 0, stagger: 0.25, duration: 0.3 });
          G.caption("each pair of numbers is a clock hand turning at its own speed");
        } },

      { rail: "SwiGLU MLP", title: "The MLP: widen, gate, narrow",
        body: `<p>After attention mixes information between positions, the <b>MLP</b> transforms each position's vector on its own. Llama's version is <b>SwiGLU</b>: two matvecs widen x from d to a larger hidden size, one of them passes through a smooth gate, they are multiplied element by element, and a third matvec narrows back to d:</p>
<div class="eq">g = Wgate · h          // d → hidden
u = Wup · h            // d → hidden
m = SiLU(g) ⊙ u        // element-wise
x = x + Wdown · m      // hidden → d, residual</div>
<p><b>SiLU</b>(z) = z / (1 + e^(−z)): close to z for large positive z, close to 0 for large negative z, smooth in between. SiLU(2) = 2 / 1.135 = 1.762; SiLU(−2) = −2 / 8.389 = −0.238. The gate decides, per hidden unit, how much of u passes.</p>
<p>Sizes: test model 64 → 160 → 64; Llama-3-8B 4,096 → 14,336 → 4,096. The widening is where the MLP's 81% of the weights comes from (step 4). After the last layer: <code>x = RMSNorm(x)</code>, then <code>logits = lm_head · x</code>, a [vocab × d] matvec: 256 × 64 in the test model, 128,256 × 4,096 = 525 M weights in Llama-3-8B.</p>`,
        scene(G) {
          G.box(24, 150, 70, 40, "h (64)", { fill: "k", size: 12 });
          G.box(150, 70, 150, 40, "Wgate · h → 160", { fill: "w", size: 11 });
          G.box(150, 230, 150, 40, "Wup · h → 160", { fill: "w", size: 11 });
          G.arrow(96, 162, 146, 96, { color: "ink" }); G.arrow(96, 178, 146, 244, { color: "ink" });
          G.box(330, 70, 110, 40, "SiLU", { fill: "q", size: 12 });
          G.arrow(302, 90, 326, 90, { color: "ink" });
          G.circle(470, 170, 18, { fill: "q" }); G.text(470, 176, "⊙", { anchor: "middle", size: 16, color: "bg" });
          G.arrow(442, 96, 458, 154, { color: "ink" }); G.arrow(302, 250, 454, 180, { color: "ink" });
          G.box(510, 150, 106, 40, "Wdown → 64", { fill: "w", size: 11 });
          G.arrow(490, 170, 506, 170, { color: "ink" });
          G.label(522, 212, "+ x (residual)", { size: 11, color: "ok" });
          // SiLU curve
          const X = (z) => 40 + (z + 4) * 36, Y = (s) => 390 - s * 30;
          G.line(40, 390, 300, 390, { color: "muted", w: 1 }); G.line(X(0), 296, X(0), 408, { color: "muted", w: 1 });
          let d = ""; for (let i = 0; i <= 70; i++) { const z = -4 + i * 0.1, s = z / (1 + Math.exp(-z)); d += (i ? " L " : "M ") + X(z).toFixed(1) + " " + Y(s).toFixed(1); }
          const curve = G.path(d, { color: "q", w: 2.5 });
          G.circle(X(2), Y(1.762), 4, { fill: "ok" }); G.label(X(2) + 8, Y(1.762) + 16, "SiLU(2) = 1.762", { size: 10, color: "ink" });
          G.circle(X(-2), Y(-0.238), 4, { fill: "hot" }); G.label(X(-2), Y(-0.238) - 12, "SiLU(−2) = −0.238", { size: 10, color: "ink", anchor: "middle" });
          G.label(400, 330, "SiLU(z) = z / (1 + e^(−z))", { size: 12, color: "q" });
          G.label(400, 352, "test model: 64 → 160 → 64", { size: 11, color: "ink" });
          G.label(400, 370, "Llama-3-8B: 4096 → 14336 → 4096", { size: 11, color: "ink" });
          G.from(curve, { opacity: 0, duration: 0.6 });
          G.caption("two wide matvecs, a smooth gate, one narrow matvec back");
        } },

      { rail: "sampling", title: "Sampling: from logits to the next token",
        body: `<p>The logits are scores, not choices. <b>Greedy</b> decoding takes the highest (argmax): deterministic, and often repetitive. To sample instead, turn logits into probabilities with softmax, after dividing by a <b>temperature</b> T: T &lt; 1 sharpens the distribution, T &gt; 1 flattens it, and T = 0 is defined as greedy (it would divide by zero).</p>
<p><b>Top-p</b> (nucleus) sampling then cuts off the unlikely tail: sort the probabilities, keep the smallest prefix whose total reaches p, renormalise, and draw.</p>
<div class="eq">p = [0.50, 0.30, 0.15, 0.05], top_p = 0.79
cumulative: 0.50, 0.80 ≥ 0.79
  → keep 2 tokens, kept mass M = 0.80
renormalised: 0.50/0.80 = 0.625, 0.30/0.80 = 0.375

draw u = 0.7: target u × M = 0.56
0.50 &lt; 0.56, then 0.80 &gt; 0.56 → token 1</div>
<p>The random number u is passed <i>in</i> rather than drawn inside the function, so a test can feed fixed values and check exact answers; exercise 4's χ² test draws 10⁵ seeded values and checks that the frequencies match {0.625, 0.375, 0, 0}. Ties in the sort are broken by token id so the result never depends on <code>std::sort</code>'s internals.</p>`,
        check: { q: "Probabilities {0.6, 0.25, 0.1, 0.05} and top_p = 0.9. Which tokens can be sampled, with what probabilities?",
          options: ["Only the first, probability 1", "The first three: 0.632, 0.263, 0.105", "All four, unchanged"], answer: 1,
          why: "Cumulative mass is 0.6, 0.85, then 0.95 ≥ 0.9, so three tokens are kept with total 0.95. Renormalised: 0.6/0.95 = 0.632, 0.25/0.95 = 0.263, 0.1/0.95 = 0.105." },
        scene(G) {
          const p = [0.5, 0.3, 0.15, 0.05];
          G.text(24, 36, "sorted probabilities, top_p = 0.79", { size: 13 });
          G.line(24, 220, 300, 220, { color: "muted" });
          const b1 = G.bars(40, 220, p, { w: 50, gap: 16, h: 160, max: 0.7, fill: (i) => (i < 2 ? "k" : "hot") });
          p.forEach((v, i) => G.label(65 + i * 66, 238, v.toFixed(2), { anchor: "middle", size: 11, color: "ink" }));
          G.label(40, 260, "cumulative 0.50, 0.80 ≥ 0.79: keep 2", { size: 11, color: "ink" });
          G.text(340, 36, "renormalised nucleus", { size: 13, color: "ok" });
          G.line(340, 220, 616, 220, { color: "muted" });
          const b2 = G.bars(356, 220, [0.625, 0.375, 0, 0], { w: 50, gap: 16, h: 160, max: 0.7, fill: "ok" });
          [0.625, 0.375, 0, 0].forEach((v, i) => G.label(381 + i * 66, 238, v ? v.toFixed(3) : "0", { anchor: "middle", size: 11, color: "ink" }));
          G.text(24, 306, "u = 0.7 → target 0.7 × 0.80 = 0.56", { size: 13 });
          G.rect(24, 322, 400, 26, { fill: "k", rx: 3, opacity: 0.85 }); G.rect(424, 322, 240 * 0.8 - 2, 26, { fill: "v", rx: 3 });
          G.label(30, 340, "token 0: 0 … 0.50", { size: 11, color: "bg" }); G.label(484, 340, "token 1: … 0.80", { size: 11, color: "bg" });
          G.line(24 + 0.56 * 800, 314, 24 + 0.56 * 800, 356, { color: "hot", w: 3 });
          G.label(24 + 0.56 * 800, 374, "0.56 → token 1", { anchor: "middle", size: 12, color: "hot" });
          G.from(b2, { attr: { height: 0, y: 220 }, stagger: 0.1, duration: 0.4, delay: 0.3 });
          G.caption("keep the smallest set reaching p, renormalise, invert the CDF at u");
        } },

      { rail: "loader · threads", title: "Putting the systems pieces together",
        body: `<p>Now look at the engine as a systems program rather than as maths.</p>
<p><b>Loading (P0.2).</b> The constructor maps <code>model.safetensors</code> and asks for each tensor by name. <code>tensor()</code> checks the dtype (v0 needs F32), the expected shape and the alignment, adds the size to <code>weight_bytes_</code>, and returns a pointer <i>into the mapped file</i>. No weight is copied. The first <code>forward</code> call touches every page and takes the page faults; later calls find them mapped. Only activations (a few vectors of d or hidden floats) and the KV cache are allocated.</p>
<p><b>Threads (P0.3).</b> Each matvec is one <code>parallel_for</code>: wake the workers, split rows, wait for all of them. Per token that is 7 per layer plus the lm_head: 2 × 7 + 1 = 15 calls for the test model, 32 × 7 + 1 = 225 for Llama-3-8B. For a 4096 × 4096 matrix each worker has thousands of rows; for the test model's 64-row matrices each worker gets a handful, and the fork/join overhead can cost more than the arithmetic. <code>perf record -g</code> shows it.</p>
<p><b>Prefill.</b> v0 feeds the prompt one token at a time through the same <code>forward</code>: simple and correct, but it reads every weight once per prompt token. Processing the whole prompt as one matrix-matrix product (P0.4's SGEMM) is how later engines make it fast (P1.2, P6).</p>
<p><b>Testing.</b> The tests compare <i>teacher-forced logits</i> (feed the same 12 fixed ids to both engines, compare every logit within <code>rtol = atol = 1e-4</code>) as well as 48 greedy tokens against <code>reference/llama_numpy.py</code>. Logits, because a tiny rounding difference can flip an argmax at a near-tie, after which the generated tokens legitimately diverge. Exercise 3.</p>`,
        scene(G) {
          G.text(24, 36, "memory of the running engine", { size: 13 });
          G.rect(24, 50, 380, 60, { fill: "w", opacity: 0.6, rx: 8 });
          G.text(36, 74, "mapped file: model.safetensors", { size: 12 });
          G.label(36, 96, "weights stay in the page cache (P0.2)", { size: 11, color: "ink" });
          const ptrs = [];
          [["embed_", 60], ["wq…w_down", 170], ["lm_head_", 300]].forEach(([n, x]) => { ptrs.push(G.arrow(x + 30, 150, x + 30, 114, { color: "w", w: 2 })); G.label(x, 166, n, { size: 11, color: "ink" }); });
          G.box(430, 50, 186, 26, "activations: few KiB", { fill: "k", size: 11 });
          G.box(430, 84, 186, 26, "KV cache: allocated", { fill: "v", size: 11 });
          G.text(24, 216, "parallel_for calls per generated token", { size: 13 });
          const bars = [[15, "test model: 2 × 7 + 1 = 15"], [225, "Llama-3-8B: 32 × 7 + 1 = 225"]].map(([n, t], i) => { G.label(24, 248 + i * 46, t, { size: 12, color: "ink" }); return G.rect(300, 234 + i * 46, (n / 225) * 316, 24, { fill: "q", rx: 3 }); });
          G.text(24, 352, "tiny matrices: fork/join can outweigh the math", { size: 13, color: "hot" });
          G.text(24, 378, "tests: teacher-forced logits + greedy tokens vs NumPy", { size: 13, color: "ok" });
          G.from(ptrs, { opacity: 0, stagger: 0.2, duration: 0.3 }); G.from(bars, { attr: { width: 0 }, stagger: 0.3, duration: 0.5 });
          G.caption("zero-copy weights, small allocations, one parallel_for per matvec");
        } },

      { rail: "predict", title: "Predict the speed, then measure it",
        body: `<p>P0.4's roofline makes a prediction before you run anything. Each decode step reads every weight once and does one multiply-add with it: 2 FLOPs per 4 bytes in fp32, I = 0.5, far left of any CPU's ridge. So the step is memory-bound:</p>
<div class="eq">decode tokens/s ≲ bandwidth B
                  ÷ weight bytes read per token</div>
<p>The module's worked example, with <b>made-up round numbers</b> to replace by yours: B = 40 GB/s and a 135M-parameter model in fp32 (135e6 × 4 B = 0.54 GB) give 40 / 0.54 ≈ <b>74 tokens/s</b> at most. Use your measured STREAM triad bandwidth from P0.4 and the "weights … MiB" line the engine prints at start-up. <code>bench/engine_bench.py</code> prints the measured tokens/s next to this ceiling.</p>
<p>Expect to land below it, because the formula ignores:</p>
<ul><li>fork/join overhead per matvec (step 10);</li>
<li>KV-cache reads, which grow with position (every step reads all stored keys and values);</li>
<li>the non-matvec ops (norms, RoPE, softmax).</li></ul>
<p>Strictly, only one row of the embedding table is read per token. For Llama-3-8B that leaves about 7.5 B of its 8.03 B weights read per step. And the test model (0.45 MiB) fits entirely in the CPU's caches, so it is not DRAM-bound at all: its limit is cache bandwidth and threading overhead. The roofline's B depends on where the data lives. Exercise 5 stores weights in int8 blocks (36 B per 32 weights instead of 128), which should lift the ceiling about 3.6× for a model too big for the caches.</p>`,
        check: { q: "Bandwidth 40 GB/s (example). A model reads 2 GB of fp32 weights per token. What is the decode ceiling, and what if the weights were int8 blocks at 36 B per 32 weights?",
          options: ["20 tokens/s; int8 the same, because the FLOPs are unchanged", "20 tokens/s; about 71 tokens/s with int8 blocks", "80 tokens/s; 320 tokens/s"], answer: 1,
          why: "40 GB/s ÷ 2 GB = 20 tokens/s. Int8 blocks shrink the bytes by 128 / 36 ≈ 3.56×, to about 0.56 GB, so 40 ÷ 0.5625 ≈ 71 tokens/s. In a memory-bound step, fewer bytes is the lever; FLOPs don't matter." },
        scene(G) {
          G.text(24, 36, "worked example (made-up round numbers)", { size: 13 });
          G.box(24, 56, 180, 60, "B = 40 GB/s", { fill: "v", size: 15 });
          G.text(224, 94, "÷", { size: 22, color: "muted" });
          G.box(254, 56, 180, 60, "W = 0.54 GB", { fill: "w", size: 15 });
          G.text(454, 94, "=", { size: 22, color: "muted" });
          const res = G.box(484, 56, 132, 60, "≲ 74 tok/s", { fill: "ok", size: 15 });
          G.label(254, 134, "135M params × 4 B (fp32)", { size: 11, color: "ink" });
          G.text(24, 184, "measured is lower: what the formula leaves out", { size: 13 });
          const gaps = ["fork/join per matvec", "KV reads grow with pos", "norms, RoPE, softmax"];
          const g = gaps.map((t, i) => G.box(24 + i * 200, 198, 188, 40, t, { stroke: "hot", color: "hot", size: 11 }));
          G.text(24, 290, "the test model (0.45 MiB) lives in cache:", { size: 13 });
          G.label(24, 312, "B is cache bandwidth there, not DRAM", { size: 12, color: "ink" });
          G.text(24, 356, "int8 blocks: 36 B per 32 weights vs 128 B", { size: 13, color: "ok" });
          G.label(24, 378, "≈ 3.6× fewer bytes → ≈ 3.6× higher ceiling (DRAM-bound models)", { size: 12, color: "ink" });
          G.pulse(res, { repeat: 3 }); G.from(g, { opacity: 0, y: 10, stagger: 0.2, duration: 0.3 });
          G.caption("decode ceiling = bandwidth ÷ bytes read per token");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>An engine is a loop: forward one token, sample, append. A forward pass is embedding lookup, then per layer x += Attention(RMSNorm(x)) and x += MLP(RMSNorm(x)), then RMSNorm and the lm_head.</li>
<li>RMSNorm rescales to unit mean square; RoPE rotates pairs of q and k by position; attention scores earlier keys, softmaxes, and blends values; SwiGLU widens, gates and narrows.</li>
<li>The KV cache stores each position's keys and values once (2 × layers × kv_dim × bytes per token); GQA maps query head h to KV head h / group.</li>
<li>Top-p keeps the smallest set of tokens reaching p, renormalises and inverts the CDF at a caller-supplied u.</li>
<li>Weights are mmap'd zero-copy; every matvec is threaded and vectorised; decode speed is bounded by bandwidth ÷ bytes read per token.</li></ul>
<p>Next, P1.1 takes this same forward pass and counts its FLOPs and bytes for real models, layer by layer.</p>`,
    sim: {
      title: "Decode speed limit for your engine",
      intro: "Pick a model, the weight format and your memory bandwidth. The chart shows the bytes one decode step must read (weights plus the KV cache, which grows with the position) and the resulting tokens/s ceiling as the conversation gets longer. Use your measured STREAM triad bandwidth from P0.4.",
      height: 300,
      controls: [
        { id: "m", label: "model", type: "select", value: "llama8", options: [["tiny", "tiny test model"], ["ex135", "135M example"], ["llama8", "Llama-3-8B"]] },
        { id: "wb", label: "weight format", type: "select", value: 4, options: [[4, "fp32 (engine v0)"], [2, "bf16 (2 B)"], [1.125, "int8 blocks (36 B / 32)"]] },
        { id: "bw", label: "memory bandwidth, GB/s (example value: measure with 03_stream)", min: 5, max: 400, step: 5, value: 40 },
        { id: "pos", label: "position in the conversation (tokens so far)", min: 0, max: 8192, step: 64, value: 1024 },
      ],
      draw(G, v) {
        const m = MODELS[v.m], B = v.bw * 1e9;
        const total = m.params * v.wb;
        const perTokW = (m.tied ? m.params : m.params - m.embed + m.d) * v.wb;
        const kvTok = 2 * m.L * m.kv * 4;                 // engine v0 keeps K/V in fp32
        const step = (t) => perTokW + kvTok * (t + 1);
        const tps = (t) => B / step(t);
        // chart: tokens/s ceiling vs position
        const x0 = 70, y0 = 250, W = 540, H = 170, tmax = 8192, ymax = tps(0) * 1.15;
        G.axes(x0, y0 - H, W, H, { ylabel: "decode ceiling, tokens/s" });
        [0, 2048, 4096, 6144, 8192].forEach((t) => G.label(x0 + (t / tmax) * W, y0 + 16, F.num(t), { anchor: "middle", size: 10 }));
        [0, 0.5, 1].forEach((f) => G.label(x0 - 6, y0 - f * H + 4, F.num(ymax * f, ymax * f < 10 ? 1 : 0), { anchor: "end", size: 10 }));
        G.label(x0 + W, y0 + 40, "position (tokens so far) →", { anchor: "end", size: 11 });
        let d = ""; for (let i = 0; i <= 100; i++) { const t = (i / 100) * tmax; d += (i ? " L " : "M ") + (x0 + (t / tmax) * W).toFixed(1) + " " + (y0 - (tps(t) / ymax) * H).toFixed(1); }
        G.path(d, { color: "k", w: 3 });
        G.circle(x0 + (v.pos / tmax) * W, y0 - (tps(v.pos) / ymax) * H, 6, { fill: "ink" });
        // bytes split bar
        const s = step(v.pos), fw = perTokW / s;
        G.rect(x0, 26, W * fw, 14, { fill: "w", rx: 2 }); G.rect(x0 + W * fw, 26, W * (1 - fw), 14, { fill: "v", rx: 2 });
        G.label(x0, 18, `bytes per step: weights ${(100 * fw).toFixed(1)}% · KV cache ${(100 * (1 - fw)).toFixed(1)}%`, { size: 10 });
        const inCache = total < 32 * 2 ** 20;
        return [
          { title: m.name, rows: [["weights in the file", F.bytes(total)], ["weights read per token", F.bytes(perTokW)], ["KV cache per token (fp32)", m.L ? F.bytes(kvTok) : "n/a"], ["KV read at this position", m.L ? F.bytes(kvTok * (v.pos + 1)) : "n/a"]],
            chip: [!inCache, inCache ? "fits in CPU caches: not DRAM-bound" : "DRAM-bound: B ÷ bytes applies"] },
          { title: "Ceiling at this position", rows: [["bytes per decode step", F.bytes(s)], ["time per step ≥", F.ms(s / B)], ["tokens/s ≤", F.num(tps(v.pos), 1)], ["tokens/s ≤ at position 0", F.num(tps(0), 1)]] },
          { title: "Model", html: `<p class="note">Lower bound on time: every weight except the embedding table (one row per token; the whole table when it doubles as the tied lm_head) plus all stored K and V, read once at bandwidth B. Ignores fork/join, non-matvec ops and cache effects. ${m.L ? "" : "The 135M example has no layer config here, so KV reads are not counted."} The 32 MiB "fits in cache" line is a rough example; check your CPU's last-level cache.</p>` },
        ];
      },
    },
    practice: {
      intro: `Run from the repository root. The exercises build <code>platform/engine/v0</code> against <b>your</b> <code>exercises/ops/ops.hpp</code>; generate the NumPy fixtures once first. Add <code>-DS2S_USE_SOLUTIONS=ON</code> to the configure step to build against the reference ops in <code>platform/engine/v0/src/ops.hpp</code>.`,
      items: [
        { title: "Build fixtures and the exercise project", tier: "setup", goal: "Write the NumPy fixtures (tiny model + expected outputs), then configure and build all five tests and your engine binary.",
          cmd: "uv run python course/P0-systems-primer/P0.5-project-tiny-engine-cpu/exercises/make_fixtures.py build/p05-fixtures && cmake -S course/P0-systems-primer/P0.5-project-tiny-engine-cpu/exercises -B build/p05-ex && cmake --build build/p05-ex -j" },
        { title: "RMSNorm and RoPE", tier: "T0 · easy", goal: "ops::rmsnorm (in place, double accumulation) and rotate_half ops::rope, within 1e-5 of NumPy at positions 0, 1, 7 and 100.",
          cmd: "ctest --test-dir build/p05-ex -R 01 --output-on-failure" },
        { title: "Attention with a KV cache (GQA)", tier: "T0 · medium", goal: "Causal attention over positions 0 … pos with head h reading KV head h / group, checked at every position.",
          cmd: "ctest --test-dir build/p05-ex -R 02 --output-on-failure" },
        { title: "Full forward pass", tier: "T0 · medium", goal: "Teacher-forced logits within 1e-4 and 48 greedy tokens matching NumPy; then break RoPE, GQA and the causal bound on purpose and watch which test fails.",
          cmd: "ctest --test-dir build/p05-ex -R 03 --output-on-failure" },
        { title: "Temperature + top-p sampling", tier: "T0 · medium", goal: "Greedy at T = 0, nucleus cut-off and renormalisation, passing a χ² test over 10⁵ seeded draws.",
          cmd: "ctest --test-dir build/p05-ex -R 04 --output-on-failure" },
        { title: "Int8 weights (runq-style)", tier: "T0 · hard", goal: "32-value int8 blocks with one scale each; the q8 matvec stays within its derived error bound. Then wire it into a copy of the engine and report the speed-up.",
          cmd: "ctest --test-dir build/p05-ex -R 05 --output-on-failure" },
        { title: "Benchmark against the ceiling", tier: "T0 · lab", goal: "Build the platform engine, make the tiny model, and print decode tokens/s next to the B ÷ W prediction.",
          cmd: "cmake -S platform/engine/v0 -B build/engine-v0 -DCMAKE_BUILD_TYPE=Release && cmake --build build/engine-v0 -j && uv run python platform/engine/v0/tools/make_tiny_llama.py build/tiny-llama && uv run python course/P0-systems-primer/P0.5-project-tiny-engine-cpu/bench/engine_bench.py --model build/tiny-llama" },
      ],
      labs: [
        { label: "The engine: main loop, forward pass, ops, loader", path: "platform/engine/v0/src/" },
        { label: "The NumPy reference every test compares against", path: "platform/engine/v0/reference/llama_numpy.py" },
        { label: "Engine tests (teacher-forced logits + greedy tokens): uv run pytest platform/engine/v0/tests", path: "platform/engine/v0/tests/test_engine.py" },
        { label: "What temperature and top-p do to a distribution", path: "course/P0-systems-primer/P0.5-project-tiny-engine-cpu/examples/03_sampling_demo.py" },
        { label: "Optional: convert a small Hugging Face Llama model (needs network and the torch extra; check its licence)", path: "platform/engine/v0/tools/convert_hf_model.py" },
        { label: "Animations reused here: first touch of the weights; roofline preset 'LLM decode, batch 1'", path: "animations/p0-vm-page-walk.html · animations/roofline.html" },
        { label: "Self-check questions", path: "course/P0-systems-primer/P0.5-project-tiny-engine-cpu/quiz.md" },
      ],
    },
  });
})();
