/* P1.1 — The transformer forward pass, by the numbers. */
(function () {
  const F = S2S.fmt;
  // Model configs as in examples/03_flop_counter.py (UNVERIFIED until checked against each config.json; exercise 1).
  const MODELS = {
    llama8: { name: "Llama-3-8B", d: 4096, L: 32, H: 32, kv: 8, f: 14336, V: 128256, tied: false },
    llama70: { name: "Llama-3-70B", d: 8192, L: 80, H: 64, kv: 8, f: 28672, V: 128256, tied: false },
    mistral: { name: "Mistral-7B", d: 4096, L: 32, H: 32, kv: 8, f: 14336, V: 32000, tied: false },
    tiny: { name: "tiny test model", d: 64, L: 2, H: 4, kv: 2, f: 160, V: 256, tied: false },
  };
  // parameter groups for a config
  function count(m) {
    const h = m.d / m.H;
    const q = m.d * m.H * h, o = m.H * h * m.d, kv = 2 * m.d * m.kv * h, mlp = 3 * m.d * m.f;
    const g = { attn: m.L * (q + o + kv), mlp: m.L * mlp, emb: m.V * m.d, head: m.tied ? 0 : m.V * m.d, norms: m.L * 2 * m.d + m.d };
    g.total = g.attn + g.mlp + g.emb + g.head + g.norms;
    g.matrices = g.attn + g.mlp + m.V * m.d;           // every matvec weight, incl. the LM head; not the embedding lookup
    return g;
  }
  // a vector drawn as a strip of cells
  const PN = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + " B" : n >= 1e6 ? (n / 1e6).toFixed(1) + " M" : F.num(n));
  const strip = (G, x, y, n, cw, ch, color, o = {}) => G.grid(x, y, 1, n, cw, ch, () => color, { gap: o.gap ?? 2, rx: 2 });

  S2S.lesson({
    id: "p1-1", n: "P1.1", title: "The transformer forward pass",
    subtitle: "Inference fundamentals · first principles · T0, no GPU needed",
    kicker: "Lesson · ≈ 45 min",
    headline: "Eight billion numbers, used twice each",
    intro: `<p>A large language model is one function, called once per token. This lesson opens it up and walks a single token from its id to the next-token scores, through every operation in a Llama-style model. At each stop it counts three things: how many weights the operation owns, how much arithmetic it does, and how many bytes it reads. Those three counts are the raw material for every later calculation in the course: latency, GPU choice, batch size and cost.</p>`,
    facts: ["11 steps", "4 checkpoints", "1 simulator", "4 exercises"],
    legend: [["blue", "activation vector"], ["w", "weights"], ["q", "query"], ["k", "key"], ["v", "value"], ["ok", "output"]],
    prev: "p0-5", next: "p1-2",
    steps: [
      { rail: "the function", title: "One token in, 128,256 scores out",
        body: `<p>Text is first cut into <b>tokens</b> (word pieces), and each token is an integer id. Llama-3-8B knows 128,256 different tokens: its <b>vocabulary</b>.</p>
<p>A <b>forward pass</b> turns the ids into one score per vocabulary entry: how strongly the model predicts each token to come next. Softmax turns the scores (called <b>logits</b>) into probabilities. The path has four stops:</p>
<ol><li><b>Embedding:</b> look up row <i>id</i> of a 128,256 × 4,096 table. The token is now a vector of 4,096 numbers.</li>
<li><b>32 layers</b>, all the same shape, each refining that vector.</li>
<li>A final <b>normalisation</b>.</li>
<li>The <b>LM head</b>: a 128,256 × 4,096 matrix times the vector gives the 128,256 logits.</li></ol>
<p>Everything in this lesson uses Llama-3-8B's published shape: hidden size <code>d = 4096</code>, <code>L = 32</code> layers, 32 query heads, 8 key/value heads, MLP width 14,336, vocabulary 128,256. The repo's presets carry these as UNVERIFIED until exercise 1 checks them against the real <code>config.json</code>.</p>`,
        scene(G) {
          G.box(24, 40, 96, 40, "id 1820", { stroke: "ink", size: 12 });
          G.label(24, 100, "token id", { size: 11 });
          G.arrow(124, 60, 160, 60, { color: "muted" });
          // embedding table
          G.rect(166, 28, 120, 150, { fill: "w", opacity: 0.45, rx: 6 });
          G.rect(166, 92, 120, 10, { fill: "blue", rx: 2 });
          G.label(166, 196, "embedding table", { size: 11 });
          G.label(166, 212, "128,256 × 4,096", { size: 11 });
          G.arrow(290, 97, 320, 97, { color: "muted" });
          // layer stack
          const layers = [];
          for (let l = 0; l < 6; l++) layers.push(G.rect(326, 150 - l * 22, 120, 16, { fill: "w", opacity: 0.35 + l * 0.08, rx: 4 }));
          G.label(326, 196, "32 layers", { size: 11 });
          G.label(326, 212, "attention + MLP", { size: 11 });
          G.arrow(450, 97, 480, 97, { color: "muted" });
          G.rect(486, 28, 120, 150, { fill: "w", opacity: 0.45, rx: 6 });
          G.label(486, 196, "norm + LM head", { size: 11 });
          G.label(486, 212, "128,256 × 4,096", { size: 11 });
          // logits bars
          const vals = [0.2, 0.5, 0.3, 0.9, 0.25, 0.4, 2.6, 0.3, 0.6, 0.2, 1.4, 0.35, 0.5, 0.2, 0.3, 0.8, 0.25, 0.4, 0.3, 0.5];
          const b = G.bars(60, 380, vals, { w: 22, gap: 6, h: 110, max: 2.6, fill: (i) => (i === 6 ? "ok" : "blue") });
          G.line(48, 380, 620, 380, { color: "line" });
          G.label(60, 252, "logits: one score per vocabulary entry (20 of 128,256 shown)", { size: 12 });
          G.label(232, 400, "highest score → most likely next token", { size: 11, color: "ok" });
          G.from(layers, { opacity: 0, stagger: 0.1, duration: 0.3 });
          G.from(b, { attr: { height: 0, y: 380 }, stagger: 0.03, duration: 0.4, delay: 0.5 });
          G.caption("id → vector → 32 layers → one score per vocabulary token");
        } },

      { rail: "residual stream", title: "Each layer adds, it never replaces",
        body: `<p>Inside the stack, the 4,096-number vector is called the <b>residual stream</b>. A layer does not overwrite it. It computes a correction and <b>adds</b> it:</p>
<div class="eq">h = rmsnorm(x);  x = x + attention(h)
h = rmsnorm(x);  x = x + mlp(h)</div>
<p>Why add instead of replace? A layer that has nothing useful to contribute can output nearly zero and pass the vector through unchanged. That makes deep stacks trainable, and it means every layer reads and writes the same shape: <code>[4096]</code> in, <code>[4096]</code> out.</p>
<p>This is the exact loop body of <code>LlamaNumpy.forward</code> in <code>platform/engine/v0/reference/llama_numpy.py</code>, the model you built in P0.5. It is the reference every C++ and CUDA engine in the course is tested against, which is why example 02 checks it against Hugging Face <code>transformers</code> logit for logit.</p>`,
        scene(G) {
          const y = 230;
          const s = G.line(30, y, 610, y, { color: "blue", w: 8 });
          G.label(30, y + 34, "x  (4,096 numbers)", { color: "blue" });
          const branch = (x0, name, col) => {
            G.path(`M ${x0} ${y} C ${x0} ${y - 90}, ${x0 + 30} ${y - 120}, ${x0 + 60} ${y - 120}`, { color: "muted", w: 2 });
            G.box(x0 + 60, y - 140, 72, 40, "norm", { stroke: "muted", size: 12 });
            G.arrow(x0 + 132, y - 120, x0 + 150, y - 120, { color: "muted" });
            G.box(x0 + 150, y - 140, 92, 40, name, { fill: col, size: 12 });
            G.path(`M ${x0 + 242} ${y - 120} C ${x0 + 270} ${y - 120}, ${x0 + 280} ${y - 90}, ${x0 + 280} ${y - 14}`, { color: "muted", w: 2, arrow: true });
            const plus = G.circle(x0 + 280, y, 14, { fill: "bg", stroke: "ink" });
            G.text(x0 + 280, y + 5, "+", { anchor: "middle", size: 16 });
            return plus;
          };
          const p1 = branch(30, "attention", "q");
          const p2 = branch(320, "MLP", "v");
          G.label(30, 330, "one layer = two read → compute → add steps", { size: 13, color: "ink" });
          G.label(30, 354, "the vector keeps its shape [4096] all the way up the stack", { size: 12 });
          G.pulse([p1, p2], { repeat: 5 });
          G.from(s, { attr: { x2: 30 }, duration: 0.8 });
          G.caption("layers write corrections into a shared stream");
        } },

      { rail: "rmsnorm", title: "RMSNorm keeps the numbers in range",
        body: `<p><b>Problem:</b> after dozens of additions, some entries of the stream can grow large while others stay small. The weight matrices were trained to expect inputs of a steady size.</p>
<p><b>Fix:</b> before each sub-block, divide the vector by its <b>root mean square</b> (square every entry, take the mean, take the square root), then multiply each entry by a learned scale <code>w</code>:</p>
<div class="eq">rms(x) = sqrt(mean(x²) + eps)
rmsnorm(x) = x / rms(x) * w

x = [2, -2, 4, 0]
mean(x²) = (4 + 4 + 16 + 0) / 4 = 6
rms = sqrt(6) = 2.449
x / rms = [0.82, -0.82, 1.63, 0]</div>
<p>The result has root mean square 1 whatever came in. Cost: about 4 operations per entry and only 4,096 weights per norm. Each layer has 2 norms, so RMSNorm owns 8,192 weights per layer: tiny next to the matrices. Unlike LayerNorm it does not subtract the mean, which saves a pass over the vector.</p>`,
        scene(G) {
          const before = [2, -2, 4, 0], after = before.map((v) => v / Math.sqrt(6));
          const draw = (x0, vals, title, sub) => {
            G.text(x0, 60, title, { size: 14 });
            G.label(x0, 80, sub, { size: 12 });
            const base = 240, sc = 34, out = [];
            G.line(x0, base, x0 + 250, base, { color: "line" });
            vals.forEach((v, i) => {
              const h = Math.abs(v) * sc;
              out.push(G.rect(x0 + 16 + i * 58, v >= 0 ? base - h : base, 40, Math.max(2, h), { fill: v >= 0 ? "blue" : "hot", rx: 3 }));
              G.text(x0 + 36 + i * 58, v >= 0 ? base + 20 : base - 8, v.toFixed(2).replace(".00", ""), { anchor: "middle", size: 12, color: "ink" });
            });
            return out;
          };
          draw(24, before, "before", "rms = 2.449");
          const b = draw(350, after, "after  x / rms", "rms = 1.000");
          G.arrow(286, 200, 340, 200, { color: "ink", w: 2 });
          G.text(24, 340, "rmsnorm(x) = x / sqrt(mean(x²) + eps) × w", { size: 14 });
          G.label(24, 366, "same direction, standard size; w is a learned per-entry scale", { size: 12 });
          G.from(b, { scaleY: 2.4, transformOrigin: "50% 240px", duration: 0.7 });
          G.caption("divide by the root mean square: the shape stays, the scale resets");
        } },

      { rail: "matvec cost", title: "A matrix times a vector: 2 FLOPs per weight",
        body: `<p>Almost all of the model's work is one operation: multiply a weight matrix by a vector. A matrix with <code>m</code> rows and <code>n</code> columns times an <code>n</code>-vector gives an <code>m</code>-vector. Each output entry is a dot product: <code>n</code> multiplies and <code>n</code> adds.</p>
<div class="eq">FLOPs = 2 × m × n   (multiply + add per weight)
bytes = m × n × bytes per weight</div>
<p>A <b>FLOP</b> is one floating-point operation. Take the query projection of Llama-3-8B, a 4,096 × 4,096 matrix:</p>
<div class="eq">FLOPs = 2 × 4096 × 4096 = 33,554,432 ≈ 33.6 M
bytes = 4096 × 4096 × 2 (bf16) = 32 MiB</div>
<p>So the rule for a single token is simple. <b>Every weight is read once and used for exactly one multiply-add.</b> Remember this ratio: 2 FLOPs for every weight, and 2 bytes per weight in bf16 (a 16-bit number format).</p>`,
        check: { q: "The LM head of Llama-3-8B is a 128,256 × 4,096 matrix. How many FLOPs does it cost per token?",
          options: ["About 0.5 billion", "About 1.05 billion", "About 4.2 billion"], answer: 1,
          why: "It has 128,256 × 4,096 = 525 M weights, and each weight is one multiply-add, so 2 × 525 M ≈ 1.05 billion FLOPs for every token generated." },
        scene(G) {
          const n = 10, m = 8, cs = 26, x0 = 60, y0 = 70;
          const W = G.grid(x0, y0, m, n, cs, cs, () => "w", { gap: 3, rx: 3 });
          G.label(x0, y0 - 14, "W: m rows × n columns", { size: 12 });
          G.text(x0 + n * cs + 22, y0 + (m * cs) / 2 + 6, "×", { size: 22, color: "muted" });
          const xv = G.grid(x0 + n * cs + 50, y0, n, 1, cs, cs * 0.8, () => "blue", { gap: 3, rx: 3 });
          G.label(x0 + n * cs + 40, y0 - 14, "x (n)", { size: 12 });
          G.text(x0 + n * cs + 100, y0 + (m * cs) / 2 + 6, "=", { size: 22, color: "muted" });
          const out = G.grid(x0 + n * cs + 128, y0, m, 1, cs, cs, (r) => (r === 2 ? "ok" : "line"), { gap: 3, rx: 3 });
          G.label(x0 + n * cs + 118, y0 - 14, "y (m)", { size: 12 });
          G.rect(x0 - 3, y0 + 2 * cs - 3, n * cs + 3, cs + 3, { stroke: "ok", sw: 2, rx: 4 });
          G.label(x0, y0 + m * cs + 26, "one output = one row · x = n multiplies + n adds", { size: 12, color: "ok" });
          G.text(24, 350, "q_proj: 2 × 4096 × 4096 = 33.6 M FLOPs", { size: 14 });
          G.text(24, 376, "        4096 × 4096 × 2 B = 32 MiB read", { size: 14 });
          G.from(W, { opacity: 0.1, stagger: { each: 0.004, from: "start" }, duration: 0.2 });
          G.from(out, { opacity: 0, stagger: 0.1, duration: 0.2, delay: 0.4 });
          G.caption("every weight: read once, one multiply, one add");
        } },

      { rail: "attention heads", title: "Attention: 32 small lookups side by side",
        body: `<p>Attention is the only step where a token uses information from <i>other</i> tokens. First the normed vector <code>h</code> is projected three ways:</p>
<div class="eq">q = W_Q · h    4096 → 4096   what this token looks for
k = W_K · h    4096 → 1024   what this token offers
v = W_V · h    4096 → 1024   what it hands over</div>
<p>The 4,096 query numbers are cut into <b>32 heads</b> of 128 numbers (<code>head_dim = 4096 / 32 = 128</code>). Each head scores the current query against the keys of earlier tokens with a dot product, turns the scores into weights with softmax, and takes the weighted sum of their values. Heads run independently, so each can learn to track a different relation (the previous word, the subject of the sentence, a matching bracket).</p>
<p>The 32 head outputs are glued back into 4,096 numbers and multiplied by <code>W_O</code> (4,096 × 4,096), which mixes the heads and returns a vector of the stream's shape. P1.2 walks through the scoring itself; here we count it.</p>`,
        scene(G) {
          G.label(24, 44, "h (4,096)", { size: 12 });
          strip(G, 24, 52, 32, 18, 18, "blue");
          G.arrow(310, 76, 310, 100, { color: "muted" });
          G.label(320, 94, "q = W_Q · h", { size: 12 });
          G.label(24, 124, "q, cut into 32 heads × 128", { size: 12, color: "q" });
          const heads = [];
          for (let i = 0; i < 32; i++) heads.push(G.rect(24 + i * 18, 132, 16, 60, { fill: "q", rx: 3, opacity: 0.55 + 0.45 * ((i % 4) === 0) }));
          const fan = [];
          [0, 9, 20, 31].forEach((i, j) => {
            const cx = 32 + i * 18;
            fan.push(G.arrow(cx, 196, 70 + j * 150, 248, { color: "q" }));
            G.box(24 + j * 150, 252, 110, 40, `head ${i + 1}`, { stroke: "q", color: "q", size: 12 });
            G.label(30 + j * 150, 310, "softmax(q·K)·V", { size: 11 });
          });
          G.label(24, 344, "each head: 128 numbers, its own attention pattern", { size: 12, color: "ink" });
          G.label(24, 370, "concat 32 × 128 = 4,096 → W_O (4096 × 4096) → back to the stream", { size: 12 });
          G.from(heads, { opacity: 0, y: -10, stagger: 0.02, duration: 0.25 });
          G.from(fan, { opacity: 0, stagger: 0.12, duration: 0.3, delay: 0.6 });
          G.caption("32 independent attention heads of 128 numbers each");
        } },

      { rail: "gqa", title: "Grouped-query attention: heads share keys",
        body: `<p>Why is <code>W_K</code> only 4,096 → 1,024? Because Llama-3-8B has <b>8 key/value heads</b> for its 32 query heads. Query heads 1–4 all read key/value head 1, heads 5–8 read head 2, and so on. This is <b>grouped-query attention (GQA)</b>. The reference code finds a head's partner with <code>kvh = hh // group</code>, where <code>group = 32 / 8 = 4</code>.</p>
<div class="eq">MHA (one K/V head per query head):
  k_proj = 4096 × 4096  = 16.8 M weights
GQA, 8 K/V heads:
  k_proj = 4096 × 1024  =  4.2 M weights</div>
<p>The real prize is not the weights. Every past token's keys and values must be kept in memory while generating (P1.2's <b>KV cache</b>), and that store shrinks by the same factor, 32 / 8 = 4×. The extremes have names: <b>MHA</b> (multi-head: as many K/V heads as query heads) and <b>MQA</b> (multi-query: one K/V head for all).</p>
<p>What GQA does <i>not</i> save: arithmetic in the scoring. Each of the 32 query heads still takes its own dot product with every key.</p>`,
        check: { q: "Llama-3-8B moves from 32 to 8 key/value heads. Which of these shrinks by 4×?",
          options: ["The FLOPs spent scoring queries against keys", "The keys and values stored per token", "The query projection W_Q"], answer: 1,
          why: "K and V are produced per KV head, so 4× fewer KV heads means 4× less K/V to store and read. All 32 query heads still score every key, so attention FLOPs stay the same, and W_Q keeps its 4096 × 4096 shape." },
        scene(G) {
          G.label(24, 44, "32 query heads", { size: 12, color: "q" });
          const qs = [];
          for (let i = 0; i < 32; i++) qs.push(G.rect(24 + i * 18.5, 54, 15, 34, { fill: "q", rx: 3 }));
          G.label(24, 264, "8 key / value heads", { size: 12, color: "k" });
          const kv = [];
          for (let j = 0; j < 8; j++) {
            const x = 24 + j * 74;
            kv.push(G.box(x + 4, 200, 30, 34, "k", { fill: "k", size: 11, rx: 4 }), G.box(x + 36, 200, 30, 34, "v", { fill: "v", size: 11, rx: 4 }));
            for (let i = 0; i < 4; i++) G.line(24 + (j * 4 + i) * 18.5 + 7.5, 90, x + 35, 198, { color: "q", w: 1.2, opacity: 0.7 });
          }
          G.text(24, 316, "group = 32 / 8 = 4 query heads per K/V head", { size: 14 });
          G.text(24, 344, "K/V stored per token: 4× smaller than MHA", { size: 14, color: "ok" });
          G.label(24, 370, "scoring FLOPs unchanged: every query head still dots every key", { size: 12 });
          G.from(kv, { opacity: 0, y: 20, stagger: 0.05, duration: 0.3 });
          G.caption("four query heads read the same key and value");
        } },

      { rail: "rope", title: "RoPE: position by rotation",
        body: `<p><b>Problem:</b> a dot product does not care where a key sits in the sequence. Shuffle the earlier tokens and every score stays the same. But "dog bites man" and "man bites dog" must differ.</p>
<p><b>Rotary position embedding (RoPE)</b> rotates <code>q</code> and <code>k</code> by an angle that grows with their position. The 128 numbers of a head are treated as 64 pairs (in this code, element <code>i</code> with element <code>i + 64</code>), and pair <code>i</code> at position <code>m</code> is turned by angle <code>m · ωᵢ</code>:</p>
<div class="eq">ωᵢ = θ^(−2i / 128)        i = 0 … 63
θ = 10,000:  ω₀ = 1 rad per position
             ω₆₃ ≈ 0.000115 rad per position</div>
<p>Fast pairs distinguish neighbours; slow pairs distinguish distant tokens, like the hands of a clock. Now the key fact. Rotating <code>q</code> by angle <code>a</code> and <code>k</code> by angle <code>b</code> leaves a dot product that depends only on <code>b − a</code>. So the score between positions <code>m</code> and <code>n</code> depends only on the <b>offset</b> <code>n − m</code>, which is exactly what exercise 4 asks you to prove numerically.</p>
<p>θ is the config's <code>rope_theta</code>. The reference code defaults to 10,000; Llama-3 sets a larger value in its <code>config.json</code> (read it there). RoPE has no weights, and costs a few operations per number.</p>`,
        scene(G) {
          const dial = (cx, cy, a, b, lab) => {
            G.circle(cx, cy, 92, { stroke: "line" });
            const r = 82, out = [];
            const ax = (ang, col, txt) => {
              const x = cx + r * Math.cos(-ang), y = cy + r * Math.sin(-ang);
              out.push(G.arrow(cx, cy, x, y, { color: col, w: 3 }));
              G.text(x + (Math.cos(-ang) > 0 ? 8 : -26), y + (Math.sin(-ang) > 0 ? 16 : -6), txt, { color: col, size: 13 });
            };
            ax(a, "q", "q"); ax(b, "k", "k");
            G.path(`M ${cx + 40 * Math.cos(-a)} ${cy + 40 * Math.sin(-a)} A 40 40 0 0 0 ${cx + 40 * Math.cos(-b)} ${cy + 40 * Math.sin(-b)}`, { color: "ok", w: 2 });
            G.label(cx - 92, cy + 124, lab, { size: 12, color: "ink" });
            return out;
          };
          const w = 0.35;
          dial(150, 170, 2 * w, 5 * w, "q at position 2, k at position 5");
          const b = dial(460, 170, 6 * w, 9 * w, "both shifted by 4: positions 6 and 9");
          G.text(24, 352, "angle between them = (5 − 2)·ω = (9 − 6)·ω", { size: 14, color: "ok" });
          G.label(24, 378, "the score depends only on the offset n − m, not on m or n", { size: 12 });
          G.from(b, { rotation: -80, svgOrigin: "460 170", duration: 0.9 });
          G.caption("rotate q and k by position: their angle encodes the distance");
        } },

      { rail: "swiglu mlp", title: "The MLP: widen, gate, narrow",
        body: `<p>Attention moves information between tokens. The <b>MLP</b> (multi-layer perceptron) then processes each token's vector on its own. Llama uses a gated version called <b>SwiGLU</b>:</p>
<div class="eq">g = W_gate · h     4096 → 14336
u = W_up   · h     4096 → 14336
y = W_down · (silu(g) * u)   14336 → 4096
silu(z) = z / (1 + e^(−z))</div>
<p>The vector is widened 3.5×, one copy decides how much of each of the 14,336 features passes (the "gate", through the smooth on/off function <code>silu</code>), the gated features are multiplied element by element, and <code>W_down</code> squeezes the result back to 4,096.</p>
<p>Count the weights: three matrices of 4,096 × 14,336 each.</p>
<div class="eq">3 × 4096 × 14336 = 176,160,768 ≈ 176 M per layer
attention q,k,v,o = 41,943,040 ≈  42 M per layer</div>
<p>The MLP holds <b>81%</b> of each layer's weights. When you later shard, quantize or offload a model, most of the bytes you move live here.</p>`,
        scene(G) {
          G.label(16, 112, "h · 4,096", { size: 12 });
          G.rect(24, 120, 16, 200, { fill: "blue", rx: 3 });
          G.line(42, 170, 116, 120, { color: "muted" }); G.line(42, 270, 116, 324, { color: "muted" });
          G.label(110, 50, "W_gate → g · 14,336", { size: 12, color: "q" });
          G.label(110, 256, "W_up → u · 14,336", { size: 12, color: "v" });
          const wide = [G.rect(120, 60, 36, 120, { fill: "q", rx: 3 }), G.rect(120, 264, 36, 120, { fill: "v", rx: 3 })];
          G.box(196, 80, 90, 80, "silu(g)", { stroke: "q", color: "q", size: 13 });
          G.arrow(158, 120, 194, 120, { color: "muted" });
          G.circle(340, 222, 20, { fill: "bg", stroke: "ink" }); G.text(340, 228, "×", { anchor: "middle", size: 18 });
          G.path("M 288 120 C 330 120, 340 160, 340 200", { color: "q", arrow: true });
          G.path("M 158 324 C 300 324, 340 290, 340 244", { color: "v", arrow: true });
          G.arrow(362, 222, 390, 222, { color: "muted" });
          G.box(394, 186, 92, 72, "W_down", { fill: "w", size: 13 });
          G.arrow(488, 222, 520, 222, { color: "muted" });
          G.rect(526, 122, 16, 200, { fill: "blue", rx: 3 });
          G.label(512, 340, "4,096", { size: 12 });
          G.label(196, 400, "176 M weights per layer = 81% of the layer", { size: 12, color: "ink" });
          G.from(wide, { scaleY: 0.2, transformOrigin: "50% 50%", duration: 0.6 });
          G.caption("widen to 14,336, gate, multiply, narrow back to 4,096");
        } },

      { rail: "parameter count", title: "Count every weight: 8.03 billion",
        body: `<p>Add up the matrices. Write <code>d</code> for the hidden size, <code>H_kv · h</code> for the K/V width, <code>d_ff</code> for the MLP width:</p>
<div class="eq">per layer
  q, o       2 × d²              33,554,432
  k, v       2 × d × H_kv·h       8,388,608
  MLP        3 × d × d_ff       176,160,768
  2 norms    2 × d                    8,192
  sum                           218,112,000

× 32 layers                   6,979,584,000
embedding   V × d               525,336,576
LM head     V × d               525,336,576
final norm  d                         4,096
total                         8,030,261,248</div>
<p>8.03 billion: the "8B" in the name, a good sign the formula is right. Two things are easy to miss. The <b>LM head</b> is half a billion weights by itself. And some models <b>tie</b> it to the embedding (one shared table, <code>tie_word_embeddings: true</code>), which removes one <code>V × d</code> term; Llama-3-8B does not.</p>
<p>Exercise 1 asks you to write exactly this as <code>count_params(cfg)</code> and test it against the summed tensor sizes of real model objects.</p>`,
        scene(G) {
          const parts = [["MLP", 32 * 176160768, "v"], ["attention", 32 * 41943040, "q"], ["embedding", 525336576, "k"], ["LM head", 525336576, "ok"]];
          const tot = 8030261248, x0 = 24, Wd = 592;
          G.text(24, 50, "Llama-3-8B: 8,030,261,248 parameters", { size: 15 });
          let x = x0; const segs = [];
          parts.forEach(([n, v, c], i) => {
            const w = (Wd * v) / tot;
            segs.push(G.rect(x, 74, w - 2, 70, { fill: c, rx: 4 }));
            const y = 180 + i * 46;
            G.rect(24, y - 14, 16, 16, { fill: c, rx: 3 });
            G.text(52, y, n, { size: 14 });
            G.text(250, y, (v / 1e9).toFixed(2) + " B", { size: 14, anchor: "end" });
            G.label(270, y, ((100 * v) / tot).toFixed(1) + "%", { size: 13 });
            x += w;
          });
          G.label(24, 376, "norms: 266,240 weights (0.003%), too thin to draw", { size: 12 });
          G.from(segs, { scaleX: 0, transformOrigin: "0% 50%", stagger: 0.2, duration: 0.4 });
          G.caption("the MLPs own most of the model; the vocabulary owns 13%");
        } },

      { rail: "2 FLOPs per param", title: "Why a token costs about 2 × P FLOPs",
        body: `<p>Generating one token uses every weight matrix exactly once, as a matrix times a vector. Step 4 says that costs 2 FLOPs per weight. So:</p>
<div class="eq">FLOPs per token ≈ 2 × (weights in matrices)
  = 2 × (6,979,584,000 − 262,144 norm weights
         + 525,336,576 LM head)
  = 2 × 7,504,658,432 ≈ 15.0 GFLOP</div>
<p>The embedding is left out: it copies one row, it doesn't multiply. That is the familiar rule <b>FLOPs ≈ 2 × P</b>. Now the bytes. Generating one token for one user reads every weight once:</p>
<div class="eq">bytes per token ≈ 8.03 B × 2 bytes = 16.1 GB (bf16)
FLOPs per byte  ≈ 15.0 G / 16.1 G ≈ 0.93 ≈ 1</div>
<p>Around one FLOP for every byte. A GPU can do far more arithmetic per byte than that, so producing one token at a time is limited by memory speed, not by arithmetic. P1.2 builds on this, and P1.4 measures "far more" for real GPUs.</p>
<p>The exception: when <code>T</code> prompt tokens go through together, each weight read from memory is used <code>T</code> times, so FLOPs per byte becomes roughly <code>T</code>. Same model, same weights, a very different bottleneck.</p>`,
        check: { q: "A 7B-parameter model stored in bf16 generates one token for one user. About how many FLOPs does it do per byte it reads?",
          options: ["About 1", "About 7", "About 14 billion"], answer: 0,
          why: "Each weight costs 2 bytes to read and is used in 2 FLOPs (one multiply-add): ≈ 14 GFLOP for ≈ 14 GB, so about 1 FLOP per byte. That is why single-token generation waits on memory." },
        scene(G) {
          G.text(24, 48, "one token, one user, Llama-3-8B in bf16", { size: 14 });
          const row = (y, label, val, max, col, txt) => {
            G.label(24, y - 8, label, { size: 12, color: "ink" });
            const r = G.rect(24, y, (590 * val) / max, 34, { fill: col, rx: 4 });
            G.text(34, y + 22, txt, { size: 13, color: "bg" });
            return r;
          };
          const a = row(92, "work: 2 FLOPs × 7.50 B matrix weights", 15.0, 16.1, "q", "15.0 GFLOP");
          const b = row(166, "traffic: 8.03 B weights × 2 bytes", 16.1, 16.1, "w", "16.1 GB read");
          G.text(24, 244, "FLOPs per byte ≈ 0.93", { size: 18, color: "hot" });
          G.label(24, 270, "one token at a time: memory speed sets the pace", { size: 12 });
          G.line(24, 300, 614, 300, { color: "line", dash: "3 4" });
          G.text(24, 330, "prompt of T tokens in one pass:", { size: 13 });
          G.label(24, 354, "weights still read once, but used T times → ≈ T FLOPs per byte", { size: 12 });
          const bs = G.bars(24, 412, [1, 2, 4, 8, 16, 32], { w: 40, gap: 30, h: 46, fill: "ok" });
          G.label(470, 404, "T = 1, 2, 4 … 32", { size: 11 });
          G.from([a, b], { attr: { width: 0 }, stagger: 0.3, duration: 0.6 });
          G.from(bs, { attr: { height: 0, y: 410 }, stagger: 0.08, duration: 0.3, delay: 0.8 });
          G.caption("2 FLOPs and 2 bytes per weight: about 1 FLOP per byte");
        } },

      { rail: "long context", title: "When 2 × P breaks: long context",
        body: `<p>One term is missing from 2 × P. When the new token attends to <code>t</code> earlier positions, each query head does two more passes over the cache: <code>q · K</code> (a dot product with every key, <code>2 · h · t</code> FLOPs) and <code>weights · V</code> (another <code>2 · h · t</code>). With 32 heads of 128 numbers:</p>
<div class="eq">attention FLOPs = 4 × H × h × t × L
                = 4 × 4096 × 32 × t
                = 524,288 × t

equals 2 × P_matrices when
  t = 15.0 G / 524,288 ≈ 28,600 tokens</div>
<p>At a 1,000-token context attention adds 3% to the work. At 8,000 tokens, 22%. At 32,000, 53%. At 128,000, 82% of every token's FLOPs are attention. (The README's "≈ 31k" uses all 8.03 B parameters instead of just the matrix weights; same idea.)</p>
<p>These are per-token FLOPs for decode. The bytes side grows too: each new token reads the whole KV cache, which is P1.2's subject.</p>`,
        check: { q: "At what context length does attention's FLOP cost roughly equal the weight FLOPs for Llama-3-8B?",
          options: ["About 4,000 tokens", "About 29,000 tokens", "About 500,000 tokens"], answer: 1,
          why: "Weight FLOPs are fixed at ≈ 15.0 G per token; attention adds 4 × d × L = 524,288 FLOPs per past token. 15.0 G / 524,288 ≈ 28,600." },
        scene(G) {
          const ts = [1, 2, 4, 8, 16, 32, 64, 128], wF = 15.009, aF = (t) => 0.524288 * t;
          G.text(24, 40, "FLOPs per decode token (GFLOP) vs context length", { size: 13 });
          G.axes(60, 60, 540, 300, {});
          G.label(600, 400, "context t (thousands of tokens)", { anchor: "end", size: 11 });
          const max = 85, X = (i) => 80 + i * 66, Y = (v) => 360 - (300 * v) / max;
          const bars = [];
          ts.forEach((t, i) => {
            const a = aF(t);
            bars.push(G.rect(X(i), Y(wF), 30, 360 - Y(wF), { fill: "w", rx: 2 }));
            bars.push(G.rect(X(i), Y(wF + a), 30, Y(wF) - Y(wF + a), { fill: "q", rx: 2 }));
            G.label(X(i) + 15, 378, `${t}k`, { anchor: "middle", size: 11 });
          });
          [0, 20, 40, 60, 80].forEach((v) => G.label(52, Y(v) + 4, String(v), { anchor: "end", size: 10 }));
          G.line(80, Y(wF), 590, Y(wF), { color: "muted", dash: "4 4" });
          G.rect(320, 74, 12, 12, { fill: "w", rx: 2 }); G.label(338, 84, "weights (2P), fixed 15.0", { size: 11 });
          G.rect(320, 94, 12, 12, { fill: "q", rx: 2 }); G.label(338, 104, "attention, 0.52 × t(k)", { size: 11 });
          G.label(320, 128, "equal at t ≈ 28.6k", { size: 12, color: "hot" });
          G.from(bars, { opacity: 0, stagger: 0.05, duration: 0.25 });
          G.caption("attention grows with context; the weight cost does not");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A forward pass is: embedding lookup → 32 × (RMSNorm, attention, add; RMSNorm, MLP, add) → final norm → LM head.</li>
<li>Attention splits into 32 query heads of 128; with GQA, 4 query heads share each of 8 key/value heads, which cuts K/V storage by 4× but not attention FLOPs.</li>
<li>RoPE encodes position by rotation, so attention scores depend only on relative offsets.</li>
<li>Llama-3-8B has 8.03 B parameters, 81% of each layer in the MLP. One token costs ≈ 2 × P FLOPs and, for one user, reads every weight once: about 1 FLOP per byte.</li>
<li>Attention adds 4 · d · L FLOPs per past token, overtaking the weights near 28.6k tokens of context.</li></ul>`,
    sim: {
      title: "Count a model",
      intro: "Pick a model and a context length. The chart shows where each decode token's FLOPs go, at context lengths from 1k to 128k; the highlighted column is your choice. The panels give the parameter breakdown and the bytes and FLOPs for one token at batch 1. Configs are the repo's UNVERIFIED presets until exercise 1 checks them.",
      controls: [
        { id: "m", label: "model", type: "select", value: "llama8", options: [["llama8", "Llama-3-8B"], ["llama70", "Llama-3-70B"], ["mistral", "Mistral-7B"], ["tiny", "tiny test model (d = 64)"]] },
        { id: "kv", label: "key/value heads (as a fraction of query heads)", type: "select", value: "cfg", options: [["cfg", "as in the config (GQA)"], ["mha", "MHA: one per query head"], ["mqa", "MQA: one in total"]] },
        { id: "t", label: "context length t (tokens)", min: 512, max: 131072, step: 512, value: 8192 },
        { id: "b", label: "bytes per weight", type: "select", value: 2, options: [[2, "bf16 / fp16 (2 bytes)"], [1, "fp8 / int8 (1 byte)"], [0.5, "int4 (0.5 byte)"]] },
      ],
      draw(G, v) {
        const base = MODELS[v.m];
        const m = Object.assign({}, base, { kv: v.kv === "mha" ? base.H : v.kv === "mqa" ? 1 : base.kv });
        const g = count(m), h = m.d / m.H, attnPer = 4 * m.H * h * m.L;
        const wF = 2 * g.matrices, ctxs = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
        const all = ctxs.concat([v.t]), max = Math.max(...all.map((t) => wF + attnPer * t));
        G.label(10, 16, "FLOPs per decode token at each context length", { size: 11 });
        G.rect(380, 7, 10, 10, { fill: "w", rx: 2 }); G.label(394, 16, "weights 2P", { size: 11 });
        G.rect(480, 7, 10, 10, { fill: "q", rx: 2 }); G.label(494, 16, "attention", { size: 11 });
        G.axes(16, 30, 610, 190, {});
        const X = (i) => 34 + i * 64;
        const col = (i, t, hi) => {
          const hw = (185 * wF) / max, ha = (185 * attnPer * t) / max;
          G.rect(X(i), 220 - hw, 40, hw, { fill: "w", rx: 2, opacity: hi ? 1 : 0.6 });
          G.rect(X(i), 220 - hw - ha, 40, ha, { fill: "q", rx: 2, opacity: hi ? 1 : 0.6 });
          G.text(X(i) + 20, 238, (t >= 1024 ? Math.round(t / 1024) + "k" : String(t)) + (hi ? " (yours)" : ""), { anchor: "middle", size: 11, color: hi ? "ink" : "muted" });
          if (hi) G.rect(X(i) - 4, 220 - hw - ha - 4, 48, hw + ha + 8, { stroke: "ok", sw: 2, rx: 4 });
        };
        ctxs.forEach((t, i) => col(i, t, false));
        col(8, v.t, true);
        const aF = attnPer * v.t, cross = wF / attnPer;
        const kvTok = 2 * m.L * m.kv * h * v.b;
        return [
          { title: "Parameters · " + m.name, rows: [["attention (q, k, v, o)", PN(g.attn)], ["MLP (gate, up, down)", PN(g.mlp)], ["embedding + LM head", PN(g.emb + g.head)], ["total P", PN(g.total)]],
            gauge: [[g.mlp / g.total, "v"], [g.attn / g.total, "q"], [(g.emb + g.head) / g.total, "k"]], gaugeText: `MLP ${(100 * g.mlp / g.total).toFixed(0)}% · attention ${(100 * g.attn / g.total).toFixed(0)}% · vocab ${(100 * (g.emb + g.head) / g.total).toFixed(0)}%` },
          { title: "FLOPs for one token at t = " + F.num(v.t), rows: [["weights, 2 × matrix params", F.si(wF) + "FLOP"], ["attention, 4·H·h·L·t", F.si(aF) + "FLOP"], ["attention share", (100 * aF / (wF + aF)).toFixed(1) + "%"], ["break-even context", F.num(cross) + " tokens"]] },
          { title: "Bytes for one token, batch 1", rows: [["weights read", (g.total * v.b / 1e9).toFixed(2) + " GB"], ["K/V stored per token", F.bytes(kvTok)], ["KV cache read at t", F.bytes(kvTok * v.t)], ["FLOPs per byte", F.num((wF + aF) / (g.total * v.b + kvTok * v.t), 2)]],
            html: `<p class="note">Counts, not measurements. Norms, RoPE and softmax are left out (well under 1% of FLOPs for real models). FLOPs per byte below about 100 means memory speed sets the pace on current GPUs; P1.4 makes that precise.</p>` },
        ];
      },
    },
    practice: {
      intro: "Run these from the repository root. Each exercise has a starter file and a test; <code>S2S_SOLUTIONS=1</code> runs the same test against the reference solution. Run all four at once with <code>uv run pytest course/P1-inference-fundamentals/P1.1-transformer-forward-pass/exercises</code>.",
      items: [
        { title: "Parameter count formula", tier: "T0 · easy", goal: "Write count_params(cfg) from the formula alone (handle tied embeddings) and match the summed tensor sizes of 4 random configs. Then verify the Llama-3-8B config values by hand.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.1-transformer-forward-pass/exercises/01-param-count" },
        { title: "FLOPs per token", tier: "T0 · easy", goal: "Write decode_flops(cfg, t): 2 × every matrix weight plus 4·H·h·t of attention per layer, checked against the instrumented counter at t = 1, 10, 100.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.1-transformer-forward-pass/exercises/02-flops" },
        { title: "GQA KV saving", tier: "T0 · easy", goal: "Compute K/V bytes per token and the saving vs MHA for MHA, GQA and MQA on a 32-head, 4096-d, 32-layer config.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.1-transformer-forward-pass/exercises/03-gqa" },
        { title: "RoPE's relative-position property", tier: "T0 · hard", goal: "Show numerically that the RoPE score depends only on n − m, and derive score_via_offset that never sees the absolute positions.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.1-transformer-forward-pass/exercises/04-rope-relative" },
      ],
      labs: [
        { label: "Example 01: the NumPy Llama with per-op FLOP and byte counters (<code>uv run python …/01_numpy_llama_counted.py</code>)", path: "course/P1-inference-fundamentals/P1.1-transformer-forward-pass/examples/01_numpy_llama_counted.py" },
        { label: "Example 02: logit-for-logit parity with Hugging Face, CPU (<code>uv run --extra torch python …/02_hf_parity.py</code>)", path: "course/P1-inference-fundamentals/P1.1-transformer-forward-pass/examples/02_hf_parity.py" },
        { label: "Example 03: parameters and FLOPs/token from a config (<code>… --preset llama3-8b</code>)", path: "course/P1-inference-fundamentals/P1.1-transformer-forward-pass/examples/03_flop_counter.py" },
        { label: "The reference forward pass", path: "platform/engine/v0/reference/llama_numpy.py" },
      ],
    },
  });
})();
