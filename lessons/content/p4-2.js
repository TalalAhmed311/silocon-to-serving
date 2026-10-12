/* P4.2 — Parallelism strategies: TP, PP, DP/FSDP, EP, and how to choose for serving. */
(function () {
  const F = S2S.fmt;
  const MOD = "course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies";
  const RC = ["k", "v", "q", "pink"];           // colour of rank 0..3

  /* ---- model of one decode step under tensor parallelism (ring all-reduce from P4.1) ---- */
  const MODELS = { l8: { name: "Llama-3-8B", L: 32, d: 4096, P: 8.0 }, l70: { name: "Llama-3-70B", L: 80, d: 8192, P: 70.6 } };
  function decode(m, tp, batch, bwGBs, linkGBs, alphaUs) {
    const weights = (m.P * 1e9 * 2) / tp;                                   // bf16 bytes read per GPU per step
    const tw = weights / (0.8 * bwGBs * 1e9);                                // 80% of the memory bandwidth, like D3's bw_util
    const S = batch * m.d * 2;                                               // one all-reduce: batch × hidden × 2 B
    const one = tp > 1 ? 2 * (tp - 1) * alphaUs * 1e-6 + ((2 * (tp - 1)) / tp) * S / (linkGBs * 1e9) : 0;
    const tc = 2 * m.L * one;                                                // 2 all-reduces per layer
    return { tw, tc, t: tw + tc, S, weights };
  }

  /* ---- port of parallel.simulate_pipeline: returns events + summary ---- */
  function pipe(p, m, schedule, tf = 1, tb = 2) {
    const order = [];
    for (let s = 0; s < p; s++) {
      if (schedule === "gpipe") {
        order.push([...Array(m)].map((_, j) => ["F", j]).concat([...Array(m)].map((_, j) => ["B", m - 1 - j])));
      } else {
        const warm = Math.min(p - s - 1, m), o = [...Array(warm)].map((_, j) => ["F", j]);
        let f = warm, b = 0;
        while (b < m) { if (f < m) { o.push(["F", f]); f++; } o.push(["B", b]); b++; }
        order.push(o);
      }
    }
    const fin_f = [...Array(p)].map(() => Array(m).fill(0)), fin_b = [...Array(p)].map(() => Array(m).fill(0));
    const done_f = [...Array(p)].map(() => Array(m).fill(false)), done_b = [...Array(p)].map(() => Array(m).fill(false));
    const free = Array(p).fill(0), idx = Array(p).fill(0), infl = Array(p).fill(0), ev = [];
    let peak = 0, remaining = order.reduce((a, o) => a + o.length, 0);
    while (remaining) {
      let prog = false;
      for (let s = 0; s < p; s++) {
        if (idx[s] >= order[s].length) continue;
        const [kind, j] = order[s][idx[s]];
        if (kind === "F") {
          if (s > 0 && !done_f[s - 1][j]) continue;
          const start = Math.max(free[s], s > 0 ? fin_f[s - 1][j] : 0);
          fin_f[s][j] = free[s] = start + tf; done_f[s][j] = true; infl[s]++; peak = Math.max(peak, infl[s]);
          ev.push({ s, kind, j, start, end: start + tf });
        } else {
          if ((s < p - 1 && !done_b[s + 1][j]) || !done_f[s][j]) continue;
          const start = Math.max(free[s], s < p - 1 ? fin_b[s + 1][j] : fin_f[s][j]);
          fin_b[s][j] = free[s] = start + tb; done_b[s][j] = true; infl[s]--;
          ev.push({ s, kind, j, start, end: start + tb });
        }
        idx[s]++; remaining--; prog = true;
      }
      if (!prog) throw new Error("deadlock");
    }
    const makespan = Math.max(...free), busy = m * (tf + tb);
    return { ev, makespan, bubble: 1 - busy / makespan, peak };
  }
  const bubbleF = (p, m) => (p - 1) / (m + p - 1);

  /* gantt chart of a pipeline schedule */
  function gantt(G, r, p, x0, y0, w, rowH, size) {
    const sc = w / r.makespan, out = [];
    for (let s = 0; s < p; s++) {
      G.rect(x0, y0 + s * (rowH + 8), w, rowH, { fill: "w", opacity: 0.12, rx: 4 });
      G.label(x0 - 8, y0 + s * (rowH + 8) + rowH / 2 + 4, `GPU ${s}`, { anchor: "end", size: 11 });
    }
    r.ev.forEach((e) => {
      const g = G.box(x0 + e.start * sc + 0.5, y0 + e.s * (rowH + 8) + 1, (e.end - e.start) * sc - 1, rowH - 2, String(e.j + 1),
        { fill: e.kind === "F" ? "k" : "v", size, rx: 3 });
      out.push(g);
    });
    return out;
  }

  /* ---- little 4×4 weight pictures for step 1 ---- */
  function miniCut(G, x, y, kind, title) {
    const out = [];
    if (kind === "tp") { for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out.push(G.rect(x + c * 28, y + r * 18, 26, 16, { fill: RC[c], rx: 3 })); }
    if (kind === "pp") { for (let r = 0; r < 4; r++) out.push(G.rect(x, y + r * 18, 110, 16, { fill: RC[r], rx: 3 })); }
    if (kind === "dp") {
      for (let r = 0; r < 4; r++) { out.push(G.rect(x + r * 28, y, 26, 40, { fill: "w", rx: 3 })); out.push(G.rect(x + r * 28, y + 46, 26, 14, { fill: RC[r], rx: 3 })); }
      G.label(x, y + 76, "batch slices", { size: 10 });
    }
    if (kind === "ep") { for (let r = 0; r < 4; r++) out.push(G.box(x + r * 28, y + 10, 26, 46, "E" + r, { fill: RC[r], size: 10, rx: 3 })); }
    G.text(x, y - 8, title, { size: 12 });
    return out;
  }

  S2S.lesson({
    id: "p4-2", n: "P4.2", title: "Parallelism strategies",
    subtitle: "Multi-GPU and distributed · first principles · T0 simulators, T3 for vLLM and FSDP2",
    kicker: "Lesson · ≈ 50 min",
    headline: "Four ways to cut a model in pieces",
    intro: `<p>One GPU is not always enough. The model may not fit, or one GPU may be too slow. This lesson builds the four ways to share a model among GPUs (<b>tensor</b>, <b>pipeline</b>, <b>data</b> and <b>expert</b> parallelism), shows what each costs in memory and communication using the α–β model from P4.1, and ends with the question that matters for serving: when is splitting worth it?</p>`,
    facts: ["11 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "rank 0"], ["v", "rank 1"], ["q", "rank 2"], ["pink", "rank 3"], ["ok", "summed result"]],
    prev: "p4-1", next: "p4-3",
    steps: [
      { rail: "why split", title: "Two reasons to use more than one GPU",
        body: `<p><b>It does not fit.</b> A model's weights take (parameters × bytes per parameter). In bf16 each parameter is 2 bytes.</p>
<div class="eq">Llama-3-8B    8.0 B × 2 B =  16 GB
Llama-3-70B  70.6 B × 2 B = 141.2 GB
(70B does not fit on one 80 GB GPU)</div>
<p>(80 GB is an example GPU size; check the spec sheet for yours.) Splitting 70B over 2 such GPUs leaves 70.6 GB of weights on each, almost no room for the KV cache (P1.2). Over 4 GPUs it is 35.3 GB each.</p>
<p><b>It is too slow.</b> Decode is memory-bound (P1.4): each token needs one read of all the weights. If four GPUs each read a quarter of the weights at the same moment, the read could take a quarter of the time.</p>
<p>There are four natural ways to cut the work. Each answers "what does each GPU hold, and what must it tell the others?"</p>
<ul><li><b>Tensor parallel (TP):</b> cut every weight matrix; each GPU does part of each layer.</li>
<li><b>Pipeline parallel (PP):</b> cut by layer; each GPU runs a slice of the layers.</li>
<li><b>Data parallel (DP):</b> copy the model; each GPU handles different inputs.</li>
<li><b>Expert parallel (EP):</b> for mixture-of-experts models, each GPU holds some experts.</li></ul>
<p>The next steps take them in this order.</p>`,
        scene(G) {
          const vals = [["8B", 16, "k"], ["70B", 141.2, "v"]];
          G.text(24, 36, "weights in bf16 (GB)", { size: 13 });
          const sc = 220 / 150, y0 = 330;
          vals.forEach(([n, v, c], i) => {
            G.rect(40 + i * 100, y0 - v * sc, 70, v * sc, { fill: c, rx: 4 });
            G.text(75 + i * 100, y0 + 18, n, { anchor: "middle", size: 12 });
            G.text(75 + i * 100, y0 - v * sc - 8, String(v), { anchor: "middle", size: 12 });
          });
          G.line(24, y0 - 80 * sc, 250, y0 - 80 * sc, { color: "hot", dash: "5 4" });
          G.label(24, y0 - 80 * sc - 6, "80 GB GPU", { color: "hot" });
          G.line(24, y0, 250, y0, { color: "line" });
          G.label(24, 372, "example GPU size, check the spec sheet", { size: 11 });
          const a = miniCut(G, 310, 70, "tp", "tensor"), b = miniCut(G, 470, 70, "pp", "pipeline");
          const c = miniCut(G, 310, 210, "dp", "data"), d = miniCut(G, 470, 210, "ep", "expert");
          G.label(310, 160, "cut every matrix", { size: 11 }); G.label(470, 160, "cut by layers", { size: 11 });
          G.label(310, 330, "copy, split inputs", { size: 11 }); G.label(470, 330, "cut by experts", { size: 11 });
          G.from([...a, ...b, ...c, ...d], { opacity: 0, stagger: 0.02, duration: 0.3 });
          G.caption("four cuts: matrices, layers, inputs, experts");
        } },

      { rail: "split a matmul", title: "A matrix product can be cut two ways",
        body: `<p>Almost all of a transformer's work is <b>y = x W</b>: a vector x times a weight matrix W. Take the smallest example, to see the two ways to cut W across two GPUs.</p>
<div class="eq">x = [1, 2]     W = [[1, 2],
               [3, 4]]
y = x W = [1·1 + 2·3,  1·2 + 2·4] = [7, 10]</div>
<p><b>Column split.</b> Each GPU gets some columns of W and the whole x. GPU 0 owns column [1, 3] and computes 1·1 + 2·3 = 7. GPU 1 owns column [2, 4] and computes 1·2 + 2·4 = 10. Each result is a finished piece of y. <b>Placing the pieces side by side is the answer: no adding, no communication.</b></p>
<p><b>Row split.</b> Each GPU gets some rows of W and the matching part of x. GPU 0 has x₀ = 1 and row [1, 2], giving [1, 2]. GPU 1 has x₁ = 2 and row [3, 4], giving [6, 8]. Each result has the full shape of y but is only a <b>partial sum</b>. The answer is [1, 2] + [6, 8] = [7, 10]: an <b>all-reduce</b> (P4.1).</p>
<p>The rule to remember: a column split gives <i>pieces</i>, a row split gives <i>partial sums</i> that need adding.</p>`,
        scene(G) {
          const half = (x0, title, mode) => {
            G.text(x0, 40, title, { size: 14 });
            G.text(x0, 76, "x", { size: 13, color: "muted" });
            if (mode === "col") { G.box(x0 + 24, 56, 34, 28, "1", { fill: "w", color: "ink" }); G.box(x0 + 62, 56, 34, 28, "2", { fill: "w", color: "ink" }); G.label(x0 + 104, 74, "on every GPU", { size: 11 }); }
            else { G.box(x0 + 24, 56, 34, 28, "1", { fill: "k" }); G.box(x0 + 62, 56, 34, 28, "2", { fill: "v" }); G.label(x0 + 104, 74, "one piece each", { size: 11 }); }
            const W = [[1, 2], [3, 4]];
            G.text(x0, 134, "W", { size: 13, color: "muted" });
            for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) G.box(x0 + 24 + c * 38, 108 + r * 34, 34, 30, String(W[r][c]), { fill: RC[mode === "col" ? c : r] });
          };
          half(16, "split columns", "col"); half(332, "split rows", "row");
          // column results
          G.arrow(60, 184, 60, 214, { color: "k" }); G.arrow(98, 184, 98, 214, { color: "v" });
          G.box(40, 218, 40, 32, "7", { fill: "k" }); G.box(88, 218, 40, 32, "10", { fill: "v" });
          G.text(24, 276, "GPU 0 holds 7, GPU 1 holds 10", { size: 12 });
          G.text(24, 298, "[7, 10] = pieces side by side", { size: 13, color: "ok" });
          G.text(24, 320, "no communication", { size: 13, color: "ok" });
          // row results
          G.arrow(376, 184, 358, 214, { color: "k" }); G.arrow(414, 184, 458, 214, { color: "v" });
          G.row(340, 218, ["1", "2"], { w: 34, h: 32, gap: 4, fill: "k" }); G.row(440, 218, ["6", "8"], { w: 34, h: 32, gap: 4, fill: "v" });
          G.text(332, 276, "each GPU holds a full-size partial", { size: 12 });
          G.text(332, 298, "[1,2] + [6,8] = [7,10]", { size: 13, color: "hot" });
          G.text(332, 320, "needs an all-reduce", { size: 13, color: "hot" });
          G.line(320, 40, 320, 340, { color: "line", dash: "3 4" });
          G.label(24, 380, "rule: column split gives pieces · row split gives partial sums", { size: 12, color: "ink" });
          G.caption("a column split needs no communication; a row split needs a sum");
        } },

      { rail: "tensor: MLP", title: "Column then row: one all-reduce per MLP",
        body: `<p>Llama's MLP block is: y = ( silu(x W<sub>gate</sub>) ⊙ (x W<sub>up</sub>) ) W<sub>down</sub>. (silu is a smooth curve applied to every number separately; ⊙ multiplies element by element.) In Llama-3-8B, x has d = 4,096 numbers, the middle layer f = 14,336.</p>
<p>The <b>Megatron-style</b> split uses the rule from the last step twice:</p>
<ol><li><b>Column-split W<sub>gate</sub> and W<sub>up</sub>.</b> With 4 GPUs each gets 14,336 / 4 = 3,584 of the middle units, plus the whole x. Each GPU produces a finished slice of the middle layer. silu and ⊙ work number by number, so every GPU applies them to its own slice. <b>No communication.</b></li>
<li><b>Row-split W<sub>down</sub></b> with the matching 3,584 rows. Each GPU turns its slice into a full-size partial y.</li>
<li><b>One all-reduce</b> sums the four partial y's.</li></ol>
<p>Why not row-split first? Because silu is not linear: silu of a sum is not the sum of silu's. With ReLU (max(0, ·)) as a stand-in: ReLU(3 + (−2)) = 1, but ReLU(3) + ReLU(−2) = 3. Summing before silu would force a communication in the middle.</p>
<div class="eq">per GPU at tp = 4: 3 × 4096 × 3584 = 44.0 M params
                   = 88.1 MB in bf16 (352 MB / 4)</div>`,
        check: { q: "Why is the first MLP matrix column-split and the second one row-split?",
          options: ["It makes the matrices fit in the cache", "Column pieces can go through silu independently; the row split then yields partial sums that one all-reduce adds", "Row splits are faster on tensor cores", "It balances the load between ranks"], answer: 1,
          why: "A column split gives finished slices of the middle layer, and silu works per number, so it needs no data from other GPUs. The row split of the next matrix consumes exactly those slices and produces partial sums, which one all-reduce adds. The other order would need an all-reduce before silu." },
        scene(G) {
          const hdr = (x, s) => G.label(x, 40, s, { size: 11 });
          hdr(16, "x"); hdr(66, "gate/up columns"); hdr(196, "silu ⊙"); hdr(262, "down rows"); hdr(388, "partial y"); hdr(520, "y");
          const parts = [];
          for (let r = 0; r < 4; r++) {
            const y = 54 + r * 72;
            G.box(16, y + 6, 34, 42, "x", { fill: "w", color: "ink" });
            G.arrow(52, y + 27, 64, y + 27, { color: "muted" });
            G.box(66, y, 120, 54, `cols ${r * 3584}…${(r + 1) * 3584 - 1}`, { fill: RC[r], size: 10 });
            G.arrow(188, y + 27, 196, y + 27, { color: "muted" });
            G.box(196, y + 10, 54, 34, "local", { stroke: "line", size: 11, color: "muted" });
            G.arrow(252, y + 27, 260, y + 27, { color: "muted" });
            G.box(262, y, 120, 54, `rows ${r * 3584}…${(r + 1) * 3584 - 1}`, { fill: RC[r], size: 10 });
            G.arrow(384, y + 27, 392, y + 27, { color: "muted" });
            parts.push(G.box(392, y + 6, 100, 42, "partial y", { fill: RC[r], size: 11 }));
            G.line(494, y + 27, 508, 54 + 4 * 72 / 2 - 6, { color: RC[r], w: 2 });
          }
          const y = G.box(512, 118, 100, 74, "Σ = y", { fill: "ok", size: 14 });
          G.text(562, 214, "all-reduce", { anchor: "middle", size: 13, color: "hot" });
          G.text(562, 232, "(the only one)", { anchor: "middle", size: 11, color: "muted" });
          G.label(16, 354, "4 GPUs · each holds 3,584 of the 14,336 middle units", { size: 12, color: "ink" });
          G.label(16, 376, "x is the same on every GPU · the middle step needs no talk", { size: 12, color: "ink" });
          G.from(parts, { opacity: 0, x: -16, stagger: 0.12, duration: 0.4 }); G.pulse(y, { repeat: 4 });
          G.caption("four partial results, one sum");
        } },

      { rail: "tensor: attention", title: "Attention splits by heads: two all-reduces per layer",
        body: `<p>Attention already has independent parts: <b>heads</b>. Each head has its own slice of W<sub>q</sub>, W<sub>k</sub>, W<sub>v</sub> and computes attention on its own. So the same trick applies:</p>
<ul><li>Give each GPU a group of heads: the W<sub>q</sub>, W<sub>k</sub>, W<sub>v</sub> <b>columns</b> of those heads. Llama-3-8B has 32 query heads and 8 KV heads (P1.2), so with 4 GPUs each owns 8 query heads and 2 KV heads, and <b>stores the KV cache of those 2 heads only</b>.</li>
<li>Attention for a head needs no other head. No communication.</li>
<li>Split W<sub>o</sub> (the output matrix) by <b>rows</b>, matching the heads. Each GPU outputs a full-size partial; an all-reduce adds them.</li></ul>
<p>Add the MLP's all-reduce from the last step and a layer has <b>2 all-reduces</b>; Llama-3-8B (32 layers) has 64 per token. Between the two all-reduces everything is local.</p>
<p>What if there are fewer KV heads than GPUs, say 8 KV heads on 16 GPUs? A KV head cannot be cut, so two GPUs hold a copy of each and the cache is duplicated (vLLM does this). That is one reason to keep TP small.</p>`,
        scene(G) {
          G.text(24, 36, "32 query heads over 4 GPUs", { size: 13 });
          for (let r = 0; r < 4; r++) {
            for (let h = 0; h < 8; h++) G.rect(24 + h * 24, 54 + r * 38, 22, 28, { fill: RC[r], rx: 4 });
            G.label(222, 74 + r * 38, `GPU ${r}: heads ${r * 8}–${r * 8 + 7} · KV heads ${2 * r},${2 * r + 1}`, { size: 11 });
          }
          G.label(24, 214, "each GPU stores the KV cache of its own 2 KV heads", { size: 12, color: "ink" });
          const b1 = G.box(24, 244, 260, 40, "attention (heads local) → W_o rows", { fill: "blue", size: 12 });
          const a1 = G.box(300, 244, 150, 40, "all-reduce #1", { fill: "hot", size: 13 });
          const b2 = G.box(24, 304, 260, 40, "MLP (cols, then rows)", { fill: "blue", size: 12 });
          const a2 = G.box(300, 304, 150, 40, "all-reduce #2", { fill: "hot", size: 13 });
          G.arrow(286, 264, 298, 264, { color: "muted" }); G.arrow(286, 324, 298, 324, { color: "muted" });
          G.arrow(374, 286, 374, 302, { color: "muted" });
          G.text(470, 268, "× 32 layers", { size: 13 }); G.text(470, 328, "= 64 per token", { size: 13, color: "hot" });
          G.label(24, 388, "everything between the all-reduces is local to one GPU", { size: 12, color: "ink" });
          G.from([b1, a1, b2, a2], { opacity: 0, y: 10, stagger: 0.15, duration: 0.4 });
          G.caption("heads are free; the sum after W_o and W_down is the price");
        }, legend: [["blue", "local compute"], ["hot", "all-reduce"]] },

      { rail: "TP cost", title: "What TP costs: 64 small all-reduces per token",
        body: `<p>Tensor parallelism cuts the weight bytes each GPU reads per token by tp. But each token now waits for 2 × layers all-reduces. Price them with P4.1's ring formula, T = 2(n−1)α + 2(n−1)/n · S/B.</p>
<p>For Llama-3-8B at batch 1, S = 4,096 × 2 B = 8 KiB (about 8,192 bytes). Use these <b>example values</b> (not measurements): GPU memory bandwidth 300 GB/s at 80% efficiency, link B = 25 GB/s, α = 10 µs.</p>
<div class="eq">tp = 1  weights 16 GB / 240 GB/s          = 66.7 ms
tp = 2  weights 33.3 ms
        one all-reduce: 20 µs + 0.33 µs = 20.3 µs
        64 of them = 1.30 ms  →  total 34.6 ms
tp = 4  weights 16.7 ms
        one all-reduce: 60 µs + 0.49 µs = 60.5 µs
        64 of them = 3.87 ms  →  total 20.5 ms
tp = 8  weights 8.3 ms
        one all-reduce: 140 µs + 0.57 µs = 140.6 µs
        64 of them = 9.00 ms  →  total 17.3 ms</div>
<p>Weight time halves each time; communication time more than doubles, because the 2(n−1)α term grows with n while the bytes hardly matter. The speed-up from tp = 1 to 4 is 66.7 / 20.5 = 3.2×, not 4×; from 4 to 8 only 1.2×. On a faster-memory GPU the weight bars shrink and the red bars decide. Try it in the simulator. This is the same code path as <code>tp_comm_seconds_per_step</code> in <code>platform/capacity/core.py</code>.</p>`,
        check: { q: "At batch 1 on 8 GPUs, each all-reduce is only 8 KiB, yet TP = 8 is barely faster than TP = 4. Why?",
          options: ["8 KiB is too large for the link", "Each all-reduce is dominated by latency, and the ring has 14 steps at n = 8 versus 6 at n = 4", "The weights stop shrinking at tp = 4", "NCCL cannot run on 8 GPUs"], answer: 1,
          why: "The transfer term for 8 KiB is under a microsecond; the cost is α per ring step. A ring over 8 GPUs has 2(n−1) = 14 steps (against 6 for n = 4), so the 64 all-reduces get about 2.3× more expensive while the weight read only halves." },
        scene(G) {
          const m = MODELS.l8, tps = [1, 2, 4, 8];
          const rs = tps.map((tp) => decode(m, tp, 1, 300, 25, 10));
          G.text(24, 34, "decode time per token (ms), Llama-3-8B, batch 1", { size: 13 });
          const base = 340, sc = 250 / 70;
          rs.forEach((r, i) => {
            const x = 60 + i * 140;
            G.rect(x, base - r.tw * 1e3 * sc, 80, r.tw * 1e3 * sc, { fill: "k", rx: 3 });
            G.rect(x, base - (r.tw + r.tc) * 1e3 * sc, 80, Math.max(1, r.tc * 1e3 * sc), { fill: "hot", rx: 3 });
            G.text(x + 40, base - r.t * 1e3 * sc - 8, (r.t * 1e3).toFixed(1) + " ms", { anchor: "middle", size: 13 });
            G.text(x + 40, base + 20, "tp = " + tps[i], { anchor: "middle", size: 13 });
            if (r.tc > 0) G.label(x + 40, base - r.tw * 1e3 * sc / 2 + 4, r.tw * 1e3 < 12 ? "" : (r.tw * 1e3).toFixed(1), { anchor: "middle", size: 11, color: "bg" });
          });
          G.line(40, base, 620, base, { color: "line" });
          G.label(60, 384, "comm per token: tp 2 = 1.30 ms · tp 4 = 3.87 ms · tp 8 = 9.00 ms", { size: 12, color: "ink" });
          G.label(60, 404, "example values: 300 GB/s memory, link 25 GB/s, α = 10 µs (not measured)", { size: 11 });
          G.from(G.svg.querySelectorAll("rect"), { opacity: 0, duration: 0.4, stagger: 0.02 });
          G.caption("weight reads shrink with tp; all-reduce latency grows with tp");
        }, legend: [["k", "read weights"], ["hot", "64 all-reduces"]] },

      { rail: "pipeline", title: "Pipeline: cut by layers, and fight the bubble",
        body: `<p>A different cut: give GPU 0 the first quarter of the layers, GPU 1 the second quarter, and so on. These are <b>stages</b>. Only the activations at stage boundaries cross a link, and only to the next stage: a single point-to-point send, not an all-reduce. This tolerates slow links (even between machines).</p>
<p>Problem: one input flows through the stages one at a time, so three of four GPUs sit idle. The fix is to split the batch into <b>microbatches</b> and keep several in flight, like cars on an assembly line.</p>
<p>The picture shows <b>GPipe</b> for training: all forwards (cyan), then all backwards (amber, twice as long as a forward here). The empty slots are the <b>bubble</b>: stage s waits for its first microbatch, and waits again at the end.</p>
<div class="eq">p = number of stages,  m = number of microbatches
bubble = (p − 1) / (m + p − 1)

p = 4, m = 4:   3 / 7   = 42.9% idle
p = 4, m = 12:  3 / 15  = 20%
p = 8, m = 8:   7 / 15  = 46.7%
under 10% at p = 8 needs  7/(m+7) &lt; 0.1  →  m &gt; 63</div>
<p>The bubble is small only when m ≫ p. The simulator behind the picture is <code>simulate_pipeline</code> in <code>${MOD}/parallel.py</code>; its test checks the formula.</p>`,
        check: { q: "A pipeline has p = 4 stages and m = 12 microbatches. What fraction of time is the bubble?",
          options: ["25%", "20%", "33%", "8%"], answer: 1,
          why: "(p − 1) / (m + p − 1) = 3 / 15 = 0.20. Note that more microbatches shrink the bubble, so the usual fix is a bigger m, as long as each microbatch still keeps the GPU busy." },
        scene(G) {
          const r = pipe(4, 4, "gpipe");
          G.text(24, 34, "GPipe · 4 stages · 4 microbatches", { size: 14 });
          const ev = gantt(G, r, 4, 70, 56, 550, 44, 12);
          G.text(70, 280, `idle = ${(r.bubble * 100).toFixed(1)}%`, { size: 15, color: "hot" });
          G.label(70, 302, `= (4 − 1) / (4 + 4 − 1) = 3 / 7 · makespan ${r.makespan} units, busy ${4 * 3} per GPU`, { size: 12, color: "ink" });
          G.label(70, 326, "numbers inside the boxes are microbatch ids", { size: 12 });
          G.label(70, 348, "stage s cannot start until stage s−1 finishes that microbatch", { size: 12 });
          G.from(ev, { opacity: 0, stagger: 0.015, duration: 0.25 });
          G.caption("empty slots at the corners are the bubble");
        }, legend: [["k", "forward"], ["v", "backward"], ["w", "idle (empty)"]] },

      { rail: "1F1B", title: "1F1B: same bubble, far less memory",
        body: `<p>Training needs each forward's activations kept until its backward has run. In GPipe, stage 0 finishes <i>all</i> m forwards first, so it holds m microbatches of activations at once.</p>
<p><b>1F1B</b> (one forward, one backward) reorders the work: after a short warm-up, each stage alternates a forward and a backward, so a microbatch's activations are freed soon after they are made. At most p microbatches are in flight at any stage.</p>
<div class="eq">p = 4, m = 8
GPipe: peak in flight = 8 microbatches
1F1B:  peak in flight = 4
bubble (both) = 3 / 11 = 27.3%</div>
<p>1F1B does <b>not</b> shrink the bubble; it saves memory, which lets you raise m (and so cut the bubble) for free. Interleaving several small stages per GPU is what cuts the bubble itself, by about the number of interleaved stages. (Exercise 2 asks you to confirm it.)</p>
<p>For <b>serving</b>, PP also adds one hop to every token's latency per stage and keeps GPUs busy only when many requests are in flight. Use it to fit a model into more GPUs than one node's TP group offers, not to make one request faster.</p>`,
        scene(G) {
          const r = pipe(4, 8, "1f1b"), g = pipe(4, 8, "gpipe");
          G.text(24, 34, "1F1B · 4 stages · 8 microbatches", { size: 14 });
          const ev = gantt(G, r, 4, 70, 52, 550, 40, 9);
          G.text(24, 270, "peak microbatches in flight on a stage", { size: 13 });
          G.bars(70, 356, [g.peak, r.peak], { w: 120, gap: 40, h: 60, max: 8, fill: (i) => (i === 0 ? "hot" : "ok"), rx: 3 });
          G.text(130, 356 - 60 * g.peak / 8 - 6, `GPipe: ${g.peak}`, { anchor: "middle", size: 13 });
          G.text(290, 356 - 60 * r.peak / 8 - 6, `1F1B: ${r.peak}`, { anchor: "middle", size: 13 });
          G.label(70, 378, `bubble: GPipe ${(g.bubble * 100).toFixed(1)}% · 1F1B ${(r.bubble * 100).toFixed(1)}% (formula 3/11)`, { size: 12, color: "ink" });
          G.label(70, 398, "after the warm-up each GPU alternates forward and backward", { size: 12 });
          G.from(ev, { opacity: 0, stagger: 0.01, duration: 0.2 });
          G.caption("same idle time, half the stored activations");
        }, legend: [["k", "forward"], ["v", "backward"], ["ok", "1F1B in flight"], ["hot", "GPipe in flight"]] },

      { rail: "data + ZeRO", title: "Data parallelism and the 16 bytes per parameter",
        body: `<p>Data parallelism (DP) keeps a full copy of the model on every GPU and gives each a different slice of the batch. After the backward pass, an all-reduce averages the gradients (P4.1 step 1) so the copies stay equal. Cheap to use, but every GPU holds everything.</p>
<p>How much is "everything" for training with the Adam optimizer in mixed precision? Per parameter:</p>
<div class="eq">bf16 weight            2 B
bf16 gradient          2 B
fp32 master weight     4 B
Adam m (momentum)      4 B     } optimizer state:
Adam v (variance)      4 B     } 12 B
                      ------
                       16 B per parameter</div>
<p>For Llama-3-8B that is 8.0 B × 16 B = <b>128 GB per GPU</b> for plain DDP, before any activations. It does not fit on an 80 GB GPU (example size). But 8 copies of the same optimizer state is waste. <b>ZeRO</b> shards it across the n data-parallel GPUs, in three stages:</p>
<table><tr><th>strategy</th><th>B / param / GPU</th><th>n = 8, 8B model</th></tr>
<tr><td>DDP</td><td>16</td><td>128 GB</td></tr>
<tr><td>ZeRO-1 (shard optimizer)</td><td>4 + 12/n</td><td>44 GB</td></tr>
<tr><td>ZeRO-2 (+ gradients)</td><td>2 + 14/n</td><td>30 GB</td></tr>
<tr><td>ZeRO-3 / FSDP (+ weights)</td><td>16/n</td><td>16 GB</td></tr></table>
<p>These are the formulas of <code>training_bytes_per_param</code> in <code>parallel.py</code>. Activations and temporary buffers come on top.</p>`,
        check: { q: "Llama-3-8B (8.0 B parameters) is trained with mixed-precision Adam, fully sharded (FSDP) over 8 GPUs. Roughly how much do weights, gradients and optimizer state take on each GPU?",
          options: ["128 GB", "44 GB", "16 GB", "2 GB"], answer: 2,
          why: "Fully sharded, every one of the 16 bytes per parameter is divided by n: 16 / 8 = 2 B per parameter, and 8.0 B × 2 B = 16 GB. 128 GB is plain DDP; 44 GB is ZeRO-1. Activations are extra." },
        scene(G) {
          const P = 8.0, n = 8;
          const rows = [["DDP", 2, 2, 12], ["ZeRO-1", 2, 2, 12 / n], ["ZeRO-2", 2, 2 / n, 12 / n], ["FSDP", 2 / n, 2 / n, 12 / n]];
          const sc = 230 / 130, y0 = 340;
          G.text(24, 34, "GB per GPU, Llama-3-8B, n = 8 (without activations)", { size: 13 });
          rows.forEach((r, i) => {
            const x = 50 + i * 140; let y = y0;
            [["k", r[1]], ["v", r[2]], ["q", r[3]]].forEach(([c, b]) => {
              const h = b * P * sc; y -= h; G.rect(x, y, 90, h, { fill: c, rx: 2 });
              if (h > 16 && Math.abs(y + h / 2 - (y0 - 80 * sc)) > 12) G.text(x + 45, y + h / 2 + 4, (b * P).toFixed(b * P % 1 ? 1 : 0), { anchor: "middle", size: 11, color: "bg" });
            });
            G.text(x + 45, y - 8, (r[1] + r[2] + r[3]).toFixed(2).replace(/\.?0+$/, "") === "" ? "" : ((r[1] + r[2] + r[3]) * P).toFixed(0) + " GB", { anchor: "middle", size: 13 });
            G.text(x + 45, y0 + 20, rows[i][0], { anchor: "middle", size: 12 });
          });
          G.line(30, y0, 620, y0, { color: "line" });
          G.line(30, y0 - 80 * sc, 620, y0 - 80 * sc, { color: "hot", dash: "5 4" });
          G.label(620, y0 - 80 * sc - 6, "80 GB GPU (example size)", { anchor: "end", color: "hot" });
          G.label(50, 388, "per parameter: weights 2 B + grads 2 B + optimizer 12 B", { size: 12, color: "ink" });
          G.from(G.svg.querySelectorAll("rect"), { opacity: 0, duration: 0.4, stagger: 0.02 });
          G.caption("sharding removes the copies, one kind of state at a time");
        }, legend: [["k", "weights"], ["v", "gradients"], ["q", "optimizer state"], ["hot", "GPU memory"]] },

      { rail: "FSDP flow", title: "FSDP: gather a layer just in time, then forget it",
        body: `<p>ZeRO-3, in PyTorch called <b>FSDP</b> (Fully Sharded Data Parallel), leaves a problem: to compute layer 5, a GPU needs all of layer 5's weights, but holds only a quarter. The answer is to <b>gather and discard</b>, layer by layer:</p>
<ol><li>At rest, each GPU holds a 1/n shard of every layer.</li>
<li>Before a layer runs: <b>all-gather</b> its weights. Each GPU briefly holds the full layer, runs it, and frees the gathered copy. Only one layer (or a few) is ever full.</li>
<li>The backward pass gathers the layer again, computes gradients, and a <b>reduce-scatter</b> leaves each GPU the summed gradient for its own shard. (P4.1: a ring reduce-scatter and all-gather are the two halves of an all-reduce.)</li></ol>
<div class="eq">DDP:  all-reduce(grads)      2(n−1)/n × 16 GB
FSDP: all-gather, forward    (n−1)/n × 16 GB
      all-gather, backward   (n−1)/n × 16 GB
      reduce-scatter(grads)  (n−1)/n × 16 GB
n = 8, 16 GB of bf16 per pass:
  DDP sends 28 GB, FSDP 42 GB per GPU
  at an example 100 GB/s: 0.28 s vs 0.42 s</div>
<p>So FSDP moves about 1.5× the bytes of DDP, for 8× less state. Because layer k+1's gather can run while layer k computes, most of the extra time hides behind the compute. Try it with <code>examples/04_fsdp2_toy.py</code>.</p>`,
        scene(G) {
          const cols = [0, 1, 2, 3], x0 = 130, cw = 120;
          cols.forEach((r) => G.text(x0 + r * cw + 46, 40, `GPU ${r}`, { anchor: "middle", size: 13 }));
          const row = (y, title, sub, fn) => {
            G.text(16, y + 16, title, { size: 12 }); G.label(16, y + 32, sub, { size: 10 });
            const out = [];
            cols.forEach((r) => { for (let c = 0; c < 4; c++) { const f = fn(r, c); out.push(G.rect(x0 + r * cw + c * 23, y, 21, 36, f ? { fill: f, rx: 3 } : { stroke: "line", rx: 3, sw: 1 })); } });
            return out;
          };
          const a = row(70, "at rest", "1/4 of the layer", (r, c) => (c === r ? RC[r] : null));
          G.arrow(300, 114, 300, 144, { color: "hot", w: 2.5 }); G.text(312, 134, "all-gather (before the layer runs)", { size: 12, color: "hot" });
          const b = row(152, "layer runs", "full weights", (r, c) => RC[c]);
          G.arrow(300, 196, 300, 226, { color: "hot", w: 2.5 }); G.text(312, 216, "free the copy; backward gathers again", { size: 12, color: "hot" });
          const c3 = row(234, "after backward", "reduce-scatter", (r, c) => (c === r ? "ok" : null));
          G.text(16, 318, "reduce-scatter: GPU r keeps the summed gradient of shard r", { size: 12, color: "ink" });
          G.label(16, 344, "bytes per GPU, n = 8, 16 GB bf16:  DDP 28 GB · FSDP 42 GB", { size: 12 });
          G.label(16, 366, "layer k+1 is gathered while layer k computes, hiding most of it", { size: 12 });
          G.from([...a, ...b, ...c3], { opacity: 0, stagger: 0.01, duration: 0.25 });
          G.caption("hold a shard, borrow the rest, give it back");
        } },

      { rail: "experts", title: "Expert parallelism: send each token to its expert",
        body: `<p>A <b>mixture-of-experts</b> (MoE) layer replaces the one MLP with E separate MLPs (the <b>experts</b>). A small router picks, per token, the k experts (often 1–2) that process it. Only those run, so the model has many parameters but does little work per token.</p>
<p>Experts are many and large, so put different experts on different GPUs: <b>expert parallelism</b> (EP). Now a token on GPU 0 may need an expert on GPU 3. Two exchanges per MoE layer:</p>
<ol><li><b>Dispatch:</b> an <b>all-to-all</b> (P4.1) sends each token to the GPU of its expert.</li>
<li>Each expert processes the tokens it received.</li>
<li><b>Combine:</b> a second all-to-all returns the results to the tokens' home GPUs.</li></ol>
<div class="eq">example config (not a real model):
  d = 4096 (8 KiB per token in bf16)
  1,024 tokens per GPU, top-2 routing, 4 GPUs
sent per GPU in one all-to-all:
  1024 tokens × 2 experts × 8 KiB = 16.8 MB
  3/4 of it leaves the GPU = 12.6 MB
  (two all-to-alls per MoE layer)</div>
<p>A trouble specific to MoE is <b>balance</b>. If the router sends most tokens to one expert, that GPU is the slowest and everyone waits for it. That is why MoE training adds a load-balancing loss and why serving systems cap how many tokens one expert accepts.</p>`,
        scene(G) {
          const tok = [[0, 1, 2, 3], [1, 1, 3, 0], [2, 0, 3, 2], [3, 2, 1, 0]];     // destination expert/GPU of each token
          G.text(24, 26, "dispatch: all-to-all", { size: 13, color: "hot" });
          const xs = (r) => 90 + r * 140;
          for (let r = 0; r < 4; r++) {
            G.rect(xs(r) - 56, 52, 112, 56, { stroke: "line", rx: 8 });
            G.text(xs(r), 46, `GPU ${r}`, { anchor: "middle", size: 11, color: "muted" });
            tok[r].forEach((d, i) => G.box(xs(r) - 50 + i * 26, 64, 22, 30, "", { fill: RC[d], rx: 4 }));
          }
          const lines = [];
          for (let r = 0; r < 4; r++) tok[r].forEach((d, i) => lines.push(G.line(xs(r) - 39 + i * 26, 96, xs(d) - 30 + ((r * 4 + i) % 4) * 15, 190, { color: RC[d], w: 1.6, opacity: 0.85, arrow: true })));
          for (let r = 0; r < 4; r++) {
            G.box(xs(r) - 56, 194, 112, 50, `expert ${r}`, { fill: RC[r], size: 13 });
          }
          G.text(24, 276, "combine: a second all-to-all returns the results", { size: 13, color: "hot" });
          G.label(24, 304, "colour = the expert (GPU) a token is routed to", { size: 12 });
          G.label(24, 326, "all-to-all, 4 GPUs: each GPU sends 3/4 of its tokens away", { size: 12 });
          G.label(24, 354, "example config: 12.6 MB leaves each GPU per all-to-all", { size: 12, color: "ink" });
          G.label(24, 376, "slowest expert sets the pace: balance matters", { size: 12, color: "ink" });
          G.from(lines, { opacity: 0, stagger: 0.03, duration: 0.3 });
          G.caption("every GPU talks to every other GPU, twice per MoE layer");
        } },

      { rail: "choosing", title: "Choosing for serving: replicas first, TP when you must",
        body: `<p>Put the pieces together for <b>serving</b>. TP costs a latency-bound all-reduce pair per layer; PP adds hops and bubbles; DP (separate <b>replicas</b>, each a full model with its own requests) needs no communication at all. The default is therefore to avoid splitting until something forces it.</p>
<table><tr><th>situation</th><th>choice</th><th>why</th></tr>
<tr><td>fits on one GPU with KV room</td><td>replicas (TP = 1)</td><td>no communication; capacity grows with GPU count</td></tr>
<tr><td>fits, but per-token latency matters</td><td>TP = 2–4 on fast links</td><td>weights read per GPU drop by tp; all-reduces are cheap on NVLink</td></tr>
<tr><td>does not fit on one GPU</td><td>smallest TP that fits with KV room</td><td>every extra rank adds communication</td></tr>
<tr><td>does not fit in one node</td><td>TP in the node × PP across nodes</td><td>TP needs fast links; PP's sends tolerate slower ones</td></tr></table>
<p>Check with the step-5 numbers. Single-stream tok/s per GPU: tp = 1: 15.0; tp = 2: 28.9 / 2 = 14.4; tp = 4: 48.7 / 4 = 12.2 (example values). Each added GPU buys less than the last. TP is a tool for latency and for fitting, not a free multiplier.</p>
<p>One honest caveat: with many concurrent requests, TP can still raise total throughput, because a larger shared batch reads the weights once for more tokens, and the freed memory holds more KV cache. Which wins depends on your link and load. That is why the module's last exercise measures vLLM at TP = 1, 2 and 4 against this prediction. Pitfalls: TP across slow links or machines; judging TP only at batch 1; ignoring that TP also shards the KV cache.</p>`,
        check: { q: "An 8B model (16 GB in bf16) fits on a 24 GB GPU. You have four such GPUs linked by PCIe and want more serving capacity. What is the sensible first move?",
          options: ["TP = 4, so every GPU holds a quarter of the weights", "Four independent replicas behind a load balancer", "PP with 4 stages", "Expert parallelism"], answer: 1,
          why: "A replica needs no communication, and each request is served on one GPU. TP over PCIe would add 64 latency-bound all-reduces per token. TP becomes the right tool only if the model does not fit, or if single-request latency matters more than capacity. PP and EP address other problems (very large models; MoE)." },
        scene(G) {
          const rows = [["fits one GPU", "replicas", "k"], ["want lower latency", "TP 2–4, fast links", "v"], ["does not fit one GPU", "smallest TP that fits", "q"], ["does not fit a node", "TP in node × PP across", "pink"]];
          rows.forEach(([a, b, c], i) => {
            const y = 40 + i * 52;
            G.box(16, y, 250, 38, a, { stroke: "ink", size: 13 });
            G.arrow(270, y + 19, 304, y + 19, { color: "muted", w: 2 });
            G.box(308, y, 306, 38, b, { fill: c, size: 13 });
          });
          G.text(16, 270, "single-stream tok/s per GPU (example values)", { size: 13 });
          const m = MODELS.l8, y0 = 380, sc = 80 / 16, base = decode(m, 1, 1, 300, 25, 10);
          [1, 2, 4].forEach((tp, i) => {
            const r = decode(m, tp, 1, 300, 25, 10), v = 1 / r.t / tp;
            G.rect(60 + i * 140, y0 - v * sc, 80, v * sc, { fill: "k", rx: 3 });
            G.text(100 + i * 140, y0 - v * sc - 8, v.toFixed(1), { anchor: "middle", size: 13 });
            G.text(100 + i * 140, y0 + 18, "tp = " + tp, { anchor: "middle", size: 12 });
          });
          G.line(40, y0, 520, y0, { color: "line" });
          G.label(480, 330, "falls as", { size: 11 }); G.label(480, 346, "tp grows", { size: 11 });
          G.from(G.svg.querySelectorAll("rect"), { opacity: 0, duration: 0.3, stagger: 0.03 });
          G.caption("split only when something forces you to");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A column split gives finished pieces; a row split gives partial sums. Megatron-style TP uses column then row, so each MLP and each attention block needs one all-reduce: 2 per layer.</li>
<li>TP divides weight reads by tp but adds 2 × layers latency-bound all-reduces per token, so speed-up is sub-linear and can stall.</li>
<li>Pipeline parallelism cuts by layers; its bubble is (p−1)/(m+p−1); 1F1B keeps the bubble but bounds stored activations to about p microbatches.</li>
<li>Mixed-precision Adam needs 16 B per parameter; ZeRO and FSDP shard it to 16/n, at the price of gathering weights just in time (about 1.5× DDP's bytes).</li>
<li>Expert parallelism needs two all-to-alls per MoE layer and is only as fast as its busiest expert.</li>
<li>For serving: replicas first; TP only to fit or for latency, within fast links; PP across nodes.</li></ul>`,
    sim: {
      title: "How much does tensor parallelism buy?",
      intro: "Pick a model, the GPU's memory bandwidth and size, the link between GPUs and the number of tokens decoded together. The bars show the time of one decode step at tp = 1, 2, 4 and 8, split into reading weights and the 2 × layers all-reduces (ring formula from P4.1). The panels describe the TP degree you select.",
      height: 300,
      controls: [
        { id: "m", label: "model", type: "select", value: "l8", options: [["l8", "Llama-3-8B"], ["l70", "Llama-3-70B"]] },
        { id: "tp", label: "selected TP degree", type: "select", value: "4", options: [["1", "1"], ["2", "2"], ["4", "4"], ["8", "8"]] },
        { id: "bw", label: "GPU memory bandwidth, GB/s (example value: check the spec sheet)", min: 100, max: 4000, step: 50, value: 300, format: (v) => v + " GB/s" },
        { id: "mem", label: "GPU memory, GB (example value)", min: 16, max: 192, step: 8, value: 80, format: (v) => v + " GB" },
        { id: "lk", label: "link bandwidth B, GB/s (example value: fit yours in P4.1)", min: 5, max: 900, step: 5, value: 25, format: (v) => v + " GB/s" },
        { id: "al", label: "latency α per ring step, µs (example value)", min: 1, max: 50, step: 1, value: 10, format: (v) => v + " µs" },
        { id: "b", label: "tokens decoded together (batch)", min: 1, max: 128, step: 1, value: 1 },
      ],
      draw(G, v) {
        const m = MODELS[v.m], tps = [1, 2, 4, 8], sel = +v.tp;
        const rs = tps.map((tp) => decode(m, tp, v.b, v.bw, v.lk, v.al));
        const top = Math.max(...rs.map((r) => r.t)) * 1e3;
        const nice = (x) => { const e = 10 ** Math.floor(Math.log10(x)); return [1, 2, 5, 10].map((k) => k * e).find((k) => k >= x); };
        const ymax = nice(top), base = 250, sc = 200 / ymax;
        G.axes(60, 36, 560, 214, {});
        [0.5, 1].forEach((f) => { G.line(60, base - 200 * f, 620, base - 200 * f, { color: "line", w: 0.5 }); G.label(54, base - 200 * f + 4, (ymax * f).toFixed(ymax * f < 10 ? 1 : 0), { anchor: "end", size: 10 }); });
        G.label(60, 28, "ms per decode step", { size: 11 });
        rs.forEach((r, i) => {
          const x = 90 + i * 135, hot = tps[i] === sel;
          G.rect(x, base - r.tw * 1e3 * sc, 80, r.tw * 1e3 * sc, { fill: "k", rx: 3, stroke: hot ? "ink" : undefined });
          if (r.tc > 0) G.rect(x, base - r.t * 1e3 * sc, 80, Math.max(1, r.tc * 1e3 * sc), { fill: "hot", rx: 3, stroke: hot ? "ink" : undefined });
          G.text(x + 40, base - r.t * 1e3 * sc - 6, (r.t * 1e3).toFixed(r.t * 1e3 < 10 ? 2 : 1), { anchor: "middle", size: 12 });
          G.text(x + 40, base + 18, "tp = " + tps[i], { anchor: "middle", size: 12, weight: hot ? "bold" : undefined });
        });
        G.rect(400, 8, 10, 10, { fill: "k", rx: 2 }); G.label(414, 17, "read weights", { size: 11 });
        G.rect(510, 8, 10, 10, { fill: "hot", rx: 2 }); G.label(524, 17, "all-reduces", { size: 11 });
        const r = rs[tps.indexOf(sel)], r1 = rs[0], ar = 2 * m.L, fit = r.weights <= 0.9 * v.mem * 1e9;
        return [
          { title: `TP = ${sel} · ${m.name}`, rows: [["step time", F.ms(r.t)], ["tokens/s per request", F.num(1 / r.t, 1)], ["speed-up vs tp = 1", (r1.t / r.t).toFixed(2) + "×"], ["tokens/s per GPU (all requests)", F.num(v.b / r.t / sel, 1)]],
            chip: [r1.t / r.t >= 0.7 * sel, r1.t / r.t >= 0.7 * sel ? "scales well" : "far from linear"] },
          { title: "Where the time goes", gauge: [[r.tw / r.t, "k"], [r.tc / r.t, "hot"]], gaugeText: `weights ${(100 * r.tw / r.t).toFixed(0)}% · all-reduce ${(100 * r.tc / r.t).toFixed(0)}%`,
            rows: [["all-reduces per step", F.num(sel > 1 ? ar : 0)], ["size of each", F.bytes(r.S)], ["comm per step", F.ms(r.tc)]] },
          { title: "Weights per GPU", gauge: [[Math.min(1, r.weights / (v.mem * 1e9))]].map(([f]) => [f, f > 0.9 ? "hot" : "ok"]), gaugeText: `${(r.weights / 1e9).toFixed(1)} GB of ${v.mem} GB`,
            chip: [fit, fit ? "fits (≤ 90% of memory)" : "does not fit: raise tp"],
            html: `<p class="note">A model of decode: weight reads at 80% of the bandwidth plus ring all-reduces with the example α and B. It ignores KV-cache reads, compute and overlap, so it is optimistic. At large batch decode becomes compute-bound (P1.4) and the weight bar is no longer the whole story.</p>` },
        ];
      },
    },
    practice: {
      items: [
        { title: "TP MLP numerics", tier: "T0 · medium", goal: "Implement tp_mlp (column-split gate/up, row-split down) in exercises/tp_mlp.py; the sum of the per-rank partials must match the unsplit MLP at fp32 (atol 1e-5) for tp 1, 2, 4, 8. Then extend it to attention heads.",
          cmd: `uv run pytest ${MOD}/exercises/test_tp_mlp.py` },
        { title: "Pipeline bubble: formula vs simulation", tier: "T0 · medium", goal: "Check that parallel.simulate_pipeline matches (p−1)/(m+p−1) for GPipe and 1F1B and that 1F1B holds at most p microbatches. Then make stage times unequal and add interleaving.",
          cmd: `uv run pytest ${MOD}/exercises/test_pipeline.py` },
        { title: "TP communication in the capacity calculator", tier: "T0 · medium", goal: "Implement tp_comm_seconds_per_step (2 ring all-reduces per layer) in exercises/tp_comm.py to match platform/capacity/core.py, then print decode ceilings for TP = 1, 2, 4 with your fitted α and B.",
          cmd: `uv run pytest ${MOD}/exercises/test_tp_capacity.py` },
        { title: "Measured vLLM TP = 1/2/4 vs the prediction", tier: "T3 · hard", goal: "On a 4-GPU box run vLLM at TP 1, 2 and 4 and compare batch-1 inter-token latency, throughput per GPU and KV capacity with your prediction. Explain every gap over 20%.",
          cmd: `bash ${MOD}/examples/03_vllm_tp.sh` },
      ],
      labs: [
        { label: "Megatron column/row split of a SwiGLU MLP on simulated ranks (<code>uv run python …</code>)", path: `${MOD}/examples/01_tp_mlp_numpy.py` },
        { label: "The same with real processes and one all-reduce (<code>uv run --extra torch python … --tp 4</code>)", path: `${MOD}/examples/02_tp_torch_gloo.py` },
        { label: "T3: vLLM with --tensor-parallel-size 1, 2, 4 and a load sweep for each", path: `${MOD}/examples/03_vllm_tp.sh` },
        { label: "T3: FSDP2 <code>fully_shard</code> vs DDP, parameters and memory per rank (<code>torchrun --nproc-per-node 4 …</code>)", path: `${MOD}/examples/04_fsdp2_toy.py` },
        { label: "Reference TP MLP, pipeline simulator (GPipe, 1F1B) and ZeRO memory accounting", path: `${MOD}/parallel.py` },
        { label: "Where TP comm cost enters the D3 capacity calculator", path: "platform/capacity/core.py" },
        { label: "AWS guide: g6.12xlarge (4 GPUs, PCIe) with cost, auto-stop and teardown", path: `${MOD}/aws.md` },
        { label: "Step-by-step animation of the TP split of an MLP", path: "animations/p4-tp-mlp.html" },
      ],
    },
  });
})();
