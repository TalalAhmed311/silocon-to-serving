/* P2.6 — Speculative decoding and CUDA graphs. Spend the idle compute of batch-1 decode on guesses, keep the output
   exactly the target's, then remove the CPU launch overhead that small steps cannot hide. */
(function () {
  const F = S2S.fmt;
  // E[tokens per target pass] and speedup: platform/specdec/core.py
  const E = (a, k) => (a >= 1 ? k + 1 : (1 - Math.pow(a, k + 1)) / (1 - a));
  const spd = (a, k, c) => E(a, k) / (k * c + 1);
  // batch model (an assumption, see step 8): a step costs max(1 weight read, tokens / R); the draft costs c per step
  const planCost = (B, R) => Math.max(1, B / R);
  const specCost = (B, k, c, R) => k * c * Math.max(1, B / R) + Math.max(1, (B * (k + 1)) / R);
  const spdB = (a, k, c, B, R) => (B * E(a, k) / specCost(B, k, c, R)) / (B / planCost(B, R));
  const bestK = (f, kmax = 16) => { let b = 1; for (let k = 2; k <= kmax; k++) if (f(k) > f(b)) b = k; return b; };

  S2S.lesson({
    id: "p2-6", n: "P2.6", title: "Speculative decoding and CUDA graphs",
    subtitle: "Serving engines · first principles · T0 maths and prototype, T2 for vLLM",
    kicker: "Lesson · ≈ 45 min",
    headline: "Guess several tokens, check them all at once",
    intro: `<p>At batch 1, decode reads every weight to produce a single token and leaves the arithmetic units almost idle (P1.2, P1.4). <b>Speculative decoding</b> puts that idle compute to work: a cheap guesser proposes several tokens, and the real model checks them all in one pass. This lesson shows why checking is as cheap as generating, the rule that keeps the output <i>exactly</i> what the big model would have produced, how to compute the speedup, and why it can hurt at high load. The second half covers <b>CUDA graphs</b>, which remove the CPU's per-kernel launch cost from small decode steps.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "target model"], ["v", "draft / proposal"], ["ok", "accepted"], ["hot", "rejected / waste"], ["w", "idle / memory"]],
    prev: "p2-5", next: "p2-7",
    steps: [
      { rail: "spare compute", title: "Checking 5 tokens costs the same as making 1",
        body: `<p>P1.2 and P1.4 showed that a batch-1 decode step is limited by <b>memory</b>: the GPU streams all the weights from memory, does a few multiplications per weight, and waits for the next load. The arithmetic units are mostly idle.</p>
<p>Now feed the model <b>5 tokens at once</b> instead of 1 (as in a short prefill). Each weight is still read once, but now it is used for 5 positions. The reading time is unchanged; only the arithmetic grows, and there was room for it.</p>
<div class="eq">step time ≈ max( weight-read time , compute time )
weight-read time:  1 unit, whatever the token count
compute time:      tokens / R units
R = token count where compute catches up (example
    value 64: peak FLOPs ÷ bandwidth, see spec sheet)

 1 token : max(1, 1/64)  = 1
 5 tokens: max(1, 5/64)  = 1
64 tokens: max(1, 64/64) = 1
128 tokens: max(1, 128/64) = 2</div>
<p>So <b>verifying</b> k proposed tokens in one pass costs about as much as <b>generating</b> one. If several of the guesses are right, you get several tokens for one weight read. That is the whole idea; the rest of the lesson is about doing it without changing the output and about knowing when it pays.</p>`,
        check: { q: "At batch 1, a decode step reads all the weights and uses little arithmetic. Why is a 5-token verification pass about as fast as a 1-token pass?",
          options: ["The GPU skips the extra tokens", "Time is set by reading the weights once, which does not depend on how many positions use them, and the extra arithmetic fits in the idle compute", "The extra tokens are computed by the draft model"], answer: 1,
          why: "A memory-bound step costs weight bytes ÷ bandwidth. Five positions reuse each loaded weight five times, so the arithmetic grows 5× but stays below the time the memory reads already take." },
        scene(G) {
          const R = 64, ns = [1, 2, 4, 8, 16, 32, 64, 128], x0 = 70, y0 = 330, u = 110;
          G.text(24, 34, "time per step vs tokens processed (R = 64, example value)", { size: 13 });
          G.line(x0, y0, x0 + 540, y0, { color: "line" });
          G.line(x0, y0, x0, y0 - 2 * u - 20, { color: "line" });
          G.label(x0 - 8, y0 - u + 4, "1", { anchor: "end", size: 11 }); G.label(x0 - 8, y0 - 2 * u + 4, "2", { anchor: "end", size: 11 });
          G.label(24, 62, "units of one weight read", { size: 11 });
          const mem = [], cmp = [];
          ns.forEach((n, i) => {
            const x = x0 + 14 + i * 66, hc = (n / R) * u;
            mem.push(G.rect(x, y0 - u, 20, u, { fill: "w", rx: 2 }));
            cmp.push(G.rect(x + 22, y0 - hc, 20, hc, { fill: n / R > 1 ? "hot" : "k", rx: 2 }));
            G.label(x + 21, y0 + 18, String(n), { anchor: "middle", size: 11 });
          });
          G.line(x0, y0 - u, x0 + 540, y0 - u, { color: "ink", dash: "4 3", w: 1 });
          G.label(x0 + 14, y0 + 38, "tokens per pass →", { size: 11 });
          G.rect(40, 376, 14, 14, { fill: "w", rx: 2 }); G.label(60, 388, "weight read (fixed)", { size: 11 });
          G.rect(210, 376, 14, 14, { fill: "k", rx: 2 }); G.label(230, 388, "compute, hidden", { size: 11 });
          G.rect(370, 376, 14, 14, { fill: "hot", rx: 2 }); G.label(390, 388, "compute, now the limit", { size: 11 });
          G.text(380, 130, "1 to 64 tokens:", { size: 13, color: "ok" }); G.label(380, 150, "same step time", { size: 12, color: "ok" });
          G.from(cmp, { attr: { height: 0 }, stagger: 0.06, duration: 0.4 });
          G.caption("up to the ridge, extra tokens in a pass are free");
        } },

      { rail: "draft & verify", title: "A cheap draft proposes, the target checks in one pass",
        body: `<p>Where do the extra tokens come from? From a <b>draft</b>: a much cheaper way to guess the next few tokens (a small model, for now). The loop, called a <b>round</b>:</p>
<ol><li>The draft generates <b>k</b> tokens one after another (cheap steps).</li><li>The <b>target</b> (the real model) runs <b>one</b> pass over the prompt plus those k tokens. Position i of its output is what it would say <i>after</i> the first i proposals.</li><li>Keep proposals for as long as they agree with the target. At the first disagreement, take the target's own token instead and stop.</li><li>If all k agree, the last output position gives one extra <b>bonus</b> token for free.</li></ol>
<p>Here is a greedy example (always take the most likely token). The prefix is "the cat", the draft proposes k = 4 tokens, the target's pass gives 5 answers:</p>
<div class="eq">draft proposes:   sat   on   the   mat
target would say: sat   on   my    (rest ignored)
                   ok    ok   no  → use "my", stop
emitted: sat on my      3 tokens from 1 target pass</div>
<p>Every emitted token is one the target itself would have chosen, so the answer is unchanged. If the draft had been right 4 times, the round would emit 5 tokens. If wrong at once, 1 token: never worse than normal decoding in tokens, only in time spent on the draft. With sampling instead of greedy, "agree" needs a careful rule (next step).</p>`,
        scene(G) {
          G.text(24, 34, "one round, k = 4, prefix: the cat", { size: 13 });
          G.text(24, 68, "1  draft proposes, one cheap step each", { size: 12, color: "v" });
          const dr = ["sat", "on", "the", "mat"].map((t, i) => G.box(60 + i * 100, 80, 84, 34, t, { fill: "v", size: 13 }));
          dr.forEach((_, i) => { if (i) G.arrow(60 + i * 100 - 14, 97, 60 + i * 100 - 2, 97, { color: "v" }); });
          G.text(24, 160, "2  target: ONE pass over 'the cat' + 4 proposals", { size: 12, color: "k" });
          G.rect(40, 172, 560, 60, { stroke: "k", rx: 8, dash: "5 4" });
          const outs = ["sat", "on", "my", "·", "·"];
          const tg = outs.map((t, i) => G.box(60 + i * 100 - (i === 4 ? 0 : 0), 186, 84, 32, t, i < 3 ? { fill: "k", size: 13 } : { stroke: "line", color: "muted", size: 13 }));
          G.label(60, 252, "output i = target's choice after the first i proposals", { size: 11 });
          G.text(24, 288, "3  compare, left to right", { size: 12, color: "ink" });
          const cmp = [["sat = sat", "ok"], ["on = on", "ok"], ["the ≠ my", "hot"]].map(([t, c], i) => G.box(60 + i * 100, 300, 84, 30, t, { stroke: c, color: c, size: 12 }));
          G.label(360, 320, "first mismatch: keep the target's token, stop", { size: 12, color: "hot" });
          G.text(24, 372, "emitted:", { size: 13 });
          const em = ["sat", "on", "my"].map((t, i) => G.box(110 + i * 70, 354, 60, 28, t, { fill: i < 2 ? "ok" : "k", size: 13 }));
          G.label(330, 373, "3 tokens, 1 target pass", { size: 13, color: "ok" });
          G.from([].concat(dr, tg, cmp, em), { opacity: 0, stagger: 0.07, duration: 0.25 });
          G.caption("guess cheaply, verify in bulk, keep what the target agrees with");
        } },

      { rail: "the rule", title: "With sampling, accept with probability min(1, p/q)",
        body: `<p>Chat models usually <i>sample</i>: the target has a probability distribution <b>p</b> over the next token, and the output is a random draw from it. For speculative decoding to leave the output distribution unchanged, "agree" cannot just mean "same top token". Leviathan et al. (arXiv 2211.17192) and Chen et al. (2302.01318) gave the rule.</p>
<p>The draft has its own distribution <b>q</b> and sampled token x from it. The target, from the same position, gives p. Then:</p>
<div class="eq">accept x with probability min(1, p(x) / q(x))</div>
<p>If the target likes x at least as much as the draft did (p ≥ q), always accept. If the draft over-proposed it, accept only a fraction p/q of the time. A tiny vocabulary of three tokens makes this concrete:</p>
<div class="eq">token      A     B     C
p (target) 0.5   0.3   0.2
q (draft)  0.2   0.6   0.2
p / q      2.5   0.5   1.0
accept     100%  50%   100%</div>
<p>The draft loves B (0.6) but the target gives it only 0.3. So when B is proposed, half the time it is thrown out. When A is proposed (the draft under-rates it), it always stays. What happens to the rejected proposals is the subtle part, next.</p>`,
        scene(G) {
          const p = [0.5, 0.3, 0.2], q = [0.2, 0.6, 0.2], names = ["A", "B", "C"], H = 135, y0 = 250;
          G.text(24, 34, "target p (cyan) vs draft q (amber), three tokens", { size: 13 });
          G.line(60, y0, 330, y0, { color: "line" });
          const bs = [];
          names.forEach((n, i) => {
            const x = 80 + i * 84;
            bs.push(G.rect(x, y0 - p[i] * 2 * H, 30, p[i] * 2 * H, { fill: "k", rx: 2 }), G.rect(x + 32, y0 - q[i] * 2 * H, 30, q[i] * 2 * H, { fill: "v", rx: 2 }));
            G.label(x + 31, y0 + 18, n, { anchor: "middle", size: 13, color: "ink" });
            G.label(x + 15, y0 - p[i] * 2 * H - 6, String(p[i]), { anchor: "middle", size: 11 }); G.label(x + 47, y0 - q[i] * 2 * H - 6, String(q[i]), { anchor: "middle", size: 11 });
          });
          G.text(370, 82, "accept probability", { size: 13 });
          G.text(370, 104, "min(1, p / q)", { size: 13, color: "ink" });
          [["A", "1", "ok"], ["B", "0.5", "hot"], ["C", "1", "ok"]].forEach(([n, a, c], i) => {
            G.box(370, 124 + i * 52, 44, 34, n, { stroke: "line", size: 13 });
            G.rect(420, 128 + i * 52, 180 * Number(a), 26, { fill: c, rx: 3, opacity: 0.9 });
            G.text(426, 146 + i * 52, a === "1" ? "always" : "50%", { size: 12, color: "bg" });
          });
          G.label(24, 300, "A: draft under-proposes it (0.2 vs 0.5) → never rejected", { size: 12, color: "ok" });
          G.label(24, 322, "B: draft over-proposes it (0.6 vs 0.3) → rejected 50%", { size: 12, color: "hot" });
          G.text(24, 372, "rejections happen exactly where q exceeds p", { size: 15, color: "ink" });
          G.from(bs, { attr: { height: 0 }, stagger: 0.08, duration: 0.4 });
          G.caption("accept less often exactly where the draft over-proposes");
        } },

      { rail: "the residual", title: "Why the replacement comes from max(0, p − q)",
        body: `<p>When a proposal is rejected, the round still has to emit a token. Which one? The tempting answer, "sample from p", is <b>wrong</b>. The right answer is to sample from the <b>residual</b>: the part of p the draft did not already cover.</p>
<div class="eq">residual(x) ∝ max(0, p(x) − q(x))</div>
<p>Check it with the three-token example. Add up every way the round can end with each token:</p>
<div class="eq">accepted proposal: q × min(1, p/q) = min(p, q)
  A: 0.2    B: 0.3    C: 0.2    (total 0.7)
rejected (prob 0.3): p − q = [0.3, 0, 0]
  normalised residual = [1, 0, 0] → always A
total: A 0.2 + 0.3×1 = 0.5
       B 0.3,  C 0.2        = p exactly</div>
<p>Accepting a proposal already delivers min(p, q) of each token's probability. The residual supplies exactly the missing p − min(p, q) = max(0, p − q). The sum is p for <i>every</i> token, whatever q is. That is the theorem: <b>the output has exactly the distribution of the target alone</b>; the draft only changes speed.</p>
<p>The wrong version, resampling from p, over-weights tokens the draft also likes: it gives A = 0.2 + 0.3×0.5 = 0.35, B = 0.3 + 0.3×0.3 = 0.39, C = 0.2 + 0.3×0.2 = 0.26. Not p. Exercise 1 plants exactly this bug and checks that a χ² test catches it.</p>`,
        check: { q: "In the 3-token example (p = 0.5/0.3/0.2, q = 0.2/0.6/0.2), what happens if a rejected proposal is replaced by a fresh sample from p instead of from the residual?",
          options: ["Nothing: the output is still exactly p", "Token A comes out 35% of the time instead of 50%, so the output is no longer the target's distribution", "Every proposal is accepted"], answer: 1,
          why: "A is accepted 0.2 of the time directly, and a rejection (prob 0.3) resampled from p gives A with probability 0.5: 0.2 + 0.3 × 0.5 = 0.35, not 0.5. The residual puts all the rejected mass back where the draft under-covered p." },
        scene(G) {
          const p = [0.5, 0.3, 0.2], acc = [0.2, 0.3, 0.2], res = [0.3, 0, 0], bad = [0.15, 0.09, 0.06], names = ["A", "B", "C"], W = 400, x0 = 90;
          const draw = (y0, title, tc, second, tag) => {
            G.text(24, y0 - 14, title, { size: 13, color: tc });
            const out = [];
            names.forEach((n, i) => {
              const y = y0 + i * 40;
              G.text(40, y + 20, n, { size: 14 });
              out.push(G.rect(x0, y, W * acc[i], 28, { fill: "ok", rx: 3 }));
              if (second[i] > 0) out.push(G.rect(x0 + W * acc[i], y, W * second[i], 28, { fill: tag, rx: 3 }));
              G.rect(x0, y - 3, W * p[i], 34, { stroke: "ink", rx: 4, dash: "3 3", sw: 1.5 });
              const tot = acc[i] + second[i];
              G.label(x0 + W * Math.max(p[i], tot) + 10, y + 19, tot.toFixed(2) + (Math.abs(tot - p[i]) < 1e-9 ? "  = p" : `  ≠ p (${p[i]})`), { size: 12, color: Math.abs(tot - p[i]) < 1e-9 ? "ok" : "hot" });
            });
            return out;
          };
          G.text(24, 28, "probability of emitting each token", { size: 13 });
          const a = draw(76, "residual resample: green accepted + amber residual", "ok", res, "v");
          const b = draw(256, "wrong: resample from p (red)", "hot", bad, "hot");
          G.label(24, 396, "dashed outline = the target's p; the bars must fill it exactly", { size: 12, color: "ink" });
          G.from(a.concat(b), { attr: { width: 0 }, stagger: 0.06, duration: 0.45 });
          G.caption("the residual tops every token up to exactly p");
        } },

      { rail: "tokens per pass", title: "How many tokens does a round emit?",
        body: `<p>Speed depends on how many proposals survive. Simplest model: each proposal is accepted with the same probability <b>α</b> (the acceptance rate), independently. A proposal at position j is only reached if all j − 1 before it were accepted, and then accepted too. So:</p>
<div class="eq">P(first j proposals all accepted) = α^j
tokens per round = 1 + α + α² + … + α^k
                 = (1 − α^(k+1)) / (1 − α)</div>
<p>The leading "1" is the token the round always emits (a correction after a rejection, or the bonus after full acceptance). Worked example, α = 0.8, k = 4:</p>
<div class="eq">1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.3616
check: (1 − 0.8⁵) / 0.2 = (1 − 0.32768) / 0.2 = 3.3616</div>
<p>Almost 3.4 tokens per target pass, from one target read of the weights. With α = 0.6 and k = 4 the same sum is 2.31; with α = 0.9 and k = 4 it is 4.10. The code is <code>expected_tokens_per_round</code> in <code>platform/specdec/core.py</code>, and exercise 2 checks it against a simulation.</p>
<p>The i.i.d. assumption is a simplification. Real acceptance clusters: easy spans (boilerplate, quotes) accept in streaks, hard spans reject at once. Exercise 2 asks whether that makes the formula optimistic or pessimistic.</p>`,
        check: { q: "α = 0.8 and k = 4. What is the expected number of tokens emitted per target pass?",
          options: ["4.0", "3.36", "0.8 × 4 = 3.2", "1.8"], answer: 1,
          why: "Sum the probabilities of reaching each position: 1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.3616. The 4.0 would need every proposal accepted; 3.2 forgets the guaranteed token and that later proposals need all earlier ones accepted." },
        scene(G) {
          const a = 0.8, k = 4, x0 = 60, y0 = 300, u = 190;
          G.text(24, 34, "chance each position is reached, α = 0.8, k = 4", { size: 13 });
          G.line(x0, y0, x0 + 400, y0, { color: "line" });
          const vals = Array.from({ length: k + 1 }, (_, j) => Math.pow(a, j));
          const bars = vals.map((v, j) => {
            const x = x0 + 14 + j * 76;
            G.label(x + 25, y0 + 18, j === 0 ? "base" : "proposal " + j, { anchor: "middle", size: 11 });
            G.label(x + 25, y0 - v * u - 8, v.toFixed(4).replace(/0+$/, "").replace(/\.$/, ""), { anchor: "middle", size: 12, color: "ink" });
            return G.rect(x, y0 - v * u, 50, v * u, { fill: j === 0 ? "k" : "ok", rx: 3 });
          });
          let cum = 0;
          vals.forEach((v, j) => { cum += v; });
          G.text(24, 352, "sum = 3.3616 tokens per target pass", { size: 16, color: "ok" });
          G.label(24, 376, "the base token is always emitted; each later one needs all earlier ones accepted", { size: 12 });
          G.text(480, 120, "α = 0.6:  2.31", { size: 13, color: "ink" });
          G.text(480, 150, "α = 0.8:  3.36", { size: 13, color: "ok" });
          G.text(480, 180, "α = 0.9:  4.10", { size: 13, color: "ink" });
          G.label(480, 204, "(k = 4 for each)", { size: 11 });
          G.from(bars, { attr: { height: 0 }, stagger: 0.1, duration: 0.4 });
          G.caption("each extra proposal adds less than the one before");
        } },

      { rail: "speedup", title: "The speedup, and how many tokens to propose",
        body: `<p>Tokens per target pass is not wall-clock speed. The draft's k steps cost time too. Let <b>c</b> be the cost of one draft step as a fraction of one target step (a draft one-twentieth the size gives c ≈ 0.05; real values depend on the draft and the kernels). One round takes <code>k·c + 1</code> target-step units, so:</p>
<div class="eq">speedup = E[tokens] / (k·c + 1)

α = 0.8, k = 4:
  c = 0.05:  3.3616 / (0.2 + 1) = 2.80×
  c = 0.20:  3.3616 / (0.8 + 1) = 1.87×
α = 0.6, k = 4:
  c = 0.05:  2.3056 / 1.2 = 1.92×
  c = 0.20:  2.3056 / 1.8 = 1.28×</div>
<p>These are the formula's values (<code>examples/01_spec_math.py</code>), not measurements. The draft cost matters a lot: at c = 0.2 the same acceptance gives a third less speedup.</p>
<p>Raising k has two sides. Each extra proposal adds α^k tokens but costs c of a step. So the speedup rises, peaks, then falls. For α = 0.8 the best k is <b>8</b> at c = 0.05 (3.09×) but only <b>4</b> at c = 0.2 (1.87×). A cheaper draft or a higher α allows longer proposals. The simulator on the Simulate tab plots this curve.</p>`,
        scene(G) {
          const x0 = 60, y0 = 330, w = 520, h = 240, kmax = 16, ymax = 3.5, a = 0.8;
          const X = (k) => x0 + (w * (k - 1)) / (kmax - 1), Y = (v) => y0 - (h * v) / ymax;
          G.text(24, 30, "speedup vs proposals per round, α = 0.8", { size: 13 });
          G.axes(x0, y0 - h, w, h, {});
          [1, 2, 3].forEach((v) => { G.line(x0, Y(v), x0 + w, Y(v), { color: v === 1 ? "ink" : "line", dash: "3 4", w: 1 }); G.label(x0 - 8, Y(v) + 4, v + "×", { anchor: "end", size: 11 }); });
          [1, 4, 8, 12, 16].forEach((k) => G.label(X(k), y0 + 16, String(k), { anchor: "middle", size: 11 }));
          G.label(x0 + w, y0 + 34, "k, proposals per round →", { anchor: "end", size: 11 });
          const curves = [[0.05, "k"], [0.2, "v"]], paths = [];
          curves.forEach(([c, col]) => {
            let d = ""; for (let k = 1; k <= kmax; k++) d += `${k > 1 ? " L" : "M"} ${X(k)} ${Y(spd(a, k, c))}`;
            paths.push(G.path(d, { color: col, w: 3 }));
            const kb = bestK((k) => spd(a, k, c)), vb = spd(a, kb, c);
            G.circle(X(kb), Y(vb), 6, { fill: col });
            G.text(X(kb) + (kb > 10 ? -10 : 10), Y(vb) - 12, `best k = ${kb}: ${vb.toFixed(2)}×`, { size: 12, color: col, anchor: kb > 10 ? "end" : "start" });
          });
          G.label(X(12), Y(spd(a, 12, 0.05)) + 24, "c = 0.05", { size: 12, color: "k" });
          G.label(X(9), Y(spd(a, 9, 0.2)) + 22, "c = 0.2", { size: 12, color: "v" });
          G.from(paths, { opacity: 0, stagger: 0.3, duration: 0.6 });
          G.caption("a cheaper draft moves the peak to longer proposals");
        } },

      { rail: "proposers", title: "Where the draft tokens come from",
        body: `<p>Anything that guesses tokens cheaply can be the draft; the verification rule keeps the output exact either way. Four common proposers, trading draft cost against acceptance:</p>
<table><tr><th>Proposer</th><th>Draft cost</th><th>Acceptance α</th><th>Notes</th></tr>
<tr><td>small draft model</td><td>a few % of target FLOPs</td><td>medium</td><td>needs a smaller model with the <b>same tokenizer</b></td></tr>
<tr><td><b>n-gram</b> (prompt lookup)</td><td>about free</td><td>high when output copies the prompt, else low</td><td>no extra weights</td></tr>
<tr><td>Medusa heads</td><td>small extra heads on the target</td><td>medium</td><td>trained heads; verify a tree of candidates</td></tr>
<tr><td>EAGLE (2401.15077)</td><td>one light layer on target features</td><td>high</td><td>trained; among the strongest published</td></tr></table>
<p>The n-gram proposer is the easiest to understand and needs no training: take the last few generated tokens, look for the same sequence earlier in the prompt (or output), and propose the tokens that followed it there. Code edits, summaries that quote, and RAG answers repeat their context constantly, so it hits often there. In open-ended chat it rarely finds a match, α falls, and the speculation is wasted.</p>
<p>So α is a property of the <b>workload</b> as much as of the draft. Exercise 4 measures it for code, chat and RAG prompts. In vLLM the code lives in <code>vllm/v1/spec_decode/</code> and the option is passed as a speculative config (<code>--speculative-config</code>; the exact JSON keys are <b>UNVERIFIED</b> at v0.31.0, check <code>vllm serve --help</code>).</p>`,
        scene(G) {
          G.text(24, 34, "n-gram proposer: copy what followed the last time", { size: 13 });
          G.label(24, 62, "prompt (already in the KV cache)", { size: 11 });
          const pr = ["def", "load", "(", "path", ")", ":", "return", "open", "(", "path", ")"];
          const cell = (arr, y, hi, col) => arr.map((t, i) => G.box(24 + i * 56, y, 52, 30, t, hi.includes(i) ? { fill: col, size: 11 } : { stroke: "line", size: 11 }));
          const a = cell(pr, 70, [8, 9], "q");
          G.label(24, 122, "last two generated tokens are '(' 'path'. Find them in the prompt…", { size: 12, color: "q" });
          G.text(24, 168, "generated so far", { size: 11, color: "muted" });
          const gen = ["def", "read", "(", "path"];
          const b = cell(gen, 176, [2, 3], "q");
          G.arrow(24 + 8.5 * 56, 104, 24 + 3.5 * 56, 170, { color: "q", w: 1.5, dash: "4 3" });
          G.text(24, 252, "…and propose the tokens that came next there:", { size: 12, color: "v" });
          const nx = [")", ":", "…"].map((t, i) => G.box(24 + i * 62, 264, 56, 30, t, { fill: "v", size: 13 }));
          G.label(220, 284, "k proposals for free: no model ran", { size: 12, color: "ink" });
          G.text(24, 340, "good at: code edits, quoting, RAG answers", { size: 13, color: "ok" });
          G.text(24, 366, "poor at: open-ended chat (no repeated text to copy)", { size: 13, color: "hot" });
          G.label(24, 394, "draft models, Medusa and EAGLE learn to guess instead of copying", { size: 12 });
          G.from([].concat(a, b, nx), { opacity: 0, stagger: 0.04, duration: 0.2 });
          G.caption("the draft can be a lookup, a small model, or trained heads");
        } },

      { rail: "high batch", title: "Why speculation can hurt at high load",
        body: `<p>Speculation spends <i>idle</i> compute. At batch 1 compute is idle; with many sequences in the batch (P2.3) it is not. Each sequence now verifies k + 1 tokens instead of 1, and the proposals that get rejected were wasted arithmetic that other sequences could have used.</p>
<p>Use the step-1 model: a pass costs <code>max(1, tokens / R)</code> with R = 64 (example value). Take α = 0.8, k = 4, c = 0.05 and count tokens per unit time against plain decoding:</p>
<div class="eq">B = 1:  plain: 1 unit, 1 token
        spec: 0.2 + max(1, 5/64) = 1.2 units
              3.36 tokens → 2.80×
B = 16: plain: max(1, 16/64) = 1 unit, 16 tokens
        spec: 0.2 + max(1, 80/64) = 1.45 units
              16 × 3.36 = 53.8 tokens
              53.8 / 1.45 = 37.1 vs 16 → 2.32×
B = 64: plain: max(1, 64/64) = 1 unit, 64 tokens
        spec: 0.2 + max(1, 320/64) = 5.2 units
              64 × 3.36 = 215 tokens
              215 / 5.2 = 41.4 vs 64 → 0.65×</div>
<p>Past the ridge, verification is compute-bound and every verified token costs time; 3.36 useful tokens out of 5 + the draft means throughput drops to 3.36 / 5.2 = 0.65 of plain decoding. Speculative decoding is a <b>latency</b> tool for lightly loaded servers, not a throughput tool for saturated ones. This is a model with an assumed R, not a measurement; #10 and exercise 4 measure it for real, at batch 1 and batch 16 on the same prompts.</p>`,
        check: { q: "A server runs at a high, saturated batch and you switch speculative decoding on. What do you expect, and why?",
          options: ["More throughput, since every pass emits more tokens", "Lower or flat throughput: compute is already busy, so verifying rejected proposals displaces other sequences' work", "No change, since output is identical"], answer: 1,
          why: "Speculation only wins by using idle compute. When the batch already keeps the GPU busy, each rejected proposal's arithmetic competes with real work. Compare speedups only at equal batch size." },
        scene(G) {
          const a = 0.8, k = 4, c = 0.05, R = 64, x0 = 70, y0 = 330, w = 500, h = 230, ymax = 3.4;
          const Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256];
          const X = (i) => x0 + (w * i) / (Bs.length - 1), Y = (v) => y0 - (h * v) / ymax;
          G.text(24, 30, "speedup vs batch size (model: α 0.8, k 4, c 0.05, R 64)", { size: 13 });
          G.axes(x0, y0 - h, w, h, {});
          [0, 1, 2, 3].forEach((v) => { if (v) G.line(x0, Y(v), x0 + w, Y(v), { color: v === 1 ? "ink" : "line", dash: "3 4", w: 1 }); G.label(x0 - 8, Y(v) + 4, v + "×", { anchor: "end", size: 11 }); });
          Bs.forEach((b, i) => G.label(X(i), y0 + 16, String(b), { anchor: "middle", size: 11 }));
          G.label(x0 + w, y0 + 34, "batch size (sequences) →", { anchor: "end", size: 11 });
          let d = ""; Bs.forEach((b, i) => { d += `${i ? " L" : "M"} ${X(i)} ${Y(spdB(a, k, c, b, R))}`; });
          const line = G.path(d, { color: "k", w: 3 });
          Bs.forEach((b, i) => G.circle(X(i), Y(spdB(a, k, c, b, R)), 4, { fill: spdB(a, k, c, b, R) < 1 ? "hot" : "k" }));
          G.label(X(0) + 8, Y(spdB(a, k, c, 1, R)) - 12, "2.80×", { size: 12, color: "k" });
          G.label(X(6), Y(spdB(a, k, c, 64, R)) + 24, "0.65× at 64", { size: 12, color: "hot", anchor: "middle" });
          G.text(x0 + 10, y0 - 30, "wins at low batch, loses past the ridge", { size: 13, color: "ink" });
          G.from(line, { opacity: 0, duration: 0.7 });
          G.caption("break-even at 1×: speculation is for low load");
        } },

      { rail: "launch cost", title: "The CPU launch cost CUDA graphs remove",
        body: `<p>The second half of this lesson is a different bottleneck. A decode step at small batch is not one big operation. It is hundreds of small <b>kernels</b> (GPU functions): a norm, a matmul, an attention, an activation, and so on, for each of 32 layers. The CPU has to <b>launch</b> each one, and each launch costs a few microseconds, plus the Python and framework work around it in eager mode.</p>
<div class="eq">example numbers, not measurements:
  ~600 kernels per decode step × ~5 µs to launch each
  = ~3 ms of CPU launch work per step</div>
<p>If a kernel runs for less time than the CPU needs to launch the next one, the GPU finishes and <b>waits</b>. At batch 1 on a fast GPU many kernels are that short, so the step is partly gaps. Suppose the kernels themselves need 8 ms and the gaps add 3 ms: 11 ms per step instead of 8, so roughly 27% of the step is overhead.</p>
<p>A <b>CUDA graph</b> fixes this by <i>recording</i> the whole sequence of kernels once, and then <i>replaying</i> it with a single launch. The GPU reads the next kernel from the graph itself, with no CPU in the loop, so the gaps disappear:</p>
<div class="eq">eager : 8 ms + 3 ms of gaps = 11 ms
graph : 8 ms          → 11 / 8 = 1.375× faster
2× bigger model: 16 + 3 = 19 ms vs 16 ms
                       → 19 / 16 = 1.19×</div>
<p>The overhead is fixed, so the bigger the step, the smaller the relative gain. That is why graphs help small models and small batches most, and why vLLM enables them by default. P2.1 exercise 3 had you measure eager against graphs at batch 1; compare your measured difference with ~3 ms.</p>`,
        check: { q: "A model twice as large has twice the GPU time per decode step but the same number of kernels. What happens to the benefit of CUDA graphs?",
          options: ["It doubles", "The saved launch overhead is the same in milliseconds, so the relative gain roughly halves", "It disappears completely"], answer: 1,
          why: "Graphs remove a fixed per-kernel launch cost. If the step is longer, that fixed saving is a smaller fraction of it (3 ms of 11 ms becomes 3 ms of 19 ms)." },
        scene(G) {
          const x0 = 30, scale = 1.35;
          const lane = (y, label, gaps) => {
            G.text(24, y - 14, label, { size: 13 });
            let x = x0; const out = [];
            for (let i = 0; i < 14; i++) {
              const w = 20 + ((i * 7) % 5) * 4;
              out.push(G.rect(x, y, w, 34, { fill: "k", rx: 2 })); x += w;
              if (gaps) { out.push(G.rect(x, y + 12, 12, 10, { fill: "hot", rx: 1, opacity: 0.85 })); x += 12; }
            }
            return [out, x];
          };
          G.text(24, 30, "GPU timeline during one decode step (schematic)", { size: 13 });
          const [e, ex] = lane(86, "eager: CPU launches kernel by kernel", true);
          const [g, gx] = lane(206, "CUDA graph: one launch, kernels back to back", false);
          G.label(x0, 146, "red = GPU idle, waiting for the CPU to launch the next kernel", { size: 12, color: "hot" });
          G.line(gx, 196, gx, 250, { color: "ok", w: 2, dash: "4 3" });
          G.line(ex, 76, ex, 130, { color: "hot", w: 2, dash: "4 3" });
          G.label(gx - 4, 266, "step ends here", { size: 12, color: "ok", anchor: "end" });
          G.label(ex - 4, 70, "eager step ends", { size: 12, color: "hot", anchor: "end" });
          G.text(24, 330, "same kernels, same work: the gaps are the only difference", { size: 14, color: "ink" });
          G.label(24, 356, "graph = record once, replay with a single launch", { size: 12 });
          G.label(24, 378, "gap size is an example (≈ 3 ms per step of 600 launches); measure yours in P2.1 exercise 3", { size: 11 });
          G.from(e.concat(g), { opacity: 0, stagger: 0.01, duration: 0.15 });
          G.caption("a graph replays the recorded kernels with no CPU gaps");
        } },

      { rail: "graph limits", title: "What a graph requires: fixed shapes, padding, memory",
        body: `<p>A recorded graph is a frozen script: same kernels, same tensor shapes, same memory addresses every replay. That brings constraints, and they explain things you see when vLLM starts.</p>
<ul><li><b>Static shapes.</b> A graph captured for batch 8 only runs batch 8. Engines capture one graph per <b>batch-size bucket</b> (for example 1, 2, 4, 8, 16, …; the exact list is configuration) and pad the real batch up to the next bucket.</li><li><b>Static addresses.</b> Inputs are copied into fixed buffers the graph reads, and outputs are read from fixed buffers.</li><li><b>No host synchronization</b> inside, since the CPU is no longer in the loop.</li><li><b>Memory per graph.</b> Each captured size reserves memory, which is one reason vLLM's startup measures memory and then spends seconds "capturing" graphs.</li></ul>
<p>Padding is wasted work. A batch of 5 runs in the size-8 graph with 3 dummy rows: 37.5% of that step's rows are padding. It is cheap because decode is memory-bound (the weights are read once either way), and far cheaper than the launch gaps it avoids. Prefill and very large batches usually run eagerly: their kernels are long enough to hide launch cost, and their shapes vary too much.</p>
<p>The flag <code>--enforce-eager</code> turns graphs off. It is a debugging aid and a way to save the capture memory; expect slower small-batch decode. P6.6 builds a capture-and-replay loop yourself; for the design read <code>docs/design/cuda_graphs.md</code> and <code>vllm/compilation/cuda_graph.py</code> at the pinned SHA.</p>`,
        scene(G) {
          G.text(24, 34, "one captured graph per batch-size bucket (example buckets)", { size: 13 });
          const bk = [1, 2, 4, 8, 16, 32];
          const bx = bk.map((b, i) => G.box(30 + i * 98, 54, 86, 38, "graph " + b, { stroke: b === 8 ? "ok" : "line", color: b === 8 ? "ok" : "muted", sw: b === 8 ? 2.5 : 1.5, size: 12 }));
          G.label(30, 112, "each graph reserves memory", { size: 12 });
          G.text(24, 168, "a batch of 5 sequences arrives", { size: 13 });
          const rows = [];
          for (let i = 0; i < 8; i++) rows.push(G.box(60 + i * 60, 188, 50, 40, i < 5 ? "S" + (i + 1) : "pad", i < 5 ? { fill: "k", size: 12 } : { stroke: "hot", color: "hot", dash: "4 3", size: 12 }));
          G.arrow(30 + 3 * 98 + 43, 94, 30 + 3 * 98 + 43, 184, { color: "ok", w: 2 });
          G.label(60, 252, "padded up to the size-8 graph: 3 of 8 rows are wasted (37.5%)", { size: 12, color: "hot" });
          G.text(24, 304, "copy inputs into the graph's fixed buffers → replay → read outputs", { size: 13 });
          const flow = [["inputs", "q"], ["replay graph 8", "ok"], ["outputs", "k"]].map(([t, c], i) => G.box(30 + i * 190, 322, 160, 38, t, { stroke: c, color: c, size: 13 }));
          G.arrow(192, 341, 218, 341, { color: "muted" }); G.arrow(382, 341, 408, 341, { color: "muted" });
          G.label(30, 392, "no growth, no host sync, no new shapes while replaying", { size: 12 });
          G.from([].concat(bx, rows, flow), { opacity: 0, stagger: 0.04, duration: 0.2 });
          G.caption("static shapes force padding to the nearest captured size");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A memory-bound pass over k + 1 tokens costs about one decode step, so a cheap draft's k guesses can be verified for free.</li>
<li>Accept each proposal with probability min(1, p/q); on rejection sample from the normalised max(0, p − q). The output then has exactly the target's distribution.</li>
<li>Tokens per pass is (1 − α^(k+1)) / (1 − α); speedup divides that by k·c + 1, and has a best k that rises with α and falls with draft cost.</li>
<li>Proposers differ in cost and acceptance (draft model, n-gram, Medusa, EAGLE); α depends heavily on the workload.</li>
<li>Speculation spends idle compute, so it helps latency at low batch and can lose throughput when the GPU is saturated.</li>
<li>CUDA graphs replay recorded kernels with one launch, removing CPU gaps at small batch; they need static shapes, so batches are padded to captured sizes.</li></ul>`,
    sim: {
      title: "Speculative decoding speedup",
      intro: "Closed-form speedup from platform/specdec/core.py (batch 1), extended with a simple batch model: a pass costs max(1, tokens ÷ R) weight reads, where R is the ridge in tokens (an example value, check your GPU's spec sheet), and the draft costs c per step. Cyan is the batch-1 curve; amber is your chosen batch. Break-even is 1×.",
      controls: [
        { id: "a", label: "acceptance rate α", min: 0.1, max: 0.95, step: 0.05, value: 0.8, format: (x) => x.toFixed(2) },
        { id: "k", label: "proposals per round k", min: 1, max: 16, value: 4 },
        { id: "c", label: "draft cost c (fraction of a target step)", min: 0.01, max: 0.5, step: 0.01, value: 0.05, format: (x) => x.toFixed(2) },
        { id: "B", label: "batch size (sequences)", type: "select", value: 32, options: [[1, "1"], [4, "4"], [8, "8"], [16, "16"], [32, "32"], [64, "64"], [128, "128"], [256, "256"]] },
        { id: "R", label: "ridge R, tokens (example value: peak FLOPs ÷ bandwidth)", min: 16, max: 256, step: 8, value: 64 },
      ],
      height: 270,
      draw(G, v) {
        const x0 = 56, y0 = 232, w = 560, h = 200, kmax = 16;
        const f1 = (k) => spdB(v.a, k, v.c, 1, v.R), fB = (k) => spdB(v.a, k, v.c, v.B, v.R);
        let ymax = 1.5; for (let k = 1; k <= kmax; k++) ymax = Math.max(ymax, f1(k), fB(k)); ymax = Math.ceil(ymax * 2) / 2 + 0.25;
        const X = (k) => x0 + (w * (k - 1)) / (kmax - 1), Y = (s) => y0 - (h * s) / ymax;
        G.axes(x0, y0 - h, w, h, {});
        G.label(x0, 14, "speedup vs proposals per round k", { size: 11 });
        for (let s = 1; s <= ymax; s++) { G.line(x0, Y(s), x0 + w, Y(s), { color: s === 1 ? "ink" : "line", dash: "3 4", w: 1 }); G.label(x0 - 6, Y(s) + 4, s + "×", { anchor: "end", size: 10 }); }
        [1, 4, 8, 12, 16].forEach((k) => G.label(X(k), y0 + 14, String(k), { anchor: "middle", size: 10 }));
        const curve = (f, col) => { let d = ""; for (let k = 1; k <= kmax; k++) d += `${k > 1 ? " L" : "M"} ${X(k)} ${Y(f(k))}`; G.path(d, { color: col, w: 2.5 }); };
        curve(f1, "k"); if (v.B !== 1) curve(fB, "v");
        G.line(X(v.k), y0 - h, X(v.k), y0, { color: "ink", dash: "2 3", w: 1 });
        G.circle(X(v.k), Y(f1(v.k)), 5, { fill: "k" }); if (v.B !== 1) G.circle(X(v.k), Y(fB(v.k)), 5, { fill: "v" });
        G.label(x0, 262, "cyan: batch 1 · amber: your batch · dotted line: your k", { size: 10 });
        const eT = E(v.a, v.k), sB = fB(v.k), s1 = f1(v.k), kb = bestK(fB), kb1 = bestK(f1);
        return [
          { title: "Per target pass", rows: [["expected tokens E", eT.toFixed(2)], ["round cost at batch 1", (v.k * v.c + 1).toFixed(2) + " target steps"], ["verify rows per sequence", String(v.k + 1)]] },
          { title: "Speedup", rows: [["batch 1", s1.toFixed(2) + "×"]].concat(v.B === 1 ? [] : [[`batch ${v.B}`, sB.toFixed(2) + "×"]], [["best k at batch 1", `${kb1} (${f1(kb1).toFixed(2)}×)`]], v.B === 1 ? [] : [[`best k at batch ${v.B}`, `${kb} (${fB(kb).toFixed(2)}×)`]]),
            chip: [sB > 1.05, sB > 1.05 ? "speculation pays here" : sB >= 0.95 ? "break-even: not worth it" : "speculation loses throughput here"] },
          { title: "Model", rows: [], html: `<p class="note">Batch 1 uses the closed form E ÷ (k·c + 1) from core.py (i.i.d. acceptance). The batch extension is an assumption: step cost max(1, tokens ÷ R), draft cost c per step scaled the same way. Real engines add scheduler, sampling and KV effects; #10 and exercise 4 measure them.</p>` },
        ];
      },
    },
    practice: {
      intro: "Exercises 1–3 (tests) run on a laptop. Exercise 3's T2 table and exercise 4 need one GPU: aws.md uses g6.xlarge, servers on 127.0.0.1 with an API key, torn down with make down. The vLLM spec-decode flag format is UNVERIFIED at v0.31.0: check vllm serve --help.",
      items: [
        { title: "The verification rule preserves the distribution", tier: "T0 · medium", goal: "Implement verify(p, q, proposals, rng) in verify.py. A χ² test checks the first and second emitted tokens against p, and a planted bug (resampling from p) must fail.",
          cmd: "uv run pytest course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/exercises/test_verify.py" },
        { title: "E[tokens] formula vs simulation", tier: "T0 · easy", goal: "Write simulate(alpha, k, rounds, seed) and match platform/specdec's expected_tokens_per_round within 3%. Then decide whether clustered acceptance makes the formula optimistic or pessimistic.",
          cmd: "uv run pytest course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/exercises/test_formula.py" },
        { title: "#10 with a KV cache, and vs vLLM", tier: "T0 test + T2 run", goal: "Add HFModelCached to platform/specdec/hf_models.py (crop the target cache to the accepted length). The test checks greedy equivalence on tiny models; the T2 part fills a table against vLLM n-gram and draft-model spec decode.",
          cmd: "uv run pytest course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/exercises/test_cached_equivalence.py" },
        { title: "α by domain", tier: "T2 · hard", goal: "Measure α, tokens per pass and ITL at batch 1 and 16 for code-edit, open-chat and RAG prompt sets (30+ prompts each); explain the ranking and when you would turn spec decode off.",
          cmd: "PYTHONPATH=platform python -m specdec.bench --target /opt/models/<target> --draft /opt/models/draft --k 1 2 4 --prompts course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/examples/prompts_code.txt --tokens 64" },
      ],
      labs: [
        { label: "Expected tokens and speedup table, best k per (α, c)", path: "course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/examples/01_spec_math.py" },
        { label: "Exactness demo: correct rule vs a planted bug (χ² p-values)", path: "course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/examples/02_spec_sampling_toy.py" },
        { label: "Draft + target as two tiny HF models on CPU", path: "course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/examples/03_tiny_hf_spec.py" },
        { label: "#10, the speculative decoding prototype (core.py, hf_models.py, bench.py)", path: "platform/specdec/" },
        { label: "AWS guide: g6.xlarge, n-gram spec decode, cost, teardown", path: "course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/aws.md" },
        { label: "vLLM speculative decoding and CUDA graphs (read-only)", path: "vllm/v1/spec_decode/, docs/features/speculative_decoding/, docs/design/cuda_graphs.md, vllm/compilation/cuda_graph.py" },
        { label: "Animation: draft, verify, accept / reject, bonus token", path: "animations/p2-speculative-decoding.html" },
        { label: "Papers: Leviathan 2211.17192 · Chen 2302.01318 · Medusa 2401.10774 · EAGLE 2401.15077", path: "SOURCES.md" },
      ],
    },
  });
})();
