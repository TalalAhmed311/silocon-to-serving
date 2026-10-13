/* P6.5 — Spec-decode verification and GPU sampling: temperature, top-k, min-p, top-p (by threshold bisection, no sort),
   per-request seeded RNG, the rejection rule that preserves the target distribution, batched on the GPU.
   Mirrors platform/engine/v1/s2s_engine/sampling.py and spec_verify.py; every worked number was run on those functions. */
(function () {
  const F = S2S.fmt;
  const P6 = [0.5, 0.2, 0.15, 0.1, 0.04, 0.01]; // the running example distribution
  const cell = (G, x, y, w, h, t, color, o = {}) => G.box(x, y, w, h, t, Object.assign({ fill: color, size: 11, rx: 3 }, o));
  // vertical probability bars: vals, colors (fn i -> color), baseline y, left x, bar width, full height for p = 1
  const pbars = (G, x, base, vals, colorOf, o = {}) => {
    const w = o.w || 40, gap = o.gap || 10, H = o.H || 150, out = [];
    vals.forEach((v, i) => {
      const h = Math.max(1, H * v);
      out.push(G.rect(x + i * (w + gap), base - h, w, h, { fill: colorOf(i), rx: 3 }));
      if (o.labels !== false) G.label(x + i * (w + gap) + w / 2, base - h - 5, v.toFixed(o.dec == null ? 2 : o.dec), { size: 10, anchor: "middle", color: "ink" });
      if (o.ids) G.label(x + i * (w + gap) + w / 2, base + 14, o.ids[i] != null ? o.ids[i] : "t" + i, { size: 10, anchor: "middle" });
    });
    return out;
  };

  /* ---- JS port of filter_probs_np + bisection, tracking which stage dropped each token ---- */
  function pipeline(z, T, topk, minp, topp, iters) {
    const V = z.length, x = z.map((v) => v / Math.max(T, 1e-6));
    const m = Math.max(...x), e = x.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0), p0 = e.map((v) => v / s);
    const stage = new Array(V).fill(0); // 0 kept, 1 top-k, 2 min-p, 3 top-p
    let p = p0.slice();
    if (topk > 0 && topk < V) { const kth = p.slice().sort((a, b) => b - a)[topk - 1]; p = p.map((v, i) => (v >= kth ? v : (stage[i] = 1, 0))); }
    if (minp > 0) { const mx = Math.max(...p); p = p.map((v, i) => (v >= minp * mx ? v : (stage[i] || (stage[i] = 2), 0))); }
    let tau = 0, exactTau = 0;
    const sum = p.reduce((a, b) => a + b, 0), q = p.map((v) => v / sum);
    if (topp < 1) {
      let lo = 0, hi = Math.max(...q);
      for (let i = 0; i < iters; i++) { const mid = (lo + hi) / 2, mass = q.reduce((a, v) => a + (v >= mid ? v : 0), 0); if (mass >= topp) lo = mid; else hi = mid; }
      tau = lo;
      const sorted = q.slice().sort((a, b) => b - a); let c = 0, k = 0;
      while (k < sorted.length && c < topp - 1e-12) { c += sorted[k]; k++; }
      exactTau = sorted[Math.min(k, sorted.length) - 1];
      p = q.map((v, i) => (v >= tau ? v : (stage[i] || (stage[i] = 3), 0)));
    } else p = q;
    const s2 = p.reduce((a, b) => a + b, 0);
    return { p0, stage, final: p.map((v) => v / s2), tau, exactTau, kept: p.filter((v) => v > 0).length, exactKept: topp < 1 ? q.filter((v) => v >= exactTau).length : q.filter((v) => v > 0).length, massKept: p.reduce((a, b) => a + b, 0) };
  }

  S2S.lesson({
    id: "p6-5", n: "P6.5", title: "Spec-decode verification and GPU sampling",
    subtitle: "Engine internals · first principles · T0 NumPy and CPU PyTorch, T2 for GPU timings",
    kicker: "Lesson · ≈ 45 min",
    headline: "Sampling is a pipeline of filters, and verification is a rejection test",
    intro: `<p>The model's last layer produces one score per vocabulary token. Turning those scores into the <i>next token</i> is the job of the sampler, and it runs on every step for every sequence. This lesson builds it from the bottom: temperature, then the three truncations (top-k, min-p, top-p) in the right order, then the trick that finds the top-p cut-off without sorting a 128,000-entry vocabulary, then how to keep each request's randomness reproducible when the batch changes every step. Then the second half: how speculative decoding (P2.6) checks several draft tokens in one target pass with a rejection rule that provably leaves the output distribution unchanged. The references are <code>platform/engine/v1/s2s_engine/sampling.py</code> and <code>spec_verify.py</code>, which you re-implement in the exercises.</p>`,
    facts: ["9 steps", "5 checkpoints", "1 simulator", "3 exercises"],
    legend: [["k", "kept"], ["w", "dropped by top-k"], ["v", "dropped by min-p / draft q"], ["hot", "dropped by top-p / residual"], ["ok", "target p / accepted"]],
    prev: "p6-4", next: "p6-6",
    steps: [
      { rail: "logits to token", title: "Scores become probabilities, and temperature reshapes them",
        body: `<p>The forward pass ends with a vector of <b>logits</b>: one real number per vocabulary token (V ≈ 128,000 for Llama-3). Higher means the model likes that token more. To pick one, turn the scores into probabilities with the <b>softmax</b>: exponentiate and divide by the total, so they are positive and sum to 1.</p>
<p>Two ways to choose: <b>greedy</b> takes the largest (<code>argmax</code>, no randomness, <code>temperature = 0</code>); <b>sampling</b> draws a token at random in proportion to its probability. Sampling gives varied text, and it is what chat and creative use want.</p>
<p><b>Temperature</b> <i>T</i> divides the logits before the softmax. It makes the distribution sharper or flatter without reordering the tokens:</p>
<div class="eq">logits z = [2, 1, 0]       p = softmax(z / T)
T = 0.5 -> z/T = [4, 2, 0]   p = [0.867, 0.117, 0.016]
T = 1.0 -> z/T = [2, 1, 0]   p = [0.665, 0.245, 0.090]
T = 2.0 -> z/T = [1, .5, 0]  p = [0.506, 0.307, 0.186]</div>
<p>Low T concentrates the mass on the top token (as T → 0 it becomes greedy); high T spreads it out. The code treats <code>temperature &lt;= 0</code> as greedy and never divides by zero. In <code>sampling.py</code>, temperature is the <b>first</b> step, because every filter that follows looks at the reshaped probabilities.</p>`,
        check: { q: "You lower the temperature from 1.0 to 0.5. What happens to the probabilities of logits [2, 1, 0]?",
          options: ["The top token's probability rises (0.665 to 0.867) and the others fall", "All tokens become equally likely", "The order of the tokens changes", "Nothing: temperature only matters for greedy decoding"], answer: 0,
          why: "Dividing by 0.5 doubles the gaps between logits: [2, 1, 0] becomes [4, 2, 0], and the softmax of that puts 0.867 on the top token. Flatter is the opposite (T = 2 gives 0.506). The ranking never changes, only how much mass the leader gets." },
        scene(G) {
          G.text(24, 30, "softmax(z / T) for logits z = [2, 1, 0]", { size: 13 });
          const sets = [["T = 0.5 (sharper)", [0.867, 0.117, 0.016]], ["T = 1.0", [0.665, 0.245, 0.09]], ["T = 2.0 (flatter)", [0.506, 0.307, 0.186]]];
          const all = [];
          sets.forEach(([t, v], i) => {
            const x = 24 + i * 200;
            G.text(x, 60, t, { size: 12, weight: 700 });
            all.push(...pbars(G, x, 280, v, (j) => ["k", "q", "w"][j], { w: 40, gap: 12, H: 180, dec: 3, ids: ["z=2", "z=1", "z=0"] }));
          });
          G.label(24, 330, "same ranking in every panel; temperature only changes the gaps", { size: 12, color: "ink" });
          G.label(24, 352, "T = 0 is special-cased as greedy: argmax, no randomness", { size: 12 });
          G.label(24, 374, "temperature is applied first, so every filter sees the reshaped p", { size: 12 });
          G.from(all, { opacity: 0, y: 12, stagger: 0.04, duration: 0.3 });
          G.caption("temperature rescales the gaps between scores");
        } },

      { rail: "top-k + min-p", title: "Truncation, part 1: cut the unlikely tail",
        body: `<p>Even at T = 1, the long tail of the vocabulary holds thousands of tokens that are individually unlikely and, summed, not negligible. Sampling from the full tail now and then picks nonsense. <b>Truncation</b> sets the tail to zero and renormalises. Take this distribution over six tokens (the running example in this lesson):</p>
<div class="eq">p = [0.50, 0.20, 0.15, 0.10, 0.04, 0.01]</div>
<ul><li><b>Top-k</b> keeps the <i>k</i> most probable tokens. k = 3 keeps [0.50, 0.20, 0.15]; renormalised by 0.85: [0.588, 0.235, 0.176]. The code finds the k-th largest probability and keeps everything at or above it. Simple, but k ignores how confident the model is: 3 tokens is generous when one has 99 % and stingy when ten are equally plausible.</li>
<li><b>Min-p</b> keeps tokens with <code>p_i ≥ min_p · max(p)</code>, a threshold that scales with confidence. With min_p = 0.1 the cut is 0.1 × 0.50 = 0.05, keeping [0.50, 0.20, 0.15, 0.10] and dropping 0.04 and 0.01; renormalised by 0.95: [0.526, 0.211, 0.158, 0.105]. If the top token had 0.95, the threshold would be 0.095 and the tail would vanish; if the distribution were flat, almost everything would stay.</li></ul>
<p>I ran both on <code>filter_probs_np</code> and got these numbers. Both are plain element-wise comparisons against a threshold, which parallelises trivially on a GPU. Note that top-k never removes the largest probability, so min-p's reference value <code>max(p)</code> is the same before and after top-k.</p>`,
        scene(G) {
          G.text(24, 30, "p = [0.50, 0.20, 0.15, 0.10, 0.04, 0.01]", { size: 13 });
          const ids = ["t0", "t1", "t2", "t3", "t4", "t5"];
          G.label(24, 52, "top-k = 3: keep the 3 largest", { size: 11, color: "ink" });
          const a = pbars(G, 24, 200, P6, (i) => (i < 3 ? "k" : "w"), { w: 36, gap: 8, H: 190, ids });
          G.label(24 + 6 * 44 + 10, 150, "→ kept mass 0.85", { size: 12, color: "k" });
          G.label(24 + 6 * 44 + 10, 168, "renormalised: .588 .235 .176", { size: 11 });
          G.label(24, 232, "min-p = 0.1: keep p ≥ 0.1 × 0.50 = 0.05", { size: 11, color: "ink" });
          const b = pbars(G, 24, 390, P6, (i) => (i < 4 ? "k" : "v"), { w: 36, gap: 8, H: 190, ids });
          G.line(24, 390 - 190 * 0.05, 24 + 6 * 44 - 8, 390 - 190 * 0.05, { color: "v", w: 1.5, dash: "4 3" });
          G.label(24 + 6 * 44 + 10, 390 - 190 * 0.05 + 4, "cut 0.05", { size: 11, color: "v" });
          G.label(24 + 6 * 44 + 10, 300, "→ kept mass 0.95", { size: 12, color: "k" });
          G.label(24 + 6 * 44 + 10, 318, "renormalised: .526 .211 .158 .105", { size: 11 });
          G.from([...a, ...b], { opacity: 0, y: 10, stagger: 0.03, duration: 0.25 });
          G.caption("top-k cuts by rank, min-p cuts relative to the leader");
        } },

      { rail: "top-p", title: "Truncation, part 2: top-p keeps the smallest set that holds the mass",
        body: `<p><b>Top-p</b> (nucleus sampling) adapts the number of tokens to the shape of the distribution: keep the <i>smallest</i> set of tokens whose probabilities add up to at least <code>top_p</code>. When the model is sure, that set is one or two tokens; when it is unsure, it is many.</p>
<div class="eq">p sorted = [0.50, 0.20, 0.15, 0.10, 0.04, 0.01]
cumulative = [0.50, 0.70, 0.85, 0.95, 0.99, 1.00]
top_p = 0.9: 0.95 is the first sum >= 0.9 (4 tokens)
keep [0.50, 0.20, 0.15, 0.10], renormalise by 0.95</div>
<p>The textbook method sorts the probabilities, takes the running sum, and cuts where it passes <code>top_p</code>. For one row of 6 that is trivial. For a real step, the sampler must do it for <b>every sequence in the batch</b> over a vocabulary of about 128,000: a sort of [B, V] per step, only to find one number per row. Sorting is not the kind of work a GPU does best, and nothing here needs the order, only the <b>threshold</b>.</p>
<p>The cut-off can be written as a value, not a rank. Keep the tokens with <code>p_i ≥ τ</code>, where τ is the probability of the last token that was kept. Here τ = 0.10: the token with p = 0.10 is the smallest one in the set, and everything smaller is out. So the question becomes: <b>what is the largest τ such that the tokens at or above it still hold at least top_p of the mass?</b> The next step finds it without a sort.</p>`,
        check: { q: "p = [0.50, 0.20, 0.15, 0.10, 0.04, 0.01] and top_p = 0.9. How many tokens does top-p keep?",
          options: ["3 (mass 0.85 is closest to 0.9)", "4 (mass 0.95 is the first to reach 0.9)", "5", "2"], answer: 1,
          why: "Top-p keeps the smallest set whose mass is at least top_p. Three tokens give 0.85, which is below 0.9, so a fourth is needed: 0.95. Stopping at 0.85 would keep less mass than requested, which is not what top_p = 0.9 means." },
        scene(G) {
          G.text(24, 30, "sorted p, running sum, top_p = 0.9", { size: 13 });
          const base = 230, H = 160, cum = [0.5, 0.7, 0.85, 0.95, 0.99, 1.0];
          const b = pbars(G, 30, base, P6, (i) => (i < 4 ? "k" : "hot"), { w: 44, gap: 14, H, labels: false, ids: ["t0 .50", "t1 .20", "t2 .15", "t3 .10", "t4 .04", "t5 .01"] });
          // cumulative line
          let d = ""; cum.forEach((c, i) => { d += `${i ? "L" : "M"} ${30 + i * 58 + 22} ${base - H * c} `; });
          const cl = G.path(d, { color: "v", w: 2.5 });
          cum.forEach((c, i) => { G.circle(30 + i * 58 + 22, base - H * c, 3.5, { fill: "v" }); G.label(30 + i * 58 + 22, base - H * c - 9, c.toFixed(2), { size: 10, anchor: "middle", color: "v" }); });
          G.line(24, base - H * 0.9, 400, base - H * 0.9, { color: "ok", w: 1.5, dash: "5 4" });
          G.label(406, base - H * 0.9 + 4, "top_p = 0.9", { size: 11, color: "ok" });
          G.line(30 + 4 * 58 - 7, base + 2, 30 + 4 * 58 - 7, base - H - 4, { color: "hot", w: 2 });
          G.label(30 + 4 * 58 - 3, base + 30, "cut: 4 kept", { size: 10, color: "hot" });
          G.line(30, base - H * 0.1, 30 + 6 * 58 - 14, base - H * 0.1, { color: "ink", w: 1, dash: "2 3" });
          G.label(404, base - H * 0.1 + 4, "τ = 0.10", { size: 11, color: "ink" });
          G.text(24, 300, "keep the smallest set with mass ≥ top_p", { size: 14, color: "ink" });
          G.label(24, 326, "sort + running sum finds it, but sorting [B, V] every step is costly", { size: 12 });
          G.label(24, 348, "the cut can be a VALUE: keep p ≥ τ, τ = 0.10 here", { size: 12 });
          G.label(24, 370, "→ find τ without sorting (next step)", { size: 12, color: "ok" });
          G.from(b, { opacity: 0, y: 10, stagger: 0.04, duration: 0.25 }); G.from(cl, { opacity: 0, duration: 0.5, delay: 0.3 });
          G.caption("top-p adapts the set size to how sure the model is");
        } },

      { rail: "bisection", title: "Finding the top-p threshold by bisection, no sort",
        body: `<p>Define <code>mass(τ) = Σ p_i over tokens with p_i ≥ τ</code>. Raising τ can only remove tokens, so <code>mass(τ)</code> never goes up as τ grows: it is <b>monotone non-increasing</b>. It starts at 1 (τ = 0 keeps everything) and falls in steps to 0 (τ above the maximum). We want the largest τ with <code>mass(τ) ≥ top_p</code>. For a monotone function that is a textbook <b>bisection</b>:</p>
<div class="eq">lo = 0, hi = max(p)
repeat 30-40 times:
    mid = (lo + hi) / 2
    if mass(mid) >= top_p: lo = mid
    else:                  hi = mid
keep tokens with p_i >= lo</div>
<p>On the six-token example with top_p = 0.9 (I ran <code>top_p_threshold_np</code> and printed each step):</p>
<table><tr><th>iter</th><th>mid</th><th>mass(mid)</th><th>action</th><th>kept at lo</th></tr>
<tr><td>1</td><td>0.25</td><td>0.50</td><td>hi = 0.25</td><td>6</td></tr>
<tr><td>2</td><td>0.125</td><td>0.85</td><td>hi = 0.125</td><td>6</td></tr>
<tr><td>3</td><td>0.0625</td><td>0.95</td><td>lo = 0.0625</td><td>4</td></tr>
<tr><td>4</td><td>0.09375</td><td>0.95</td><td>lo = 0.09375</td><td>4</td></tr>
<tr><td>5</td><td>0.109</td><td>0.85</td><td>hi = 0.109</td><td>4</td></tr>
<tr><td>8</td><td>0.0996</td><td>0.95</td><td>lo = 0.0996</td><td>4</td></tr></table>
<p><code>lo</code> closes in on 0.10 from below (after 40 iterations it is 0.09999999999991, and <code>p ≥ lo</code> keeps the 0.10 token), matching the sort-based <code>top_p_threshold_sorted</code>. Each iteration is "compare every probability to a number and sum the ones above": one parallel reduction over V. The GPU version <code>sample_torch</code> does 30 iterations for the whole [B, V] matrix at once, a separate τ per row, so the cost is about 30 reads of the probabilities. 30 halvings narrow the interval to max(p) / 2³⁰, about 10⁻⁹ of the maximum. Because <code>lo</code> only moves up when the mass test passes, the set it keeps always holds at least <code>top_p</code> of the mass, even if you stop early.</p>`,
        check: { q: "Why can the top-p cut-off be found by bisection on τ?",
          options: ["Probabilities are sorted, so binary search applies to the array", "mass(τ), the total probability of tokens with p ≥ τ, is monotone non-increasing in τ, so the largest τ with mass ≥ top_p can be bracketed and halved", "Bisection is exact after 2 iterations", "Softmax outputs are always uniform"], answer: 1,
          why: "Raising τ can only drop tokens, so mass(τ) never increases. A monotone function with a threshold test is exactly what bisection solves: if mass(mid) is still enough, the answer is at or above mid; otherwise below. The probabilities are not sorted (that is the point), and each iteration is one parallel sum." },
        scene(G) {
          G.text(24, 30, "mass(τ) falls as τ rises (top_p = 0.9)", { size: 13 });
          const x0 = 60, y0 = 70, W = 540, H = 200;
          G.axes(x0, y0, W, H);
          G.label(x0 - 6, y0 + 4, "1.0", { size: 10, anchor: "end" }); G.label(x0 - 6, y0 + H + 4, "0", { size: 10, anchor: "end" });
          G.label(x0 + W, y0 + H + 18, "τ (0 to 0.27)", { size: 10, anchor: "end" });
          const X = (t) => x0 + (W * t) / 0.27, Y = (m) => y0 + H - H * m;
          // step function: mass(τ)
          const edges = [[0, 1.0], [0.01, 0.99], [0.04, 0.95], [0.1, 0.85], [0.15, 0.65], [0.2, 0.5], [0.5, 0]];
          let d = `M ${X(0)} ${Y(1.0)}`;
          [[0.01, 1.0, 0.99], [0.04, 0.99, 0.95], [0.1, 0.95, 0.85], [0.15, 0.85, 0.65], [0.2, 0.65, 0.5]].forEach(([t, a, b]) => { d += ` L ${X(t)} ${Y(a)} L ${X(t)} ${Y(b)}`; });
          d += ` L ${X(0.27)} ${Y(0.5)}`;
          const st = G.path(d, { color: "k", w: 3 });
          G.line(x0, Y(0.9), x0 + W, Y(0.9), { color: "ok", w: 1.5, dash: "5 4" });
          G.label(x0 + W - 4, Y(0.9) - 6, "top_p = 0.9", { size: 11, color: "ok", anchor: "end" });
          const mids = [[0.25, 1], [0.125, 2], [0.0625, 3], [0.09375, 4], [0.109, 5], [0.1016, 6]];
          const mk = [];
          mids.forEach(([t, i]) => { const m = P6.reduce((a, v) => a + (v >= t ? v : 0), 0); mk.push(G.circle(X(t), Y(m), 5, { fill: m >= 0.9 ? "ok" : "hot" })); G.label(X(t) + (i === 5 ? 9 : i === 6 ? -9 : 0), Y(m) - 10 - (i === 4 ? 12 : 0), String(i), { size: 11, anchor: "middle", color: "ink" }); });
          G.line(X(0.1), y0 + H, X(0.1), y0 + 10, { color: "ink", w: 1, dash: "2 3" });
          G.label(X(0.1) + 4, y0 + H + 18, "τ = 0.10", { size: 11, color: "ink" });
          G.label(24, 330, "green dot: mass ≥ top_p → lo moves up · red dot: too little → hi moves down", { size: 11, color: "ink" });
          G.label(24, 352, "numbers = iteration; they close in on the step at τ = 0.10", { size: 11 });
          G.text(24, 386, "each iteration = one parallel sum over V", { size: 13, color: "ink" });
          G.from(st, { opacity: 0, duration: 0.5 }); G.from(mk, { opacity: 0, scale: 0.2, stagger: 0.25, duration: 0.2, delay: 0.4 });
          G.caption("a monotone staircase: halve the interval until it pins the step");
        } },

      { rail: "order", title: "The filters run in a fixed order, and renormalising matters",
        body: `<p>The order of operations in <code>filter_probs_np</code> (and <code>sample_torch</code>) is:</p>
<ol><li><b>Temperature</b> and softmax: reshape the distribution.</li>
<li><b>Top-k</b>: drop everything below the k-th largest probability.</li>
<li><b>Min-p</b>: drop everything below <code>min_p · max(p)</code>.</li>
<li><b>Renormalise</b>, then <b>top-p</b> by threshold search.</li>
<li>Renormalise again and <b>sample</b>.</li></ol>
<p>Why this order? Temperature first, because every threshold is a statement about probabilities, and temperature changes them: if you applied top-p first and then temperature, the kept set would not be the nucleus of the distribution the user asked for. Renormalise before top-p, because top-k and min-p set entries to zero, so the vector no longer sums to 1, and top-p's test <code>mass ≥ top_p</code> is about mass <i>relative to what is left</i>.</p>
<div class="eq">p = [.50, .20, .15, .10, .04, .01]
top_k = 5      drops .01             mass .99
min_p = 0.1    cut .1 * .5 = .05     drops .04
               left [.5,.2,.15,.1]   mass .95
renormalise    [.526, .211, .158, .105]
top_p = 0.85   cumulative .526, .737, .895
               first >= .85 is .895 (3 tokens)
keep [.526, .211, .158] / .895
final p = [.588, .235, .176]</div>
<p>The final line is what <code>filter_probs_np</code> returned when I ran it with those settings. The sampler then draws one token from this vector. A request with <code>temperature = 0</code> skips all of it: <code>sample_np</code> returns the argmax.</p>
<p>A slip here is silent: nothing crashes, the text is just a bit worse or the settings behave unlike the same settings in another engine. That is why the tests compare the filtered distribution to a reference, not just the sampled tokens.</p>`,
        scene(G) {
          G.text(24, 30, "the pipeline, one row of 6 logits", { size: 13 });
          const stages = [
            ["softmax(z/T)", [0.5, 0.2, 0.15, 0.1, 0.04, 0.01], (i) => "k", "mass 1.00"],
            ["top-k = 5", [0.5, 0.2, 0.15, 0.1, 0.04, 0], (i) => (i < 5 ? "k" : "w"), "mass 0.99"],
            ["min-p = 0.1", [0.5, 0.2, 0.15, 0.1, 0, 0], (i) => (i < 4 ? "k" : "v"), "mass 0.95"],
            ["top-p = 0.85", [0.526, 0.211, 0.158, 0, 0, 0], (i) => (i < 3 ? "k" : "hot"), "after renorm"],
          ];
          const out = [];
          stages.forEach(([t, v, c, m], r) => {
            const x = 24 + (r % 2) * 310, y = 66 + Math.floor(r / 2) * 150;
            G.text(x, y, (r + 1) + ". " + t, { size: 12, weight: 700 });
            G.label(x + 190, y, m, { size: 10 });
            const full = [0.5, 0.2, 0.15, 0.1, 0.04, 0.01];
            full.forEach((f, i) => { G.rect(x + i * 46, y + 12 + 100 - 100 * f * 1.6, 38, 100 * f * 1.6, { stroke: "line", rx: 2, dash: "2 3" }); });
            out.push(...pbars(G, x, y + 112, v.map((q) => q), c, { w: 38, gap: 8, H: 160, dec: 2, labels: false }));
            v.forEach((q, i) => { if (q > 0) G.label(x + i * 46 + 19, y + 112 - 160 * q - 4, q.toFixed(2).replace(/^0/, ""), { size: 9, anchor: "middle", color: "ink" }); });
          });
          G.text(24, 372, "final after the last renormalise: .588 .235 .176", { size: 13, color: "ok" });
          G.label(24, 396, "dashed outline = the original softmax bar", { size: 10 });
          G.from(out, { opacity: 0, stagger: 0.02, duration: 0.2 });
          G.caption("temperature, then truncations, renormalising before top-p");
        } },

      { rail: "seeded RNG", title: "Reproducible per request: one generator each",
        body: `<p>A user passes <code>seed = 42</code> and expects the same output on every run. But the engine's batch changes every step: other requests join, finish, get preempted. If the whole engine shared one random number generator, a request's draw on step 5 would depend on <i>how many other tokens were sampled before it in that step</i>, which depends on who else happened to be running. Same seed, different text.</p>
<p>The fix is structural: <b>one generator per request</b>, consumed exactly once per sampled token. The engine (<code>engine.py</code>) creates it when the request is added:</p>
<div class="eq">self.rngs[seq.seq_id] = np.random.default_rng(
    params.seed if params.seed is not None
    else seq.seq_id)
# per step, per sequence that produced a token:
sampled[sid] = sample_np(row, seq.params,
                         self.rngs[sid])</div>
<p>Now a request's stream is a function of its seed and its own history only. Another request in the batch never touches its generator. The same idea applies on the GPU with a <code>torch.Generator</code> per request (or, in a fused kernel, a per-row Philox seed and offset; exercise 3). Three details:</p>
<ul><li>The generator is advanced <i>only</i> when a token is actually sampled. A chunked-prefill step that produces no token (P6.2) must not consume a draw, or the stream would shift with the chunking.</li>
<li>A request with no seed still gets one (its <code>seq_id</code>), so runs are repeatable inside one process; give users a real seed if you want cross-process repeatability.</li>
<li>Reproducibility here means the <i>sampling</i> is deterministic given the logits. Logits themselves can differ in the last bits when the batch shape changes, because floating-point sums are order-dependent (P5.4).</li></ul>
<p><code>test_seeded_sampling_reproducible</code> checks the sampling half: same seed and logits, same tokens.</p>`,
        scene(G) {
          G.text(24, 30, "same seed, different batch-mates, same stream", { size: 13 });
          const run = (y, title, others) => {
            G.text(24, y, title, { size: 12, weight: 700 });
            const out = [];
            others.forEach((o, i) => {
              const yy = y + 14 + i * 34;
              G.label(24, yy + 20, o[0], { size: 11, color: o[1] });
              o[2].forEach((t, j) => out.push(cell(G, 100 + j * 70, yy, 62, 28, t, o[1], { size: 12 })));
            });
            return out;
          };
          const a = run(52, "run 1: A is batched with B", [["A seed 42", "k", ["71", "5", "309", "12"]], ["B", "w", ["8", "8", "90", "4"]]]);
          const b = run(176, "run 2: A is batched with B and C", [["A seed 42", "k", ["71", "5", "309", "12"]], ["B", "w", ["8", "8", "90", "4"]], ["C", "w", ["33", "7", "7", "61"]]]);
          G.rect(94, 64, 296 + 2, 32, { stroke: "ok", rx: 6, sw: 2, dash: "4 3" }); G.rect(94, 188, 298, 32, { stroke: "ok", rx: 6, sw: 2, dash: "4 3" });
          G.label(400, 82, "identical tokens for A", { size: 12, color: "ok" });
          G.label(400, 206, "identical tokens for A", { size: 12, color: "ok" });
          G.text(24, 322, "each request owns np.random.default_rng(seed)", { size: 13, color: "ink" });
          G.label(24, 346, "advanced once per sampled token, never by other requests", { size: 12 });
          G.label(24, 368, "a step that yields no token (mid-prompt chunk) draws nothing", { size: 12 });
          G.label(24, 392, "illustrative token ids, not a measurement", { size: 10 });
          G.from([...a, ...b], { opacity: 0, x: -8, stagger: 0.02, duration: 0.2 });
          G.caption("one generator per request keeps its stream independent of the batch");
        } },

      { rail: "verification rule", title: "Speculative verification: accept with probability min(1, p/q)",
        body: `<p>P2.6 introduced speculative decoding: a small, cheap <b>draft</b> model guesses <i>k</i> tokens ahead, and the large <b>target</b> model checks all of them in <i>one</i> forward pass (decode is memory-bound, so checking <i>k</i>+1 positions costs little more than one). If the guesses are right, you got several tokens for the price of one target pass. The question this lesson answers: what exactly counts as "right", if you also need the output to follow the target model's distribution <i>exactly</i>?</p>
<p>Let the draft's distribution at a position be <i>q</i> and the target's be <i>p</i>. The draft samples a token <i>x</i> from <i>q</i>. The verifier (in <code>spec_verify.verify_batch</code>):</p>
<ol><li><b>Accept</b> <i>x</i> with probability <code>min(1, p(x) / q(x))</code>. If the target likes <i>x</i> at least as much as the draft did, always accept.</li>
<li>If <b>rejected</b>, sample a replacement from the <b>residual</b> <code>max(p − q, 0)</code> (renormalised) and stop: later draft tokens were conditioned on a wrong one.</li>
<li>If all <i>k</i> are accepted, sample one <b>bonus</b> token from the target's distribution at position <i>k</i> + 1. The same forward pass already computed it.</li></ol>
<p>Check that it preserves <i>p</i> on a tiny vocabulary of 3 tokens:</p>
<div class="eq">p = [0.5, 0.3, 0.2]     q = [0.2, 0.5, 0.3]
accept prob: t0 min(1, 2.5) = 1
             t1 min(1, 0.6) = 0.6
             t2 min(1, 0.667) = 0.667
residual max(p-q,0) = [0.3, 0, 0] -> all on t0
P(emit t0) = 0.2*1 + (0.5*0.4 + 0.3*0.333)*1 = 0.5
P(emit t1) = 0.5 * 0.6              = 0.3
P(emit t2) = 0.3 * 0.667            = 0.2</div>
<p>The emitted token is distributed exactly as <i>p</i>. I also ran <code>verify_batch</code> 20,000 times on this case and got [0.500, 0.299, 0.201]. This is Theorem 1 of Leviathan et al.; the residual is what repairs the probability the draft over-claimed on tokens 1 and 2. Another case: <i>p</i> = 0.1 and <i>q</i> = 0.4 give acceptance 0.1 / 0.4 = 0.25.</p>`,
        check: { q: "A draft token has p = 0.1 under the target and q = 0.4 under the draft. With what probability is it accepted, and what happens on rejection?",
          options: ["1.0; nothing", "0.25; sample from max(p − q, 0) renormalised at that position, then stop", "0.25; keep going with the next draft token", "0.4; resample from p"], answer: 1,
          why: "Acceptance is min(1, p/q) = min(1, 0.1/0.4) = 0.25. On rejection the verifier samples a replacement from the residual max(p − q, 0), renormalised, and stops, because the remaining draft tokens were conditioned on a token that was not accepted. Resampling from p itself would double-count the part of p the draft already covers." },
        scene(G) {
          G.text(24, 30, "p = target, q = draft, 3 tokens", { size: 13 });
          const p = [0.5, 0.3, 0.2], q = [0.2, 0.5, 0.3], res = [1, 0, 0];
          const base = 200, H = 130;
          const draw = (x, vals, col, title) => { G.text(x, 62, title, { size: 12, weight: 700, color: col }); return pbars(G, x, base, vals, () => col, { w: 38, gap: 10, H, ids: ["t0", "t1", "t2"], dec: 2 }); };
          const a = draw(24, p, "ok", "p (target)");
          const b = draw(200, q, "v", "q (draft)");
          const c = draw(376, [0.3, 0, 0], "hot", "max(p − q, 0)");
          G.label(376, base + 32, "renormalised: [1, 0, 0]", { size: 11, color: "hot" });
          G.text(24, 250, "accept prob = min(1, p/q):   t0: 1    t1: 0.6    t2: 0.667", { size: 12, color: "ink" });
          G.label(24, 280, "P(emit t0) = 0.2×1 + (0.5×0.4 + 0.3×0.333) × 1 = 0.5 = p0", { size: 12 });
          G.label(24, 302, "P(emit t1) = 0.5 × 0.6   = 0.3 = p1", { size: 12 });
          G.label(24, 324, "P(emit t2) = 0.3 × 0.667 = 0.2 = p2", { size: 12 });
          G.text(24, 372, "accept the draft, or repair with the residual: always lands on p", { size: 13, color: "ok" });
          G.from([...a, ...b, ...c], { opacity: 0, y: 10, stagger: 0.04, duration: 0.25 });
          G.caption("the residual puts back exactly the mass the draft over-claimed");
        } },

      { rail: "tokens per pass", title: "What it buys: expected tokens per target pass",
        body: `<p>Suppose each draft token is accepted with probability α, independently (a simplification). A round proposes <i>k</i> tokens. The tokens emitted are: the accepted prefix, plus one more token (the replacement after the first rejection, or the bonus if all <i>k</i> were accepted). So at least 1 and at most <i>k</i> + 1 tokens per target pass:</p>
<div class="eq">E[tokens] = 1 + α + α^2 + ... + α^k
          = (1 - α^(k+1)) / (1 - α)

α = 0.8:  k=1  1.800    k=2  2.440    k=3  2.952
          k=4  3.362    k=8  4.329</div>
<p>Those are <code>platform/specdec/core.py::expected_tokens_per_round</code> (<code>(1 − α^{k+1}) / (1 − α)</code>; 3.36 at k = 4). Gains flatten: going from k = 4 to k = 8 doubles the draft work and adds under one token. And a round isn't free: the draft runs <i>k</i> times. If one draft step costs <i>c</i> times a target step (in the same units), the speedup over plain decoding is roughly:</p>
<div class="eq">speedup = E[tokens] / (k * c + 1)
α = 0.8, k = 4, c = 0.1: 3.362 / 1.4 = 2.40</div>
<p>(<code>expected_speedup</code> in the same file; <i>c</i> = 0.1 is an example value, measure yours.) The batched verifier itself is cheap: it reads <code>B · (k + 1) · V</code> probabilities once, an element-wise accept test plus a first-rejection scan per row (on the GPU, a prefix scan over the accept flags, Lane B L6 #87) and one residual sample. Nearly all the time is in the target forward pass.</p>
<p>A cost the formula hides: rejected positions already had K/V written and blocks allocated. In the engine (P6.2/P6.3) a spec-decode sequence's decode chunk becomes <code>1 + k</code> tokens, and <code>update()</code> must keep only the accepted ones and free blocks that held rejected positions. P6.7's stretch list covers it.</p>`,
        check: { q: "k = 4 draft tokens, all accepted. How many tokens does that step emit?",
          options: ["4", "5", "1", "8"], answer: 1,
          why: "The k = 4 accepted drafts plus a bonus token sampled from the target's distribution at position k + 1, which the same forward pass already computed. That is why the maximum is k + 1 per round and the minimum is 1 (the first draft rejected, so you emit its replacement)." },
        scene(G) {
          G.text(24, 30, "E[tokens per target pass], α = 0.8", { size: 13 });
          const x0 = 70, y0 = 60, W = 520, H = 220;
          G.axes(x0, y0, W, H);
          G.label(x0 - 6, y0 + 4, "5", { size: 10, anchor: "end" }); G.label(x0 - 6, y0 + H + 4, "1", { size: 10, anchor: "end" });
          G.label(x0 + W, y0 + H + 34, "k (draft tokens per round)", { size: 11, anchor: "end" });
          const X = (k) => x0 + 20 + ((W - 40) * (k - 1)) / 7, Y = (e) => y0 + H - (H * (e - 1)) / 4;
          const E = (k) => (1 - Math.pow(0.8, k + 1)) / 0.2;
          let d = ""; for (let k = 1; k <= 8; k++) d += `${k > 1 ? "L" : "M"} ${X(k)} ${Y(E(k))} `;
          const ln = G.path(d, { color: "k", w: 3 });
          const mk = [];
          for (let k = 1; k <= 8; k++) { mk.push(G.circle(X(k), Y(E(k)), 4, { fill: k === 4 ? "v" : "k" })); G.label(X(k), y0 + H + 16, String(k), { size: 10, anchor: "middle" }); }
          G.line(x0, Y(5), x0 + W, Y(5), { color: "line", dash: "3 4" }); G.label(x0 + W, Y(5) - 4, "ceiling for k=4: 5", { size: 10, anchor: "end" });
          G.text(X(4) + 8, Y(E(4)) + 22, "k = 4: 3.362", { size: 12, color: "v", weight: 700 });
          G.text(X(8) - 6, Y(E(8)) - 12, "k = 8: 4.329", { size: 12, color: "k", anchor: "end" });
          G.text(24, 322, "E = (1 − α^(k+1)) / (1 − α)", { size: 14, color: "ink" });
          G.label(24, 348, "gains flatten; every extra draft step costs c target steps", { size: 12 });
          G.label(24, 370, "speedup = E / (k·c + 1);  α=.8, k=4, c=.1 → 3.362 / 1.4 = 2.40", { size: 12 });
          G.label(24, 392, "c = 0.1 is an example value; measure your draft model", { size: 10 });
          G.from(ln, { opacity: 0, duration: 0.6 }); G.from(mk, { opacity: 0, stagger: 0.07, duration: 0.2 });
          G.caption("more drafts help less and less");
        } },

      { rail: "on the GPU", title: "Batched on the GPU: the engine's sampling step",
        body: `<p>At engine level (<code>engine.py</code>) the step after the forward pass has three parts: take the rows of logits for the sequences that produced a token; apply each sequence's own parameters and generator; hand the token ids back to the scheduler (P6.2 <code>update()</code>). <code>LLMEngine.step</code> uses <code>sample_np</code> row by row: simple and enough for T0. The GPU path does it for all rows at once:</p>
<ul><li>Gather the rows that produced a token into <code>logits [B, V]</code>.</li>
<li>Build per-row parameter tensors <b>once per step</b>: <code>temperature [B]</code>, <code>top_k [B]</code>, <code>top_p [B]</code>, <code>min_p [B]</code>. Each request can have different settings; a row with <code>temperature == 0</code> takes the argmax (<code>torch.where(greedy, argmax, sampled)</code>).</li>
<li>Call <code>sample_torch</code>: softmax of <code>logits / T</code>, a <code>topk</code> for top-k, a compare for min-p, 30 bisection iterations for top-p (each: one [B, V] comparison and a row sum), then <code>torch.multinomial</code>.</li>
<li>Copy B integers back to the CPU.</li></ul>
<p>Why this matters for speed: the sampler's work is a few passes over a [B, V] matrix. For B = 256 and V = 128,256 that is 32.8 million floats per pass (V of Llama-3 from P1.2). A sort-based top-p adds a [B, V] sort; bisection replaces it with about 30 more passes of a simple reduction. Which is faster depends on the hardware, which is why <code>bench/sampler_bench.py</code> compares both; the GPU rows of its table are <code>TODO(run-on: g6.xlarge)</code> and on a CPU the numbers mean little. Production engines go further (FlashInfer's sorting-free rejection sampling, fused kernels like exercise 3).</p>
<p>The verifier gets the same treatment: <code>verify_batch</code> is the NumPy reference with a Python loop over sequences and positions, and exercise 1's T2 half asks for the batched PyTorch version with no loop over positions: an accept mask <code>[B, k]</code>, a first-rejection index via <code>cumprod</code>, one <code>multinomial</code> for all residuals.</p>
<p>Common mistakes the tests and README list: top-p before temperature; a threshold search on an unnormalised vector after top-k/min-p; one global RNG for the batch; a residual taken from <code>p − q</code> without clamping at 0; forgetting the bonus token when all <i>k</i> are accepted.</p>`,
        scene(G) {
          const stage = (x, y, w, h, t, sub, col) => { G.box(x, y, w, h, "", { stroke: col, rx: 10 }); G.text(x + 10, y + 22, t, { size: 12, weight: 700, color: col }); G.label(x + 10, y + 40, sub, { size: 10 }); };
          stage(24, 46, 170, 54, "forward pass", "logits for all rows", "w");
          G.arrow(196, 73, 226, 73, { color: "ink", w: 2 });
          stage(230, 46, 180, 54, "gather B rows", "[B, V] logits", "k");
          G.arrow(412, 73, 442, 73, { color: "ink", w: 2 });
          stage(446, 46, 170, 54, "per-row params", "T, top_k, top_p, min_p [B]", "v");
          G.arrow(530, 102, 530, 128, { color: "ink", w: 2 });
          G.box(24, 130, 592, 130, "", { stroke: "ok", rx: 12 });
          G.text(36, 154, "sample_torch(logits, T, top_k, top_p, min_p)", { size: 13, weight: 700, color: "ok" });
          ["softmax(logits / T)", "top-k: topk + compare", "min-p: compare to min_p · max", "top-p: 30 × (compare, row-sum)", "multinomial (per-request generator)"].forEach((t, i) => G.label(40 + (i % 2) * 290, 180 + Math.floor(i / 2) * 24, "· " + t, { size: 12, color: "ink" }));
          G.arrow(320, 262, 320, 292, { color: "ink", w: 2 });
          stage(24, 296, 280, 54, "B token ids → CPU", "to the scheduler: update()", "q");
          stage(330, 296, 286, 54, "spec decode: verify_batch", "accept mask · first rejection", "hot");
          G.label(24, 380, "one pass of simple kernels, no [B, V] sort", { size: 12, color: "ink" });
          G.label(24, 400, "GPU timings are TODO(run-on: g6.xlarge): none quoted here", { size: 10 });
          G.caption("the whole batch is sampled in a few passes over [B, V]");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Logits become probabilities with softmax; temperature divides the logits first (0 means greedy). Then the truncations in a fixed order: top-k, min-p, renormalise, top-p, sample.</li>
<li>Top-p keeps the smallest set with mass ≥ top_p. Because <code>mass(τ)</code> is monotone in τ, the threshold is found by 30 to 40 bisection steps of one parallel sum each, no [B, V] sort.</li>
<li>Reproducibility needs one generator per request, consumed once per sampled token.</li>
<li>Verification accepts a draft token with probability <code>min(1, p/q)</code>, resamples from <code>max(p − q, 0)</code> on the first rejection, and adds a bonus token if all <i>k</i> pass. The output distribution is exactly the target's.</li>
<li>Expected tokens per pass are <code>(1 − α^(k+1)) / (1 − α)</code> (3.36 for α = 0.8, k = 4); the speedup divides by <code>k·c + 1</code>.</li>
<li>On the GPU the sampler and verifier run batched over [B, V] with per-row parameters.</li></ul>`,
    sim: {
      title: "The sampling pipeline and the speculative-decoding payoff",
      intro: "Left: a made-up distribution over 24 tokens (logits 4 / (1 + 0.3 i), an example, not a model). Choose the temperature and filters; bars show the probabilities after temperature and are coloured by what dropped them: grey top-k, amber min-p, red top-p. Cut the bisection short to see that top-p keeps a bit too much until enough iterations have run. Right: the speculative-decoding formulas from platform/specdec/core.py.",
      height: 280,
      controls: [
        { id: "T", label: "temperature", min: 0.2, max: 3, step: 0.1, value: 1, format: (x) => x.toFixed(1) },
        { id: "topk", label: "top-k (0 = off)", min: 0, max: 24, step: 1, value: 0 },
        { id: "minp", label: "min-p", min: 0, max: 0.5, step: 0.01, value: 0, format: (x) => x.toFixed(2) },
        { id: "topp", label: "top-p (1 = off)", min: 0.05, max: 1, step: 0.01, value: 0.9, format: (x) => x.toFixed(2) },
        { id: "iters", label: "bisection iterations", min: 0, max: 30, step: 1, value: 30 },
        { id: "alpha", label: "acceptance α", min: 0.1, max: 0.99, step: 0.01, value: 0.8, format: (x) => x.toFixed(2) },
        { id: "k", label: "draft tokens k", min: 1, max: 12, step: 1, value: 4 },
        { id: "c", label: "draft cost c (target steps, example)", min: 0.01, max: 0.5, step: 0.01, value: 0.1, format: (x) => x.toFixed(2) },
      ],
      draw(G, v) {
        const V = 24, z = Array.from({ length: V }, (_, i) => 4 / (1 + 0.3 * i));
        const r = pipeline(z, v.T, v.topk, v.minp, v.topp, v.iters);
        const x0 = 40, W = 580, base = 230, H = 190, bw = W / V;
        G.axes(x0, base - H, W, H);
        G.label(10, 16, "p after temperature, token 0 … 23 (colour = which filter dropped it)", { size: 11 });
        const mx = Math.max(...r.p0) * 1.05;
        const col = ["k", "w", "v", "hot"];
        r.p0.forEach((p, i) => { const h = (H * p) / mx; G.rect(x0 + i * bw + 1, base - h, bw - 2, h, { fill: col[r.stage[i]], rx: 1 }); });
        if (v.topp < 1) {
          const tauN = r.tau * 1; // relative to renormalised vector; draw on the original scale via kept mass
          const keptMass0 = r.p0.reduce((a, p, i) => a + (r.stage[i] === 1 || r.stage[i] === 2 ? 0 : p), 0);
          G.line(x0, base - (H * (r.tau * keptMass0)) / mx, x0 + W, base - (H * (r.tau * keptMass0)) / mx, { color: "ink", dash: "3 4" });
          G.label(x0 + W, base - (H * (r.tau * keptMass0)) / mx - 4, "top-p threshold τ", { size: 10, anchor: "end" });
        }
        G.label(x0, base + 18, "token 0", { size: 10 }); G.label(x0 + W, base + 18, "token 23", { size: 10, anchor: "end" });
        G.label(10, 270, "cyan kept · grey top-k · amber min-p · red top-p · dashed line: τ rescaled to this chart", { size: 10 });
        const E = v.alpha >= 1 ? v.k + 1 : (1 - Math.pow(v.alpha, v.k + 1)) / (1 - v.alpha);
        const spd = E / (v.k * v.c + 1);
        const extra = r.kept - r.exactKept;
        return [
          { title: "Filters", rows: [["tokens kept", `${r.kept} of ${V}`], ["by exact (sorted) top-p", String(r.exactKept)], ["extra from short bisection", String(Math.max(0, extra))], ["mass kept before renorm.", (100 * r.massKept).toFixed(1) + " %"], ["largest probability", r.final[0].toFixed(3)]],
            chip: [extra <= 0, extra <= 0 ? "same set as the sort-based reference" : `${extra} extra token(s): too few iterations`] },
          { title: "Speculative decoding", rows: [["E[tokens per target pass]", E.toFixed(3)], ["ceiling (k + 1)", String(v.k + 1)], ["speedup E / (k·c + 1)", spd.toFixed(2) + "×"]],
            chip: [spd > 1, spd > 1 ? "faster than plain decoding" : "slower: drafts cost too much for this α"],
            html: `<p class="note">Formulas from platform/specdec/core.py, assuming i.i.d. acceptance α. The draft cost c is an example value, measure your own.</p>` },
        ];
      },
      note: "A model: synthetic logits, the filter order of filter_probs_np, bisection on the renormalised vector as in top_p_threshold_np, and the i.i.d. acceptance formula of platform/specdec. It is not a benchmark of any sampler.",
    },
    practice: {
      intro: "Exercises 1 (T0 half) and 2 are pure NumPy; the torch parts need the torch extra and run on CPU. The GPU timings in exercises 1 (T2) and 3 are TODO(run-on: g6.xlarge); aws.md in the module has the cost, auto-stop and teardown. Implement in exercises/my_sampling.py; S2S_SOLUTIONS=1 runs the tests against the reference.",
      items: [
        { title: "Distribution preservation of batched verification", tier: "T0 → T2 · hard", goal: "Implement verify_batch; the first emitted token must be distributed as p0 (tolerance 0.03 over 5,000 draws). T2: batched PyTorch version with no loop over positions, and every position's marginal tested.",
          cmd: "uv run pytest course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling/exercises -q -k verify" },
        { title: "Top-p by threshold search vs a sort", tier: "T0 · medium", goal: "Implement top_p_threshold by bisection; it must keep exactly the sort reference's set on heavy-tailed Dirichlet distributions. With torch, check sample_torch against the NumPy filters.",
          cmd: "uv run pytest course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling/exercises -q -k top_p" },
        { title: "Fused min-p + top-k sampler, benchmarked", tier: "T2 · hard", goal: "One Triton or CUDA kernel: online softmax, min-p, top-k by threshold search, sample with a per-row Philox seed. Compare against sample_torch and a sort-based sampler.",
          cmd: "uv run --extra torch python course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling/bench/sampler_bench.py --device cuda" },
        { title: "The engine's own tests", tier: "T0", goal: "Run the reference sampling and verification tests (same-set top-p, seeded reproducibility, first-token distribution).",
          cmd: "uv run pytest platform/engine/v1/tests/test_sampling.py platform/engine/v1/tests/test_spec_verify.py -q" },
        { title: "Sampler bench on CPU", tier: "T0 (+ torch)", goal: "See the sort-based and bisection samplers side by side at V = 32k and 128k. On CPU the numbers mean little; it checks the harness before a GPU run.",
          cmd: "uv run --extra torch python course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling/bench/sampler_bench.py --device cpu" },
      ],
      labs: [
        { label: "The reference sampler (NumPy and batched PyTorch)", path: "platform/engine/v1/s2s_engine/sampling.py" },
        { label: "The reference verifier", path: "platform/engine/v1/s2s_engine/spec_verify.py" },
        { label: "Speculative decoding formulas (P2.6): expected_tokens_per_round, expected_speedup", path: "platform/specdec/core.py" },
        { label: "Per-request generators in the engine step", path: "platform/engine/v1/s2s_engine/engine.py" },
        { label: "AWS guide for the GPU timings (g6.xlarge, SSM only, make down)", path: "course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling/aws.md" },
        { label: "Quiz", path: "course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling/quiz.md" },
        { label: "Lane B kernels: L4 #60 top-p, #104 min-p, L6 #87 verification (names only; problem texts are CC BY-NC-ND, link only)", path: "https://leetgpu.com/challenges" },
        { label: "Papers: Leviathan et al., Fast Inference from Transformers via Speculative Decoding (ICML '23); Holtzman et al., The Curious Case of Neural Text Degeneration; Nguyen et al., Min-p sampling (arXiv 2407.01082)", path: "arXiv 2407.01082" },
      ],
    },
  });
})();
