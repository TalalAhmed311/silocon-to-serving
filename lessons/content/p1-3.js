/* P1.3 — Latency and throughput metrics. */
(function () {
  const F = S2S.fmt;
  // The six requests of examples/sample_tokens.jsonl (arrival and token timestamps in seconds).
  const SAMPLE = [
    { id: "r1", a: 0.0, t: [0.21, 0.24, 0.27, 0.3, 0.33] },
    { id: "r2", a: 0.05, t: [0.26, 0.29, 0.32, 0.351] },
    { id: "r3", a: 0.1, t: [0.89, 0.93, 0.97, 1.01, 1.05, 1.09] },
    { id: "r4", a: 0.4, t: [0.62] },
    { id: "r5", a: 0.5, t: [0.7, 0.76, 0.82] },
    { id: "r6", a: 0.6, t: [2.1, 2.13, 2.16, 2.19] },
  ];

  /* ---- the mock server's cost model (platform/mockllm/server.py), re-implemented for the simulator ----
     step_ms = base + per_seq × running + prefill_per_token × (prompt tokens admitted this step)
     FIFO admission while running < max_num_seqs and the KV budget holds prompt + max output. */
  const MOCK = { base: 8, perSeq: 0.25, prefill: 0.05, kvCap: 200000 };
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const memo = new Map();
  function simulate(rate, P, O, maxSeqs, D = 30) {
    const key = [rate, P, O, maxSeqs, D].join(",");
    if (memo.has(key)) return memo.get(key);
    const r = rng(12345 + Math.round(rate * 100)), arr = [];
    for (let t = -Math.log(1 - r()) / rate; t < D; t += -Math.log(1 - r()) / rate) arr.push({ a: t, gen: 0, first: null, end: null });
    let T = 0, i = 0, kv = 0, outTok = 0;
    const waiting = [], running = [], cap = 6 * D;
    while ((i < arr.length || waiting.length || running.length) && T < cap) {
      while (i < arr.length && arr[i].a <= T) waiting.push(arr[i++]);
      if (!waiting.length && !running.length) { T = arr[i].a; continue; }
      let adm = 0;
      while (waiting.length && running.length < maxSeqs && kv + P + O <= MOCK.kvCap) { running.push(waiting.shift()); kv += P + O; adm += P; }
      const step = MOCK.base + MOCK.perSeq * running.length + MOCK.prefill * adm;
      T += step / 1000;
      for (let k = running.length - 1; k >= 0; k--) {
        const s = running[k]; s.gen++; if (T <= D) outTok++;
        if (s.first === null) s.first = T;
        if (s.gen >= O) { s.end = T; kv -= P + O; running.splice(k, 1); }
      }
    }
    // requests never served before the cap: their TTFT is at least cap − arrival
    arr.forEach((s) => { if (s.first === null) s.first = cap; if (s.end === null) s.end = cap + O; });
    const res = { reqs: arr.map((s) => ({ ttft: s.first - s.a, tpot: O > 1 ? (s.end - s.first) / (O - 1) : null, end: s.end })), outTokPerS: outTok / D, D };
    memo.set(key, res);
    return res;
  }
  function pct(vals, q) { // numpy "linear" percentile
    if (!vals.length) return NaN;
    const v = vals.slice().sort((x, y) => x - y), rank = (q / 100) * (v.length - 1), lo = Math.floor(rank), hi = Math.ceil(rank);
    return v[lo] + (v[hi] - v[lo]) * (rank - lo);
  }
  const msOrS = (s) => (isFinite(s) ? F.ms(s) : "–");

  // a horizontal timeline helper: seconds → x
  const TL = (x0, x1, t0, t1) => (t) => x0 + ((x1 - x0) * (t - t0)) / (t1 - t0);

  S2S.lesson({
    id: "p1-3", n: "P1.3", title: "Latency and throughput metrics",
    subtitle: "Inference fundamentals · first principles · T0, no GPU needed",
    kicker: "Lesson · ≈ 35 min",
    headline: "\"Is it fast?\" is three questions",
    intro: `<p>A user asks "why is it slow?" An operator asks "how many requests can one GPU take?" Both questions have precise answers, but only once you name the clock you are reading. This lesson defines every number used to judge an LLM server: when the first token arrives, how evenly the rest stream, how much work the server finishes per second, and how many requests were actually good enough. Then it shows how those numbers behave as load rises, using the course's own mock server.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["q", "waiting in queue"], ["w", "prefill"], ["k", "token"], ["hot", "missed the target"], ["ok", "good"]],
    prev: "p1-2", next: "p1-4",
    steps: [
      { rail: "a request's clock", title: "Every request leaves a trail of timestamps",
        body: `<p>A chat client sends a prompt and receives the answer as a <b>stream</b>: one small message per generated token, so text appears while the model is still working. Each streamed request therefore produces a list of times:</p>
<div class="eq">arrival      when the request reached the server
token 1      when the first token was sent
token 2 … n  when each later token was sent</div>
<p>Between arrival and token 1, three things happen. The request may <b>wait in a queue</b> because the server is full. Then its prompt goes through <b>prefill</b> (P1.2), which produces the first token. Then <b>decode</b> adds one token per step.</p>
<p>Every metric in this lesson is arithmetic on these timestamps. The repo keeps six example requests in <code>examples/sample_tokens.jsonl</code>; we use them throughout. The first one:</p>
<div class="eq">{"id": "r1", "arrival": 0.000,
 "tokens": [0.210, 0.240, 0.270, 0.300, 0.330]}</div>`,
        scene(G) {
          const X = TL(40, 600, 0, 0.36);
          G.text(24, 50, "request r1 · times in seconds", { size: 14 });
          G.line(40, 200, 600, 200, { color: "line" });
          [0, 0.1, 0.2, 0.3].forEach((t) => { G.line(X(t), 196, X(t), 204, { color: "muted" }); G.label(X(t), 222, t.toFixed(1), { anchor: "middle", size: 11 }); });
          G.rect(X(0), 140, X(0.08) - X(0), 40, { fill: "q", rx: 4, opacity: 0.85 }); G.label(X(0) + 4, 132, "queue", { size: 11, color: "q" });
          G.rect(X(0.08), 140, X(0.21) - X(0.08), 40, { fill: "w", rx: 4 }); G.label(X(0.08) + 4, 132, "prefill", { size: 11 });
          const ticks = SAMPLE[0].t.map((t) => G.rect(X(t) - 5, 140, 10, 40, { fill: "k", rx: 3 }));
          G.circle(X(0), 200, 6, { fill: "ink" }); G.label(X(0), 250, "arrival", { size: 12, color: "ink" });
          SAMPLE[0].t.forEach((t, i) => G.label(X(t), 250, String(i + 1), { anchor: "middle", size: 12, color: "k" }));
          G.label(X(0.27), 272, "token number", { anchor: "middle", size: 11 });
          G.label(24, 330, "queue and prefill split is illustrative: the log only records", { size: 12 });
          G.label(24, 348, "arrival and token times. Metrics never need more than that.", { size: 12 });
          G.from(ticks, { opacity: 0, stagger: 0.25, duration: 0.2 });
          G.caption("one request: an arrival time, then one timestamp per token");
        } },

      { rail: "ttft", title: "TTFT: how long until anything happens",
        body: `<p>The first thing a user notices is the blank screen. <b>Time to first token (TTFT)</b> measures it:</p>
<div class="eq">TTFT = first token time − arrival
r1:   0.210 − 0.000 = 0.210 s
r3:   0.890 − 0.100 = 0.790 s
r6:   2.100 − 0.600 = 1.500 s</div>
<p>TTFT is the sum of everything before the first token: queueing, plus prefill, plus (when measured at the client) the network. Prefill grows with prompt length, because every prompt token goes through the model. Queueing grows with load. So a TTFT that jumps from 0.2 s to 1.5 s, like r6, is almost never the model being slow at maths. It is the request waiting its turn.</p>`,
        scene(G) {
          const X = TL(70, 610, 0, 2.25);
          G.text(24, 46, "TTFT of the six sample requests", { size: 14 });
          SAMPLE.forEach((r, i) => {
            const y = 76 + i * 50;
            G.label(24, y + 20, r.id, { size: 13, color: "ink" });
            const bar = G.rect(X(r.a), y + 4, X(r.t[0]) - X(r.a), 24, { fill: r.t[0] - r.a > 0.5 ? "hot" : "q", rx: 4, opacity: 0.85 });
            r.t.forEach((t) => G.rect(X(t) - 2, y + 4, 4, 24, { fill: "k", rx: 1 }));
            G.label(X(r.t[0]) + 8 + (r.t.length * 0) , y - 2, (r.t[0] - r.a).toFixed(2) + " s", { size: 11, color: r.t[0] - r.a > 0.5 ? "hot" : "muted" });
            G.from(bar, { attr: { width: 0 }, duration: 0.5, delay: i * 0.1 });
          });
          G.line(70, 384, 610, 384, { color: "line" });
          [0, 0.5, 1, 1.5, 2].forEach((t) => G.label(X(t), 402, t + " s", { anchor: "middle", size: 11 }));
          G.caption("bar = arrival → first token · ticks = tokens");
        } },

      { rail: "itl and tpot", title: "ITL and TPOT: how smoothly it streams",
        body: `<p>After the first token, the user watches text appear. The gaps between consecutive tokens are the <b>inter-token latencies (ITL)</b>. One request has many of them, so ITL is a list, not a number:</p>
<div class="eq">r3 tokens: 0.89 0.93 0.97 1.01 1.05 1.09
ITL:          0.04 0.04 0.04 0.04 0.04</div>
<p>To summarise one request, take the mean gap. That is the <b>time per output token (TPOT)</b>:</p>
<div class="eq">TPOT = (E2E − TTFT) / (n − 1)
r3:  (0.99 − 0.79) / (6 − 1) = 0.040 s</div>
<p>Why <code>n − 1</code>? Six tokens have five gaps; the first token's wait is already TTFT. A one-token answer has no gaps, so its TPOT is undefined (exercise 1 returns <code>None</code>). ITL is set by the decode step time, which grows with how many requests share each step (P1.2's batching).</p>`,
        check: { q: "A request arrives at t = 0. Its first token comes at 0.4 s and its 11th (last) token at 1.4 s. What is its TPOT?",
          options: ["0.091 s", "0.1 s", "0.127 s"], answer: 1,
          why: "Eleven tokens have ten gaps after the first one. (E2E − TTFT) / (n − 1) = (1.4 − 0.4) / 10 = 0.1 s. Dividing by 11, or including TTFT, mixes the queue and prefill wait into the streaming speed." },
        scene(G) {
          const r = SAMPLE[2], X = TL(60, 600, 0.85, 1.12);
          G.text(24, 50, "request r3 · tokens after the first", { size: 14 });
          G.line(60, 200, 600, 200, { color: "line" });
          const gaps = [];
          r.t.forEach((t, i) => {
            G.rect(X(t) - 6, 150, 12, 50, { fill: "k", rx: 3 });
            G.label(X(t), 222, t.toFixed(2), { anchor: "middle", size: 11 });
            if (i) {
              gaps.push(G.path(`M ${X(r.t[i - 1]) + 6} 140 Q ${(X(r.t[i - 1]) + X(t)) / 2} 100 ${X(t) - 6} 140`, { color: "v", w: 2 }));
              G.text((X(r.t[i - 1]) + X(t)) / 2, 96, "0.04", { anchor: "middle", size: 12, color: "v" });
            }
          });
          G.label(60, 270, "ITL = each gap  ·  TPOT = mean gap = 0.04 s", { size: 13, color: "ink" });
          G.label(60, 296, "5 gaps for 6 tokens → divide by n − 1", { size: 12 });
          G.from(gaps, { opacity: 0, stagger: 0.15, duration: 0.3 });
          G.caption("the gaps between tokens are the streaming speed");
        } },

      { rail: "e2e", title: "End-to-end latency ties them together",
        body: `<p><b>End-to-end latency (E2E)</b> is the total time a request took, from arrival to its last token:</p>
<div class="eq">E2E = last token time − arrival
    = TTFT + (n − 1) × TPOT
r3: 0.79 + 5 × 0.04 = 0.99 s  ✓</div>
<p>Which number matters depends on who is waiting. A person reading a chat answer cares about TTFT first and smooth ITL second; they start reading before the end. A program that needs the whole answer (an agent step, a batch job summarising documents) only sees E2E.</p>
<p>The formula also tells you where to look when E2E is bad. Long answers multiply TPOT by a large <code>n</code>: a 500-token answer at 40 ms per token spends 20 s streaming even with an instant first token.</p>`,
        scene(G) {
          const r = SAMPLE[2], X = TL(60, 600, 0, 1.1);
          G.text(24, 50, "request r3", { size: 14 });
          const ttft = G.rect(X(r.a), 130, X(r.t[0]) - X(r.a), 40, { fill: "q", rx: 4, opacity: 0.85 });
          const stream = G.rect(X(r.t[0]), 130, X(r.t[5]) - X(r.t[0]), 40, { fill: "v", rx: 4, opacity: 0.85 });
          r.t.forEach((t) => G.rect(X(t) - 2, 130, 4, 40, { fill: "k", rx: 1 }));
          G.label((X(r.a) + X(r.t[0])) / 2, 122, "TTFT 0.79 s", { anchor: "middle", size: 12, color: "q" });
          G.label(X(r.t[0]) - 4, 196, "5 × TPOT = 0.20 s", { size: 12, color: "v" });
          G.line(X(r.a), 230, X(r.t[5]), 230, { color: "ink", w: 2 });
          G.line(X(r.a), 222, X(r.a), 238, { color: "ink", w: 2 }); G.line(X(r.t[5]), 222, X(r.t[5]), 238, { color: "ink", w: 2 });
          G.text((X(r.a) + X(r.t[5])) / 2, 256, "E2E = 0.99 s", { anchor: "middle", size: 14 });
          G.label(24, 320, "a chat user feels TTFT, then ITL", { size: 13, color: "ink" });
          G.label(24, 344, "a program waiting for the full answer feels only E2E", { size: 13, color: "ink" });
          G.from(stream, { attr: { width: 0 }, duration: 0.6, delay: 0.3 });
          G.caption("E2E = TTFT + (n − 1) × TPOT");
        } },

      { rail: "throughput", title: "Throughput: what the operator pays for",
        body: `<p>Users judge one request at a time. Whoever pays for the GPU judges the whole server: how much work does it finish per second? That is <b>throughput</b>, counted over a time window, either as completed <b>requests per second</b> or as <b>output tokens per second</b>.</p>
<p>The two goals pull against each other. P1.2 showed that putting more requests into each decode step reads the weights once for all of them, so the server makes more tokens per second. But each step gets a bit slower, so every user's ITL rises.</p>
<p>The course's mock server (<code>platform/mockllm/server.py</code>) models exactly this with made-up but realistically shaped costs:</p>
<div class="eq">step_ms = 8 + 0.25 × running sequences
                (+ 0.05 per admitted prompt token)
batch  1:  8.25 ms → ITL  8.25 ms,   121 tok/s
batch 64: 24 ms    → ITL 24 ms,    2,667 tok/s</div>
<p>Going from 1 to 64 concurrent streams makes each user's tokens 2.9× slower and the server's output 22× larger. The constants are the mock's settings, not measurements; P2 replaces them with vLLM on a real GPU.</p>`,
        scene(G) {
          const bs = [1, 8, 16, 32, 48, 64];
          G.text(24, 44, "mock server: batch size vs ITL and tokens/s", { size: 13 });
          G.axes(60, 70, 240, 260, {});
          G.axes(370, 70, 240, 260, {});
          G.label(60, 60, "ITL per user (ms)", { size: 11, color: "v" });
          G.label(370, 60, "server tokens/s", { size: 11, color: "ok" });
          const itl = (b) => 8 + 0.25 * b, tps = (b) => (1000 * b) / itl(b);
          const xa = (b) => 60 + (230 * b) / 64, xb = (b) => 370 + (230 * b) / 64;
          const ya = (v) => 330 - (240 * v) / 30, yb = (v) => 330 - (240 * v) / 3000;
          G.path("M " + bs.map((b) => `${xa(b)} ${ya(itl(b))}`).join(" L "), { color: "v", w: 3 });
          G.path("M " + bs.map((b) => `${xb(b)} ${yb(tps(b))}`).join(" L "), { color: "ok", w: 3 });
          const dots = [];
          bs.forEach((b) => { dots.push(G.circle(xa(b), ya(itl(b)), 4, { fill: "v" }), G.circle(xb(b), yb(tps(b)), 4, { fill: "ok" })); G.label(xa(b), 350, String(b), { anchor: "middle", size: 10 }); G.label(xb(b), 350, String(b), { anchor: "middle", size: 10 }); });
          [0, 10, 20, 30].forEach((v) => G.label(54, ya(v) + 4, String(v), { anchor: "end", size: 10 }));
          [0, 1000, 2000, 3000].forEach((v) => G.label(364, yb(v) + 4, v ? v / 1000 + "k" : "0", { anchor: "end", size: 10 }));
          G.label(180, 370, "requests per step", { anchor: "middle", size: 11 }); G.label(490, 370, "requests per step", { anchor: "middle", size: 11 });
          G.text(xa(64) - 40, ya(24) - 12, "24 ms", { size: 12, color: "v" }); G.text(xb(64) - 40, yb(2667) - 12, "2,667", { size: 12, color: "ok" });
          G.from(dots, { opacity: 0, stagger: 0.06, duration: 0.25 });
          G.caption("bigger batches: slower for each user, more output overall");
        } },

      { rail: "percentiles", title: "Percentiles: the mean hides the tail",
        body: `<p>Six requests, six TTFTs. Sorted:</p>
<div class="eq">0.20 0.21 0.21 0.22 0.79 1.50   (seconds)
mean = 3.13 / 6        = 0.52 s</div>
<p>The mean says "about half a second". Yet four of six users waited about 0.2 s, and one waited 1.5 s. Nobody waited 0.52 s. Latency distributions are <b>skewed</b>: most requests are quick, and a few that queued behind others are very slow. A mean blends them into a number that describes no one.</p>
<p>A <b>percentile</b> answers "how long did the slowest X% wait?" The p50 (median) is the middle; the p90 is beaten by 90% of requests; the p99 is the experience of one user in a hundred. With the "linear" rule <code>numpy.percentile</code> uses (rank = q × (n − 1), interpolate between neighbours):</p>
<div class="eq">p50: rank 2.5 → (0.21 + 0.22) / 2      = 0.215 s
p90: rank 4.5 → 0.79 + 0.5 × (1.50 − 0.79) = 1.145 s</div>
<p>Report p50, p90 and p99 together with the request count. A benchmark that gives only a mean is hiding something.</p>`,
        check: { q: "Four of the six requests waited about 0.2 s for their first token. What does the mean TTFT of 0.52 s tell you?",
          options: ["The typical user waits about half a second", "Very little: a couple of slow requests pull it far from what most users saw", "That the p99 is about 0.52 s"], answer: 1,
          why: "The mean adds the 1.5 s and 0.79 s outliers into everyone's number. The median (0.215 s) describes the typical user and the p90 (1.145 s) the unlucky ones; the mean describes neither." },
        scene(G) {
          const v = [0.2, 0.21, 0.21, 0.22, 0.79, 1.5], X = TL(60, 600, 0, 1.6);
          G.text(24, 46, "TTFT of the six sample requests", { size: 14 });
          G.line(60, 200, 600, 200, { color: "line" });
          [0, 0.4, 0.8, 1.2, 1.6].forEach((t) => G.label(X(t), 222, t.toFixed(1) + " s", { anchor: "middle", size: 11 }));
          const dots = v.map((t, i) => G.circle(X(t), 180 - (i > 0 && i < 4 ? i * 14 : 0), 8, { fill: t > 0.5 ? "hot" : "k" }));
          const mark = (t, txt, col, y) => { G.line(X(t), 90, X(t), 200, { color: col, dash: "4 3", w: 2 }); G.text(X(t) + 6, y, txt, { size: 12, color: col }); };
          mark(0.215, "p50 0.215", "ok", 84);
          mark(0.5217, "mean 0.52", "v", 104);
          mark(1.145, "p90 1.145", "hot", 84);
          G.label(60, 280, "nobody waited 0.52 s: the mean sits in the gap", { size: 13, color: "ink" });
          G.label(60, 304, "median = typical user · p90 / p99 = the unlucky ones", { size: 12 });
          G.from(dots, { opacity: 0, y: -30, stagger: 0.1, duration: 0.35 });
          G.caption("a skewed distribution: report p50, p90 and p99");
        } },

      { rail: "histograms", title: "Servers keep histograms, not samples",
        body: `<p>A busy server handles millions of requests. Keeping every latency to sort later is too expensive, so monitoring systems like <b>Prometheus</b> keep a <b>histogram</b>: a fixed list of upper bounds called <code>le</code> ("less than or equal"), and for each one a running count of requests at or below it. The mock server uses the bounds 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, … seconds. Our six TTFTs give these cumulative counts:</p>
<div class="eq">le:     0.2   0.5   1   2   +Inf
count:   1     4    5   6     6</div>
<p>To estimate a percentile, <code>histogram_quantile</code> computes the rank <code>q × total</code>, finds the first bucket whose count reaches it, and <b>assumes the samples are spread evenly inside that bucket</b>:</p>
<div class="eq">p90: rank 0.9 × 6 = 5.4  → bucket (1, 2]
     1 + (2 − 1) × (5.4 − 5) / (6 − 5) = 1.4 s
p50: rank 3 → bucket (0.2, 0.5]
     0.2 + 0.3 × (3 − 1) / (4 − 1)    = 0.4 s</div>
<p>The true values were 1.145 s and 0.215 s. A histogram percentile is only as precise as its bucket widths: here it can land anywhere inside the bucket. Exercise 2 implements this exact rule and measures its error on 10,000 samples.</p>`,
        check: { q: "Buckets are …, 0.5, 1, 2.5, … and the true p90 is 0.6 s. What can histogram_quantile report?",
          options: ["Exactly 0.6 s", "Some value between 0.5 and 1 s", "Always 1 s, the bucket's upper bound"], answer: 1,
          why: "Only the bucket counts are stored. The function knows the p90 rank falls in (0.5, 1] and interpolates linearly as if samples were spread evenly there, so the answer can be anywhere in that bucket depending on how the counts fall." },
        scene(G) {
          const les = ["≤0.2", "≤0.5", "≤1", "≤2", "+Inf"], cum = [1, 4, 5, 6, 6];
          G.text(24, 44, "cumulative bucket counts (le)", { size: 13 });
          const b = G.bars(60, 250, cum, { w: 70, gap: 30, h: 170, max: 6, fill: (i) => (i === 3 ? "hot" : "k") });
          cum.forEach((c, i) => { G.text(95 + i * 100, 240 - (170 * c) / 6, String(c), { anchor: "middle", size: 13 }); G.label(95 + i * 100, 270, les[i], { anchor: "middle", size: 12, color: "ink" }); });
          G.line(50, 250 - (170 * 5.4) / 6, 560, 250 - (170 * 5.4) / 6, { color: "hot", dash: "5 4" });
          G.label(60, 250 - (170 * 5.4) / 6 - 8, "rank 5.4 (p90)", { size: 12, color: "hot" });
          // interpolation inside (1, 2]
          const X = TL(120, 520, 1, 2);
          G.line(120, 340, 520, 340, { color: "line", w: 2 });
          G.label(120, 362, "1 s · count 5", { anchor: "middle", size: 11 }); G.label(520, 362, "2 s · count 6", { anchor: "middle", size: 11 });
          const p = G.circle(X(1.4), 340, 8, { fill: "hot" });
          G.label(X(1.4), 326, "estimate 1.4 s", { anchor: "middle", size: 12, color: "hot" });
          G.circle(X(1.145), 340, 6, { stroke: "ok", sw: 2 });
          G.label(X(1.145), 386, "true 1.145 s", { anchor: "middle", size: 12, color: "ok" });
          G.from(b, { attr: { height: 0, y: 250 }, stagger: 0.1, duration: 0.35 });
          G.from(p, { attr: { cx: 120 }, duration: 0.8, delay: 0.6 });
          G.caption("interpolate inside the bucket that holds the rank");
        } },

      { rail: "load and the knee", title: "Load: where TTFT explodes",
        body: `<p>The same server gives very different numbers at 5 and at 40 requests per second. The cause is the <b>queue</b>. Each request needs a slot in the running batch (the mock allows at most 64, <code>max_num_seqs</code>). When requests arrive faster than slots free up, the extras wait.</p>
<p>Estimate the mock's capacity with its default costs, 64-token prompts and 64-token answers. At a full batch of 64, a step takes 24 ms and each request needs 64 steps; its prompt adds 64 × 0.05 = 3.2 ms of prefill to one step. One new request per completed one means each request costs the server <code>24 + 3.2 = 27.2 ms</code> of step time, so:</p>
<div class="eq">capacity ≈ 1000 ms / 27.2 ms ≈ 37 requests/s</div>
<p>Below that rate, TTFT stays near one step plus prefill. Near it, short random bursts start to queue. Above it, the queue grows every second, and so does every new request's TTFT, without limit. ITL, by contrast, is capped: the batch can't grow past 64, so steps can't get slower than about 24 ms. <b>TTFT is the metric that breaks first.</b> The point where latency turns sharply upward is called the <b>knee</b>. The chart is computed with the mock's cost model (the simulator tab runs the same code).</p>`,
        check: { q: "Arrivals rise past the server's capacity. Which metric keeps growing without limit?",
          options: ["ITL, because steps get slower", "TTFT, because the waiting queue grows every second", "Output tokens per second"], answer: 1,
          why: "Batch size is capped by max_num_seqs, so step time (and ITL) levels off. The excess requests pile up in the queue, and each new arrival waits behind all of them, so TTFT climbs without bound while throughput plateaus." },
        scene(G) {
          const rates = [5, 10, 15, 20, 25, 30, 33, 36, 39, 42, 46, 50];
          const p90 = rates.map((r) => pct(simulate(r, 64, 64, 64, 20).reqs.map((x) => x.ttft), 90));
          const thr = rates.map((r) => simulate(r, 64, 64, 64, 20).outTokPerS);
          G.text(24, 28, "mock server · 64-token prompts and answers", { size: 13 });
          const X = (r) => 70 + (r / 52) * 520, Yt = (s) => 200 - (150 * Math.log10(Math.max(s, 0.01) / 0.01)) / 3, Yh = (v) => 390 - (130 * v) / 3000;
          G.axes(70, 50, 530, 150, {}); G.axes(70, 260, 530, 130, {});
          G.label(76, 64, "p90 TTFT (log scale)", { size: 11, color: "hot" });
          [0.01, 0.1, 1, 10].forEach((s) => G.label(64, Yt(s) + 4, s >= 1 ? s + " s" : s * 1000 + " ms", { anchor: "end", size: 10 }));
          G.path("M " + rates.map((r, i) => `${X(r)} ${Yt(p90[i])}`).join(" L "), { color: "hot", w: 3 });
          G.label(76, 274, "output tokens/s", { size: 11, color: "ok" });
          [0, 1000, 2000, 3000].forEach((v) => G.label(64, Yh(v) + 4, v ? v / 1000 + "k" : "0", { anchor: "end", size: 10 }));
          const ln = G.path("M " + rates.map((r, i) => `${X(r)} ${Yh(thr[i])}`).join(" L "), { color: "ok", w: 3 });
          [10, 20, 30, 40, 50].forEach((r) => G.label(X(r), 408, String(r), { anchor: "middle", size: 10 }));
          G.label(600, 424, "arrival rate (requests/s)", { anchor: "end", size: 11 });
          G.line(X(36.8), 50, X(36.8), 390, { color: "muted", dash: "4 4" });
          G.label(X(36.8) + 6, 236, "≈ 37 req/s capacity", { size: 11 });
          G.from(ln, { opacity: 0, duration: 0.6 });
          G.caption("past capacity, TTFT climbs while throughput flattens");
        } },

      { rail: "goodput", title: "Goodput: count only the requests that were good enough",
        body: `<p>Throughput rewards a server for finishing requests even when every user gave up waiting. The fix is to agree on a <b>service-level objective (SLO)</b>, a target each request must meet, and count only the requests that met it. That count per second is <b>goodput</b>.</p>
<p>A typical SLO has two parts, one per thing users feel. Take TTFT ≤ 0.5 s and TPOT ≤ 0.05 s for the six samples:</p>
<div class="eq">r1  TTFT 0.21  TPOT 0.030   good
r2  TTFT 0.21  TPOT 0.030   good
r3  TTFT 0.79  TPOT 0.040   bad (TTFT)
r4  TTFT 0.22  one token    good
r5  TTFT 0.20  TPOT 0.060   bad (TPOT)
r6  TTFT 1.50  TPOT 0.030   bad (TTFT)</div>
<p>Three of six met the SLO: <b>50% attainment</b>. Over a 3-second window that is 2 requests/s of throughput but only 1 request/s of goodput. Note r5: a fast start, but streaming too slowly; a one-part SLO would have missed it.</p>
<p>Past the knee, throughput keeps climbing or plateaus while goodput falls, because more and more requests miss the target. That is why serving research (DistServe, for example) optimises goodput rather than raw throughput.</p>`,
        check: { q: "Server A and server B both complete 10 requests/s. A has 9.5 good requests/s and B has 6, but B has the lower mean latency. Which serves users better?",
          options: ["B, its mean latency is lower", "A: 95% of its requests meet the SLO against 60% for B", "They are equal, both do 10 requests/s"], answer: 1,
          why: "Goodput counts requests that met the target. B's low mean can come from many very fast requests plus a tail that misses the SLO; A's users are almost all inside the target, which is what an SLO promises." },
        scene(G) {
          const rows = [["r1", 0.21, 0.03], ["r2", 0.21, 0.0303], ["r3", 0.79, 0.04], ["r4", 0.22, null], ["r5", 0.2, 0.06], ["r6", 1.5, 0.03]];
          G.text(24, 42, "SLO: TTFT ≤ 0.5 s and TPOT ≤ 0.05 s", { size: 14 });
          G.label(110, 72, "TTFT", { size: 12 }); G.label(330, 72, "TPOT", { size: 12 });
          const marks = [];
          rows.forEach(([id, a, b], i) => {
            const y = 90 + i * 44, okA = a <= 0.5, okB = b === null || b <= 0.05, good = okA && okB;
            G.label(24, y + 20, id, { size: 13, color: "ink" });
            G.rect(110, y + 6, Math.min(200, a * 130), 20, { fill: okA ? "k" : "hot", rx: 3 });
            G.line(110 + 0.5 * 130, y, 110 + 0.5 * 130, y + 32, { color: "muted", dash: "3 3" });
            if (b !== null) G.rect(330, y + 6, b * 2400, 20, { fill: okB ? "k" : "hot", rx: 3 }); else G.label(330, y + 21, "1 token", { size: 11 });
            G.line(330 + 0.05 * 2400, y, 330 + 0.05 * 2400, y + 32, { color: "muted", dash: "3 3" });
            marks.push(G.box(520, y + 2, 92, 28, good ? "good" : "missed", { fill: good ? "ok" : "hot", size: 12 }));
          });
          G.text(24, 372, "throughput 6 / 3 s = 2 req/s", { size: 14 });
          G.text(24, 398, "goodput    3 / 3 s = 1 req/s", { size: 14, color: "ok" });
          G.from(marks, { opacity: 0, x: 20, stagger: 0.12, duration: 0.3 });
          G.caption("dashed lines are the SLO limits");
        } },

      { rail: "who measures", title: "Where you measure changes the number",
        body: `<p>The same request has two TTFTs. The <b>server</b> starts its clock when the request reaches the engine and stops it when the first token leaves. The <b>client</b> starts before the HTTP request is even written and stops when the first bytes arrive. The client's number also contains the network, TLS, any proxies or gateways, and the server's HTTP handling, so <b>client TTFT ≥ server TTFT</b>. Exercise 4 builds a streaming client that proves it against the mock server.</p>
<p>Streams arrive as <b>Server-Sent Events</b>: lines of <code>data: {json}</code>, ending with <code>data: [DONE]</code>. A client timestamps each line that carries text.</p>
<p>In production, vLLM exports these metrics in Prometheus format (names as read from the docs at the repo's pinned vLLM commit; the mock server uses the same names so dashboards carry over):</p>
<table><tr><th>concept</th><th>vLLM metric</th></tr>
<tr><td>TTFT</td><td><code>vllm:time_to_first_token_seconds</code></td></tr>
<tr><td>ITL</td><td><code>vllm:inter_token_latency_seconds</code></td></tr>
<tr><td>TPOT</td><td><code>vllm:request_time_per_output_token_seconds</code></td></tr>
<tr><td>queue time</td><td><code>vllm:request_queue_time_seconds</code></td></tr>
<tr><td>load</td><td><code>vllm:num_requests_running</code>, <code>…_waiting</code></td></tr></table>`,
        scene(G) {
          const lanes = [["client", 70], ["network + proxy", 170], ["server", 270]];
          lanes.forEach(([n, y]) => { G.label(24, y - 10, n, { size: 12, color: "ink" }); G.line(24, y + 20, 616, y + 20, { color: "line" }); });
          const X = TL(150, 600, 0, 1);
          G.rect(X(0), 60, X(0.9) - X(0), 22, { stroke: "v", sw: 2, rx: 4 });
          G.text(X(0) + 6, 76, "client TTFT", { size: 12, color: "v" });
          G.rect(X(0.12), 260, X(0.78) - X(0.12), 22, { stroke: "k", sw: 2, rx: 4 });
          G.text(X(0.12) + 6, 276, "server TTFT", { size: 12, color: "k" });
          const req = G.arrow(X(0.02), 92, X(0.11), 252, { color: "muted", w: 2 });
          const resp = G.arrow(X(0.79), 252, X(0.89), 92, { color: "muted", w: 2 });
          G.label(X(0.13), 180, "request", { size: 11 }); G.label(X(0.79) - 8, 200, "first token", { size: 11, anchor: "end" });
          G.text(24, 350, "client TTFT − server TTFT = network, TLS, proxies,", { size: 13 });
          G.text(24, 372, "gateways and HTTP handling: worth its own dashboard", { size: 13 });
          G.from([req, resp], { opacity: 0, stagger: 0.5, duration: 0.4 });
          G.caption("the client's clock always includes more than the server's");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>TTFT = first token − arrival (queue + prefill + network). ITL is each gap between tokens; TPOT is one request's mean gap, (E2E − TTFT) / (n − 1).</li>
<li>E2E = TTFT + (n − 1) × TPOT. Chat users feel TTFT and ITL; programs feel E2E.</li>
<li>Throughput (requests/s or tokens/s) rises with batch size while each user's ITL rises too.</li>
<li>Latency is skewed: report p50, p90, p99 and a count. Prometheus histograms interpolate inside buckets, so their percentiles are only as precise as the bucket widths.</li>
<li>Past capacity the queue grows without bound and TTFT breaks first. Goodput, the requests per second that met the SLO, is the honest target.</li></ul>`,
    sim: {
      title: "Load the mock server",
      intro: "This runs the cost model of <code>platform/mockllm/server.py</code> (step = 8 ms + 0.25 ms per running sequence + 0.05 ms per admitted prompt token; FIFO queue; at most max_num_seqs running) on 30 simulated seconds of random (Poisson) arrivals. The chart sweeps the arrival rate: p90 TTFT on a log scale on top, throughput and goodput below. The marker is your chosen rate; the panels give its full numbers.",
      height: 300,
      controls: [
        { id: "rate", label: "arrival rate (requests/s)", min: 1, max: 80, value: 30 },
        { id: "seqs", label: "max_num_seqs (batch cap)", min: 4, max: 256, step: 4, value: 64 },
        { id: "p", label: "prompt tokens", min: 16, max: 2048, step: 16, value: 64 },
        { id: "o", label: "output tokens", min: 4, max: 512, step: 4, value: 64 },
        { id: "sttft", label: "SLO: TTFT ≤ (ms)", min: 100, max: 5000, step: 100, value: 1000 },
        { id: "stpot", label: "SLO: TPOT ≤ (ms)", min: 10, max: 200, step: 5, value: 50 },
      ],
      draw(G, v) {
        const rates = [], maxR = 80;
        for (let r = 2; r <= maxR; r += 3) rates.push(r);
        const good = (x) => x.ttft <= v.sttft / 1000 && (x.tpot === null || x.tpot <= v.stpot / 1000);
        const stat = (r) => { const s = simulate(r, v.p, v.o, v.seqs); const done = s.reqs.filter((x) => x.end <= s.D); return { s, p90: pct(s.reqs.map((x) => x.ttft), 90), thr: done.length / s.D, gp: done.filter(good).length / s.D }; };
        const pts = rates.map(stat), cur = stat(v.rate);
        const X = (r) => 50 + ((r - 0) / maxR) * 570;
        const Yt = (s) => 120 - (100 * Math.log10(Math.min(Math.max(s, 0.005), 200) / 0.005)) / Math.log10(200 / 0.005);
        const top = Math.max(10, ...pts.map((p) => p.thr), cur.thr) * 1.4, Yr = (q) => 270 - (110 * q) / top;
        G.axes(50, 20, 570, 100, {}); G.axes(50, 160, 570, 110, {});
        G.label(56, 32, "p90 TTFT", { size: 11, color: "hot" });
        [0.01, 0.1, 1, 10, 100].forEach((s) => G.label(44, Yt(s) + 4, s >= 1 ? s + "s" : s * 1000 + "ms", { anchor: "end", size: 9 }));
        G.line(50, Yt(v.sttft / 1000), 620, Yt(v.sttft / 1000), { color: "muted", dash: "4 4" });
        G.label(616, Yt(v.sttft / 1000) - 4, "TTFT SLO", { anchor: "end", size: 10 });
        G.path("M " + pts.map((p, i) => `${X(rates[i])} ${Yt(p.p90)}`).join(" L "), { color: "hot", w: 2.5 });
        G.label(56, 172, "completed requests/s", { size: 11, color: "k" }); G.label(220, 172, "goodput (met both SLOs)", { size: 11, color: "ok" });
        [0, top / 2, top].forEach((q) => G.label(44, Yr(q) + 4, q.toFixed(0), { anchor: "end", size: 9 }));
        G.path("M " + pts.map((p, i) => `${X(rates[i])} ${Yr(p.thr)}`).join(" L "), { color: "k", w: 2.5 });
        G.path("M " + pts.map((p, i) => `${X(rates[i])} ${Yr(p.gp)}`).join(" L "), { color: "ok", w: 2.5, dash: "6 3" });
        [0, 20, 40, 60, 80].forEach((r) => G.label(X(r), 288, String(r), { anchor: "middle", size: 10 }));
        G.label(620, 298, "arrival rate (req/s)", { anchor: "end", size: 10 });
        G.line(X(v.rate), 20, X(v.rate), 270, { color: "v", w: 2 });
        const R = cur.s.reqs, tt = R.map((x) => x.ttft), tp = R.filter((x) => x.tpot !== null).map((x) => x.tpot);
        const mean = tt.reduce((a, b) => a + b, 0) / Math.max(1, tt.length);
        const att = R.length ? R.filter(good).length / R.length : 0;
        const cap = Math.min(v.seqs, Math.floor(MOCK.kvCap / (v.p + v.o)));
        const fullStep = MOCK.base + MOCK.perSeq * cap, capRate = 1000 / ((fullStep * v.o) / cap + MOCK.prefill * v.p);
        return [
          { title: `TTFT at ${v.rate} req/s (${R.length} requests)`, rows: [["p50", msOrS(pct(tt, 50))], ["p90", msOrS(pct(tt, 90))], ["p99", msOrS(pct(tt, 99))], ["mean", msOrS(mean)]] },
          { title: "Streaming", rows: [["TPOT p50", msOrS(pct(tp, 50))], ["TPOT p90", msOrS(pct(tp, 90))], ["output tokens/s", F.num(cur.s.outTokPerS)]] },
          { title: "Throughput vs goodput", rows: [["completed req/s", F.num(cur.thr, 1)], ["goodput req/s", F.num(cur.gp, 1)], ["SLO attainment", (100 * att).toFixed(0) + "%"]],
            gauge: [[att, "ok"], [1 - att, "hot"]], chip: [v.rate < capRate, v.rate < capRate ? "below estimated capacity" : "above estimated capacity: queue grows"] },
          { title: "Capacity estimate", rows: [["max running (batch or KV cap)", F.num(cap)], ["step at full batch", fullStep.toFixed(2) + " ms"], ["≈ capacity", F.num(capRate, 1) + " req/s"]],
            html: `<p class="note">Steady-state estimate: a full batch, with each request costing o × step / batch plus its prefill. The mock's costs are made up (they are its defaults, not measurements); the shape, not the numbers, carries over to vLLM.</p>` },
        ];
      },
    },
    practice: {
      intro: "Run these from the repository root. Each exercise has a starter file and a test; <code>S2S_SOLUTIONS=1</code> runs the same test against the reference solution. Exercise 4 starts the mock server in-process, so nothing leaves your machine.",
      items: [
        { title: "TTFT / ITL / TPOT / E2E from timestamps", tier: "T0 · easy", goal: "Turn an arrival time and token timestamps into the per-request metrics, with percentiles that match numpy's linear method; handle the one-token answer.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/exercises/01-request-metrics" },
        { title: "histogram_quantile like Prometheus", tier: "T0 · medium", goal: "Implement bucket interpolation, check it is within one bucket width of the exact percentile on 10,000 samples, and handle empty and +Inf cases.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/exercises/02-histogram-quantile" },
        { title: "Goodput under a two-part SLO", tier: "T0 · easy", goal: "Count requests that meet both TTFT and TPOT targets, as a rate and as an attainment fraction.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/exercises/03-goodput" },
        { title: "SSE client: client-side vs server-side TTFT", tier: "T0 · hard", goal: "Stream a completion, timestamp each text chunk, and show the client's TTFT is at least the server's, scraped from the mock's /metrics.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/exercises/04-sse-client" },
      ],
      labs: [
        { label: "Start the mock engine first: <code>uv run uvicorn mockllm.server:app --app-dir platform --port 8001 &amp;</code>", path: "platform/mockllm/server.py" },
        { label: "Example 01: metrics and percentiles from the sample log", path: "course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/examples/01_metrics_from_log.py" },
        { label: "Example 02: a streaming client against the mock (<code>--url http://localhost:8001 --n 20</code>)", path: "course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/examples/02_stream_client.py" },
        { label: "Bench: open-loop load at several rates (<code>--rates 5 20 50 100</code>, or <code>--concurrency 64</code>)", path: "course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/bench/mock_load.py" },
        { label: "The six sample requests", path: "course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/examples/sample_tokens.jsonl" },
      ],
    },
  });
})();
