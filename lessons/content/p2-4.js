/* P2.4 — Benchmarking methodology. Why most LLM benchmark numbers can't be trusted, and the method that makes yours
   trustworthy: open-loop load, coordinated omission, warm-up, percentiles with n, repeats, tool reconciliation, the report. */
(function () {
  const F = S2S.fmt;

  /* ---------- simulator: closed loop vs open loop against a server with periodic stalls ---------- */
  function rng(seed) { let a = seed * 2654435761 + 7; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  // service of S ms starting at `start`; progress freezes during stalls [k*P, k*P + D) (first stall at P/2)
  function finish(start, S, P, D) {
    let t = start, rem = S;
    for (let guard = 0; guard < 10000; guard++) {
      const k = Math.floor((t - P / 2) / P), s0 = P / 2 + k * P, s1 = s0 + D;
      if (t >= s0 && t < s1) { t = s1; continue; }
      const next = t < P / 2 ? P / 2 : s0 + P;
      if (t + rem <= next) return t + rem;
      rem -= next - t; t = next + D;
    }
    return t + rem;
  }
  // FIFO server with C parallel slots
  function serve(arrivals, C, S, P, D) {
    const free = new Array(C).fill(0);
    return arrivals.map((a) => {
      let j = 0; for (let i = 1; i < C; i++) if (free[i] < free[j]) j = i;
      const st = Math.max(a, free[j]), fin = finish(st, S, P, D); free[j] = fin;
      return [a, fin - a];
    });
  }
  function closedLoop(users, C, S, P, D, T) {
    const free = new Array(C).fill(0), next = new Array(users).fill(0).map((_, i) => i * 0.01), out = [];
    for (let guard = 0; guard < 200000; guard++) {
      let u = 0; for (let i = 1; i < users; i++) if (next[i] < next[u]) u = i;
      const a = next[u]; if (a >= T) break;
      let j = 0; for (let i = 1; i < C; i++) if (free[i] < free[j]) j = i;
      const st = Math.max(a, free[j]), fin = finish(st, S, P, D); free[j] = fin;
      out.push([a, fin - a]); next[u] = fin;
    }
    return out;
  }
  function openLoop(rate, C, S, P, D, T, seed) {
    const R = rng(seed), arr = []; let t = 0;
    for (;;) { t += (-Math.log(1 - R()) * 1000) / rate; if (t >= T) break; arr.push(t); }
    return serve(arr, C, S, P, D);
  }
  const pct = (sorted, q) => { if (!sorted.length) return NaN; const x = q * (sorted.length - 1), i = Math.floor(x), f = x - i; return sorted[i] + (sorted[Math.min(i + 1, sorted.length - 1)] - sorted[i]) * f; };

  S2S.lesson({
    id: "p2-4", n: "P2.4", title: "Benchmarking methodology",
    subtitle: "Serving engines · first principles · T0 mock, T2 tools",
    kicker: "Lesson · ≈ 35 min",
    headline: "The load generator that lied",
    intro: `<p>In P2.3 you found a knee. Before you trust that number, or anyone else's, you need to know how it was measured. This lesson builds the method from first principles: how a load generator decides when to send, why the most common way hides the worst latencies, how many samples a percentile needs, how to handle warm-up and run-to-run noise, why three tools give three answers, and what a report must contain for a stranger to reproduce it.</p>`,
    facts: ["9 steps", "4 checkpoints", "1 simulator", "3 exercises"],
    legend: [["k", "requests"], ["v", "queued / waiting"], ["hot", "missed or hidden"], ["ok", "measured correctly"], ["w", "server"]],
    prev: "p2-3", next: "p2-5",
    steps: [
      { rail: "a number needs a method", title: "\"2,000 tokens/s\" means nothing on its own",
        body: `<p>A benchmark number is the answer to a question. "2,000 tokens per second" does not say which question: on which GPU, with which engine version and flags, which model and precision, how long the prompts and outputs were, how many requests were in flight, how requests were sent, and whether the first slow minutes were counted.</p>
<p>Change any one of those and the number moves, often by more than the difference you were trying to measure. Most numbers in blog posts and vendor slides leave half of them out, so they cannot be checked or compared.</p>
<p>The rest of this lesson builds a method in which every one of those choices is <b>deliberate</b> and <b>written down</b>. Your capstone teardown (C2) is only as credible as this method.</p>`,
        scene(G) {
          G.box(170, 30, 300, 56, "2,000 tokens/s", { stroke: "ink", size: 22 });
          G.label(320, 104, "…measured how?", { anchor: "middle", size: 13, color: "ink" });
          const qs = ["GPU, driver, CUDA", "engine + version + flags", "model, revision, dtype", "prompt / output lengths", "load model: open or closed", "rate or concurrency", "warm-up discarded?", "n samples, repeats"];
          const boxes = qs.map((q, i) => G.box(30 + (i % 2) * 300, 130 + Math.floor(i / 2) * 58, 280, 42, q + "   ?", { stroke: "hot", dash: "5 4", color: "hot", size: 13 }));
          G.text(24, 386, "each unknown can move the number more than the effect you test", { size: 13 });
          G.from(boxes, { opacity: 0, y: 10, stagger: 0.08, duration: 0.3 });
          G.caption("a result without its method can't be reproduced or compared");
        } },

      { rail: "closed loop", title: "Closed loop: users who wait politely",
        body: `<p>The simplest load generator starts N virtual users. Each sends a request, <b>waits for the answer</b>, then sends the next. This is a <b>closed loop</b>: the load generator and the server form one cycle.</p>
<p>Little's law (from queueing theory) ties three quantities together for any stable system: <b>requests in flight = arrival rate × time each spends inside</b>. In a closed loop the number in flight is fixed at N, so the rate is set by the server:</p>
<div class="eq">rate = N / latency

32 users, latency 2 s  →  16 requests/s
server slows to 4 s    →   8 requests/s</div>
<p>When the server gets slower, the users automatically send less. The load backs off exactly when the server struggles. That makes a closed loop useful for one question ("what does the server do with exactly 32 requests in flight?", which is what <code>--max-concurrency</code> asks) and useless for another: "what happens when more people show up than the server can handle?"</p>`,
        check: { q: "A closed-loop test runs 32 users. The server's latency rises from 2 s to 4 s. What happens to the load the test offers?",
          options: ["It stays at 16 requests/s", "It halves to 8 requests/s, because each user waits twice as long before sending again", "It doubles, because requests pile up"], answer: 1,
          why: "With N fixed, Little's law gives rate = N / latency = 32 / 4 = 8 requests/s. The test backs off exactly when the server slows, so it never pushes the server into overload." },
        scene(G) {
          G.box(380, 150, 200, 120, "", { fill: "w", boxOpacity: 0.5 });
          G.text(480, 205, "server", { anchor: "middle", size: 15 });
          G.label(480, 228, "latency L", { anchor: "middle", size: 12, color: "ink" });
          const us = [];
          for (let i = 0; i < 4; i++) {
            const y = 80 + i * 76;
            us.push(G.box(40, y, 110, 40, "user " + (i + 1), { stroke: "k", color: "k", size: 12 }));
            G.path(`M 152 ${y + 12} C 260 ${y + 12}, 300 190, 378 ${200 + (i - 1.5) * 12}`, { color: "k", arrow: true, w: 1.3 });
            G.path(`M 378 ${214 + (i - 1.5) * 12} C 300 230, 260 ${y + 30}, 154 ${y + 30}`, { color: "muted", arrow: true, w: 1.3, dash: "4 3" });
          }
          G.label(190, 60, "send → wait for the answer → send again", { size: 12, color: "ink" });
          G.text(24, 400, "in flight is always N  →  rate = N / L", { size: 14, color: "ok" });
          G.from(us, { opacity: 0, x: -16, stagger: 0.1, duration: 0.3 });
          G.caption("the server's speed sets the load: slow server, fewer requests");
        } },

      { rail: "open loop", title: "Open loop: arrivals that don't wait",
        body: `<p>Real users are independent. Somebody opening a chat window does not check whether the server finished your request first. An <b>open-loop</b> generator models this: requests arrive on a <b>timetable</b> fixed in advance, whatever the server is doing.</p>
<p>The usual timetable is a <b>Poisson process</b> at rate λ: the gaps between arrivals are random, independent, and exponentially distributed with mean 1/λ. That is how arrivals look when many people act independently, and it includes realistic <b>bursts</b>: several requests landing close together by chance.</p>
<div class="eq">t = 0
for each request:
    t += random.expovariate(rate)   # mean gap 1/rate
    send at time t, never earlier, never later</div>
<p>If λ is below what the server can complete, the queue empties between bursts. If λ is above it, the queue grows without limit and latency climbs for as long as the test runs. That growth is the overload you need to see. <code>platform/loadgen/workload.py</code> builds a seeded Poisson timetable, so two engines can be given exactly the same requests at the same moments.</p>`,
        scene(G) {
          const R = rng(4), arr = []; let t = 0;
          while (t < 560) { t += -Math.log(1 - R()) * 26; if (t < 560) arr.push(t); }
          G.text(24, 40, "Poisson arrivals at a fixed rate: gaps random, mean 1/λ", { size: 13 });
          G.line(40, 110, 600, 110, { color: "line" });
          const ticks = arr.map((a) => G.line(40 + a, 84, 40 + a, 110, { color: "k", w: 2 }));
          G.label(40, 130, "time →", { size: 11 });
          G.label(40, 160, "bursts happen by chance; nobody waits for the server", { size: 12, color: "ink" });
          G.text(24, 214, "queue length when λ is above capacity", { size: 13 });
          G.axes(40, 230, 560, 140, {});
          G.path("M 40 368 L 120 352 L 200 330 L 280 312 L 360 284 L 440 262 L 520 236 L 600 214", { color: "hot", w: 2.5 });
          G.path("M 40 368 L 100 360 L 140 366 L 200 356 L 260 364 L 320 354 L 380 366 L 440 358 L 520 364 L 600 360", { color: "ok", w: 2 });
          G.label(250, 280, "λ > capacity: grows", { color: "hot", size: 12 });
          G.label(470, 350, "λ < capacity: drains", { color: "ok", size: 12 });
          G.label(40, 392, "sketch of the shape, not a measurement", { size: 11 });
          G.from(ticks, { opacity: 0, stagger: 0.02, duration: 0.1 });
          G.caption("an open loop keeps sending, so overload becomes visible");
        } },

      { rail: "coordinated omission", title: "Coordinated omission: the hidden requests",
        body: `<p>Now the trap. Take a server that answers in 10 ms but freezes for 1 second once during a 10-second test (a garbage-collection pause, a long prefill, a CUDA-graph capture).</p>
<p><b>Closed loop, one user.</b> It sends every 10 ms, so it records about 900 fast requests and one that took about 1 s. During the freeze it sent nothing: it was waiting.</p>
<div class="eq">closed: 901 samples, 900 × 10 ms + 1 × ~1,000 ms
  p99 = 10 ms     mean ≈ 11 ms</div>
<p><b>Open loop at 100 requests/s</b> (the rate the user had before the freeze). About 100 requests arrive during the freeze. The first waits about 1,000 ms, the last about 10 ms, so their waits are spread evenly between.</p>
<div class="eq">open: 1,000 samples, 100 of them delayed 10–1,000 ms
  worst 1% = 10 requests that waited ≥ 900 ms
  p99 ≈ 900 ms    mean ≈ 60 ms</div>
<p>Same server, same freeze: p99 of 10 ms versus 900 ms. The closed-loop client <b>coordinated</b> with the server and <b>omitted</b> the 99 requests that real users would have sent during the freeze. Gil Tene named this <b>coordinated omission</b>. It makes tail latencies look best precisely when the system is worst.</p>`,
        check: { q: "A closed-loop benchmark reports a flat, low p99 even at very high concurrency. What is the most likely explanation?",
          options: ["The server has no tail latency", "The clients stop sending while they wait, so slow periods produce few samples", "p99 is always flat for LLM servers"], answer: 1,
          why: "In a closed loop, a stall delays the next send of every waiting user. The requests that would have arrived during the stall are never sent, never measured, and the percentile is computed without them." },
        scene(G) {
          const x0 = 40, sx = 0.056; // 10,000 ms → 560 px
          const stall0 = 4000, stall1 = 5000;
          G.rect(x0 + stall0 * sx, 40, 1000 * sx, 222, { fill: "hot", opacity: 0.12, rx: 2 });
          G.label(x0 + stall0 * sx + 4, 56, "1 s freeze", { color: "hot", size: 12 });
          G.text(x0, 90, "closed loop, 1 user", { size: 13 });
          const c = [];
          for (let t = 0; t < 10000; t += 125) if (t < stall0 || t >= stall1) c.push(G.line(x0 + t * sx, 104, x0 + t * sx, 128, { color: "k", w: 1.5 }));
          c.push(G.line(x0 + stall0 * sx, 104, x0 + stall0 * sx, 128, { color: "hot", w: 3 }));
          G.label(x0 + stall0 * sx + 4, 146, "one slow sample, then silence", { color: "hot", size: 11 });
          G.text(x0, 200, "open loop, 100 req/s", { size: 13 });
          const o = [];
          for (let t = 0; t < 10000; t += 125) o.push(G.line(x0 + t * sx, 214, x0 + t * sx, 238, { color: t >= stall0 && t < stall1 ? "hot" : "k", w: 1.5 }));
          G.label(x0 + stall0 * sx + 4, 256, "every arrival in the freeze is measured", { color: "hot", size: 11 });
          G.label(x0, 280, "(each tick here stands for about 12 requests)", { size: 11 });
          G.text(x0, 320, "closed:  p99 = 10 ms", { size: 15, color: "ok" });
          G.text(x0, 350, "open:    p99 ≈ 900 ms", { size: 15, color: "hot" });
          G.label(x0 + 300, 336, "same server, same freeze", { size: 13, color: "ink" });
          G.from(o, { opacity: 0, stagger: 0.008, duration: 0.1 });
          G.caption("the closed loop never sent the requests that would have suffered");
        } },

      { rail: "warm-up", title: "Warm-up: the first minute is a different system",
        body: `<p>The first requests to a freshly started server do work that later requests don't:</p>
<ul><li>CUDA graphs are captured for batch sizes not seen yet (P2.1), and compiled kernels are built and cached;</li><li>the operating system's page cache is still filling with weight files, so anything that touches disk is slow;</li><li>memory allocators grow their pools, and the prefix cache (P2.2, P2.3) starts empty.</li></ul>
<p>Mixing those requests into your statistics measures start-up, not serving. The fix is simple: run a <b>warm-up phase</b> first (same workload, a fixed number of requests or seconds), discard it, then measure. Say in the report how long the warm-up was.</p>
<p>The opposite mistake also exists: if your real users hit a cold server (scale-to-zero, P3.6), the cold-start latency is exactly what you must measure, separately and on purpose.</p>`,
        scene(G) {
          G.axes(50, 50, 560, 290, { ylabel: "latency of each request" });
          const R = rng(9), pts = [];
          for (let i = 0; i < 90; i++) {
            const warm = Math.max(0, 1 - i / 14);
            const y = 300 - (warm * 210 + 8 + R() * 22 + (R() < 0.05 ? 40 : 0));
            pts.push(G.circle(56 + i * 6, y, 3, { fill: i < 16 ? "v" : "k" }));
          }
          G.rect(50, 50, 16 * 6 + 6, 290, { fill: "v", opacity: 0.12, rx: 2 });
          G.label(60, 70, "warm-up: discard", { color: "v", size: 12 });
          G.label(300, 190, "steady state: measure this", { color: "k", size: 12 });
          G.label(610, 362, "request number →", { anchor: "end", size: 11 });
          G.label(50, 392, "illustrative shape: graph capture, caches and page cache filling", { size: 11 });
          G.from(pts, { opacity: 0, stagger: 0.01, duration: 0.1 });
          G.caption("the first requests measure start-up, not serving");
        } },

      { rail: "percentiles and n", title: "A percentile is only as good as its sample count",
        body: `<p>Latency is a distribution, not a number. Sort all the samples: the <b>p50</b> (median) is the value halfway along, the <b>p90</b> is the value 90% of the way along, the <b>p99</b> is 99% of the way. Tail percentiles matter because a user making 100 requests will hit the p99 about once.</p>
<p>But a percentile is computed from very few samples near the end. With n samples, roughly <code>n × (1 − q)</code> of them lie above the q-th percentile:</p>
<div class="eq">n = 120:    p90 has 12 samples above it
            p99 has ~1 sample above it
n = 1,000:  p99 has 10 samples above it</div>
<p>With 120 samples, p99 sits between the second- and third-largest values: one unlucky request moves it. So: always <b>report n</b> next to each percentile, prefer p50 and p90 for small runs, and report p99 only when n is at least about 1,000. Exercise 2 makes <code>report.py</code> warn when n &lt; 100.</p>`,
        check: { q: "Your run has 150 successful requests. What is the honest way to report the tail?",
          options: ["Report p99.9: it is the most informative", "Report p50 and p90 with n = 150, and treat p99 as roughly the 2nd-largest value", "Report the mean: it uses all the samples"], answer: 1,
          why: "Only about 1.5 samples lie above the p99 of 150 values, so it is essentially the 2nd-largest sample and very noisy. p90 has 15 samples behind it. The mean hides the tail entirely." },
        scene(G) {
          const R = rng(3), n = 120, v = [];
          for (let i = 0; i < n; i++) v.push(Math.exp(R() * 1.2 + (R() < 0.04 ? 1.2 : 0)));
          v.sort((a, b) => a - b);
          const mx = v[n - 1];
          G.text(24, 36, "120 latency samples, sorted", { size: 13 });
          const bars = G.bars(30, 300, v, { w: 3.6, gap: 1, h: 230, max: mx, fill: (i) => (i >= 118 ? "hot" : i >= 108 ? "v" : "k") });
          const at = (q) => 30 + q * (n - 1) * 4.6 + 1.8;
          [["p50", 0.5, "ok"], ["p90", 0.9, "v"], ["p99", 0.99, "hot"]].forEach(([l, q, c], i) => {
            G.line(at(q), 306, at(q), 330 + i * 18, { color: c });
            G.label(at(q) - 4, 342 + i * 18, l, { color: c, size: 12, anchor: "end" });
          });
          G.label(24, 410, "red: the ~1 sample above p99 · amber: the 12 above p90", { size: 12, color: "ink" });
          G.from(bars, { attr: { height: 0, y: 300 }, stagger: 0.004, duration: 0.2 });
          G.caption("the tail percentiles rest on a handful of samples");
        } },

      { rail: "repeats", title: "One run is an anecdote: repeat and report the spread",
        body: `<p>Two runs of the same benchmark on the same machine give different numbers: other processes, clock and thermal behaviour, network jitter, the random timetable. Before claiming that a change helped, you need to know how big that noise is.</p>
<ul><li><b>Repeat the whole run</b> at least 3 times, from warm-up onwards.</li><li>Report the <b>median</b> of the run-level statistic and its <b>spread</b> (min–max for 3–5 runs).</li><li><b>Fix the seed</b> of the workload, so every run and every engine sees the same requests (<code>loadgen --seed</code>).</li><li><b>Change one thing at a time</b>, so a difference has one cause.</li></ul>
<div class="eq">made-up example: p90 TTFT over 3 runs
  baseline:  0.41, 0.38, 0.52 s → 0.41 s (0.38–0.52)
  change:    0.36, 0.40, 0.35 s → 0.36 s (0.35–0.40)</div>
<p>The medians differ by 0.05 s, but the ranges overlap. Honest conclusion: "no clear difference at 3 runs". Run more, or reduce the noise, before writing "12% faster".</p>`,
        scene(G) {
          const sx = 900, base = 0.3, x = (s) => 120 + (s - base) * sx;
          G.text(24, 40, "p90 TTFT per run (made-up example values)", { size: 13 });
          const rows = [["baseline", [0.41, 0.38, 0.52], "k", 110], ["change", [0.36, 0.40, 0.35], "v", 210]];
          const dots = [];
          rows.forEach(([name, vals, c, y]) => {
            G.text(24, y + 5, name, { size: 13 });
            const lo = Math.min(...vals), hi = Math.max(...vals), med = vals.slice().sort()[1];
            G.line(x(lo), y, x(hi), y, { color: c, w: 3 });
            vals.forEach((s) => dots.push(G.circle(x(s), y, 7, { fill: c })));
            G.line(x(med), y - 18, x(med), y + 18, { color: "ink", w: 2 });
            G.label(x(med), y + 34, "median " + med.toFixed(2) + " s", { anchor: "middle", size: 11, color: "ink" });
          });
          G.rect(x(0.38), 70, x(0.40) - x(0.38), 170, { fill: "hot", opacity: 0.15, rx: 2 });
          G.label(x(0.39), 262, "ranges overlap", { anchor: "middle", color: "hot", size: 12 });
          [0.3, 0.35, 0.4, 0.45, 0.5, 0.55].forEach((s) => G.label(x(s), 300, s.toFixed(2), { anchor: "middle", size: 11 }));
          G.line(x(0.3), 284, x(0.55), 284, { color: "line" });
          G.text(24, 350, "verdict: not distinguishable at n = 3 runs", { size: 14, color: "hot" });
          G.from(dots, { opacity: 0, scale: 0.3, transformOrigin: "center", stagger: 0.08, duration: 0.25 });
          G.caption("a difference smaller than the run-to-run spread is not a result");
        } },

      { rail: "the tools", title: "Three tools, three answers: reconcile before you trust",
        body: `<p>You will use three load tools on the same server (exercise 3):</p>
<ul><li><b><code>platform/loadgen</code></b> (#4): open-loop Poisson, seeded, yours line by line;</li><li><b>GuideLLM</b> (pinned <code>v0.8.0</code>): standard sweeps (synchronous, throughput, constant or Poisson rates) against any OpenAI-compatible server;</li><li><b><code>vllm bench serve</code></b>: dataset-driven, at a request rate or as fast as possible, comparable with vLLM's own CI.</li></ul>
<p>Run them at the same rate and workload shape and they still disagree. Before trusting any of them, find out why. The usual causes are definitions:</p>
<ul><li><b>Token counting.</b> loadgen approximates output tokens by counting words; the other tools count with a tokenizer or the server's <code>usage</code> field (check which at the pinned version). Tokens per second differ by that ratio.</li><li><b>Where the clock starts.</b> TTFT from "request sent", from "connection opened", or from the scheduled arrival time.</li><li><b>Output length.</b> Whether <code>ignore_eos</code> forces the full length or the model may stop early.</li><li><b>Warm-up</b> and <b>load model</b>: open loop, closed loop, or "as fast as possible".</li></ul>
<p>And never compare <code>vllm bench throughput</code> or <code>vllm bench latency</code> with a serving number: those run offline, without HTTP, with all prompts available at once.</p>`,
        check: { q: "Why can't an offline <code>vllm bench throughput</code> result be compared with a serving benchmark?",
          options: ["It uses a different model", "It has no HTTP and no arrivals: all prompts are available up front and batched maximally, so it measures engine capacity, not latency under load", "It only measures prefill"], answer: 1,
          why: "Offline throughput answers 'how fast can the engine chew through a pile of prompts?'. A serving benchmark answers 'what latency do users see when requests arrive over time through the API?'. Different questions, different numbers." },
        scene(G) {
          G.text(24, 36, "one streamed request: where can the clock start and stop?", { size: 13 });
          const y = 110, xs = [60, 170, 290, 420, 560];
          const lab = ["scheduled", "connect", "send", "first byte", "first token"];
          G.line(40, y, 610, y, { color: "line", w: 2 });
          xs.forEach((x, i) => { G.circle(x, y, 7, { fill: i === 4 ? "ok" : "k" }); G.label(x, y - 18, lab[i], { anchor: "middle", size: 12, color: "ink" }); });
          const spans = [[60, "loadgen (scheduled → first token)", "k", 150], [290, "a tool timing from send", "v", 180], [170, "a tool timing from connect", "q", 210]];
          const sp = spans.map(([x0, l, c, yy]) => { const r = G.rect(x0, yy, 560 - x0, 12, { fill: c, rx: 3 }); G.label(x0, yy - 4, l, { size: 11, color: c }); return r; });
          G.text(24, 270, "tokens/s depends on what a \"token\" is", { size: 13 });
          G.box(40, 290, 170, 50, "word count", { stroke: "k", color: "k", size: 13 });
          G.box(230, 290, 170, 50, "tokenizer", { stroke: "v", color: "v", size: 13 });
          G.box(420, 290, 170, 50, "usage field", { stroke: "ok", color: "ok", size: 13 });
          G.label(40, 364, "same text, different counts → different tokens/s", { size: 12, color: "ink" });
          G.label(40, 386, "exact definitions: read each tool's source at the pinned version", { size: 12 });
          G.from(sp, { attr: { width: 0 }, stagger: 0.2, duration: 0.5 });
          G.caption("disagreement is usually a definition, not a bug in the server");
        } },

      { rail: "the report", title: "The report a stranger can reproduce",
        body: `<p>A benchmark is finished when someone else can rerun it and get the same answer within your stated spread. That needs every choice from this lesson written down. <code>examples/report.py</code> renders a report from a loadgen JSON and <b>refuses</b> to render if a required field is missing:</p>
<ul><li><b>hardware:</b> GPU model and count, driver, CUDA, instance type and region;</li><li><b>software:</b> engine and version (commit SHA);</li><li><b>model:</b> repository and revision, dtype, quantization;</li><li><b>flags:</b> every non-default engine flag;</li><li><b>load:</b> tool and version, open or closed loop, rates, duration, warm-up, seed, prompt and output length distributions, tokenizer;</li><li><b>statistics:</b> n per point, median and p90 (p99 only with n ≥ 1,000), number of repeats and spread;</li><li><b>raw data:</b> the JSON, committed next to the report.</li></ul>
<div class="eq">uv run python $D/examples/report.py results/lg.json \\
  --hardware "1x L4 (g6.xlarge, us-east-1) ..." \\
  --versions "vllm 0.31.0 (d1f3d8b8)" \\
  --model "&lt;repo&gt;@&lt;sha&gt; bf16" --flags "..."</div>
<p>Exercise 2 extends it: a variance column for repeated runs, a warning when n &lt; 100, and a refusal when the hardware field names no actual device ("my machine" is not reproducible).</p>`,
        scene(G) {
          G.rect(100, 24, 470, 380, { stroke: "ink", rx: 10 });
          G.text(120, 54, "# Benchmark report", { size: 15 });
          const f = [["hardware", "1x L4, driver, CUDA, region", true], ["software", "vllm 0.31.0 (d1f3d8b8)", true], ["model", "<repo>@<sha>, bf16", true], ["flags", "every non-default flag", true], ["load", "open loop, seed, warm-up", true], ["statistics", "n, p50, p90, repeats", true], ["raw data", "results/*.json committed", true], ["hardware = \"my machine\"", "refused: not reproducible", false]];
          const rows = f.map(([k, v, ok], i) => {
            const y = 78 + i * 38, g = G.group();
            G.circle(130, y + 10, 9, { fill: ok ? "ok" : "hot", parent: g });
            G.text(130, y + 14, ok ? "✓" : "×", { anchor: "middle", size: 12, color: "bg", parent: g });
            G.text(150, y + 14, k, { size: 13, color: ok ? "ink" : "hot", parent: g });
            G.label(370, y + 14, v, { size: 11, parent: g });
            return g;
          });
          G.from(rows, { opacity: 0, x: -10, stagger: 0.1, duration: 0.3 });
          G.caption("report.py refuses to render without the fields that make it reproducible");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A closed loop sets its own rate from the server's latency (rate = N / latency), so it can't overload the server. An open loop sends on a timetable and can.</li>
<li>Coordinated omission: a closed-loop client stops sending during stalls, so the worst latencies are never sampled.</li>
<li>Discard warm-up. Report n beside every percentile; p99 needs about 1,000 samples.</li>
<li>Repeat runs, fix the seed, change one thing at a time, and report the median with its spread.</li>
<li>Tools disagree over definitions (token counting, where TTFT starts, load model). Reconcile before trusting, and never compare offline numbers with serving numbers.</li></ul>`,
    sim: {
      title: "Closed loop vs open loop on a stalling server",
      intro: "A server with C parallel slots answers each request in S ms, but freezes for D ms once every P seconds. First a closed loop of N users runs for 30 s; then an open loop sends Poisson arrivals at the same mean rate the closed loop achieved. Each dot is one request: its send time across, its latency up. Press Replay to watch it unfold.",
      controls: [
        { id: "N", label: "closed-loop users (N)", min: 1, max: 32, value: 4 },
        { id: "C", label: "server slots (C)", min: 1, max: 32, value: 8 },
        { id: "S", label: "service time per request, ms (S)", min: 10, max: 400, step: 10, value: 100 },
        { id: "D", label: "stall length, ms (D)", min: 0, max: 3000, step: 100, value: 1000 },
        { id: "P", label: "one stall every P seconds", min: 2, max: 30, value: 15 },
        { id: "seed", label: "open-loop seed", min: 0, max: 9, value: 0 },
      ],
      height: 300,
      run: { label: "Replay", frames: 40, ms: 50 },
      draw(G, v, t) {
        const T = 30000, P = v.P * 1000;
        const closed = closedLoop(v.N, v.C, v.S, P, v.D, T);
        const lam = closed.length / (T / 1000);
        const open = openLoop(lam, v.C, v.S, P, v.D, T, v.seed);
        const ymax = Math.max(v.S * 2, ...closed.map((r) => r[1]), ...open.map((r) => r[1]));
        const x0 = 110, W = 520, now = T * t;
        const lanes = [["closed loop", closed, "k", 16], [`open, ${lam.toFixed(1)}/s`, open, "v", 150]];
        lanes.forEach(([name, rs, c, y]) => {
          const h = 110;
          G.text(8, y + 50, name, { size: 12 });
          G.label(8, y + 68, `n = ${rs.length}`, { size: 11 });
          G.rect(x0, y, W, h, { stroke: "line", rx: 3 });
          for (let s = P / 2; s < T; s += P) G.rect(x0 + (W * s) / T, y, Math.max(2, (W * v.D) / T), h, { fill: "hot", opacity: 0.15, rx: 0 });
          rs.forEach(([a, l]) => { if (a <= now) G.circle(x0 + (W * a) / T, y + h - 3 - (h - 8) * Math.sqrt(l / ymax), 1.6, { fill: c }); });
          G.label(x0 - 4, y + 10, F.ms(ymax / 1000), { anchor: "end", size: 10 });
        });
        G.label(x0, 278, "0 s", { size: 10 }); G.label(x0 + W, 278, "30 s", { size: 10, anchor: "end" });
        G.label(x0 + W / 2, 278, "send time → · latency up (square-root scale) · red bands = stalls", { size: 10, anchor: "middle" });
        const st = (rs) => { const l = rs.map((r) => r[1]).sort((a, b) => a - b); return { p50: pct(l, 0.5), p90: pct(l, 0.9), p99: pct(l, 0.99), max: l[l.length - 1], mean: l.reduce((s, x) => s + x, 0) / l.length }; };
        const a = st(closed), b = st(open);
        const ms = (x) => F.ms(x / 1000);
        return [
          { title: "Tail: closed · open loop", rows: [["p90", `${ms(a.p90)} · ${ms(b.p90)}`], ["p99", `${ms(a.p99)} · ${ms(b.p99)}`], ["p99 ratio", (b.p99 / a.p99).toFixed(1) + "×"]],
            chip: [b.p99 <= a.p99 * 1.5, b.p99 <= a.p99 * 1.5 ? "tails agree" : "closed loop hides the tail"] },
          { title: "Closed loop", rows: [["requests (n)", F.num(closed.length)], ["achieved rate", lam.toFixed(1) + " req/s"], ["p50 · p90", `${ms(a.p50)} · ${ms(a.p90)}`], ["mean · max", `${ms(a.mean)} · ${ms(a.max)}`]] },
          { title: "Open loop (Poisson)", rows: [["requests (n)", F.num(open.length)], ["p50 · p90", `${ms(b.p50)} · ${ms(b.p90)}`], ["mean · max", `${ms(b.mean)} · ${ms(b.max)}`]],
            html: `<p class="note">Model: a FIFO server with C identical slots and fixed service time; stalls freeze all progress. Real engines batch and vary, but coordinated omission works the same way. Exercise 1 shows it on the mock server.</p>` },
        ];
      },
    },
    practice: {
      intro: "Exercises 1 and 2 are T0 (the mock server on a laptop; the coordinated-omission test is marked slow, so select it with -m slow). Exercise 3 needs one GPU: aws.md has the three tool commands, the cost lookup and make down.",
      items: [
        { title: "Show coordinated omission", tier: "T0 · medium", goal: "Run the closed-vs-open example on a mock with 8 slots, then explain in three sentences why the closed-loop p99 is a lie. The test asserts open-loop p99 TTFT is higher at the same mean rate.",
          cmd: "uv run pytest -m slow course/P2-serving-engines/P2.4-benchmarking-methodology/exercises/test_co.py" },
        { title: "The report generator", tier: "T0 · easy", goal: "Extend render(): a variance column for repeated runs, a warning when n < 100, and a refusal when --hardware names no device. Delete each pytest.skip as you finish it.",
          cmd: "uv run pytest course/P2-serving-engines/P2.4-benchmarking-methodology/exercises/test_report.py" },
        { title: "GuideLLM vs #4 vs vllm bench serve", tier: "T2 · medium", goal: "Run all three against the same server at 4 req/s, fill the reconciliation table, and trace every >10% difference to a file and function in the tools' source.",
          cmd: "PYTHONPATH=platform python -m loadgen.cli --url http://127.0.0.1:8000 --rates 2 4 8 --duration 60 --out results/p24_loadgen.json" },
      ],
      labs: [
        { label: "Closed vs open loop at the same mean rate", path: "course/P2-serving-engines/P2.4-benchmarking-methodology/examples/01_closed_vs_open.py" },
        { label: "Report generator (required fields enforced)", path: "course/P2-serving-engines/P2.4-benchmarking-methodology/examples/report.py" },
        { label: "AWS guide: GuideLLM and vllm bench serve commands (flags UNVERIFIED at the pinned versions), teardown", path: "course/P2-serving-engines/P2.4-benchmarking-methodology/aws.md" },
        { label: "The seeded Poisson workload and knee analysis", path: "platform/loadgen/workload.py, platform/loadgen/analysis.py" },
      ],
    },
  });
})();
