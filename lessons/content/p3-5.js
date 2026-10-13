/* P3.5 — Observability: metrics, histograms, SLOs and burn-rate alerts, traces, cardinality, and $/1M tokens (#13, #2). */
(function () {
  const F = S2S.fmt;

  /* ---- burn-rate alert simulation, the same logic as platform/observability/rules/slo.yaml ----
     One error ratio per minute (share of requests slower than the SLO threshold). A window ratio is the mean over the
     window (equal request rate every minute), with a healthy history before minute 0. */
  const HORIZON = 600, START = 30;
  function burnSim(p, L, budget) {
    const e = (m) => (m >= START && m < START + L ? p : 0);
    const win = (m, w) => { let s = 0; for (let k = 0; k < w; k++) s += e(m - k); return s / w; };
    const th1 = 14.4 * budget, th2 = 6 * budget;
    const r5 = [], r30 = [], r1h = [], r6h = [], page = [], tick = [];
    for (let m = 0; m < HORIZON; m++) {
      r5.push(win(m, 5)); r30.push(win(m, 30)); r1h.push(win(m, 60)); r6h.push(win(m, 360));
      page.push(r1h[m] > th1 && r5[m] > th1); tick.push(r6h[m] > th2 && r30[m] > th2);
    }
    const span = (a) => { const f = a.indexOf(true); if (f < 0) return null; let l = f; while (l + 1 < a.length && a[l + 1]) l++; return [f, l + 1]; };
    return { r5, r30, r1h, r6h, page, tick, th1, th2, pageSpan: span(page), tickSpan: span(tick) };
  }

  S2S.lesson({
    id: "p3-5", n: "P3.5", title: "Observability",
    subtitle: "Deployment and infra · first principles · T0 for rules, tracing and cost; T3 for live dashboards",
    kicker: "Lesson · ≈ 50 min",
    headline: "Alert on how fast you are spending trust, not on a number",
    intro: `<p>P3.4 gave you a cluster. Now it is running, and a user says "it was slow around three o'clock". What do you look at? This lesson builds the answer from the ground up: the three kinds of telemetry and the question each one answers, how Prometheus metrics work (counters, gauges, histograms), how to turn a histogram into a percentile without lying, how to define a service-level objective and an error budget, and how to alert on the <b>speed</b> at which the budget burns using two time windows at once. Then traces across the gateway, why a request id must never be a metric label, and the arithmetic that turns GPU prices and token counts into dollars per million tokens. You build two projects: #13 (the observability spine) and #2 (the cost dashboard).</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "5 exercises"],
    legend: [["k", "signal"], ["v", "window / budget"], ["ok", "healthy"], ["hot", "alert / problem"]],
    prev: "p3-4", next: "p3-6",
    steps: [
      /* 1 */
      { rail: "three signals", title: "Three questions, three kinds of evidence",
        body: `<p>A running system produces three kinds of evidence, <b>telemetry</b>. They differ in cost and in the question they can answer.</p>
<table><tr><th>question</th><th>signal</th><th>example</th></tr>
<tr><td>Is it healthy right now? Is it getting worse?</td><td><b>metrics</b>: numbers sampled over time, aggregated, cheap</td><td>p90 time to first token, KV-cache usage, waiting requests</td></tr>
<tr><td>Why was <i>this</i> request slow?</td><td><b>traces</b>: the path of one request through the services, in timed pieces, usually sampled</td><td>gateway 3 ms, attempt 1 failed in 80 ms, attempt 2 took 1.9 s to first token</td></tr>
<tr><td>What exactly happened?</td><td><b>logs</b>: one text line per event, verbose</td><td>the vLLM error line, a preemption message</td></tr></table>
<p>Use them in that order. A metric tells you <i>that</i> something is wrong (and is small enough to keep for months). A trace tells you <i>where</i> in one slow request the time went. A log tells you the detail of the event. Trying to answer "is it healthy?" from logs means grepping millions of lines; trying to answer "why was this one slow?" from an average is impossible.</p>
<p>This lesson spends most of its time on metrics because they are what you alert on and what autoscaling (P3.6) reads.</p>`,
        scene(G) {
          const cols = [["metrics", "k", "numbers over time", "cheap · aggregated", "is it healthy?", "p90 TTFT 1.8 s"], ["traces", "v", "one request, in spans", "sampled", "why was THIS slow?", "attempt 2: 1.9 s"], ["logs", "q", "one line per event", "verbose · costly", "what exactly happened?", "preempted request 41"]];
          const g = cols.map(([n, c, a, b, q, ex], i) => {
            const x = 24 + i * 200, gg = G.group();
            G.rect(x, 60, 184, 300, { stroke: c, rx: 10, parent: gg });
            G.text(x + 92, 90, n, { anchor: "middle", size: 16, color: c, parent: gg });
            G.text(x + 92, 126, a, { anchor: "middle", size: 12, parent: gg });
            G.text(x + 92, 148, b, { anchor: "middle", size: 12, color: "muted", parent: gg });
            G.line(x + 20, 172, x + 164, 172, { color: "line", parent: gg });
            G.text(x + 92, 206, q, { anchor: "middle", size: 13, color: c, parent: gg });
            G.box(x + 12, 250, 160, 44, ex, { stroke: c, color: "ink", size: 11, parent: gg });
            G.text(x + 92, 330, "step " + (i + 1) + " of the hunt", { anchor: "middle", size: 11, color: "muted", parent: gg });
            return gg;
          });
          G.arrow(210, 36, 410, 36, { color: "muted", w: 1.5 }); G.label(24, 40, "start here", { size: 11 });
          G.label(24, 394, "metrics say THAT, traces say WHERE, logs say WHAT", { color: "ink" });
          G.from(g, { opacity: 0, y: 20, stagger: 0.2, duration: 0.4 });
          G.caption("each signal answers one question cheaply; none answers all three");
        } },

      /* 2 */
      { rail: "metric types", title: "Counters go up, gauges move, histograms count by bucket",
        body: `<p><b>Prometheus</b> is the metrics database. Every <b>scrape interval</b> (say 15 s) it fetches the <code>/metrics</code> text page of each pod (vLLM serves one) and stores each number with a timestamp. Three metric types cover almost everything.</p>
<ul><li>A <b>counter</b> only ever increases (it resets to 0 when the process restarts). Example: <code>gateway_requests_total</code>. The raw value is useless; what matters is how fast it grows. <code>rate(x[5m])</code> is the increase over the last 5 minutes divided by the seconds. A counter that read 1,200 and then 1,800 five minutes later grew by 600 over 300 s: <b>2 requests per second</b>.</li>
<li>A <b>gauge</b> is a value that goes up and down, read as it is. Example: <code>vllm:num_requests_waiting</code>, <code>vllm:kv_cache_usage_perc</code>.</li>
<li>A <b>histogram</b> records how values are distributed. It is the right type for latency, and it needs its own step.</li></ul>
<p>One rule of thumb: the <code>rate()</code> window needs at least four scrapes in it. With a 15 s scrape, use at least 60 s; <code>[5m]</code> is the usual choice. A window shorter than that gives empty or jumpy answers.</p>`,
        scene(G) {
          // counter
          G.label(24, 40, "counter: requests_total", { color: "k", size: 12 });
          G.axes(24, 50, 180, 120, {});
          const cp = [[24, 170], [54, 160], [84, 138], [114, 118], [144, 98], [174, 80], [204, 62]];
          G.path("M " + cp.map((p) => p.join(" ")).join(" L "), { color: "k", w: 3 });
          G.label(24, 192, "only goes up", {}); G.label(24, 212, "rate() = slope", { color: "ink" });
          // gauge
          G.label(236, 40, "gauge: num_requests_waiting", { color: "v", size: 12 });
          G.axes(236, 50, 180, 120, {});
          const gp = [[236, 140], [260, 100], [284, 130], [308, 70], [332, 120], [356, 90], [380, 150], [416, 110]];
          G.path("M " + gp.map((p) => p.join(" ")).join(" L "), { color: "v", w: 3 });
          G.label(236, 192, "up and down", {}); G.label(236, 212, "read as is", { color: "ink" });
          // histogram
          G.label(448, 40, "histogram: TTFT", { color: "q", size: 12 });
          G.axes(448, 50, 176, 120, {});
          G.bars(456, 170, [30, 55, 38, 14, 6, 2], { w: 22, gap: 6, h: 110, max: 60, fill: "q" });
          ["0.1", "0.5", "1", "2", "5", "+"].forEach((l, i) => G.label(467 + i * 28, 186, l, { anchor: "middle", size: 9 }));
          G.label(448, 212, "count per bucket", { color: "ink" });
          // worked rate
          G.rect(24, 250, 592, 150, { stroke: "line", rx: 10 });
          G.text(40, 276, "rate() by hand", { size: 13, color: "ink" });
          G.text(40, 304, "t = 0 min:   counter = 1,200", { size: 13 });
          G.text(40, 328, "t = 5 min:   counter = 1,800", { size: 13 });
          G.text(40, 356, "(1,800 − 1,200) ÷ 300 s = 2 requests/s", { size: 14, color: "ok" });
          G.label(40, 384, "window ≥ 4 scrapes: 15 s scrape → at least 60 s", {});
          G.from(G.svg.querySelectorAll("path"), { opacity: 0, duration: 0.8 });
          G.caption("the type decides which query makes sense");
        } },

      /* 3 */
      { rail: "histograms", title: "A percentile from buckets, and why you never average one",
        body: `<p>"Average latency" hides the slow tail users complain about, so you track <b>percentiles</b>: p90 is the value that 90% of requests beat. But the server cannot store every request's latency forever. A <b>histogram</b> stores counts per <b>bucket</b>: "how many requests took at most 0.5 s", "at most 2 s", "at most +Inf (all)". The counts are <i>cumulative</i>, and the bucket's upper bound is its <code>le</code> label ("less than or equal").</p>
<p>To get a percentile, <code>histogram_quantile</code> finds the bucket where the cumulative count crosses the rank and <b>interpolates</b> linearly inside it. Worked example (exercise 1): 50% of requests are ≤ 0.5 s, 95% are ≤ 2 s, 100% are ≤ +Inf. The 90% mark falls inside the (0.5, 2] bucket, which holds 45% of requests:</p>
<div class="eq">p90 = 0.5 + (0.90 − 0.50) ÷ (0.95 − 0.50) × (2 − 0.5)
    = 0.5 + 0.889 × 1.5 = 1.83 s</div>
<div class="eq">histogram_quantile(0.90,
  sum by (le) (rate(
    vllm:time_to_first_token_seconds_bucket[5m])))</div>
<p>Note the <code>sum by (le)</code>: combine the <b>buckets</b> of all pods first, then take the quantile. Averaging each pod's p90 is wrong. Example: pod A serves 900 requests, all ≤ 1 s (its p90 is 0.9 s). Pod B serves 100 requests, all between 2 and 5 s (its p90 is 4.7 s). The mean of the two p90s is 2.8 s, but across all 1,000 requests exactly 900 are ≤ 1 s, so the fleet p90 is <b>1.0 s</b>. The percentile of a mixture is not a mixture of percentiles.</p>
<p>The interpolation is only as good as the buckets: if no boundary falls near your SLO threshold, the answer is a guess inside a wide bucket.</p>`,
        check: { q: "Pod A: 900 requests, p90 0.9 s. Pod B: 100 requests, p90 4.7 s. What is the right way to get the fleet p90?",
          options: ["avg(p90 of each pod) = 2.8 s", "Sum the histogram buckets across pods, then histogram_quantile", "Take the larger of the two p90s"], answer: 1,
          why: "Percentiles do not average, because the pods carry different numbers of requests and different distributions. Summing the bucket counts reconstructs the combined distribution, from which any quantile can be read. Here that gives 1.0 s, not 2.8 s." },
        scene(G) {
          G.text(24, 34, "cumulative share of requests per bucket (le)", { size: 13 });
          const x0 = 70, base = 270, H = 200;
          const Y = (f) => base - H * f;
          G.axes(x0 - 10, base - H, 520, H, {});
          [0, 0.5, 0.9, 1].forEach((f) => G.label(x0 - 16, Y(f) + 4, Math.round(f * 100) + "%", { anchor: "end", size: 10 }));
          const bs = [["le 0.5", 0.5, 120], ["le 2", 0.95, 280], ["le +Inf", 1, 440]];
          const rs = bs.map(([l, f, x]) => { const r = G.rect(x, Y(f), 90, H * f, { fill: "q", rx: 3, opacity: 0.85 }); G.label(x + 45, base + 18, l, { anchor: "middle", size: 12 }); G.text(x + 45, Y(f) - 6, Math.round(f * 100) + "%", { anchor: "middle", size: 12 }); return r; });
          G.line(x0 - 10, Y(0.9), 600, Y(0.9), { color: "hot", dash: "5 4", w: 2 }); G.label(x0 - 6, Y(0.9) - 6, "rank 90%", { color: "hot", size: 11 });
          G.line(210, Y(0.5), 370, Y(0.9), { color: "ok", w: 2 }); G.circle(370, Y(0.9), 5, { fill: "ok" });
          G.label(84, Y(0.7), "interpolate inside", { color: "ok", size: 11 }); G.label(84, Y(0.7) + 15, "the (0.5, 2] bucket", { color: "ok", size: 11 });
          G.text(24, 322, "p90 = 0.5 + 0.889 × 1.5 = 1.83 s", { size: 14, color: "ok" });
          G.rect(24, 340, 592, 66, { stroke: "hot", rx: 8, dash: "5 4" });
          G.text(36, 364, "avg of pod p90s: (0.9 + 4.7) ÷ 2 = 2.8 s   ✗", { size: 12, color: "hot" });
          G.text(36, 388, "sum buckets first: 900 of 1000 ≤ 1 s → p90 = 1.0 s   ✓", { size: 12, color: "ok" });
          G.from(rs, { attr: { height: 0 }, stagger: 0.2, duration: 0.5 });
          G.caption("combine buckets, then take the quantile");
        } },

      /* 4 */
      { rail: "SLO & budget", title: "Say what \"good enough\" means, then count the slack",
        body: `<p>"Keep TTFT low" cannot be alerted on. A service-level objective makes it precise in three layers:</p>
<ul><li><b>SLI</b> (indicator): a measured fraction. Here: the share of requests whose TTFT is under 2 s.</li>
<li><b>SLO</b> (objective): the target for the SLI. Here: at least 99%, over a rolling 30 days.</li>
<li><b>Error budget</b>: what the SLO allows to go wrong. 100% − 99% = <b>1%</b> of requests may be slow.</li></ul>
<p>The budget is the point. A perfect system is not the goal: it would never deploy, experiment, or run on cheap spot GPUs (P3.6). The budget says how much unreliability you may <i>spend</i>. While budget remains, ship. When it is gone, freeze risky changes.</p>
<p>Worked example at 100 requests per minute:</p>
<div class="eq">requests in 30 days = 100 × 1440 × 30 = 4,320,000
budget = 1% of that = 43,200 slow requests</div>
<p>The SLI is computed from the histogram: <code>bucket{le="2"} / count</code>, the share at most 2 s. That only works if the histogram has a bucket boundary at exactly 2. Choose your SLO threshold at an existing boundary of vLLM's buckets (check the list at the pinned version) rather than the other way round.</p>`,
        scene(G) {
          G.text(24, 34, "30 days at 100 req/min = 4,320,000 requests", { size: 13 });
          G.rect(24, 56, 592, 44, { fill: "ok", rx: 6, opacity: 0.55 });
          G.rect(24 + 592 * 0.99, 56, 592 * 0.01, 44, { fill: "hot", rx: 2 });
          G.text(320, 84, "99%: fast enough (TTFT under 2 s)", { anchor: "middle", size: 13 });
          G.line(24 + 592 * 0.995, 100, 24 + 592 * 0.995, 128, { color: "hot", w: 2 }); G.label(24 + 592 * 0.995 - 4, 146, "1% budget = 43,200 slow requests", { color: "hot", anchor: "end", size: 12 });
          G.rect(24, 180, 592, 130, { stroke: "line", rx: 10 });
          [["SLI", "share of requests with TTFT < 2 s", "k"], ["SLO", "SLI ≥ 99% over 30 days", "v"], ["budget", "1% may be slow: spend it on deploys, spot, experiments", "ok"]].forEach(([n, t, c], i) => { G.box(36, 196 + i * 38, 70, 28, n, { fill: c, size: 12 }); G.text(120, 215 + i * 38, t, { size: 12, color: "ink" }); });
          G.label(24, 340, "bucket {le=\"2\"} ÷ count needs a bucket boundary at exactly 2", { color: "ink" });
          G.label(24, 364, "budget left → ship; budget gone → freeze risky changes", {});
          G.caption("the error budget turns \"reliable\" into a number you can spend");
        } },

      /* 5 */
      { rail: "burn rate", title: "Alert on how fast the budget burns",
        body: `<p>Why not alert on "p90 TTFT is above 2 s right now"? Because it is noisy (one slow minute pages someone) and blind to slow leaks (1.5% slow for a week passes every check yet breaks the SLO).</p>
<p>Define the <b>burn rate</b> as the speed the budget is being used, relative to the rate that would use exactly all of it in the window:</p>
<div class="eq">burn rate = error ratio ÷ budget
20% slow ÷ 1% budget = 20×
0.5% slow ÷ 1% budget = 0.5×   (never runs out)</div>
<p>Burn rate 1 spends the whole 30-day budget in exactly 30 days. A burn rate of <i>B</i> exhausts it in 30 ÷ <i>B</i> days. That gives a natural alert ladder, from the Google SRE workbook:</p>
<div class="eq">14.4× → 30 ÷ 14.4 = 2.08 days   → page a human
 6×   → 30 ÷  6   = 5 days      → open a ticket
1h at 14.4× = 14.4 × 1 ÷ 720 h = 2% of budget gone</div>
<p>Why 14.4? Because one hour at that pace burns exactly 2% of a 30-day (720 hour) budget, a size most teams agree is worth waking someone for. Six hours at 6× burns 6 × 6 ÷ 720 = 5%, worth a ticket. The repo encodes both in <code>platform/observability/rules/slo.yaml</code> as <code>TTFTBudgetFastBurn</code> (severity page) and <code>TTFTBudgetSlowBurn</code> (severity ticket).</p>`,
        check: { q: "The SLO is 99% over 30 days and 20% of requests are currently slow. If nothing changes, how long until the whole error budget is spent?",
          options: ["About 36 hours", "About 5 days", "About 30 days"], answer: 0,
          why: "Burn rate = 20% ÷ 1% = 20×. The budget lasts 30 days ÷ 20 = 1.5 days, which is 36 hours. This is far above the 14.4× page threshold, so a human is paged." },
        scene(G) {
          const x0 = 60, x1 = 600, yb = 330, yt = 70;
          const X = (d) => x0 + ((x1 - x0) * d) / 30, Y = (f) => yb - (yb - yt) * f;
          G.axes(x0, yt, x1 - x0, yb - yt, { xlabel: "days →" });
          G.label(x0, yt - 12, "error budget remaining", { size: 11 });
          [0, 5, 10, 20, 30].forEach((d) => G.label(X(d), yb + 16, String(d), { anchor: "middle", size: 10 }));
          [0, 0.5, 1].forEach((f) => G.label(x0 - 6, Y(f) + 4, Math.round(f * 100) + "%", { anchor: "end", size: 10 }));
          const lines = [[1, "ok", "1×  lasts 30 days"], [6, "v", "6×  empty on day 5"], [14.4, "hot", "14.4×  day 2.08"]];
          const ps = lines.map(([b, c, l]) => { const d = Math.min(30, 30 / b); const p = G.line(X(0), Y(1), X(d), Y(1 - d / (30 / b)), { color: c, w: 3 }); G.circle(X(d), Y(0), 5, { fill: c }); return p; });
          G.text(300, 130, "14.4×: empty in 2.08 days · page", { size: 12, color: "hot" });
          G.text(300, 154, "6×: empty in 5 days · ticket", { size: 12, color: "v" });
          G.text(300, 178, "1×: lasts the full 30 days", { size: 12, color: "ok" });
          G.rect(24, 360, 592, 50, { stroke: "line", rx: 8 });
          G.text(36, 382, "burn = error ratio ÷ budget · empty in 30 ÷ burn days", { size: 12 });
          G.label(36, 400, "1 h at 14.4× = 2% of the budget · 6 h at 6× = 5%", {});
          G.from(ps, { opacity: 0, stagger: 0.3, duration: 0.5 });
          G.caption("the steeper the line, the sooner a human must know");
        } },

      /* 6 */
      { rail: "two windows", title: "Two windows: one to be sure, one to reset",
        body: `<p>A burn-rate alert still needs a time window to measure the error ratio over. Pick one and you get a trade-off:</p>
<ul><li>A <b>long window</b> (1 h) ignores blips, but keeps firing for a long time after you fix the problem, because the bad minutes stay inside the window.</li>
<li>A <b>short window</b> (5 min) reacts and resets fast, but a 2-minute blip can trip it.</li></ul>
<p>The fix is to require <b>both</b>: fire only when the 1 h <i>and</i> the 5 min error ratios exceed 14.4 × 1%. The long window proves it is not a blip. The short window makes the alert stop within minutes of the repair.</p>
<div class="eq">expr: ratio_rate1h  > (14.4 * 0.01)
  and ratio_rate5m > (14.4 * 0.01)</div>
<p>Trace it with a healthy history and a 60-minute incident where 20% of requests are slow. The 1 h ratio rises 0.2 ÷ 60 per minute, so it crosses 14.4% at minute 0.144 × 60 ÷ 0.2 = <b>43.2</b>: the page fires about 43 minutes in. After the fix the 5-minute ratio falls below 14.4% within two minutes (after 2 clean minutes it is 3/5 × 20% = 12%), so the alert resets even though the 1 h ratio is still high. The scene is computed with the same rules. (In exercise 2's synthetic series there is no healthy history, so Prometheus's <code>rate()</code> sees a short window and the alert fires earlier. Real systems have history.)</p>
<p>The slow-burn ticket uses the same trick with 6 h and 30 min windows at 6×.</p>`,
        check: { q: "Why does the page alert also require the 5-minute window to be above the threshold?",
          options: ["To make it fire sooner", "So it stops firing soon after the problem is fixed, instead of waiting for the 1 h window to drain", "Because Prometheus cannot evaluate a 1 h window alone"], answer: 1,
          why: "After a fix, the 1 h ratio stays above the threshold for tens of minutes because the bad minutes are still inside it. The 5-minute ratio drops below the threshold within a few minutes, and since both must be true, the alert resets quickly." },
        scene(G) {
          const s = burnSim(0.20, 60, 0.01), T = 150;
          const x0 = 50, x1 = 610, yb = 250, yt = 60, mx = 0.22;
          const X = (m) => x0 + ((x1 - x0) * m) / T, Y = (v) => yb - ((yb - yt) * Math.min(v, mx)) / mx;
          G.axes(x0, yt, x1 - x0, yb - yt, { xlabel: "minutes →" });
          G.label(x0, yt - 12, "error ratio (share of requests slower than 2 s)", { size: 11 });
          [0, 5, 10, 15, 20].forEach((v) => G.label(x0 - 6, Y(v / 100) + 4, v + "%", { anchor: "end", size: 10 }));
          [0, 30, 60, 90, 120, 150].forEach((m) => G.label(X(m), yb + 16, String(m), { anchor: "middle", size: 10 }));
          G.line(x0, Y(s.th1), x1, Y(s.th1), { color: "hot", dash: "5 4" }); G.label(x1 - 2, Y(s.th1) - 6, "14.4% threshold", { color: "hot", size: 11, anchor: "end" });
          G.rect(X(30), Y(0.2), X(90) - X(30), yb - Y(0.2), { fill: "hot", opacity: 0.12, rx: 0 });
          const path = (a, color, w) => G.path("M " + a.slice(0, T).map((v, m) => `${X(m)} ${Y(v)}`).join(" L "), { color, w });
          const l1 = path(s.r1h, "v", 2.5), l5 = path(s.r5, "k", 2);
          const [f, l] = s.pageSpan;
          G.rect(X(f), 276, X(l) - X(f), 18, { fill: "hot", rx: 3 });
          G.label(X(f), 310, `page fires at minute ${f}`, { size: 11, color: "hot" }); G.label(X(l), 326, `resets at minute ${l}`, { size: 11, color: "ok", anchor: "start" });
          G.rect(36, 346, 10, 10, { fill: "k", rx: 2 }); G.label(50, 355, "5 min ratio", { size: 11 });
          G.rect(150, 346, 10, 10, { fill: "v", rx: 2 }); G.label(164, 355, "1 h ratio", { size: 11 });
          G.label(36, 392, "incident: minutes 30 to 90, 20% slow. Both must exceed the line to page.", { color: "ink" });
          G.from([l1, l5], { opacity: 0, stagger: 0.4, duration: 0.6 });
          G.caption("the 1 h line proves it; the 5 min line lets it stop");
        } },

      /* 7 */
      { rail: "what to scrape", title: "The few metrics that matter, and one that lies",
        body: `<p>Do not graph everything. Each layer has a handful of numbers worth keeping. Names below are from the pinned vLLM docs and dcgm-exporter 4.8.4, plus the gateway's own metrics in <code>platform/gateway/app.py</code>.</p>
<ul><li><b>Engine (vLLM), what users feel:</b> <code>vllm:time_to_first_token_seconds</code>, <code>vllm:inter_token_latency_seconds</code>, <code>vllm:request_queue_time_seconds</code>.</li>
<li><b>Engine, load:</b> <code>vllm:num_requests_running</code> and <code>vllm:num_requests_waiting</code>. The queue is the autoscaling signal (P3.6).</li>
<li><b>Engine, KV pressure (P2.3):</b> <code>vllm:kv_cache_usage_perc</code>, <code>vllm:num_preemptions</code>.</li>
<li><b>GPU (DCGM), memory, power, hardware faults:</b> <code>DCGM_FI_DEV_FB_USED</code>, <code>DCGM_FI_DEV_POWER_USAGE</code>, <code>DCGM_FI_DEV_XID_ERRORS</code>.</li>
<li><b>Gateway (#6), per-tenant usage, errors, the cost input:</b> <code>gateway_requests_total</code>, <code>gateway_tokens_total</code>, <code>gateway_ttft_seconds</code>, <code>gateway_retries_total</code>.</li></ul>
<p><b>The metric that lies:</b> <code>DCGM_FI_DEV_GPU_UTIL</code> is the fraction of time <i>any</i> kernel was running. Decode keeps a kernel running almost constantly even at batch 1, so it reads near 100% when the tensor cores are mostly idle (P3.6 step 2). Judge efficiency with tokens per second against the capacity you measured in #4.</p>
<p>The repo also alerts on two causes rather than symptoms: <code>KVCacheSaturated</code> (usage above 0.95 for 10 min) and <code>GPUXidErrors</code> (any XID hardware error in 10 min).</p>`,
        scene(G) {
          const L = [["gateway", "gateway_requests_total · gateway_tokens_total", "gateway_ttft_seconds · retries", "v"], ["engine · vLLM", "TTFT · ITL · queue time (histograms)", "running · waiting · kv_cache_usage_perc", "k"], ["GPU · DCGM", "FB_USED · POWER_USAGE · XID_ERRORS", "GPU_UTIL: a kernel ran, not efficiency", "q"]];
          const g = L.map(([n, a, b, c], i) => {
            const y = 50 + i * 104, gg = G.group();
            G.box(24, y, 110, 88, n, { fill: c, size: 13, parent: gg });
            G.rect(144, y, 472, 88, { stroke: "line", rx: 8, parent: gg });
            G.text(158, y + 34, a, { size: 12, color: "ink", parent: gg });
            G.text(158, y + 62, b, { size: 12, color: i === 2 ? "hot" : "muted", parent: gg });
            return gg;
          });
          G.label(24, 380, "alert on causes too: kv_cache_usage > 0.95 for 10 min · any XID error", { color: "ink" });
          G.label(24, 402, "utilisation is not efficiency: use tokens/s vs your #4 knee", {});
          G.from(g, { opacity: 0, x: -20, stagger: 0.2, duration: 0.4 });
          G.caption("three layers, a handful of numbers each");
        } },

      /* 8 */
      { rail: "traces", title: "A trace is one request's timeline",
        body: `<p>Metrics say p90 TTFT is bad. They cannot say <i>why this request</i> took 2 s, because they have thrown the individual request away. A <b>trace</b> keeps it.</p>
<p>A trace is a tree of <b>spans</b>. A span is one timed piece of work with a name, a start, a duration and key-value <b>attributes</b>. All spans of one request share a <b>trace id</b>, and each span records its <b>parent</b>. <code>platform/gateway/tracing.py</code> creates a <code>gateway.request</code> span per request and a <code>gateway.backend_attempt</code> child per try.</p>
<p>The trace crosses processes through an HTTP header. The gateway adds W3C <code>traceparent</code> to its call to the backend, so that any span the backend creates joins the same trace:</p>
<div class="eq">traceparent: 00-&lt;trace id&gt;-&lt;span id&gt;-01</div>
<p>The four fields are a version (<code>00</code>), a 32-hex-digit trace id, the 16-hex-digit id of the calling span (the new span's parent), and flags (<code>01</code> = sampled).</p>
<p>The payoff is a retried request. In logs it is two unrelated lines. In a trace it is one tree: the request, a failed attempt (503 after 80 ms), then a second attempt that succeeded with a TTFT of 1.9 s. Now you see that the slowness was the backend's prefill, not the retry.</p>
<p>Turn it on with <code>S2S_TRACING=console</code> on a laptop (spans print to the terminal), <code>memory</code> in tests, or <code>otlp</code> with an OpenTelemetry Collector and Tempo or Jaeger in the cluster. It is <code>off</code> by default, with no overhead. Exercise 3 extends it to streaming responses.</p>`,
        scene(G) {
          const x0 = 150, W = 460, total = 2000, X = (ms) => x0 + (W * ms) / total;
          G.text(24, 34, "one trace (illustrative timings)", { size: 13 });
          G.axes(x0, 50, W, 190, {});
          [0, 500, 1000, 1500, 2000].forEach((ms) => G.label(X(ms), 258, ms + " ms", { anchor: "middle", size: 10 }));
          const sp = [
            ["gateway.request", 0, 2000, "k", 0, "trace id 7c1e…"],
            ["route + auth", 0, 12, "w", 1, ""],
            ["backend_attempt 1", 12, 92, "hot", 1, "503 · 80 ms"],
            ["backend_attempt 2", 100, 2000, "ok", 1, "200 · TTFT 1.9 s"],
          ];
          const bars = sp.map(([n, a, b, c, d, note], i) => {
            const y = 66 + i * 44;
            G.label(24 + d * 12, y + 17, n, { size: 11, color: "ink" });
            const r = G.rect(X(a), y, Math.max(3, X(b) - X(a)), 26, { fill: c, rx: 4 });
            if (note) { const wide = X(b) - X(a) > 120; G.text(wide ? X(a) + 8 : X(b) + 8, y + 18, note, { size: 11, color: wide ? "bg" : "ink" }); }
            return r;
          });
          G.rect(24, 290, 592, 112, { stroke: "line", rx: 10 });
          G.text(40, 314, "traceparent: 00-<trace id>-<span id>-01", { size: 12 });
          G.label(40, 338, "sent to the backend so its spans join the same trace", {});
          G.text(40, 366, "metrics: THAT it was slow · trace: WHERE the time went", { size: 12, color: "ok" });
          G.label(40, 390, "attributes: s2s.backend, status, tokens", {});
          G.from(bars, { attr: { width: 0 }, stagger: 0.25, duration: 0.4 });
          G.caption("a retry is one tree, not two unrelated log lines");
        } },

      /* 9 */
      { rail: "cardinality", title: "What goes on a metric label, and what never does",
        body: `<p>A <b>label</b> splits one metric into many <b>time series</b>: <code>gateway_requests_total{tenant="a", model="llama-3-8b", backend="g6-1", outcome="ok"}</code> is one series. Prometheus stores and indexes every distinct combination, in memory. The number of series is the <b>product</b> of the number of values of each label: the metric's <b>cardinality</b>.</p>
<div class="eq">20 tenants × 3 models × 4 backends × 4 outcomes
   = 960 series      (fine)
× 1,000,000 distinct request ids
   = 960,000,000 series  (the server falls over)</div>
<p>Put a request id, a prompt, or a user id on a metric and every request creates a new series that never receives a second sample. The rule: <b>labels hold a small, bounded set of values</b> (tenant, model, outcome). Anything per-request belongs in a <b>trace attribute</b> or a <b>log field</b>, where one-off values are expected.</p>
<p>The other common mistakes from this module, each with the reason: averaging percentiles (step 3); a <code>rate()</code> window shorter than four scrapes (step 2); reading GPU utilisation as efficiency (step 7); and charging tenants for idle capacity without saying so (next step).</p>`,
        check: { q: "You want to find the slowest individual requests for one tenant. Where should the request id be recorded?",
          options: ["As a label on gateway_ttft_seconds", "As a trace attribute or log field", "In the metric name"], answer: 1,
          why: "A request id is unbounded: as a label it creates one new time series per request, exploding memory and query cost. Traces and logs are built for per-request values. Use the tenant label for the aggregate and the trace for the individual request." },
        scene(G) {
          G.text(24, 34, "series = product of label value counts", { size: 13 });
          const f = [["tenant", 20, "k"], ["model", 3, "v"], ["backend", 4, "q"], ["outcome", 4, "ok"]];
          let x = 24;
          const bx = f.map(([n, v, c], i) => { const g = G.box(x, 66, 112, 62, "", { stroke: c }); G.text(x + 56, 92, n, { anchor: "middle", size: 12, color: c }); G.text(x + 56, 116, v + " values", { anchor: "middle", size: 12 }); if (i < 3) G.text(x + 122, 104, "×", { size: 18, anchor: "middle", color: "muted" }); x += 144; return g; });
          G.box(24, 168, 270, 56, "= 960 series", { fill: "ok", size: 16 });
          G.label(24, 244, "small, bounded, cheap to store and query", { color: "ok" });
          const hot = G.box(24, 276, 592, 70, "× 1,000,000 request ids = 960,000,000 series", { fill: "hot", size: 17 });
          G.label(24, 372, "per-request values go in traces and logs, never in labels", { color: "ink" });
          G.label(24, 396, "metrics: bounded labels · traces/logs: per-request detail", {});
          G.from(bx, { opacity: 0, y: -12, stagger: 0.15, duration: 0.3 }); G.from(hot, { opacity: 0, scale: 0.9, transformOrigin: "50% 50%", delay: 0.8, duration: 0.5 });
          G.caption("cardinality is the price of every label");
        } },

      /* 10 */
      { rail: "cost per token", title: "From a GPU price to dollars per million tokens",
        body: `<p>Project #2 answers "what does a token cost us?" with one identity:</p>
<div class="eq">$ per 1M tokens = (GPU $ per hour) ÷ (tokens per hour) × 1,000,000</div>
<p>Everything else is bookkeeping, and it lives in <code>platform/cost/model.py</code> as pure functions. Every price is an <b>input</b>, never hard-coded. The rules: a node's price per hour is split across its GPUs; a replica costs its GPU share (TP=2 is two GPUs, a 1/7 MIG slice is 1/7 of a GPU); tenants are charged by their token share on each replica; spot nodes use the price you actually pay times (1 + interruption overhead).</p>
<p>A worked example with <b>round example numbers, not a price quote and not a measurement</b>: a node at $1.00 per hour whose replica sustains 1,000 output tokens/s (your #4 knee).</p>
<div class="eq">capacity = 1,000 × 3,600 = 3,600,000 tokens/h
100% busy: 1.00 ÷ 3.6M × 1M = $0.278 per 1M
 30% busy: 0.278 ÷ 0.30      = $0.926 per 1M  (3.3×)
spot at 40% of price, 5% replayed:
   0.40 × 1.05 = 0.42 of the on-demand price</div>
<p><b>Waste</b> is reported <i>separately</i>: waste $/h = (1 − served ÷ capacity) × replica $/h. At 30% utilisation, 70% of the $1.00 hour is idle: $0.70. Smearing it into each tenant's rate hides the real lever (utilisation: autoscaling in P3.6, packing, MIG in P4.4) and makes tenants' rates depend on each other's behaviour. <code>breakeven_utilization()</code> returns the utilisation above which self-hosting beats a per-token API price. With an API price of, say, $0.50 per 1M (an example; look up a real one), breakeven is 0.278 ÷ 0.50 = 56%.</p>`,
        check: { q: "A replica is 30% utilised. Why does the cost model report the idle 70% as separate \"waste\" instead of adding it to the tenants' $/1M?",
          options: ["It is easier to compute", "It keeps tenant rates fair and shows that utilisation is the lever to pull", "Idle GPUs are free"], answer: 1,
          why: "If idle cost were smeared into tenant rates, a tenant's price would rise whenever other tenants left, and nobody would see that the real fix is higher utilisation (autoscaling, packing, MIG). Separate waste keeps both the rates and the lever visible." },
        scene(G) {
          G.text(24, 34, "$ per 1M output tokens vs utilisation (example numbers)", { size: 13 });
          G.label(24, 54, "node $1.00/h, capacity 1,000 tokens/s: round example numbers", {});
          const us = [1, 0.6, 0.3], vals = us.map((u) => 0.2778 / u);
          const bars = G.bars(60, 300, vals, { w: 90, gap: 50, h: 190, max: 1, fill: (i) => (i === 0 ? "ok" : i === 1 ? "v" : "hot") });
          us.forEach((u, i) => { G.text(105 + i * 140, 322, Math.round(u * 100) + "% busy", { anchor: "middle", size: 12 }); G.text(105 + i * 140, 300 - 190 * vals[i] - 8, "$" + vals[i].toFixed(3), { anchor: "middle", size: 13 }); });
          G.line(40, 300, 480, 300, { color: "line" });
          G.rect(490, 100, 126, 160, { stroke: "line", rx: 8 });
          G.text(502, 124, "at 30% busy", { size: 12, color: "ink" });
          G.rect(502, 136, 102, 20, { fill: "ok", rx: 3, opacity: 0.8 }); G.rect(502, 156, 102, 46, { fill: "hot", rx: 3, opacity: 0.7 });
          G.text(553, 150, "used $0.30", { anchor: "middle", size: 10, color: "bg" }); G.text(553, 184, "waste $0.70", { anchor: "middle", size: 11, color: "bg" });
          G.label(502, 224, "report waste", { size: 11 }); G.label(502, 240, "separately", { size: 11 });
          G.label(24, 360, "$0.278 ÷ 0.60 = $0.463 · $0.278 ÷ 0.30 = $0.926", { color: "ink" });
          G.label(24, 384, "breakeven vs an API at $0.50/1M: 0.278 ÷ 0.50 = 56% busy (example)", {});
          G.from(bars, { attr: { height: 0 }, stagger: 0.2, duration: 0.5 });
          G.caption("the same GPU costs 3.3× more per token when it is a third busy");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Metrics say <i>that</i>, traces say <i>where</i>, logs say <i>what</i>. Alert on metrics.</li>
<li>Counters need <code>rate()</code>; gauges are read as they are; histograms give percentiles only if you sum the buckets across pods first and then call <code>histogram_quantile</code>.</li>
<li>An SLO of 99% leaves a 1% error budget (43,200 requests in 30 days at 100 req/min). Burn rate = error ratio ÷ budget, and 14.4× empties it in about 2 days.</li>
<li>Page only when a long and a short window both exceed the threshold: the long one proves it, the short one lets the alert reset after the fix.</li>
<li>Traces connect gateway and backend through <code>traceparent</code>. Request ids belong in traces and logs, never in labels (cardinality is the product of label values).</li>
<li>$/1M tokens = GPU $/h ÷ tokens/h × 1M. Utilisation moves it 3×, and idle cost is reported as waste, not smeared across tenants.</li></ul>`,
    sim: {
      title: "When does the page fire, and when does it stop?",
      intro: "A healthy service has an incident at minute 30: for the chosen length, a fraction of requests miss the TTFT threshold. The simulator applies the rules from <code>slo.yaml</code>: page when the 1 h and 5 min error ratios both exceed 14.4 × budget, ticket when the 6 h and 30 min ratios both exceed 6 × budget. Press Run to sweep the clock.",
      height: 300,
      controls: [
        { id: "p", label: "share of requests slow during the incident (%)", min: 0.5, max: 50, step: 0.5, value: 20, format: (v) => v.toFixed(1) },
        { id: "len", label: "incident length (minutes)", min: 5, max: 360, step: 5, value: 60 },
        { id: "slo", label: "SLO target", type: "select", value: 0.99, options: [[0.99, "99% (budget 1%)"], [0.995, "99.5% (budget 0.5%)"], [0.999, "99.9% (budget 0.1%)"]] },
      ],
      run: { label: "Run the clock", frames: 40, ms: 60 },
      draw(G, v, t) {
        const budget = +(1 - v.slo).toFixed(4), p = v.p / 100;
        const s = burnSim(p, v.len, budget);
        const x0 = 40, x1 = 630, yt = 30, yb = 230, now = Math.max(1, Math.round(HORIZON * t));
        const mx = Math.max(0.05, p * 1.15, s.th1 * 1.3);
        const X = (m) => x0 + ((x1 - x0) * m) / HORIZON, Y = (val) => yb - ((yb - yt) * Math.min(val, mx)) / mx;
        G.axes(x0, yt, x1 - x0, yb - yt, {});
        [0, 120, 240, 360, 480, 600].forEach((m) => G.label(X(m), yb + 14, m + " min", { anchor: m === 600 ? "end" : "middle", size: 10 }));
        [0, 0.5, 1].forEach((f) => G.label(x0 - 4, Y(mx * f) + 4, F.num(mx * f * 100, 1) + "%", { anchor: "end", size: 10 }));
        G.line(x0, Y(s.th1), x1, Y(s.th1), { color: "hot", dash: "5 4" }); G.label(x1 - 2, Y(s.th1) - 5, "14.4 × budget", { size: 10, color: "hot", anchor: "end" });
        G.line(x0, Y(s.th2), x1, Y(s.th2), { color: "v", dash: "5 4" }); G.label(x1 - 2, Y(s.th2) - 5, "6 × budget", { size: 10, color: "v", anchor: "end" });
        const pth = (a, color, w) => G.path("M " + a.slice(0, now).map((val, m) => `${X(m)} ${Y(val)}`).join(" L "), { color, w });
        pth(s.r6h, "q", 1.5); pth(s.r1h, "v", 2); pth(s.r30, "pink", 1.5); pth(s.r5, "k", 2.5);
        // alert bands
        const band = (a, y, color) => a.slice(0, now).forEach((on, m) => { if (on) G.rect(X(m), y, Math.max(1, X(1) - x0), 10, { fill: color, rx: 0 }); });
        G.label(x0 - 4, 263, "page", { size: 10, color: "hot", anchor: "end" }); band(s.page, 254, "hot");
        G.label(x0 - 4, 281, "ticket", { size: 10, color: "v", anchor: "end" }); band(s.tick, 272, "v");
        [["k", "5 min"], ["pink", "30 min"], ["v", "1 h"], ["q", "6 h"]].forEach(([c, l], i) => { G.rect(x0 + 80 + i * 70, 8, 10, 10, { fill: c, rx: 2 }); G.label(x0 + 94 + i * 70, 17, l, { size: 10 }); });
        const burn = p / budget, pg = s.pageSpan, tk = s.tickSpan;
        const used = (p * v.len) / (budget * 43200);
        return [
          { title: "The incident", rows: [["burn rate", F.num(burn, 1) + "×"], ["budget empty in", burn > 1 ? (30 / burn < 2 ? F.num(30 * 24 / burn, 1) + " h" : F.num(30 / burn, 1) + " days") : "never"], ["budget spent by it", F.num(used * 100, 1) + "%"]],
            chip: [burn < 1, burn < 1 ? "below budget pace: no alert is needed" : burn >= 14.4 ? "fast burn: pageable" : "slow burn: ticket-worthy"] },
          { title: "Page (1 h and 5 min)", rows: [["fires at", pg ? "minute " + pg[0] + " (" + (pg[0] - START) + " min in)" : "never"], ["resets at", pg ? "minute " + pg[1] + " (" + (pg[1] - START - v.len + 1) + " clean min)" : "-"]],
            chip: [!!pg || burn < 14.4, pg ? "a human is paged" : burn < 14.4 ? "pace is below 14.4×" : "incident ended before the 1 h window filled"] },
          { title: "Ticket (6 h and 30 min)", rows: [["fires at", tk ? "minute " + tk[0] + " (" + (tk[0] - START) + " min in)" : "never"], ["resets at", tk ? "minute " + tk[1] : "-"]],
            chip: [!!tk || burn < 6, tk ? "a ticket is opened" : burn < 6 ? "pace is below 6×" : "incident ended before the 6 h window filled"] },
          { title: "Model", html: `<p class="note">One error ratio per minute with equal request volume each minute, a healthy history before the incident, and exact window means: the same arithmetic as <code>rate()</code> ratios in <code>slo.yaml</code>, without scrape jitter. Short incidents can end before the long window crosses its threshold; that is by design: they cost little budget. Budget spent = slow share × minutes ÷ (budget × 43,200 minutes in 30 days).</p>` },
        ];
      },
    },
    practice: {
      intro: "Everything here is T0 (a laptop). Exercises 1, 2 and 5 need the <code>promtool</code> binary from a Prometheus release tarball; pytest skips them if it is missing. The live dashboards are T3: after P3.2's <code>make addons</code> on EKS, apply the PodMonitors and the PrometheusRule; start with <code>make up</code> and end with <code>make down</code> (cost: control plane, NAT and GPU node-hours; the GPU group scales to zero when idle; the budget alarm in <code>infra/aws/guardrails</code> is your backstop). Grafana and Prometheus stay cluster-internal: use <code>kubectl port-forward</code>.",
      items: [
        { title: "PromQL p90 TTFT and the SLI, tested", tier: "T0", goal: "Run the SLO rule tests with promtool, then add a test for the p90 recording rule: three buckets with 50% under 0.5 s and 95% under 2 s. Work out the answer by hand first (step 3: 1.83 s).",
          cmd: "promtool test rules platform/observability/rules/slo_test.yaml" },
        { title: "Burn-rate alerts: prove they fire and reset", tier: "T0", goal: "Add a test to slo_test.yaml: 40 minutes at 20% slow, then 60 minutes at 0%. Assert the fast-burn alert fires, then stops 5 to 10 minutes after recovery. Compute the budget those 40 minutes used (800 of 43,200).",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.5-observability/exercises/test_promtool.py" },
        { title: "Trace propagation across gateway and backend", tier: "T0", goal: "Check one trace id, a gateway.request parent with a failed (503) and a successful (200) gateway.backend_attempt child, and a traceparent header on the backend call. Then add streaming spans and un-skip test_stream_spans.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.5-observability/exercises/test_tracing.py" },
        { title: "The cost model", tier: "T0", goal: "Pass the allocation, TP, MIG, spot-overhead, waste and breakeven tests, then write platform/cost/prices.yaml with your region's real prices (date and source in a comment) and compute this lesson's numbers with them.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.5-observability/exercises/test_cost.py" },
        { title: "Cost-spike alert on a synthetic tenant flood", tier: "T0 · hard", goal: "Write cost_test.yaml: tenant a at 1,000 tokens/min for 25 hours, then 10,000 for 30 minutes; tenant b steady. TenantTokenSpike must fire for a only, after its 15 minute hold. Explain why the rule compares with offset 1h of a 1-day rate.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.5-observability/exercises" },
      ],
      labs: [
        { label: "Run console tracing on a laptop: <code>S2S_TRACING=console GATEWAY_CONFIG=platform/gateway/example.yaml uv run uvicorn gateway.app:app --app-dir platform --port 9000</code>", path: "platform/gateway/tracing.py" },
        { label: "SLO rules, promtool tests, dashboards, scrape configs", path: "platform/observability/" },
        { label: "Cost model and exporter (#2)", path: "platform/cost/" },
        { label: "Grafana dashboard: TTFT, ITL, running/waiting, KV, burn rate, $/1M", path: "platform/observability/dashboards/llm-serving.json" },
        { label: "EKS cluster guide (cost, teardown)", path: "infra/aws/eks/README.md" },
      ],
    },
  });
})();
