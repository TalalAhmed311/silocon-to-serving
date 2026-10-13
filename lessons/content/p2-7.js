/* P2.7 — Gateway basics. What sits in front of the engines: authentication, rate limits, token budgets, routing,
   retries that do not make outages worse, the no-retry-after-first-byte rule, circuit breakers and accounting.
   Everything here is #6, platform/gateway, tested in virtual time against mockllm. */
(function () {
  const F = S2S.fmt;

  /* ---------- simulator: the real TokenBucket logic (platform/gateway/ratelimit.py) in virtual time ---------- */
  function bucketRun(rate, burst, offered, group, dur) {
    const gap = group / offered, ev = [];          // groups of `group` requests arrive together every `gap` seconds
    for (let t = 0; t < dur - 1e-9; t += gap) for (let i = 0; i < group; i++) ev.push(t);
    let tokens = burst, last = 0, ok = 0, bad = 0;
    const reqs = [], curve = [[0, burst]];
    const refill = (now) => { tokens = Math.min(burst, tokens + (now - last) * rate); last = now; };
    ev.forEach((t) => {
      refill(t); curve.push([t, tokens]);
      const allowed = tokens >= 1; if (allowed) { tokens -= 1; ok++; } else bad++;
      reqs.push([t, allowed]); curve.push([t, tokens]);
    });
    refill(dur); curve.push([dur, tokens]);
    return { reqs, curve, ok, bad };
  }

  S2S.lesson({
    id: "p2-7", n: "P2.7", title: "Gateway basics",
    subtitle: "Serving engines · first principles · T0, runs against mock backends",
    kicker: "Lesson · ≈ 40 min",
    headline: "The service in front of the engines",
    intro: `<p>An engine like vLLM turns prompts into tokens. A <b>gateway</b> sits in front of one or many engines and handles everything else a real service needs: who is calling, how fast they may call, how many tokens they may spend, which replica should take the request, what to do when a replica fails, and how to count what happened. This lesson builds each of those from the problem it solves, with worked numbers, and ties each to <code>platform/gateway</code> (project #6), where every rule is code you can test in virtual time.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "5 exercises"],
    legend: [["k", "request path"], ["ok", "allowed / success"], ["hot", "rejected / failed"], ["v", "retry / probe"], ["w", "state"]],
    prev: "p2-6", next: "p3-0",
    steps: [
      { rail: "request path", title: "What a gateway owns: one request, seven checks",
        body: `<p>Why not let clients talk to the engines directly? Because three things are always true in production:</p>
<ul><li><b>Many backends.</b> Several models, several replicas of each, maybe an external API as a spare.</li><li><b>Many tenants.</b> Teams or customers, each with a quota and a bill.</li><li><b>Failures are normal.</b> Replicas restart, scale to zero, run out of memory.</li></ul>
<p>The gateway is the one place that can see all of that. Every request walks the same path, and any step can stop it:</p>
<div class="eq">1 authenticate   Bearer key → tenant       else 401
2 rate limit     token bucket               else 429
3 token budget   estimate vs window         else 429
4 route          candidate backends         else 404
5 try backend    retry / fall back on error else 502
6 stream through bytes to the client
7 bill + metrics actual usage, counters</div>
<p>Steps 1–3 protect the platform and the bill <i>before</i> any GPU time is spent. Steps 4–5 hide failures from the client. Steps 6–7 produce the platform's numbers: per-tenant tokens (the cost dashboard), end-to-end latency, error rates. The gateway speaks the OpenAI-compatible API on both sides (P1.3), so a client cannot tell it from an engine.</p>
<p>The code is <code>platform/gateway/app.py</code> (<code>Gateway._forward</code> is this list, in order). The rest of the lesson goes through it piece by piece.</p>`,
        scene(G) {
          G.text(24, 32, "one request through the gateway (platform/gateway/app.py)", { size: 13 });
          const names = ["client", "auth", "rate limit", "budget", "route", "engine"], codes = [null, "401", "429", "429", "404", null];
          const bx = names.map((n, i) => G.box(14 + i * 104, 70, 90, 44, n, i === 0 ? { stroke: "ink", size: 13 } : i === 5 ? { fill: "k", size: 13 } : { stroke: "ok", color: "ok", size: 13 }));
          for (let i = 0; i < 5; i++) G.arrow(104 + i * 104 + 1, 92, 118 + i * 104 - 1, 92, { color: "muted" });
          codes.forEach((c, i) => { if (c) { G.arrow(59 + i * 104, 116, 59 + i * 104, 168, { color: "hot", w: 1.5 }); G.box(34 + i * 104, 170, 50, 28, c, { stroke: "hot", color: "hot", size: 13 }); } });
          G.label(14, 220, "each check can stop the request before any GPU time is used", { size: 12, color: "hot" });
          G.text(24, 270, "after the engine:", { size: 13 });
          const post = [["retry / fall back", "v"], ["stream bytes through", "k"], ["bill usage + metrics", "ok"]].map(([t, c], i) => G.box(24 + i * 200, 286, 180, 40, t, { stroke: c, color: c, size: 13 }));
          for (let i = 0; i < 2; i++) G.arrow(206 + i * 200, 306, 222 + i * 200, 306, { color: "muted" });
          G.label(24, 360, "tenants, keys, quotas, backends and retry settings come from one YAML file", { size: 12 });
          G.label(24, 380, "platform/gateway/example.yaml", { size: 12 });
          G.from([].concat(bx, post), { opacity: 0, stagger: 0.06, duration: 0.25 });
          G.caption("cheap checks first, GPU work last");
        } },

      { rail: "authentication", title: "Who is calling? Keys stored as hashes",
        body: `<p>Each tenant gets an API key. The gateway must map the key in the <code>Authorization: Bearer …</code> header to a tenant, and the key list must be safe to store. If the config held raw keys, anyone who read the config, a backup or a log line could impersonate every tenant.</p>
<p>So the config stores only the <b>SHA-256 hash</b> of each key. A hash is a one-way fingerprint: easy to compute from the key, infeasible to reverse. At request time the gateway hashes the key it received and looks for the same hash:</p>
<div class="eq">sha256("dev-key-1") = 1bcefe2243eced99cd50…
                      (64 hex characters in all)
config stores: key_sha256: "1bcefe22…"
               (never the key itself)</div>
<p>One more detail: comparing two strings with <code>==</code> usually stops at the <i>first differing byte</i>, so a wrong guess that matches more leading characters takes slightly longer. Measuring that difference can leak the secret one byte at a time (a <b>timing attack</b>). The gateway therefore uses <code>hmac.compare_digest</code>, which takes the same time wherever the strings differ. Comparing hashes already makes the attack impractical; constant-time compare is the standard second layer.</p>
<p>In <code>app.py</code> this is <code>Gateway.tenant_for</code>; backend keys are never written in the YAML either, only the <i>name</i> of an environment variable (<code>api_key_env: S2S_API_KEY</code>). A missing or unknown key is a <code>401</code>.</p>`,
        scene(G) {
          G.text(24, 32, "the key never appears in the config", { size: 13 });
          G.box(24, 60, 150, 40, "dev-key-1", { stroke: "ink", size: 13 });
          G.label(24, 118, "what the client sends", { size: 11 });
          G.arrow(176, 80, 226, 80, { color: "k" });
          G.box(228, 60, 130, 40, "sha256", { fill: "k", size: 14 });
          G.arrow(360, 80, 410, 80, { color: "k" });
          G.box(412, 60, 200, 40, "1bcefe22 43eced99 …", { stroke: "k", color: "k", size: 12 });
          G.text(24, 168, "stored tenant hashes", { size: 13 });
          const rows = [["team-a", "1bcefe22 43eced99 …", true], ["team-b", "a different hash …", false]];
          const rb = [];
          rows.forEach(([n, h, hit], i) => {
            const y = 184 + i * 46;
            rb.push(G.box(24, y, 100, 34, n, { stroke: hit ? "ok" : "line", color: hit ? "ok" : "muted", size: 13 }));
            rb.push(G.box(130, y, 220, 34, h, { stroke: hit ? "ok" : "line", color: hit ? "ok" : "muted", size: 12 }));
            G.text(366, y + 22, hit ? "match → tenant team-a" : "no match", { size: 13, color: hit ? "ok" : "muted" });
          });
          G.text(24, 306, "compare in constant time", { size: 13, color: "ink" });
          G.label(24, 328, "== stops at the first wrong byte: a longer match takes longer, which leaks", { size: 12, color: "hot" });
          G.label(24, 348, "hmac.compare_digest: same time wherever the strings differ", { size: 12, color: "ok" });
          G.label(24, 378, "no key (401) → unknown key (401) → tenant found: go to the rate limit", { size: 12 });
          G.from(rb, { opacity: 0, stagger: 0.1, duration: 0.25 });
          G.caption("a leaked config does not leak usable credentials");
        } },

      { rail: "token bucket", title: "Rate limits: the token bucket",
        body: `<p>A tenant with a loop and a bug can send thousands of requests a second. A <b>rate limit</b> protects the service. The standard tool is the <b>token bucket</b> (not to be confused with LLM tokens): a bucket holds up to <b>burst</b> tickets and refills at <b>rate</b> tickets per second. Each request takes one ticket; with none left, the request is refused.</p>
<p>It allows short bursts (up to the bucket size) while capping the long-run average at <i>rate</i>. It needs no timer: refill lazily when a request arrives.</p>
<div class="eq">tokens = min(burst, tokens + elapsed × rate)
allow if tokens ≥ 1, then tokens −= 1

rate 5/s, burst 10, 15 requests at t = 0:
  10 allowed (tokens 10 → 0), 5 refused
  Retry-After = (1 − 0) / 5 = 0.2 s
  at t = 1 s: tokens = min(10, 0 + 1×5) = 5
  at t = 2 s: min(10, 10) = 10, full again</div>
<p>A refused request gets HTTP <code>429</code> and a <code>Retry-After</code> header telling the client how long to wait (the gateway sends whole seconds, at least 1). The class is <code>TokenBucket</code> in <code>platform/gateway/ratelimit.py</code>; it takes a <code>clock</code> argument so the tests run in virtual time with no sleeps. That is exercise 1, and the simulator on the Simulate tab runs this exact logic.</p>`,
        check: { q: "A bucket has rate 5 per second and burst 10, and starts full. A client sends 15 requests at once, then waits 1 second and sends 15 more at once. How many requests are allowed in total?",
          options: ["10", "15", "20", "30"], answer: 1,
          why: "The first batch drains the 10 tokens (10 allowed, 5 refused). One second later the bucket has refilled 5 tokens, so 5 of the second batch pass. 10 + 5 = 15 allowed in total. The sustained rate is 5 per second; only the first burst is larger." },
        scene(G) {
          const x0 = 70, y0 = 300, w = 500, h = 190, tmax = 3, X = (t) => x0 + (w * t) / tmax, Y = (v) => y0 - (h * v) / 10;
          G.text(24, 32, "tokens in the bucket (rate 5/s, burst 10)", { size: 13 });
          G.axes(x0, y0 - h, w, h, {});
          [0, 5, 10].forEach((v) => G.label(x0 - 8, Y(v) + 4, String(v), { anchor: "end", size: 11 }));
          [0, 1, 2, 3].forEach((t) => G.label(X(t), y0 + 16, t + " s", { anchor: "middle", size: 11 }));
          G.line(x0, Y(10), x0 + w, Y(10), { color: "ink", dash: "3 4", w: 1 }); G.label(x0 + w - 4, Y(10) - 6, "burst = 10", { anchor: "end", size: 11 });
          const path = G.path(`M ${X(-0.0)} ${Y(10)} L ${X(0)} ${Y(0)} L ${X(2)} ${Y(10)} L ${X(3)} ${Y(10)}`, { color: "k", w: 3 });
          G.circle(X(0), Y(10), 5, { fill: "k" }); G.circle(X(0), Y(0), 5, { fill: "hot" });
          G.label(X(0) + 10, Y(8.6), "15 requests arrive at t = 0", { size: 12, color: "ink" });
          G.label(X(0.55), Y(0.7), "10 allowed, 5 refused (429)", { size: 12, color: "hot" });
          G.circle(X(1), Y(5), 4, { fill: "ok" }); G.label(X(1) + 8, Y(5) + 20, "1 s: 5 tokens", { size: 12, color: "ok" });
          G.label(X(2) + 10, Y(10) + 18, "2 s: full again", { size: 12, color: "ok" });
          G.text(24, 352, "refill = elapsed × rate, capped at burst; no timer needed", { size: 13 });
          G.label(24, 376, "refused: 429 + Retry-After = (1 − tokens) / rate seconds", { size: 12 });
          G.from(path, { opacity: 0, duration: 0.7 });
          G.caption("bursts up to the bucket, then the long-run rate");
        } },

      { rail: "token budget", title: "Token budgets: protect the bill, not the server",
        body: `<p>A request rate says nothing about cost. One request can ask for 50 output tokens or 4,000. A tenant who pays for, or is allowed, N LLM tokens a day needs a different limit: a <b>token budget</b>, counted in prompt plus completion tokens over a window (for example 2,000,000 per 86,400 s).</p>
<p>Problem: before the request runs, you do not know its cost. The completion length is unknown until it finishes. So the gateway uses two numbers:</p>
<ul><li>an <b>estimate</b> to decide whether to admit the request: prompt characters ÷ 4 (a rough tokens-per-character rule) plus <code>max_tokens</code>, the most it may generate;</li><li>the <b>actual usage</b> reported by the engine, committed once the request ends.</li></ul>
<div class="eq">prompt 2,000 chars, max_tokens 256
  estimate = 2000 // 4 + 256 = 756   (check only)
engine reports prompt 500, completion 90
  commit   = 590                     (what is billed)</div>
<p>The window <b>slides</b>. Each commit is stored with its time and expires exactly one window later, instead of everything resetting at midnight (which would let a tenant spend two days' budget around the reset). Limit 1,000, window 60 s:</p>
<div class="eq">t=0: commit 400   t=20: commit 300   t=40: commit 200
t=40: used 900, remaining 100
t=60: the t=0 event expires → used 500, remaining 500
t=80: the t=20 event expires → used 200, remaining 800</div>
<p>The class is <code>TokenBudget</code> (<code>check</code>, <code>commit</code>, <code>remaining</code>) in <code>ratelimit.py</code>; exercise 4 writes it.</p>`,
        check: { q: "A tenant has 100 tokens left in its budget. A request arrives with an estimate of 756 tokens, though it would really use only about 90. What does the gateway do, and why?",
          options: ["Admits it, since actual usage will fit", "Refuses it with 429: admission is decided on the estimate, because the real length is unknown until the request ends", "Admits it and bills the estimate"], answer: 1,
          why: "check(estimate) must pass before the request starts, and 100 < 756. The estimate is deliberately conservative (it assumes the full max_tokens). Billing then uses the real usage, never the estimate." },
        scene(G) {
          const x0 = 50, w = 540, tmax = 90, X = (t) => x0 + (w * t) / tmax, y0 = 190, u = 0.14;
          G.text(24, 32, "sliding window: limit 1,000 tokens, window 60 s", { size: 13 });
          G.line(x0, y0, x0 + w, y0, { color: "line" });
          [0, 20, 40, 60, 80].forEach((t) => G.label(X(t), y0 + 16, t + " s", { anchor: "middle", size: 11 }));
          const ev = [[0, 400, "q"], [20, 300, "v"], [40, 200, "k"]];
          const bars = ev.map(([t, n, c]) => { G.label(X(t) + 18, y0 - n * u - 8, String(n), { anchor: "middle", size: 12, color: c }); return G.rect(X(t), y0 - n * u, 36, n * u, { fill: c, rx: 3 }); });
          // window at now = 60 covers (0, 60]
          G.rect(X(0), 66, X(60) - X(0), 130, { stroke: "ok", rx: 6, dash: "5 4" });
          G.label(X(0) + 6, 82, "window at t = 60: covers (0, 60]", { size: 12, color: "ok" });
          G.line(X(60), 62, X(60), y0, { color: "ok", w: 2 });
          G.text(24, 252, "t = 40", { size: 13 }); G.label(110, 252, "used 900 → remaining 100", { size: 13, color: "ink" });
          G.text(24, 280, "t = 60", { size: 13 }); G.label(110, 280, "the t = 0 commit expires → used 500, remaining 500", { size: 13, color: "ok" });
          G.text(24, 308, "t = 80", { size: 13 }); G.label(110, 308, "the t = 20 commit expires → used 200, remaining 800", { size: 13, color: "ok" });
          G.text(24, 356, "check(estimate) before · commit(actual) after", { size: 14, color: "ink" });
          G.label(24, 380, "usage expires one event at a time, never all at midnight", { size: 12 });
          G.from(bars, { attr: { height: 0 }, stagger: 0.12, duration: 0.4 });
          G.caption("each commit ages out exactly one window after it was made");
        } },

      { rail: "routing", title: "Routing: why least outstanding requests",
        body: `<p>With several replicas serving one model, which gets the next request? <b>Round-robin</b> alternates A, B, A, B. That is fair only if requests cost the same. LLM requests do not: a 10-token answer and a 2,000-token answer differ by about 100× in time.</p>
<p>The cheap fix is <b>least outstanding requests</b> (LOR): send the request to the backend with the fewest requests <i>currently in flight</i>, divided by its weight. In-flight count is a free, always-fresh proxy for how busy a replica is: a replica stuck on long requests keeps a high count, a replica finishing short ones returns to zero.</p>
<p>Six requests arrive one time unit apart; durations 20, 1, 20, 1, 20, 1 (long, short, long, short, …):</p>
<div class="eq">round-robin: A gets R1 R3 R5 → 20+20+20 = 60 units
             B gets R2 R4 R6 →  1+ 1+ 1 =  3 units
LOR (ties go to A):
  R1→A (0 vs 0, tie)  R2→B (A busy)
  R3→B (R2 done: A 1, B 0)
  R4→A (1 vs 1, tie)  R5→A (1 vs 1, tie)
  R6→B (A 2, B 1)
  A: 20+1+20 = 41     B: 1+20+1 = 22</div>
<p>Round-robin piles all three long requests on A. LOR, because it sees A still busy, splits them 2 and 1. Mean latency drops because less work waits behind a long request. It is a router for <i>load</i>; routing by <i>prefix</i> or KV state (so a request lands where its cached prefix lives) is the next level, in P3.4. In code: <code>Router.candidates</code> in <code>router.py</code> returns every backend for the model, best first, so the gateway can fall back down the list.</p>`,
        check: { q: "Why is least-outstanding-requests a better default than round-robin for LLM backends?",
          options: ["It uses fewer CPU cycles", "Request durations vary by orders of magnitude, and in-flight count tracks real load while round-robin assumes equal cost", "It guarantees every backend gets the same number of requests"], answer: 1,
          why: "Round-robin balances the count of requests sent, not the work. A backend holding long requests stays 'in flight' longer, so LOR steers new requests to the one that is actually free." },
        scene(G) {
          const sc = 8, x0 = 90;
          const lane = (y, name, segs, col) => {
            G.text(24, y + 24, name, { size: 13, color: col });
            let x = x0; const out = [];
            segs.forEach(([lab, d, c]) => { out.push(G.box(x, y, d * sc - 3, 38, d > 5 ? lab : "", { fill: c, size: 12 })); x += d * sc; });
            G.label(x + 6, y + 24, String(segs.reduce((s, a) => s + a[1], 0)) + " units", { size: 12, color: "ink" });
            return out;
          };
          const L = "v", S = "k";
          G.text(24, 32, "work assigned to each backend (1 unit = 8 px)", { size: 13 });
          G.label(24, 56, "round-robin", { size: 12, color: "hot" });
          const a = lane(66, "A", [["R1", 20, L], ["R3", 20, L], ["R5", 20, L]], "ink").concat(lane(116, "B", [["R2", 1, S], ["R4", 1, S], ["R6", 1, S]], "ink"));
          G.label(24, 190, "least outstanding requests", { size: 12, color: "ok" });
          const b = lane(200, "A", [["R1", 20, L], ["R4", 1, S], ["R5", 20, L]], "ink").concat(lane(250, "B", [["R2", 1, S], ["R3", 20, L], ["R6", 1, S]], "ink"));
          G.rect(x0 + 41 * sc, 196, 2, 90, { fill: "ok", rx: 0, opacity: 0.0 });
          G.text(24, 330, "busiest backend: 60 units vs 41 units", { size: 15, color: "ok" });
          G.label(24, 354, "amber = long (R1 R3 R5, 20 units), thin cyan = short (R2 R4 R6)", { size: 12 });
          G.label(24, 376, "LOR sees A still busy and sends R3 to B", { size: 12 });
          G.from(a.concat(b), { opacity: 0, stagger: 0.07, duration: 0.25 });
          G.caption("in-flight count steers long requests apart");
        } },

      { rail: "retries", title: "Retries: only the retryable, with backoff and jitter",
        body: `<p>A backend call can fail. Some failures are worth retrying, because a second try, usually on another replica, may succeed: <code>408</code> (timeout), <code>429</code>, <code>500</code>, <code>502</code>, <code>503</code>, <code>504</code>, and transport errors (connection refused or reset). Others never will: a <code>400</code> (bad request) stays bad on every replica. Retrying it only wastes capacity.</p>
<p>When to retry is half the problem; <b>when to send the retry</b> is the other half. Retrying immediately, or after a fixed delay, makes things worse: if 100 clients fail at the same moment, they all retry at the same moment, and the recovering backend gets hit by a synchronized wave again. The fix is <b>exponential backoff with full jitter</b>:</p>
<div class="eq">sleep = uniform(0, min(cap, base × 2^attempt))

base 0.1 s, cap 2 s:
attempt 0: 0 … 0.1 s    attempt 3: 0 … 0.8 s
attempt 1: 0 … 0.2 s    attempt 4: 0 … 1.6 s
attempt 2: 0 … 0.4 s    attempt 5: 0 … 2.0 s
                        (capped: 0.1 × 32 = 3.2)</div>
<p>The window doubles each attempt, so a struggling backend gets progressively more room. Choosing a <i>random</i> point in the window spreads the clients out. (The idea is from the AWS Architecture Blog's "Exponential Backoff and Jitter".) The code is <code>backoff_s</code> and <code>should_retry</code> in <code>platform/gateway/retry.py</code>; exercise 2 writes both.</p>`,
        scene(G) {
          const caps = [0.1, 0.2, 0.4, 0.8, 1.6, 2.0], draws = [0.07, 0.04, 0.31, 0.52, 1.12, 0.9];
          G.text(24, 32, "backoff window per attempt (base 0.1 s, cap 2 s)", { size: 13 });
          const x0 = 70, sc = 240, bars = [];
          caps.forEach((c, i) => {
            const y = 56 + i * 32;
            G.label(24, y + 17, "try " + i, { size: 11 });
            bars.push(G.rect(x0, y, c * sc, 22, { stroke: "k", rx: 3 }));
            bars.push(G.circle(x0 + draws[i] * sc, y + 11, 5, { fill: "v" }));
            G.label(x0 + c * sc + 8, y + 16, "0 … " + c + " s", { size: 11, color: "ink" });
          });
          G.label(x0 + 260, 82, "orange dot = one random draw", { size: 11, color: "v" });
          G.text(24, 276, "100 clients fail at once", { size: 13 });
          const hist = (x, label, vals, col) => {
            G.label(x, 296, label, { size: 11, color: col });
            return vals.map((v, i) => G.rect(x + i * 18, 372 - v * 0.7, 14, v * 0.7, { fill: col, rx: 1 }));
          };
          const a = hist(30, "fixed delay: all retry together", [0, 0, 100, 0, 0, 0, 0, 0], "hot");
          const b = hist(330, "full jitter: spread out", [14, 13, 12, 13, 12, 12, 13, 11], "ok");
          G.line(30, 372, 30 + 8 * 18, 372, { color: "line" }); G.line(330, 372, 330 + 8 * 18, 372, { color: "line" });
          G.label(30, 392, "time of retry →", { size: 11 }); G.label(330, 392, "time of retry →", { size: 11 });
          G.from(bars.concat(a, b), { opacity: 0, stagger: 0.03, duration: 0.2 });
          G.caption("random delays turn a synchronized wave into a trickle");
        } },

      { rail: "retry budget", title: "The retry budget: don't turn an outage into a storm",
        body: `<p>Retries have a dark side. Suppose a replica dies and clients retry, up to <code>max_attempts</code> = 3 times. Each request that keeps failing now costs <b>3 attempts</b> instead of 1, so the load on the system triples exactly when it can least afford it. Survivors overload, more requests fail, more retries follow: a <b>retry storm</b>.</p>
<p>The cure is a <b>retry budget</b>: retries are allowed only up to a fraction of all requests, <i>globally</i>, not per request. The gateway's rule (<code>RetryBudget</code>) is:</p>
<div class="eq">retries allowed = max(min_retries, ratio × requests)
ratio 0.2, min_retries 10:
   20 requests →  max(10, 4)   = 10
  200 requests →  max(10, 40)  = 40
 1000 requests →  max(10, 200) = 200

worst case during an outage:
  no budget:   3 attempts per request → 3.0× load
  with budget: ≤ 1 + 0.2 requests worth → ≤ 1.2× load</div>
<p>The floor <code>min_retries</code> keeps a few retries available at low traffic, where 20% of 3 requests would round to zero. Once the budget is spent, <code>should_retry</code> returns false and the client gets the error immediately: failing fast is better than amplifying load.</p>
<p>Two companion ideas: retries must go to a <i>different</i> backend where possible (the gateway's <code>_attempt_order</code> walks the candidate list), and a <b>circuit breaker</b> (step 9) stops sending to a backend that is clearly down. The Google SRE book chapters on handling overload and cascading failures describe the pattern.</p>`,
        check: { q: "All backends start failing. Each of 1,000 requests would retry twice (max_attempts = 3). With a retry budget of 20%, roughly how many retries does the gateway send?",
          options: ["2,000", "About 200, since retries are capped at 20% of requests", "0, since the budget blocks all retries"], answer: 1,
          why: "Without the budget the load is 3×: 1,000 first tries + 2,000 retries. The budget allows max(10, 0.2 × 1000) = 200 retries in total, so the load rises to at most 1.2× and the rest fail fast." },
        scene(G) {
          G.text(24, 32, "load multiplier when a backend keeps failing", { size: 13 });
          const x0 = 40, y0 = 190, u = 40;
          G.line(x0, y0, x0 + 260, y0, { color: "line" });
          const b1 = G.rect(x0 + 20, y0 - 3 * u, 80, 3 * u, { fill: "hot", rx: 3 });
          const b2 = G.rect(x0 + 140, y0 - 1.2 * u, 80, 1.2 * u, { fill: "ok", rx: 3 });
          G.text(x0 + 60, y0 - 3 * u - 8, "3.0×", { anchor: "middle", size: 14, color: "hot" });
          G.text(x0 + 180, y0 - 1.2 * u - 8, "1.2×", { anchor: "middle", size: 14, color: "ok" });
          G.label(x0 + 60, y0 + 18, "no budget", { anchor: "middle", size: 12 });
          G.label(x0 + 180, y0 + 18, "budget 20%", { anchor: "middle", size: 12 });
          G.line(x0, y0 - u, x0 + 260, y0 - u, { color: "ink", dash: "3 4", w: 1 }); G.label(x0 + 262, y0 - u + 4, "1×", { size: 11 });
          // allowed retries vs requests
          const cx = 360, cy = 200, cw = 250, ch = 110;
          G.text(cx - 10, 72, "retries allowed", { size: 13 });
          G.axes(cx, cy - ch, cw, ch, {});
          const X = (r) => cx + (cw * r) / 1000, Y = (v) => cy - (ch * v) / 200;
          let d = `M ${X(0)} ${Y(10)}`; for (let r = 0; r <= 1000; r += 10) d += ` L ${X(r)} ${Y(Math.max(10, 0.2 * r))}`;
          const p = G.path(d, { color: "ok", w: 3 });
          [0, 100, 200].forEach((v) => G.label(cx - 6, Y(v) + 4, String(v), { anchor: "end", size: 10 }));
          [0, 500, 1000].forEach((r) => G.label(X(r), cy + 14, String(r), { anchor: "middle", size: 10 }));
          G.label(cx + cw, cy + 30, "requests so far →", { anchor: "end", size: 10 });
          G.label(X(90), Y(10) + 18, "floor: 10", { size: 11, color: "ink" });
          G.circle(X(1000), Y(200), 5, { fill: "ok" }); G.label(X(1000) - 6, Y(200) - 8, "200 at 1,000", { anchor: "end", size: 11, color: "ok" });
          G.text(24, 276, "max(min_retries, ratio × requests)", { size: 14, color: "ink" });
          G.label(24, 302, "once spent, errors go straight back to the client: fail fast", { size: 12 });
          G.label(24, 326, "retries should also move to a different backend (fallback)", { size: 12 });
          G.label(24, 350, "platform/gateway/retry.py · RetryBudget, should_retry", { size: 12 });
          G.from([b1, b2], { attr: { height: 0 }, stagger: 0.15, duration: 0.4 }); G.from(p, { opacity: 0, duration: 0.6 });
          G.caption("a global cap keeps retries from multiplying an outage");
        } },

      { rail: "streaming", title: "Streaming: never retry after the first byte",
        body: `<p>For a normal HTTP call, a retry is invisible to the client: the first attempt failed, the second succeeded, the client sees one response. A <b>streamed</b> LLM response breaks that. Tokens reach the user as they are produced (SSE, P1.3). Suppose a backend sends "The cat" and then dies.</p>
<p>If the gateway retried on another backend, it would produce a <i>new</i> completion (sampling is random, and the second backend has no idea what the first said). Forwarding that would give the client "The cat" followed by an unrelated start: duplicated or spliced text, silently. Once <b>any</b> byte has reached the client, the response is <b>committed</b>.</p>
<p>The rule, then, has two cases:</p>
<ul><li><b>Failure before the first byte:</b> retry or fall back freely. The client never knew.</li><li><b>Failure after the first byte:</b> stop, and send an <b>explicit error event</b> in the stream so the client can decide what to do (show partial text, resend the request). Never splice.</li></ul>
<p>In <code>Gateway._stream</code> it is one line, <code>started = True</code>, set when the first line is forwarded. After that, the failure path yields <code>{"error": {"type": "upstream_interrupted"}}</code> and returns instead of calling <code>should_retry</code>. Exercise 5 runs a mock that cuts every stream after 3 tokens and checks the client gets exactly 3 tokens, then one error event, never a 4th token from a silent retry.</p>
<p>Bonus: <code>stream_options.include_usage</code> must be set so the final chunk reports usage; the gateway adds it by default so it can bill (next steps).</p>`,
        check: { q: "A backend streams 3 tokens and then the connection drops. What should the gateway do?",
          options: ["Retry on another backend and forward its tokens after the first 3", "Retry silently and discard the first 3 tokens", "Stop, and send the client an explicit error event: the response is already committed, and a retry would splice a different completion"], answer: 2,
          why: "The client has already seen output. A second run would sample different text, and forwarding it would duplicate or splice tokens. The gateway reports the interruption explicitly and lets the client choose; before the first byte, retrying is allowed." },
        scene(G) {
          const tok = (x, y, t, c) => G.box(x, y, 46, 28, t, { fill: c, size: 12 });
          G.text(24, 32, "two failures, two different answers", { size: 13 });
          G.text(24, 66, "A  fails before the first byte → retry is invisible", { size: 13, color: "ok" });
          const a1 = G.box(24, 82, 120, 30, "backend 1 ✕", { stroke: "hot", color: "hot", size: 12 });
          G.arrow(146, 97, 174, 97, { color: "v" });
          const a2 = G.box(176, 82, 120, 30, "backend 2 ✓", { stroke: "ok", color: "ok", size: 12 });
          G.arrow(298, 97, 326, 97, { color: "k" });
          const ca = ["The", "cat", "sat"].map((t, i) => tok(330 + i * 52, 82, t, "k"));
          G.label(24, 134, "client sees: one normal stream", { size: 12, color: "ok" });
          G.text(24, 200, "B  fails after the first byte → committed", { size: 13, color: "hot" });
          const cb = ["The", "cat"].map((t, i) => tok(24 + i * 52, 216, t, "k"));
          G.box(132, 216, 140, 28, "✕ connection lost", { stroke: "hot", color: "hot", size: 12 });
          G.arrow(274, 230, 300, 230, { color: "hot" });
          const err = G.box(304, 212, 310, 36, "error: upstream_interrupted", { fill: "hot", size: 12 });
          G.label(24, 268, "client sees: 2 tokens, then an explicit error event", { size: 12, color: "hot" });
          G.text(24, 322, "never: ", { size: 13, color: "muted" });
          G.box(82, 306, 46, 28, "The", { fill: "k", size: 12 }); G.box(134, 306, 46, 28, "cat", { fill: "k", size: 12 });
          G.box(186, 306, 56, 28, "A dog", { fill: "v", size: 12, dash: "3 3" });
          G.line(184, 340, 244, 300, { color: "hot", w: 3 });
          G.label(262, 326, "spliced output from a retry", { size: 12, color: "hot" });
          G.label(24, 376, "gateway: started = True on the first forwarded line → no retry path", { size: 12 });
          G.from([].concat(ca, cb, [a1, a2, err]), { opacity: 0, stagger: 0.07, duration: 0.25 });
          G.caption("committed responses are never silently retried");
        } },

      { rail: "circuit breaker", title: "Circuit breakers and fallbacks",
        body: `<p>Retries handle a bad <i>request</i>. If a whole backend is down, trying it first for every request wastes time (each attempt waits for a timeout) and spends retry budget. A <b>circuit breaker</b> remembers that a backend is failing and skips it, like a fuse.</p>
<p>It has three states:</p>
<ul><li><b>closed</b>: normal, requests flow. Count consecutive failures.</li><li><b>open</b>: after <i>threshold</i> consecutive failures, stop sending for <i>cooldown_s</i> seconds.</li><li><b>half-open</b>: after the cooldown, let <b>one probe</b> request through. If it succeeds, close the breaker. If it fails, open again and restart the cooldown.</li></ul>
<div class="eq">threshold 5, cooldown 30 s
t=0..4   five failures in a row   → open
t=5..34  requests skip this backend (fall back)
t=35     cooldown over → half-open, one probe
         probe ok → closed
         probe fails → open for 30 s more</div>
<p>While a breaker is open, the <b>fallback</b> does the real work: the gateway just takes the next backend in the router's candidate list. With breaker plus fallback, a dead replica costs a few failed probes instead of a timeout on every request.</p>
<p>The gateway already falls back to the next backend (<code>Gateway._unary</code> and <code>_stream</code>) and marks backends unhealthy with simple <code>healthy</code> flags in <code>router.py</code>. Exercise 3 adds a real <code>CircuitBreaker(threshold, cooldown_s, clock)</code> state machine, tested in virtual time, and checks that with one of two mock backends at <code>FAIL_RATE=1.0</code>, 30 requests all still succeed.</p>`,
        scene(G) {
          G.text(24, 32, "circuit breaker states (threshold 5, cooldown 30 s)", { size: 13 });
          const c = (x, y, t, col) => { G.circle(x, y, 44, { stroke: col, sw: 2.5 }); G.text(x, y + 5, t, { anchor: "middle", size: 14, color: col }); };
          c(100, 130, "closed", "ok"); c(320, 130, "open", "hot"); c(210, 290, "half-open", "v");
          G.arrow(146, 120, 274, 120, { color: "hot", w: 2 }); G.label(168, 108, "5 failures", { size: 11, color: "hot" });
          G.arrow(298, 172, 244, 254, { color: "v", w: 2 }); G.label(186, 206, "30 s passed", { size: 11, color: "v" });
          G.arrow(176, 258, 120, 176, { color: "ok", w: 2 }); G.label(24, 222, "probe ok", { size: 11, color: "ok" });
          G.arrow(256, 280, 340, 176, { color: "hot", w: 2, dash: "4 3" }); G.label(350, 252, "probe fails:", { size: 11, color: "hot" }); G.label(350, 268, "open again", { size: 11, color: "hot" });
          G.label(400, 120, "skip this backend,", { size: 11 }); G.label(400, 136, "fall back to the next", { size: 11 });
          G.text(24, 372, "one probe at a time: a dead backend costs a probe, not a timeout", { size: 13 });
          G.caption("fail, back off, probe once, close");
        } },

      { rail: "accounting", title: "Billing from usage, metrics, and the cost of the hop",
        body: `<p>The gateway is where the platform's numbers come from, because it sees every request once. Two rules keep them honest.</p>
<p><b>Bill the actual usage, never the estimate.</b> The estimate only gated admission. The engine reports exact token counts in <code>usage</code>. For a unary request that is in the response body. For a stream it is in the <i>last</i> chunk, and only if the request set <code>stream_options.include_usage</code>; without it an OpenAI-compatible stream reports no usage at all, so the gateway sets it by default. The gateway watches each forwarded line for a <code>usage</code> field and calls <code>TokenBudget.commit(prompt + completion)</code> at the end (<code>Gateway._bill</code>). Even a stream cut mid-way is billed for what was reported.</p>
<p><b>Export counters.</b> The gateway has its own Prometheus <code>/metrics</code> (P1.3), labelled by tenant, model and backend:</p>
<ul><li><code>gateway_requests_total{tenant, model, backend, outcome}</code>: includes <code>err_503</code>-style outcomes;</li><li><code>gateway_tokens_total{…, kind="prompt"|"completion"}</code>: the per-tenant bill;</li><li><code>gateway_request_seconds</code>, <code>gateway_ttft_seconds</code>: latency per backend;</li><li><code>gateway_retries_total</code> and <code>gateway_inflight</code>: how much the retry budget and routing are working.</li></ul>
<p>One more cost to know: the hop itself. A Python async proxy parses JSON, hashes the key and forwards bytes, so it adds some latency to every request. README guidance is a few hundred microseconds to a few milliseconds on localhost, small against a request that takes a second. Do not trust that: <code>bench/gateway_overhead.py</code> measures p50 and p99 added latency directly, and you should paste your own result.</p>`,
        scene(G) {
          G.text(24, 32, "streaming response → usage in the last chunk → bill", { size: 13 });
          const ch = ["data: {\"text\": \"The\"}", "data: {\"text\": \" cat\"}", "data: {\"text\": \" sat\"}", "data: {\"usage\": {\"prompt_tokens\": 500, \"completion_tokens\": 90}}", "data: [DONE]"];
          const boxes = ch.map((t, i) => G.box(24, 56 + i * 38, 440, 30, t, { stroke: i === 3 ? "ok" : "line", color: i === 3 ? "ok" : "ink", size: 11, sw: i === 3 ? 2.5 : 1.5 }));
          G.label(480, 76, "tokens: forwarded", { size: 11 }); G.label(480, 152, "usage chunk:", { size: 11, color: "ok" }); G.label(480, 168, "requires include_usage", { size: 11, color: "ok" });
          G.arrow(244, 252, 244, 282, { color: "ok", w: 2 });
          G.box(24, 286, 220, 38, "commit(500 + 90 = 590)", { fill: "ok", size: 13 });
          G.arrow(246, 305, 290, 305, { color: "muted" });
          G.box(292, 286, 220, 38, "gateway_tokens_total", { stroke: "k", color: "k", size: 13 });
          G.label(24, 348, "billed = what the engine reported, even if the stream broke later", { size: 12, color: "ink" });
          G.label(24, 370, "estimate (756 in step 4) only decided whether to admit", { size: 12 });
          G.label(24, 392, "overhead of the hop: measure with bench/gateway_overhead.py", { size: 12 });
          G.from(boxes, { opacity: 0, stagger: 0.1, duration: 0.25 });
          G.caption("one place sees every request: count it exactly once");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A gateway owns auth, rate limits, token budgets, routing, retries, fallbacks and accounting, in that order, before and after the GPU.</li>
<li>Keys are stored as hashes and compared in constant time. A token bucket allows bursts and caps the rate; a sliding token budget caps spend, admitted on an estimate and billed on actual usage.</li>
<li>Least outstanding requests beats round-robin when request costs vary by 100×.</li>
<li>Retry only retryable errors, with full-jitter backoff and a global retry budget so an outage is not multiplied.</li>
<li>After the first streamed byte there are no retries: send an explicit error event, never a spliced completion.</li>
<li>A circuit breaker skips a dead backend and probes it once per cooldown.</li></ul>`,
    sim: {
      title: "Token bucket and retry amplification",
      intro: "Top: the exact TokenBucket logic from platform/gateway/ratelimit.py, run in virtual time for 20 seconds. Clients send groups of requests together; green ticks are allowed, red are refused (429). Bottom panels: how many attempts a failing backend costs per request, with and without a retry budget. Press Replay to watch time pass.",
      controls: [
        { id: "rate", label: "bucket rate, tokens per second", min: 1, max: 40, value: 5 },
        { id: "burst", label: "bucket size (burst)", min: 1, max: 40, value: 10 },
        { id: "offered", label: "offered load, requests per second", min: 1, max: 60, value: 8 },
        { id: "group", label: "requests sent together in each group", type: "select", value: 1, options: [[1, "1 (smooth)"], [2, "2"], [5, "5"], [10, "10"], [20, "20 (bursty)"]] },
        { id: "f", label: "chance an attempt fails (retry model)", min: 0, max: 1, step: 0.05, value: 1, format: (x) => x.toFixed(2) },
        { id: "M", label: "max attempts per request", min: 1, max: 6, value: 3 },
        { id: "ratio", label: "retry budget ratio", min: 0, max: 0.5, step: 0.05, value: 0.2, format: (x) => x.toFixed(2) },
      ],
      height: 270,
      run: { label: "Replay time", frames: 40, ms: 50 },
      draw(G, v, t) {
        const DUR = 20, r = bucketRun(v.rate, v.burst, v.offered, v.group, DUR), now = DUR * t;
        const x0 = 50, w = 570, yt = 20, ht = 150, X = (s) => x0 + (w * s) / DUR, Y = (n) => yt + ht - (ht * n) / v.burst;
        G.label(x0, 12, "tokens in the bucket over 20 s", { size: 11 });
        G.rect(x0, yt, w, ht, { stroke: "line", rx: 3 });
        G.line(x0, Y(v.burst), x0 + w, Y(v.burst), { color: "line", dash: "3 4", w: 1 });
        G.label(x0 - 4, Y(v.burst) + 4, String(v.burst), { anchor: "end", size: 10 }); G.label(x0 - 4, Y(0) + 4, "0", { anchor: "end", size: 10 });
        let d = ""; r.curve.forEach(([s, n], i) => { if (s <= now) d += `${d ? " L" : "M"} ${X(s)} ${Y(n)}`; });
        if (d) G.path(d, { color: "k", w: 2 });
        let ok = 0, bad = 0;
        r.reqs.forEach(([s, a]) => { if (s > now) return; if (a) ok++; else bad++; G.line(X(s), 190, X(s), a ? 206 : 226, { color: a ? "ok" : "hot", w: 1.2 }); });
        G.label(x0 - 4, 206, "ok", { anchor: "end", size: 10, color: "ok" }); G.label(x0 - 4, 226, "429", { anchor: "end", size: 10, color: "hot" });
        [0, 5, 10, 15, 20].forEach((s) => G.label(X(s), 248, s + " s", { anchor: "middle", size: 10 }));
        G.line(X(now), yt, X(now), 232, { color: "ink", dash: "2 3", w: 1 });
        const total = r.reqs.length, maxOk = Math.min(total, Math.floor(v.burst + v.rate * DUR));
        const A = (() => { let s = 0, p = 1; for (let i = 0; i < v.M; i++) { s += p; p *= v.f; } return s; })();
        const cap = Math.min(A, 1 + v.ratio);
        return [
          { title: "Token bucket, whole 20 s", rows: [["requests sent", String(total)], ["allowed", String(r.ok)], ["refused (429)", String(r.bad)], ["average allowed rate", (r.ok / DUR).toFixed(1) + " /s"], ["most the bucket could allow", `${maxOk} (burst + rate × 20)`]],
            chip: [r.bad === 0, r.bad === 0 ? "nothing refused" : `${(100 * r.bad / total).toFixed(0)}% refused`] },
          { title: "Retries on a failing backend", rows: [["attempts per request, no budget", A.toFixed(2) + "×"], ["with budget (≤ 1 + ratio)", cap.toFixed(2) + "×"], ["load at your offered rate, no budget", F.num(v.offered * A) + " /s"], ["load with budget", F.num(v.offered * cap) + " /s"]],
            html: `<p class="note">Bucket: virtual-time port of TokenBucket.allow (refill = elapsed × rate, capped at burst). Retries: if each attempt fails independently with chance f, a request makes Σ f^i attempts up to max attempts; the budget caps average attempts at 1 + ratio (the min_retries floor is ignored). A model, not a measurement.</p>` },
        ];
      },
    },
    practice: {
      intro: "All five exercises are T0: they run on a laptop against mock backends, in virtual time where a clock matters. Set S2S_SOLUTIONS=1 to run the same tests against the reference solutions. Nothing here needs a GPU; point the gateway at your P2.1 vLLM server if you want a real backend.",
      items: [
        { title: "Token bucket", tier: "T0 · easy", goal: "TokenBucket(rate, burst, clock) with allow(n) and retry_after(n), refilling lazily. A virtual clock checks the burst, refusal when empty, partial refill, the cap and an exact Retry-After.",
          cmd: "uv run pytest course/P2-serving-engines/P2.7-gateway-basics/exercises/test_bucket.py" },
        { title: "Retry policy with budget and jitter", tier: "T0 · medium", goal: "RetryBudget, backoff_s (full jitter) and should_retry: never after streaming started, never on a 400, retry a 503 and transport errors, respect max_attempts, cap retries at 20% of 1000 requests.",
          cmd: "uv run pytest course/P2-serving-engines/P2.7-gateway-basics/exercises/test_retry.py" },
        { title: "Circuit breaker and fallback", tier: "T0 · medium", goal: "CircuitBreaker(threshold, cooldown_s, clock) with closed, open and half-open states and one probe. Also an end-to-end test: one of two mock backends at FAIL_RATE=1.0, and 30 requests must all succeed.",
          cmd: "uv run pytest course/P2-serving-engines/P2.7-gateway-basics/exercises/test_breaker.py" },
        { title: "Per-tenant token budget", tier: "T0 · medium", goal: "TokenBudget(limit, window_s, clock): remaining, check(estimate) that consumes nothing, commit(actual), expiry exactly at the window edge, and a gateway that answers 429 once a tiny budget is used up.",
          cmd: "uv run pytest course/P2-serving-engines/P2.7-gateway-basics/exercises/test_budget.py" },
        { title: "SSE passthrough with mid-stream failure", tier: "T0 · hard", goal: "With MOCKLLM_FAIL_MIDSTREAM_RATE=1.0 the client must get 3 tokens, then one upstream_interrupted event, never a silent retry; a failure before the first byte is retried on the healthy backend. Then write a design for resumable streams.",
          cmd: "uv run pytest course/P2-serving-engines/P2.7-gateway-basics/exercises/test_midstream.py" },
      ],
      labs: [
        { label: "#6, the gateway: read ratelimit.py, retry.py, router.py, then app.py", path: "platform/gateway/" },
        { label: "Run it against two mock replicas, one flaky (walkthrough in the module README)", path: "course/P2-serving-engines/P2.7-gateway-basics/README.md" },
        { label: "Measure the latency the gateway adds (p50 / p99 vs direct)", path: "course/P2-serving-engines/P2.7-gateway-basics/bench/gateway_overhead.py" },
        { label: "The mock server the gateway talks to, with failure knobs", path: "platform/mockllm/server.py" },
        { label: "Animation: gateway to queue to autoscaler to pod", path: "animations/p3-request-path.html" },
        { label: "Compare with LiteLLM (router, fallbacks) and Envoy AI Gateway (token rate limits), pinned in SOURCES.md", path: "SOURCES.md" },
      ],
    },
  });
})();
