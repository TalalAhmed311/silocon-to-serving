/* P3.6 — Autoscaling GPUs: queue depth, cold starts and spot. */
(function () {
  const F = S2S.fmt;

  /* ---- the policy from platform/autoscaler/policy.py, ported line by line ---- */
  const POL = { target: 24, min: 0, max: 4, cooldown: 600, window: 300, up: 2, down: 1 };
  const rawDesired = (w, p) => (w <= 0 ? 0 : Math.ceil(w / p.target));
  function polStep(s, t, w, p) {
    if (w > 0) s.lastWork = t;
    const want = rawDesired(w, p);
    s.hist = s.hist.filter(([ht]) => t - ht <= p.window).concat([[t, want]]);
    let nw;
    if (want > s.replicas) nw = Math.min(want, s.replicas + p.up);
    else {
      const stable = Math.max(...s.hist.map(([, d]) => d));
      nw = stable < s.replicas ? Math.max(stable, s.replicas - p.down) : s.replicas;
    }
    if (nw === 0 && t - s.lastWork < p.cooldown) nw = 1;
    nw = Math.max(p.min, Math.min(p.max, nw));
    s.replicas = nw; return nw;
  }
  const newState = (r = 0) => ({ replicas: r, lastWork: -1e18, hist: [] });

  /* seeded noise so scenes are the same every time */
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { const u = Math.max(1e-9, r()), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  /* a small "pod" glyph: GPU box with a state label */
  const pod = (G, x, y, label, fill, o = {}) => G.box(x, y, o.w || 64, o.h || 40, label, Object.assign({ fill, size: 11, rx: 6 }, o));

  /* step-line through (x, y) points */
  const stepPath = (pts) => pts.map(([x, y], i) => (i ? `H ${x} V ${y}` : `M ${x} ${y}`)).join(" ");

  S2S.lesson({
    id: "p3-6", n: "P3.6", title: "Autoscaling GPUs",
    subtitle: "Deployment and infrastructure · queue depth, cold starts, spot · T0 + T3",
    kicker: "Lesson · ≈ 45 min",
    headline: "Rent GPUs for the traffic you have, not the traffic you fear",
    intro: `<p>Your cluster from P3.4 serves a model on a fixed number of GPU pods, and P3.5 gave you the metrics to watch it. Traffic is not fixed: it rises in the morning, spikes when a launch goes out and falls to nothing at night. This lesson builds an autoscaler from the problem up: which number to scale on, how fast to move in each direction, what happens when there are zero pods, why a new GPU pod takes minutes, and how to use cheap <b>spot</b> GPUs that the cloud can take back with two minutes' notice.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "demand"], ["ok", "capacity"], ["hot", "overload"], ["w", "idle GPUs"]],
    prev: "p3-5", next: "p3-7",
    steps: [
      { rail: "the problem", title: "A fixed fleet is wrong twice a day",
        body: `<p>A <b>replica</b> is one copy of the serving pod: one vLLM process on one GPU. It can hold a limited number of requests at once before latency climbs. In P2.4 you measured that limit as the <b>knee</b>: the concurrency where TTFT turns sharply upwards.</p>
<p>Suppose one replica's knee is 30 concurrent requests, and the daily demand swings from about 10 concurrent requests at night to about 90 at the peak. A fixed fleet has two bad choices:</p>
<ul><li><b>Size for the peak</b> (ceil(90 ÷ 30) = 3 replicas, 4 with headroom): at night three of the four GPUs sit idle, and you pay for every one of their hours.</li>
<li><b>Size for the average</b>: at the peak, requests queue, TTFT explodes and the SLO from P3.5 burns.</li></ul>
<p><b>Autoscaling</b> changes the number of replicas to follow demand. The rest of the lesson is about doing that without making things worse.</p>
<div class="analogy"><b>Picture it</b>A restaurant that staffs every table for Saturday night pays waiters to stand around on Tuesday morning. One that staffs for Tuesday turns people away on Saturday.</div>`,
        scene(G) {
          const x0 = 50, x1 = 610, yb = 360, yt = 70, cap = 120; // y scale: concurrency 0..120
          const Y = (v) => yb - ((yb - yt) * v) / cap, X = (h) => x0 + ((x1 - x0) * h) / 24;
          const d = (h) => 10 + 80 * Math.exp(-((h - 14) ** 2) / 18);
          G.axes(x0, yt, x1 - x0, yb - yt, { xlabel: "hour of day →" });
          [0, 30, 60, 90, 120].forEach((v) => G.label(x0 - 8, Y(v) + 4, String(v), { anchor: "end", size: 10 }));
          G.label(x0, yt - 14, "concurrent requests", { size: 11 });
          // fixed fleet of 2 replicas (60) vs demand
          const fix = 60;
          let over = "", idle = "";
          const pts = []; for (let i = 0; i <= 96; i++) { const h = (24 * i) / 96; pts.push([X(h), Y(d(h))]); }
          const line = "M " + pts.map((p) => p.join(" ")).join(" L ");
          idle = `M ${X(0)} ${Y(fix)} ` + pts.map(([x, y]) => `L ${x} ${Math.max(y, Y(fix))}`).join(" ") + ` L ${X(24)} ${Y(fix)} Z`;
          over = `M ${X(0)} ${Y(fix)} ` + pts.map(([x, y]) => `L ${x} ${Math.min(y, Y(fix))}`).join(" ") + ` L ${X(24)} ${Y(fix)} Z`;
          G.path(idle, { fill: "w", opacity: 0.45 });
          const ov = G.path(over, { fill: "hot", opacity: 0.6 });
          G.line(X(0), Y(fix), X(24), Y(fix), { color: "ok", w: 2, dash: "6 4" });
          G.label(X(0.3), Y(fix) - 6, "fixed fleet: 2 replicas × 30 = 60", { color: "ok", size: 11 });
          G.path(line, { color: "k", w: 2.5 });
          G.label(X(14), Y(92), "peak ≈ 90", { color: "k", anchor: "middle", size: 11 });
          G.label(X(3), Y(20), "idle GPU-hours", { color: "ink", size: 11 });
          G.text(70, 400, "grey = paid but idle    red = demand above capacity: queueing", { size: 12, color: "muted" });
          G.pulse(ov, { lo: 0.25, hi: 0.7, repeat: 5 });
          G.caption("any fixed size either wastes GPUs or breaks the SLO");
        } },

      { rail: "wrong signals", title: "GPU utilisation can't see the load",
        legend: [["hot", "GPU utilisation"], ["k", "running + waiting"]],
        body: `<p>Kubernetes' built-in <b>Horizontal Pod Autoscaler (HPA)</b> adds or removes replicas to keep a metric near a target. By default that metric is CPU utilisation. For an LLM server every obvious metric misleads:</p>
<table><tr><th>signal</th><th>what goes wrong</th></tr>
<tr><td>CPU utilisation</td><td>the CPU mostly waits while the GPU works; it never crosses the target</td></tr>
<tr><td>GPU utilisation (<code>DCGM_FI_DEV_GPU_UTIL</code>)</td><td>it measures the fraction of time <i>any</i> kernel is running. One decoding request keeps a kernel running almost all the time, so it reads near 100% with 1 user and with 400</td></tr>
<tr><td>requests per second</td><td>ignores length: 10 req/s of 8k-token prompts is far more work than 10 req/s of short chat</td></tr></table>
<p>A signal is only useful for scaling if it keeps growing as the load grows. GPU utilisation is flat from the first request onwards: it is <b>saturated</b>, so it carries no information about how many more GPUs you need.</p>`,
        check: { q: "A replica reports 98% GPU utilisation. What can you conclude about whether to add a replica?",
          options: ["It is overloaded: scale up", "Almost nothing: one decoding request already reads near 100%", "It is idle: scale down"], answer: 1,
          why: "GPU utilisation counts time with any kernel running. Decode keeps kernels running constantly even at batch 1, so 98% is consistent with 1 user or with a queue of hundreds. The metric is saturated and can't rank loads." },
        scene(G) {
          const users = [1, 10, 50, 100, 400];
          G.text(24, 40, "same 5 load levels, two metrics", { size: 14 });
          // left: GPU util (illustrative shape, from the definition)
          G.axes(40, 80, 250, 250, {}); G.label(40, 70, "GPU utilisation", { size: 12, color: "hot" });
          const gu = G.bars(56, 330, [0.97, 0.98, 0.99, 0.99, 0.99], { w: 34, gap: 14, h: 230, max: 1, fill: "hot" });
          // right: running + waiting
          G.axes(350, 80, 260, 250, {}); G.label(350, 70, "running + waiting requests", { size: 12, color: "k" });
          const rw = G.bars(366, 330, users, { w: 34, gap: 14, h: 230, max: 400, fill: "k" });
          users.forEach((u, i) => { G.label(73 + i * 48, 350, String(u), { anchor: "middle", size: 11 }); G.label(383 + i * 48, 350, String(u), { anchor: "middle", size: 11 }); });
          G.label(165, 372, "concurrent users", { anchor: "middle", size: 11 }); G.label(480, 372, "concurrent users", { anchor: "middle", size: 11 });
          G.text(40, 404, "flat: can't tell 1 user from 400", { color: "hot", size: 12 });
          G.text(350, 404, "grows with the load", { color: "k", size: 12 });
          G.from(rw, { attr: { height: 0, y: 330 }, stagger: 0.12, duration: 0.4 });
          G.caption("shapes from the metrics' definitions, not a benchmark");
        } },

      { rail: "the right signal", title: "Scale on the work in the system",
        legend: [["k", "one request"], ["line", "free slot"]],
        body: `<p>vLLM exports two gauges (P3.5): <code>vllm:num_requests_running</code>, the requests in the current batch, and <code>vllm:num_requests_waiting</code>, those queued for a slot. Their sum is the <b>work in the system</b>, and it grows with load without saturating.</p>
<p>How big will it be? <b>Little's law</b>: on average, the number of requests in a system equals the arrival rate times the time each one spends inside.</p>
<div class="eq">in system = arrival rate × time per request
          = 4 req/s × 6 s = 24 requests</div>
<p>Pick a <b>target per replica</b> a bit below the knee, about 0.8 × knee, so a replica runs just short of where TTFT climbs. Then:</p>
<div class="eq">desired = ceil( (Σ running + Σ waiting) / target )

knee 30 → target 0.8 × 30 = 24
work 70 → ceil(70 / 24) = ceil(2.92) = 3 replicas</div>
<p>In the repo this is <code>raw_desired()</code> in <code>platform/autoscaler/policy.py</code>, and <code>threshold: "24"</code> in <code>scaledobject.yaml</code>.</p>`,
        check: { q: "Target is 24 per replica. The fleet reports 49 running + waiting. How many replicas does the formula ask for?",
          options: ["2", "3", "49"], answer: 1,
          why: "49 ÷ 24 = 2.04, and ceil rounds up to 3. Two replicas would hold 48 at target, so the 49th request already needs a third." },
        scene(G) {
          const work = 70, T = 24;
          G.text(24, 40, "work = 70 requests (running + waiting) · target = 24 / replica", { size: 13 });
          const cells = [];
          for (let r = 0; r < 3; r++) {
            const y = 80 + r * 96;
            G.rect(24, y, 592, 76, { stroke: "line", rx: 10 });
            G.label(34, y + 18, `replica ${r + 1}`, { size: 11 });
            for (let i = 0; i < T; i++) {
              const n = r * T + i, on = n < work;
              cells.push(G.rect(34 + i * 24, y + 30, 20, 34, { fill: on ? "k" : "line", rx: 3, opacity: on ? 1 : 0.5 }));
            }
          }
          G.text(24, 400, "ceil(70 / 24) = ceil(2.92) = 3 replicas", { color: "ok", size: 15 });
          G.from(cells, { opacity: 0, stagger: 0.012, duration: 0.15 });
          G.caption("each square is one request; 70 − 48 = 22 land on the third replica");
        } },

      { rail: "the loop", title: "KEDA closes the loop",
        legend: [["k", "pods"], ["q", "metrics"], ["v", "KEDA"], ["ok", "HPA"]],
        body: `<p>The formula needs to run every few seconds against live metrics. Three pieces do that:</p>
<ol><li><b>Prometheus</b> scrapes every vLLM pod and stores the gauges (P3.5).</li>
<li><b>KEDA</b> (Kubernetes Event-Driven Autoscaling, pinned at v2.21.0) runs your PromQL query every <code>pollingInterval</code> seconds and hands the value to the HPA as an external metric.</li>
<li>The <b>HPA</b> divides by the <code>threshold</code>, applies its <code>behavior</code> rules, and edits the Deployment's replica count. The scheduler then places the new pods.</li></ol>
<div class="eq">triggers:
  - type: prometheus
    metadata:
      query: sum(vllm:num_requests_running)
           + sum(vllm:num_requests_waiting)
      threshold: "24"     # target per replica
pollingInterval: 15       # seconds</div>
<p>Each turn of the loop is cheap. The slow part is what happens after the replica count changes: a new pod needs a GPU node, an image and the weights. Keep that in mind; it drives every design choice that follows.</p>`,
        scene(G) {
          const nodes = [[60, 90, "vLLM pods", "k"], [430, 90, "Prometheus", "q"], [430, 290, "KEDA", "v"], [60, 290, "HPA → Deployment", "ok"]];
          nodes.forEach(([x, y, l, c]) => G.box(x, y, 150, 56, l, { stroke: c, color: c, size: 13 }));
          const a = [
            G.arrow(212, 118, 426, 118, { color: "q", w: 2 }),
            G.arrow(505, 148, 505, 286, { color: "v", w: 2 }),
            G.arrow(428, 318, 214, 318, { color: "ok", w: 2 }),
            G.arrow(135, 288, 135, 150, { color: "k", w: 2 }),
          ];
          G.label(320, 108, "scrape gauges", { anchor: "middle", size: 11 });
          G.label(515, 210, "PromQL every 15 s", { size: 11 });
          G.label(320, 340, "external metric: 70", { anchor: "middle", size: 11 });
          G.label(145, 210, "replicas = 3", { size: 11 });
          G.text(320, 210, "≈ seconds", { anchor: "middle", size: 14, color: "ok" });
          G.box(40, 372, 560, 34, "new pod → GPU node + image + weights: minutes", { stroke: "hot", color: "hot", size: 12 });
          G.from(a, { opacity: 0, stagger: 0.3, duration: 0.3 });
          G.caption("deciding is fast; acting on the decision is slow");
        } },

      { rail: "asymmetry", title: "Up fast, down slow",
        legend: [["hot", "raw formula"], ["ok", "stabilised policy"]],
        body: `<p>Demand is noisy. Feed a noisy number into <code>ceil(work / target)</code> and the replica count jumps up and down every poll. That is <b>flapping</b>, and on GPUs it is expensive: every scale-down that is reversed a minute later costs a full cold start.</p>
<p>The two mistakes have very different prices. Too few replicas breaks the SLO <i>now</i>. Too many for a few minutes only costs money. So the policy is deliberately <b>asymmetric</b>:</p>
<ul><li><b>Scale up</b> as soon as demand rises, but at most +2 pods per minute.</li>
<li><b>Scale down</b> only to the <i>maximum</i> desired count seen over the last 300 s (the <b>stabilisation window</b>), and at most −1 pod per 2 minutes.</li></ul>
<div class="eq">behavior:
  scaleUp:
    stabilizationWindowSeconds: 0
    policies: [{type: Pods, value: 2, periodSeconds: 60}]
  scaleDown:
    stabilizationWindowSeconds: 300
    policies: [{type: Pods, value: 1, periodSeconds: 120}]</div>
<p>The scene runs the exact <code>step()</code> logic from <code>policy.py</code> on a noisy trace (mean 50, target 20, polled every 15 s) and counts how often each version changes the replica count.</p>`,
        check: { q: "Why does the scale-down rule use the maximum desired count over the window, not the latest one?",
          options: ["To save money faster", "So one quiet sample can't remove a GPU that the next sample needs again", "Because Prometheus only stores maxima"], answer: 1,
          why: "Taking the maximum over 300 s means capacity is only removed once demand has stayed low for the whole window. A single dip no longer triggers a scale-down that would be reversed, with a cold start, a minute later." },
        scene(G) {
          const r = rng(7), N = 60, p = Object.assign({}, POL, { target: 20, max: 10, up: 10 });
          const trace = []; for (let i = 0; i < N; i++) trace.push(Math.max(0, 50 + 15 * gauss(r)));
          const raw = trace.map((w) => rawDesired(w, p));
          const s = newState(0), pol = trace.map((w, i) => polStep(s, i * 15, w, p));
          const ch = (xs) => xs.reduce((n, x, i) => n + (i && x !== xs[i - 1] ? 1 : 0), 0);
          const X = (i) => 50 + i * 9.4;
          const panel = (y0, vals, color, title, n) => {
            const Y = (v) => y0 + 130 - v * 24;
            G.axes(50, y0, 564, 130, {}); [0, 2, 4].forEach((v) => G.label(42, Y(v) + 4, String(v), { anchor: "end", size: 10 }));
            G.label(50, y0 - 8, `${title} · ${n} changes`, { color, size: 12 });
            return G.path(stepPath(vals.map((v, i) => [X(i), Y(v)]).concat([[X(N), Y(vals[N - 1])]])), { color, w: 2.5 });
          };
          const a = panel(50, raw, "hot", "raw ceil(work / 20)", ch(raw));
          const b = panel(240, pol, "ok", "policy.py step(): up fast, down after 300 s", ch(pol));
          G.label(614, 398, "time (15 s polls) →", { anchor: "end", size: 11 });
          G.from([a, b], { opacity: 0, stagger: 0.5, duration: 0.5 });
          G.caption("same noisy demand; the stabilised policy barely moves");
        } },

      { rail: "scale to zero", title: "Zero replicas, and the signal that disappears",
        legend: [["v", "waiting request"], ["k", "gateway signal"], ["ok", "activation"]],
        body: `<p>Plain HPA never goes below one replica. KEDA can go to <b>zero</b>: after <code>cooldownPeriod</code> seconds (600 here) with no demand, it removes the last pod, and the GPU node pool (minimum 0) then releases the node. No traffic, no GPU bill.</p>
<p>Now try to scale back up. The query reads <code>vllm:num_requests_waiting</code>, but that gauge is exported <i>by the vLLM pods</i>. With zero pods the metric does not exist, so the demand signal is gone exactly when you need it.</p>
<p>The fix is to measure demand somewhere that is always running. The gateway from P2.7 sits in front of every request, so its <code>gateway_inflight</code> gauge is the second trigger in <code>scaledobject.yaml</code>. <code>activationThreshold: "0"</code> means any value above 0 wakes the Deployment from zero. Requests that arrive during the cold start wait at the gateway, or get a fast <code>503</code> with <code>Retry-After</code> if you prefer to fail quickly.</p>
<p>The trade-off: zero replicas saves every idle GPU-hour, and the first user after a quiet spell waits a whole cold start. Good for dev and batch; rarely acceptable for an interactive SLO.</p>`,
        check: { q: "The Deployment is at zero replicas and a request arrives. Which metric can wake it?",
          options: ["vllm:num_requests_waiting", "gateway_inflight from the gateway", "DCGM GPU utilisation"], answer: 1,
          why: "At zero replicas there is no vLLM pod and no GPU, so neither the vLLM gauges nor DCGM exist. The gateway is always up and sees the request, so its in-flight gauge is the only signal that survives scale-to-zero." },
        scene(G) {
          G.box(24, 170, 120, 60, "client", { stroke: "muted", color: "ink" });
          const gw = G.box(210, 160, 150, 80, "gateway", { stroke: "k", color: "k" });
          G.arrow(146, 200, 206, 200, { color: "muted" });
          const q = []; for (let i = 0; i < 5; i++) q.push(G.rect(222 + i * 26, 250, 20, 20, { fill: "v", rx: 3 }));
          G.label(222, 290, "requests wait here", { size: 11, color: "v" });
          G.rect(430, 130, 180, 140, { stroke: "line", dash: "6 5", rx: 10 });
          G.text(520, 190, "0 replicas", { anchor: "middle", size: 14, color: "muted" });
          G.text(520, 214, "no pod → no vllm:*", { anchor: "middle", size: 11, color: "hot" });
          G.arrow(362, 200, 426, 200, { color: "line", dash: "4 4" });
          G.box(210, 40, 150, 46, "Prometheus", { stroke: "q", color: "q", size: 12 });
          G.box(430, 40, 180, 46, "KEDA", { stroke: "v", color: "v", size: 12 });
          G.arrow(285, 158, 285, 90, { color: "k", w: 2 }); G.label(292, 128, "gateway_inflight = 5", { size: 11, color: "k" });
          G.arrow(362, 63, 426, 63, { color: "q" });
          G.arrow(520, 88, 520, 126, { color: "ok", w: 2 }); G.label(528, 112, "activate: 0 → 1", { size: 11, color: "ok" });
          G.text(24, 340, "cooldownPeriod: 600 s of zero demand before 1 → 0", { size: 12 });
          G.text(24, 364, "activationThreshold: 0 · any demand wakes it", { size: 12 });
          G.from(q, { opacity: 0, x: -20, stagger: 0.15, duration: 0.3 });
          G.caption("the wake-up signal must come from something that is always up");
        } },

      { rail: "cold start", title: "Where a cold start's minutes go",
        body: `<p>A <b>cold start</b> is the time from "the autoscaler wants a pod" to "that pod serves its first token". When there is no free GPU node, it has four phases, in order:</p>
<ol><li><b>Node provisioning.</b> The pod is <code>Pending</code>. The cloud launches a GPU instance, it boots, joins the cluster, and the GPU Operator (P3.3) installs the driver and device plugin before <code>nvidia.com/gpu</code> appears.</li>
<li><b>Image pull.</b> A CUDA + PyTorch + vLLM image is often over 10 GB (P3.1).</li>
<li><b>Weight load.</b> The model is copied from storage to the GPU: <code>model bytes ÷ effective bandwidth</code> (P3.7).</li>
<li><b>Warm-up.</b> vLLM allocates the KV cache, captures CUDA graphs (P2.6) and maybe compiles. Readiness (P3.3) only passes after this.</li></ol>
<p>You can estimate phases 2 and 3 before measuring anything. The bandwidths below are <b>example values</b>; measure yours in P3.7.</p>
<div class="eq">weights: Llama-3-8B bf16 ≈ 16 GB
  at 0.5 GB/s → 16 / 0.5 = 32 s
  at 2.0 GB/s → 16 / 2.0 =  8 s
image: 10 GB at 0.25 GB/s → 40 s</div>
<p>Node provisioning is usually "a few minutes" and is the hardest to predict. <code>python -m autoscaler.coldstart --pod &lt;pod&gt;</code> turns the pod's Kubernetes events (Scheduled, Pulling, Pulled, Started, Ready) into the real phase table.</p>`,
        legend: [["blue", "node"], ["v", "image"], ["q", "weights"], ["k", "warm-up"]],
        scene(G) {
          const ph = [["node provisioning", "blue", "Pending → Scheduled"], ["image pull", "v", "Pulling → Pulled"], ["weight load", "q", "Started → …"], ["warm-up", "k", "… → Ready"]];
          const w = [230, 120, 110, 70], x0 = 40; let x = x0;
          G.text(24, 40, "one cold start, phase by phase (widths illustrative: measure yours)", { size: 13 });
          const bars = ph.map(([l, c, ev], i) => { const b = G.rect(x, 90 + i * 64, w[i], 40, { fill: c, rx: 5 }); G.text(x + 8, 115 + i * 64, l, { size: 12, color: "bg" }); G.label(x + w[i] + 8, 115 + i * 64, ev, { size: 11 }); x += w[i]; return b; });
          G.line(x0, 360, x, 360, { color: "ink", w: 2 }); G.line(x0, 352, x0, 368, { color: "ink", w: 2 }); G.line(x, 352, x, 368, { color: "ink", w: 2 });
          G.text((x0 + x) / 2, 386, "TTFT spike users see = the whole bar", { anchor: "middle", size: 13, color: "hot" });
          G.label(x0, 404, "pod created", { size: 11 }); G.label(x, 404, "Ready", { size: 11, anchor: "end" });
          G.from(bars, { attr: { width: 0 }, stagger: 0.35, duration: 0.4 });
          G.caption("each phase waits for the previous one; the sum is the cold start");
        } },

      { rail: "mitigations", title: "Attack each phase separately",
        body: `<p>Each phase has its own fix, and they add up because the phases are sequential.</p>
<table><tr><th>phase</th><th>mitigation</th></tr>
<tr><td>node provisioning</td><td><b>warm pool</b>: keep spare GPU nodes. One trick is <b>balloon pods</b>: placeholder pods with a low PriorityClass that hold a GPU node; a real pod pre-empts one instantly. Or <code>min_size ≥ 1</code> during business hours</td></tr>
<tr><td>image pull</td><td>pre-pull with a DaemonSet or bake into the node image; a smaller image (P3.1); a registry in the same region</td></tr>
<tr><td>weight load</td><td>node-local NVMe cache, streaming loaders, safetensors mmap (P3.7)</td></tr>
<tr><td>warm-up</td><td>fewer CUDA-graph capture sizes; cache compiled artifacts</td></tr></table>
<p>Why bother? Because requests keep arriving during the cold start and pile up. By Little's law the backlog is the arrival rate times the wait:</p>
<div class="eq">backlog = extra arrival rate × cold start
  2 req/s × 180 s = 360 requests waiting
  2 req/s ×  45 s =  90 requests waiting</div>
<p>This is also why the target is 0.8 × knee and not the knee itself: the 20% headroom absorbs growth while the next replica is still starting.</p>`,
        legend: [["blue", "node"], ["v", "image"], ["q", "weights"], ["k", "warm-up"]],
        scene(G) {
          const cols = ["blue", "v", "q", "k"];
          const before = [230, 120, 110, 70], after = [0, 10, 40, 50];
          G.text(24, 40, "same four phases, before and after mitigations (illustrative)", { size: 13 });
          const drawBar = (y, ws, label) => { let x = 40; G.label(40, y - 8, label, { size: 12, color: "ink" }); return ws.map((w, i) => { const r = G.rect(x, y, w, 40, { fill: cols[i], rx: 4 }); x += w; return r; }); };
          drawBar(80, before, "cold: new GPU node, image not cached, weights from S3");
          const b = drawBar(190, after, "mitigated: balloon pod node, pre-pulled image, NVMe weights");
          G.label(40, 250, "node provisioning removed · pull ≈ 0 · weights from local disk", { size: 11 });
          // backlog bars
          G.text(24, 300, "backlog that piles up meanwhile (2 extra req/s)", { size: 13 });
          G.rect(40, 316, 360, 26, { fill: "hot", rx: 4 }); G.label(408, 334, "180 s → 360 waiting", { size: 12, color: "hot" });
          G.rect(40, 352, 90, 26, { fill: "ok", rx: 4 }); G.label(138, 370, "45 s → 90 waiting", { size: 12, color: "ok" });
          G.from(b, { attr: { width: 0 }, stagger: 0.2, duration: 0.4 });
          G.caption("shorter cold start → smaller queue → smaller TTFT spike");
        } },

      { rail: "spot", title: "Spot GPUs: cheap, with a two-minute warning",
        body: `<p><b>Spot</b> (AWS's name; "preemptible" or "spot VMs" elsewhere) is spare capacity sold at a discount. The catch: the cloud can take the instance back. On AWS you get a <b>two-minute interruption notice</b> (EC2 docs, UNVERIFIED wording).</p>
<p>Two minutes is enough to leave gracefully if every layer cooperates:</p>
<ol><li><b>Detect.</b> Karpenter or the AWS Node Termination Handler sees the notice, then <b>cordons</b> the node (no new pods) and drains it.</li>
<li><b>Stop taking work.</b> The <code>preStop</code> hook from P3.3 makes the pod fail readiness, so the Service and gateway stop sending it requests.</li>
<li><b>Finish in-flight work</b> within <code>terminationGracePeriodSeconds</code>. Keep it under about 90 s to leave margin before the 120 s kill.</li>
<li><b>Retry the rest.</b> The gateway (P2.7) retries requests that failed <i>before the first byte</i> on another replica. A stream that already sent tokens can't be replayed transparently: the client has seen half an answer.</li></ol>
<p>A stream still running when the grace period ends is cut. If responses take <i>R</i> seconds and the notice lands at a random moment, the time a stream has left is spread evenly between 0 and <i>R</i>, so:</p>
<div class="eq">fraction cut ≈ max(0, 1 − grace / R)
  grace 90 s, R = 30 s  → 0          (all finish)
  grace 90 s, R = 180 s → 1 − 0.5 = 0.5</div>
<p>That is why spot replicas often cap <code>max_tokens</code>: short responses always finish inside the window.</p>`,
        legend: [["v", "notice"], ["ok", "finishes"], ["hot", "cut"], ["k", "rerouted"]],
        check: { q: "A spot node gets its notice. Which requests can the gateway save by retrying on another replica?",
          options: ["All of them", "Those that have not received any bytes yet", "Only streams that are more than half done"], answer: 1,
          why: "A request with no bytes sent can be resent elsewhere and the client never knows. Once tokens have streamed, a retry would produce a different answer from the start, so the gateway must not retry it: it either finishes within the grace period or fails." },
        scene(G) {
          const X = (s) => 40 + (s + 60) * 3.15; // -60..120 s
          G.axes(40, 70, 570, 280, {});
          [-60, -30, 0, 30, 60, 90, 120].forEach((s) => G.label(X(s), 372, s + " s", { anchor: "middle", size: 11 }));
          G.line(X(0), 60, X(0), 350, { color: "v", w: 2 }); G.label(X(0) + 4, 56, "notice", { color: "v", size: 11 });
          G.line(X(90), 60, X(90), 350, { color: "ok", dash: "5 4" }); G.label(X(90) - 4, 56, "grace ends", { color: "ok", size: 11, anchor: "end" });
          G.line(X(120), 60, X(120), 350, { color: "hot", w: 2 }); G.label(X(120), 56, "killed", { color: "hot", size: 11, anchor: "end" });
          const reqs = [[-40, 55], [-20, 70], [-55, 130], [-10, 150], [-30, 200]];
          const bars = reqs.map(([st, len], i) => {
            const y = 86 + i * 40, end = Math.min(st + len, 120), ok = st + len <= 90;
            const r = G.rect(X(st), y, X(end) - X(st), 26, { fill: ok ? "ok" : "hot", rx: 4, opacity: 0.9 });
            if (ok) G.label(X(end) + 6, y + 18, "done at " + (st + len) + " s", { size: 11, color: "ok" });
            else G.label(X(st) + 6, y + 18, "still streaming at 120 s: cut", { size: 11, color: "bg" });
            return r;
          });
          G.box(X(0) + 6, 300, 230, 30, "new requests → other replica", { fill: "k", size: 11 });
          G.label(40, 400, "bars = streams already running when the notice arrives", { size: 11 });
          G.from(bars, { attr: { width: 0 }, stagger: 0.12, duration: 0.4 });
          G.caption("drain inside the grace period; only long streams are lost");
        } },

      { rail: "fallback", title: "Fall back to on-demand, and check the bill",
        body: `<p>Spot capacity can also be <i>unavailable</i>: no instances of that type in that zone right now. A replica that can't be placed is a replica you don't have. <b>Karpenter</b> (a node autoscaler: it launches nodes to fit Pending pods) handles this with two <b>NodePools</b> in <code>platform/autoscaler/karpenter-nodepools.yaml</code>:</p>
<ul><li><code>gpu-spot</code>, <code>weight: 100</code>, tried first, allows <code>g6.xlarge</code>, <code>g6.2xlarge</code> and <code>g5.xlarge</code>.</li>
<li><code>gpu-on-demand</code>, <code>weight: 10</code>, used when spot can't launch.</li></ul>
<p>Listing several instance types matters: interruptions and shortages happen per instance pool, so diversity makes it unlikely that one event takes every replica at once.</p>
<p>Is spot still cheaper once you count the work redone after interruptions? P3.5's cost model calls that fraction <code>spot_overhead</code>. With example prices (check your region's pricing page):</p>
<div class="eq">effective spot cost = spot price × (1 + overhead)
spot at 40% of on-demand, 10% redone:
  0.40 × 1.10 = 0.44 of on-demand → still 56% cheaper
spot at 90% of on-demand, 15% redone:
  0.90 × 1.15 = 1.035 → more expensive than on-demand</div>
<p>Note how the discount shrinks: at 60% of on-demand with 20% redone you pay 0.60 × 1.20 = 0.72, so 28% cheaper rather than the headline 40%.</p>`,
        legend: [["ok", "spot"], ["blue", "on-demand"], ["hot", "dearer than on-demand"]],
        scene(G) {
          G.box(230, 30, 180, 46, "Pending GPU pod", { stroke: "ink", size: 13 });
          G.box(40, 130, 250, 150, "", { stroke: "ok" });
          G.text(56, 156, "gpu-spot · weight 100", { color: "ok", size: 13 });
          ["g6.xlarge", "g6.2xlarge", "g5.xlarge"].forEach((t, i) => G.box(56, 172 + i * 34, 140, 26, t, { fill: "ok", size: 11, rx: 4 }));
          G.label(206, 190, "tried first", { size: 11 });
          G.box(350, 130, 250, 150, "", { stroke: "blue" });
          G.text(366, 156, "gpu-on-demand · weight 10", { color: "blue", size: 13 });
          G.box(366, 172, 140, 26, "g6.xlarge", { fill: "blue", size: 11, rx: 4 });
          G.label(366, 226, "used when spot can't launch", { size: 11 });
          const a1 = G.arrow(290, 78, 170, 126, { color: "ok", w: 2 });
          const a2 = G.arrow(350, 78, 470, 126, { color: "blue", w: 1.5, dash: "5 4" });
          G.text(24, 312, "effective spot cost = price × (1 + overhead), on-demand = 1", { size: 13 });
          G.bars(40, 410, [1, 0.44, 0.72, 1.035], { w: 90, gap: 40, h: 60, max: 1.1, fill: (i, v) => (v > 1 ? "hot" : i ? "ok" : "blue") });
          ["on-demand", "0.40 × 1.10", "0.60 × 1.20", "0.90 × 1.15"].forEach((l, i) => G.label(85 + i * 130, 338, l, { anchor: "middle", size: 11 }));
          G.from([a1, a2], { opacity: 0, stagger: 0.5, duration: 0.4 });
          G.caption("spot first, on-demand when spot runs out; check overhead");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Scale LLM servers on the work in the system (running + waiting), not CPU or GPU utilisation, which saturate at one request.</li>
<li><code>desired = ceil(work / target)</code>, with target ≈ 0.8 × the knee you measured in P2.4.</li>
<li>KEDA runs the query and feeds the HPA. Scale up fast and bounded, scale down only after the window's maximum drops.</li>
<li>Scale-to-zero needs a wake-up signal from something always up: the gateway.</li>
<li>A cold start is node + image + weights + warm-up, in sequence. Shrink each; the backlog grows with arrival rate × cold start.</li>
<li>Spot saves money if you drain inside two minutes, retry only unstarted requests, fall back to on-demand and count the replayed work.</li></ul>`,
    sim: {
      title: "Run an autoscaler for two hours",
      intro: "A traffic pattern arrives at the gateway. Every 15 s the policy from <code>policy.py</code> reads running + waiting and sets the replica count; each new pod then needs a cold start before it serves. Change the knobs and press Run. Watch the red area: requests waiting because capacity hasn't arrived yet.",
      height: 300,
      controls: [
        { id: "pat", label: "traffic pattern", type: "select", value: "wave", options: [["wave", "daily wave"], ["spike", "launch spike"], ["bursty", "bursty on/off"]] },
        { id: "svc", label: "time per request, s (example)", min: 5, max: 180, step: 5, value: 20 },
        { id: "knee", label: "replica knee (concurrent requests)", min: 8, max: 64, step: 2, value: 30 },
        { id: "tgt", label: "target per replica (fraction of knee)", min: 0.5, max: 1, step: 0.05, value: 0.8, format: (v) => v.toFixed(2) },
        { id: "cold", label: "cold start, s (measure yours)", min: 15, max: 600, step: 15, value: 240 },
        { id: "win", label: "scale-down window, s", min: 0, max: 900, step: 60, value: 300 },
        { id: "min", label: "min replicas", min: 0, max: 3, value: 0 },
        { id: "max", label: "max replicas", min: 1, max: 10, value: 6 },
        { id: "spot", label: "spot interruption", type: "select", value: "none", options: [["none", "none"], ["70", "one replica reclaimed at minute 70"]] },
      ],
      run: { label: "Run two hours", frames: 40, ms: 50 },
      draw(G, v, t) {
        const dt = 15, T = 7200, N = T / dt, grace = 90;
        const lam = (s) => {
          const m = s / 60;
          if (v.pat === "spike") return m >= 40 && m < 75 ? 6 : 1;
          if (v.pat === "bursty") return Math.floor(m / 15) % 2 ? 0.2 : 3.5;
          return 0.3 + 3.7 * Math.exp(-((m - 60) ** 2) / 600);
        };
        const target = Math.max(1, v.knee * v.tgt);
        const p = { target, min: v.min, max: v.max, cooldown: 600, window: v.win, up: 2, down: 1 };
        const st = newState(v.min);
        let pods = []; for (let i = 0; i < v.min; i++) pods.push(-1e9); // ready times
        let W = 0, gpuS = 0, qS = 0, arr = 0, peakQ = 0, slowS = 0, starts = 0, cut = 0, lastUp = -1e9, lastDown = -1e9;
        const rows = [];
        for (let s = 0; s < T; s++) {
          if (s % dt === 0) {
            // spot interruption: one ready pod is reclaimed
            if (v.spot === "70" && s === 70 * 60) {
              const k = pods.findIndex((r) => r <= s);
              if (k >= 0) { const ready = pods.filter((r) => r <= s).length; const run = Math.min(W, ready * v.knee) / ready; cut += run * Math.max(0, 1 - grace / v.svc); pods.splice(k, 1); st.replicas = pods.length; }
            }
            // policy every 15 s; HPA rate limits: +up per 60 s, -down per 120 s
            const before = st.replicas; let want = polStep(st, s, W, p);
            if (want > before && s - lastUp < 60) want = before;
            if (want < before && s - lastDown < 120) want = before;
            st.replicas = want;
            if (want > pods.length) { for (let k = pods.length; k < want; k++) { pods.push(s + v.cold); starts++; } lastUp = s; }
            if (want < pods.length) { pods.sort((a, b) => b - a); pods = pods.slice(pods.length - want); lastDown = s; }
          }
          const ready = pods.filter((r) => r <= s).length, cap = ready * v.knee;
          const a = lam(s); arr += a; W += a;                       // 1-second fluid step
          const running = Math.min(W, cap), q = Math.max(0, W - cap);
          W -= running / v.svc;
          gpuS += pods.length; qS += q; peakQ = Math.max(peakQ, q); if (q > 0.5) slowS += 1;
          if (s % dt === 0) rows.push({ s, W, cap, q, n: pods.length, ready });
        }
        // chart
        const x0 = 40, x1 = 630, yt = 30, yb = 260, shown = Math.max(1, Math.round(N * t));
        const maxY = Math.max(10, ...rows.map((r) => Math.max(r.W, r.cap))) * 1.1;
        const X = (s) => x0 + ((x1 - x0) * s) / T, Y = (y) => yb - ((yb - yt) * Math.min(y, maxY)) / maxY;
        G.axes(x0, yt, x1 - x0, yb - yt, {});
        [0, 30, 60, 90, 120].forEach((m) => G.label(X(m * 60), yb + 14, m + " min", { anchor: m === 120 ? "end" : "middle", size: 10 }));
        [0, 0.5, 1].forEach((f) => G.label(x0 - 4, Y(maxY * f / 1.1) + 4, F.num((maxY * f) / 1.1), { anchor: "end", size: 10 }));
        const R = rows.slice(0, shown);
        G.path(stepPath(R.map((r) => [X(r.s), Y(r.cap)])), { color: "ok", w: 2 });
        G.path(stepPath(R.map((r) => [X(r.s), Y(r.n * v.knee)])), { color: "ok", w: 1, dash: "3 3" });
        R.forEach((r) => { if (r.q > 0.5) G.rect(X(r.s), Y(r.W), Math.max(1, X(dt) - x0), Y(r.cap) - Y(r.W), { fill: "hot", rx: 0, opacity: 0.55 }); });
        G.path("M " + R.map((r) => `${X(r.s)} ${Y(r.W)}`).join(" L "), { color: "k", w: 2 });
        if (v.spot === "70") { G.line(X(4200), yt, X(4200), yb, { color: "v", dash: "4 3" }); G.label(X(4200) + 4, yt + 10, "spot reclaim", { color: "v", size: 10 }); }
        G.rect(x0 + 6, 8, 10, 10, { fill: "k", rx: 2 }); G.label(x0 + 20, 17, "work (running + waiting)", { size: 11 });
        G.rect(x0 + 200, 8, 10, 10, { fill: "ok", rx: 2 }); G.label(x0 + 214, 17, "ready capacity (pods × knee)", { size: 11 });
        G.rect(x0 + 420, 8, 10, 10, { fill: "hot", rx: 2 }); G.label(x0 + 434, 17, "waiting", { size: 11 });
        const R0 = rows[rows.length - 1];
        const fixed = v.max * T;
        return [
          { title: "User impact", rows: [["peak waiting requests", F.num(peakQ)], ["minutes with a queue", F.num(slowS / 60, 1)], ["avg extra wait per request (Little)", F.ms(arr ? qS / arr : 0)]],
            chip: [slowS < 300, slowS < 300 ? "queue under 5 min in total" : "users waited for capacity"] },
          { title: "GPU bill (2 h)", gauge: [[gpuS / fixed, "ok"]], gaugeText: `autoscaled vs ${v.max} replicas always on`,
            rows: [["GPU-hours used", F.num(gpuS / 3600, 2)], ["GPU-hours, fixed at max", F.num(fixed / 3600, 2)], ["saved", F.num(100 * (1 - gpuS / fixed), 0) + "%"]] },
          { title: "Scaling", rows: [["target per replica", F.num(target, 1)], ["pods started (cold starts)", F.num(starts)], ["replicas at the end", F.num(R0.n)], ["streams cut by spot", F.num(cut, 1)]] },
          { title: "Model", html: `<p class="note">A fluid model in 1-second steps: each ready replica serves up to <i>knee</i> requests at once, each taking the chosen time. Arrivals are smooth, not Poisson. HPA rate limits (+2/min, −1 per 2 min) and the 600 s cooldown are applied. The dashed line counts pods still in their cold start: you pay for those GPU-hours before they serve. Prices are not modelled: GPU-hours are the unit. Use your own cold-start and knee numbers.</p>` },
        ];
      },
    },
    practice: {
      items: [
        { title: "The scaling policy as a pure function", tier: "T0", goal: "Implement <code>raw_desired()</code> and <code>step()</code>: activation from zero, bounded scale-up, windowed scale-down, cooldown, and less flapping than the raw formula.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.6-autoscaling/exercises/test_policy.py" },
        { title: "KEDA ScaledObject on kind", tier: "T0", goal: "Install KEDA v2.21.0 on your kind cluster, scale the mock backend on queue depth, record the timeline and check it scales up then back to zero.",
          cmd: "python course/P3-deployment-and-infra/P3.6-autoscaling/exercises/check_scale_log.py results/p3.6-kind.log --deployment mockllm --expect-zero" },
        { title: "#3 on EKS: load ramp and scale-up timeline", tier: "T3", goal: "Ramp load through the gateway, capture the replica timeline and the cold-start phase table, then apply one mitigation and measure again.",
          cmd: "uv run python -m autoscaler.coldstart --namespace s2s --pod <the pod that scaled up> | tee results/p3.6-coldstart.md" },
        { title: "Spot interruption with zero failed requests", tier: "T0 sim · hard", goal: "Understand why draining saves every request, then hard-kill a streaming backend and add a mitigation that keeps failures within your budget.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.6-autoscaling/exercises/test_spot_sim.py" },
      ],
      labs: [
        { label: "Run all T0 exercises", path: "uv run pytest course/P3-deployment-and-infra/P3.6-autoscaling/exercises" },
        { label: "The policy, ScaledObject, Karpenter NodePools and cold-start tool", path: "platform/autoscaler/" },
        { label: "Kind version with short windows", path: "platform/autoscaler/kind/scaledobject-mock.yaml" },
        { label: "EKS guide (cost, quota, teardown)", path: "course/P3-deployment-and-infra/P3.6-autoscaling/aws.md" },
        { label: "Request path animation, cold vs warm", path: "animations/p3-request-path.html" },
      ],
    },
  });
})();
