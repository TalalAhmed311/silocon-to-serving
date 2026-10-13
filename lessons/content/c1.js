/* C1 — Multi-region failover. Teaches what state matters, RTO/RPO, topologies, hysteresis, authenticated probing,
   DNS caching, the decision policy, G-counter budgets and the game day. Code: platform/failover, infra/aws/failover. */
(function () {
  const F = S2S.fmt;
  const PF = "platform/failover";
  const col = (G, x, y, w, h, label, o) => G.box(x, y, w, h, label, o);

  S2S.lesson({
    id: "c1", n: "C1", title: "Multi-region failover",
    subtitle: "Capstone · first principles · T0 logic, T3 game day (two regions)",
    kicker: "Lesson · ≈ 50 min",
    headline: "When the whole region disappears, what do you still need?",
    intro: `<p>A cloud region is a group of data centres. Sometimes one becomes unreachable: a power failure, a bad network change, an outage of a service everything depends on. A serving platform that lives in one region goes down with it.</p>
<p>This lesson builds a failover design from the problem upward: which state must survive, how fast you must recover (RTO and RPO), how to <i>notice</i> an outage without crying wolf, how DNS moves the traffic, how a controller decides, how token budgets stay correct in two places at once, and how a game day proves it. The code is in <code>platform/failover</code>.</p>`,
    facts: ["9 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["ok", "safe / healthy"], ["hot", "lost / down"], ["v", "time or delay"], ["k", "traffic or probe"], ["q", "decision"]],
    prev: "p6-7", next: "c2",
    steps: [
      { rail: "what dies", title: "Losing a region loses compute, not much data",
        body: `<p>Start with the question that decides the whole design: if everything in one region vanished this second, what would you actually miss?</p>
<p>An LLM serving region is mostly <b>stateless compute on expensive, slow-to-start GPUs</b>. Most of what lives in it can be rebuilt:</p>
<ul><li><b>KV caches and prefix caches</b> sit in GPU memory. They are gone, and the other region just recomputes them: the first requests see a slower time to first token.</li>
<li><b>In-flight requests</b> are lost, but clients retry (the gateway's retry budget from P2.7).</li></ul>
<p>A few things are <i>not</i> rebuildable, and these are the real failover work: <b>tenant API keys</b> (replicate the secret), <b>per-tenant token budgets</b> (or a tenant overspends), the <b>audit log</b> tail that was not yet shipped, and the <b>model weights</b> (they must already be in the other region, or loading them from scratch becomes your recovery time).</p>
<div class="analogy"><b>Picture it</b>A restaurant kitchen burns down. The recipes (weights), the customers' tabs (budgets) and the till receipts (audit log) must exist in another building. The half-chopped onions (caches) do not.</div>`,
        scene(G) {
          const rows = [["KV cache, prefix cache", "recompute", "ok"], ["in-flight requests", "client retries", "ok"], ["model weights", "pre-stage in B", "hot"], ["tenant API keys", "replicate secret", "hot"], ["token budgets", "sync counters", "hot"], ["audit-log tail", "ship often", "hot"]];
          G.text(24, 36, "state in the dying region", { size: 13, color: "muted" }); G.text(330, 36, "needed in region B?", { size: 13, color: "muted" });
          const out = [];
          rows.forEach(([a, b, c], i) => {
            const y = 52 + i * 58;
            const l = G.box(24, y, 220, 42, a, { stroke: "hot", dash: "5 4", size: 13 });
            const ar = G.arrow(250, y + 21, 322, y + 21, { color: "muted" });
            const r = G.box(330, y, 290, 42, c === "ok" ? "no: " + b : "yes: " + b, { fill: c, size: 13 });
            out.push(l, ar, r);
          });
          G.from(out, { opacity: 0, x: -14, stagger: 0.07, duration: 0.3 });
          G.caption("green: disposable · red: must exist in the other region before the failure");
        } },

      { rail: "RTO and RPO", title: "Two numbers define the goal: RTO and RPO",
        body: `<p>"Survive a region failure" is too vague to build or test. Two numbers make it concrete:</p>
<ul><li><b>RTO</b> (recovery time objective): how long users may be without service. It is measured from the failure until requests succeed again.</li>
<li><b>RPO</b> (recovery point objective): how much <i>data</i> you accept losing, measured as time: "the newest copy is at most this old". It is set by how often you copy state out of the region.</li></ul>
<p>For serving, RTO is a sum of delays in a chain, and every link is something you design:</p>
<div class="eq">RTO ≈ detect + decide + DNS change + client caches
      + capacity ready in the other region</div>
<p>RPO matters for the audit log: if you ship it to another region every 60 seconds, you can lose up to 60 seconds of entries. Caches and in-flight requests have no RPO because you never promised to keep them.</p>
<p>Write both targets <i>before</i> you build. A drill can then pass or fail against numbers you chose in advance.</p>`,
        scene(G) {
          const segs = [["detect", "k", 70], ["alarm", "v", 90], ["DNS + caches", "q", 130], ["capacity", "blue", 110]];
          let x = 70; const bars = [];
          G.text(24, 50, "time to recover (illustrative proportions, not measured)", { size: 12, color: "muted" });
          G.line(70, 82, 70, 150, { color: "hot", w: 2 }); G.text(70, 70, "region fails", { anchor: "middle", size: 12, color: "hot" });
          segs.forEach(([t, c, w]) => { bars.push(G.box(x, 96, w - 4, 40, t, { fill: c, size: 12 })); x += w; });
          G.line(x - 4, 82, x - 4, 150, { color: "ok", w: 2 }); G.text(x - 4, 70, "service back", { anchor: "middle", size: 12, color: "ok" });
          G.path(`M 70 166 L 70 176 L ${x - 4} 176 L ${x - 4} 166`, { color: "ink", w: 1.5 });
          G.text((70 + x - 4) / 2, 196, "RTO: how long users wait", { anchor: "middle", size: 14 });
          // RPO row
          G.text(24, 258, "audit log shipped to the other region every N seconds", { size: 12, color: "muted" });
          const ships = [90, 200, 310, 420].map((sx) => G.line(sx, 280, sx, 330, { color: "ok", w: 3 }));
          G.line(70, 305, 540, 305, { color: "line", w: 1.5 });
          G.line(500, 280, 500, 330, { color: "hot", w: 2 }); G.text(500, 272, "failure", { anchor: "middle", size: 12, color: "hot" });
          G.rect(420, 296, 80, 18, { fill: "hot", opacity: 0.45, rx: 3 });
          G.path("M 420 346 L 420 356 L 500 356 L 500 346", { color: "ink", w: 1.5 });
          G.text(460, 376, "RPO: entries not yet copied", { anchor: "middle", size: 14 });
          G.from(bars, { opacity: 0, scaleX: 0, transformOrigin: "left center", stagger: 0.15, duration: 0.3 }); G.from(ships, { opacity: 0, stagger: 0.1, duration: 0.2 });
          G.caption("RTO is a chain of delays; RPO is the age of your newest copy");
        } },

      { rail: "topologies", title: "Active-passive or active-active?",
        body: `<p>Two ways to arrange the regions:</p>
<ul><li><b>Active-passive.</b> Region A serves everyone. Region B is a warm standby: weights are pre-staged and a minimum number of GPU replicas are already running. You pay for those idle GPUs. State has a single writer, so it stays simple.</li>
<li><b>Active-active.</b> Both regions serve traffic all the time. Nothing sits idle, but each region needs <i>headroom</i> to absorb the other's load, and state now has two writers (step 8).</li></ul>
<p>Headroom is easy to get wrong. With <i>N</i> equal regions and traffic split evenly, losing one means each survivor must take <i>N / (N − 1)</i> times its old load. So the busiest you can run normally is:</p>
<div class="eq">max safe utilisation = (N − 1) / N
N = 2:  1/2 = 50%
N = 3:  2/3 = 66.7%
N = 4:  3/4 = 75%</div>
<p>Check it for N = 2: each region at 50% of its capacity; one fails; the survivor carries 50% + 50% = 100%. Any higher and the failover itself overloads the survivor.</p>`,
        check: { q: "Three regions share traffic evenly and each runs at 80% of its capacity. One region fails. What happens to each survivor?",
          options: ["It runs at 80%, nothing changes", "It carries 80% × 3/2 = 120% of its capacity: overloaded", "It carries 80% × 2/3 = 53%"], answer: 1,
          why: "The failed region's 80% is split over two survivors, so each takes 3/2 of its old load: 0.8 × 1.5 = 1.2. The safe maximum for N = 3 is 2/3 (66.7%): 0.667 × 1.5 = 1.0." },
        scene(G) {
          const cap = 150, base = 360;
          const draw = (x, label, util, c1) => { G.rect(x, base - cap, 60, cap, { stroke: "ink", sw: 1.5, rx: 3 }); const f = G.rect(x + 2, base - cap * util, 56, cap * util - 2, { fill: c1, rx: 2, opacity: 0.9 }); G.text(x + 30, base + 18, label, { anchor: "middle", size: 12 }); return f; };
          G.text(24, 50, "active-passive", { size: 14 });
          draw(30, "A", 1, "k"); const pb = draw(110, "B warm", 0.25, "w");
          G.label(24, 100, "B idles at its warm minimum", { size: 11 });
          G.text(300, 50, "active-active, N = 2 at 50%", { size: 14 });
          draw(310, "A", 0.5, "k"); draw(390, "B", 0.5, "k");
          G.line(280, 60, 280, 380, { color: "line", dash: "3 4" });
          G.text(455, 150, "A fails:", { anchor: "middle", size: 13, color: "hot" });
          G.arrow(458, 215, 494, 215, { color: "hot", w: 2 });
          draw(500, "B alone", 1, "ok");
          G.text(530, 200, "100%", { anchor: "middle", size: 13, color: "ok" });
          G.label(24, 410, "survivor load = N/(N−1) × normal · N=3 → ×1.5 · N=4 → ×1.33", { size: 12, color: "ink" });
          G.from(pb, { opacity: 0, duration: 0.5 });
          G.caption("the survivor must absorb the dead region's traffic");
        } },

      { rail: "health signal", title: "Noticing an outage without false alarms",
        body: `<p>Failover begins when something <i>decides</i> a region is down. Probe results are noisy: a single timeout is a network blip, not an outage, and one success in the middle of an outage is luck, not recovery. Acting on every raw result would flap the traffic back and forth, and each move costs errors and cold caches.</p>
<p>The fix is <b>hysteresis</b>: change state only after several consecutive probes agree. <code>health.py</code> uses two thresholds:</p>
<div class="eq">UP   → DOWN after fail_threshold consecutive failures
DOWN → UP   after rise_threshold consecutive successes
detection time ≈ fail_threshold × probe interval</div>
<p>A probe also counts as failed if it is slower than the latency SLO: a region answering in 30 seconds is down for users. Making <code>rise_threshold</code> larger than <code>fail_threshold</code> means the system is quick to give up on a region and slow to trust it again.</p>
<p><b>How many failures?</b> If a probe fails by chance with probability <i>p</i>, a false DOWN needs <i>k</i> failures in a row. The chance that a run of <i>k</i> starts at a given probe is about (1 − <i>p</i>)·<i>p<sup>k</sup></i>. With <i>p</i> = 5% and a probe every 10 s (8,640 probes a day):</p>
<table><tr><th>k</th><th>detection</th><th>false DOWNs per day</th></tr>
<tr><td>1</td><td>10 s</td><td>0.95 × 0.05 × 8,640 ≈ 410</td></tr>
<tr><td>2</td><td>20 s</td><td>0.95 × 0.0025 × 8,640 ≈ 20.5</td></tr>
<tr><td>3</td><td>30 s</td><td>0.95 × 0.000125 × 8,640 ≈ 1.03</td></tr>
<tr><td>5</td><td>50 s</td><td>0.95 × 3.125e-7 × 8,640 ≈ 0.0026 (once per ~390 days)</td></tr></table>
<p>Each extra required failure cuts false alarms about 20× (1 / <i>p</i>) and adds one interval of detection delay.</p>`,
        check: { q: "Probes run every 10 s with 5% random loss. You need detection within 60 s and fewer than one false alarm a month. Which fail_threshold works?",
          options: ["k = 1", "k = 3", "k = 5"], answer: 2,
          why: "k = 5 detects in 50 s and gives about 0.0026 false DOWNs a day (once per ~390 days). k = 3 gives about one false alarm a day; k = 1 gives hundreds. The alarm in the next step adds its own delay on top, so leave margin." },
        scene(G) {
          const N = 34, x0 = 24, pitch = 17.5;
          const ok = (i) => !([5, 11].includes(i) || (i >= 16 && i <= 24));
          G.text(24, 40, "probe results, one every 10 s", { size: 12, color: "muted" });
          for (let i = 0; i < N; i++) G.circle(x0 + 6 + i * pitch, 64, 6, { fill: ok(i) ? "ok" : "hot" });
          G.text(24, 112, "raw: follows every probe", { size: 12, color: "muted" });
          let run = [];
          for (let i = 0; i < N; i++) G.rect(x0 + i * pitch, 120, pitch - 1, 22, { fill: ok(i) ? "ok" : "hot", opacity: 0.8, rx: 2 });
          G.text(24, 188, "hysteresis: 3 failures to go down, 5 successes to come up", { size: 12, color: "muted" });
          for (let i = 0; i < N; i++) { const down = i >= 18 && i < 29; G.rect(x0 + i * pitch, 196, pitch - 1, 22, { fill: down ? "hot" : "ok", opacity: 0.8, rx: 2 }); }
          G.line(x0 + 18 * pitch, 224, x0 + 18 * pitch, 232, { color: "ink", dash: "3 3" });
          G.text(x0 + 18 * pitch - 6, 248, "3rd failure:", { size: 12, anchor: "end" }); G.text(x0 + 18 * pitch - 6, 264, "DOWN", { size: 13, anchor: "end", color: "hot" });
          G.line(x0 + 29 * pitch, 224, x0 + 29 * pitch, 232, { color: "ink", dash: "3 3" });
          G.text(x0 + 29 * pitch + 6, 248, "5th success:", { size: 12 }); G.text(x0 + 29 * pitch + 6, 264, "UP", { size: 13, color: "ok" });
          G.label(24, 120 - 0, "", {});
          G.label(24, 320, "raw state flips 6 times in this window; hysteresis flips twice", { size: 13, color: "ink" });
          G.label(24, 344, "the two single blips (probes 6 and 12) are ignored", { size: 12 });
          G.label(24, 368, "detection = 3 × 10 s = 30 s after the outage starts", { size: 12 });
          G.from(G.svg.querySelectorAll("circle"), { opacity: 0, scale: 0, transformOrigin: "center", stagger: 0.02, duration: 0.2 });
          G.caption("a blip is ignored; a run is believed");
        } },

      { rail: "authenticated probe", title: "Probe like a user, report through the cloud's own signals",
        body: `<p>Where should the probe come from, and what should it test? The obvious option, a public <code>/health</code> URL checked by the DNS provider, has two problems. It forces an endpoint to be reachable from the whole internet, which this course forbids. And a shallow <code>/health</code> can return 200 while the engines behind it are dead.</p>
<p>The design in this repo (<code>prober.py</code>, <code>infra/aws/failover</code>) turns both around:</p>
<ol><li>A <b>prober runs inside each region</b>. Every few seconds it sends a real <b>one-token completion</b> through the gateway using a dedicated probe tenant's API key. If it gets a good answer, the whole path works: gateway, auth, engine, GPU.</li>
<li>It publishes the result (<code>RegionUp</code>, after the hysteresis from step 4) as a <b>CloudWatch metric</b>.</li>
<li>A <b>CloudWatch alarm</b> watches the metric. Missing data counts as <i>breaching</i>: if the prober itself dies, you cannot see the region, so you treat it as down.</li>
<li>A <b>Route 53 health check</b> reads the alarm's state, and the DNS failover record follows the health check.</li></ol>
<p>No unauthenticated endpoint faces the internet: the gateway load balancers only accept your client CIDRs, and every request, including the probe's, needs a key.</p>`,
        scene(G) {
          G.rect(16, 40, 420, 100, { stroke: "muted", dash: "5 4", rx: 10 }); G.label(26, 34, "inside one region", {});
          const a = G.box(30, 70, 100, 50, "prober", { fill: "k", size: 13 });
          const b = G.box(170, 70, 100, 50, "gateway", { fill: "w", size: 13 });
          const c = G.box(310, 70, 110, 50, "engine + GPU", { fill: "w", size: 13 });
          G.arrow(130, 95, 170, 95, { color: "ink" }); G.arrow(270, 95, 310, 95, { color: "ink" });
          G.label(132, 62, "1-token", { size: 10 }); G.label(132, 76, "+ key", { size: 10 });
          G.rect(466, 66, 154, 58, { stroke: "line", rx: 8 }); G.text(543, 92, "probe tenant key", { anchor: "middle", size: 11 }); G.text(543, 108, "in Secrets Manager", { anchor: "middle", size: 11 });
          G.path("M 80 120 L 80 190", { color: "k", w: 2, arrow: true });
          G.box(20, 190, 160, 50, "CloudWatch RegionUp", { fill: "v", size: 11 });
          G.arrow(180, 215, 215, 215, { color: "ink" });
          G.box(215, 190, 100, 50, "alarm", { fill: "q", size: 13 });
          G.arrow(315, 215, 350, 215, { color: "ink" });
          G.box(350, 190, 180, 50, "Route 53 health check", { fill: "blue", size: 11 });
          G.label(215, 262, "no data = breaching = down", { size: 11, color: "hot" });
          G.arrow(440, 240, 440, 300, { color: "ink" });
          G.box(330, 300, 250, 46, "DNS failover record", { stroke: "ink", size: 13 });
          G.text(330, 372, "PRIMARY → region A load balancer", { size: 12, color: "ink" });
          G.text(330, 392, "SECONDARY → region B load balancer", { size: 12, color: "ink" });
          G.label(24, 300, "no public /health endpoint exists", { size: 12, color: "ok" });
          G.label(24, 320, "LBs accept only your client CIDRs", { size: 12, color: "ok" });
          G.from([a, b, c], { opacity: 0, x: -12, stagger: 0.12, duration: 0.3 });
          G.caption("health is a real completion, reported from inside, read by DNS");
        } },

      { rail: "DNS and clients", title: "DNS moves traffic, but not instantly",
        body: `<p>Once Route 53 declares the primary unhealthy, it starts answering DNS queries with the secondary's address. But who asks? Clients and resolvers <b>cache</b> an answer for its <b>TTL</b> (time to live; the Route 53 records in <code>infra/aws/failover</code> default to 30 s, and this lesson's examples use 60 s), and some caches keep it longer. A client that looked up the name 5 seconds before the flip keeps using the old address for another 55 seconds.</p>
<p>Worse, many clients hold a pooled connection that never looks the name up again until it breaks. So the delay users feel is longer than the flip itself:</p>
<div class="eq">worst case after the flip ≈ TTL
    + resolvers that overrun the TTL
    + pooled connections' retry time</div>
<p>That is why you measure the RTO <b>from a client outside both regions</b>, with a loop that logs every request. Measuring from inside the surviving region shows you a number users never see.</p>
<p>DNS is the right tool for <i>region</i> failover. For a single failed <i>backend</i> inside a region, the gateway's own fallbacks (P2.7) react faster than any DNS change can.</p>`,
        check: { q: "Route 53 flipped the record 90 s after the kill, yet users saw errors for 4 minutes. Which is NOT a plausible cause?",
          options: ["Resolvers and clients caching the old answer past the TTL", "Pooled connections still pointing at the dead region's load balancer", "KV caches being copied to the secondary before it can serve"], answer: 2,
          why: "KV caches are never copied; they are recomputed (step 1). The delay comes from DNS caching, pooled connections, and the secondary scaling up for the extra load." },
        scene(G) {
          const x0 = 150, sx = 2.6, flip = 0;
          const t = (s) => x0 + s * sx;            // seconds after the flip -> x
          G.text(24, 36, "seconds after Route 53 flips the record", { size: 12, color: "muted" });
          [0, 60, 120, 180].forEach((s) => { G.line(t(s), 50, t(s), 330, { color: "line", dash: "3 4" }); G.label(t(s), 346, String(s), { anchor: "middle", size: 11 }); });
          G.line(t(0), 46, t(0), 334, { color: "ink", w: 2 }); G.label(t(0) + 4, 62, "flip", { size: 11, color: "ink" });
          const rows = [["resolver A", 55, "cached 5 s before"], ["resolver B", 20, "cached 40 s before"], ["resolver C", 170, "keeps entry 180 s"], ["pooled conn.", 120, "until it breaks"]];
          const out = [];
          rows.forEach(([n, end, note], i) => {
            const y = 82 + i * 58;
            G.text(24, y + 22, n, { size: 13 }); G.label(24, y + 38, note, { size: 10 });
            out.push(G.rect(t(0), y + 4, end * sx, 28, { fill: "hot", opacity: 0.85, rx: 4 }));
            out.push(G.rect(t(end), y + 4, (185 - end) * sx, 28, { fill: "ok", opacity: 0.85, rx: 4 }));
          });
          G.label(24, 390, "red: still sent to the dead region · green: reaches the secondary", { size: 12, color: "ink" });
          G.label(24, 410, "illustrative values (TTL 60 s); the last row depends on the client", { size: 11 });
          G.from(out, { opacity: 0, scaleX: 0, transformOrigin: "left center", stagger: 0.05, duration: 0.3 });
          G.caption("the flip is not the recovery: every cache has its own clock");
        } },

      { rail: "decision policy", title: "Who decides, and when to go back",
        body: `<p>Health signals say what each region looks like. Something still has to <i>choose</i> where traffic goes. <code>policy.py</code> makes this a pure function, <code>decide()</code>, so you can test it with a table. Its rules, in order:</p>
<ol><li><b>Dwell:</b> if we switched less than <code>min_dwell_s</code> ago (120 s by default), do nothing. This stops ping-pong when signals disagree.</li>
<li><b>Fail over</b> if the active region is down <i>and</i> the other is up.</li>
<li><b>Hold</b> if both are down. Moving traffic to another dead region only adds a DNS change to the incident.</li>
<li><b>Fail back</b> to the primary only after it has been continuously up for <code>failback_after_s</code> (600 s by default), or never automatically.</li>
<li>Otherwise: steady.</li></ol>
<p>Why is failing back slower than failing over? A region that just recovered is the one most likely to fail again, and every move costs errors and cold caches. Many teams fail back by hand, in business hours.</p>
<div class="eq">A dies at t=0, B is up      → switch to B at t=30
A recovers at t=200         → since_up[A] = 200
A stays up for 600 s        → fail back at t=800</div>`,
        check: { q: "The active region is down, and the other region is also down. What should the policy do?",
          options: ["Switch anyway: any region is better than a dead one", "Hold: a switch adds a DNS change and sends users to another dead region", "Switch back and forth every probe until one answers"], answer: 1,
          why: "Rule 3 holds. A switch only helps if the target is up. The decide() table test checks this case, and the dwell rule prevents the third option." },
        scene(G) {
          const rules = [["1  inside min_dwell_s?", "hold"], ["2  active down, other up?", "fail over"], ["3  both down?", "hold"], ["4  on secondary, primary stable ≥ failback_after_s?", "fail back"]];
          rules.forEach(([a, b], i) => {
            const y = 44 + i * 52;
            G.box(24, y, 400, 38, a, { stroke: "q", size: 13 });
            G.arrow(424, y + 19, 462, y + 19, { color: "muted" });
            G.box(462, y, 150, 38, b, { fill: b === "hold" ? "w" : "q", size: 13 });
            if (i < 3) G.arrow(224, y + 38, 224, y + 52, { color: "muted" });
          });
          G.label(24, 262, "no rule fired: steady", { size: 12 });
          const x0 = 40, sx = 0.56; const T = (s) => x0 + s * sx;
          G.text(24, 296, "active-region timeline (seconds)", { size: 12, color: "muted" });
          G.rect(T(0), 310, 30 * sx, 22, { fill: "k", opacity: 0.9, rx: 2 }); G.rect(T(30), 310, 770 * sx, 22, { fill: "blue", opacity: 0.9, rx: 2 }); G.rect(T(800), 310, 100 * sx, 22, { fill: "k", opacity: 0.9, rx: 2 });
          G.text(T(0), 350, "A", { size: 12, color: "k" }); G.text(T(400), 350, "B serves", { size: 12, color: "blue", anchor: "middle" }); G.text(T(850), 350, "A", { size: 12, color: "k", anchor: "middle" });
          [[0, "A dies"], [200, "A up"], [800, "fail back"]].forEach(([s, l], i) => { G.line(T(s), 300, T(s), 366, { color: "ink", dash: "3 3" }); G.label(T(s) + (i === 2 ? -4 : 4), 386 + (i % 2) * 0, l, { size: 11, anchor: i === 2 ? "end" : "start" }); });
          G.rect(T(30), 366, 120 * sx, 6, { fill: "v", rx: 2 }); G.label(T(30), 404, "dwell 120 s", { size: 10, color: "v" });
          G.rect(T(200), 366, 600 * sx, 6, { fill: "ok", rx: 2 }); G.label(T(500), 404, "primary must stay up 600 s", { size: 10, color: "ok", anchor: "middle" });
          G.caption("decide() is a pure function: test it with a table");
        } },

      { rail: "budgets", title: "Two regions, one budget",
        body: `<p>In active-active, a tenant with a monthly token budget sends requests to both regions. Each region counts the tokens it served. How do they agree on the total without a slow cross-region lock?</p>
<p>Use a <b>G-counter</b> (grow-only counter), in <code>budgets.py</code>. Keep one counter per (tenant, region). A region only ever increases <i>its own</i> entry. To merge two replicas, take the <b>element-wise maximum</b>. The tenant's usage is the sum over regions.</p>
<div class="eq">A's view: {A: 300, B: 0}
B's view: {A: 0, B: 200}
merge (max per entry): {A: 300, B: 200}  used = 500</div>
<p>Max is commutative, associative and idempotent, so syncs can arrive in any order, twice, or late, and every replica still converges to the same answer.</p>
<p>Convergence is not enough, though. Between syncs each region sees only its own spend. Say 1,000 tokens remain at the last sync. A <b>naive</b> gate ("used + n ≤ budget") lets <i>each</i> region spend the whole 1,000: total 2,000, double the budget. The <b>split</b> gate gives each region <code>remaining / regions</code> = 500 per sync interval, so the total can never exceed 1,000. The price: if 90% of traffic hits region A, then A, demanding 900, is cut off at 500, refusing 400 tokens of requests while the tenant still has budget left globally. Exercise 4 measures this and asks for a better split.</p>`,
        check: { q: "1,000 tokens remain at the last sync and two regions use the split gate. In the next interval region A gets 900 tokens of requests and region B gets 100. How many does A admit, and does the tenant overspend?",
          options: ["900; no overspend", "500; no overspend, but 400 are refused early", "1,000; overspend by 100"], answer: 1,
          why: "Each region's allowance is 1,000 / 2 = 500. A admits 500 and refuses 400, B admits its 100. The total stays under 1,000 (no overspend); the refusals are the price of not coordinating." },
        scene(G) {
          G.text(24, 36, "naive gate vs split gate (1,000 tokens left, 2 regions)", { size: 12, color: "muted" });
          const base = 250, sc = 0.085;                      // 1000 tokens = 85 px
          const colx = [60, 220];
          G.rect(40, base - 1000 * sc, 330, 1000 * sc, { fill: "ok", opacity: 0.12, rx: 0 });
          G.line(40, base - 1000 * sc, 370, base - 1000 * sc, { color: "ok", w: 2, dash: "5 3" }); G.label(380, base - 1000 * sc + 4, "budget 1,000", { size: 11, color: "ok" });
          const b1 = [G.rect(colx[0], base - 1000 * sc, 50, 1000 * sc, { fill: "k", rx: 0 }), G.rect(colx[0], base - 2000 * sc, 50, 1000 * sc, { fill: "hot", rx: 0 })];
          G.text(colx[0] + 25, base + 18, "naive", { anchor: "middle", size: 12 }); G.text(colx[0] + 25, base - 2000 * sc - 8, "2,000", { anchor: "middle", size: 13, color: "hot" });
          G.label(colx[0] + 56, base - 500 * sc, "A: 1,000", { size: 10 }); G.label(colx[0] + 56, base - 1500 * sc, "B: 1,000", { size: 10 });
          const b2 = [G.rect(colx[1], base - 500 * sc, 50, 500 * sc, { fill: "k", rx: 0 }), G.rect(colx[1], base - 1000 * sc, 50, 500 * sc, { fill: "blue", rx: 0 })];
          G.text(colx[1] + 25, base + 18, "split", { anchor: "middle", size: 12 }); G.text(colx[1] + 25, base - 1000 * sc - 8, "1,000", { anchor: "middle", size: 13, color: "ok" });
          G.label(colx[1] + 56, base - 250 * sc, "A: 500", { size: 10 }); G.label(colx[1] + 56, base - 750 * sc, "B: 500", { size: 10 });
          // merge
          G.line(24, 290, 616, 290, { color: "line" });
          G.text(24, 316, "G-counter merge: element-wise max", { size: 12, color: "muted" });
          G.box(24, 328, 170, 40, "A: {A:300, B:0}", { stroke: "k", size: 12 }); G.box(214, 328, 170, 40, "B: {A:0, B:200}", { stroke: "blue", size: 12 });
          G.arrow(204, 380, 204, 392, { color: "muted" });
          G.box(104, 394, 200, 30, "{A:300, B:200}  used = 500", { fill: "ok", size: 12 });
          G.from([...b1, ...b2], { attr: { height: 0 }, duration: 0.5, stagger: 0.1 });
          G.caption("max-merge converges; the split gate keeps the sum within budget");
        } },

      { rail: "game day", title: "Prove it: kill a region and measure",
        body: `<p>A failover design you have not exercised is a hypothesis. A <b>game day</b> is a planned drill: fill in the runbook (<code>runbook.md</code>) <i>before</i>, then kill a region on purpose and record what happens.</p>
<ol><li>Write targets for detection, client-side failover, error rate and budget overspend, plus the abort criteria ("secondary error rate over 20% for 5 minutes: restore the primary").</li>
<li>Start a client <i>outside both regions</i> that logs one JSON line per request: <code>{"t", "end", "ok", "region"}</code>.</li>
<li><code>make drill-kill</code> scales the primary gateway to zero and records <code>.kill_t</code>.</li>
<li>Watch the alarm, the health check, <code>dig</code> answers and the client log. Then <code>make drill-restore</code> and watch failback.</li></ol>
<p><code>gameday.py</code> turns the log into the numbers: <b>detection</b> (kill to health-check flip), <b>first success</b> in the other region, <b>failover time</b> (kill until 20 consecutive successes there) and the <b>error rate</b>. As a worked example (hypothetical, not measured): a client sending 5 requests a second that fails for 120 s loses 5 × 120 = 600 requests; if 300 s of log follow the kill (1,500 requests), the error rate is 600 / 1,500 = 40%.</p>
<p>The deliverable is the filled runbook, the measured table and a "what surprised us" section. Be careful with cost: Route 53 health checks bill monthly, so destroy them afterwards (<code>make down</code>).</p>`,
        scene(G) {
          const ev = [["kill", 0, "hot"], ["first client error", 2, "hot"], ["alarm ALARM", 30, "q"], ["health check flips", 40, "q"], ["DNS answer changes", 55, "blue"], ["first success in B", 70, "ok"], ["20 in a row: failed over", 90, "ok"]];
          const x0 = 40, sx = 5.4;
          G.text(24, 36, "one game-day timeline (positions illustrative, not measured)", { size: 12, color: "muted" });
          G.line(x0, 210, x0 + 100 * sx, 210, { color: "line", w: 2 });
          ev.forEach(([n, s, c], i) => {
            const x = x0 + s * sx, up = i % 2 === 0, ty = up ? 120 - Math.floor(i / 2) * 18 : 254 + Math.floor(i / 2) * 15;
            G.line(x, 210, x, up ? ty + 6 : ty - 12, { color: c, w: 1.5 });
            G.circle(x, 210, 6, { fill: c });
            G.text(Math.min(Math.max(x, 64), 540), ty, n, { size: 11, anchor: "middle", color: c });
          });
          G.rect(x0, 300, 70 * sx, 14, { fill: "hot", opacity: 0.6, rx: 3 }); G.rect(x0 + 70 * sx, 300, 30 * sx, 14, { fill: "ok", opacity: 0.6, rx: 3 });
          G.label(x0, 336, "requests failing  →  requests succeeding in B", { size: 12, color: "ink" });
          G.label(x0, 372, "detection = health flip − kill", { size: 12 }); G.label(x0, 392, "failover = 20th consecutive success in B − kill", { size: 12 });
          G.caption("targets first, drill second, numbers third");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A region failure loses caches and in-flight requests (recomputed, retried) but not keys, budgets, the audit-log tail or weights: those must already be in the other region.</li>
<li>RTO is a chain: detect + decide + DNS + client caches + capacity. RPO is the age of your newest copy of state. Write both before you build.</li>
<li>Active-active needs headroom: N regions can run at most (N − 1) / N full. Active-passive pays for a warm minimum.</li>
<li>Hysteresis gives fast detection without flapping: each extra required failure cuts false alarms by about 1/p.</li>
<li>Probe from inside with a real authenticated completion, publish to CloudWatch, treat missing data as down; keep every endpoint private.</li>
<li>Failing back is slower than failing over. G-counters merge by max; a split gate keeps the total within budget.</li></ul>
<p>Next, C2 turns everything you built in the course into a report a stranger can reproduce.</p>`,
    sim: {
      title: "Failover time budget",
      intro: "Set the knobs of the chain from steps 2 to 6 and see where the recovery time goes. The bar adds the worst case of each stage; the panel also gives the false-alarm rate your hysteresis setting implies.",
      height: 230,
      controls: [
        { id: "iv", label: "probe interval (s)", min: 5, max: 60, step: 5, value: 10 },
        { id: "k", label: "fail_threshold (consecutive failures)", min: 1, max: 8, step: 1, value: 5 },
        { id: "ap", label: "alarm evaluation periods (1 min each)", min: 1, max: 5, step: 1, value: 3 },
        { id: "hc", label: "health-check to DNS delay (s)", min: 0, max: 120, step: 10, value: 30 },
        { id: "ttl", label: "DNS TTL (s)", min: 10, max: 300, step: 10, value: 30 },
        { id: "over", label: "resolvers/clients overrunning the TTL (s)", min: 0, max: 300, step: 10, value: 0 },
        { id: "cap", label: "secondary scale-up time (s)", min: 0, max: 600, step: 30, value: 0 },
        { id: "loss", label: "random probe loss (%)", min: 0, max: 20, step: 1, value: 5 },
        { id: "target", label: "RTO target (s)", min: 60, max: 900, step: 30, value: 300 },
      ],
      draw(G, v) {
        const st = [["detect", v.k * v.iv, "k"], ["alarm", v.ap * 60, "v"], ["health → DNS", v.hc, "q"], ["DNS caches", v.ttl + v.over, "blue"], ["capacity", v.cap, "pink"]];
        const total = st.reduce((a, s) => a + s[1], 0);
        const scale = Math.max(total, v.target) * 1.1, W = 580, x0 = 30;
        const px = (s) => (s / scale) * W;
        let x = x0;
        st.forEach(([n, s, c], i) => { if (s > 0) { G.rect(x, 40, Math.max(1, px(s)), 46, { fill: c, rx: 0 }); if (px(s) > 52) G.text(x + px(s) / 2, 68, F.num(s) + " s", { anchor: "middle", size: 12, color: "bg" }); } x += px(s); });
        G.line(x0 + px(v.target), 28, x0 + px(v.target), 100, { color: total <= v.target ? "ok" : "hot", w: 2.5 });
        G.text(x0 + px(v.target), 20, "target " + v.target + " s", { anchor: "middle", size: 12, color: total <= v.target ? "ok" : "hot" });
        st.forEach(([n, s, c], i) => { const lx = 30 + (i % 3) * 200, ly = 140 + Math.floor(i / 3) * 24; G.rect(lx, ly - 10, 12, 12, { fill: c, rx: 2 }); G.text(lx + 18, ly, `${n}: ${F.num(s)} s`, { size: 12 }); });
        G.label(30, 205, "worst case: stages are added; in a real drill some overlap", { size: 11 });
        const p = v.loss / 100, perDay = (86400 / v.iv) * (1 - p) * Math.pow(p, v.k);
        return [
          { title: "Recovery time (worst case)", rows: [["detect = k × interval", `${v.k} × ${v.iv} = ${v.k * v.iv} s`], ["alarm", `${v.ap} × 60 = ${v.ap * 60} s`], ["DNS", `${v.ttl} + ${v.over} = ${v.ttl + v.over} s`], ["total", `${F.num(total)} s (${(total / 60).toFixed(1)} min)`]],
            gauge: [[Math.min(1, total / v.target), total <= v.target ? "ok" : "hot"]], gaugeText: "share of the RTO target used", chip: [total <= v.target, total <= v.target ? "within target" : "over target"] },
          { title: "False alarms", rows: [["probes per day", F.num(86400 / v.iv)], ["false DOWN per day", perDay >= 0.01 ? perDay.toFixed(2) : perDay.toExponential(1)], ["one every", perDay > 0 ? (perDay >= 1 ? (24 / perDay).toFixed(1) + " h" : (1 / perDay).toFixed(0) + " days") : "never"]],
            chip: [perDay < 1 / 30, perDay < 1 / 30 ? "under one a month" : "too noisy"],
            html: `<p class="note">Model: independent losses, rate ≈ (1 − p) · p<sup>k</sup> per probe. The alarm adds its own hysteresis on top. Health-check delay, scale-up time and overrun are example inputs you must measure.</p>` },
        ];
      },
      note: "Teaching model, not a measurement. The CloudWatch period (60 s per evaluation period) is the common default; check your alarm's settings.",
    },
    practice: {
      intro: `<p>The T0 exercises are pure Python in <code>platform/failover</code> and run on a laptop; the module is already implemented, so read it, predict the result, then run the tests. Exercise 3 needs two AWS regions (T3) and costs money: read <code>infra/aws/failover/README.md</code> for cost, safety and teardown first; plan one 4-hour window. The two EKS stacks dominate the cost and Route 53 health checks bill monthly (look up current prices on the AWS pricing pages). Teardown: <code>make down</code> in <code>infra/aws/failover</code>, then in each region's <code>infra/aws/eks</code>, then check Route 53 for leftover health checks. No endpoint is public: the load balancers accept only your client CIDRs and every request needs a key.</p>`,
      items: [
        { title: "Failover decision logic", tier: "T0 · medium", goal: "Write the decision table for decide() before reading the tests, then extend it with one missing rule (manual pin, capacity check or a switch limit) and test it. Bound the worst-case switches per hour.",
          cmd: "uv run pytest platform/failover -q -k \"decide_table or no_ping_pong\"" },
        { title: "A health check that doesn't flap", tier: "T0 · medium", goal: "Compute detection time and false-down rate for fail_threshold 1, 2, 3, 5 under 5% loss, simulate it, pick values that detect within 60 s and false-alarm under once a month, and add a test.",
          cmd: "uv run pytest platform/failover -q -k \"blips or outage or slow_is_down or flapping\"" },
        { title: "The game day", tier: "T3 · two regions · costs money", goal: "Fill the runbook before the drill, run the outside client, kill the primary, then turn the client log into the timeline table.",
          cmd: "PYTHONPATH=platform uv run python -m failover.gameday client.jsonl --kill-t $(cat infra/aws/failover/.kill_t) --switch-t <health-check-flip-time>" },
        { title: "Active-active budgets", tier: "T0 → T3 · hard", goal: "Measure naive overspend against sync interval and region count, measure the split gate's refusals on 90/10 traffic, then design a split proportional to recent traffic and prove it never overspends.",
          cmd: "uv run pytest platform/failover -q -k \"budget or gcounter\"" },
        { title: "Timeline analysis (supporting test)", tier: "T0 · easy", goal: "See how a client log becomes detection, first-success, failover time and error rate.",
          cmd: "uv run pytest platform/failover -q -k gameday_timeline" },
      ],
      labs: [
        { label: "Module README, exercises and quiz", path: "course/capstone/C1-multi-region-failover/" },
        { label: "Runbook template to fill in before the drill", path: "course/capstone/C1-multi-region-failover/runbook.md" },
        { label: "Decision policy, hysteresis, budgets, prober, game-day analysis", path: PF },
        { label: "Route 53 failover stack (cost, safety and teardown in its README)", path: "infra/aws/failover" },
        { label: "Per-region platform the failover stack points at", path: "infra/aws/eks" },
      ],
    },
  });
})();
