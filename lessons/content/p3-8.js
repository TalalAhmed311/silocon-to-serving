/* P3.8 — Multi-tenancy and security: tenancy models, side channels, keys, quotas and the noisy neighbour, NetworkPolicy, sandboxes, secrets, rotation, audit (#8). */
(function () {
  const F = S2S.fmt;

  /* ---- noisy-neighbour model: two tenants, one backend with S sequence slots, a gateway in front ----
     Tenant A offers `lam` requests/s for 60 s; tenant B sends one short request per second. Each request holds a slot for
     tokens / vel seconds. The gateway can rate-limit A (token bucket), clamp A's max_tokens, and cap A's in-flight requests. */
  const T_END = 60, B_TOK = 32;
  function noisy(v) {
    const arr = [];
    for (let k = 0; k < Math.floor(T_END * v.lam); k++) arr.push([1 + k / v.lam, "A"]);
    for (let k = 0; k < T_END; k++) arr.push([0.5 + k, "B"]);
    arr.sort((a, b) => a[0] - b[0]);
    const free = new Array(v.slots).fill(0), aEnds = [], B = [];
    let bucket = v.rps, last = 0, a429 = 0, aOk = 0, demand = 0;
    const limited = v.rps < 50;                                    // 50 = "no rate limit"
    for (const [t, who] of arr) {
      let dur;
      if (who === "A") {
        if (limited) { bucket = Math.min(v.rps, bucket + (t - last) * v.rps); last = t; if (bucket < 1) { a429++; continue; } }
        if (v.cap > 0 && aEnds.filter((e) => e > t).length >= v.cap) { a429++; continue; }
        if (limited) bucket -= 1;
        dur = Math.min(v.tok, v.clamp > 0 ? v.clamp : 1e9) / v.vel;
      } else dur = B_TOK / v.vel;
      let i = 0; for (let k = 1; k < v.slots; k++) if (free[k] < free[i]) i = k;
      const st = Math.max(t, free[i]); free[i] = st + dur; demand += dur;
      if (who === "A") { aEnds.push(st + dur); aOk++; } else B.push([t, st + dur - t]);
    }
    const l = B.map((x) => x[1]).sort((a, b) => a - b);
    return { B, p90: l[Math.floor(0.9 * (l.length - 1))], base: B_TOK / v.vel, a429, aOk, demand, supply: v.slots * T_END };
  }

  const tag = (G, x, y, w, t, c) => G.box(x, y, w, 26, t, { stroke: c, color: c, size: 11, rx: 13 });

  S2S.lesson({
    id: "p3-8", n: "P3.8", title: "Multi-tenancy and security",
    subtitle: "Deployment and infra · first principles · T0 for everything, T3 optional on EKS",
    kicker: "Lesson · ≈ 50 min",
    headline: "Many customers, one GPU fleet, and no way for one to hurt another",
    intro: `<p>Until now the platform had one user: you. A real fleet serves many <b>tenants</b> (teams or customers) from the same GPUs. That creates risks that a single-user system never had: one tenant slowing the others, one tenant learning something about another, a stolen key, a pod that reaches the cloud's credential service, a log that someone quietly edits. This lesson builds the defences in layers, from first principles: how to pick a tenancy model, the cache timing leak, how API keys should be stored, why limiting requests per second is not enough, default-deny networking, sandboxes for untrusted code, secrets and cloud roles, key rotation with zero downtime, and a tamper-evident audit log. It is project #8 (a stretch goal); everything runs on a laptop.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "tenant request"], ["ok", "allowed"], ["hot", "blocked / risk"], ["v", "control"]],
    prev: "p3-7", next: "p3-9",
    steps: [
      /* 1 */
      { rail: "the problem", title: "Sharing GPUs is cheap, and sharing creates new ways to fail",
        body: `<p>A GPU costs the same whether one customer or ten use it, so sharing is how you make the fleet pay for itself (P3.5's utilisation story). But sharing means a tenant's behaviour reaches the others. Four distinct things can go wrong, and each needs its own control:</p>
<ul><li><b>The noisy neighbour.</b> One tenant sends so much, or such heavy requests, that everyone else's latency collapses. Control: <b>quotas</b>.</li>
<li><b>The leak.</b> One tenant learns something about another through a shared resource, such as the prefix cache. Control: <b>isolation</b> of that resource.</li>
<li><b>The stolen or over-powerful credential.</b> A leaked API key, or a pod whose cloud role can read everything. Control: <b>identity and least privilege</b>, plus <b>network</b> and <b>runtime</b> limits to contain a compromised pod.</li>
<li><b>The unaccountable change.</b> Someone does something and the record is edited afterwards. Control: a <b>tamper-evident audit log</b>.</li></ul>
<p>The design principle is <b>defence in depth</b>: no single control is trusted, each layer assumes the one above it failed. The layers in this repo are identity (keys, IAM), quota (rate, tokens, concurrency), network (NetworkPolicy), runtime (Pod Security, seccomp, sandboxes) and audit.</p>`,
        scene(G) {
          G.text(24, 34, "one fleet, many tenants, four risks", { size: 13 });
          const t = ["tenant A", "tenant B", "tenant C"].map((n, i) => G.box(24, 66 + i * 70, 100, 50, n, { stroke: "k", color: "k", size: 12 }));
          const gw = G.box(190, 100, 120, 110, "gateway", { stroke: "v", color: "v", size: 13 }); G.label(200, 190, "keys · quotas", { size: 11, color: "v" });
          [0, 1, 2].forEach((i) => G.arrow(126, 91 + i * 70, 186, 150, { color: "k", w: 1.5 }));
          G.rect(370, 66, 246, 190, { stroke: "ink", rx: 10 }); G.text(382, 90, "shared GPU pool", { size: 13 });
          [0, 1, 2, 3].forEach((i) => G.box(384 + (i % 2) * 114, 104 + Math.floor(i / 2) * 66, 104, 52, "GPU " + (i + 1), { fill: "w", size: 12 }));
          G.arrow(312, 150, 366, 150, { color: "k", w: 2 });
          const tg = [tag(G, 24, 292, 180, "noisy neighbour", "hot"), tag(G, 214, 292, 200, "cache timing leak", "hot"), tag(G, 24, 326, 180, "stolen credential", "hot"), tag(G, 214, 326, 200, "edited log", "hot")];
          G.label(430, 310, "→ quotas · isolation", { size: 11 }); G.label(430, 330, "→ identity · network", { size: 11 }); G.label(430, 350, "→ runtime · audit", { size: 11 });
          G.label(24, 392, "defence in depth: each layer assumes the one above failed", { color: "ink" });
          G.from(tg, { opacity: 0, y: 10, stagger: 0.15, duration: 0.3 });
          G.caption("sharing pays for the fleet and opens four failure modes");
        } },

      /* 2 */
      { rail: "tenancy models", title: "Three ways to share: a pool, a deployment each, or a hardware slice",
        body: `<p>How much to share is the first decision, and it trades <b>isolation</b> against <b>cost</b>.</p>
<table><tr><th>model</th><th>isolation</th><th>cost</th><th>use when</th></tr>
<tr><td><b>shared pool</b>: one deployment, the gateway enforces quotas</td><td>logical: keys, quotas, separate metrics</td><td>best: bursts share the GPUs</td><td>internal teams, same trust level</td></tr>
<tr><td><b>dedicated deployment</b>: own namespace and pods per tenant</td><td>process, network, scheduling</td><td>worse: each tenant's idle capacity is waste</td><td>a noisy or regulated tenant, a custom model or LoRA set</td></tr>
<tr><td><b>MIG slice</b> per tenant (P4.4)</td><td>hardware partition: memory, cache, SMs</td><td>good for small models</td><td>small models with strict isolation on A100/H100-class GPUs</td></tr></table>
<p>Worked example. Three tenants each generate traffic worth <b>0.1 of a GPU</b> (their load would keep a GPU 10% busy).</p>
<div class="eq">shared pool: 0.1 × 3 = 0.3 GPU of work
  → reserve 1 GPU (30% busy)
dedicated: one GPU each, 10% busy
  → reserve 3 GPUs
MIG: one 1/7 slice each (0.143 ≥ 0.1)
  → reserve 3/7 = 0.43 GPU (70% busy)</div>
<p>Dedicated costs three times the shared pool for the same traffic, and the idle share is pure waste (P3.5). MIG is the middle path: a hard partition at a fraction of the cost, provided the model fits a slice. Many fleets mix: a shared pool by default, a dedicated deployment for the one tenant who needs it.</p>`,
        scene(G) {
          const cols = [["shared pool", "ok", 1, [0.3], "1 GPU · 30% busy"], ["dedicated", "hot", 3, [0.1, 0.1, 0.1], "3 GPUs · 10% busy each"], ["MIG slices", "v", 1, [3 / 7], "3 of 7 slices · 70% busy"]];
          G.text(24, 34, "three tenants, each worth 0.1 GPU of traffic", { size: 13 });
          cols.forEach(([n, c, gpus, _u, txt], i) => {
            const x = 24 + i * 200; G.text(x + 8, 66, n, { size: 14, color: c });
            if (i === 0) { G.rect(x + 8, 80, 150, 150, { stroke: "line", rx: 8 }); G.rect(x + 8, 80 + 150 * 0.7, 150, 150 * 0.3, { fill: "ok", rx: 4, opacity: 0.85 }); G.text(x + 83, 205, "A B C", { anchor: "middle", size: 14, color: "bg" }); }
            if (i === 1) [0, 1, 2].forEach((k) => { G.rect(x + 8 + k * 52, 80, 46, 150, { stroke: "line", rx: 6 }); G.rect(x + 8 + k * 52, 80 + 135, 46, 15, { fill: "hot", rx: 3, opacity: 0.85 }); G.text(x + 31 + k * 52, 222, "ABC"[k], { anchor: "middle", size: 12 }); });
            if (i === 2) { G.rect(x + 8, 80, 150, 150, { stroke: "line", rx: 8 }); for (let k = 0; k < 7; k++) { G.rect(x + 12, 84 + k * 21, 142, 18, { fill: k < 3 ? "v" : "line", rx: 3, opacity: k < 3 ? 0.9 : 0.4 }); if (k < 3) G.text(x + 83, 98 + k * 21, "ABC"[k], { anchor: "middle", size: 12, color: "bg" }); } }
            G.label(x + 8, 256, txt, { size: 11, color: "ink" });
          });
          [["1 GPU", "logical isolation"], ["3 GPUs", "process + network"], ["3/7 ≈ 0.43 GPU", "hardware partition"]].forEach(([a, b], i) => { G.text(32 + i * 200, 296, "reserve " + a, { size: 13, color: "ink" }); G.label(32 + i * 200, 320, b, { size: 11 }); });
          G.label(24, 380, "mix them: pool by default, dedicated or MIG for the one who needs it", {});
          G.caption("the more isolated the tenant, the more idle GPU you pay for");
        } },

      /* 3 */
      { rail: "cache leak", title: "The prefix cache can tell tenant B what tenant A asked",
        body: `<p>Isolation is not only about CPU and memory. In a shared vLLM pool, <b>prefix caching</b> (P2.3) stores the KV blocks of earlier prompts and reuses them for any request that starts with the same tokens, <i>whoever sent it</i>. That is good for speed and bad for secrecy.</p>
<p>Here is the <b>timing side channel</b>. A cache hit skips the prefill, so the first token arrives sooner. Suppose tenant B guesses a prompt prefix (a company's confidential system prompt) and sends it. If the first token comes back unusually fast, B learns that <i>someone already sent that prefix</i>. With example numbers (a 2,000-token prefix: a miss costs about 0.4 s of prefill, a hit about 0.04 s) the difference is easy to measure, and B can repeat the guess for different prefixes.</p>
<p>Options, from cheapest to strongest:</p>
<ul><li><b>Per-tenant cache salting</b>: mix a tenant-specific value into the hash that names each KV block, so identical text from two tenants never shares blocks. vLLM supports a per-request salt (its <code>cache_salt</code> field; <b>UNVERIFIED for your version</b>, check the docs at the pinned SHA).</li>
<li><b>Turn prefix caching off</b> for mixed-trust traffic. Costs the hit rate from P3.4.</li>
<li><b>Separate deployments</b> per tenant. Strongest, most expensive (step 2).</li></ul>
<p>The general lesson: any resource that is shared and observable (a cache, a queue, a timing) can carry information between tenants.</p>`,
        scene(G) {
          G.text(24, 34, "tenant B guesses tenant A's secret prefix (example timings)", { size: 13 });
          G.box(24, 60, 110, 44, "tenant A", { stroke: "k", color: "k" }); G.box(24, 150, 110, 44, "tenant B", { stroke: "hot", color: "hot" });
          G.rect(240, 60, 200, 134, { stroke: "k", rx: 10, dash: "5 4" }); G.text(340, 84, "shared prefix cache", { anchor: "middle", size: 12, color: "k" });
          const blocks = [0, 1, 2, 3].map((i) => G.rect(256 + i * 44, 100, 36, 28, { fill: "k", rx: 3 }));
          G.label(256, 150, "A's secret prefix is cached", { size: 11 });
          G.arrow(136, 82, 236, 100, { color: "k", w: 2 }); G.arrow(136, 172, 236, 128, { color: "hot", w: 2 });
          G.rect(24, 236, 592, 164, { stroke: "line", rx: 10 });
          G.text(40, 262, "B's guess is the cached prefix: hit", { size: 12, color: "ok" }); G.rect(330, 248, 20, 18, { fill: "ok", rx: 3 }); G.text(360, 262, "0.04 s", { size: 12 });
          G.text(40, 296, "B's guess is a new prefix: miss", { size: 12, color: "hot" }); G.rect(330, 282, 200, 18, { fill: "hot", rx: 3 }); G.text(540, 296, "0.4 s", { size: 12 });
          G.text(40, 336, "fast answer = someone already sent it: a leak", { size: 13, color: "hot" });
          G.label(40, 364, "fixes: per-tenant cache salt · no cross-tenant caching ·", {}); G.label(40, 384, "dedicated deployments (cache_salt: check your vLLM version)", {});
          G.from(blocks, { opacity: 0, x: -12, stagger: 0.1, duration: 0.3 });
          G.caption("shared and observable means it can carry a message");
        } },

      /* 4 */
      { rail: "keys", title: "Identity: keys you can check but never read back",
        body: `<p>The gateway must know <i>who</i> is calling. The simplest identity is a per-tenant <b>API key</b>, a secret string sent in the <code>Authorization</code> header. The design question is how the gateway stores them.</p>
<p>If the config holds the keys, anyone who reads a leaked config (a Git history, a backup, a log) can impersonate every tenant. So store only a <b>hash</b>: <code>key_sha256 = SHA-256(key)</code>. The gateway hashes the incoming key and compares. A leaked config does not reveal any usable key.</p>
<p>Passwords need a deliberately <i>slow</i> hash. Why is a fast SHA-256 fine here? Because of entropy. <code>platform/tenancy/keys.py</code> makes keys with <code>secrets.token_urlsafe(32)</code>: 32 random bytes, <b>256 bits</b>, with a recognisable prefix <code>s2s_</code> so leaked keys are easy to find with secret scanners. Compare the guessing effort at an example attacker speed of 10¹⁰ hashes per second:</p>
<div class="eq">8 lowercase letters: 26^8 = 2.1 × 10^11
  → 2.1e11 ÷ 1e10 ≈ 21 seconds
random 256-bit key: 2^256 ≈ 1.2 × 10^77
  → 1.2e77 ÷ 1e10 ≈ 1.2 × 10^67 seconds
  (the universe is about 4 × 10^17 s old)</div>
<p>A human password is guessable, so it needs a slow hash to make each guess expensive. A 256-bit random key is not guessable at any hash speed. Keys are shown once at creation, delivered through the secret store (never chat or email), and rotated (step 9).</p>`,
        check: { q: "Why is a fast hash like SHA-256 acceptable for storing these API keys, when it would be a poor choice for user passwords?",
          options: ["The keys are short, so the hash is quick anyway", "The keys are random 256-bit values, so brute force is infeasible whatever the hash speed; passwords are low-entropy and need a slow hash", "SHA-256 can be reversed, which the gateway needs"], answer: 1,
          why: "Security of a stored hash depends on how many guesses an attacker needs. A human password has few possibilities (about 2^37 for 8 lowercase letters), so a slow hash is needed. A random 256-bit key has 2^256, which no hash speed makes feasible. SHA-256 is one-way, so a leaked config does not give usable keys." },
        scene(G) {
          G.text(24, 34, "how a key is made, stored and checked", { size: 13 });
          const a = G.box(24, 60, 150, 44, "1  generate", { stroke: "k", color: "k", size: 13 });
          const b = G.box(244, 60, 150, 44, "2  hand over once", { stroke: "v", color: "v", size: 13 });
          const c = G.box(464, 60, 152, 44, "3  store the hash", { stroke: "ok", color: "ok", size: 13 });
          G.arrow(176, 82, 240, 82, { color: "muted", w: 2 }); G.arrow(396, 82, 460, 82, { color: "muted", w: 2 });
          G.label(24, 122, "256 random bits, s2s_ prefix", { size: 11 }); G.label(244, 122, "via the secret store", { size: 11 }); G.label(464, 122, "config: SHA-256(key) only", { size: 11 });
          G.rect(24, 140, 592, 70, { stroke: "line", rx: 10 }); G.text(40, 166, "request: Authorization: Bearer s2s_…", { size: 12 });
          G.text(40, 192, "gateway: sha256(incoming) == key_sha256 ?  yes → tenant A", { size: 12, color: "ok" });
          G.rect(24, 246, 592, 154, { stroke: "line", rx: 10 }); G.text(40, 270, "guessing, at 10^10 hashes per second (example)", { size: 13, color: "ink" });
          G.rect(40, 286, 12, 22, { fill: "hot", rx: 2 }); G.text(60, 302, "8-letter password: 21 s", { size: 12, color: "hot" });
          G.rect(40, 322, 480, 22, { fill: "ok", rx: 2, opacity: 0.8 }); G.text(60, 338, "256-bit key: 10^67 s", { size: 12, color: "bg" });
          G.label(40, 372, "universe age ≈ 4 × 10^17 s: the bar is not to scale, nothing could be", {}); G.label(40, 390, "entropy decides the hash: random keys fast, passwords slow", { color: "ink" });
          G.from([a, b, c], { opacity: 0, y: -10, stagger: 0.2, duration: 0.3 });
          G.caption("a leaked config reveals no usable key");
        } },

      /* 5 */
      { rail: "quotas", title: "Limit work, not requests: the noisy neighbour",
        body: `<p>The #6 gateway (P2.7) limits each tenant's request rate with a <b>token bucket</b>: a bucket holds up to <i>burst</i> tokens and refills at <i>rps</i> per second; each request takes one, and with none left the gateway answers <b>429</b>. This stops the classic flood. Exercise 4's setup: backend with 4 sequence slots (the number of requests it decodes at once), tenant A flooding 100 concurrent requests with <code>rps: 5, burst: 5</code>. About 95 of the 100 get a 429, tenant B (one short request per second) stays near its solo latency.</p>
<p>But a requests-per-second limit counts <i>arrivals</i>, not <i>work</i>. Let tenant A stay <b>inside</b> 5 rps and ask for <code>max_tokens: 2000</code>. At an example 100 tokens/s per sequence a request holds a slot for 20 s. By Little's law (P3.6) the slots A needs are arrival rate × time in system:</p>
<div class="eq">slots A wants = 5 req/s × 20 s = 100 slots
slots that exist = 4
→ all 4 are taken by A; B waits in the queue</div>
<p>The remedies, in increasing effort (exercise 4): clamp <code>max_tokens</code> per tenant; give each tenant a <b>concurrency limit</b> (a semaphore per tenant: at most <i>C</i> requests in flight, the rest queue or get 429); or weighted fair queuing in front of the backend. Concurrency limits are the direct fix because they cap exactly the resource being fought over: slots. Limit tokens as well as requests, and cap concurrency per tenant. Try it in the simulator.</p>`,
        check: { q: "Tenant A stays under its 5 requests/second limit but every request asks for 2,000 tokens, and tenant B's latency explodes. What control addresses the cause?",
          options: ["Lower A's requests-per-second limit slightly", "A per-tenant concurrency limit (or max_tokens cap) so A cannot hold all the sequence slots", "Add more replicas of the gateway"], answer: 1,
          why: "The resource A is exhausting is sequence slots, held for the duration of long generations. Requests per second does not measure that. Capping A's in-flight requests (or clamping max_tokens) bounds the slots A can occupy, leaving the rest for B." },
        scene(G) {
          G.text(24, 34, "4 sequence slots; A's requests are 2,000 tokens (20 s each, example)", { size: 12 });
          const row = (y, title, slots, color) => {
            G.text(24, y, title, { size: 13, color });
            slots.forEach((s, i) => G.box(24 + i * 98, y + 12, 90, 44, s[0], { fill: s[1], size: 12 }));
          };
          row(70, "rps limit only: A needs 5 × 20 = 100 slots", [["A 20 s", "k"], ["A 20 s", "k"], ["A 20 s", "k"], ["A 20 s", "k"]], "hot");
          G.box(430, 82, 186, 44, "B waits in the queue", { stroke: "hot", color: "hot", size: 12 });
          row(190, "concurrency cap 2 for A", [["A", "k"], ["A", "k"], ["free for B", "ok"], ["free for B", "ok"]], "ok");
          G.box(430, 202, 186, 44, "B served at once", { stroke: "ok", color: "ok", size: 12 });
          G.rect(24, 300, 592, 100, { stroke: "line", rx: 10 });
          G.text(40, 326, "slots in use = arrival rate × time held (Little's law)", { size: 13, color: "ink" });
          G.label(40, 352, "rps limit counts arrivals; the scarce resource is slots × seconds", {});
          G.label(40, 374, "fixes: per-tenant concurrency cap · max_tokens clamp · fair queueing", { color: "ok" });
          G.caption("limit what is scarce: concurrent slots, not just arrivals");
        } },

      /* 6 */
      { rail: "network", title: "Default-deny networking, in both directions",
        body: `<p>By default every pod in a Kubernetes cluster can talk to every other pod and to the internet. If one pod is compromised, so is everything it can reach. A <b>NetworkPolicy</b> is a firewall rule for pods. The strategy is <b>default deny, then allow only the paths the platform needs</b>. From <code>platform/tenancy/networkpolicies.yaml</code>:</p>
<ul><li><code>default-deny</code>: selects every pod and blocks all <b>ingress</b> (traffic in) and <b>egress</b> (traffic out).</li>
<li><code>allow-dns</code>: everyone may reach cluster DNS on port 53.</li>
<li><code>gateway</code>: ingress only from the ingress controller's namespace on 9000; egress only to pods labelled <code>s2s/role: backend</code> on 8000 and 8001.</li>
<li><code>backends</code>: ingress only from the gateway and from Prometheus; egress only to port 443, excluding <code>169.254.169.254/32</code> (the cloud's instance metadata service, <b>IMDS</b>) and the private ranges.</li></ul>
<p>Why <b>egress</b> too? Ingress rules stop strangers reaching the backends. Egress rules stop a <i>compromised</i> pod from calling out: stealing the node's cloud credentials from IMDS, sending data to the internet, or probing other internal services. Blocking IMDS matters because that address hands out the node role's credentials to any process that can reach it.</p>
<p>Two practical traps. First, a policy does nothing on a network plugin (CNI) that does not enforce it, and everything stays allowed without any error. <code>np_probe.sh</code> checks it live: expect <code>gateway → mockllm: allowed</code>, <code>stray pod → mockllm: blocked</code>, <code>mockllm → 169.254.169.254: blocked</code>. If all are allowed, your CNI ignores policies. Second, default-deny ingress alone leaves egress open.</p>`,
        check: { q: "What does default-deny egress protect against that default-deny ingress does not?",
          options: ["Strangers connecting to the backends", "A compromised pod calling out: stealing credentials from the metadata service, exfiltrating data, or reaching other internal services", "Slow image pulls"], answer: 1,
          why: "Ingress rules control who can connect to a pod. Once a pod is compromised, the attacker uses its outbound connections. Egress rules (with the metadata address blocked) cut off credential theft and data exfiltration." },
        scene(G) {
          const n = (x, y, w, t, c) => G.box(x, y, w, 44, t, { stroke: c, color: c, size: 12 });
          n(24, 60, 122, "ingress ctrl", "k"); n(24, 150, 122, "Prometheus", "v"); n(24, 240, 122, "stray pod", "hot");
          n(240, 60, 122, "gateway", "ok"); n(470, 60, 146, "backend (vLLM)", "ok");
          n(430, 232, 150, "IMDS 169.254…", "hot"); n(500, 310, 116, "internet :443", "muted");
          G.arrow(148, 82, 236, 82, { color: "ok", w: 2 }); G.label(170, 74, ":9000", { size: 10, color: "ok" });
          G.arrow(364, 82, 466, 82, { color: "ok", w: 2 }); G.label(392, 74, ":8000", { size: 10, color: "ok" });
          G.path("M 148 172 L 420 172 L 420 98 L 466 98", { color: "ok", w: 1.5, dash: "4 3", arrow: true }); G.label(190, 164, "scrape :8000", { size: 10, color: "ok" });
          G.line(148, 262, 380, 262, { color: "hot", w: 2, dash: "3 3" }); G.text(392, 270, "✕", { size: 20, color: "hot" }); G.label(160, 254, "stray pod → backend: blocked", { size: 10, color: "hot" });
          G.arrow(500, 106, 500, 228, { color: "hot", w: 2, dash: "3 3" }); G.text(494, 170, "✕ IMDS blocked", { size: 11, color: "hot", anchor: "end" });
          G.arrow(600, 106, 600, 306, { color: "muted", w: 1.5 }); G.label(540, 296, "443 only", { size: 10 });
          G.label(24, 350, "default-deny both ways · then allow each needed path", { color: "ink" });
          G.label(24, 374, "egress closed: a hijacked pod cannot call out for credentials", {});
          G.label(24, 396, "check with a live probe: the CNI must enforce policies", {});
          G.caption("every arrow is an explicit rule; the rest is denied");
        } },

      /* 7 */
      { rail: "sandboxes", title: "Untrusted code needs a real sandbox. seccomp is not one",
        body: `<p>If your platform runs <b>model-generated code</b> (an agent's tool call, a code interpreter), treat that code as hostile: a user can steer the model to write anything. A normal container is not enough, because a container is an ordinary process on the <b>shared host kernel</b> (P3.1). A kernel bug is an escape route.</p>
<p>The hardening layers, from weakest to strongest:</p>
<ol><li><b>Pod Security "restricted"</b>: non-root, read-only root filesystem, all Linux capabilities dropped, no privilege escalation (P3.3's <code>policy.py</code> checks these).</li>
<li><b>seccomp</b>: a filter on <b>system calls</b> (the requests a process makes to the kernel). <code>platform/tenancy/seccomp-inference.json</code> is a deny-list for inference pods. It narrows the kernel surface, but the code still talks to the host kernel. Not a sandbox.</li>
<li><b>gVisor</b> (<code>runsc</code>, a Kubernetes RuntimeClass): a kernel written in user space that intercepts the container's system calls, so the host kernel sees far fewer of them.</li>
<li><b>Firecracker microVMs</b>: each workload gets its own small virtual machine with its own kernel, behind hardware virtualisation.</li></ol>
<p>Whichever you choose, also give the sandbox <b>no credentials</b>, and let its <b>egress</b> go only through an allow-listed proxy (step 6). Isolation that leaves a cloud role attached is only half done.</p>`,
        scene(G) {
          G.text(24, 34, "from model-generated code down to the hardware", { size: 13 });
          const L = [["untrusted code", "hot", 0], ["Pod Security + seccomp: narrows syscalls, same kernel", "w", 1], ["gVisor: a user-space kernel intercepts syscalls", "v", 2], ["Firecracker: own microVM and kernel", "q", 3], ["host kernel · hardware", "muted", 4]];
          const g = L.map(([t, c, i]) => { const y = 56 + i * 62, gg = G.group(); G.box(24 + i * 14, y, 592 - i * 28, 48, t, c === "muted" ? { stroke: "muted", color: "muted", size: 12, parent: gg } : { fill: c, size: 12, parent: gg }); return gg; });
          G.arrow(320, 106, 320, 116, { color: "muted" });
          G.rect(24, 372, 592, 42, { stroke: "hot", rx: 8, dash: "5 4" });
          G.text(36, 398, "plus: no credentials in the sandbox · egress only via an allow-list proxy", { size: 12, color: "hot" });
          G.from(g, { opacity: 0, y: -12, stagger: 0.15, duration: 0.3 });
          G.caption("a container shares the host kernel; stronger boundaries add a kernel layer");
        } },

      /* 8 */
      { rail: "secrets & roles", title: "Secrets reach the pod at run time, through a role that sees only one",
        body: `<p>Your platform holds secrets: the Hugging Face token, backend API keys, the audit-log signing key. Three rules cover them.</p>
<ol><li><b>Never in the repo, a ConfigMap, a Terraform default, a Helm values file or an image layer</b> (P3.1 step 5: a deleted file survives in an earlier layer). <code>policy.py</code> enforces part of this with a <code>secretKeyRef</code> rule.</li>
<li><b>Store them in a secret store</b> (AWS Secrets Manager or SSM Parameter Store as a SecureString). They reach the pod at <b>run time</b> through the Secrets Store CSI driver or an External Secrets operator, as an environment variable or a mounted file.</li>
<li><b>Least privilege through workload identity.</b> On EKS, <b>IRSA</b> (IAM Roles for Service Accounts) or <b>EKS Pod Identity</b> lets one Kubernetes ServiceAccount assume one IAM role. Give each workload its <b>own</b> role, scoped to what it needs: read-only on one S3 prefix (P3.7), or read on one secret. One shared role for every pod means one compromise reaches everything.</li></ol>
<p>The vLLM Deployment in <code>platform/deploy/base/vllm.yaml</code> shows the pattern: the API key comes from a <code>secretKeyRef</code> to a Secret that is created out-of-band, never committed. If you repeat the IRSA checks on EKS (optional, T3), use <code>make up</code> and <code>make down</code>: you pay for the control plane, the NAT gateway and any GPU nodes while they exist (look up your region's prices), the GPU pool is min 0, the budget alarm in <code>infra/aws/guardrails</code> is the backstop, and the cluster exposes no public endpoint (use <code>kubectl port-forward</code>).</p>`,
        scene(G) {
          const a = G.box(24, 60, 150, 56, "Secrets Manager / SSM", { stroke: "v", color: "v", size: 11 });
          const b = G.box(244, 60, 150, 56, "CSI / External Secrets", { stroke: "k", color: "k", size: 11 });
          const c = G.box(464, 60, 152, 56, "pod: env var or file", { stroke: "ok", color: "ok", size: 11 });
          G.arrow(176, 88, 240, 88, { color: "v", w: 2 }); G.arrow(396, 88, 460, 88, { color: "k", w: 2 });
          G.text(24, 158, "one ServiceAccount → one IAM role → one resource", { size: 13, color: "ink" });
          const sa = G.box(24, 176, 150, 46, "SA: vllm", { stroke: "k", color: "k", size: 11 });
          const rl = G.box(244, 176, 150, 46, "IAM role: S3 read", { stroke: "v", color: "v", size: 11 });
          const rs = G.box(464, 176, 152, 46, "s3://registry/models/", { stroke: "ok", color: "ok", size: 11 });
          G.arrow(176, 199, 240, 199, { color: "k", w: 2 }); G.arrow(396, 199, 460, 199, { color: "v", w: 2 }); G.label(250, 238, "IRSA / Pod Identity", { size: 10 });
          G.rect(24, 262, 592, 138, { stroke: "hot", rx: 10, dash: "5 4" }); G.text(40, 286, "never put a secret in", { size: 13, color: "hot" });
          ["the Git repo", "a ConfigMap", "a Terraform default", "a Helm values file", "an image layer"].forEach((t, i) => G.box(40 + (i % 3) * 190, 300 + Math.floor(i / 3) * 44, 176, 34, "✕ " + t, { stroke: "hot", color: "hot", size: 11 }));
          G.from([a, b, c, sa, rl, rs], { opacity: 0, y: -8, stagger: 0.1, duration: 0.25 });
          G.caption("secrets arrive at run time; each workload gets a role of its own");
        } },

      /* 9 */
      { rail: "rotation", title: "Rotate a key without anyone noticing",
        body: `<p>Keys leak, staff leave, policies expire. You must be able to replace a key without a moment when the tenant's working key is rejected. The trick is to let the gateway accept <b>two</b> keys for a while: <code>key_sha256</code> (the new one) and <code>previous_key_sha256</code> (a list of old ones). <code>platform/tenancy/keys.py</code> implements it with <code>start_rotation()</code> and <code>finish_rotation()</code>.</p>
<div class="eq">phase 1: key_sha256 = H(new)
         previous_key_sha256 = [H(old)]
         roll the gateway (maxUnavailable 0)
         deliver the new key via the secret store
         tenant switches after the rollout
phase 2: when metrics show the old key unused:
         previous_key_sha256 = []   → roll again</div>
<p>Why this is safe. During the rollout some gateway replicas still have the old config (they accept only the old key) and some have the new (they accept both). A tenant still sending the <i>old</i> key works on every replica, old or new. Only after the rollout completes can the tenant switch to the new key, because then every replica accepts it. The deployment's <code>maxUnavailable: 0</code> and readiness probes (<code>platform/deploy/base/gateway.yaml</code>) ensure capacity never dips while pods are replaced.</p>
<p>When may you do phase 2? When you can <b>prove</b> the old key is no longer used, not when you hope so. Exercise 2 adds a <code>key_generation</code> label (<code>current</code> or <code>previous</code>) to <code>gateway_requests_total</code>, so a PromQL query can show zero requests with <code>previous</code> over a quiet period. The same two-phase pattern rotates backend keys and the audit HMAC key (record which key generation signed each record).</p>`,
        check: { q: "When is it safe to remove the old key's hash from previous_key_sha256 (phase 2)?",
          options: ["Immediately after rolling out the new config", "Once metrics show no requests are still using the old key", "When the tenant says it has probably switched"], answer: 1,
          why: "Removing it earlier risks rejecting a tenant that has not switched yet, which is exactly the downtime rotation is meant to avoid. A per-generation request counter turns 'probably unused' into evidence." },
        scene(G) {
          const x0 = 150, X = (s) => x0 + s * 4.6;
          G.text(24, 34, "time →  (phases not to scale)", { size: 13 });
          const row = (y, label, segs) => { G.label(24, y + 22, label, { size: 11, color: "ink" }); segs.forEach(([a, b, c, t]) => { G.rect(X(a), y, X(b) - X(a), 30, { fill: c, rx: 4 }); G.text(X(a) + 6, y + 20, t, { size: 11, color: "bg" }); }); };
          row(64, "gateway 1", [[0, 20, "w", "old only"], [20, 80, "ok", "old + new"], [80, 100, "k", "new only"]]);
          row(112, "gateway 2", [[0, 30, "w", "old only"], [30, 80, "ok", "old + new"], [80, 100, "k", "new only"]]);
          row(176, "tenant sends", [[0, 40, "v", "old key"], [40, 100, "q", "new key"]]);
          G.line(X(40), 50, X(40), 224, { color: "hot", dash: "4 3" }); G.label(X(40) + 4, 244, "rollout done: tenant switches", { size: 11, color: "hot" });
          G.line(X(80), 50, X(80), 224, { color: "ok", dash: "4 3" }); G.label(X(80) - 4, 262, "old key unused (metrics): phase 2", { size: 11, color: "ok", anchor: "end" });
          G.rect(24, 290, 592, 112, { stroke: "line", rx: 10 });
          G.text(40, 316, "every request is accepted at every moment", { size: 13, color: "ok" });
          G.label(40, 342, "a replica on the new config accepts both keys", {});
          G.label(40, 364, "the tenant moves only after every replica has rolled", {});
          G.label(40, 386, "phase 2 waits for evidence: requests by key generation = 0", { color: "ink" });
          G.caption("overlap the old and new key; remove the old one only when unused");
        } },

      /* 10 */
      { rail: "audit log", title: "A log that shows if it was edited",
        body: `<p>When something goes wrong you need a record of <b>who did what, and with what outcome</b>. If the service that writes the log can also rewrite it, the log proves little. A <b>hash chain</b> makes tampering visible. <code>platform/tenancy/audit.py</code> stores, for each record, <code>prev</code> (the previous record's hash) and <code>hash = HMAC-SHA256(key, prev ‖ canonical(record))</code>. An HMAC is a hash that needs a secret key to compute.</p>
<p>What each tamper does to the chain (the verifier you write in exercise 3 reports these):</p>
<table><tr><th>tamper</th><th>what verify sees</th></tr>
<tr><td>edit a field of record 7</td><td>hash mismatch at line 7: the recomputed hash differs from the stored one</td></tr>
<tr><td>delete record 7</td><td>broken link at line 7: its <code>prev</code> no longer matches the record before it</td></tr>
<tr><td>swap records 7 and 8</td><td>broken link at line 7</td></tr>
<tr><td>truncate the last 5 records</td><td><b>nothing</b>: the remaining chain is internally consistent</td></tr>
<tr><td>rewrite everything with another key</td><td>mismatch at line 1 when verified with the right key</td></tr></table>
<p>The truncation hole is why you also <b>checkpoint</b>: periodically copy the latest hash somewhere the writer cannot change, such as S3 with Object Lock or a CloudWatch Logs group in a separate account. Verify against it and a missing tail shows up as "checkpoint hash not found". On AWS, the gateway's role must <b>not</b> have permission to delete or overwrite that checkpoint object (no <code>s3:DeleteObject</code>, no ability to shorten the retention).</p>
<p>Log who (tenant, key-hash prefix), what (action, model, backend), outcome and token counts. <b>Never</b> prompts, completions or full keys: the audit log would become the most sensitive data you hold.</p>`,
        check: { q: "A hash chain on its own does not detect one kind of tampering. Which, and what closes the gap?",
          options: ["Editing a record: close it with a longer hash", "Truncating the end of the log: close it by storing the latest hash somewhere the writer cannot modify", "Reordering records: close it by sorting by timestamp"], answer: 1,
          why: "Deleting the last records leaves a valid shorter chain, because nothing after them refers back. Only an independent copy of the latest hash (an Object Lock bucket, another account) reveals that the log used to be longer." },
        scene(G) {
          G.text(24, 34, "each record commits to the one before it", { size: 13 });
          const xs = [24, 176, 328, 480], recs = ["rec 1", "rec 2", "rec 3", "rec 4"];
          const bx = xs.map((x, i) => { const g = G.group(); G.rect(x, 60, 136, 100, { stroke: "k", rx: 8, parent: g }); G.text(x + 10, 82, recs[i], { size: 13, color: "k", parent: g }); G.label(x + 10, 106, "prev = h" + i, { size: 11, parent: g }); G.text(x + 10, 134, "hash = h" + (i + 1), { size: 12, color: "ok", parent: g }); return g; });
          [0, 1, 2].forEach((i) => G.arrow(xs[i] + 138, 110, xs[i + 1] - 2, 110, { color: "ok", w: 2 }));
          G.label(24, 182, "h(i) = HMAC(key, h(i−1) ‖ record i)", { size: 12, color: "ink" });
          G.rect(24, 206, 592, 196, { stroke: "line", rx: 10 });
          [["edit rec 2", "hash mismatch at line 2", "hot"], ["delete rec 2", "broken link at line 2", "hot"], ["swap 2 and 3", "broken link at line 2", "hot"], ["cut the tail", "invisible alone → need an off-box checkpoint", "v"]].forEach(([a, b, c], i) => { G.text(40, 234 + i * 30, a, { size: 12, color: "ink" }); G.text(190, 234 + i * 30, "→ " + b, { size: 12, color: c }); });
          G.label(40, 358, "checkpoint = latest hash stored where the writer cannot change it", {});
          G.label(40, 380, "log who, what, outcome, token counts: never prompts or full keys", {});
          G.from(bx, { opacity: 0, y: 12, stagger: 0.15, duration: 0.3 });
          G.caption("any edit breaks every later link; the tail needs an outside witness");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Sharing GPUs opens four risks: a noisy neighbour, leaks through shared resources, stolen or over-powerful credentials, and unaccountable changes. Defence in depth answers each with its own layer.</li>
<li>Tenancy trades isolation for cost: a shared pool is cheapest, a dedicated deployment costs idle capacity (3 GPUs instead of 1 in the example), a MIG slice is a hardware partition in between.</li>
<li>Prefix caching is a timing side channel across tenants; salt the cache per tenant, disable it, or separate deployments.</li>
<li>Store API keys as hashes of 256-bit random values. Limit work, not just arrivals: per-tenant concurrency and token caps protect the scarce resource (slots).</li>
<li>Default-deny NetworkPolicy in both directions, with the metadata service blocked; sandbox untrusted code with gVisor or Firecracker; give each workload its own least-privilege role and fetch secrets at run time.</li>
<li>Rotate keys by accepting old and new together and removing the old one only when metrics show it unused. Keep a hash-chained audit log with an off-box checkpoint.</li></ul>`,
    sim: {
      title: "Can one tenant break another's SLO?",
      intro: "Tenant A offers load for 60 s; tenant B sends one short request (32 tokens) per second. Both share a backend with a few sequence slots; each request holds a slot for tokens ÷ decode speed. The gateway can rate-limit A, clamp its max_tokens and cap its in-flight requests. The chart shows B's latency per request, queueing included. Press Run to sweep time.",
      height: 280,
      controls: [
        { id: "lam", label: "A offered load (req/s)", min: 1, max: 40, step: 1, value: 20 },
        { id: "tok", label: "A max_tokens", type: "select", value: 32, options: [[32, "32 (short)"], [256, "256"], [2000, "2000 (long)"]] },
        { id: "rps", label: "A rate limit (req/s; 50 = off)", min: 1, max: 50, step: 1, value: 5 },
        { id: "cap", label: "A concurrency cap (0 = off)", min: 0, max: 4, step: 1, value: 0 },
        { id: "clamp", label: "gateway max_tokens clamp (0 = off)", min: 0, max: 512, step: 32, value: 0 },
        { id: "slots", label: "backend sequence slots", min: 2, max: 16, step: 1, value: 4 },
        { id: "vel", label: "decode speed per sequence, tokens/s (example)", min: 20, max: 200, step: 10, value: 100 },
      ],
      run: { label: "Run 60 seconds", frames: 40, ms: 50 },
      draw(G, v, t) {
        const r = noisy(v), base = r.base, slo = 3 * base + 0.05;
        const x0 = 40, x1 = 630, yt = 26, yb = 220;
        const maxY = Math.max(2, Math.min(60, Math.max(...r.B.map((b) => b[1])) * 1.1));
        const X = (s) => x0 + ((x1 - x0) * s) / T_END, Y = (l) => yb - ((yb - yt) * Math.min(l, maxY)) / maxY;
        G.axes(x0, yt, x1 - x0, yb - yt, {});
        [0, 15, 30, 45, 60].forEach((s) => G.label(X(s), yb + 14, s + " s", { size: 10, anchor: s === 60 ? "end" : "middle" }));
        [0, 0.5, 1].forEach((f) => G.label(x0 - 4, Y(maxY * f) + 4, F.num(maxY * f, maxY < 5 ? 1 : 0), { size: 10, anchor: "end" }));
        G.label(x0, yt - 8, "tenant B latency per request, seconds", { size: 11 });
        G.line(x0, Y(slo), x1, Y(slo), { color: "hot", dash: "5 4" }); G.label(x1 - 2, Y(slo) - 5, "3× solo baseline", { size: 10, color: "hot", anchor: "end" });
        const now = t * T_END;
        r.B.forEach(([ts, l]) => { if (ts <= now) G.circle(X(ts), Y(l), 3.5, { fill: l > slo ? "hot" : "ok" }); });
        G.line(X(now), yt, X(now), yb, { color: "muted", dash: "3 3" });
        G.text(x0, 256, `baseline ${F.ms(base)} · B p90 ${r.p90 > 60 ? ">60 s" : F.ms(r.p90)}`, { size: 12, color: r.p90 < slo ? "ok" : "hot" });
        const okB = r.p90 < slo;
        return [
          { title: "Tenant B (the victim)", rows: [["solo latency", F.ms(base)], ["p90 with A running", r.p90 > 60 ? ">60 s (queue never drains)" : F.ms(r.p90)], ["limit (3× baseline)", F.ms(slo)]],
            chip: [okB, okB ? "B stays within its SLO" : "B's SLO is broken by A"] },
          { title: "Tenant A (the flooder)", rows: [["requests offered", F.num(Math.floor(T_END * v.lam))], ["accepted", F.num(r.aOk)], ["rejected with 429", F.num(r.a429)]] },
          { title: "Backend slots", gauge: [[Math.min(1, r.demand / r.supply), r.demand > r.supply ? "hot" : r.demand > 0.7 * r.supply ? "v" : "ok"]], gaugeText: `${F.num(r.demand)} slot-s demanded of ${F.num(r.supply)} available`,
            rows: [["A's time per request", F.ms(Math.min(v.tok, v.clamp > 0 ? v.clamp : 1e9) / v.vel)]] },
          { title: "Model", html: `<p class="note">FIFO queue onto the earliest free slot, one request per slot for tokens ÷ speed. A's rate limit is a token bucket (burst = rate). The concurrency cap counts A's in-flight requests, running or queued, and rejects the excess with 429. Prefill and network time are ignored. It reproduces the shape of <code>test_noisy.py</code>, not its absolute numbers.</p>` },
        ];
      },
    },
    practice: {
      intro: "All four exercises are T0: they run on a laptop with the mock backend; the NetworkPolicy probe needs your kind cluster from P3.3 (and a CNI that enforces policies). Optionally repeat the NetworkPolicy and IRSA checks on EKS (T3): start with <code>make up</code>, finish with <code>make down</code>; the cluster costs the control plane and NAT per hour plus GPU nodes while they run (the GPU pool is min 0 and the budget alarm is a backstop). Nothing is exposed publicly: use <code>kubectl port-forward</code>.",
      items: [
        { title: "NetworkPolicy tests on kind", tier: "T0", goal: "Check the policies statically, apply them to kind and run the live probe. Break them on purpose (delete backends, then default-deny) to see which lines flip, then add a rule that lets Prometheus scrape the gateway on 9000.",
          cmd: "bash course/P3-deployment-and-infra/P3.8-multi-tenancy-and-security/exercises/np_probe.sh" },
        { title: "Key rotation without downtime", tier: "T0", goal: "Make test_rotation.py pass, then complete the xfail test (two gateway instances, a client that switches keys halfway, no request fails). Add a key_generation label to gateway_requests_total and write the PromQL that proves the old key is unused.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.8-multi-tenancy-and-security/exercises/test_rotation.py" },
        { title: "Tamper-evident audit log", tier: "T0", goal: "Implement verify(path, key, checkpoint) in audit_impl.py so it names the line of an edit, deletion or reorder, and detects truncation only with a checkpoint. Then log one record per request without prompt text.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.8-multi-tenancy-and-security/exercises/test_audit.py" },
        { title: "Noisy neighbour", tier: "T0 · hard", goal: "Predict the number of 429s, then make test_long_outputs_within_rps pass: tenant A stays within its rps but asks for 2,000 tokens. Choose a max_tokens cap, a per-tenant concurrency limit, or fair queueing, and keep every P2.7 test green.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.8-multi-tenancy-and-security/exercises/test_noisy.py" },
      ],
      labs: [
        { label: "Run all T0 exercises", path: "uv run pytest course/P3-deployment-and-infra/P3.8-multi-tenancy-and-security/exercises" },
        { label: "Keys, audit log, NetworkPolicies and the seccomp profile (#8)", path: "platform/tenancy/" },
        { label: "Generate a key: <code>PYTHONPATH=platform uv run python -c \"from tenancy import keys; k=keys.generate(); print(keys.key_hash(k))\"</code>", path: "platform/tenancy/keys.py" },
        { label: "Gateway Deployment: 2 replicas, maxUnavailable 0", path: "platform/deploy/base/gateway.yaml" },
        { label: "Static policy checks for manifests", path: "platform/deploy/policy.py" },
        { label: "Request path animation: where auth and quota sit", path: "animations/p3-request-path.html" },
      ],
    },
  });
})();
