/* P4.4 — GPU sharing: time-slicing, MPS and MIG (project #11, stretch). */
(function () {
  const F = S2S.fmt;
  const MOD = "course/P4-multi-gpu-and-distributed/P4.4-gpu-sharing";
  const RC = ["k", "v", "q", "pink"];

  const jain = (xs) => { const s2 = xs.reduce((a, x) => a + x * x, 0); return s2 ? xs.reduce((a, x) => a + x, 0) ** 2 / (xs.length * s2) : 1; };

  /* toy fair-share models (units: fraction of one whole GPU's throughput) */
  function waterfill(D, cap) {                       // max-min fair split of `cap` among demands D (time-slicing)
    const out = D.map(() => 0), idx = D.map((_, i) => i).sort((a, b) => D[a] - D[b]);
    let left = cap;
    idx.forEach((i, k) => { const share = left / (idx.length - k); out[i] = Math.min(D[i], share); left -= out[i]; });
    return out;
  }
  function share(mode, D, p) {
    const n = D.length, sum = D.reduce((a, x) => a + x, 0);
    if (mode === "mig") return D.map((d) => Math.min(d, 1 / n));
    if (mode === "ts") return waterfill(D, 1 - p.o);
    const s = Math.min(1, p.g / sum);                // MPS: kernels overlap (gain g), but no fair split
    return D.map((d) => (p.cap ? Math.min(d * s, 1 / n) : d * s));
  }
  const MODES = { ts: "Time-slicing", mps: "MPS", mig: "MIG" };

  S2S.lesson({
    id: "p4-4", n: "P4.4", title: "GPU sharing",
    subtitle: "Multi-GPU and distributed · first principles · T0 planner and maths, T3 for the contention matrix",
    kicker: "Lesson · ≈ 40 min",
    headline: "Many tenants, one GPU",
    intro: `<p>A small model rarely fills a modern data-centre GPU. Renting a whole GPU for each one wastes money; putting several on one GPU saves it, but now the tenants can disturb each other. This lesson explains the three ways to share a GPU (<b>time-slicing</b>, <b>MPS</b> and <b>MIG</b>), what each one isolates and what it leaves shared, how to pack tenants into MIG slices, how to configure each mode in Kubernetes, and how to measure fairness. It prepares project #11, the GPU partitioning lab (a stretch project).</p>`,
    facts: ["9 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "tenant A"], ["v", "tenant B"], ["q", "tenant C"], ["pink", "tenant D"], ["w", "idle"]],
    prev: "p4-3", next: "p5-1",
    steps: [
      { rail: "why share", title: "A small model leaves most of a big GPU idle",
        body: `<p>Count what one small model actually uses. A 1-billion-parameter model in bf16 has 1.0 B × 2 B = <b>2 GB</b> of weights. Add its KV cache and server overhead and an example budget is about <b>10 GB</b> (example value; it depends on context length and batch). On an 80 GB GPU that is 1/8 of the memory.</p>
<p>Memory is only half of the waste. A GPU is built from many <b>SMs</b> (streaming multiprocessors, the units that run kernels in parallel; an A100 has 108). A small model's kernels are small: each launches work for only a handful of SMs, so the rest wait. At a low request rate the GPU is also idle between requests.</p>
<div class="eq">1 tenant / GPU:   10 GB of 80 GB used = 12.5%
8 tenants / GPU:  8 × 10 GB  = 80 GB       = 100%</div>
<p>So sharing can turn eight GPUs into one. The catch is the word <b>tenant</b>: each of the eight is a separate server, maybe for a separate customer or team. If one floods the GPU with requests, or crashes, or runs out of memory, what happens to the others? That is the <b>isolation</b> question, and the three sharing modes answer it differently.</p>`,
        scene(G) {
          G.text(24, 34, "80 GB GPU memory, in 10 GB blocks (example budgets)", { size: 13 });
          const row = (y, n, lab) => {
            G.text(24, y - 8, lab, { size: 12, color: "ink" });
            const out = [];
            for (let i = 0; i < 8; i++) out.push(G.box(24 + i * 74, y, 68, 44, i < n ? "10 GB" : "", i < n ? { fill: i < 4 ? RC[i] : "blue", size: 12 } : { stroke: "line", dash: "4 3", size: 12 }));
            return out;
          };
          const a = row(70, 1, "one tenant per GPU: 12.5% of memory used");
          const b = row(160, 8, "eight tenants share it: 100% used");
          G.text(24, 244, "and compute: a toy GPU with 32 SMs (an A100 has 108)", { size: 12, color: "ink" });
          const sm = [];
          for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) sm.push(G.rect(24 + c * 26, 258 + r * 22, 22, 18, r === 0 && c < 3 ? { fill: "k", rx: 3 } : { fill: "w", opacity: 0.25, rx: 3 }));
          G.label(250, 276, "one small kernel lights up 3 SMs", { size: 12, color: "ink" });
          G.label(250, 296, "the other 29 wait for work", { size: 12 });
          G.label(24, 372, "2 GB of weights + KV cache + overhead ≈ 10 GB (example value)", { size: 12 });
          G.from([...a, ...b], { opacity: 0, stagger: 0.04, duration: 0.3 });
          G.caption("a whole GPU per small model wastes memory and SMs");
        } },

      { rail: "time-slicing", title: "Time-slicing: take turns on the whole GPU",
        body: `<p>The simplest way to share: let each process use the entire GPU, one at a time. The driver gives process A a short turn, then saves its state and switches to B, then C, and so on in a loop. This is <b>time-slicing</b>. It works on any GPU and needs no setup beyond starting several servers.</p>
<p>What a tenant feels: while the other tenants are busy, it only runs a fraction of the time. With the ideal round-robin of n busy tenants, a request that needs 10 ms of GPU time on an idle GPU spends about:</p>
<div class="eq">n = 4 busy tenants:
  10 ms of work ≈ 4 × 10 ms = 40 ms
(ideal round-robin, no switching cost)</div>
<p>Switching is not free: the GPU must change context on every turn, so some time is lost (the overhead you can set in the simulator). And there is <b>no isolation</b>: all tenants share the same GPU memory, so one that grabs too much can make another run out of memory (OOM), and an error in one process can disturb the rest. Tail latency (the slowest 1% of requests) suffers most, because a request that arrives just after its turn ended waits for everyone else's.</p>`,
        scene(G) {
          G.text(24, 34, "the GPU, one tenant at a time", { size: 13 });
          const out = [], u = 34;
          for (let i = 0; i < 16; i++) out.push(G.box(24 + i * (u + 1), 56, u, 40, "ABCD"[i % 4], { fill: RC[i % 4], size: 13, rx: 4 }));
          G.label(24, 118, "time →   each turn: one tenant owns every SM", { size: 12 });
          G.text(24, 160, "tenant A's view", { size: 13 });
          const a = [];
          for (let i = 0; i < 16; i++) a.push(G.rect(24 + i * (u + 1), 176, u, 28, i % 4 === 0 ? { fill: "k", rx: 4 } : { stroke: "line", dash: "3 3", rx: 4 }));
          G.label(24, 222, "A works in 1 turn out of 4; the rest is waiting", { size: 12, color: "ink" });
          G.text(24, 276, "a request needing 10 ms of GPU time", { size: 13 });
          G.rect(24, 292, 56, 22, { fill: "k", rx: 3 }); G.label(90, 308, "alone: 10 ms", { size: 12 });
          G.rect(24, 324, 224, 22, { fill: "k", rx: 3, opacity: 0.55 }); G.label(258, 340, "with 3 busy neighbours: ≈ 40 ms", { size: 12, color: "hot" });
          G.label(24, 380, "no memory or fault isolation: one process can OOM the others", { size: 12, color: "ink" });
          G.from(out, { opacity: 0, x: -8, stagger: 0.03, duration: 0.2 });
          G.caption("everyone waits for everyone else's turn");
        } },

      { rail: "MPS", title: "MPS: let kernels from several tenants run side by side",
        body: `<p>Time-slicing has a flaw for small models: during A's turn, A's small kernel lights up only a few SMs and the rest sit idle, yet B is not allowed to use them. <b>MPS</b> (Multi-Process Service, an NVIDIA daemon) removes that rule. Kernels from different processes can run on the GPU <b>at the same time</b>, each on its own SMs, as long as SMs are free.</p>
<div class="eq">toy GPU: 32 SMs, each tenant's kernel fills 8
time-slicing: 8 of 32 busy = 25%
              (one tenant at a time)
MPS, 4 tenants: 4 × 8 = 32 busy = 100%</div>
<p>This is the highest-utilisation mode for many small kernels. What you give up is isolation. MPS clients share one fault domain: an error in one client can take down all of them. Memory limits per client are possible but not the default. SM share is only partly controllable: the course names <code>CUDA_MPS_ACTIVE_THREAD_PERCENTAGE</code> as the per-client limit (exercise 4; check the MPS documentation for your driver, since the variable name is unverified here). Without limits, a client that launches bigger or more kernels simply takes more.</p>
<p>MPS fits many small models or workers from <b>one trust domain</b> (one team): you want the utilisation and are not worried about one tenant attacking another.</p>`,
        check: { q: "When is MPS the right choice?",
          options: ["Untrusted tenants that need hard isolation", "Many small models from one trust domain whose kernels alone do not fill the GPU", "A single large model that fills the GPU", "Dev work that needs the GPU only occasionally"], answer: 1,
          why: "MPS raises utilisation by overlapping small kernels, but clients share a fault domain and SM share is only softly controlled. That suits one team's small workers. Hard isolation needs MIG; occasional dev use is where plain time-slicing is enough." },
        scene(G) {
          const grid = (x0, y0, title, owners) => {
            G.text(x0, y0 - 18, title, { size: 13 });
            const out = [];
            for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) {
              const o = owners(r, c);
              out.push(G.rect(x0 + c * 34, y0 + r * 34, 30, 30, o === null ? { fill: "w", opacity: 0.22, rx: 4 } : { fill: RC[o], rx: 4 }));
            }
            return out;
          };
          const a = grid(24, 66, "time-slicing: A's turn", (r) => (r === 0 ? 0 : null));
          G.text(310, 110, "8 of 32 SMs busy = 25%", { size: 13, color: "hot" });
          const b = grid(24, 238, "MPS: all four at once", (r) => r);
          G.text(310, 282, "32 of 32 SMs busy = 100%", { size: 13, color: "ok" });
          G.label(24, 394, "toy GPU with 32 SMs; a real A100 has 108", { size: 12 });
          G.from([...a, ...b], { opacity: 0.2, stagger: 0.01, duration: 0.25 });
          G.caption("MPS fills the idle SMs with other tenants' kernels");
        }, legend: [["k", "tenant A"], ["v", "tenant B"], ["q", "tenant C"], ["pink", "tenant D"], ["w", "idle SM"]] },

      { rail: "MIG", title: "MIG: cut the GPU into separate, smaller GPUs in hardware",
        body: `<p>The strongest option splits the GPU itself. <b>MIG</b> (Multi-Instance GPU, available on data-centre A100, H100 and later) divides one GPU into up to seven <b>instances</b>. Each instance has its own SMs, its own slice of the L2 cache, its own memory and its own memory bandwidth. To a program an instance looks like a smaller GPU, and one instance cannot touch another's memory or crash it.</p>
<p>The budget is counted in <b>slices</b>: the GPU has 7 compute slices and 8 memory slices. A <b>profile</b> named <code>Ng.Mgb</code> takes N compute slices and a fixed amount of memory. For an A100 80 GB (table from the MIG user guide, as coded in <code>platform/partitioning/mig.py</code>; check it against your driver's guide):</p>
<table><tr><th>profile</th><th>compute</th><th>memory slices</th><th>memory</th></tr>
<tr><td>1g.10gb</td><td>1</td><td>1</td><td>10 GB</td></tr>
<tr><td>2g.20gb</td><td>2</td><td>2</td><td>20 GB</td></tr>
<tr><td>3g.40gb</td><td>3</td><td>4</td><td>40 GB</td></tr>
<tr><td>4g.40gb</td><td>4</td><td>4</td><td>40 GB</td></tr>
<tr><td>7g.80gb</td><td>7</td><td>8</td><td>80 GB</td></tr></table>
<p>A layout is valid if the compute slices sum to at most 7 and the memory slices to at most 8. Example: 3g.40gb + 2g.20gb + 1g.10gb + 1g.10gb uses 3+2+1+1 = 7 compute and 4+2+1+1 = 8 memory slices: a full GPU with four tenants. (Real hardware also limits <i>where</i> each profile may start; the planner here ignores that and says so.)</p>`,
        scene(G) {
          G.text(24, 34, "A100 80 GB: 7 compute slices, 8 memory slices", { size: 13 });
          const inst = [["chat", "3g.40gb", [0, 3], [0, 4], 0], ["small", "2g.20gb", [3, 5], [4, 6], 1], ["rerank", "1g.10gb", [5, 6], [6, 7], 2], ["embed", "1g.10gb", [6, 7], [7, 8], 3]];
          G.text(24, 80, "compute", { size: 12, color: "muted" }); G.text(24, 182, "memory", { size: 12, color: "muted" });
          const out = [], cw = 70, mw = 61;
          for (let i = 0; i < 7; i++) G.rect(100 + i * (cw + 4), 60, cw, 40, { stroke: "line", rx: 4, sw: 1 });
          for (let i = 0; i < 8; i++) G.rect(100 + i * (mw + 4), 160, mw, 40, { stroke: "line", rx: 4, sw: 1 });
          inst.forEach(([name, prof, c, m, k]) => {
            out.push(G.box(100 + c[0] * (cw + 4), 60, (c[1] - c[0]) * (cw + 4) - 4, 40, `${name}`, { fill: RC[k], size: 12, rx: 4 }));
            out.push(G.box(100 + m[0] * (mw + 4), 160, (m[1] - m[0]) * (mw + 4) - 4, 40, prof, { fill: RC[k], size: 11, rx: 4 }));
          });
          G.label(100, 122, "3 + 2 + 1 + 1 = 7 compute slices", { size: 12, color: "ink" });
          G.label(100, 222, "4 + 2 + 1 + 1 = 8 memory slices", { size: 12, color: "ink" });
          G.label(24, 280, "each instance: own SMs, own L2 slice, own memory, own bandwidth", { size: 12, color: "ink" });
          G.label(24, 302, "one instance cannot read, starve or crash another", { size: 12, color: "ink" });
          G.label(24, 340, "profile table: NVIDIA MIG guide (unverified here); the planner checks slice totals only", { size: 11 });
          G.from(out, { opacity: 0, y: 8, stagger: 0.1, duration: 0.35 });
          G.caption("a fixed menu of slice sizes, hard walls between tenants");
        }, legend: [["k", "chat"], ["v", "small"], ["q", "rerank"], ["pink", "embed"]] },

      { rail: "isolation", title: "What one bad tenant can do to the others",
        body: `<p>Compare the three modes by asking what a misbehaving tenant can do: use too much memory (OOM), hang or crash, hit an <b>Xid</b> (the NVIDIA driver's error report), or hog SM time.</p>
<table><tr><th></th><th>time-slicing</th><th>MPS</th><th>MIG</th></tr>
<tr><td>memory isolation</td><td>none: one pod can OOM the others</td><td>partial: per-client limits possible</td><td><b>yes</b>, in hardware</td></tr>
<tr><td>fault isolation</td><td>none</td><td>none: a fault can take down all clients</td><td><b>yes</b>, per instance</td></tr>
<tr><td>performance isolation</td><td>none: tail latency suffers</td><td>partial: thread-percentage limits</td><td><b>yes</b>: fixed SMs and bandwidth</td></tr>
<tr><td>utilisation</td><td>high</td><td>highest for small kernels</td><td>can strand capacity</td></tr>
<tr><td>GPUs</td><td>any</td><td>any (Volta and later)</td><td>A100, H100 and later</td></tr></table>
<p>The pattern is a trade: the modes that share the most (MPS) use the GPU best and protect it least; MIG protects best and can waste slices that no tenant fits. Note also that MIG reconfiguration needs the GPU idle: the MIG manager drains it, so <i>every</i> pod on that GPU is evicted when you change the layout.</p>`,
        check: { q: "Which sharing mode gives memory isolation and fault isolation?",
          options: ["Time-slicing", "MPS", "MIG", "All three, if configured correctly"], answer: 2,
          why: "Only MIG partitions memory, SMs and bandwidth in hardware, so an OOM or a fault stays inside one instance. Time-slicing and MPS put every tenant in one memory space and one fault domain; MPS can add limits, but a fault can still take all clients down." },
        scene(G) {
          const cols = ["time-slicing", "MPS", "MIG"], rows = [["memory", [["none", "hot"], ["partial", "v"], ["hardware", "ok"]]], ["faults", [["shared", "hot"], ["shared", "hot"], ["isolated", "ok"]]], ["performance", [["none", "hot"], ["partial", "v"], ["isolated", "ok"]]], ["utilisation", [["high", "k"], ["highest", "k"], ["can strand", "w"]]]];
          const out = [];
          cols.forEach((c, j) => G.text(190 + j * 150 + 66, 44, c, { anchor: "middle", size: 13 }));
          rows.forEach(([lab, cells], i) => {
            const y = 60 + i * 66;
            G.text(24, y + 32, lab, { size: 13 });
            cells.forEach(([t, c], j) => out.push(G.box(190 + j * 150, y, 132, 52, t, { fill: c, size: 13, color: c === "w" ? "ink" : "bg" })));
          });
          G.label(24, 350, "isolation of memory, faults and performance, and how fully the GPU is used", { size: 12, color: "ink" });
          G.label(24, 372, "sharing more = using more, protecting less", { size: 12, color: "ink" });
          G.from(out, { opacity: 0, stagger: 0.04, duration: 0.3 });
          G.caption("only MIG puts hard walls between tenants");
        }, legend: [["ok", "isolated"], ["v", "partial"], ["hot", "shared"], ["k", "good utilisation"]] },

      { rail: "MIG layouts", title: "Packing tenants into MIG slices",
        body: `<p>Given tenants with different needs, which MIG profiles do you choose, and how many GPUs does it take? Two steps:</p>
<ol><li><b>Pick each tenant's smallest profile</b> with enough memory <i>and</i> enough compute slices. A tenant's SLO may need more compute than its memory implies, so give each tenant a <code>min_compute</code> measured in a load test (P2.4).</li>
<li><b>Pack the profiles onto as few GPUs as possible</b>, never exceeding 7 compute or 8 memory slices per GPU.</li></ol>
<p>Example from the module's CLI: chat needs 35 GB and 3 compute slices, rerank 9 GB, embed 8 GB, small 18 GB. Profiles: chat → 3g.40gb (3c, 4m); small → 2g.20gb (2c, 2m); rerank and embed → 1g.10gb each (1c, 1m). Sum 7 compute, 8 memory: all four tenants fit on <b>one</b> A100-80GB.</p>
<p>Waste is built in. A tenant that needs 12 GB gets a 20 GB profile (the next one up), and the extra compute slice that comes with it may sit idle: stranded capacity.</p>
<p>The planner uses <b>first-fit decreasing</b> (place the biggest first, put each in the first GPU where it fits). It is fast but not always optimal. Take two 40 GB tenants (x, y: 3c/4m each), two 20 GB tenants (p, q: 2c/2m) and four 10 GB tenants (s0–s3: 1c/1m). Total 14 compute and 16 memory slices, so 2 GPUs is the lower bound. First-fit puts x and y together (6c, 8m), which strands a compute slice, and needs <b>3</b> GPUs. A better packing is {x, p, q} (7c, 8m) and {y, s0–s3} (7c, 8m): exactly 2. Exercise 1 asks you to find it.</p>`,
        check: { q: "A tenant needs 12 GB of memory and 1 compute slice on an A100-80GB. Which MIG profile does the planner pick?",
          options: ["1g.10gb", "2g.20gb", "3g.40gb", "7g.80gb"], answer: 1,
          why: "1g.10gb has only 10 GB, which is too small. The smallest profile with at least 12 GB is 2g.20gb, so the tenant also receives a second compute slice it may not need: that is how MIG strands capacity." },
        scene(G) {
          const u = 50, gpu = (y, label, chips, c, m) => {
            G.text(24, y + 20, label, { size: 12 });
            G.rect(80, y, 8 * u, 30, { stroke: "line", rx: 4 });
            let x = 80;
            chips.forEach(([n, mem, col]) => { G.box(x + 1, y + 1, mem * u - 2, 28, n, { fill: col, size: 11, rx: 3 }); x += mem * u; });
            G.label(80 + 8 * u + 10, y + 20, `${c}/7 c · ${m}/8 m`, { size: 11 });
          };
          const X = ["x 3g", 4, "k"], Y = ["y 3g", 4, "v"], P = ["p 2g", 2, "q"], Q = ["q 2g", 2, "pink"];
          const S = (i) => [`s${i}`, 1, "blue"];
          G.text(24, 36, "first-fit decreasing: 3 GPUs", { size: 13, color: "hot" });
          gpu(52, "GPU 0", [X, Y], 6, 8); gpu(92, "GPU 1", [P, Q, S(0), S(1), S(2)], 7, 7); gpu(132, "GPU 2", [S(3)], 1, 1);
          G.text(24, 214, "a better packing: 2 GPUs", { size: 13, color: "ok" });
          gpu(230, "GPU 0", [X, P, Q], 7, 8); gpu(270, "GPU 1", [Y, S(0), S(1), S(2), S(3)], 7, 8);
          G.label(24, 340, "chip width = memory slices (8 per GPU) · label = compute slices", { size: 12 });
          G.label(24, 362, "x, y: 40 GB · p, q: 20 GB · s0–s3: 10 GB (from the exercise's test mix)", { size: 12 });
          G.label(24, 384, "lower bound: 14 compute, 16 memory slices needs 2 GPUs", { size: 12, color: "ink" });
          G.from(G.svg.querySelectorAll("rect"), { opacity: 0, stagger: 0.01, duration: 0.25 });
          G.caption("greedy placement can strand a slice and cost a GPU");
        } },

      { rail: "Kubernetes", title: "Turning each mode on with the GPU Operator",
        body: `<p>In Kubernetes (P3.3), pods ask for GPUs through the device plugin, which advertises <code>nvidia.com/gpu</code> resources. The NVIDIA GPU Operator configures sharing from a ConfigMap.</p>
<p><b>Time-slicing and MPS</b> (device-plugin sharing config; field names per the k8s-device-plugin README, unverified for your version): each physical GPU is advertised as several.</p>
<div class="eq">version: v1
sharing:
  timeSlicing:           # or  mps:
    failRequestsGreaterThanOne: true
    resources:
      - name: nvidia.com/gpu
        replicas: 4</div>
<p>One A100 now appears as <b>4</b> × <code>nvidia.com/gpu</code>. The flag <code>failRequestsGreaterThanOne</code> matters: without it, a pod asking for <code>nvidia.com/gpu: 2</code> would get two replicas of the <b>same</b> GPU and think it has two GPUs. The repo's <code>sharing_config.py</code> validates it.</p>
<p><b>MIG</b> (mig-parted config read by the operator's MIG manager): you list the instances per GPU, then label the node to apply it. With the "mixed" strategy pods request the profile by name.</p>
<div class="eq">mig-devices:
  "3g.40gb": 1
  "2g.20gb": 1
  "1g.10gb": 2
kubectl label node &lt;n&gt; \
  nvidia.com/mig.config=s2s-mix --overwrite
pod requests: nvidia.com/mig-3g.40gb: 1</div>
<p>Remember that applying a MIG layout evicts the pods on that GPU. The generated config for your tenant mix: <code>PYTHONPATH=platform uv run python -m partitioning.cli mig --gpu A100-80GB --tenant chat:35:3 …</code>. Compare with <code>platform/partitioning/k8s/mig-config.yaml</code> and <code>time-slicing.yaml</code>.</p>`,
        check: { q: "A time-sliced node advertises 4 replicas per GPU. A pod requests nvidia.com/gpu: 2 and the config lacks failRequestsGreaterThanOne. What does the pod get?",
          options: ["Two different physical GPUs", "Two replicas of the same physical GPU", "A MIG instance", "The pod is always rejected"], answer: 1,
          why: "Replicas are just extra advertised copies of one GPU. The scheduler may hand the pod two of them, so it sees two devices that are really time-shared slices of the same GPU. The flag makes such requests fail instead." },
        scene(G) {
          G.text(24, 34, "time-slicing / MPS: one GPU, 4 replicas", { size: 13 });
          G.box(24, 54, 150, 70, "1 physical GPU", { fill: "w", size: 13, color: "ink" });
          const r = [];
          for (let i = 0; i < 4; i++) { G.arrow(176, 70 + i * 14, 236, 66 + i * 22, { color: "muted" }); r.push(G.box(240, 52 + i * 26, 220, 22, "nvidia.com/gpu", { fill: RC[i], size: 11, rx: 4 })); }
          G.label(470, 90, "×4 advertised", { size: 12 });
          G.text(24, 214, "MIG (mixed strategy): instances by name", { size: 13 });
          G.box(24, 234, 150, 70, "1 physical GPU", { fill: "w", size: 13, color: "ink" });
          const m = [["nvidia.com/mig-3g.40gb", 0], ["nvidia.com/mig-2g.20gb", 1], ["nvidia.com/mig-1g.10gb ×2", 2]];
          m.forEach(([t, i]) => { G.arrow(176, 258 + i * 14, 236, 250 + i * 28, { color: "muted" }); r.push(G.box(240, 236 + i * 28, 250, 24, t, { fill: RC[i], size: 11, rx: 4 })); });
          G.label(24, 340, "failRequestsGreaterThanOne: true  → a request for 2 shared GPUs fails", { size: 12, color: "ink" });
          G.label(24, 362, "MIG layout change: the GPU is drained, its pods are evicted", { size: 12, color: "hot" });
          G.from(r, { opacity: 0, x: -10, stagger: 0.06, duration: 0.3 });
          G.caption("replicas are copies of one GPU; MIG instances are separate devices");
        } },

      { rail: "fairness", title: "Fairness: one number for \"did everyone get a fair share?\"",
        body: `<p>Total throughput hides who got it. If four tenants share a GPU and one starves, the average can still look fine. <b>Jain's fairness index</b> condenses per-tenant throughput x<sub>1</sub> … x<sub>n</sub> into one number between 1/n and 1:</p>
<div class="eq">J(x) = (Σ x_i)² / (n · Σ x_i²)
1   = perfectly even
1/n = one tenant takes everything</div>
<p>Worked examples with four tenants (tok/s):</p>
<div class="eq">30, 30, 30, 30:  120²/(4×3,600)  = 1.00
30, 30, 30, 10:  100²/(4×2,800)  = 0.89
40, 10, 10, 10:   70²/(4×1,900)  = 0.64
40, 40,  0,  0:   80²/(4×3,200)  = 0.50
100,  0,  0,  0: 100²/(4×10,000) = 0.25</div>
<p>If tenants pay for <b>different shares</b>, divide each x<sub>i</sub> by its share first (<code>jain_weighted</code> in <code>platform/partitioning/fairness.py</code>), otherwise a tenant that is entitled to more looks unfairly treated. Pair fairness with <b>p99 inter-token latency</b> (P2.4's load test): contention shows up in the tail and in J, rarely in the mean.</p>`,
        check: { q: "Four tenants get 40, 40, 0 and 0 tok/s. What is Jain's index?",
          options: ["1.00", "0.75", "0.50", "0.25"], answer: 2,
          why: "(40 + 40 + 0 + 0)² = 6,400 and n · Σx² = 4 × (1,600 + 1,600) = 12,800, so J = 0.5. Equivalently, only 2 of 4 tenants are being served, and J measures roughly the fraction served: it ranges from 1 (all) down to 1/n (just one)." },
        scene(G) {
          const sets = [[30, 30, 30, 30], [30, 30, 30, 10], [40, 10, 10, 10], [40, 40, 0, 0], [100, 0, 0, 0]];
          G.text(24, 34, "tok/s per tenant → Jain's index", { size: 13 });
          const out = [];
          sets.forEach((xs, k) => {
            const x0 = 24 + k * 120, J = jain(xs);
            xs.forEach((v, i) => { const h = (v / 100) * 200; out.push(G.rect(x0 + i * 22, 270 - h, 18, Math.max(1, h), { fill: RC[i], rx: 2 })); });
            G.line(x0 - 2, 270, x0 + 88, 270, { color: "line" });
            G.text(x0 + 43, 296, "J = " + J.toFixed(2), { anchor: "middle", size: 14, color: J > 0.95 ? "ok" : J > 0.6 ? "v" : "hot" });
            G.label(x0 + 43, 316, xs.join(" / "), { anchor: "middle", size: 10 });
          });
          G.label(24, 360, "1 = even · 1/4 = one tenant takes all (4 tenants)", { size: 12, color: "ink" });
          G.from(out, { attr: { height: 0 }, stagger: 0.02, duration: 0.4 });
          G.caption("the more uneven the shares, the lower J");
        }, legend: [["k", "tenant A"], ["v", "tenant B"], ["q", "tenant C"], ["pink", "tenant D"]] },

      { rail: "choosing", title: "Choosing a mode, and predicting a flood",
        body: `<p>A rule of thumb for serving:</p>
<ul><li><b>MIG</b> for multi-tenant isolation with known, steady models. You pay with fixed profiles and possible stranded slices.</li>
<li><b>MPS</b> for many small models from one trust domain. You pay with a shared fault domain and soft SM limits.</li>
<li><b>Time-slicing</b> only for dev and bursty, latency-tolerant work. It has the weakest isolation and the worst tail latency.</li></ul>
<p>The module's "predict first": 4 tenants, each serving a 1B-class model, on one A100-80GB. Rank exclusive (4 GPUs), 4 × 1g.10gb MIG, time-slicing × 4 and MPS × 4 by aggregate tok/s, p99 ITL and Jain when one tenant floods. The shape to expect (from the module README, with the real numbers still <code>TODO(run-on: p4d.24xlarge)</code>): MIG has the flattest p99 and Jain near 1 under a flood; time-slicing has the worst tails; MPS has the highest aggregate throughput with small models and the lowest fairness without limits.</p>
<p>The simulator makes this concrete with <b>toy models</b>: time-slicing as an equal split of GPU time with a switching overhead, MPS as free overlap without a fair split, MIG as equal hard partitions. Turn up the flood factor and watch who suffers. Common mistakes: treating time-slicing as isolation; choosing MIG profiles by memory only; reporting average throughput only.</p>`,
        scene(G) {
          const rows = [["MIG", "multi-tenant isolation, known steady models", "ok"], ["MPS", "many small models, one trust domain", "k"], ["time-slicing", "dev and bursty, latency-tolerant work", "v"]];
          const out = [];
          rows.forEach(([a, b, c], i) => {
            const y = 40 + i * 70;
            out.push(G.box(24, y, 150, 50, a, { fill: c, size: 14 }));
            G.arrow(178, y + 25, 206, y + 25, { color: "muted", w: 2 });
            out.push(G.box(210, y, 400, 50, b, { stroke: "ink", size: 13, color: "ink" }));
          });
          G.text(24, 276, "pays for it with:", { size: 13 });
          G.label(24, 298, "MIG: fixed profiles, stranded slices, drain on reconfigure", { size: 12, color: "ink" });
          G.label(24, 320, "MPS: shared fault domain, soft SM limits", { size: 12, color: "ink" });
          G.label(24, 342, "time-slicing: no isolation, worst tail latency", { size: 12, color: "ink" });
          G.label(24, 380, "expected ordering is from the module README; real numbers are not yet measured", { size: 11 });
          G.from(out, { opacity: 0, x: -12, stagger: 0.12, duration: 0.35 });
          G.caption("pick by what a bad neighbour is allowed to do");
        }, legend: [["ok", "MIG"], ["k", "MPS"], ["v", "time-slicing"]] },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Small models leave most of a big GPU's memory and SMs idle, so sharing saves money; the price is isolation.</li>
<li>Time-slicing takes turns on the whole GPU (any GPU, no isolation, bad tails); MPS overlaps kernels from several processes (best utilisation, shared fault domain); MIG partitions the GPU in hardware (full isolation, fixed profiles, A100/H100+).</li>
<li>MIG budget: 7 compute and 8 memory slices per GPU; pick each tenant's smallest profile, then pack. Greedy packing can cost an extra GPU, and a profile can strand compute.</li>
<li>The GPU Operator configures time-slicing and MPS through device-plugin replicas (use <code>failRequestsGreaterThanOne</code>) and MIG through mig-parted; a MIG change drains the GPU.</li>
<li>Jain's index (Σx)² / (n·Σx²) measures fairness from 1/n to 1; always report it with p99 latency, not just average throughput.</li></ul>`,
    sim: {
      title: "A flood on a shared GPU: who suffers?",
      intro: "Each tenant would use a fraction of one GPU if it had the GPU alone. Tenant 0 floods: its demand is multiplied. The chart shows, for the mode you pick, how much of each tenant's demand is served (dashed outline = demand, filled = served). These are toy fair-share models, not measurements: time-slicing splits the GPU time fairly among busy tenants minus a switching overhead; MPS overlaps kernels (a gain) but shares by demand; MIG gives each tenant a fixed 1/n partition. Exercise 3 measures the real thing.",
      height: 300,
      controls: [
        { id: "mode", label: "sharing mode", type: "select", value: "ts", options: [["ts", "Time-slicing"], ["mps", "MPS"], ["mig", "MIG (equal partitions)"]] },
        { id: "n", label: "tenants", min: 2, max: 8, step: 1, value: 4 },
        { id: "d", label: "demand per tenant, as % of one whole GPU", min: 5, max: 60, step: 1, value: 20, format: (v) => v + "%" },
        { id: "k", label: "flood factor for tenant 0 (× its demand)", min: 1, max: 8, step: 0.5, value: 4, format: (v) => v + "×" },
        { id: "o", label: "time-slicing switching overhead (example value)", min: 0, max: 30, step: 1, value: 10, format: (v) => v + "%" },
        { id: "g", label: "MPS overlap gain (example value: depends on kernel size)", min: 1, max: 2, step: 0.05, value: 1.3, format: (v) => v.toFixed(2) + "×" },
        { id: "cap", label: "MPS per-client limit of 1/n (thread percentage)", type: "select", value: "0", options: [["0", "off"], ["1", "on"]] },
      ],
      draw(G, v) {
        const n = v.n, D = [...Array(n)].map((_, i) => (v.d / 100) * (i === 0 ? v.k : 1));
        const p = { o: v.o / 100, g: v.g, cap: v.cap === "1" };
        const S = share(v.mode, D, p);
        const ymax = Math.max(1, ...D, ...S), Y = (x) => 240 - (200 * x) / ymax;
        G.axes(60, 40, 560, 200, {});
        G.label(60, 30, "fraction of one GPU's throughput", { size: 11 });
        [0.5, 1].forEach((f) => { if (f <= ymax) { G.line(60, Y(f), 620, Y(f), { color: "line", w: 0.5 }); G.label(54, Y(f) + 4, (f * 100).toFixed(0) + "%", { anchor: "end", size: 10 }); } });
        const bw = Math.min(54, 520 / n - 10);
        D.forEach((d, i) => {
          const x = 76 + i * (520 / n), h = (200 * S[i]) / ymax;
          G.rect(x, Y(d), bw, (200 * d) / ymax, { stroke: "muted", dash: "4 3", rx: 3, sw: 1.5 });
          G.rect(x, 240 - h, bw, h, { fill: i === 0 ? "hot" : RC[i % 4], rx: 3, opacity: 0.9 });
          G.text(x + bw / 2, 258, i === 0 ? "T0 flood" : "T" + i, { anchor: "middle", size: 11 });
        });
        const agg = S.reduce((a, x) => a + x, 0), cap = v.mode === "ts" ? 1 - p.o : v.mode === "mps" ? p.g : 1;
        const norm = S.slice(1).map((s, i) => s / D[i + 1]), worst = Math.min(...norm);
        const all = ["ts", "mps", "mig"].map((m) => { const s = share(m, D, p); return [m, s.reduce((a, x) => a + x, 0), jain(s), Math.min(...s.slice(1).map((x, i) => x / D[i + 1]))]; });
        return [
          { title: `${MODES[v.mode]}: this setting`, rows: [["total demand", (100 * D.reduce((a, x) => a + x, 0)).toFixed(0) + "% of a GPU"], ["aggregate served", (100 * agg).toFixed(0) + "%"], ["Jain's index of served", jain(S).toFixed(3)], ["worst normal tenant served", (100 * worst).toFixed(0) + "% of its demand"]],
            chip: [worst > 0.97, worst > 0.97 ? "normal tenants unaffected" : "normal tenants hurt"] },
          { title: "All three modes, same load", rows: all.map(([m, a, j, w]) => [MODES[m], `${(100 * a).toFixed(0)}% · J ${j.toFixed(2)} · normal ${(100 * w).toFixed(0)}%`]),
            html: `<p class="note">Columns: aggregate served, Jain's index, worst normal tenant's fraction of demand served. MIG may leave capacity stranded: with a flood it serves tenant 0 only its 1/n. Above 100% means overlapped kernels did more work than one tenant could alone.</p>` },
          { title: "How to read it", html: `<p class="note">Toy models. Time-slicing: max-min fair split of (1 − overhead). MPS: all demand shrinks by the same factor once demand exceeds the overlap gain; the optional limit caps each client at 1/n. MIG: each tenant gets min(demand, 1/n). Real MIG instances come in 7 compute slices, so equal partitions are an idealisation. Measure the real thing in exercise 3.</p>` },
        ];
      },
    },
    practice: {
      items: [
        { title: "Choose a MIG layout for a tenant mix", tier: "T0 · medium", goal: "Implement plan(gpu, tenants) in exercises/layout.py: smallest profile per tenant, then bin-pack within 7 compute and 8 memory slices per GPU. Stretch: beat first-fit decreasing on the 2-GPU mix (remove the xfail).",
          cmd: `uv run pytest ${MOD}/exercises/test_layout.py` },
        { title: "Time-slicing and MPS configs pass schema checks", tier: "T0 · easy", goal: "Read platform/partitioning/sharing_config.py, explain failRequestsGreaterThanOne, add an MPS memory-limit rule, and list what one misbehaving pod can do in each mode.",
          cmd: `uv run pytest ${MOD}/exercises/test_sharing_config.py` },
        { title: "#11 contention matrix", tier: "T3 · hard", goal: "On an A100/H100, run N vLLM servers under time-slicing, MPS and MIG, load them all at once, and fill in tok/s, p99 ITL and Jain; repeat with one tenant at 4× the rate. Needs S2S_API_KEY and a ~1B model.",
          cmd: `MODE=timeslice N=4 MODEL=<model dir> bash platform/partitioning/contention.sh` },
        { title: "A fair scheduler across MPS clients", tier: "T3 · hard", goal: "Measure MPS unfairness with one 4× client, fix it with a per-client thread percentage, then fix it above the GPU with per-tenant concurrency limits in the #6 gateway; compare the two.",
          cmd: `MODE=mps N=4 MODEL=<model dir> bash platform/partitioning/contention.sh` },
      ],
      labs: [
        { label: "Plan a MIG layout for a tenant mix (<code>PYTHONPATH=platform uv run python -m partitioning.cli mig --gpu A100-80GB --tenant chat:35:3 …</code>)", path: "platform/partitioning/cli.py" },
        { label: "MIG profile tables and the first-fit-decreasing planner", path: "platform/partitioning/mig.py" },
        { label: "Build and validate time-slicing and MPS sharing configs", path: "platform/partitioning/sharing_config.py" },
        { label: "Jain's fairness index, plain and share-weighted", path: "platform/partitioning/fairness.py" },
        { label: "GPU Operator ConfigMaps for MIG and for time-slicing / MPS", path: "platform/partitioning/k8s/mig-config.yaml" },
        { label: "T3: the contention script (<code>MODE=timeslice|mps|mig N=2|4</code>)", path: "platform/partitioning/contention.sh" },
        { label: "AWS guide: p4d.24xlarge or a Capacity Block for MIG; cost, auto-stop, teardown", path: `${MOD}/aws.md` },
      ],
    },
  });
})();
