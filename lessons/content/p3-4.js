/* P3.4 — Serving stacks on Kubernetes. What a stack adds over a Deployment; routing that respects the KV cache; prefill/decode disaggregation and its transfer bill; zero-drop rollouts (#1). */
(function () {
  const F = S2S.fmt;

  /* a 32-bit string hash (FNV-1a + a final mix) used for the rendezvous demo; the real router should use hashlib, see step 5 */
  function h32(s) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b) >>> 0; h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35) >>> 0; h ^= h >>> 16;
    return h >>> 0;
  }
  const score = (sess, b) => h32(sess + "|" + b) / 4294967296;
  const owner = (sess, bs) => bs.reduce((best, b) => (score(sess, b) > score(sess, best) ? b : best), bs[0]);

  S2S.lesson({
    id: "p3-4", n: "P3.4", title: "Serving stacks on Kubernetes",
    subtitle: "Deployment and infra · first principles · T0 for routing, T3 for the cluster",
    kicker: "Lesson · ≈ 45 min",
    headline: "Where the request goes matters as much as the GPU it lands on",
    intro: `<p>P3.3 left you with a vLLM Deployment that Kubernetes keeps alive. That is one server, replicated. This lesson asks what a real cluster needs on top: a standard way to declare and roll out models, a router that knows the KV cache exists, and, for the heaviest workloads, a split of prefill and decode onto different GPUs. You will meet four stacks (KServe, Ray Serve, NVIDIA Dynamo, llm-d), learn what each one adds and what it costs, work out the arithmetic of KV transfer, and finish with the checklist for project #1: the cluster that serves a model with no dropped requests during a rollout.</p>`,
    facts: ["9 steps", "4 checkpoints", "1 simulator", "5 exercises"],
    legend: [["k", "request / KV"], ["ok", "fine"], ["v", "you build it"], ["hot", "problem"]],
    prev: "p3-3", next: "p3-5",
    steps: [
      /* 1 ------------------------------------------------------------ */
      { rail: "the problem", title: "A Deployment keeps pods alive. It knows nothing about LLMs",
        body: `<p>In P3.3 you wrote a <b>Deployment</b> (it keeps N copies of the pod running) and a <b>Service</b> (one stable address that spreads connections over the ready pods). That is already a lot: restarts, rolling updates, probes, a GPU request.</p>
<p>But look at what Kubernetes does <i>not</i> know. It treats every pod as identical and every request as the same size. For an LLM server that is wrong in at least five ways:</p>
<ul><li><b>Routing.</b> A Service picks a pod without looking at what is in each pod's KV cache (P2.3's prefix cache), or how long its queue is.</li>
<li><b>Scaling.</b> The default signals (CPU) are blind to GPU work (P3.6).</li>
<li><b>Declaration.</b> "Serve this model, from this storage, with these flags" is spread over a Deployment, a Service, a PVC, a PDB and a ScaledObject, written by hand for every model.</li>
<li><b>Release.</b> Sending 10% of traffic to a new model version is not something a Service does.</li>
<li><b>Shape.</b> One pod does prefill and decode. At large scale you may want them on different GPUs.</li></ul>
<p>A <b>serving stack</b> is software that adds some of these on top of Kubernetes. The rule of this lesson: start with the plain Deployment, and adopt a stack only for a problem you have <i>measured</i>.</p>`,
        scene(G) {
          G.text(24, 34, "what Kubernetes gives you  ·  what you still build", { size: 14 });
          G.rect(24, 52, 250, 300, { stroke: "ok", rx: 10 }); G.text(36, 74, "Deployment + Service", { color: "ok", size: 13 });
          const have = ["pods restarted when they die", "rolling update (maxSurge)", "probes: ready / alive", "one GPU per pod", "one stable address"];
          const g1 = have.map((t, i) => G.box(38, 90 + i * 50, 222, 38, t, { stroke: "ok", color: "ink", size: 12 }));
          G.rect(366, 52, 250, 300, { stroke: "v", dash: "6 4", rx: 10 }); G.text(378, 74, "left for you", { color: "v", size: 13 });
          const need = ["route by KV-cache contents", "scale on queue depth", "one object per model", "canary / traffic split", "split prefill and decode"];
          const g2 = need.map((t, i) => G.box(380, 90 + i * 50, 222, 38, t, { stroke: "v", color: "ink", size: 12, dash: "5 4" }));
          G.arrow(276, 202, 362, 202, { color: "muted", w: 2 }); G.label(290, 192, "not enough", { size: 11, color: "hot" });
          G.label(24, 384, "a serving stack adds some of the right-hand column", { color: "ink" });
          G.label(24, 406, "adopt one only for a problem you have measured", {});
          G.from(g1, { opacity: 0, x: -16, stagger: 0.08, duration: 0.3 }); G.from(g2, { opacity: 0, x: 16, stagger: 0.08, duration: 0.3, delay: 0.4 });
          G.caption("Kubernetes sees identical pods; an LLM fleet is not");
        } },

      /* 2 ------------------------------------------------------------ */
      { rail: "four stacks", title: "Four stacks, four different things they add",
        body: `<p>All four below still run vLLM (or a similar engine) in the pods. They differ in what they wrap <i>around</i> it.</p>
<table><tr><th>stack (pinned)</th><th>you get</th><th>you pay</th></tr>
<tr><td><b>KServe</b> v0.21.0: one <code>InferenceService</code> object stands for "this model, served"</td><td>a standard API, canary traffic splits, optional scale-to-zero, a Hugging Face / vLLM runtime</td><td>CRDs and a controller; Knative mode adds much more</td></tr>
<tr><td><b>Ray Serve</b> (via KubeRay) Ray 2.59.0: Python-first composition of deployments</td><td>model pipelines, Python pre/post-processing, autoscaling on ongoing requests</td><td>a Ray cluster to operate; one more layer above vLLM</td></tr>
<tr><td><b>NVIDIA Dynamo</b> v1.5.0: a distributed inference framework</td><td>disaggregated prefill/decode, KV-aware routing, KV transfer, a planner</td><td>a newer, faster-moving project with more components</td></tr>
<tr><td><b>llm-d</b> v0.10.0: Kubernetes-native, vLLM-based, Envoy inference scheduler</td><td>prefix and load aware endpoint picking, disaggregation guides</td><td>the Gateway API and inference-extension stack</td></tr></table>
<p>Two of them (Dynamo, llm-d) attack the <i>physics</i> of LLM serving: where the KV cache lives and where each phase runs. The other two (KServe, Ray Serve) attack the <i>operations</i>: declaring, releasing and composing models. Steps 3 to 7 take these in turn.</p>`,
        scene(G) {
          const rows = [
            ["Deployment", "w", "a pod per replica, round-robin Service", "you build routing, scaling, canaries"],
            ["KServe", "k", "one InferenceService, traffic splits", "CRDs + controller (Knative: more)"],
            ["Ray Serve", "q", "Python pipelines, autoscale on load", "a Ray cluster to run"],
            ["Dynamo", "v", "disaggregation, KV-aware routing, planner", "newer, more components"],
            ["llm-d", "ok", "prefix-aware scheduler on Envoy", "Gateway API + extension stack"],
          ];
          G.text(24, 34, "what each adds (top line) and costs (bottom line)", { size: 13 });
          const g = rows.map(([n, c, a, p], i) => {
            const y = 50 + i * 72, gg = G.group();
            G.box(24, y, 120, 58, n, { fill: c, size: 13, parent: gg });
            G.rect(150, y, 466, 58, { stroke: "line", rx: 8, parent: gg });
            G.text(162, y + 24, "+ " + a, { size: 12, color: "ink", parent: gg });
            G.text(162, y + 45, "– " + p, { size: 12, color: "muted", parent: gg });
            return gg;
          });
          G.from(g, { opacity: 0, x: 24, stagger: 0.12, duration: 0.35 });
          G.caption("same engine in the pods; different wrappers around it");
        } },

      /* 3 ------------------------------------------------------------ */
      { rail: "KServe", title: "A wrapper changes the objects you manage, not the engine",
        body: `<p>KServe's idea: you write <i>one</i> small object that says what you want, and a <b>controller</b> (a program in the cluster that watches the object and creates whatever it needs) builds the rest. From <code>platform/deploy/kserve/inferenceservice.yaml</code>:</p>
<div class="eq">kind: InferenceService
metadata:
  name: llm
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 2
    model:
      modelFormat: { name: huggingface }
      storageUri: pvc://models/model
      args: ["--max-model-len=8192"]</div>
<p>In <b>RawDeployment</b> mode the controller creates an ordinary Deployment, Service and autoscaler. The pod inside runs the same vLLM with the same flags as your hand-written one, so latency should match within noise. In <b>Knative</b> (serverless) mode KServe also installs Knative Serving, which adds scale-to-zero and request-based scaling, at the price of a heavier control plane and an extra proxy hop.</p>
<p>The lesson in this: a stack is a trade of <b>operational objects</b> for <b>moving parts</b>. Compare them fairly by pinning the same engine image and flags across all stacks (exercise 1). Ray Serve plays a similar role in Python (<code>platform/deploy/rayserve/app.py</code> wraps vLLM's async engine in a <code>@serve.deployment</code> with <code>target_ongoing_requests: 32</code>) and costs you a Ray cluster. The field names in these manifests are <b>UNVERIFIED</b> against the pinned versions; run <code>kubectl apply</code> and let the API server correct you.</p>`,
        scene(G) {
          const a = G.box(24, 60, 170, 78, "", { stroke: "k" });
          G.text(36, 84, "InferenceService", { color: "k", size: 13 }); G.label(36, 104, "you write ~15 lines", { size: 11 }); G.label(36, 122, "name, model, replicas", { size: 11 });
          const c = G.box(250, 60, 140, 78, "controller", { stroke: "v", color: "v", size: 13 });
          G.arrow(196, 99, 246, 99, { color: "k", w: 2 });
          const objs = ["Deployment", "Service", "autoscaler"];
          const ob = objs.map((t, i) => G.box(450, 50 + i * 44, 160, 36, t, { stroke: "ok", color: "ink", size: 12 }));
          const ar = objs.map((t, i) => G.arrow(392, 99, 446, 68 + i * 44, { color: "v", w: 1.5 }));
          G.arrow(530, 186, 530, 214, { color: "ok", w: 2 });
          G.box(400, 216, 210, 52, "pod: same vLLM, same flags", { fill: "k", size: 11 });
          G.text(24, 214, "RawDeployment: plain K8s objects", { size: 12, color: "ok" });
          G.text(24, 238, "Knative mode: + Knative Serving,", { size: 12, color: "v" });
          G.text(24, 256, "scale to zero, one more proxy hop", { size: 12, color: "v" });
          G.rect(24, 300, 592, 96, { stroke: "line", rx: 10 });
          G.text(40, 326, "fair comparison rule", { size: 13, color: "ink" });
          G.label(40, 350, "same vLLM image · same flags · same seed and rates", {});
          G.label(40, 372, "otherwise you measure your mistakes, not the stack", {});
          G.from([a, c], { opacity: 0, y: -12, stagger: 0.2, duration: 0.3 }); G.from(ob.concat(ar), { opacity: 0, stagger: 0.08, duration: 0.25, delay: 0.5 });
          G.caption("one object in, ordinary Kubernetes objects out");
        } },

      /* 4 ------------------------------------------------------------ */
      { rail: "prefix routing", title: "Round-robin throws the KV cache away",
        body: `<p>P2.3's <b>prefix cache</b> keeps the KV blocks of prompts it has already processed, so a request that starts with the same tokens skips recomputing them. The cache lives <i>inside one replica's GPU memory</i>. Other replicas cannot see it.</p>
<p>Think of a chat. Turn 6 of a conversation sends turns 1 to 5 again as its prefix, plus the new message. If the replica that handled turn 5 receives turn 6, almost the whole prompt is a cache hit. If another replica receives it, nothing hits and the whole prompt is prefilled again.</p>
<p>A Kubernetes Service spreads requests without looking at content, so with N replicas a turn lands on the replica that holds its prefix with probability about 1/N. Worked example: turn 6 has a 2,000-token prompt, of which 1,800 tokens are earlier turns (cached on one replica) and 200 are new. With 4 replicas:</p>
<div class="eq">sticky:       prefill 200 tokens
round-robin:  0.25 × 200 + 0.75 × 2000
            = 50 + 1500 = 1550 tokens
1550 ÷ 200 = 7.75× more prefill work</div>
<p>Every extra prefilled token is GPU time other requests wait for, and TTFT grows with it. <b>Prefix-aware routing</b> sends a request to the replica that already holds its prefix, unless that replica is badly overloaded. llm-d's inference scheduler and Dynamo's router do this by scoring replicas on expected cache hits and load. Your own gateway can approximate it more cheaply with <b>session affinity</b>: the same conversation id always goes to the same replica (next step).</p>`,
        check: { q: "A fleet of 4 replicas sits behind a plain round-robin Service. Conversations share long prefixes. What happens to the prefix-cache hit rate as you add replicas?",
          options: ["It rises, because there is more total cache", "It falls, because a request lands on the replica holding its prefix less often", "It stays the same, because each replica caches independently"], answer: 1,
          why: "Each replica has its own cache and the Service ignores content, so the chance that a request reaches the replica that holds its prefix is about 1/N. More replicas means a lower chance, so more prefill work and higher TTFT for the same traffic." },
        scene(G) {
          G.text(24, 34, "turns 1..4 of one conversation, 4 replicas", { size: 13 });
          const R = ["A", "B", "C", "D"];
          const panel = (x0, title, color, sticky) => {
            G.text(x0, 66, title, { size: 13, color });
            R.forEach((n, i) => {
              const y = 84 + i * 58, hold = sticky ? (i === 0 ? 4 : 0) : 1;
              G.rect(x0, y, 130, 46, { stroke: "line", rx: 6 }); G.label(x0 + 8, y + 16, "replica " + n, { size: 11 });
              for (let k = 0; k < (sticky ? 4 : 1); k++) {
                const on = sticky ? i === 0 : true;
                if (on) G.rect(x0 + 8 + k * 24, y + 24, 20, 16, { fill: "k", rx: 3 });
              }
              if (!sticky) G.label(x0 + 38, y + 36, "turn " + (i + 1), { size: 10 });
            });
          };
          panel(24, "round-robin Service", "hot", false);
          panel(334, "session affinity", "ok", true);
          G.label(24, 330, "1 turn per replica: 1 hit in 4", { color: "hot", size: 11 });
          G.label(334, 330, "A holds all 4 turns: every turn hits", { color: "ok", size: 11 });
          G.text(24, 372, "turn 6 prefill, expected tokens:   1550   vs   200", { size: 14, color: "ink" });
          G.label(24, 398, "(2000-token prompt, 1800 cached on one replica, 4 replicas)", {});
          G.caption("the cache is per replica, so the router must know where it is");
        } },

      /* 5 ------------------------------------------------------------ */
      { rail: "affinity", title: "Pin a session to a backend, and lose almost nothing when one dies",
        body: `<p>Plan: give every request a session id (header <code>X-Session-Id</code>) and always send a session to the same backend. The obvious recipe is <code>hash(session) % N</code>. It has a flaw.</p>
<p>Add a fourth backend to three. A session keeps its backend only if <code>h % 3 == h % 4</code>. Over any 12 consecutive values of <i>h</i> that holds for just 3 (h = 0, 1, 2), so <b>75% of sessions move</b> and 75% of your warm caches become useless. Remove a backend (3 to 2) and 2/3 of sessions move.</p>
<p><b>Rendezvous hashing</b> (highest random weight) fixes it. For each session, give every backend a score from a hash of <code>(session, backend name)</code>, and choose the highest score:</p>
<div class="eq">score(s, b) = hash(s + "|" + b)
backend(s)   = argmax over healthy b of score(s, b)</div>
<p>A session moves only when its top-scoring backend disappears, and then it simply takes its <i>runner-up</i>. Nobody else changes. Adding a fourth backend steals exactly the sessions that score it highest, about 1 in 4 (25%). Removing one of three moves only that backend's third (about 33%). The scene computes real scores; the simulator measures the moved fraction on 1,000 sessions.</p>
<p>Two implementation notes for exercise 5 (<code>platform/gateway/router.py</code>). Python's built-in <code>hash()</code> of a string changes between processes, so use <code>hashlib</code>. And always fall back to the next-ranked backend when the first is unhealthy: affinity <i>without</i> failover black-holes a dead replica's sessions.</p>`,
        check: { q: "Three backends serve 1,000 sessions with rendezvous hashing. One backend fails. About how many sessions change backend?",
          options: ["All 1,000", "About 667, as with modulo hashing", "About 333, only the sessions that were on the failed backend"], answer: 2,
          why: "A session's choice is its highest-scoring healthy backend. Sessions whose winner is still healthy keep it. Only the failed backend's sessions (about one third) fall to their runner-up. Modulo hashing would remap about two thirds." },
        scene(G) {
          const B = ["a", "b", "c"], S = ["s8", "s9", "s1"], col = { a: "k", b: "v", c: "q" };
          G.text(24, 34, "score = hash(session | backend); highest wins", { size: 13 });
          const x0 = 40, W = 56, gap = 12, base = 210, H = 130;
          const bars = [];
          S.forEach((s, si) => {
            const bx = x0 + si * 200;
            G.label(bx, 58, "session " + s, { size: 12, color: "ink" });
            const win = owner(s, B);
            B.forEach((b, bi) => {
              const v = score(s, b), x = bx + bi * (W + gap);
              bars.push(G.rect(x, base - H * v, W, H * v, { fill: col[b], rx: 3, opacity: b === win ? 1 : 0.35 }));
              G.label(x + W / 2, base + 16, b, { anchor: "middle", size: 12 });
              G.label(x + W / 2, base - H * v - 5, v.toFixed(2), { anchor: "middle", size: 10 });
            });
            G.text(bx, 252, "→ " + win, { size: 14, color: col[win] });
          });
          G.line(x0 - 6, base, 610, base, { color: "line" });
          G.rect(24, 276, 592, 130, { stroke: "line", rx: 10 });
          G.text(40, 300, "backend a fails", { size: 13, color: "hot" });
          S.forEach((s, si) => {
            const w0 = owner(s, B), w1 = owner(s, ["b", "c"]);
            G.text(40, 326 + si * 24, `${s}:  ${w0} → ${w1}` + (w0 === w1 ? "   (unchanged)" : "   (moves to its runner-up)"), { size: 12, color: w0 === w1 ? "muted" : "hot" });
          });
          G.text(330, 326, "3 → 4 backends:", { size: 12 }); G.label(330, 346, "modulo moves 75%, rendezvous 25%", { color: "ink" });
          G.label(330, 374, "3 → 2: modulo 67%, rendezvous 33%", { color: "ink" });
          G.from(bars, { attr: { height: 0 }, stagger: 0.04, duration: 0.4 });
          G.caption("a failure moves only the sessions that ranked the dead backend first");
        } },

      /* 6 ------------------------------------------------------------ */
      { rail: "disaggregation", title: "Prefill and decode fight over the same GPU",
        body: `<p>Recall P1.2. <b>Prefill</b> processes the whole prompt at once: lots of arithmetic, compute-bound, short but heavy. <b>Decode</b> then produces one token per step: little arithmetic per byte of weights read, memory-bound, long and steady.</p>
<p>On one GPU the two phases take turns. A 4,000-token prefill occupies the GPU for hundreds of milliseconds, and during that time every <i>other</i> user's decode step waits. They see a stall in their stream: a spike in <b>ITL</b> (inter-token latency). Chunked prefill (P2.3) cuts the big prefill into slices so the stalls are shorter, but the two phases still compete.</p>
<p><b>Prefill–decode disaggregation</b> (the DistServe paper, arXiv 2401.09670) removes the competition:</p>
<ul><li>a <b>prefill pool</b> of workers computes the KV cache for the prompt;</li>
<li>it <b>ships the KV cache over the network</b> to a <b>decode pool</b>;</li>
<li>the decode workers only decode, so their steps are never interrupted;</li>
<li>each pool is sized and parallelised for its own bottleneck.</li></ul>
<p>Dynamo and llm-d both implement this. You pay with a new cost, the transfer in the middle (next step), and with two pools that must stay balanced: too few prefill workers and TTFT queues up, too few decode workers and the KV caches wait.</p>`,
        scene(G) {
          G.text(24, 34, "one GPU, shared", { size: 13, color: "hot" });
          G.label(24, 56, "decode steps of other users:", { size: 11 });
          const ticks = [];
          for (let i = 0; i < 6; i++) ticks.push(G.rect(24 + i * 26, 64, 20, 26, { fill: "ok", rx: 3 }));
          const pf = G.rect(24 + 6 * 26, 64, 150, 26, { fill: "hot", rx: 3 }); G.text(24 + 6 * 26 + 75, 82, "4k prefill", { size: 12, anchor: "middle", color: "bg" });
          for (let i = 0; i < 4; i++) ticks.push(G.rect(24 + 6 * 26 + 156 + i * 26, 64, 20, 26, { fill: "ok", rx: 3, opacity: 0.9 }));
          G.line(24 + 6 * 26, 100, 24 + 6 * 26 + 150, 100, { color: "hot", w: 2 }); G.label(24 + 6 * 26, 118, "no token for everyone else here: ITL spike", { color: "hot", size: 11 });
          G.line(16, 150, 624, 150, { color: "line", dash: "4 4" });
          G.text(24, 182, "split into two pools", { size: 13, color: "ok" });
          G.rect(24, 200, 190, 110, { stroke: "v", rx: 10 }); G.text(36, 222, "prefill pool", { color: "v", size: 13 });
          const p = [0, 1].map((i) => G.box(38 + i * 84, 238, 76, 56, "GPU " + (i + 1), { fill: "v", size: 12 }));
          G.rect(426, 200, 190, 110, { stroke: "k", rx: 10 }); G.text(438, 222, "decode pool", { color: "k", size: 13 });
          const d = [0, 1].map((i) => G.box(440 + i * 84, 238, 76, 56, "GPU " + (i + 3), { fill: "k", size: 12 }));
          const a = G.arrow(216, 255, 424, 255, { color: "ok", w: 3 }); G.text(320, 244, "KV cache", { anchor: "middle", size: 12, color: "ok" });
          G.label(320, 278, "over the network", { anchor: "middle" });
          G.label(24, 340, "prefill: compute-bound, bursty  ·  decode: memory-bound, steady", { color: "ink" });
          G.label(24, 364, "each pool is sized for its own bottleneck", {});
          G.label(24, 388, "new cost: moving the KV cache, and balancing two pools", { color: "hot" });
          G.from(pf, { attr: { width: 0 }, duration: 0.6 }); G.from(a, { opacity: 0, duration: 0.6, delay: 0.5 });
          G.pulse(pf, { repeat: 4, lo: 0.5 });
          G.caption("decode never waits for a prefill once the phases have separate GPUs");
        } },

      /* 7 ------------------------------------------------------------ */
      { rail: "transfer bill", title: "What the KV hand-over costs, in bytes and seconds",
        body: `<p>Disaggregation pays only if shipping the KV cache is cheap compared with what it saves. So do the arithmetic. From P1.2, Llama-3-8B in bf16 stores <b>128 KiB of KV per token</b> (32 layers × 2 (K and V) × 8 KV heads × 128 dims × 2 bytes = 131,072 bytes). A 4,000-token prompt therefore moves:</p>
<div class="eq">4000 tokens × 131,072 B = 524,288,000 B ≈ 500 MiB
link 25 Gb/s = 3.125 GB/s
transfer = 524.288 MB ÷ 3,125 MB/s ≈ 0.168 s
link ceiling = 3.125 GB/s ÷ 0.524 GB
             ≈ 5.96 req/s</div>
<p>Two consequences. First, each request waits an extra ≈ 0.17 s before decoding can start (whether that lands on TTFT or on the first inter-token gap depends on the stack). Second, and easier to miss, the <b>link is a shared resource</b>: at 10 req/s of 4k prompts you would need 5.2 GB/s, more than a 25 Gb/s link can carry at all, and the queue grows without bound.</p>
<p>Now compare with the saving. If prefill of 4,000 tokens takes about 0.75 s on your GPU (example value from the module's L4 range of 0.5 to 1 s; measure yours), a 0.17 s hand-over is a 22% tax on that prefill. Faster links (100 Gb/s, or RDMA-class fabrics in a training-style cluster) cut it to ≈ 0.04 s; the numbers for your network come from an <code>iperf3</code> pod-to-pod test, not from a spec sheet. <b>UNVERIFIED here:</b> real transfer engines overlap the copy with compute layer by layer, which hides part of this cost.</p>`,
        check: { q: "4,000-token prompts, 128 KiB of KV per token, a 25 Gb/s link. Roughly how many requests per second can that link carry at most?",
          options: ["About 0.6", "About 6", "About 60"], answer: 1,
          why: "Each request is 524.288 MB, or 4.19 Gb. 25 Gb/s ÷ 4.19 Gb ≈ 5.96 req/s. Above that the link is the bottleneck whatever the GPUs can do. This is why disaggregation on slow networks disappoints, and why the lesson says to measure the network first." },
        scene(G) {
          const sc = 440, x0 = 24; // seconds -> px: 1 s = 440 px? scale so 0.75 s -> 330
          const bars = [
            ["prefill 4k tokens (example)", 0.75, "v"],
            ["send 500 MiB at 25 Gb/s", 0.168, "hot"],
            ["send 500 MiB at 100 Gb/s", 0.042, "ok"],
          ];
          G.text(24, 36, "time per request, to scale (seconds)", { size: 13 });
          const rs = bars.map(([l, t, c], i) => {
            const y = 66 + i * 70;
            G.label(x0, y - 6, l, { size: 12, color: "ink" });
            const r = G.rect(x0, y, t * sc, 30, { fill: c, rx: 4 });
            G.text(x0 + t * sc + 8, y + 21, t.toFixed(3) + " s", { size: 13 });
            return r;
          });
          G.rect(24, 288, 592, 112, { stroke: "line", rx: 10 });
          G.text(40, 312, "the link also has a ceiling", { size: 13, color: "ink" });
          G.text(40, 338, "25 Gb/s ÷ 4.19 Gb per request ≈ 5.96 req/s", { size: 13, color: "hot" });
          G.text(40, 362, "100 Gb/s ÷ 4.19 Gb per request ≈ 23.8 req/s", { size: 13, color: "ok" });
          G.label(40, 386, "KV per token 128 KiB (Llama-3-8B, bf16) · link speeds are example values", {});
          G.from(rs, { attr: { width: 0 }, stagger: 0.3, duration: 0.5 });
          G.caption("the hand-over is cheap on a fast fabric and fatal on a slow one");
        } },

      /* 8 ------------------------------------------------------------ */
      { rail: "choosing", title: "Pick the smallest stack that solves a measured problem",
        body: `<p>Put the pieces together as questions to ask in order:</p>
<ol><li><b>One model, steady traffic, one or two replicas?</b> Use the plain Deployment. Fewest moving parts, full control.</li>
<li><b>Many models, canaries, a platform team that wants one declarative object per model?</b> KServe, in RawDeployment mode until you really need scale-to-zero.</li>
<li><b>A pipeline: pre-processing, several models, custom Python?</b> Ray Serve.</li>
<li><b>Shared prefixes (chat, agents, RAG with a fixed system prompt) across many replicas, and a cache hit rate you can measure falling?</b> Prefix-aware routing: llm-d's scheduler, Dynamo's router, or session affinity in your own gateway.</li>
<li><b>Long prompts, strict ITL targets under bursty load, a fast interconnect, enough GPUs for two pools?</b> Disaggregation, through Dynamo or llm-d. Measure first (exercise 4).</li></ol>
<p>Common mistakes the module warns about: comparing stacks at different vLLM versions or flags; running KServe in Knative mode before you need scale-to-zero; disaggregating on a slow network and blaming the framework; and session affinity without failover.</p>
<p>Both Dynamo and llm-d move fast. This course pins Dynamo v1.5.0 and llm-d v0.10.0 and tells you to follow the pinned repository's deployment guide (<code>platform/deploy/dynamo/README.md</code> and <code>platform/deploy/llmd/README.md</code>) rather than copying manifests that will go stale.</p>`,
        scene(G) {
          const q = (x, y, w, t) => G.box(x, y, w, 46, t, { stroke: "line", color: "ink", size: 11 });
          const a = (x, y, w, t, c) => G.box(x, y, w, 40, t, { fill: c, size: 12 });
          q(24, 40, 330, "one model, steady load, 1-2 replicas?"); a(420, 40, 196, "plain Deployment", "w");
          q(24, 108, 330, "many models, canaries, one object each?"); a(420, 108, 196, "KServe (RawDeployment)", "k");
          q(24, 176, 330, "multi-model Python pipeline?"); a(420, 176, 196, "Ray Serve", "q");
          q(24, 244, 330, "cache hit rate falls as replicas grow?"); a(420, 244, 196, "prefix-aware routing", "ok");
          q(24, 312, 330, "long prompts, strict ITL, fast network?"); a(420, 312, 196, "disaggregation", "v");
          [0, 1, 2, 3, 4].forEach((i) => G.arrow(356, 63 + i * 68, 416, 60 + i * 68, { color: "ok", w: 2 }));
          G.label(24, 394, "yes → that row's answer;  otherwise go to the next row", { color: "ink" });
          G.caption("adopt a stack for a measured problem, not for the logo");
        } },

      /* 9 ------------------------------------------------------------ */
      { rail: "#1 cluster", title: "Project #1: a cluster that rolls out with zero dropped requests",
        body: `<p>Project #1 is the self-hosted inference cluster: EKS with a tainted GPU node group, the GPU Operator and Prometheus, the vLLM Deployment from <code>platform/deploy/base/vllm.yaml</code>, and the #6 gateway in front as the <b>only entry point</b>. It is done when the checklist in <code>results/p34_cluster.md</code> is true, including a rolling update with <b>0 failed requests</b>.</p>
<p>Why that is harder with GPUs than with web servers. The Deployment says <code>maxUnavailable: 0, maxSurge: 1</code>: Kubernetes must start the new pod and wait until it is <i>Ready</i> before it may stop the old one. The new pod needs a free GPU. With one GPU per node and a pool that scales from zero, the surge pod makes the cluster add a node (minutes), pull the image and load the weights. Only then does the old pod receive the <code>preStop</code> (<code>sleep 20</code>, so the endpoint removal reaches the gateway) and a <code>terminationGracePeriodSeconds</code> of 120 s to finish its streams. If any link fails (the probe passes too early, the sleep is too short, the grace period too short) a user sees a broken stream. <code>drain_test.py</code> detects exactly that, and you record how long the surge waited for a node.</p>
<p><b>Cost, auto-stop and teardown (EKS).</b> You pay for the EKS control plane, the NAT gateway and the system node every hour the cluster exists, plus GPU nodes while pods run (look up your region's prices; this module needs about 6 GPU-node-hours). The GPU group is min 0, so it scales back to zero with no pods, and the budget alarm in <code>infra/aws/guardrails</code> is the backstop. The control plane and NAT keep billing until you run <code>make down</code>. The vLLM Service stays <code>ClusterIP</code>; reach it with <code>kubectl port-forward</code> and tenant keys from a Secret. No unauthenticated public endpoint.</p>`,
        check: { q: "A rollout with maxUnavailable 0 and maxSurge 1 on a scale-from-zero GPU pool drops requests at the very end of the old pod's life. Which setting do you check first?",
          options: ["The surge pod's GPU request", "The preStop sleep and terminationGracePeriodSeconds on the old pod", "The Service type"], answer: 1,
          why: "By then the new pod is Ready, so capacity exists. Drops at the end mean the old pod was still receiving requests (the endpoint removal had not reached the gateway: preStop too short) or was killed before its streams finished (grace period too short)." },
        scene(G) {
          const x0 = 24, X = (s) => x0 + s * 1.3;
          G.text(24, 34, "rolling update, one GPU per pod (phase widths illustrative)", { size: 13 });
          G.label(24, 64, "old pod", { size: 12, color: "ink" });
          const old = G.rect(x0, 72, 560, 26, { fill: "ok", rx: 4, opacity: 0.85 });
          G.text(x0 + 8, 90, "serves traffic", { size: 12, color: "bg" });
          G.label(24, 128, "new pod (surge)", { size: 12, color: "ink" });
          const ph = [["node", "blue", 150], ["pull", "v", 55], ["load", "q", 70], ["warm", "k", 40]];
          let x = x0 + 60;
          const bars = ph.map(([l, c, w]) => { const r = G.rect(x, 136, w, 26, { fill: c, rx: 4 }); G.text(x + w / 2, 154, l, { size: 11, anchor: "middle", color: "bg" }); x += w; return r; });
          const rdy = x; G.line(rdy, 70, rdy, 200, { color: "hot", dash: "4 3" }); G.label(rdy + 4, 214, "new pod Ready", { color: "hot", size: 11 });
          const ps = G.rect(rdy, 72, 60, 26, { fill: "v", rx: 4 }); G.text(rdy + 30, 90, "preStop", { size: 11, anchor: "middle", color: "bg" });
          G.rect(rdy, 172, 200, 26, { fill: "hot", rx: 4, opacity: 0.8 }); G.text(rdy + 8, 190, "new pod serves", { size: 11, color: "bg" });
          G.label(24, 214, "Pending until a node exists", { size: 11 });
          G.rect(24, 250, 592, 150, { stroke: "line", rx: 10 });
          G.text(40, 274, "#1 is done when", { size: 13, color: "ink" });
          ["gateway is the only entry; keys in a Secret", "Ready probes + drain survive a rollout: 0 failed", "latency report from #4 (report.py) attached", "make down run; cost.sh confirms teardown"].forEach((t, i) => G.text(40, 300 + i * 24, "[ ] " + t, { size: 12, color: "muted" }));
          G.from(bars, { attr: { width: 0 }, stagger: 0.25, duration: 0.4 });
          G.caption("the old pod must outlive the new pod's whole cold start");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A Deployment keeps pods alive but ignores the KV cache, queue depth and release strategy. A serving stack adds some of these, always at the price of more moving parts.</li>
<li>KServe trades hand-written objects for an InferenceService and a controller. Ray Serve composes Python pipelines. Neither changes the engine's speed; compare them with the same image and flags.</li>
<li>The prefix cache is per replica, so round-robin makes the hit rate fall as replicas grow (about 1/N). Rendezvous hashing pins a session to a backend and moves only the dead backend's sessions.</li>
<li>Disaggregation separates compute-bound prefill from memory-bound decode and ships the KV between pools. 4,000 tokens of Llama-3-8B KV is about 500 MiB, 0.17 s at 25 Gb/s, and caps that link near 6 req/s.</li>
<li>A zero-drop rollout on GPUs needs a Ready probe that tells the truth, a preStop that outlasts endpoint propagation and a grace period long enough for streams.</li></ul>`,
    sim: {
      title: "Does disaggregation pay on this network?",
      intro: "Choose the model, prompt length, link speed and request rate. The simulator computes the KV bytes that must move from the prefill pool to the decode pool, how long that takes, and how much of the link the traffic uses. Prefill speed is an example value: measure yours with <code>loadgen</code> (P2.4). Press Run to watch one request's timeline.",
      height: 330,
      controls: [
        { id: "kv", label: "model (KV per token)", type: "select", value: 131072, options: [[131072, "Llama-3-8B · 128 KiB"], [327680, "Llama-3-70B · 320 KiB"]] },
        { id: "prompt", label: "prompt (tokens)", min: 250, max: 32000, step: 250, value: 4000 },
        { id: "pfs", label: "prefill tokens/s (example)", min: 1000, max: 40000, step: 1000, value: 5000 },
        { id: "link", label: "link, Gb/s (iperf3)", min: 5, max: 400, step: 5, value: 25 },
        { id: "rate", label: "requests/s", min: 0.5, max: 24, step: 0.5, value: 2 },
      ],
      run: { label: "Run one request", frames: 40, ms: 50 },
      draw(G, v, t) {
        const bytes = v.kv * v.prompt, gb = (bytes * 8) / 1e9;
        const tx = gb / v.link, pf = v.prompt / v.pfs, maxRate = v.link / gb, util = (v.rate * gb) / v.link;
        const tot = pf + tx, now = t * tot, sc = 560 / Math.max(tot, 0.01);
        G.axes(40, 40, 580, 120, {});
        G.text(44, 36, "one request: prefill pool, then KV transfer, then decode starts", { size: 11, color: "muted" });
        G.rect(44, 70, pf * sc, 34, { fill: "line", rx: 3 }); G.rect(44, 70, Math.min(pf, now) * sc, 34, { fill: "v", rx: 3 });
        G.rect(44 + pf * sc, 70, tx * sc, 34, { fill: "line", rx: 3 }); G.rect(44 + pf * sc, 70, Math.max(0, Math.min(tx, now - pf)) * sc, 34, { fill: "hot", rx: 3 });
        G.text(44, 128, `prefill ${F.ms(pf)}`, { size: 12, color: "v" });
        G.text(44 + Math.min(pf * sc + 8, 400), 148, `KV transfer ${F.ms(tx)}`, { size: 12, color: "hot" });
        G.line(44 + now * sc, 54, 44 + now * sc, 160, { color: "ink", dash: "3 3" });
        // utilisation curve vs request rate
        const x0 = 40, y0 = 300, w = 580, h = 70, rmax = 24;
        G.label(x0, y0 - h - 10, "link utilisation vs request rate", { size: 11 });
        G.axes(x0, y0 - h, w, h, {});
        const Y = (u) => y0 - Math.min(u, 1.5) / 1.5 * h, X = (r) => x0 + (r / rmax) * w;
        G.line(x0, Y(1), x0 + w, Y(1), { color: "hot", dash: "5 4" }); G.label(x0 + w - 4, Y(1) - 4, "100% of link", { size: 10, color: "hot", anchor: "end" });
        G.path(`M ${X(0)} ${Y(0)} L ${X(rmax)} ${Y((rmax * gb) / v.link)}`, { color: "k", w: 2.5 });
        G.circle(X(v.rate), Y(util), 5, { fill: util > 1 ? "hot" : "ok" });
        [0, 6, 12, 18, 24].forEach((r) => G.label(X(r), y0 + 14, r + " req/s", { size: 10, anchor: "middle" }));
        return [
          { title: "KV hand-over", rows: [["KV per request", F.bytes(bytes)], ["data on the wire", F.num(gb, 2) + " Gb"], ["transfer time", F.ms(tx)], ["prefill time", F.ms(pf)], ["added to prefill", F.num((100 * tx) / pf, 0) + "%"]],
            chip: [tx < 0.25 * pf, tx < 0.25 * pf ? "hand-over is small next to prefill" : "hand-over rivals prefill: weigh it carefully"] },
          { title: "The link", gauge: [[Math.min(util, 1), util > 1 ? "hot" : util > 0.7 ? "v" : "ok"]], gaugeText: `${F.num(util * 100, 0)}% of ${v.link} Gb/s`,
            rows: [["traffic", F.num(v.rate * gb / 8, 2) + " GB/s"], ["link carries at most", F.num(maxRate, 1) + " req/s"]],
            chip: [util <= 0.7, util > 1 ? "over capacity: the queue grows without bound" : util > 0.7 ? "busy: bursts will queue" : "headroom"] },
          { title: "Model", html: `<p class="note">KV bytes = layers × 2 × KV heads × head_dim × 2 bytes per token (Llama-3-8B: 32 × 2 × 8 × 128 × 2 = 131,072; Llama-3-70B: 80 × 2 × 8 × 128 × 2 = 327,680). Transfer = bytes ÷ link speed, with no protocol overhead and no overlap with compute, so it is an upper bound on the hand-over; real engines overlap copy and compute and the wire never reaches 100%. Whether the saved decode interference outweighs this cost is what exercise 4 measures.</p>` },
        ];
      },
    },
    practice: {
      intro: "Exercise 5 is T0 (pytest, no cluster). Exercises 1 to 4 are T3 on EKS (about 6 GPU-node-hours); rehearse the rollout on kind first. Every T3 session starts with <code>make up</code> and ends with <code>make down</code> then <code>cost.sh</code>.",
      items: [
        { title: "Same model, two stacks, one table", tier: "T3", goal: "Deploy the plain Deployment and KServe RawDeployment with the same image and flags, run loadgen at the same rates, and tabulate knee, TTFT, TPOT and objects managed. Measure noise by repeating one stack three times.",
          cmd: "PYTHONPATH=platform python -m loadgen.cli --url http://127.0.0.1:8000 --rates 1 2 4 --duration 60 --out results/p34_deploy.json" },
        { title: "Rolling update with zero dropped requests", tier: "T3 (rehearse on kind)", goal: "Run drain_test.py inside the cluster while you restart the deployment; it must exit 0. If it fails, find which mechanism broke: preStop, grace period, readiness probe, or the surge waiting for a node.",
          cmd: "kubectl -n s2s rollout restart deploy/vllm" },
        { title: "#1 checklist and cluster report", tier: "T3", goal: "Complete the checklist and write results/p34_cluster.md with Setup, Architecture, Health checks, Results, Rollout and Teardown sections.",
          cmd: "uv run python course/P3-deployment-and-infra/P3.4-serving-stacks-on-k8s/exercises/check_cluster_report.py results/p34_cluster.md" },
        { title: "Disaggregated vs aggregated", tier: "T3 · hard (2 GPUs)", goal: "On Dynamo or llm-d, compare 2 aggregated replicas with 1 prefill + 1 decode at long prompts. Explain the result with this lesson's KV-bytes and bandwidth arithmetic, using your own iperf3 and prefill measurements.",
          cmd: "PYTHONPATH=platform python -m loadgen.cli --url <gateway> --prompt-mean 4000 --output-mean 200 --rates 0.5 1 2 4 --out results/p34_disagg.json" },
        { title: "Session-affinity routing in the gateway", tier: "T0", goal: "Add rendezvous hashing on X-Session-Id to Router.candidates in platform/gateway/router.py: stable, evenly spread, and only a failed backend's sessions move.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.4-serving-stacks-on-k8s/exercises/test_affinity.py" },
      ],
      labs: [
        { label: "Manifest checks for the serving stacks (T0)", path: "course/P3-deployment-and-infra/P3.4-serving-stacks-on-k8s/exercises/test_stack_manifests.py" },
        { label: "KServe InferenceService (field names UNVERIFIED)", path: "platform/deploy/kserve/inferenceservice.yaml" },
        { label: "Ray Serve app and RayService", path: "platform/deploy/rayserve/" },
        { label: "Dynamo v1.5.0 deployment contract", path: "platform/deploy/dynamo/README.md" },
        { label: "llm-d v0.10.0 deployment contract", path: "platform/deploy/llmd/README.md" },
        { label: "vLLM Deployment with probes, drain and PDB", path: "platform/deploy/base/vllm.yaml" },
        { label: "EKS cluster guide (cost, access, teardown)", path: "infra/aws/eks/README.md" },
        { label: "Request path animation, cold vs warm", path: "animations/p3-request-path.html" },
      ],
    },
  });
})();
