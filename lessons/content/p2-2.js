/* P2.2 — SGLang and its runtime. A second engine, the flag map, and the radix-tree prefix cache that sets it apart. */
(function () {
  const F = S2S.fmt;

  // draw a radix-tree node: rounded box with a token-run label and a size
  function node(G, x, y, w, label, sub, o = {}) {
    const g = G.group();
    G.rect(x, y, w, 40, { fill: o.fill, stroke: o.stroke ?? (o.fill ? undefined : "ink"), rx: 8, opacity: o.opacity, parent: g, dash: o.dash });
    G.text(x + w / 2, y + 17, label, { anchor: "middle", size: 12, color: o.fill ? "bg" : (o.color || "ink"), parent: g });
    if (sub) G.text(x + w / 2, y + 32, sub, { anchor: "middle", size: 10, color: o.fill ? "bg" : "muted", parent: g });
    return g;
  }
  const edge = (G, x1, y1, x2, y2, col) => G.path(`M ${x1} ${y1} C ${x1} ${(y1 + y2) / 2}, ${x2} ${(y1 + y2) / 2}, ${x2} ${y2}`, { color: col || "muted", w: 1.5 });

  S2S.lesson({
    id: "p2-2", n: "P2.2", title: "SGLang and its runtime",
    subtitle: "Serving engines · first principles · T2 (one GPU), tests T0",
    kicker: "Lesson · ≈ 35 min",
    headline: "Never prefill the same words twice",
    intro: `<p>vLLM is not the only engine. SGLang does the same job with the same building blocks, but it was designed around one observation: real traffic repeats itself. Chat turns resend the whole conversation, agents resend the same instructions, few-shot prompts resend the same examples. This lesson maps SGLang onto what you know from P2.1, then builds its central idea, a <b>radix-tree prefix cache</b>, from the causal mask you met in P1.2.</p>`,
    facts: ["10 steps", "4 checkpoints", "1 simulator", "3 exercises"],
    legend: [["k", "cached KV"], ["hot", "recomputed"], ["ok", "new tokens"], ["q", "scheduler"], ["w", "weights"]],
    prev: "p2-1", next: "p2-3",
    steps: [
      { rail: "same job", title: "Same job, different design choices",
        body: `<p>SGLang (pinned at <code>v0.5.21</code>) is a serving engine like vLLM: an HTTP server with the OpenAI-compatible API, a scheduler that batches every running request into one forward pass per step, a KV cache handed out in pages, and a model runner with CUDA graphs.</p>
<p>So why learn two? Because "which engine?" is a question about <b>workloads</b>, and the engines differ exactly where workloads differ:</p>
<ul><li><b>Prefix cache design.</b> SGLang keeps reusable KV in a <b>radix tree</b> (this lesson). vLLM hashes fixed-size blocks (P2.3).</li>
<li><b>Scheduling policy.</b> SGLang can reorder its queue to maximise cache hits.</li>
<li><b>Kernels and defaults</b>: attention backends, chunk sizes, memory fractions.</li>
<li><b>Frontend.</b> SGLang also ships a small language for structured generation.</li></ul>
<p>The honest answer to "vLLM or SGLang?" is a measured table for your workload. Building that table fairly is the skill.</p>`,
        scene(G) {
          const parts = ["HTTP API", "scheduler", "paged KV cache", "model runner"];
          [["vLLM v0.31.0", 24], ["SGLang v0.5.21", 336]].forEach(([t, x]) => {
            G.rect(x, 40, 280, 300, { stroke: "line", rx: 12 });
            G.text(x + 140, 68, t, { anchor: "middle", size: 15 });
            parts.forEach((p, i) => G.box(x + 20, 88 + i * 54, 240, 40, p, { fill: i === 0 || i === 3 ? "w" : "k", size: 13 }));
          });
          G.label(164, 324, "prefix cache: block hashes", { anchor: "middle", color: "v" });
          G.label(476, 324, "prefix cache: radix tree", { anchor: "middle", color: "v" });
          const d = [G.circle(288, 175, 9, { fill: "v" }), G.circle(600, 175, 9, { fill: "v" }), G.circle(288, 229, 9, { fill: "v" }), G.circle(600, 229, 9, { fill: "v" })];
          G.label(24, 372, "same four parts; amber dots mark where the designs differ", { color: "ink" });
          G.pulse(d, { repeat: 5 });
          G.caption("compare engines on your workload, not on reputation");
        } },

      { rail: "flag map", title: "Translating the flags",
        body: `<p>Every vLLM flag from P2.1 has an SGLang counterpart, because both set the same physical quantities. The launch in <code>examples/01_serve.sh</code>:</p>
<div class="eq">python -m sglang.launch_server \\
  --model-path /opt/models/&lt;model&gt; \\
  --host 127.0.0.1 --port 30000 \\
  --api-key "$S2S_API_KEY" \\
  --context-length 8192 \\
  --mem-fraction-static 0.88</div>
<table><tr><th>quantity</th><th>vLLM</th><th>SGLang</th></tr>
<tr><td>memory fraction</td><td><code>--gpu-memory-utilization</code></td><td><code>--mem-fraction-static</code></td></tr>
<tr><td>max sequence</td><td><code>--max-model-len</code></td><td><code>--context-length</code></td></tr>
<tr><td>max running</td><td><code>--max-num-seqs</code></td><td><code>--max-running-requests</code></td></tr>
<tr><td>prefill chunk</td><td><code>--max-num-batched-tokens</code></td><td><code>--chunked-prefill-size</code></td></tr>
<tr><td>no CUDA graphs</td><td><code>--enforce-eager</code></td><td><code>--disable-cuda-graph</code></td></tr>
<tr><td>prefix cache off</td><td><code>--no-enable-prefix-caching</code></td><td><code>--disable-radix-cache</code></td></tr></table>
<p>These names are <b>UNVERIFIED</b> until exercise 1 captures each engine's <code>--help</code> at the pinned version and <code>test_flag_map.py</code> checks every row of <code>exercises/flag_map.yaml</code>. A renamed flag then fails a test, not a deployment.</p>`,
        check: { q: "You compare vLLM at gpu-memory-utilization 0.90 with SGLang at mem-fraction-static 0.88 on an 8B model, 24 GiB GPU. What is wrong with the comparison?",
          options: ["Nothing; the numbers are close enough", "The engines get different KV budgets, so concurrency differs for a reason unrelated to engine design", "SGLang ignores the memory fraction"], answer: 1,
          why: "Weights are fixed, so the 0.02 × 24 GiB ≈ 0.48 GiB difference lands entirely on the KV cache: roughly 0.48 of 5.6 GiB, about 8% fewer KV tokens in this example. Set equal KV capacity before comparing." },
        scene(G) {
          const rows = [["memory fraction", "--gpu-memory-utilization", "--mem-fraction-static"], ["max sequence", "--max-model-len", "--context-length"], ["max running", "--max-num-seqs", "--max-running-requests"], ["prefill chunk", "--max-num-batched-tokens", "--chunked-prefill-size"], ["no CUDA graphs", "--enforce-eager", "--disable-cuda-graph"], ["prefix cache off", "--no-enable-prefix-caching", "--disable-radix-cache"]];
          G.text(24, 36, "vLLM", { size: 14 }); G.text(400, 36, "SGLang", { size: 14 });
          const ls = rows.map(([c, a, b], i) => {
            const y = 60 + i * 56;
            G.box(24, y, 210, 34, a, { stroke: "k", size: 11, color: "k" });
            G.box(400, y, 216, 34, b, { stroke: "v", size: 11, color: "v" });
            const l = G.line(238, y + 17, 396, y + 17, { color: "muted", dash: "3 3" });
            G.label(317, y + 12, c, { anchor: "middle", size: 11, color: "ink" });
            return l;
          });
          G.from(ls, { opacity: 0, stagger: 0.1, duration: 0.3 });
          G.caption("same quantity, different spelling: verify against --help");
        } },

      { rail: "repetition", title: "Real traffic repeats itself",
        body: `<p>Look at what a chat client actually sends. The API is stateless: the server remembers nothing between requests, so turn <i>k</i> of a conversation resends the system prompt and every earlier message.</p>
<p><code>examples/02_multiturn.py</code> builds exactly this: a system prompt of about 1,500 tokens, a short question each turn (about 12 tokens) and a reply of up to 64 tokens. The prompt at turn <i>k</i> is roughly:</p>
<div class="eq">prompt(k) ≈ 1500 + (k − 1) × (12 + 64) + 12

turn 1:  1,512 tokens   (12 new)
turn 6:  1,892 tokens   (12 new)
6 turns: 10,212 prompt tokens, of which 72 are new</div>
<p>Without reuse, the engine prefills all 10,212 tokens. Prefill time grows with prompt length (P1.2), so TTFT grows turn after turn even though each turn only adds a dozen tokens. The same pattern appears in agents (the same tool instructions every call), RAG (the same documents for many questions) and few-shot prompts.</p>`,
        scene(G) {
          const T = 6, base = 1500, step = 76;
          G.text(24, 36, "prompt tokens sent at each turn of one conversation", { size: 13 });
          G.axes(60, 50, 540, 320, {});
          const sc = 250 / 2000;
          const bars = [];
          for (let k = 0; k < T; k++) {
            const x = 90 + k * 86, sys = base * sc, hist = k * step * sc, nw = 12 * sc;
            bars.push(G.rect(x, 370 - sys, 56, sys, { fill: "w", rx: 2 }));
            if (hist) bars.push(G.rect(x, 370 - sys - hist, 56, hist, { fill: "v", rx: 2, opacity: 0.8 }));
            bars.push(G.rect(x, 370 - sys - hist - Math.max(4, nw), 56, Math.max(4, nw), { fill: "ok", rx: 1 }));
            G.label(118 + k * 86, 390, "turn " + (k + 1), { anchor: "middle", size: 11 });
            G.text(118 + k * 86, 362 - sys - hist - 6, String(base + k * step + 12), { anchor: "middle", size: 11 });
          }
          [["w", "system prompt (same every turn)"], ["v", "earlier messages (resent)"], ["ok", "new question"]].forEach(([c, l], i) => { G.rect(80, 58 + i * 20, 12, 12, { fill: c, rx: 2 }); G.label(98, 69 + i * 20, l, { size: 11, color: "ink" }); });
          G.from(bars, { opacity: 0, stagger: 0.03, duration: 0.2 });
          G.caption("almost every token sent was already seen by the server");
        } },

      { rail: "why reuse works", title: "Why a shared prefix has the same KV",
        body: `<p>Recall P1.2: attention is causal, so token <i>i</i>'s key and value in every layer depend only on tokens 1…<i>i</i>. Two consequences:</p>
<ul><li>If two prompts start with the <b>same tokens</b>, the keys and values for that shared start are <b>identical</b>, whatever comes after. Compute them once, reuse them.</li>
<li>If two prompts share an <b>ending</b> but differ earlier, the KV of the shared ending is <b>different</b> in each, because each token has looked at a different past. Only prefixes can be shared.</li></ul>
<p>A <b>prefix cache</b> keeps the KV of finished requests in GPU memory after they end. When a new request arrives, the engine finds the longest cached prefix of its tokens, points the request at that KV, and prefills only the remainder. TTFT then depends on the <b>uncached</b> tokens:</p>
<div class="eq">TTFT ≈ queue wait
     + uncached tokens × prefill cost per token</div>
<p>Matching is exact and token-by-token. One changed token, such as a timestamp at the top of a system prompt, makes everything after it a miss.</p>`,
        check: { q: "Request A is [system prompt][question 1]. Request B is [different intro][system prompt][question 1]. Can B reuse A's cached KV for the system prompt?",
          options: ["Yes, the system prompt tokens are the same", "No: in B those tokens come after a different intro, so their keys and values differ"], answer: 1,
          why: "Each token's KV depends on every token before it. In B the system prompt follows a different intro, so its KV is different. Only a shared start (a prefix) is reusable. Put stable content first and variable content last." },
        scene(G) {
          const A = ["You", "are", "a", "helpful", "bot", "Q1"], B = ["You", "are", "a", "helpful", "bot", "Q2"];
          G.label(24, 40, "request A (finished, KV kept)", { color: "ink" });
          A.forEach((t, i) => G.box(24 + i * 98, 52, 86, 34, t, { fill: "k", size: 12 }));
          G.label(24, 128, "request B (new)", { color: "ink" });
          const bb = B.map((t, i) => G.box(24 + i * 98, 140, 86, 34, t, { fill: i < 5 ? "k" : "ok", size: 12 }));
          G.rect(18, 46, 486, 134, { stroke: "k", dash: "6 4", rx: 10 });
          G.label(24, 200, "shared prefix → identical KV → reuse 5 tokens, prefill 1", { color: "k", size: 13 });
          G.label(24, 262, "request C: a different start", { color: "ink" });
          ["Hi!", "You", "are", "a", "helpful", "bot"].forEach((t, i) => G.box(24 + i * 98, 274, 86, 34, t, { fill: "hot", size: 12 }));
          G.label(24, 334, "same words, shifted after a new first token → every KV differs", { color: "hot", size: 13 });
          G.from(bb.slice(0, 5), { opacity: 0, y: -40, stagger: 0.08, duration: 0.35 });
          G.caption("only a common start can be shared");
        } },

      { rail: "radix tree", title: "Storing many prefixes: a radix tree",
        body: `<p>A server sees thousands of requests that share prefixes in a branching pattern: one system prompt, then many conversations, each with several turns. The natural structure is a <b>tree of token sequences</b>:</p>
<ul><li>The root is the empty sequence.</li><li>Each edge holds a run of tokens, plus pointers to where those tokens' KV sits in GPU memory.</li><li>Reading the edges from the root down to any node spells out one cached prefix.</li></ul>
<p>It is called a <b>radix tree</b> because runs of tokens with no branching are merged into one edge, instead of one node per token. SGLang's version is <code>python/sglang/srt/mem_cache/radix_cache.py</code>, and the idea is from the SGLang paper ("RadixAttention", arXiv 2312.07104).</p>
<p>A lookup walks down from the root, comparing the request's tokens with each edge, and stops at the first mismatch. Everything matched is a hit. The cost of the walk is proportional to the prompt length, which is tiny next to running the model over those tokens.</p>`,
        scene(G) {
          const root = node(G, 270, 30, 100, "root", "");
          const sys = node(G, 230, 100, 180, "system prompt", "1,500 tokens", { fill: "w" });
          edge(G, 320, 70, 320, 100);
          const convs = [["conv 1: Q1 A1", 40], ["conv 2: Q1 A1", 250], ["conv 3: Q1", 460]];
          const cs = convs.map(([t, x]) => { edge(G, 320, 140, x + 70, 190); return node(G, x, 190, 140, t, "76 tokens", { fill: "k" }); });
          edge(G, 110, 230, 110, 270); const t2 = node(G, 40, 270, 140, "Q2 A2", "76 tokens", { fill: "k" });
          edge(G, 110, 310, 110, 340); const t3 = node(G, 40, 340, 140, "Q3", "12 tokens", { fill: "ok" });
          edge(G, 320, 230, 320, 270); node(G, 250, 270, 140, "Q2 A2", "76 tokens", { fill: "k" });
          G.label(400, 300, "path root → leaf =", { color: "ink" }); G.label(400, 318, "one cached prefix", { color: "ink" });
          [[270, 30, 100], [230, 100, 180], [40, 190, 140], [40, 270, 140], [40, 340, 140]].forEach(([x, y, w]) => G.rect(x - 4, y - 4, w + 8, 48, { stroke: "v", sw: 2.5, rx: 10, dash: "5 3" }));
          G.label(196, 372, "← lookup for conv 1, turn 3", { color: "v" });
          G.from([sys, ...cs, t2, t3], { opacity: 0, y: -10, stagger: 0.15, duration: 0.3 });
          G.caption("one system prompt stored once, conversations branch below it");
        } },

      { rail: "insert & split", title: "Inserting: when a new prefix branches mid-edge",
        body: `<p>When a request finishes, SGLang inserts its full token sequence (prompt plus generated output) into the tree, so the next turn can reuse the reply's KV too. Insertion walks down like a lookup. Three cases:</p>
<ol><li><b>Full match.</b> The sequence is already there: nothing to add.</li>
<li><b>Runs off the end of a node.</b> Add a new child edge holding the remaining tokens.</li>
<li><b>Diverges in the middle of an edge.</b> <b>Split</b> that edge in two at the divergence point, then add the new branch under the split.</li></ol>
<div class="eq">edge "You are a helpful bot. Answer briefly."
insert "You are a helpful bot. Use JSON."
→ split after "You are a helpful bot."
   ├─ "Answer briefly."
   └─ "Use JSON."</div>
<p>Splitting moves no KV data. The KV for each token stays where it is in GPU memory; only the tree's bookkeeping changes, so a split is cheap.</p>`,
        scene(G) {
          G.label(24, 36, "before", { color: "ink", size: 13 });
          node(G, 24, 50, 120, "root", "");
          edge(G, 84, 90, 84, 120);
          node(G, 24, 120, 260, "You are a helpful bot.", "Answer briefly.", { fill: "k" });
          G.label(24, 200, "insert:", { color: "ok", size: 12 }); G.label(24, 218, "You are a helpful bot. Use JSON.", { color: "ok", size: 12 });
          G.line(316, 40, 316, 400, { color: "line", dash: "3 3" });
          G.label(340, 36, "after the split", { color: "ink", size: 13 });
          node(G, 400, 50, 120, "root", "");
          edge(G, 460, 90, 460, 120);
          const mid = node(G, 360, 120, 200, "You are a helpful bot.", "shared", { fill: "k" });
          edge(G, 420, 160, 410, 220); edge(G, 500, 160, 540, 220);
          const l = node(G, 336, 220, 148, "Answer briefly.", "old branch", { fill: "k" });
          const r = node(G, 494, 220, 122, "Use JSON.", "new branch", { fill: "ok" });
          G.label(340, 300, "no KV is copied: each token's KV", { color: "muted" });
          G.label(340, 318, "stays in its GPU slot; only the", { color: "muted" });
          G.label(340, 336, "tree's edges are rearranged", { color: "muted" });
          G.from([mid, l, r], { opacity: 0, y: -12, stagger: 0.25, duration: 0.35 });
          G.caption("divergence inside an edge splits it in two");
        } },

      { rail: "eviction", title: "Memory runs out: evict the least recently used leaves",
        body: `<p>Cached KV lives in the same GPU memory as the KV of running requests. When new requests need room, the cache must give some back. Which part?</p>
<ul><li>Only <b>leaves</b> can go: removing an inner node would cut off every prefix below it.</li>
<li>Never evict what a running request is using. Each node has a <b>reference count</b> of running requests that pass through it; nodes with a count above zero are locked.</li>
<li>Among the rest, evict the <b>least recently used</b> (LRU) leaf first, then repeat, since its parent may now be a leaf.</li></ul>
<p>The shared system prompt is touched by every request, so it is always recent and stays. A conversation the user abandoned an hour ago is evicted first. When the cache is too small for the working set, though, LRU can evict each conversation just before its next turn arrives: the hit rate then collapses to the system prompt only. The simulator shows this.</p>`,
        check: { q: "The cache is full. Which node does SGLang's LRU policy evict first?",
          options: ["The system-prompt node, because it is the largest", "An unlocked leaf that was used least recently", "Any node used by a running request, because it will be recomputed anyway"], answer: 1,
          why: "Inner nodes hold prefixes that other entries depend on, and locked nodes (reference count above zero) are being read by running requests. Among unlocked leaves, the least recently used is the least likely to be needed again soon." },
        scene(G) {
          node(G, 230, 30, 180, "system prompt", "used 1 s ago · ref 3", { fill: "w" });
          const xs = [24, 184, 344, 504], info = [["conv 1", "8 s ago", "ref 1", "k"], ["conv 2", "2 min ago", "ref 0", "k"], ["conv 3", "47 min ago", "ref 0", "hot"], ["conv 4", "3 s ago", "ref 0", "k"]];
          const nn = xs.map((x, i) => { edge(G, 320, 70, x + 56, 140); return node(G, x, 140, 112, info[i][0], info[i][1], { fill: info[i][3] }); });
          info.forEach(([, , r], i) => G.label(xs[i] + 56, 200, r, { anchor: "middle", color: r === "ref 1" ? "q" : "muted" }));
          G.rect(18, 132, 124, 56, { stroke: "q", rx: 10, sw: 2 });
          G.label(24, 228, "ref 1 = a running request reads it: locked", { color: "q" });
          G.text(344, 260, "✕ evict first", { color: "hot", size: 14 });
          G.label(24, 300, "order: oldest unlocked leaf → its parent if it became a leaf → …", { color: "ink" });
          G.label(24, 324, "the system prompt is touched by every request, so it stays", { color: "muted" });
          G.pulse(nn[2], { repeat: 6 });
          G.caption("least recently used, unlocked leaves go first");
        } },

      { rail: "scheduling", title: "Cache-aware scheduling: reorder the queue",
        body: `<p>A cache hit only helps if the cached KV is still there when the request runs. When many requests wait, the <b>order</b> in which they are admitted changes the hit rate.</p>
<p>First-come, first-served ignores the cache. SGLang's <code>--schedule-policy</code> can instead prefer the waiting requests with the <b>longest prefix match</b> ("lpm"). Requests that share a prefix then run close together, while that prefix is hot, and fewer tokens need prefill overall.</p>
<p>The cost is <b>fairness</b>: a request that shares nothing with anyone keeps getting passed over, and its queue wait (and so its TTFT) grows. Every scheduling policy trades total throughput against the worst-case wait of some request; P6.2 builds schedulers and measures this.</p>`,
        check: { q: "With longest-prefix-match scheduling, which request is most likely to wait longer than under FCFS?",
          options: ["A request that shares the popular system prompt", "A one-off request whose prompt shares nothing with the cache", "The first request of the day"], answer: 1,
          why: "LPM admits requests with long cache matches first. A request with no match ranks last every time new matching requests arrive, so its queue time grows: throughput is bought with fairness." },
        scene(G) {
          const q = [["r1", 0, "hot"], ["r2", 1500, "k"], ["r3", 1576, "k"], ["r4", 0, "hot"], ["r5", 1500, "k"]];
          G.label(24, 40, "waiting queue (arrival order) · number = cached prefix tokens", { color: "ink" });
          q.forEach(([r, m, c], i) => { G.box(24 + i * 120, 54, 104, 44, r, { fill: c, size: 13 }); G.label(76 + i * 120, 116, String(m), { anchor: "middle", color: "ink" }); });
          G.label(24, 170, "FCFS admits:", { color: "muted" });
          G.row(140, 154, ["r1", "r2", "r3", "r4", "r5"], { w: 80, h: 30, gap: 10, fill: (i) => (i === 0 || i === 3 ? "hot" : "k"), size: 12 });
          G.label(24, 240, "LPM admits:", { color: "muted" });
          const lp = G.row(140, 224, ["r3", "r2", "r5", "r1", "r4"], { w: 80, h: 30, gap: 10, fill: (i) => (i > 2 ? "hot" : "k"), size: 12 });
          G.label(140, 290, "cache hits run together while the prefix is hot", { color: "k" });
          G.label(140, 312, "r1 and r4 (no match) drift to the back", { color: "hot" });
          G.from(lp, { x: 60, opacity: 0, stagger: 0.12, duration: 0.35 });
          G.caption("more hits, but some requests wait longer");
        } },

      { rail: "the loop", title: "Where SGLang forms a batch",
        body: `<p>Open <code>python/sglang/srt/managers/scheduler.py</code> at the pinned commit and find the event loop. Each iteration:</p>
<ol><li><b>Receive</b> new requests from the HTTP side and put them in the waiting queue.</li>
<li>Try to build a <b>prefill batch</b>: for each waiting request (in policy order), match its prefix in the radix cache, and admit it if its uncached tokens fit the token budget and its KV fits in memory.</li>
<li>If no prefill batch was built, run one <b>decode step</b> for the running batch.</li>
<li><b>Process outputs</b>: stream tokens back, finish requests, insert finished sequences into the tree, release references.</li></ol>
<p>vLLM's V1 scheduler (<code>vllm/v1/core/sched/scheduler.py</code>) does the same job with a single per-step token budget shared by prefill and decode. P2.3 explains that design; P6.2 has you write one.</p>`,
        scene(G) {
          const st = [["1 receive", 180, 40, "w"], ["2 prefill batch?", 420, 140, "k"], ["3 decode step", 180, 240, "ok"], ["4 process outputs", 420, 330, "q"]];
          const bs = st.map(([t, x, y, c]) => G.box(x - 90, y, 180, 46, t, { fill: c, size: 13 }));
          G.arrow(274, 66, 330, 150, { color: "ink" });
          G.arrow(330, 176, 274, 252, { color: "ink" }); G.label(250, 214, "nothing to admit", { color: "muted", size: 11, anchor: "end" });
          G.arrow(274, 270, 330, 344, { color: "ink" });
          G.arrow(420, 188, 420, 326, { color: "k" }); G.label(428, 262, "run prefill", { color: "k", size: 11 });
          G.path("M 330 360 C 40 360, 40 100, 88 66", { color: "muted", dash: "4 4", arrow: true });
          G.label(522, 132, "match prefix", { color: "ink", size: 11 }); G.label(522, 148, "in radix tree", { color: "ink", size: 11 });
          G.label(522, 330, "insert finished", { color: "ink", size: 11 }); G.label(522, 346, "into the tree", { color: "ink", size: 11 });
          G.from(bs, { opacity: 0, stagger: 0.2, duration: 0.3 });
          G.caption("the cache is consulted on admission and updated on finish");
        } },

      { rail: "fair comparison", title: "Measure, then explain",
        body: `<p>The multi-turn replay is the workload where the prefix cache should win. Predict first with the TTFT model from step 4, using the mock server's prefill cost as a stand-in (0.05 ms per token: the mock's default, which its README says is made up):</p>
<div class="eq">turn 6, no cache:  1,892 tokens × 0.05 ms ≈ 95 ms
turn 6, cached:       12 tokens × 0.05 ms ≈ 0.6 ms
                       (plus queueing and one decode step)</div>
<p>So with a working cache, TTFT stays nearly flat across turns; without one it climbs with the history. Exercise 2 first checks the climb on the cache-less mock (<code>test_multiturn_mock.py</code>), then fills a table on the GPU with the cache on and off in both engines.</p>
<p>A fair comparison keeps everything else equal: same model and revision, same KV capacity, same dataset, same sampling settings, warmed up, several runs. Then look for the workload where the other engine wins (exercise 3), and explain it with a mechanism you can point to in code or docs. With no shared prefixes, for instance, a prefix cache gives nothing and the comparison falls back to kernels and scheduling defaults.</p>`,
        scene(G) {
          const T = 6, pre = 0.05, sc = 260 / 100;
          G.text(24, 36, "predicted prefill part of TTFT per turn (mock cost 0.05 ms/token)", { size: 12 });
          G.axes(60, 60, 540, 280, { ylabel: "ms" });
          const off = [], on = [];
          for (let k = 0; k < T; k++) { off.push((1512 + 76 * k) * pre); on.push(12 * pre); }
          const px = (k) => 110 + k * 90, py = (v) => 340 - v * sc;
          const pOff = G.path("M " + off.map((v, k) => `${px(k)} ${py(v)}`).join(" L "), { color: "hot", w: 3 });
          const pOn = G.path("M " + on.map((v, k) => `${px(k)} ${py(v)}`).join(" L "), { color: "k", w: 3 });
          off.forEach((v, k) => { G.circle(px(k), py(v), 4, { fill: "hot" }); G.label(px(k), py(v) - 10, v.toFixed(0), { anchor: "middle", color: "hot", size: 11 }); G.label(px(k), 360, "turn " + (k + 1), { anchor: "middle", size: 11 }); });
          on.forEach((v, k) => G.circle(px(k), py(v), 4, { fill: "k" }));
          G.label(380, 300, "cache on: 12 new tokens ≈ 0.6 ms", { color: "k" });
          G.label(380, 160, "cache off: grows with history", { color: "hot" });
          G.from([pOff, pOn], { opacity: 0, stagger: 0.3, duration: 0.6 });
          G.caption("a model to test, not a result: measure it on both engines");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>SGLang and vLLM share the same four parts; they differ in prefix-cache design, scheduling policy, kernels and frontend.</li>
<li>Flags map one to one (<code>--mem-fraction-static</code> ↔ <code>--gpu-memory-utilization</code>, …); verify them against <code>--help</code> and compare engines at equal KV capacity.</li>
<li>Causal attention makes the KV of a shared <b>prefix</b> identical, so it can be reused; a shared suffix cannot.</li>
<li>A radix tree stores many prefixes once, matches the longest one per request, splits edges on divergence, and evicts unlocked least-recently-used leaves.</li>
<li>Cache-aware scheduling raises hit rates at a cost in fairness.</li>
<li>TTFT ≈ queue + uncached tokens × per-token prefill cost: multi-turn chat with a working cache stays flat.</li></ul>`,
    sim: {
      title: "Multi-turn chat against a prefix cache",
      intro: "Several conversations take turns round-robin against one server, like examples/02_multiturn.py. The chart shows the prefill part of TTFT at each turn with no cache (red) and with an LRU prefix cache of the capacity you choose (cyan). Shrink the cache below the working set and watch the hit rate collapse.",
      height: 280,
      controls: [
        { id: "sys", label: "system prompt tokens", min: 0, max: 4000, step: 100, value: 1500 },
        { id: "q", label: "question tokens per turn", min: 4, max: 400, step: 4, value: 12 },
        { id: "a", label: "reply tokens per turn (max_tokens)", min: 16, max: 1024, step: 16, value: 64 },
        { id: "turns", label: "turns per conversation", min: 2, max: 12, value: 6 },
        { id: "convs", label: "conversations (round-robin)", min: 1, max: 64, value: 8 },
        { id: "cap", label: "cache capacity, tokens", min: 1000, max: 100000, step: 1000, value: 40000 },
        { id: "pf", label: "prefill cost, ms per token (mock default 0.05; made up)", min: 0.01, max: 0.5, step: 0.01, value: 0.05, format: (x) => x.toFixed(2) },
      ],
      draw(G, v) {
        // segments: system (shared) and one per conversation (its cached history). Whole-segment LRU eviction: a simplification.
        const S = v.sys, Q = v.q, A = v.a, T = v.turns, N = v.convs, cap = v.cap;
        let clock = 0; const seg = { sys: { len: 0, t: 0 } }; for (let c = 0; c < N; c++) seg["c" + c] = { len: 0, t: 0 };
        const used = () => Object.values(seg).reduce((s, x) => s + x.len, 0);
        const onT = Array(T).fill(0), offT = Array(T).fill(0); let tot = 0, hit = 0;
        for (let k = 0; k < T; k++) for (let c = 0; c < N; c++) {
          clock++;
          const hist = k * (Q + A), prompt = S + hist + Q;
          let h = 0;
          if (seg.sys.len >= S) { h = S; if (seg["c" + c].len >= hist) h += hist; else h += 0; }
          else h = 0;
          tot += prompt; hit += h;
          onT[k] += (prompt - h) * v.pf / N; offT[k] += prompt * v.pf / N;
          seg.sys.len = S; seg.sys.t = clock;
          seg["c" + c].len = hist + Q + A; seg["c" + c].t = clock;
          // evict LRU segments (never the one just used) until within capacity
          while (used() > cap) {
            const cands = Object.entries(seg).filter(([key, x]) => x.len > 0 && x.t < clock);
            if (!cands.length) break;
            cands.sort((a, b) => (a[0] === "sys") - (b[0] === "sys") || a[1].t - b[1].t);
            cands[0][1].len = 0;
          }
        }
        const mx = Math.max(...offT, 0.001);
        G.label(10, 16, "prefill part of TTFT per turn, average over conversations (ms)", { size: 11 });
        G.rect(420, 7, 10, 10, { fill: "hot", rx: 2 }); G.label(434, 16, "no cache", { size: 11 });
        G.rect(510, 7, 10, 10, { fill: "k", rx: 2 }); G.label(524, 16, "LRU cache", { size: 11 });
        G.axes(40, 30, 580, 210, { xlabel: "turn →" });
        const bw = Math.min(40, 560 / T / 2.6);
        for (let k = 0; k < T; k++) {
          const x = 60 + (k * 560) / T, h1 = (200 * offT[k]) / mx, h2 = Math.max(1.5, (200 * onT[k]) / mx);
          G.rect(x, 240 - h1, bw, h1, { fill: "hot", rx: 2, opacity: 0.85 });
          G.rect(x + bw + 3, 240 - h2, bw, h2, { fill: "k", rx: 2 });
          G.text(x + bw, 258, String(k + 1), { anchor: "middle", size: 11, color: "muted" });
        }
        const ws = S + N * (T * (Q + A));
        return [
          { title: "Cache effect", gauge: [[hit / tot, "k"], [1 - hit / tot, "hot"]], gaugeText: `hit rate ${(100 * hit / tot).toFixed(1)}% of prompt tokens`,
            chip: [ws <= cap, ws <= cap ? "working set fits" : "working set exceeds cache"],
            rows: [["prompt tokens sent", F.num(tot)], ["prefilled, no cache", F.num(tot)], ["prefilled, with cache", F.num(tot - hit)], ["work saved", (100 * hit / tot).toFixed(1) + "%"]] },
          { title: "Last turn (prefill only)", rows: [["no cache", F.ms(offT[T - 1] / 1000)], ["with cache", F.ms(onT[T - 1] / 1000)], ["working set, tokens", F.num(ws)], ["cache capacity", F.num(cap)]],
            html: `<p class="note">Simplified model: whole conversations are evicted LRU (real radix caches trim leaf by leaf), queueing and decode are ignored, and the reply's KV is assumed reusable.</p>` },
        ];
      },
      note: "Costs are the mock server's made-up defaults, not GPU measurements. Use the shape, then measure the real thing in exercise 2.",
    },
    practice: {
      intro: "Use a separate venv for SGLang: its torch and flashinfer pins collide with vLLM's. Same instance and safety rules as P2.1 (aws.md).",
      items: [
        { title: "Flag-mapping table, verified by --help", tier: "T2 capture · T0 test", goal: "Capture both engines' help text at the pinned versions, then make <code>test_flag_map.py</code> pass and add three rows of your own.",
          cmd: "uv run pytest course/P2-serving-engines/P2.2-sglang/exercises/test_flag_map.py" },
        { title: "Multi-turn replay: measure the prefix-cache win", tier: "T0 test · T2", goal: "Check TTFT grows with history on the cache-less mock, then fill the cache on/off table for both engines and explain the growth column.",
          cmd: "uv run pytest course/P2-serving-engines/P2.2-sglang/exercises/test_multiturn_mock.py" },
        { title: "Find a workload where vLLM wins", tier: "T2 · hard", goal: "Find one workload each engine wins on goodput at equal memory fraction, and tie each result to a mechanism in code or docs.",
          cmd: "python course/P2-serving-engines/P2.2-sglang/examples/02_multiturn.py --url http://127.0.0.1:30000 --api-key \"$S2S_API_KEY\"" },
      ],
      labs: [
        { label: "Launch SGLang (localhost + API key)", path: "course/P2-serving-engines/P2.2-sglang/examples/01_serve.sh" },
        { label: "Multi-turn replay: TTFT per turn", path: "course/P2-serving-engines/P2.2-sglang/examples/02_multiturn.py" },
        { label: "The flag map under test", path: "course/P2-serving-engines/P2.2-sglang/exercises/flag_map.yaml" },
        { label: "AWS guide: cost, auto-stop, teardown", path: "course/P2-serving-engines/P2.2-sglang/aws.md" },
        { label: "Radix tree animation (rebuilt in P6.4)", path: "animations/p6-radix-tree.html" },
      ],
    },
  });
})();
