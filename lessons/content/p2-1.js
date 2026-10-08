/* P2.1 — First serve with vLLM. From "a model in a notebook" to a running, measured, safely exposed server. */
(function () {
  const F = S2S.fmt;
  const GiB = 2 ** 30;
  // Llama-3-8B (P1.1): 8,030,261,248 params → 16.06 GB = 14.96 GiB in bf16; KV 128 KiB/token (P1.2)
  const W8 = 14.96, KV_TOK = 131072;

  // a horizontal memory bar: parts = [[GiB, color, label], ...] drawn left to right over `total` GiB
  function memBar(G, x, y, w, h, total, parts, o = {}) {
    G.rect(x, y, w, h, { stroke: "ink", rx: 6 });
    let cx = x; const out = [];
    parts.forEach(([gb, col, lab, lc]) => {
      const pw = (w * gb) / total;
      out.push(G.rect(cx + 1, y + 1, Math.max(0, pw - 2), h - 2, { fill: col, rx: 4, opacity: o.opacity ?? 0.85 }));
      if (lab && pw > 46) G.text(cx + pw / 2, y + h / 2 + 5, lab, { anchor: "middle", size: 12, color: lc || "bg" });
      cx += pw;
    });
    return out;
  }

  // presets: shapes from the models' public config.json files (verify before citing; see platform/capacity/presets)
  const MODELS = {
    llama8: { name: "Llama-3-8B", L: 32, kv: 8, hd: 128, P: 8.030261 },
    mistral: { name: "Mistral-7B", L: 32, kv: 8, hd: 128, P: 7.24 },
    llama3b: { name: "Llama-3.2-3B", L: 28, kv: 8, hd: 128, P: 3.21 },
    llama1b: { name: "Llama-3.2-1B", L: 16, kv: 8, hd: 64, P: 1.24 },
  };
  // memory and bandwidth: the course's gpu_specs.yaml values, all UNVERIFIED there
  const GPUS = { L4: { mem: 24, bw: 300, fp8: true, bf16: true }, A10G: { mem: 24, bw: 600, fp8: false, bf16: true }, T4: { mem: 16, bw: 320, fp8: false, bf16: false } };

  S2S.lesson({
    id: "p2-1", n: "P2.1", title: "First serve with vLLM",
    subtitle: "Serving engines · first principles · T2 (one GPU), client parts T0",
    kicker: "Lesson · ≈ 40 min",
    headline: "From a function call to a service",
    intro: `<p>In P1 you treated the model as a function you call. A real deployment is different: many people send requests over the network at once, each one wants its tokens streamed back quickly, and one GPU has to serve all of them. A <b>serving engine</b> is the program that does this. This lesson runs the most widely used open-source one, vLLM, and explains every number it prints at start-up using the arithmetic you already know from P1.2, P1.4 and P1.5.</p>`,
    facts: ["10 steps", "4 checkpoints", "1 simulator", "4 exercises"],
    legend: [["w", "weights"], ["k", "KV cache"], ["q", "activations"], ["muted", "reserved"], ["hot", "problem"]],
    prev: "p1-5", next: "p2-2",
    steps: [
      { rail: "why a server", title: "One GPU, many users: why a serving engine exists",
        body: `<p>In P1.2 you wrote the generation loop for one prompt. Now imagine 50 people chatting with the same model. Running their loops one after another would make person 50 wait for 49 complete answers. Running 50 separate copies of the model would need 50 copies of the 16 GB of weights.</p>
<p>A <b>serving engine</b> keeps <b>one</b> copy of the weights on the GPU and runs everyone's decode steps <b>together</b>, as one batch (P1.2 step 9: one weight read serves every user in the batch). To do that it needs four parts:</p>
<ul><li>an <b>HTTP server</b> that accepts requests and streams tokens back;</li><li>a <b>scheduler</b> that decides, every step, which requests run;</li><li>a <b>KV-cache manager</b> that hands out GPU memory for each request's cache;</li><li>the <b>model runner</b> that executes one forward pass for the whole batch.</li></ul>
<p>vLLM (pinned in this course at <code>v0.31.0</code>) is one such engine. SGLang (P2.2) is another. Both have the same four parts.</p>`,
        scene(G) {
          const users = [];
          for (let i = 0; i < 6; i++) users.push(G.box(24, 40 + i * 58, 92, 40, "user " + (i + 1), { stroke: "ink", sw: 1, size: 12 }));
          const arrows = [];
          for (let i = 0; i < 6; i++) arrows.push(G.arrow(120, 60 + i * 58, 196, 205, { color: "muted", w: 1.2 }));
          G.rect(200, 40, 420, 340, { stroke: "k", dash: "6 4", rx: 12 });
          G.label(214, 62, "serving engine (vLLM)", { color: "k", size: 13 });
          G.box(220, 80, 180, 44, "HTTP server", { fill: "blue", size: 13 });
          G.box(220, 150, 180, 44, "scheduler", { fill: "q", size: 13 });
          G.box(420, 150, 180, 44, "KV-cache manager", { fill: "k", size: 13 });
          G.box(220, 220, 380, 44, "model runner: one forward pass", { fill: "ok", size: 13 });
          G.arrow(310, 126, 310, 146, { color: "ink" }); G.arrow(310, 196, 310, 216, { color: "ink" }); G.line(402, 172, 418, 172, { color: "ink" });
          G.rect(220, 290, 380, 70, { fill: "w", opacity: 0.5, rx: 8 });
          G.text(410, 320, "GPU: ONE copy of the weights", { anchor: "middle", size: 13 });
          G.text(410, 342, "+ every user's KV cache", { anchor: "middle", size: 13, color: "k" });
          G.from(arrows, { opacity: 0, stagger: 0.08, duration: 0.3 });
          G.caption("many requests in, one batched forward pass per step");
        } },

      { rail: "the API", title: "Talking to it: the OpenAI-compatible API",
        body: `<p>Clients talk to vLLM over HTTP with JSON, using the same request format as OpenAI's API. That shared format is why the same client script works against vLLM, SGLang and this repo's mock server.</p>
<div class="eq">POST /v1/chat/completions
Authorization: Bearer $S2S_API_KEY
{"model": "...", "max_tokens": 64, "temperature": 0,
 "messages": [{"role": "user", "content": "Hi"}],
 "stream": true}</div>
<p>Without <code>"stream": true</code> the server answers once, at the end, with the full text and a <code>usage</code> field (prompt and completion token counts). With streaming it sends <b>server-sent events</b> (SSE): one <code>data: {...}</code> line per new piece of text, then <code>data: [DONE]</code>.</p>
<p>Streaming is what makes the P1.3 metrics visible from the outside. The time to the first <code>data:</code> line is the <b>TTFT</b>; the gaps between later lines are the <b>ITL</b>. <code>examples/02_client.py</code> timestamps each line exactly this way.</p>`,
        scene(G) {
          G.box(24, 40, 120, 44, "client", { stroke: "ink" });
          G.box(496, 40, 120, 44, "vLLM :8000", { fill: "k" });
          G.arrow(150, 56, 490, 56, { color: "ink" }); G.label(320, 48, "POST /v1/chat/completions", { anchor: "middle", color: "ink" });
          G.line(80, 96, 80, 400, { color: "line" }); G.line(556, 96, 556, 400, { color: "line" });
          const ys = [170, 214, 244, 274, 304, 334];
          const ev = ys.map((y, i) => {
            const a = G.arrow(550, y, 88, y, { color: i === ys.length - 1 ? "muted" : "k", w: 1.5 });
            G.label(110, y - 6, i === ys.length - 1 ? "data: [DONE]" : `data: {"delta": "tok${i + 1}"}`, { size: 11, color: i === ys.length - 1 ? "muted" : "ink" });
            return a;
          });
          G.line(62, 70, 62, 170, { color: "v", w: 3 }); G.label(16, 126, "TTFT", { color: "v", size: 12 });
          G.line(36, 214, 60, 214, { color: "q" }); G.line(36, 244, 60, 244, { color: "q" });
          G.label(16, 234, "ITL", { color: "q", size: 12 });
          G.label(330, 140, "prefill happens here", { color: "muted", size: 12 });
          G.label(330, 390, "time flows down", { color: "muted", size: 11 });
          G.from(ev, { opacity: 0, stagger: 0.25, duration: 0.2 });
          G.caption("one SSE line per new chunk: the gaps are what you measure");
        } },

      { rail: "safe launch", title: "Starting it without exposing it",
        body: `<p>The minimal launch, from <code>examples/01_serve.sh</code>:</p>
<div class="eq">vllm serve /opt/models/&lt;model&gt; \\
  --host 127.0.0.1 --port 8000 \\
  --api-key "$S2S_API_KEY" \\
  --max-model-len 8192 \\
  --gpu-memory-utilization 0.90</div>
<p>Two flags are about safety, not speed. <code>--host 127.0.0.1</code> makes the server listen only on the machine's own loopback address, so nothing on the internet can reach it, even if a firewall rule is wrong. <code>--api-key</code> makes every request carry a secret. A GPU endpoint left open is a bill anyone can run up.</p>
<p>You still reach it from your laptop through an <b>SSM port-forward</b> (<code>make forward</code> in <code>infra/aws/single-node</code>): AWS's Session Manager opens a tunnel that is authenticated by your AWS identity, and your laptop's <code>localhost:8000</code> becomes the instance's <code>localhost:8000</code>. No inbound port is ever opened.</p>`,
        scene(G) {
          G.box(24, 150, 130, 60, "your laptop", { stroke: "ink" });
          G.label(30, 232, "localhost:8000", { color: "ink", size: 12 });
          G.rect(340, 80, 280, 230, { stroke: "ink", rx: 12 });
          G.label(352, 102, "g6.xlarge (1× L4)", { color: "ink", size: 13 });
          G.box(380, 160, 200, 50, "vllm on 127.0.0.1", { fill: "k", size: 13 });
          G.label(380, 236, "+ --api-key required", { color: "k", size: 12 });
          const tun = G.path("M 158 180 C 240 180, 280 185, 376 185", { color: "ok", w: 4, arrow: true });
          G.label(196, 168, "SSM tunnel", { color: "ok", size: 12 });
          G.label(196, 208, "(your AWS identity)", { color: "ok", size: 11 });
          G.box(150, 330, 140, 44, "internet", { stroke: "muted", color: "muted" });
          G.line(290, 340, 382, 300, { color: "hot", w: 2, dash: "5 4" });
          G.text(318, 352, "✕ no open port", { color: "hot", size: 13 });
          G.from(tun, { opacity: 0, duration: 0.8 });
          G.caption("the server never listens on a public address");
        } },

      { rail: "startup", title: "What happens before the first request",
        body: `<p>vLLM does four things before it accepts requests, and its log reports each one. Read them in order, because each step uses what the previous one left over:</p>
<ol><li><b>Load the weights.</b> Llama-3-8B has 8,030,261,248 parameters (P1.1). At 2 bytes each that is 16.06 GB, or <b>14.96 GiB</b> (1 GiB = 2³⁰ bytes, which is how the log counts).</li>
<li><b>Profile.</b> It runs one forward pass with the largest batch it will allow and measures the <b>peak activation memory</b>: the temporary tensors a pass needs while it runs.</li>
<li><b>Size the KV cache.</b> Whatever is left of its memory budget becomes KV-cache blocks. It prints that capacity in tokens and the maximum concurrency at <code>--max-model-len</code>.</li>
<li><b>Capture CUDA graphs</b> for a set of batch sizes (step 10 says why).</li></ol>
<p>The exact wording of these log lines changes between versions. <code>examples/04_d3_vs_log.py</code> searches the log for the KV-capacity line and prints candidate lines if its pattern no longer matches.</p>`,
        scene(G) {
          const st = [["1", "load weights", "14.96 GiB onto the GPU", "w"], ["2", "profile a pass", "measure peak activations", "q"], ["3", "size KV cache", "leftover → blocks → tokens", "k"], ["4", "capture graphs", "one per batch size", "ok"]];
          const boxes = st.map(([n, t, d, col], i) => {
            const y = 40 + i * 92;
            const g = G.group();
            G.circle(48, y + 26, 20, { fill: col, parent: g }); G.text(48, y + 31, n, { anchor: "middle", color: "bg", size: 15, parent: g });
            G.rect(84, y, 300, 52, { stroke: col, rx: 8, parent: g });
            G.text(100, y + 22, t, { size: 15, parent: g }); G.text(100, y + 42, d, { size: 12, color: "muted", parent: g });
            if (i < 3) G.arrow(48, y + 50, 48, y + 86, { color: "muted", parent: g });
            return g;
          });
          G.rect(420, 40, 196, 336, { fill: "line", opacity: 0.35, rx: 8 });
          G.label(432, 64, "vllm.log (paraphrased)", { color: "ink", size: 12 });
          ["model weights … GiB", "profiling run …", "peak activation …", "KV cache size: N tokens", "max concurrency …", "capturing graphs …", "ready on :8000"].forEach((s, i) => G.label(432, 96 + i * 34, s, { size: 11, color: i === 3 ? "k" : "muted" }));
          G.from(boxes, { opacity: 0, x: -20, stagger: 0.3, duration: 0.35 });
          G.caption("each stage spends memory the next stage can no longer use");
        } },

      { rail: "the budget", title: "The memory budget: --gpu-memory-utilization",
        body: `<p>vLLM does not take the whole GPU. <code>--gpu-memory-utilization</code> (default 0.90 in the launch script) is the fraction of GPU memory it allows itself. The rest is headroom for the CUDA context, other processes and fragmentation.</p>
<p>The KV cache gets what remains after weights and activations. This is the same formula as your D3 calculator (P1.5, <code>platform/capacity/core.py</code>):</p>
<div class="eq">KV budget = util × GPU memory − weights − activations

L4: 24 GiB (example value; check nvidia-smi)
Llama-3-8B bf16, util 0.90, activations 1.0 GiB:
  0.90 × 24      = 21.60 GiB
  − 14.96        weights
  − 1.00         activations
  = 5.64 GiB for the KV cache

5.64 GiB ÷ 128 KiB per token ≈ 46,223 tokens</div>
<p>Notice how sensitive this is. Weights are fixed, so every change in the utilization lands entirely on the KV cache: at 0.80 the budget falls to 3.24 GiB (about 26,500 tokens); at 0.95 it rises to 6.84 GiB (about 56,000 tokens).</p>
<details><summary>Go deeper: why the prediction is never exact</summary><p>The activation peak is measured, not assumed; vLLM also reserves memory for CUDA graphs and the non-PyTorch CUDA context, and an L4 reports slightly less than 24 GiB usable. Exercise 1 tunes one assumption at a time until D3 agrees with the log within 5%, then checks the fit still holds at a different utilization.</p></details>`,
        check: { q: "You raise <code>--gpu-memory-utilization</code> from 0.90 to 0.95 on the 24 GiB L4. Roughly how much more KV cache do you get?",
          options: ["About 5% more KV cache", "About 1.2 GiB more, which is about 21% more KV cache", "Nothing: the KV cache is fixed by <code>--max-model-len</code>"], answer: 1,
          why: "0.05 × 24 GiB = 1.2 GiB more budget. Weights and activations don't change, so all of it goes to the KV cache: 5.64 → 6.84 GiB, a 21% increase. Small utilization changes are large KV changes." },
        scene(G) {
          G.text(24, 40, "L4 · 24 GiB · Llama-3-8B bf16 · util 0.90", { size: 14 });
          const parts = [[W8, "w", "weights 14.96"], [1.0, "q", ""], [5.64, "k", "KV 5.64"], [2.4, "muted", ""]];
          const bars = memBar(G, 24, 64, 592, 64, 24, parts);
          G.label(24 + (592 * 14.96) / 24 + 2, 150, "↑ activations 1.0", { color: "q", size: 12 });
          G.line(24 + (592 * 21.6) / 24, 56, 24 + (592 * 21.6) / 24, 136, { color: "ink", dash: "4 3", w: 2 });
          G.label(24 + (592 * 21.6) / 24 - 4, 52, "0.90 × 24 = 21.6", { anchor: "end", color: "ink", size: 12 });
          const rows = [["0.80", 3.24, "26,562"], ["0.90", 5.64, "46,223"], ["0.95", 6.84, "56,053"]];
          G.label(24, 200, "util", { size: 12 }); G.label(90, 200, "KV budget", { size: 12 }); G.label(500, 200, "KV tokens", { size: 12 });
          const kvb = rows.map(([u, gb, tok], i) => {
            const y = 214 + i * 50;
            G.text(24, y + 24, u, { size: 15 });
            const r = G.rect(90, y, (300 * gb) / 7, 34, { fill: "k", rx: 4 });
            G.text(98 + (300 * gb) / 7, y + 23, gb.toFixed(2) + " GiB", { size: 13, color: "k" });
            G.text(500, y + 24, tok, { size: 15, color: "ink" });
            return r;
          });
          G.from(bars, { attr: { width: 0 }, stagger: 0.2, duration: 0.5 });
          G.from(kvb, { attr: { width: 0 }, stagger: 0.15, duration: 0.5, delay: 0.6 });
          G.caption("weights are fixed, so the KV cache absorbs every change");
        } },

      { rail: "blocks", title: "From bytes to blocks to concurrency",
        body: `<p>vLLM does not give each request one long slab of memory. It cuts the KV budget into fixed-size <b>blocks</b> of 16 tokens each (the default block size) and hands blocks to requests as they grow. P2.3 explains why (no wasted reservations); here you only need the count.</p>
<div class="eq">46,223 tokens ÷ 16 per block ≈ 2,888 blocks

max concurrency at --max-model-len 8192:
  46,223 ÷ 8,192 ≈ 5.6 sequences</div>
<p>The log's "maximum concurrency" is a worst case: it assumes every request grows to the full <code>--max-model-len</code>. Real chats are usually much shorter. If requests average 1,000 tokens (prompt plus output), the same 2,888 blocks hold about 46 of them at once.</p>
<p>So the KV budget, measured in <b>tokens</b>, is the real capacity of the server. <code>--max-num-seqs</code> (how many requests may run at once) only matters if it is the smaller limit.</p>`,
        scene(G) {
          const cols = 24, rows = 8, cw = 24, ch = 24, x0 = 24, y0 = 60;
          G.text(24, 40, "KV budget as blocks (each square = 16 tokens; 192 drawn)", { size: 13 });
          const owner = (r, c) => { const i = r * cols + c; return i < 34 ? "k" : i < 52 ? "v" : i < 88 ? "q" : i < 101 ? "pink" : i < 140 ? "blue" : "line"; };
          const cells = G.grid(x0, y0, rows, cols, cw, ch, owner, { gap: 3, rx: 3 });
          [["k", "request A · 540 tokens"], ["v", "request B · 280"], ["q", "request C · 570"], ["pink", "request D · 200"], ["blue", "request E · 620"], ["line", "free blocks"]].forEach(([c, l], i) => {
            const x = 24 + (i % 3) * 200, y = 280 + Math.floor(i / 3) * 30;
            G.rect(x, y, 14, 14, { fill: c, rx: 3 }); G.label(x + 22, y + 12, l, { color: "ink" });
          });
          G.text(24, 370, "2,888 blocks ≈ 46,223 tokens", { size: 14, color: "k" });
          G.label(24, 394, "at 8,192 tokens each: 5.6 requests · at ~1,000 tokens each: ~46", { size: 12, color: "ink" });
          G.from(cells, { opacity: 0, stagger: 0.004, duration: 0.15 });
          G.caption("capacity is counted in tokens, handed out in blocks");
        } },

      { rail: "max-model-len", title: "Why a long --max-model-len can refuse to start",
        body: `<p><code>--max-model-len</code> is the longest sequence (prompt plus output) the server accepts. At start-up vLLM checks that <b>one</b> such sequence fits in the KV budget. If it doesn't, the server exits with an error instead of failing later on a long request.</p>
<p>Take Llama-3.1-8B, which has the same shapes as Llama-3-8B (so the same 128 KiB per token) but a much longer native context:</p>
<div class="eq">32,768 tokens × 128 KiB = 4.0 GiB   fits
65,536 tokens × 128 KiB = 8.0 GiB   &gt; 5.64 GiB
largest that fits ≈ 46,223 tokens (util 0.90)</div>
<p>Three ways to make 65,536 start, each with a cost:</p>
<ul><li><code>--kv-cache-dtype fp8</code>: 1 byte per element, so 64 KiB per token; 65,536 tokens need 4.0 GiB. Quality cost to measure (P2.5).</li>
<li>Smaller weights (<code>--quantization</code>, P2.5): fp8 weights free about 7.5 GiB.</li>
<li>Higher <code>--gpu-memory-utilization</code>: here 0.95 gives only 6.84 GiB, still too little, and less headroom.</li></ul>`,
        check: { q: "vLLM refuses to start with <code>--max-model-len 65536</code> on the L4. Which single change makes it fit in our example?",
          options: ["<code>--max-num-seqs 1</code>", "<code>--kv-cache-dtype fp8</code>", "<code>--gpu-memory-utilization 0.95</code>"], answer: 1,
          why: "The check is about one sequence's KV fitting in the budget. <code>--max-num-seqs</code> doesn't change bytes per token. 0.95 raises the budget to 6.84 GiB, still below 8.0 GiB. fp8 KV halves the bytes per token, so 65,536 tokens need 4.0 GiB, which fits in 5.64 GiB." },
        scene(G) {
          const sc = 400 / 9;
          G.text(24, 40, "one sequence's KV vs the 5.64 GiB budget", { size: 14 });
          const cases = [["32k, bf16 KV", 4.0, "ok"], ["65k, bf16 KV", 8.0, "hot"], ["65k, fp8 KV", 4.0, "ok"]];
          const bs = cases.map(([l, gb, col], i) => {
            const y = 80 + i * 90;
            G.label(24, y - 8, l, { color: "ink", size: 13 });
            const r = G.rect(24, y, gb * sc, 40, { fill: col, rx: 4, opacity: 0.85 });
            G.text(Math.max(32 + gb * sc, 34 + 5.64 * sc), y + 26, gb.toFixed(1) + " GiB" + (col === "hot" ? "  ✕ won't start" : "  ✓ fits"), { size: 13, color: col });
            return r;
          });
          const bx = 24 + 5.64 * sc;
          G.line(bx, 60, bx, 330, { color: "k", w: 2, dash: "6 4" });
          G.label(bx + 6, 346, "KV budget 5.64 GiB", { color: "k", size: 12 });
          G.from(bs, { attr: { width: 0 }, stagger: 0.25, duration: 0.5 });
          G.caption("the start-up check: one full-length sequence must fit");
        } },

      { rail: "the knobs", title: "The flags are the P1 concepts",
        body: `<p>Almost every important vLLM flag sets one quantity you already computed by hand. Learn them as a map, not a list:</p>
<ul><li><code>--gpu-memory-utilization</code> sets the KV budget. Higher costs headroom.</li>
<li><code>--max-model-len</code> sets the worst-case KV per sequence. Higher costs concurrency, and the server may not start.</li>
<li><code>--max-num-seqs</code> caps the batch. Higher raises throughput until ITL passes your target.</li>
<li><code>--max-num-batched-tokens</code> caps the tokens processed per step: the prefill chunk (P2.3). Higher speeds long prompts but makes ITL spikier for everyone else.</li>
<li><code>--kv-cache-dtype fp8</code> halves the bytes per KV number. Costs some quality (P2.5).</li>
<li><code>--quantization</code> shrinks the bytes per weight. Costs some quality (P2.5).</li>
<li><code>--enable-prefix-caching</code> reuses the KV of shared prompt prefixes (P2.3). On by default in recent versions.</li>
<li><code>--enforce-eager</code> turns CUDA graphs off (step 10). Slower decode at small batch, faster start-up.</li>
<li><code>--tensor-parallel-size</code> splits the model across GPUs (P4.2). Costs an all-reduce in every layer.</li></ul>
<p>Defaults move between releases. Always check <code>vllm serve --help</code> for the pinned version before trusting a default.</p>`,
        scene(G) {
          G.text(24, 36, "each flag moves one part of the machine", { size: 14 });
          memBar(G, 24, 60, 592, 50, 24, [[W8, "w", "weights"], [1.0, "q", ""], [5.64, "k", "KV"], [2.4, "muted", ""]]);
          const fl = [["--quantization", 150, "w"], ["--gpu-memory-utilization", 548, "ink"], ["--kv-cache-dtype", 470, "k"]];
          fl.forEach(([t, x, col], i) => { const ly = i === 1 ? 190 : 160; G.arrow(x, ly - 14, x, 116, { color: col }); G.label(i === 0 ? x - 40 : x + 6, ly, t, { anchor: i === 0 ? "start" : "end", color: col, size: 12 }); });
          G.rect(24, 210, 290, 170, { stroke: "line", rx: 10 }); G.label(36, 232, "one engine step (the batch)", { color: "ink" });
          const seqs = G.row(40, 250, ["r1", "r2", "r3", "r4", "r5"], { w: 44, h: 30, gap: 8, fill: "q", size: 11 });
          G.label(40, 306, "--max-num-seqs: how many rows", { color: "q" });
          G.label(40, 330, "--max-num-batched-tokens:", { color: "v" });
          G.label(40, 348, "tokens processed this step", { color: "v" });
          G.rect(332, 210, 284, 170, { stroke: "line", rx: 10 }); G.label(344, 232, "one sequence", { color: "ink" });
          G.rect(344, 256, 256, 26, { fill: "k", opacity: 0.4, rx: 4 }); G.rect(344, 256, 120, 26, { fill: "k", rx: 4 });
          G.label(344, 306, "--max-model-len: the full bar", { color: "k" });
          G.label(344, 330, "--enforce-eager / graphs:", { color: "ok" });
          G.label(344, 348, "how each step is launched", { color: "ok" });
          G.from(seqs, { opacity: 0, stagger: 0.1, duration: 0.25 });
          G.caption("memory flags on top, scheduling flags below");
        } },

      { rail: "the ceiling", title: "How fast can one request decode?",
        body: `<p>From P1.4: a batch-1 decode step must read every weight once, so its time can't be less than <code>bytes ÷ bandwidth</code>.</p>
<div class="eq">L4 bandwidth ≈ 300 GB/s (gpu_specs.yaml, UNVERIFIED)
16.06 GB ÷ 300 GB/s = 53.5 ms per token
                    → 18.7 tokens/s ceiling
D3 with bw_util 0.8 → 15.0 tokens/s</div>
<p>Now add users. At short context each sequence's KV is tiny next to the weights (256 tokens × 128 KiB = 32 MiB), so a batch of 8 reads about 16.3 GB per step instead of 16.06 GB. The step barely slows down, yet 8 tokens come out instead of 1:</p>
<div class="eq">batch 8: 16.33 GB ÷ 300 GB/s = 54.4 ms per step
         8 ÷ 0.0544 s ≈ 147 tokens/s in total
         (8 × 18.7 = 150 if it were perfectly free)</div>
<p><code>bench/decode_ceiling.py</code> measures exactly this at batch 1, 2, 4 and 8 and prints each next to its ceiling. Write your prediction down before you run it.</p>`,
        check: { q: "At short context, why does total throughput rise almost 8× from batch 1 to batch 8?",
          options: ["The GPU runs 8 times more cores", "The weights are read once per step for the whole batch, and the extra KV reads are small", "vLLM skips prefill at batch 8"], answer: 1,
          why: "Decode is memory-bound: step time ≈ bytes read ÷ bandwidth. Weights (16 GB) dominate the bytes and are shared by the batch, so the step only grows by each sequence's small KV read, about 32 MiB at 256 tokens." },
        scene(G) {
          const B = [1, 2, 4, 8], ceil = B.map((b) => { const bytes = 16.06e9 + b * 256 * KV_TOK; return b / (bytes / 300e9); });
          G.text(24, 36, "ceiling: total tokens/s vs batch (L4, 300 GB/s example)", { size: 13 });
          G.axes(60, 60, 520, 300, { ylabel: "tok/s" });
          const bars = G.bars(110, 360, ceil, { w: 70, gap: 50, h: 280, max: 160, fill: "k" });
          ceil.forEach((c, i) => { G.text(145 + i * 120, 352 - (280 * c) / 160, c.toFixed(0), { anchor: "middle", size: 13, color: "ink" }); G.label(145 + i * 120, 380, "batch " + B[i], { anchor: "middle" }); });
          G.path("M 145 " + (360 - (280 * 18.7) / 160) + " L 505 " + (360 - (280 * 149.6) / 160), { color: "muted", dash: "4 4" });
          G.label(76, 90, "dashed: 18.7 × batch (perfect sharing)", { color: "muted", size: 11 });
          G.from(bars, { attr: { height: 0, y: 360 }, stagger: 0.15, duration: 0.45 });
          G.caption("one weight read per step, shared by every sequence in it");
        } },

      { rail: "the gap", title: "Explaining the gap to the ceiling",
        body: `<p>Your measured batch-1 number will be below 18.7 tokens/s. The README expects something like 70–90% of the ceiling, but that is a range to test, not a result. Explain the gap one term at a time instead of shrugging at it:</p>
<ul><li><b>Achievable bandwidth.</b> No GPU reaches its datasheet bandwidth. Your measured copy speed from P1.4 exercise 4 sets a lower, honest ceiling.</li>
<li><b>KV reads.</b> They grow with position. Rerun with a 4,000-token prompt and watch ITL rise.</li>
<li><b>Launch gaps.</b> One decode step runs hundreds of small GPU kernels (several per layer, 32 layers). Launching each one from the CPU costs time during which the GPU may sit idle. A <b>CUDA graph</b> records the whole sequence of launches once and replays it with one call. <code>--enforce-eager</code> turns that off: exercise 3 measures the difference.</li>
<li><b>Sampling, detokenization, HTTP.</b> Compare your client-side ITL with the server's own <code>vllm:inter_token_latency_seconds</code> metric (P1.3).</li></ul>
<p>Always warm up first: the first request pays for graph capture and compilation caches.</p>`,
        check: { q: "Why do CUDA graphs help most at batch 1 and matter less at batch 8?",
          options: ["At batch 1 each kernel is short, so fixed launch overhead is a bigger share of the step", "Graphs are disabled above batch 4", "Batch 8 uses fewer kernels"], answer: 0,
          why: "The number of kernel launches per step is about the same at any batch size, so launch overhead is roughly fixed per step. When each kernel has little work (batch 1), that fixed cost is a large fraction of the step; with more work per kernel it shrinks in proportion." },
        scene(G) {
          G.text(24, 36, "one batch-1 decode step on a timeline", { size: 14 });
          G.label(24, 70, "eager: CPU launches each kernel; the GPU waits in the gaps", { color: "ink" });
          const ek = []; for (let i = 0; i < 12; i++) ek.push(G.rect(24 + i * 50, 84, 34, 30, { fill: "k", rx: 3 }));
          for (let i = 0; i < 11; i++) G.rect(58 + i * 50, 92, 16, 14, { fill: "hot", rx: 2, opacity: 0.7 });
          G.label(24, 170, "CUDA graph: one replay, kernels back to back", { color: "ink" });
          const gk = []; for (let i = 0; i < 12; i++) gk.push(G.rect(24 + i * 36, 184, 34, 30, { fill: "k", rx: 3 }));
          G.line(458, 176, 458, 222, { color: "ok", dash: "3 3" }); G.label(466, 204, "step ends sooner", { color: "ok" });
          G.text(24, 280, "other gaps between ceiling and measurement:", { size: 13 });
          [["achievable < datasheet bandwidth", "w"], ["KV reads grow with position", "k"], ["sampling · detokenize · HTTP", "v"]].forEach(([t, c], i) => { G.rect(24, 298 + i * 30, 14, 14, { fill: c, rx: 3 }); G.label(46, 310 + i * 30, t, { color: "ink" }); });
          G.from(gk, { opacity: 0, stagger: 0.05, duration: 0.15 });
          G.caption("red gaps: GPU idle while the CPU launches the next kernel");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A serving engine keeps one copy of the weights and runs many requests' steps as one batch; vLLM's parts are an HTTP server, a scheduler, a KV-cache manager and a model runner.</li>
<li>Clients use the OpenAI-compatible API; streaming (SSE) makes TTFT and ITL measurable from outside.</li>
<li>Bind to <code>127.0.0.1</code>, require an API key, and reach the server through an SSM port-forward.</li>
<li>KV budget = utilization × memory − weights − activations. For Llama-3-8B on a 24 GiB L4 at 0.90 that is about 5.64 GiB ≈ 46,000 tokens ≈ 2,888 blocks.</li>
<li>One full <code>--max-model-len</code> sequence must fit, or the server won't start.</li>
<li>Batch-1 decode is capped at bytes ÷ bandwidth (≈ 18.7 tok/s here); batching shares the weight read. Explain any gap term by term.</li></ul>`,
    sim: {
      title: "Plan a vLLM launch",
      intro: "Choose a model, GPU and flags. The bar shows how vLLM would split GPU memory at start-up, the panels say whether the server starts and how many sequences fit, and the chart shows the decode ceiling for each batch size. Same arithmetic as platform/capacity (D3).",
      height: 300,
      controls: [
        { id: "m", label: "model", type: "select", value: "llama8", options: [["llama8", "Llama-3-8B"], ["mistral", "Mistral-7B"], ["llama3b", "Llama-3.2-3B"], ["llama1b", "Llama-3.2-1B"]] },
        { id: "g", label: "GPU (memory, bandwidth: UNVERIFIED values from gpu_specs.yaml)", type: "select", value: "L4", options: [["L4", "L4 · 24 GiB · 300 GB/s"], ["A10G", "A10G · 24 GiB · 600 GB/s"], ["T4", "T4 · 16 GiB · 320 GB/s"]] },
        { id: "wd", label: "weights dtype", type: "select", value: 2, options: [[2, "bf16 / fp16 (2 bytes)"], [1, "fp8 (1 byte)"], [0.5, "int4 (0.5 byte)"]] },
        { id: "kd", label: "KV cache dtype (kv-cache-dtype flag)", type: "select", value: 2, options: [[2, "auto (2 bytes)"], [1, "fp8 (1 byte)"]] },
        { id: "u", label: "GPU memory utilization (gpu-memory-utilization flag)", min: 0.5, max: 0.98, step: 0.01, value: 0.9, format: (x) => x.toFixed(2) },
        { id: "act", label: "activation peak, GiB (assumption; vLLM measures it)", min: 0.25, max: 4, step: 0.25, value: 1, format: (x) => x.toFixed(2) },
        { id: "len", label: "max model length (max-model-len flag)", min: 1024, max: 131072, step: 1024, value: 8192 },
        { id: "ctx", label: "average tokens per running sequence", min: 128, max: 32768, step: 128, value: 1024 },
        { id: "bwu", label: "bandwidth utilization (bw_util)", min: 0.5, max: 1, step: 0.05, value: 0.8, format: (x) => x.toFixed(2) },
      ],
      draw(G, v) {
        const m = MODELS[v.m], g = GPUS[v.g];
        const w = (m.P * 1e9 * v.wd) / GiB, tot = g.mem, usable = tot * v.u;
        const kvb = usable - w - v.act, tokB = 2 * m.L * m.kv * m.hd * v.kd;
        const kvTok = kvb > 0 ? Math.floor((kvb * GiB) / tokB) : 0, blocks = Math.floor(kvTok / 16);
        const starts = kvb > 0 && kvTok >= v.len, conc = kvTok / v.len, fitAvg = Math.floor(kvTok / v.ctx);
        // memory bar
        G.label(10, 18, `${m.name} on ${v.g}: how GPU memory is split at start-up (GiB)`, { size: 11 });
        const sc = 600 / tot; let x = 20;
        const seg = (gb, col, lab) => { const pw = Math.max(0, gb) * sc; if (pw <= 0) return; G.rect(x, 28, Math.min(pw, 620 - x), 36, { fill: col, rx: 3, opacity: 0.85 }); if (pw > 70) G.text(x + pw / 2, 51, lab, { anchor: "middle", size: 11, color: "bg" }); x += pw; };
        seg(Math.min(w, tot), "w", "weights " + w.toFixed(1)); seg(Math.min(v.act, Math.max(0, tot - w)), "q", "act"); seg(Math.max(0, kvb), "k", "KV " + Math.max(0, kvb).toFixed(2)); seg(Math.max(0, tot - Math.max(usable, w + v.act)), "muted", "reserved");
        G.rect(20, 28, 600, 36, { stroke: "ink", rx: 3 });
        if (w + v.act > usable) G.text(20, 84, "weights + activations exceed the budget: vLLM cannot start", { color: "hot", size: 12 });
        // decode ceiling chart
        const Bs = [1, 2, 4, 8, 16, 32, 64].filter((b) => b * v.ctx <= Math.max(kvTok, v.ctx));
        const bw = g.bw * 1e9 * v.bwu, wB = m.P * 1e9 * v.wd;
        const agg = Bs.map((b) => b / ((wB + b * v.ctx * tokB) / bw));
        const mx = Math.max(...agg, 1);
        G.label(10, 108, "decode ceiling: total tokens/s at each batch (memory-bound model)", { size: 11 });
        G.axes(40, 118, 580, 150, { xlabel: "batch size →" });
        agg.forEach((a, i) => { const h = (140 * a) / mx; G.rect(60 + i * 80, 268 - h, 50, h, { fill: "k", rx: 2 }); G.text(85 + i * 80, 262 - h, F.num(a, 0), { anchor: "middle", size: 11 }); G.text(85 + i * 80, 284, String(Bs[i]), { anchor: "middle", size: 11, color: "muted" }); });
        if (!Bs.length || kvTok <= 0) G.text(60, 200, "no batch fits", { color: "hot" });
        const warn = [];
        if (v.wd === 1 && !g.fp8) warn.push(`${v.g} has no FP8 support`);
        if (v.kd === 1 && !g.fp8) warn.push(`fp8 KV needs FP8 hardware support, which ${v.g} lacks in the course's specs`);
        if (v.wd === 2 && !g.bf16) warn.push("T4 has no bf16: use fp16 (same 2 bytes)");
        return [
          { title: "Start-up check", gauge: [[Math.min(w, tot) / tot, "w"], [Math.min(v.act, Math.max(0, tot - w)) / tot, "q"], [Math.max(0, kvb) / tot, "k"]],
            gaugeText: `budget ${usable.toFixed(2)} of ${tot} GiB`, chip: [starts, starts ? "server starts" : kvb <= 0 ? "no room for KV" : "--max-model-len too long"],
            rows: [["weights", w.toFixed(2) + " GiB"], ["KV budget", Math.max(0, kvb).toFixed(2) + " GiB"], ["one full-length sequence needs", F.bytes(v.len * tokB)]] },
          { title: "KV capacity", rows: [["per token", F.bytes(tokB)], ["KV tokens", F.num(kvTok)], ["blocks of 16", F.num(blocks)], ["max concurrency at max-model-len", conc.toFixed(1)], ["sequences at the average length", F.num(fitAvg)]] },
          { title: "Decode ceiling", rows: [["batch 1, tokens/s", F.num(bw / (wB + v.ctx * tokB), 1)], ["batch 1, ITL lower bound", F.ms((wB + v.ctx * tokB) / bw)], ["largest batch drawn", Bs.length ? String(Bs[Bs.length - 1]) : "–"]],
            html: `<p class="note">Ceiling = bw_util × bandwidth ÷ (weights + batch × KV per sequence). Real engines land below it; explain the gap term by term.${warn.length ? "<br><b>" + warn.join(" · ") + "</b>" : ""}</p>` },
        ];
      },
      note: "Model shapes come from the public config.json files and GPU numbers from the course's gpu_specs.yaml, which marks them UNVERIFIED. vLLM measures activations and reserves memory for CUDA graphs, so its log will differ by a few percent: calibrating that is exercise 1.",
    },
    practice: {
      intro: "The T0 parts run on a laptop against platform/mockllm. The rest need one GPU (aws.md: g6.xlarge, bound to 127.0.0.1, reached through make forward, torn down with make down).",
      items: [
        { title: "Make D3 match the startup log", tier: "T2 · medium", goal: "Tune one D3 assumption at a time until its KV-token prediction is within 5% of vLLM's log, then check it still holds at utilization 0.80.",
          cmd: "uv run python course/P2-serving-engines/P2.1-first-serve-with-vllm/exercises/check_results.py results/p21.json" },
        { title: "Find the max-model-len that fails", tier: "T2 · easy", goal: "Raise MAX_MODEL_LEN until vLLM refuses to start, explain the error with D3, then make it start by changing one other flag.",
          cmd: "MAX_MODEL_LEN=65536 bash course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/01_serve.sh /opt/models/<model>" },
        { title: "Eager vs CUDA graphs at batch 1", tier: "T2 · medium", goal: "Run the decode-ceiling bench with and without <code>--enforce-eager</code> and explain why the gap shrinks at batch 8.",
          cmd: "python course/P2-serving-engines/P2.1-first-serve-with-vllm/bench/decode_ceiling.py --url http://127.0.0.1:8000 --gpu L4 --model-config /opt/models/<model>/config.json" },
        { title: "Reproduce vllm bench latency", tier: "T0 test + T2 · hard", goal: "Write measure_e2e() in my_latency.py, pass the test against the mock server, then match vllm bench latency within 10%.",
          cmd: "uv run pytest course/P2-serving-engines/P2.1-first-serve-with-vllm/exercises" },
      ],
      labs: [
        { label: "Launch script (localhost + API key, logs to vllm.log)", path: "course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/01_serve.sh" },
        { label: "Client: streaming, TTFT and ITL", path: "course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/02_client.py" },
        { label: "Offline batch API (LLM.generate)", path: "course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/03_offline_llm.py" },
        { label: "D3 prediction vs the startup log", path: "course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/04_d3_vs_log.py" },
        { label: "Bench: decode tok/s vs ceiling at batch 1–8", path: "course/P2-serving-engines/P2.1-first-serve-with-vllm/bench/decode_ceiling.py" },
        { label: "AWS guide: cost, auto-stop, teardown", path: "course/P2-serving-engines/P2.1-first-serve-with-vllm/aws.md" },
        { label: "The capacity model (D3)", path: "platform/capacity/core.py" },
        { label: "Mock server for T0 runs", path: "platform/mockllm/server.py" },
      ],
    },
  });
})();
