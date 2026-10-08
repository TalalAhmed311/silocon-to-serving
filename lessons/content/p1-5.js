/* P1.5 — Project: a capacity calculator. The simulator re-implements platform/capacity/core.py:plan() in JS. */
(function () {
  const F = S2S.fmt;
  const GiB = 2 ** 30;
  // platform/capacity/presets/*.json (UNVERIFIED author recollection of the HF config.json files).
  const MODELS = {
    "llama3-8b": { name: "Llama-3-8B", d: 4096, ff: 14336, L: 32, H: 32, kv: 8, V: 128256, tie: false, E: 0, k: 0 },
    "llama3-70b": { name: "Llama-3-70B", d: 8192, ff: 28672, L: 80, H: 64, kv: 8, V: 128256, tie: false, E: 0, k: 0 },
    "mixtral-8x7b": { name: "Mixtral-8x7B", d: 4096, ff: 14336, L: 32, H: 32, kv: 8, V: 32000, tie: false, E: 8, k: 2 },
  };
  // P1.4's gpu_specs.yaml (every entry UNVERIFIED). mem in GB as written there; core.py multiplies it by 2^30.
  const GPUS = {
    T4: { mem: 16, bw: 320, fp16: 65, fp8: null }, L4: { mem: 24, bw: 300, fp16: 121, fp8: 242 },
    L40S: { mem: 48, bw: 864, fp16: 362, fp8: 733 }, "A100-80GB-SXM": { mem: 80, bw: 2039, fp16: 312, fp8: null },
    "H100-SXM": { mem: 80, bw: 3350, fp16: 989, fp8: 1979 },
  };
  const DT = { fp32: 4, fp16: 2, bf16: 2, fp8: 1, int8: 1, int4: 0.5 };

  // ---- the same formulas as platform/capacity/core.py ----
  const hd = (m) => m.d / m.H;
  const attnPL = (m) => m.d * m.H * hd(m) * 2 + m.d * m.kv * hd(m) * 2;
  const mlpPE = (m) => 3 * m.d * m.ff;
  function totalParams(m) {
    const router = m.E ? m.d * m.E : 0;
    const per = attnPL(m) + Math.max(1, m.E) * mlpPE(m) + router + 2 * m.d;
    return m.L * per + m.V * m.d + (m.tie ? 0 : m.V * m.d) + m.d;
  }
  function activeParams(m) {
    const router = m.E ? m.d * m.E : 0;
    const per = attnPL(m) + (m.E ? m.k : 1) * mlpPE(m) + router + 2 * m.d;
    return m.L * per + m.V * m.d + m.d;   // LM head read every token; embedding lookup excluded
  }
  const kvTok = (m, b) => 2 * m.L * m.kv * hd(m) * b;
  function plan(m, gName, wd, kd, tp, ctx, memUtil, act, bwUtil, mfu) {
    const g = GPUS[gName], wb = DT[wd], kb = DT[kd];
    const weights = (totalParams(m) * wb) / tp;
    const usable = g.mem * GiB * memUtil;
    const budget = usable - weights - act * GiB;
    const kt = kvTok(m, kb) / tp;
    const maxTok = budget > 0 ? Math.floor(budget / kt) : 0;
    const p = { g, weights, usable, budget, kvPerTok: kvTok(m, kb), maxTok, maxSeqs: Math.floor(maxTok / ctx), fits: budget > 0, decode: {}, prefill: {} };
    const bw = g.bw * 1e9 * bwUtil * tp, readW = activeParams(m) * wb;
    p.decodeAt = (B) => { const s = (readW + B * (ctx / 2) * kvTok(m, kb)) / bw; return [1 / s, B / s, s]; };
    const peak = wd === "fp8" ? g.fp8 : g.fp16;
    p.peak = peak;
    p.prefillAt = (T) => (peak ? (2 * activeParams(m) * T + 2 * m.L * m.H * hd(m) * T * T) / (peak * 1e12 * mfu * tp) : null);
    return p;
  }
  const gib = (b) => (b / GiB).toFixed(1) + " GiB";

  S2S.lesson({
    id: "p1-5", n: "P1.5", title: "Project: a capacity calculator",
    subtitle: "Inference fundamentals · project · T0 (one exercise compares with vLLM on a GPU)",
    kicker: "Project lesson · ≈ 30 min + build",
    headline: "Will it fit, how many users, how fast?",
    intro: `<p>Renting a GPU to find out whether a model fits is slow and costs money. Everything needed to answer in advance is already in the last four lessons: parameter counts (P1.1), the KV cache (P1.2), what "fast" means (P1.3) and rooflines (P1.4). This project wires them into one tool, <code>platform/capacity</code>, that every later phase uses to size deployments. The lesson walks through each formula with Llama-3-8B on an L4, then the simulator lets you run the whole calculator on any combination.</p>
<p>Model configs (<code>platform/capacity/presets</code>) and GPU specs (<code>gpu_specs.yaml</code>) are both <b>UNVERIFIED</b>, so every number below is a prediction from those inputs, not a measurement.</p>`,
    facts: ["10 steps", "4 checkpoints", "1 simulator", "5 exercises"],
    legend: [["w", "weights"], ["k", "KV cache"], ["muted", "reserved"], ["q", "compute"], ["hot", "doesn't fit"], ["ok", "result"]],
    prev: "p1-4", next: "p2-1",
    steps: [
      { rail: "the question", title: "Three questions before renting a GPU",
        body: `<p>Someone asks: "Can we serve Llama-3-8B on an L4, for how many users, and how fast?" You could rent one and try. Or you could compute it, because every part of the answer is arithmetic you have already done:</p>
<ol><li><b>Does it fit?</b> Weight bytes against GPU memory (P1.1).</li>
<li><b>How many users?</b> The memory left over, divided by each conversation's KV cache (P1.2).</li>
<li><b>How fast?</b> Bytes per decode step against bandwidth, FLOPs per prefill against compute (P1.4), reported as tokens/s and TTFT (P1.3).</li></ol>
<p>The calculator is one function, <code>plan()</code> in <code>platform/capacity/core.py</code>. Its inputs are a model's <code>config.json</code>, a GPU from <code>gpu_specs.yaml</code>, data types, the number of GPUs (TP) and a context length, plus <b>three named assumptions</b> it prints with every answer. You build your own version in <code>exercises/calculator.py</code>, and the tests are the specification.</p>
<div class="eq">PYTHONPATH=platform uv run python \\
  -m capacity.cli --model llama3-8b --gpu L4</div>`,
        scene(G) {
          const ins = [["config.json", "ink"], ["gpu_specs.yaml", "v"], ["dtypes · TP · context", "blue"], ["mem_util · bw_util · mfu", "muted"]];
          const a = ins.map(([t, c], i) => G.box(24, 60 + i * 70, 190, 46, t, { stroke: c, color: c, size: 12 }));
          ins.forEach((_, i) => G.arrow(218, 83 + i * 70, 258, 200, { color: "muted" }));
          const core = G.box(262, 160, 120, 80, "plan()", { fill: "q", size: 15 });
          G.label(264, 258, "platform/capacity/core.py", { size: 10 });
          const outs = [["weights per GPU", "P1.1"], ["KV budget, max users", "P1.2"], ["decode tokens/s", "P1.4"], ["prefill time (TTFT)", "P1.3"]];
          const b = outs.map(([t, l], i) => { G.arrow(386, 200, 418, 83 + i * 70, { color: "ok" }); G.label(616, 120 + i * 70, l, { size: 10, anchor: "end" }); return G.box(422, 60 + i * 70, 194, 46, t, { fill: "ok", size: 12 }); });
          G.label(24, 380, "inputs on the left are UNVERIFIED or assumptions: the tool prints them", { size: 12 });
          G.from(a, { opacity: 0, x: -20, stagger: 0.1, duration: 0.3 });
          G.from(core, { opacity: 0, scale: 0.6, transformOrigin: "center", duration: 0.4, delay: 0.4 });
          G.from(b, { opacity: 0, x: 20, stagger: 0.1, duration: 0.3, delay: 0.7 });
          G.caption("four lessons of formulas in one function");
        } },

      { rail: "weights", title: "First: the weights must fit",
        body: `<p>Weight bytes are parameters times bytes per parameter, from the formula in P1.1:</p>
<div class="eq">weights = total_params × bytes(dtype) ÷ TP

Llama-3-8B, bf16:
  8,030,261,248 × 2 = 16.06e9 bytes
                    = 14.96 GiB</div>
<p>Note the two units. <b>GB</b> is 10⁹ bytes; <b>GiB</b> is 2³⁰ = 1,073,741,824 bytes, 7.4% more. The calculator reports GiB, as vLLM does. Mixing them silently loses a gigabyte or two, which on a 24 GB card is several users.</p>
<p>Exercise 1 makes the formula exact: <code>weights_bytes(cfg, dtype)</code> must equal, to the byte, the summed tensor sizes of a tiny checkpoint written by <code>platform/engine/v0/tools/make_tiny_llama.py</code>, with and without tied embeddings (a tied model stores its LM head and embedding once). On real models, the <code>*.safetensors</code> files are within about 2% (headers and optional tensors).</p>
<p>Quantizing changes only <code>bytes(dtype)</code>: fp8 halves the weights to 7.48 GiB, int4 quarters them.</p>`,
        scene(G) {
          const sc = 560 / 24;
          G.text(24, 40, "L4 · 24 GiB as the calculator reads it", { size: 13 });
          G.rect(24, 60, 560, 56, { stroke: "ink", rx: 8 });
          const segs = [["bf16", 14.96, "w"], ["fp8", 7.48, "w"], ["int4", 3.74, "w"]];
          const bars = segs.map(([n, v, c], i) => {
            const y = 160 + i * 70;
            G.label(24, y - 8, `${n}: 8.03 B × ${DT[n]} B`, { size: 12, color: "ink" });
            G.rect(24, y, 560, 36, { stroke: "line", rx: 6 });
            const r = G.rect(24, y, v * sc, 36, { fill: c, rx: 6 });
            G.text(34 + v * sc, y + 24, `${v.toFixed(2)} GiB`, { size: 13 });
            return r;
          });
          G.rect(24, 60, 14.96 * sc, 56, { fill: "w", rx: 8, opacity: 0.8 });
          G.text(36, 94, "bf16 weights 14.96 GiB", { size: 13, color: "ink" });
          G.label(24, 390, "16.06e9 B = 16.06 GB = 14.96 GiB: same bytes, different units", { size: 12 });
          G.from(bars, { attr: { width: 0 }, stagger: 0.2, duration: 0.5 });
          G.caption("weights = params × bytes per param ÷ TP");
        } },

      { rail: "the budget", title: "What is left for the KV cache",
        body: `<p>Not all GPU memory is available for the cache. The calculator copies how vLLM sizes it:</p>
<div class="eq">usable    = memory × mem_util
KV budget = usable − weights − activation reserve

L4, Llama-3-8B bf16:
  24 GiB × 0.90       = 21.60 GiB
  − weights           = 14.96 GiB
  − activations       =  1.00 GiB
  = KV budget         ≈  5.64 GiB</div>
<p><code>mem_util = 0.90</code> is vLLM's default <code>--gpu-memory-utilization</code>: the share of the card the engine may claim, leaving room for the CUDA context and other processes. The <b>activation reserve</b> covers the temporary tensors of a forward pass and the buffers CUDA graphs need; vLLM measures its own at startup, and the calculator assumes 1 GiB.</p>
<p>The result is sobering. Of a 24 GB card, the conversations get less than a quarter.</p>
<p>A detail you can check in the code: <code>core.py</code> multiplies <code>memory_gb</code> from the YAML by 2³⁰, so it treats the datasheet's "24 GB" as 24 GiB. Whether that matches what <code>nvidia-smi</code> reports is something to verify when you first get a GPU (P2.1).</p>`,
        scene(G) {
          const sc = 560 / 24, y = 90;
          G.text(24, 50, "L4 memory, Llama-3-8B bf16 (GiB)", { size: 13 });
          const parts = [[14.96, "w", "weights 14.96"], [1, "q", "act 1.0"], [5.64, "k", "KV 5.64"], [2.4, "muted", "10% kept free 2.4"]];
          let x = 24; const segs = [];
          parts.forEach(([v, c, t], i) => {
            segs.push(G.rect(x, y, v * sc - 2, 64, { fill: c, rx: 4, opacity: c === "muted" ? 0.4 : 0.9 }));
            const lx = x + (v * sc) / 2, ly = i % 2 ? 196 : 182;
            G.line(lx, y + 66, lx, ly - 14, { color: "line" });
            G.text(lx, ly, t, { anchor: "middle", size: 12, color: c === "muted" || c === "w" ? "ink" : c });
            x += v * sc;
          });
          G.line(24 + 21.6 * sc, 70, 24 + 21.6 * sc, 166, { color: "ink", dash: "4 3" });
          G.label(24 + 21.6 * sc - 4, 66, "mem_util 0.90 → 21.6", { anchor: "end", size: 11, color: "ink" });
          G.text(24, 260, "21.60 − 14.96 − 1.00 = 5.64 GiB for every conversation's cache", { size: 14, color: "ok" });
          G.label(24, 290, "weights are fixed; only the KV slice grows or shrinks with your choices", { size: 12 });
          G.from(segs, { scaleX: 0, transformOrigin: "0% 50%", stagger: 0.2, duration: 0.4 });
          G.caption("the KV budget is what remains after weights and reserves");
        } },

      { rail: "max users", title: "Divide the budget into conversations",
        body: `<p>Each token of context costs the KV bytes from P1.2, so the budget converts to a token count and then to whole conversations:</p>
<div class="eq">KV per token = 2 × L × H_kv × h × bytes
             = 2 × 32 × 8 × 128 × 2 = 128 KiB
max tokens   = 5.64 GiB ÷ 128 KiB  = 46,223
at 4,096 ctx: 46,223 ÷ 4,096       = 11 sequences</div>
<p>Eleven users with full 4k contexts. Now switch both weights and KV cache to fp8:</p>
<div class="eq">weights   7.48 GiB   → budget 13.12 GiB
KV/token  64 KiB     → 214,978 tokens
at 4,096 ctx         → 52 sequences</div>
<p>Halving every byte gave <b>4.7×</b> the users, not 2×, because the smaller weights also free memory for the cache. Context is the other big lever: the same budget holds 22 conversations at 2k tokens but only 1 at 32k. That is why serving engines cap <code>--max-model-len</code>.</p>`,
        check: { q: "Switching Llama-3-8B on an L4 from bf16 to fp8 (weights and KV) takes max sequences from 11 to 52. Why more than 2×?",
          options: ["fp8 arithmetic is faster", "Each sequence's cache halves AND the halved weights leave a much bigger budget", "fp8 removes the activation reserve"], answer: 1,
          why: "The budget is what's left after weights. Weights drop from 14.96 to 7.48 GiB, so the budget grows from 5.64 to 13.12 GiB (2.3×), and each token's cache halves (2×). Together ≈ 4.7×." },
        scene(G) {
          const row = (y, title, n, cells, col) => {
            G.text(24, y - 10, title, { size: 13 });
            const out = [];
            for (let i = 0; i < cells; i++) out.push(G.rect(24 + (i % 26) * 22.5, y + Math.floor(i / 26) * 26, 19, 21, { fill: col, rx: 3, opacity: i < n ? 1 : 0.15 }));
            return out;
          };
          const a = row(60, "bf16: 5.64 GiB ÷ (4,096 × 128 KiB) = 11 sequences", 11, 11, "k");
          const b = row(140, "fp8: 13.12 GiB ÷ (4,096 × 64 KiB) = 52 sequences", 52, 52, "ok");
          G.label(24, 236, "one square = one 4k-token conversation", { size: 11 });
          G.text(24, 286, "context sets the count too (bf16, 46,223 tokens):", { size: 13 });
          const ctxs = [[2048, 22], [4096, 11], [8192, 5], [32768, 1]];
          const bars = G.bars(60, 400, ctxs.map((c) => c[1]), { w: 70, gap: 60, h: 80, max: 22, fill: "k" });
          ctxs.forEach(([c, n], i) => { G.label(95 + i * 130, 416, `${c / 1024}k ctx`, { anchor: "middle", size: 11 }); G.text(95 + i * 130, 400 - (80 * n) / 22 - 6, String(n), { anchor: "middle", size: 12 }); });
          G.from(a.concat(b), { opacity: 0, stagger: 0.012, duration: 0.15 });
          G.from(bars, { attr: { height: 0, y: 400 }, stagger: 0.1, duration: 0.3, delay: 0.6 });
          G.caption("max sequences = ⌊KV budget ÷ (context × KV per token)⌋");
        } },

      { rail: "decode ceiling", title: "How fast can it decode?",
        body: `<p>P1.4 said decode is memory-bound: each step reads the weights once plus every running sequence's KV cache. The calculator adds one refinement each:</p>
<div class="eq">step bytes = active weight bytes
           + B × (context ÷ 2) × KV per token
step time  = step bytes ÷ (bw_util × bandwidth × TP)</div>
<p><code>context ÷ 2</code> because, in a running batch, sequences are on average half way through. <code>bw_util = 0.80</code> because no real kernel reaches the datasheet bandwidth (P1.4 exercise 4 measures yours). On the L4 at 4k context:</p>
<div class="eq">B = 1:  15.01e9 + 1 × 2048 × 131,072 = 15.28e9 B
        ÷ 240 GB/s = 63.7 ms → 15.7 tok/s
B = 8:  15.01e9 + 8 × 268e6        = 17.16e9 B
        ÷ 240 GB/s = 71.5 ms → 14.0 tok/s each,
                                  112 tok/s total</div>
<p>The weights dominate: eight users cost only 12% more time per step than one, so aggregate throughput rises about 7×. As the batch and context grow, the KV term takes over and per-user speed falls. These are <b>ceilings</b>; P2.1 measures how close vLLM gets.</p>`,
        scene(G) {
          const Bs = [1, 2, 4, 8, 11], kvB = 2048 * 131072, w = 15.01e9, bw = 240e9;
          G.text(24, 40, "Llama-3-8B bf16 · L4 · 4k context · bytes read per step", { size: 13 });
          const sc = 330 / 18.0e9, bars = [];
          Bs.forEach((B, i) => {
            const y = 70 + i * 50, kv = B * kvB, s = (w + kv) / bw;
            G.label(24, y + 22, `B = ${B}`, { size: 12, color: "ink" });
            bars.push(G.rect(84, y, w * sc, 30, { fill: "w", rx: 4 }));
            bars.push(G.rect(84 + w * sc, y, kv * sc, 30, { fill: "k", rx: 2 }));
            G.label(84 + (w + kv) * sc + 8, y + 13, `${(s * 1e3).toFixed(1)} ms`, { size: 11, color: "ink" });
            G.label(84 + (w + kv) * sc + 8, y + 28, `${(1 / s).toFixed(1)} each · ${(B / s).toFixed(0)} total`, { size: 11, color: "ok" });
          });
          G.rect(84, 334, 12, 12, { fill: "w", rx: 2 }); G.label(102, 344, "active weights 15.01 GB (same for any batch)", { size: 11 });
          G.rect(84, 356, 12, 12, { fill: "k", rx: 2 }); G.label(102, 366, "KV read: B × 2,048 × 128 KiB", { size: 11 });
          G.label(24, 396, "B = 11 is the most that fits at 4k (step 4)", { size: 11 });
          G.from(bars, { attr: { width: 0 }, stagger: 0.06, duration: 0.4 });
          G.caption("weights are shared by the batch; each user adds their own KV");
        } },

      { rail: "active vs total", title: "Stored is not the same as read",
        body: `<p>The decode formula used <b>active</b> parameters (7.50 B), not the total (8.03 B). Two rules decide what a decode step reads:</p>
<ul><li>The <b>embedding</b> is a table lookup: one row of 4,096 numbers per token, so its 525 M weights are stored but barely read. Excluded.</li>
<li>The <b>LM head</b> is a full 128,256 × 4,096 matrix times a vector, every token. Included, even when it is tied to the embedding.</li></ul>
<p>The difference becomes huge for a <b>mixture-of-experts (MoE)</b> model. Mixtral-8x7B has 8 expert MLPs per layer, and a small <b>router</b> sends each token to just 2 of them. All 8 must be stored; only 2 are read:</p>
<div class="eq">per layer: attention 41.9 M + router 32,768
  + experts: 8 × 176.2 M stored, 2 × 176.2 M read
total  = 46.70 B   → memory (all experts)
active = 12.75 B   → decode bytes (2 experts)</div>
<p>Using the total for decode bandwidth overestimates bytes read by 3.7×. Using the active count for memory says it fits when it does not. Exercise 5 implements both counts.</p>`,
        check: { q: "For Mixtral-8x7B, which parameter count belongs in the decode-bandwidth formula?",
          options: ["Total (46.7 B): every expert must be loaded", "Active (12.75 B): each token reads only its 2 routed experts, plus attention and the LM head", "Total ÷ 8, one expert's worth"], answer: 1,
          why: "A decode step reads only the weights it multiplies: attention, the 2 chosen experts' MLPs, the router and the LM head. All 8 experts still occupy memory, so VRAM uses the total. (At large batch, different tokens pick different experts and more of them get read.)" },
        scene(G) {
          G.text(24, 40, "one Mixtral layer, one token", { size: 13 });
          G.box(24, 64, 120, 46, "attention", { fill: "q", size: 12 });
          G.box(24, 130, 120, 36, "router", { stroke: "ok", color: "ok", size: 12 });
          const ex = [];
          for (let e = 0; e < 8; e++) {
            const on = e === 0 || e === 4, x = 180 + (e % 4) * 110, y = 64 + Math.floor(e / 4) * 70;
            ex.push(G.box(x, y, 96, 52, `expert ${e + 1}`, { fill: on ? "v" : undefined, stroke: on ? undefined : "line", color: on ? undefined : "muted", size: 12 }));
            if (on) G.arrow(146, 148, x - 2, y + 26, { color: "ok", w: 2 });
          }
          G.label(180, 214, "stored: 8 experts · read this token: 2", { size: 12, color: "ink" });
          const tot = 46.70, act = 12.75, sc = 440 / 46.70;
          G.label(24, 266, "total (VRAM)", { size: 12, color: "ink" });
          const b1 = G.rect(130, 252, tot * sc, 24, { fill: "w", rx: 4 });
          G.text(140 + tot * sc - 70, 269, "46.70 B", { size: 12, color: "bg" });
          G.label(24, 312, "active (decode)", { size: 12, color: "ink" });
          const b2 = G.rect(130, 298, act * sc, 24, { fill: "v", rx: 4 });
          G.text(140 + act * sc, 315, "12.75 B", { size: 12, color: "v" });
          G.label(24, 360, "Llama-3-8B: total 8.03 B, active 7.50 B (no embedding lookup)", { size: 12 });
          G.from(ex, { opacity: 0, stagger: 0.06, duration: 0.25 });
          G.from([b1, b2], { attr: { width: 0 }, stagger: 0.3, duration: 0.5, delay: 0.4 });
          G.caption("memory pays for all experts; bandwidth only for the ones used");
        } },

      { rail: "prefill", title: "Prefill time: the TTFT floor",
        body: `<p>Prefill is compute-bound (P1.4), so its time is FLOPs over achievable FLOP/s. The FLOPs have two parts: every prompt token through every weight matrix (2 × active params per token, from P1.1), plus attention, where token <i>t</i> scores against <i>t</i> earlier tokens. Summed over a causal prompt that is about <code>2·L·H·h·T²</code>:</p>
<div class="eq">FLOPs = 2·active·T + 2·L·H·h·T²
time  = FLOPs ÷ (mfu × peak × TP)

L4, mfu 0.5 → 60.5 TFLOP/s achievable
T = 2,048:  3.07e13 + 0.11e13 = 3.18e13 → 526 ms
T = 8,192:  1.23e14 + 0.18e14 = 1.41e14 → 2.32 s</div>
<p>Four times the prompt costs 4.4 times the time: the T² term is 3% of the FLOPs at 2k and 13% at 8k. <b>mfu</b> (model-FLOPs utilization) is the fraction of datasheet peak a real prefill reaches; 0.5 is the calculator's default guess. This number is a floor for TTFT: queueing (P1.3) comes on top.</p>`,
        scene(G) {
          const Ts = [512, 2048, 4096, 8192], A = 7.5049e9, att = 262144;
          const W = (T) => 2 * A * T, At = (T) => att * T * T, tt = (T) => (W(T) + At(T)) / 60.5e12;
          G.text(24, 40, "prefill time on an L4 (mfu 0.5), Llama-3-8B bf16", { size: 13 });
          const sc = 470 / 2.4, bars = [];
          Ts.forEach((T, i) => {
            const y = 74 + i * 64, w = W(T) / 60.5e12, a = At(T) / 60.5e12;
            G.label(24, y + 22, `T = ${F.num(T)}`, { size: 12, color: "ink" });
            bars.push(G.rect(110, y, w * sc, 32, { fill: "q", rx: 3 }));
            bars.push(G.rect(110 + w * sc, y, Math.max(2, a * sc), 32, { fill: "hot", rx: 2 }));
            G.label(116 + (w + a) * sc, y + 21, tt(T) >= 1 ? tt(T).toFixed(2) + " s" : Math.round(tt(T) * 1e3) + " ms", { size: 12, color: "ink" });
          });
          G.rect(110, 342, 12, 12, { fill: "q", rx: 2 }); G.label(128, 352, "2 · active · T (weights)", { size: 11 });
          G.rect(330, 342, 12, 12, { fill: "hot", rx: 2 }); G.label(348, 352, "2 · L · H · h · T² (attention)", { size: 11 });
          G.label(24, 388, "the red slice grows with T²: 4× the prompt → 4.4× the time at 8k", { size: 12 });
          G.from(bars, { attr: { width: 0 }, stagger: 0.08, duration: 0.4 });
          G.caption("TTFT floor = prefill FLOPs ÷ (mfu × peak)");
        } },

      { rail: "tensor parallel", title: "When one GPU is not enough",
        body: `<p>Llama-3-70B in bf16 needs 70.55 B × 2 = 131.4 GiB of weights. One 80 GB H100 offers 72 GiB usable. It cannot fit.</p>
<p><b>Tensor parallelism (TP)</b> splits every weight matrix across <i>N</i> GPUs, each holding <code>1/N</code> of the columns or rows, and also splits the KV heads. All N GPUs read their shards at the same time, so bandwidth adds up too. The calculator divides weights and KV per token by TP and multiplies bandwidth and peak by TP:</p>
<div class="eq">Llama-3-70B bf16 on 4× H100 (calculator output):
  weights per GPU    32.9 GiB
  KV budget per GPU  38.1 GiB
  KV per token       320 KiB ÷ 4 = 80 KiB per GPU
  max sequences      122 at 4k context
  decode B = 1       76.7 tok/s (ceiling)</div>
<p>It ignores the cost of the GPUs talking to each other: two all-reduces per layer per step. P4.1 measures those links and P4.2 adds them back (<code>link_gbs</code> in <code>plan()</code>).</p>
<p>Try the README's third walkthrough line, Mixtral on 2× L40S in bf16: the calculator answers <b>DOES NOT FIT</b>, because 43.5 GiB of weights per GPU exceeds the 43.2 GiB usable. Its options: TP 4, fp8 weights, or a bigger card.</p>`,
        check: { q: "Llama-3-70B bf16 (131 GiB of weights) moves from 1 to 4 H100s with TP. What happens to the decode ceiling at batch 1, in the calculator?",
          options: ["It stays the same: the model is the same size", "It rises about 4×: each GPU reads a quarter of the weights in parallel", "It falls: GPUs must communicate"], answer: 1,
          why: "Each GPU reads only its 1/4 shard, all at once, so the step's bytes are spread over 4× the bandwidth. The calculator ignores communication; in reality the all-reduces take some of that gain back (P4.2)." },
        scene(G) {
          const sc = 300 / 140;
          G.text(24, 40, "Llama-3-70B bf16 · H100 80 GB (72 GiB usable)", { size: 13 });
          G.label(24, 72, "TP = 1", { size: 12, color: "ink" });
          G.rect(84, 56, 72 * sc, 28, { stroke: "ink", rx: 4 });
          const over = G.rect(84, 56, 131.4 * sc, 28, { fill: "hot", rx: 4, opacity: 0.85 });
          G.text(92, 75, "weights 131.4 GiB: does not fit", { size: 12, color: "bg" });
          G.line(84 + 72 * sc, 48, 84 + 72 * sc, 92, { color: "ink", w: 2 });
          G.label(84 + 72 * sc + 4, 104, "72 GiB", { size: 11, color: "ink" });
          G.label(24, 150, "TP = 4: each GPU holds a quarter", { size: 12, color: "ink" });
          const gp = [];
          for (let i = 0; i < 4; i++) {
            const x = 24 + i * 150, y = 166;
            G.rect(x, y, 136, 180, { stroke: "line", rx: 8 });
            G.label(x + 8, y + 18, `H100 #${i + 1}`, { size: 11 });
            const hW = (32.9 / 72) * 140, hK = (38.1 / 72) * 140;
            gp.push(G.rect(x + 10, y + 168 - hW, 116, hW, { fill: "w", rx: 4 }));
            gp.push(G.rect(x + 10, y + 168 - hW - hK + 2, 116, hK - 4, { fill: "k", rx: 4, opacity: 0.85 }));
            G.text(x + 68, y + 168 - hW / 2 + 4, "32.9 w", { anchor: "middle", size: 11, color: "bg" });
            G.text(x + 68, y + 168 - hW - hK / 2 + 4, "38.1 KV", { anchor: "middle", size: 11, color: "bg" });
          }
          G.text(24, 380, "4 × 3,350 GB/s read in parallel → 76.7 tok/s at B = 1 (ceiling)", { size: 13, color: "ok" });
          G.label(24, 404, "communication between GPUs not included (P4.2)", { size: 11 });
          G.from(over, { attr: { width: 0 }, duration: 0.7 });
          G.from(gp, { opacity: 0, y: 20, stagger: 0.08, duration: 0.3, delay: 0.5 });
          G.caption("TP divides weights and KV per GPU and adds bandwidth");
        } },

      { rail: "assumptions", title: "Physics and guesses, kept apart",
        body: `<p>Most of the calculator is physics: parameter counts, bytes, FLOPs. Three numbers are not. They are guesses about how efficiently real software runs, so they are parameters with names, printed next to every result:</p>
<table><tr><th>assumption</th><th>default</th><th>where it comes from</th></tr>
<tr><td><code>mem_util</code></td><td>0.90</td><td>vLLM's <code>--gpu-memory-utilization</code> default</td></tr>
<tr><td><code>bw_util</code></td><td>0.80</td><td>achieved ÷ peak bandwidth; P1.4 exercise 4 measures it</td></tr>
<tr><td><code>mfu</code></td><td>0.50</td><td>achieved ÷ peak FLOP/s in prefill; a guess until P2</td></tr></table>
<p>Their effect is direct. Decode ceilings scale linearly with <code>bw_util</code>: 0.9 instead of 0.8 is 12.5% more tokens/s. Prefill time scales with <code>1 / mfu</code>. <code>mem_util</code> moves the KV budget by 2.4 GiB per 0.1 on an L4, which is 4 to 5 sequences at 4k.</p>
<p>Underneath sit two more layers of uncertainty: the model presets and <code>gpu_specs.yaml</code> are UNVERIFIED. The CLI prints the GPU's status in its header. The path to trust is calibration: in P2.1 you start vLLM on a <code>g6.xlarge</code>, read the KV-cache size it logs at startup (exercise 3 parses that line), and compare it with your prediction. The gap tells you what <code>mem_util</code> and the activation reserve really are.</p>`,
        check: { q: "Which of these calculator inputs are guesses you should calibrate, rather than facts?",
          options: ["The KV bytes per token formula", "bw_util, mfu and the activation reserve", "The number of layers in the config"], answer: 1,
          why: "Bytes per token and layer counts follow from the model's config (once verified). How close real kernels get to peak bandwidth and FLOP/s, and how much memory the engine reserves, are properties of the software and hardware together, so you measure them (P1.4 exercise 4, P2.1)." },
        scene(G) {
          G.text(24, 40, "a prediction = physics × assumptions", { size: 13 });
          G.box(24, 60, 280, 150, "", { stroke: "ok" });
          G.text(40, 86, "physics (from inputs)", { size: 13, color: "ok" });
          ["params × bytes", "2·L·H_kv·h·bytes per token", "2 · active FLOPs per token", "bytes ÷ bandwidth"].forEach((t, i) => G.label(40, 114 + i * 24, t, { size: 12, color: "ink" }));
          G.box(336, 60, 280, 150, "", { stroke: "v" });
          G.text(352, 86, "assumptions (calibrate)", { size: 13, color: "v" });
          const kn = [["mem_util", 0.9], ["bw_util", 0.8], ["mfu", 0.5]].map(([n, v], i) => {
            const y = 112 + i * 30;
            G.label(352, y + 4, n, { size: 12, color: "ink" });
            G.rect(440, y - 6, 150, 10, { fill: "line", rx: 5 });
            return G.rect(440, y - 6, 150 * v, 10, { fill: "v", rx: 5 });
          });
          G.text(24, 252, "inputs marked UNVERIFIED", { size: 13, color: "hot" });
          G.label(24, 276, "presets/*.json · gpu_specs.yaml", { size: 12 });
          G.text(24, 320, "calibrate in P2.1 (g6.xlarge)", { size: 13, color: "ok" });
          G.label(24, 344, "vLLM's logged KV-cache tokens vs max_sequences × context", { size: 12 });
          G.label(24, 366, "measured tokens/s vs the decode ceiling → your real bw_util", { size: 12 });
          G.from(kn, { attr: { width: 0 }, stagger: 0.2, duration: 0.5 });
          G.caption("assumptions are named, printed and measured, never hidden");
        } },

      { rail: "build it", title: "Build it: the tests are the specification",
        body: `<p>The project is to write <code>exercises/calculator.py</code> yourself, then compare it with the reference in <code>platform/capacity/core.py</code>. Each function is one row of this lesson:</p>
<div class="eq">total_params, active_params    weights, stored vs read
weights_bytes(cfg, dtype, tp)  weights, TP
kv_bytes_per_token(cfg, kd, tp)  max users, TP
max_sequences(cfg, gpu, …)     the budget, max users
parse_vllm_kv_tokens(log)      physics and guesses</div>
<p>The tests check them against hand-computed values: 131,072 bytes per token for Llama-3-8B in bf16, between 8 and 13 sequences on a 24 GB fixture GPU, more than twice as many in fp8, and Mixtral totals within 1% of 46.70 B and 12.75 B.</p>
<p>Exercise 4 is manual: enter three configurations into the Modular handbook's GPU memory calculator and into yours, fill in <code>exercises/crosscheck.md</code>, and explain every difference (GB vs GiB, overhead estimates, how each tool handles GQA, MoE and the LM head).</p>
<p>Before P2.1, run the two "predict first" commands and write down the max sequences at 4k and the batch-1 decode ceiling. You will check them against a real GPU.</p>`,
        scene(G) {
          const rows = [["max sequences at 4k", "11", "52"], ["KV per token", "128 KiB", "64 KiB"], ["weights per GPU", "15.0 GiB", "7.5 GiB"], ["decode B=1 ceiling", "15.7 tok/s", "31.4 tok/s"], ["prefill 2,048 tokens", "526 ms", "263 ms"]];
          G.text(24, 40, "predict first: Llama-3-8B on an L4 (calculator output)", { size: 13 });
          G.label(330, 74, "bf16", { size: 12, color: "ink" }); G.label(450, 74, "--weights fp8 --kv fp8", { size: 12, color: "ink" });
          const cells = [];
          rows.forEach(([n, a, b], i) => {
            const y = 88 + i * 46;
            G.rect(24, y, 592, 38, { fill: "line", rx: 6, opacity: 0.35 });
            G.text(36, y + 24, n, { size: 13 });
            cells.push(G.text(330, y + 24, a, { size: 14, color: "k" }), G.text(450, y + 24, b, { size: 14, color: "ok" }));
          });
          G.label(24, 340, "UNVERIFIED specs and presets · mem_util 0.9 · bw_util 0.8 · mfu 0.5", { size: 11, color: "v" });
          G.label(24, 364, "write these down; P2.1 measures them on a g6.xlarge", { size: 12, color: "ink" });
          G.from(cells, { opacity: 0, stagger: 0.08, duration: 0.25 });
          G.caption("your numbers to beat, before you rent anything");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Weights per GPU = total params × bytes ÷ TP. The KV budget is memory × mem_util − weights − an activation reserve.</li>
<li>Max sequences = ⌊budget ÷ (context × KV per token)⌋. fp8 weights and KV gave Llama-3-8B on an L4 4.7× the sequences, not 2×.</li>
<li>Decode ceiling = bw_util × bandwidth × TP ÷ (active weight bytes + B × context/2 × KV per token). Active excludes the embedding and unused experts; total decides memory.</li>
<li>Prefill floor = (2·active·T + 2·L·H·h·T²) ÷ (mfu × peak × TP).</li>
<li>Everything is a prediction from UNVERIFIED configs and specs plus three named assumptions you calibrate in P2.</li></ul>`,
    sim: {
      title: "Run the capacity calculator",
      intro: "This is <code>plan()</code> from <code>platform/capacity/core.py</code>, re-implemented in the page with the same formulas, presets and <code>gpu_specs.yaml</code> values (all UNVERIFIED). The top bar is one GPU's memory; the chart is the decode ceiling at each batch size that fits. Communication between TP ranks is ignored, as in core.py without <code>link_gbs</code>.",
      height: 300,
      controls: [
        { id: "m", label: "model preset", type: "select", value: "llama3-8b", options: Object.keys(MODELS).map((k) => [k, k]) },
        { id: "g", label: "GPU", type: "select", value: "L4", options: Object.keys(GPUS).map((k) => [k, k]) },
        { id: "w", label: "weights", type: "select", value: "bf16", options: [["bf16", "bf16"], ["fp8", "fp8"], ["int4", "int4"]] },
        { id: "kd", label: "KV cache", type: "select", value: "bf16", options: [["bf16", "bf16"], ["fp8", "fp8"]] },
        { id: "tp", label: "TP (GPUs)", type: "select", value: 1, options: [[1, "1"], [2, "2"], [4, "4"], [8, "8"]] },
        { id: "ctx", label: "context (tokens)", min: 512, max: 32768, step: 512, value: 4096 },
        { id: "mu", label: "mem_util", min: 0.5, max: 0.95, step: 0.05, value: 0.9, format: (x) => x.toFixed(2) },
        { id: "bu", label: "bw_util", min: 0.4, max: 1, step: 0.05, value: 0.8, format: (x) => x.toFixed(2) },
        { id: "mfu", label: "mfu", min: 0.2, max: 0.8, step: 0.05, value: 0.5, format: (x) => x.toFixed(2) },
      ],
      draw(G, v) {
        const m = MODELS[v.m], p = plan(m, v.g, v.w, v.kd, v.tp, v.ctx, v.mu, 1.0, v.bu, v.mfu);
        const memB = p.g.mem * GiB, sc = 600 / memB;
        G.label(20, 16, `one ${v.g} (${p.g.mem} GiB as core.py reads it) · ${m.name} · TP ${v.tp}`, { size: 11 });
        G.rect(20, 24, 600, 34, { stroke: "ink", sw: 1, rx: 4 });
        let x = 20;
        const seg = (b, c, op) => { const wpx = Math.max(0, Math.min(620 - x, b * sc)); G.rect(x, 24, wpx, 34, { fill: c, rx: 3, opacity: op ?? 0.9 }); x += wpx; };
        seg(p.weights, p.fits ? "w" : "hot"); seg(GiB, "q"); if (p.budget > 0) seg(p.budget, "k"); seg(Math.max(0, memB - p.usable), "muted", 0.35);
        const lg = [["w", "weights"], ["q", "activations"], ["k", "KV budget"], ["muted", "outside mem_util"]];
        lg.forEach(([c, t], i) => { G.rect(20 + i * 130, 66, 10, 10, { fill: c, rx: 2 }); G.label(34 + i * 130, 75, t, { size: 10 }); });
        if (!p.fits) {
          G.text(320, 170, "DOES NOT FIT: weights + activations exceed usable memory", { anchor: "middle", size: 14, color: "hot" });
          G.label(320, 194, "raise TP, quantize the weights, or pick a bigger GPU", { anchor: "middle", size: 12 });
        } else {
          const Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256], res = Bs.map((B) => p.decodeAt(B));
          const maxAgg = Math.max(...res.filter((_, i) => Bs[i] <= Math.max(1, p.maxSeqs)).map((r) => r[1]), 1);
          G.label(20, 100, "aggregate decode tokens/s (ceiling) by batch size · label = tokens/s per sequence", { size: 10 });
          G.axes(40, 108, 580, 150, {});
          Bs.forEach((B, i) => {
            const X = 52 + i * 63, ok = B <= p.maxSeqs, [per, agg] = res[i];
            if (ok) {
              const h = (140 * agg) / maxAgg;
              G.rect(X, 258 - h, 40, h, { fill: "k", rx: 3 });
              G.text(X + 20, 252 - h, per.toFixed(1), { anchor: "middle", size: 10, color: "ink" });
            } else G.rect(X, 238, 40, 20, { fill: "hot", rx: 3, opacity: 0.35 });
            G.text(X + 20, 274, String(B), { anchor: "middle", size: 10, color: ok ? "ink" : "hot" });
          });
          G.label(620, 292, "batch (red: more sequences than fit at this context)", { anchor: "end", size: 10 });
        }
        const pre = [512, 2048, 8192].map((T) => [F.num(T) + " tokens", p.prefillAt(T) === null ? "no " + v.w + " peak" : F.ms(p.prefillAt(T))]);
        const d1 = p.decodeAt(1);
        return [
          { title: `Plan · ${p.fits ? "fits" : "does not fit"}`, chip: [p.fits, p.fits ? `${p.maxSeqs} sequences at ${F.num(v.ctx)} ctx` : "DOES NOT FIT"],
            rows: [["total / active params", `${(totalParams(m) / 1e9).toFixed(2)} / ${(activeParams(m) / 1e9).toFixed(2)} B`], ["weights per GPU", gib(p.weights)], ["KV budget per GPU", gib(Math.max(0, p.budget))], ["KV per token (all GPUs)", F.bytes(p.kvPerTok)], ["max KV tokens", F.num(p.maxTok)], ["max sequences", F.num(p.maxSeqs)]] },
          { title: "Decode ceiling", rows: p.fits ? [["batch 1, per sequence", F.num(d1[0], 1) + " tok/s"], ["batch 1, step time", F.ms(d1[2])]].concat(p.maxSeqs >= 1 ? [[`batch ${p.maxSeqs} (max), per sequence`, F.num(p.decodeAt(p.maxSeqs)[0], 1) + " tok/s"], [`batch ${p.maxSeqs}, aggregate`, F.num(p.decodeAt(p.maxSeqs)[1], 0) + " tok/s"]] : []) : [["—", "does not fit"]] },
          { title: "Prefill (TTFT floor)", rows: pre },
          { title: "Assumptions (printed by the CLI)", rows: [["mem_util", v.mu.toFixed(2)], ["activation reserve", "1.0 GiB"], ["bw_util", v.bu.toFixed(2)], ["mfu", v.mfu.toFixed(2)]],
            html: `<p class="note">Predictions from UNVERIFIED presets and specs. Decode assumes sequences half way through their context; TP communication ignored. Check against the CLI: <code>PYTHONPATH=platform uv run python -m capacity.cli --model ${v.m} --gpu ${v.g} --weights ${v.w} --kv ${v.kd} --tp ${v.tp} --context ${v.ctx}</code></p>` },
        ];
      },
    },
    practice: {
      intro: "All five exercises share one test file; <code>-k</code> picks one. <code>S2S_SOLUTIONS=1</code> runs the tests against the reference solution. Run everything with <code>uv run pytest course/P1-inference-fundamentals/P1.5-project-capacity-calculator/exercises</code>.",
      items: [
        { title: "Weights bytes = checkpoint size", tier: "T0 · easy", goal: "weights_bytes(cfg, dtype) must equal, to the byte, the tensor sizes of tiny generated checkpoints, tied and untied. Then compare 3 real models' safetensors sizes by hand.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.5-project-capacity-calculator/exercises -k weights_bytes" },
        { title: "KV bytes per token and max sequences", tier: "T0 · easy", goal: "kv_bytes_per_token and max_sequences: 128 KiB per token for Llama-3-8B, 8–13 sequences on a 24 GB GPU, more than double in fp8.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.5-project-capacity-calculator/exercises -k kv_and_capacity" },
        { title: "Predicted vs vLLM-reported KV capacity", tier: "T0 parser · T2 comparison", goal: "Parse the KV-cache token count from vLLM's startup log (format UNVERIFIED: fix the regex if yours differs), then compare with your prediction on a g6.xlarge.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.5-project-capacity-calculator/exercises -k parse_vllm" },
        { title: "Cross-check with the Modular handbook calculator", tier: "T0 · medium · manual", goal: "Fill in exercises/crosscheck.md for three configurations and explain each difference.",
          cmd: "$EDITOR course/P1-inference-fundamentals/P1.5-project-capacity-calculator/exercises/crosscheck.md" },
        { title: "MoE and TP", tier: "T0 · hard", goal: "total_params and active_params for Mixtral-shaped configs (46.70 B / 12.75 B), and TP that halves weights and KV per GPU.",
          cmd: "uv run pytest course/P1-inference-fundamentals/P1.5-project-capacity-calculator/exercises -k moe_and_tp" },
      ],
      labs: [
        { label: "The reference calculator: plan(), Model.total_params / active_params", path: "platform/capacity/core.py" },
        { label: "The CLI (<code>PYTHONPATH=platform uv run python -m capacity.cli --model llama3-8b --gpu L4</code>)", path: "platform/capacity/cli.py" },
        { label: "Model presets (UNVERIFIED until replaced by real config.json files)", path: "platform/capacity/presets/" },
        { label: "Your starter file", path: "course/P1-inference-fundamentals/P1.5-project-capacity-calculator/exercises/calculator.py" },
        { label: "GPU specs (P1.4)", path: "course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/gpu_specs.yaml" },
      ],
    },
  });
})();
