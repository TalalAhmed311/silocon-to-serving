/* L6 — Attention and Triton (Lane B). Concepts behind the L6 LeetGPU problems: why naive attention is a memory problem,
   online softmax over K/V tiles (FlashAttention), masks and tile skipping, heads/GQA/varlen, decode attention over an
   int8 KV cache, speculative-decoding verification, Triton's tile model, and assembling a transformer block. */
(function () {
  const F = S2S.fmt;
  const LV = "laneB-cuda/L6-attention-and-triton";
  const mono = (h) => h.replace(/<code>(.*?)<\/code>/g, '<span style="font-family:var(--f-mono);font-size:13px">$1</span>');
  const P = (slug, title, diff, prio, goal, cmd) => ({
    title: `#${slug.split("-")[0]} ${title}`, tier: `${diff} · ${prio}`,
    goal: mono(`${goal} Hints and ${cmd === null ? "outline" : "reference solution"}: <code>${LV}/solutions/${slug}/</code>`),
    cmd: cmd === null ? `# no local test: follow the outline in ${LV}/solutions/${slug}/README.md` : `ctest --test-dir build/laneB -R '^L6_${slug}$' --output-on-failure`,
  });
  const nums = (G, x, y, vals, cw, ch, hl, base = "k", gap = 4) => vals.map((v, i) => {
    const c = hl ? hl(i) : null;
    return G.box(x + i * (cw + gap), y, cw, ch, String(v), { fill: c || base, size: 12, boxOpacity: c ? 1 : 0.5 });
  });
  // visited (query tile, key tile) pairs under the L6 kernel's rule: causal upper bound, optional window lower bound
  function visited(N, BM, BN, causal, window) {
    const Tq = Math.ceil(N / BM); let pairs = 0;
    for (let t = 0; t < Tq; t++) {
      const i0 = t * BM; let hi = N, lo = 0;
      if (causal) hi = Math.min(N, i0 + BM);
      if (window > 0) lo = Math.max(0, i0 - window + 1);
      lo = Math.floor(lo / BN) * BN;
      pairs += Math.max(0, Math.ceil((hi - lo) / BN));
    }
    return { pairs, Tq, Tk: Math.ceil(N / BN) };
  }

  S2S.lesson({
    id: "l6", n: "L6", title: "Attention and Triton",
    subtitle: "Lane B · CUDA on LeetGPU · T2 harness for timing; Triton checks run on CPU",
    kicker: "Lesson · ≈ 80 min",
    headline: "Never write down the N × N matrix",
    intro: `<p>Attention is the operation that makes a transformer a transformer, and it is where the sequence length bites: its natural implementation builds an N × N table of scores. For long sequences that table is far larger than the inputs and outputs combined, and moving it to and from memory dominates the time.</p>
<p>The dark panel builds <b>FlashAttention</b> from what you already know: the online softmax of L4 applied to tiles of keys, so the table is never stored. Then masks (and skipping work), many heads sharing a KV head (GQA), packed variable-length batches, decode attention over a quantized KV cache, verification for speculative decoding, how Triton expresses the same tile program, and how the pieces of L1 to L6 assemble into a transformer block.</p>`,
    facts: ["9 steps", "5 checkpoints", "1 simulator", "7 core problems"],
    legend: [["k", "query / key"], ["v", "value / state"], ["hot", "skipped or wasted"], ["ok", "computed / good"]],
    prev: "l5",
    steps: [
      { rail: "naive cost", title: "Naive attention writes an N × N table to memory",
        body: `<p>For one head, attention takes queries Q, keys K and values V, each N × d (N tokens, d numbers per token), and computes <code>O = softmax(Q·Kᵀ / √d) · V</code>:</p>
<ol><li><b>Scores</b> S = Q·Kᵀ / √d: an N × N matrix, row i holding how much token i matches every token j.</li>
<li><b>Softmax</b> along each row: P = softmax(S), the attention weights.</li>
<li><b>Mix</b> O = P·V: each output row is a weighted average of the value rows.</li></ol>
<p>Take Llama-3-8B's head dimension d = 128 and a 4,096-token sequence, in fp32 (the L6 reference solutions are fp32):</p>
<div class="eq">S = 4096 * 4096 * 4 B = 64 MiB per head
32 heads: 2 GiB just for S

FLOPs per head: QK^T + PV = 4 * N^2 * d
  = 4 * 16,777,216 * 128 = 8.6 GFLOP
S and P through DRAM: write S, read S,
  write P, read P = 4 * 64 MiB = 256 MiB / head
Q, K, V, O: 4 * 4096 * 128 * 4 B = 8 MiB</div>
<p>The N² table costs 32 times the bytes of all four real tensors, and it grows with the <i>square</i> of the sequence length: double N and S quadruples. The arithmetic (8.6 GFLOP per head) is not the problem; the table is. A kernel that never writes S and P to DRAM removes that cost entirely. That is FlashAttention, and the idea behind it is already in your toolbox from L4.</p>`,
        scene(G) {
          G.text(24, 30, "naive: three kernels, two N × N tables", { size: 13 });
          G.box(24, 56, 60, 100, "Q\nN×d", { fill: "k", size: 11 }); G.box(94, 56, 100, 60, "Kᵀ  d×N", { fill: "k", size: 11 });
          G.arrow(200, 90, 240, 90, { color: "muted", w: 2 });
          G.rect(246, 56, 120, 120, { fill: "hot", rx: 6, opacity: 0.85 }); G.text(262, 112, "S", { size: 20, color: "bg", weight: 700 }); G.label(250, 192, "N×N = 64 MiB / head", { size: 11, color: "hot" });
          G.arrow(372, 90, 398, 90, { color: "muted", w: 2 });
          G.rect(404, 56, 120, 120, { fill: "hot", rx: 6, opacity: 0.55 }); G.text(430, 112, "P", { size: 20, color: "bg", weight: 700 }); G.label(408, 192, "P = softmax(S): 64 MiB", { size: 11, color: "hot" });
          G.label(24, 230, "each table: written, then read back from DRAM", { size: 12, color: "ink" });
          G.box(24, 252, 130, 38, "V  N×d", { fill: "q", size: 12 }); G.text(168, 278, "→", { size: 20, color: "ink" }); G.box(196, 252, 130, 38, "O = P·V", { fill: "ok", size: 12 });
          G.text(24, 336, "traffic of the tables: 256 MiB per head", { size: 14, color: "hot" });
          G.text(24, 362, "the real tensors Q, K, V, O: 8 MiB", { size: 14, color: "ok" });
          G.text(24, 394, "N² grows with the square of the sequence length", { size: 13, color: "ink" });
          G.caption("d = 128, N = 4096, fp32: the scores table is 32× the inputs");
        } },

      { rail: "online tiles", title: "Online softmax over tiles: the table is never stored",
        body: `<p>L4's online softmax kept a running <code>(m, d)</code>: the max so far and the sum of <code>e^(x − m)</code>. Attention needs the same for each query row, plus the running <i>weighted sum of values</i>. Keep three things per query row: the max <code>m</code>, the softmax denominator <code>l</code>, and the unnormalized output <code>o = Σ e^(s_j − m) · v_j</code>. Process the keys in <b>tiles</b>. For each new tile with scores s<sub>j</sub>:</p>
<div class="eq">m_new = max(m, max_j s_j)
corr  = e^(m - m_new)      (1 if max unchanged)
l = l * corr + sum_j e^(s_j - m_new)
o = o * corr + sum_j e^(s_j - m_new) * v_j
at the end:  out = o / l</div>
<p>One query row, four keys, scalar values (d = 1), in two tiles:</p>
<div class="eq">scores [2, 0 | 1, 3]    values [30, 40 | 10, 20]
exact: e^2*30 + 1*40 + e*10 + e^3*20 = 690.57
       denominator 7.389 + 1 + 2.718 + 20.086 = 31.193
       out = 690.57 / 31.193 = 22.14

tile 1: m = 2, l = 1 + e^-2 = 1.1353
        o = 1*30 + 0.1353*40 = 35.412
tile 2: max 3 > 2, corr = e^-1 = 0.3679
        l = 1.1353*0.3679 + e^-2 + 1 = 1.5530
        o = 35.412*0.3679 + 0.1353*10 + 1*20 = 34.381
out = 34.381 / 1.5530 = 22.14   ✓</div>
<p>Same answer, and at no moment did we hold more than one tile of scores. The rescale by <code>corr</code> is what lets us fix an earlier tile after the max moved. Two partial states also merge: <code>M = max(m_a, m_b)</code>, <code>l = l_a·e^(m_a−M) + l_b·e^(m_b−M)</code>, <code>o</code> likewise. That merge is associative, exactly as in L3.</p>`,
        check: { q: "A query row has state (m, l, o) = (2, 1.1353, 35.412). The next key tile has max score 3. By what factor must the old l and o be multiplied before adding the new tile's terms?",
          options: ["e^(3 − 2) = 2.718", "e^(2 − 3) = 0.3679", "1: the old terms are already correct"], answer: 1,
          why: "The old terms were computed as e^(s − 2). Relative to the new max 3 they must be e^(s − 3) = e^(s − 2) · e^(2 − 3), so they are scaled by e^(m_old − m_new) = e^−1 = 0.3679, which is below 1. Multiplying by e^(+1) would blow the terms up instead of re-expressing them against the larger max." },
        scene(G) {
          G.text(24, 30, "one query row · 4 keys · 2 tiles · scalar values", { size: 13 });
          G.label(24, 56, "tile 1: scores 2, 0 → values 30, 40", { size: 11, color: "k" });
          G.box(24, 64, 60, 30, "2", { fill: "k", size: 13 }); G.box(90, 64, 60, 30, "0", { fill: "k", size: 13 });
          G.label(330, 56, "tile 2: scores 1, 3 → values 10, 20", { size: 11, color: "q" });
          G.box(330, 64, 60, 30, "1", { fill: "q", size: 13 }); G.box(396, 64, 60, 30, "3", { fill: "q", size: 13 });
          const row = (y, name, m, l, o, c, note) => { G.label(24, y - 8, name, { size: 11, color: c }); G.box(24, y, 80, 36, `m = ${m}`, { fill: c, size: 12 }); G.box(110, y, 130, 36, `l = ${l}`, { fill: c, size: 12 }); G.box(246, y, 150, 36, `o = ${o}`, { fill: c, size: 12 }); G.label(406, y + 22, note, { size: 11 }); };
          row(140, "after tile 1", "2", "1.1353", "35.412", "k", "");
          G.arrow(130, 180, 130, 214, { color: "hot", w: 2 }); G.label(142, 200, "tile 2 raises max to 3: scale by e^(2−3) = 0.368", { size: 11, color: "hot" });
          row(236, "after tile 2", "3", "1.5530", "34.381", "v", "o / l = 22.14");
          G.text(24, 322, "exact softmax(scores)·values = 22.14  ✓", { size: 14, color: "ok" });
          G.label(24, 350, "never more than one tile of scores in memory", { size: 12, color: "ink" });
          G.label(24, 372, "merge of two states: M = max, rescale each by e^(m − M)", { size: 12, color: "ink" });
          G.caption("the correction factor re-expresses old terms against the new max");
        } },

      { rail: "the kernel", title: "One block per query tile, K/V tiles streamed through shared memory",
        body: `<p>Turn the tile recurrence into a kernel. The reference (<code>${LV}/solutions/_shared/attn_core.cuh</code>, used by every L6 solution) follows the FlashAttention-2 structure:</p>
<ol><li>One block owns <code>FBR = 64</code> query rows of one (sequence, head); one thread per row, keeping <code>q</code>, the output accumulator <code>o</code> and the state <code>(m, l)</code> in registers.</li>
<li>Loop over the keys in tiles of <code>FBC = 32</code>: load the K and V tile into shared memory (L2), barrier.</li>
<li>Each thread computes its 32 scores against the tile, takes the tile max, rescales <code>(o, l)</code> <b>once per tile</b>, and accumulates <code>p · v</code>.</li>
<li>After the last tile, write <code>o / l</code>: one division per output element.</li></ol>
<p>The S tile (a thread's 32 scores) lives in registers and is discarded. DRAM sees Q once, O once, and K and V once <i>per query tile</i>. Count K and V tile loads in fp32:</p>
<div class="eq">query tiles = N / BM
K, V per query tile: 2 * N * d * 4 B
total: (N/BM) * 2*N*d*4 = 8*N^2*d / BM bytes
naive S, P: 4 passes * N^2 * 4 B = 16*N^2 bytes
ratio naive / flash = 2*BM / d

N = 4096, d = 64, BM = 128:
  flash 64 MiB vs naive 256 MiB = 4x</div>
<p>Two honest caveats. First, this is the traffic of tile <i>requests</i>: the K/V tiles are the same for many blocks running at once, so L2 serves many of them, and real DRAM traffic can be lower. The papers' simplification, O(N·d) traffic, assumes ideal reuse. Second, the savings grow with the tile height BM, which is limited by registers and shared memory. The footprint saving is unconditional: extra memory drops from N² to just <code>(m, l)</code>, 8 bytes per query row.</p>
<p>FlashAttention-2 (Dao, 2023) versus the first version: parallelize over query tiles (the grid's x dimension) as well as batch and heads, keep the output unnormalized and divide once at the end, and split work between warps to avoid shared-memory traffic. FA-2 matters because the matrix multiplications use the tensor cores (L5) while the exponentials and rescales do not; removing non-matmul work is where the speed comes from. The L6 exit check is FA-2 forward ≥ 5× faster than your naive kernel at sequence length 4,096 (fp16), against PyTorch SDPA for correctness, measured on the P5.8 harness (<code>TODO(run-on: g6.xlarge)</code>).</p>`,
        scene(G) {
          G.text(24, 30, "block = 64 query rows, loops over K/V tiles", { size: 13 });
          G.box(24, 58, 90, 150, "Q tile", { fill: "k", size: 12 }); G.label(30, 224, "64 rows, regs", { size: 10 });
          [0, 1, 2, 3, 4].forEach((t) => G.box(150 + t * 62, 58, 56, 70, `K${t}`, { fill: t === 2 ? "v" : "line", size: 12, boxOpacity: t === 2 ? 1 : 0.6 }));
          [0, 1, 2, 3, 4].forEach((t) => G.box(150 + t * 62, 138, 56, 70, `V${t}`, { fill: t === 2 ? "v" : "line", size: 12, boxOpacity: t === 2 ? 1 : 0.6 }));
          G.label(150, 240, "K/V tiles stream through shared memory (amber = current)", { size: 11, color: "v" });
          G.box(480, 58, 140, 150, "m, l, o", { fill: "q", size: 13 }); G.label(486, 224, "state per row", { size: 10 });
          G.arrow(118, 120, 146, 120, { color: "muted", w: 1.5 }); G.arrow(466, 120, 478, 120, { color: "muted", w: 1.5 });
          G.text(24, 276, "tile loads: 8 · N² · d / BM bytes (fp32)", { size: 14, color: "v" });
          G.text(24, 304, "naive S, P traffic: 16 · N² bytes", { size: 14, color: "hot" });
          G.text(24, 332, "ratio = 2 · BM / d   (N=4096, d=64, BM=128: 4×)", { size: 13, color: "ok" });
          G.label(24, 362, "extra memory: N² → 8 bytes per row (m, l)", { size: 12, color: "ink" });
          G.label(24, 384, "FA-2: parallel over query tiles, one divide at the end", { size: 12, color: "ink" });
          G.caption("the score tile is born and dies in registers");
        } },

      { rail: "masks and skipping", title: "Causal and windowed attention: skip the tiles that cannot matter",
        body: `<p>A language model generating text must not look at the future: token i may attend only to j ≤ i. This is a <b>causal</b> mask: scores with j &gt; i are set to −∞ <i>before</i> the max, so they contribute e<sup>−∞</sup> = 0 and cannot move m. In tile terms, the picture is a triangle. A K/V tile that lies entirely after the last query row of the block is fully masked, so the kernel never loads it:</p>
<div class="eq">j_hi = min(Lk, i0 + FBR)   (causal loop bound)
only the diagonal tile needs per-element masks</div>
<p>Count the saving. (Tile size here is conceptual; the reference kernel uses 64 query rows by 32 keys. The ratios barely change.) N = 2,048 with 64 × 64 tiles gives a 32 × 32 grid of tile pairs. Causal attention visits the lower triangle: 32 · 33 / 2 = 528 of 1,024, or 51.6%: about half the work.</p>
<p><b>Sliding-window attention</b> (each query sees only the last w keys, as in several long-context models) also cuts the lower end: <code>j_lo = max(0, i0 − w + 1)</code>, rounded down to a tile. With w = 512 (8 tiles), query tile t visits key tiles t−8 … t: 9 tiles once t ≥ 8, and t + 1 before that:</p>
<div class="eq">t = 0 … 7:   1 + 2 + ... + 8 = 36
t = 8 … 31:  24 tiles * 9      = 216
total 252 of 1,024 tile pairs = 24.6%</div>
<p>The cost is O(N · w) instead of O(N²). ALiBi (#55) adds a per-head bias <code>−slope · (i − j)</code> to the scores inside the tile; attention sinks (#112) add one extra softmax term with no value vector (the kernel starts with <code>m = sink, l = 1</code>). One shared core handles all of them through parameters. Careful with the all-masked row: if a tile has every score at −∞, then <code>m_new</code> stays −∞ and <code>e^(−∞ − (−∞))</code> is NaN, which is why the kernel checks <code>m_new != -INFINITY</code> first.</p>`,
        check: { q: "Causal attention, N = 2,048, tiles of 64 × 64 (a 32 × 32 grid of tile pairs). How many tile pairs does the kernel visit?",
          options: ["1,024: all of them", "528: the lower triangle including the diagonal", "496: only strictly below the diagonal"], answer: 1,
          why: "Query tile t must see key tiles 0 … t (the diagonal tile is partially masked but still needed). That is t + 1 tiles for t = 0 … 31: 1 + 2 + … + 32 = 32 · 33 / 2 = 528, which is 51.6% of 1,024. Skipping the diagonal tile (496) would drop the keys with j ≤ i inside it." },
        scene(G) {
          const T = 32, cs = 8.6, g = 0.6;
          const map = (ox, oy, name, vis, c) => {
            G.label(ox, oy - 8, name, { size: 11, color: c });
            for (let t = 0; t < T; t++) for (let k = 0; k < T; k++) G.rect(ox + k * cs, oy + t * cs, cs - g, cs - g, { fill: vis(t, k) ? c : "line", rx: 1, opacity: vis(t, k) ? 0.9 : 0.35 });
          };
          G.text(24, 28, "tile pairs, N = 2048, 64×64 tiles (rows: query tile, cols: key tile)", { size: 12 });
          map(24, 56, "causal: 528 of 1,024 = 51.6%", (t, k) => k <= t, "ok");
          map(332, 56, "window 512: 252 of 1,024 = 24.6%", (t, k) => k <= t && k >= t - 8, "v");
          G.label(24, 350, "gray tiles are never loaded or computed", { size: 12, color: "ink" });
          G.label(24, 372, "diagonal tile (and band edge) still needs per-element masking", { size: 12 });
          G.label(24, 396, "mask → −∞ before the max: masked keys cannot move m", { size: 12, color: "hot" });
          G.caption("work follows the shape of what each query may see");
        } },

      { rail: "heads, GQA, varlen", title: "Many heads, shared KV heads, and sequences of different lengths",
        body: `<p><b>Multi-head attention.</b> A model runs h independent attention heads, each with its own d<sub>h</sub>-dimensional slice of the projections. Do not transpose the tensor into per-head layout with a separate kernel. Put the head in the grid (<code>blockIdx.y</code>) and address through strides: token stride d<sub>model</sub>, head stride d<sub>h</sub> (#12). The output is written in the same packed layout, ready for the output projection (L5). A <b>cross-attention</b> variant (#26) has Lq ≠ Lk: the same code with different lengths.</p>
<p><b>Grouped-query attention (GQA).</b> Keys and values for decoding are cached, and the cache is the memory hog. GQA lets several query heads share one K/V head. Llama-3-8B has 32 query heads and 8 KV heads (head dim 128): each group of 4 query heads reads one KV head, <code>kv_head = h / (H / Hkv)</code>. The cache per token (2 tensors × KV heads × d × 2 bytes × 32 layers in bf16) shrinks four times:</p>
<div class="eq">GQA: 2 * 8 * 128 * 2 B * 32 layers
   = 131,072 B = 128 KiB per token
MHA: 2 * 32 * 128 * 2 B * 32 = 512 KiB per token
4096 tokens: 512 MiB (GQA) versus 2 GiB (MHA)</div>
<p>MQA is the extreme Hkv = 1; MHA is Hkv = H. The test covers both. A kernel that handles a whole group of query heads in one block reads each K/V tile once instead of four times: the next optimization to try (#80).</p>
<p><b>Variable-length batches (#102).</b> Padding every sequence to the longest wastes compute and memory. Instead, pack the sequences back to back along one token axis and pass prefix offsets <code>cu_seqlens</code>:</p>
<div class="eq">lengths 5, 3, 8 → cu_seqlens = [0, 5, 8, 16]
packed: 16 token slots
padded to 8: 3 * 8 = 24 slots
padding waste = 8 / 24 = 33%</div>
<p>The grid is (query tiles of the <i>longest</i> sequence, batch × heads). Each block reads its sequence's offset and length from <code>cu_seqlens</code> and exits at once if its tile starts past the end of that sequence. Attention never crosses sequence boundaries, so the same offsets select the K/V range. This is the layout continuous batching produces (P2.3 and P6.2), and FlashAttention's <code>varlen</code> API takes exactly these offsets.</p>`,
        check: { q: "Llama-3-8B has 32 query heads and 8 KV heads. Query head 13 reads which KV head, and how many query heads share each KV head?",
          options: ["KV head 3; 4 query heads per KV head", "KV head 13; 1 per KV head", "KV head 1; 8 per KV head"], answer: 0,
          why: "The group size is H / Hkv = 32 / 8 = 4, so kv_head = h / 4 using integer division: 13 / 4 = 3. Heads 12, 13, 14 and 15 all read KV head 3. The KV cache therefore holds 8 heads instead of 32: a 4× saving." },
        scene(G) {
          G.text(24, 28, "GQA: 8 query heads (drawn) share 2 KV heads", { size: 13 });
          for (let h = 0; h < 8; h++) G.box(24 + h * 74, 54, 62, 32, `Q${h}`, { fill: h < 4 ? "k" : "q", size: 12 });
          [0, 1].forEach((g) => { G.box(24 + g * 296 + 40, 150, 200, 38, `KV head ${g}`, { fill: g ? "q" : "k", size: 12 }); for (let t = 0; t < 4; t++) G.line(24 + (g * 4 + t) * 74 + 31, 86, 24 + g * 296 + 140, 150, { color: g ? "q" : "k", w: 1 }); });
          G.label(24, 206, "Llama-3-8B: 32 query heads / 8 KV heads → 4 per group", { size: 11, color: "ink" }); G.label(24, 222, "KV cache 128 KiB per token (MHA: 512 KiB)", { size: 11, color: "ink" });
          G.text(24, 252, "varlen packing: lengths 5, 3, 8", { size: 13, color: "v" });
          G.label(24, 274, "padded to 8 → 24 slots", { size: 11, color: "hot" });
          [5, 3, 8].forEach((n, s) => { for (let i = 0; i < 8; i++) G.rect(210 + i * 26, 262 + s * 22, 22, 18, i < n ? { fill: ["k", "v", "q"][s], rx: 3, opacity: 0.9 } : { fill: "hot", rx: 3, opacity: 0.4 }); });
          G.label(24, 342, "packed → 16 slots, cu_seqlens = [0, 5, 8, 16]", { size: 11, color: "ok" });
          let x = 24; [5, 3, 8].forEach((n, s) => { for (let i = 0; i < n; i++) { G.rect(x + i * 26, 352, 22, 18, { fill: ["k", "v", "q"][s], rx: 3, opacity: 0.9 }); } x += n * 26 + 2; });
          G.label(24, 392, "padding waste 8 / 24 = 33% avoided; block exits if its tile starts past its sequence", { size: 11, color: "ink" });
          G.caption("share K/V across a head group; pack sequences, don't pad them");
        } },

      { rail: "decode, int8 KV", title: "Decode attention is memory-bound on the KV cache",
        body: `<p>When a model generates one token at a time, each sequence has a single query (the newest token) attending to the whole cache of N past keys and values. There is almost no arithmetic to amortize: per cached token the kernel reads one K row and one V row and does 2·d multiply-adds for each. It is a memory-bound streaming read of the KV cache (P1.2). Example, Llama-3-8B at 4,096 tokens of context, one sequence:</p>
<div class="eq">KV read per step: 4096 * 128 KiB = 512 MiB (bf16)
int8 cache:       256 MiB (+ scales)
at 300 GB/s (example value):
  1.79 ms versus 0.90 ms per step</div>
<p>(Lower bounds for the cache reads of one decode step; they add to the weight reads of L5 step 8.) An <b>int8 KV cache</b> (#96) halves the bytes, and also doubles the number of tokens that fit in GPU memory. The speed-up is only real if dequantization happens in registers as the data streams past, never as a separate pass. Two tricks from the reference README:</p>
<ul><li><b>Scale the dot product, not every element.</b> With a per-token key scale, <code>s = (q · k_int8) · k_scale</code>: one multiply per token instead of d.</li>
<li><b>Fold the value scale into the weight.</b> The contribution of token j is <code>(p_j · v_scale_j) · v_int8_j</code>.</li></ul>
<p>Parallelism: a single query row gives little to parallelize over, so one block takes one (sequence, query head), warps take tokens round-robin and lanes split the head dimension. Each warp keeps its own online-softmax state, and the warps <b>merge</b> at the end with the formula from step 2 (this is the idea behind "flash-decoding" split-K attention). Check the merge with the earlier example: state A = (m = 2, l = 1.1353, o = 35.412) from keys [2, 0], state B = (3, 1.1353, 21.353) from keys [1, 3]. M = 3, l = 1.1353 · e<sup>−1</sup> + 1.1353 = 1.5530, o = 35.412 · 0.3679 + 21.353 = 34.381, out = 22.14, the same as before.</p>`,
        check: { q: "Decode attention reads the whole KV cache each step. Llama-3-8B at 4,096 tokens, bf16 cache, is 512 MiB. Switching to an int8 cache (ignoring scales) changes the bytes read per step to:",
          options: ["256 MiB, half", "128 MiB, a quarter", "512 MiB: dequantization restores the size"], answer: 0,
          why: "int8 is 1 byte per element versus 2 for bf16, so the cache read halves to 256 MiB (the per-token scales are a small extra). Dequantization happens in registers after the load, so it does not add global-memory traffic. Because decode is memory-bound, the time falls by about the same factor." },
        scene(G) {
          G.text(24, 30, "decode: one query reads the whole KV cache", { size: 13 });
          G.box(24, 54, 70, 40, "q (1×d)", { fill: "k", size: 12 });
          for (let t = 0; t < 12; t++) G.box(110 + t * 42, 54, 38, 40, "", { fill: t < 6 ? "v" : "q", boxOpacity: 0.6 });
          G.label(110, 112, "4,096 cached tokens (K and V)", { size: 11, color: "ink" });
          G.text(24, 160, "bytes read per step (Llama-3-8B, 4096 tokens)", { size: 13 });
          G.rect(24, 176, 512 / 2, 30, { fill: "w", rx: 5 }); G.label(24 + 256 + 8, 196, "bf16: 512 MiB · 1.79 ms at 300 GB/s (ex.)", { size: 11 });
          G.rect(24, 216, 256 / 2, 30, { fill: "ok", rx: 5 }); G.label(24 + 128 + 8, 236, "int8: 256 MiB · 0.90 ms", { size: 11, color: "ok" });
          G.text(24, 290, "split across warps, then merge states:", { size: 13, color: "v" });
          [0, 1, 2, 3].forEach((w) => G.box(24 + w * 100, 304, 90, 34, `warp ${w}`, { fill: "k", size: 11 }));
          G.arrow(300, 340, 300, 360, { color: "muted", w: 2 }); G.box(180, 362, 240, 30, "merge: M = max, rescale by e^(m − M)", { fill: "v", size: 11 });
          G.label(24, 412, "dequantize in registers: s = (q·k_int8)·k_scale, weight p·v_scale", { size: 11, color: "ink" });
          G.caption("fewer cache bytes is a direct speed-up; the merge is the L3/L4 pattern");
        } },

      { rail: "speculative verify", title: "Speculative decoding: accept or reject draft tokens exactly",
        body: `<p>Decoding is slow because each token needs a full pass of the big <i>target</i> model. <b>Speculative decoding</b> lets a small <i>draft</i> model propose k tokens cheaply, then the target scores all k at once (one pass, like prefill) and the system keeps as many as it can. It must keep the target's output distribution <b>exactly</b>. The rule is rejection sampling. Let q(x) be the draft's probability of its proposed token x and p(x) the target's:</p>
<ul><li>Accept x with probability <code>min(1, p(x) / q(x))</code>. Example: p = 0.6, q = 0.8 → accept with probability 0.75; draw u = 0.7, and since 0.7 &lt; 0.75, accept.</li>
<li>At the first rejection, stop and sample a replacement from the <b>residual</b> <code>max(0, p − q)</code>, normalized.</li>
<li>If all k are accepted, sample a bonus token from the target's next distribution.</li></ul>
<p>Why is this exact? Take p = [0.5, 0.4, 0.1] and q = [0.2, 0.3, 0.5] over three tokens. The chance the draft proposes x and it is accepted is q(x)·min(1, p/q) = min(p, q) = [0.2, 0.3, 0.1]. The chance of rejection is 1 − Σmin = 0.4. The residual is max(0, p − q) = [0.3, 0.1, 0] with total 0.4, normalized [0.75, 0.25, 0]. So the output probabilities are:</p>
<div class="eq">token 0: 0.2 + 0.4 * 0.75 = 0.5   = p(0)
token 1: 0.3 + 0.4 * 0.25 = 0.4   = p(1)
token 2: 0.1 + 0.4 * 0    = 0.1   = p(2)</div>
<p>It always works: the total of the positive part Σ(p − q)<sub>+</sub> equals the rejection probability, so min(p, q) + (p − q)<sub>+</sub> = p. The kernel (#87) takes the random numbers as inputs (<code>u_accept</code>, <code>u_sample</code>), which makes it deterministic and testable against a CPU reference. The accept chain is sequential but k ≤ 8, so one thread does it. The expensive part is sampling from a vocabulary-sized distribution: a block-wide sum of the weights, a chunked prefix scan (L3), and the first index whose cumulative sum exceeds <code>u · total</code>. Each sequence accepts a different count, so the kernel returns <code>n_out</code> per sequence. Doing this on the GPU avoids copying (k + 1) · V probabilities to the host each step.</p>`,
        check: { q: "Target p(x) = 0.3 and draft q(x) = 0.6 for the proposed token x. With what probability is x accepted?",
          options: ["0.5", "1.0", "0.3"], answer: 0,
          why: "The accept probability is min(1, p / q) = min(1, 0.3 / 0.6) = 0.5. The draft over-proposes this token (q is twice p), so it is accepted only half the time, which corrects the excess. If instead p ≥ q the token is always accepted." },
        scene(G) {
          G.text(24, 28, "p = [0.5, 0.4, 0.1] (target), q = [0.2, 0.3, 0.5] (draft)", { size: 12 });
          const px = [0.5, 0.4, 0.1], qx = [0.2, 0.3, 0.5], mn = [0.2, 0.3, 0.1], res = [0.3, 0.1, 0];
          const col = (x0, vals, c, nm) => { G.label(x0, 52, nm, { size: 11, color: c }); vals.forEach((v, i) => { G.rect(x0 + i * 52, 188 - v * 220, 44, v * 220, { fill: c, rx: 4, opacity: 0.9 }); G.label(x0 + i * 52 + 6, 204, v.toFixed(1), { size: 11 }); }); };
          col(24, px, "k", "target p"); col(200, qx, "q", "draft q"); col(376, mn, "ok", "accepted: min(p, q)");
          G.label(24, 232, "reject chance = 1 − Σ min = 0.4  = Σ max(0, p − q) = 0.3 + 0.1", { size: 12, color: "hot" });
          G.label(24, 262, "residual max(0, p − q) = [0.3, 0.1, 0] → normalized [0.75, 0.25, 0]", { size: 12, color: "v" });
          G.text(24, 306, "token 0: 0.2 + 0.4 · 0.75 = 0.5 = p(0)", { size: 13, color: "ok" });
          G.text(24, 332, "token 1: 0.3 + 0.4 · 0.25 = 0.4 = p(1)", { size: 13, color: "ok" });
          G.text(24, 358, "token 2: 0.1 + 0.4 · 0    = 0.1 = p(2)", { size: 13, color: "ok" });
          G.label(24, 394, "accept prob = min(1, p/q) · stop at the first rejection · bonus token if all k accepted", { size: 11, color: "ink" });
          G.caption("accepted mass plus residual mass reproduces the target distribution exactly");
        } },

      { rail: "Triton", title: "Triton: the same tile program, written for a tile",
        body: `<p>In CUDA you wrote per-thread code and managed threads, shared memory, barriers and tensor-core fragments yourself. <b>Triton</b> is a Python-embedded language where one <i>program</i> owns a tile and you write tensor operations on whole tiles (<code>tl.load</code>, <code>tl.dot</code>, <code>tl.max</code>, <code>tl.sum</code>). The compiler picks the threads, stages data in shared memory, inserts barriers and uses tensor cores for <code>tl.dot</code>. Lane A's P5.9 teaches the model from a vector add; here is the same FlashAttention loop of step 2 and 3, from <code>platform/kernels/triton_kernels/flash_attention.py</code>:</p>
<div class="eq">q = tl.load(Q + ...)              # [BM, D] tile
m_i = tl.full([BM], -inf)
l_i = tl.zeros([BM])
acc = tl.zeros([BM, D])
for start_n in range(0, hi, BN):  # hi: N or causal
    k = tl.load(K + ...)          # K tile
    s = tl.dot(q, k) * scale      # [BM, BN]
    s = tl.where(valid, s, -inf)  # mask first
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp(m_i - m_new)   # correction
    p = tl.exp(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p, v)
    m_i = m_new
acc = acc / l_i[:, None]          # divide once</div>
<p>Every line matches a step you have seen: the state <code>(m_i, l_i, acc)</code>, the correction <code>alpha</code> (step 2), the mask before the max and the causal upper bound <code>hi</code> (step 4). Differences from the CUDA core: both matrix products are <code>tl.dot</code> and run on tensor cores, there is no explicit shared memory or <code>__syncthreads</code>, and the grid is <code>(cdiv(N, BM), B · H)</code>: for N = 4,096, BM = 64, B = 1 and 32 heads, 64 × 32 = 2,048 programs. Tile sizes are <code>tl.constexpr</code> values, so Triton compiles a specialized kernel per choice and an autotune list searches over them (and over <code>num_stages</code>, the pipelining depth that plays the role of double buffering from L5).</p>
<p>What you give up is control: you cannot place a specific register or choose the exact shared-memory layout. For many kernels that is a good trade, and it is easy to test: Triton's interpreter (<code>TRITON_INTERPRET=1</code>) runs the kernels on CPU tensors, so correctness checks need no GPU. The L6 Triton pass: rewrite softmax, RMSNorm, GEMM and FA-2 in Triton and compare them with your CUDA versions (P5.9).</p>`,
        scene(G) {
          G.text(24, 28, "CUDA (L1 to L6)", { size: 13, color: "k" }); G.text(330, 28, "Triton (P5.9)", { size: 13, color: "v" });
          const rows = [["thread, warp, threadIdx", "program on a tile: tl.program_id"], ["shared memory + __syncthreads", "implicit staging by the compiler"], ["fragments, ldmatrix, mma.sync", "tl.dot(a, b)"], ["cp.async double buffering", "num_stages"], ["if (i < N) guard", "mask= on tl.load / tl.store"], ["tile sizes as templates", "tl.constexpr + autotune"], ["per-thread (m, l, o) row state", "m_i[BM], l_i[BM], acc[BM, D]"]];
          rows.forEach(([a, b], i) => { const y = 48 + i * 40; G.box(24, y, 290, 32, a, { fill: "k", size: 11, boxOpacity: 0.55 }); G.arrow(318, y + 16, 326, y + 16, { color: "muted", w: 1.5 }); G.box(330, y, 290, 32, b, { fill: "v", size: 11, boxOpacity: 0.7 }); });
          G.text(24, 352, "grid = (cdiv(N, BM), B·H)", { size: 14, color: "ink" });
          G.label(24, 376, "N = 4096, BM = 64, 32 heads → 64 × 32 = 2,048 programs", { size: 12, color: "ok" });
          G.label(24, 398, "TRITON_INTERPRET=1 runs kernels on CPU tensors: no GPU for correctness", { size: 11, color: "ink" });
          G.caption("same algorithm, written for tiles instead of threads");
        } },

      { rail: "a whole block", title: "Putting it together: one Llama layer from your kernels",
        body: `<p>The hard problems (#74 GPT-2 block, #93 Llama block, #116 diffusion-transformer block) ask you to assemble a full transformer layer from kernels you wrote in L1 to L6. A Llama layer for a batch of tokens x:</p>
<ol><li><b>RMSNorm</b> (L4): one block per row.</li>
<li><b>QKV projection</b>: a GEMM (L5) with the weights of Q, K, V concatenated, so x is read once.</li>
<li><b>RoPE</b> (L4) on q and k, ideally in the GEMM epilogue (#115).</li>
<li><b>Attention</b> (L6): FlashAttention prefill, or decode attention over the KV cache; GQA via the head mapping.</li>
<li><b>Output projection</b> GEMM with the residual add fused (L5 epilogue).</li>
<li><b>Fused residual + RMSNorm</b> (L4).</li>
<li><b>Gate and up GEMM</b> with the SwiGLU epilogue (L5).</li>
<li><b>Down projection</b> GEMM with the residual add.</li></ol>
<p>Roughly eight kernel launches per layer, 32 layers. Where do the parameters (and so the weight reads at decode, L5) sit? In Llama-3-8B, with hidden size 4,096, 8 KV heads of 128 (so K and V are 4,096 × 1,024) and MLP width 14,336:</p>
<div class="eq">Q: 4096*4096 = 16,777,216
K, V: 4096*1024 = 4,194,304 each
O: 16,777,216  → attention 41,943,040
MLP: 3 * 4096 * 14336 = 176,160,768
layer: 218,103,808; * 32 = 6,979,321,856
embedding + head: 2 * 128,256 * 4096
  = 1,050,673,152
matrices 8,029,995,008
+ RMSNorm weights 266,240
total 8,030,261,248 ≈ 8.0 billion</div>
<p>(Untied embedding and output matrices; the total matches P1.1.) The MLP holds 81% of each layer's parameters and the attention projections 19%. At decode, GEMMs read weights and attention reads the KV cache: the sum of the two streams, in bytes, sets the time per token. At prefill (many tokens) the GEMMs become compute-bound and attention's N² behaviour matters. Profile with <code>nsys</code> to see which kernel dominates at batch 1 and at 4,096 tokens, as #74's outline suggests, then ask the question this whole lane trained: <i>how many bytes does this kernel move, and how many does it have to?</i> The reference for the GPT-2 structure is llm.c's <code>train_gpt2.c</code> forward pass (MIT-licensed; read it, don't copy it).</p>`,
        scene(G) {
          const steps = [["RMSNorm", "L4", "k"], ["QKV GEMM", "L5", "q"], ["RoPE", "L4", "k"], ["attention", "L6", "ok"], ["O GEMM + residual", "L5", "q"], ["add + RMSNorm", "L4", "k"], ["gate/up GEMM + SwiGLU", "L5", "q"], ["down GEMM + residual", "L5", "q"]];
          G.text(24, 28, "one Llama layer: 8 kernels, 32 layers", { size: 13 });
          steps.forEach(([n, l, c], i) => { const y = 46 + i * 38; G.box(24, y, 230, 30, n, { fill: c, size: 12, boxOpacity: 0.9 }); G.label(262, y + 20, l, { size: 11, color: c }); if (i < 7) G.arrow(139, y + 31, 139, y + 37, { color: "muted", w: 1 }); });
          const px = 330;
          G.label(px, 52, "parameters per layer (Llama-3-8B)", { size: 11, color: "ink" });
          G.rect(px, 62, 41.9 / 218.1 * 280, 26, { fill: "ok", rx: 4 }); G.label(px, 104, "attention 41.9 M (19%)", { size: 11, color: "ok" });
          G.rect(px, 116, 280, 26, { fill: "q", rx: 4, opacity: 0.4 }); G.rect(px, 116, 176.2 / 218.1 * 280, 26, { fill: "q", rx: 4 }); G.label(px, 158, "MLP 176.2 M (81%)", { size: 11, color: "q" });
          G.text(px, 210, "× 32 layers = 6.98 B", { size: 13, color: "ink" });
          G.text(px, 236, "+ embed / head 1.05 B", { size: 13, color: "ink" });
          G.text(px, 262, "= 8.03 B parameters", { size: 14, color: "v" });
          G.label(px, 300, "decode: weights + KV cache bytes set the time", { size: 11 });
          G.label(px, 320, "prefill: GEMMs compute-bound, attention ~ N²", { size: 11 });
          G.label(px, 344, "profile with nsys; count bytes per kernel", { size: 11, color: "ok" });
          G.caption("every kernel of the lane appears in one layer");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Naive attention materializes N × N tables (64 MiB per head at N = 4,096, fp32, 32× the real tensors) and is limited by them.</li>
<li>FlashAttention keeps (m, l, o) per query row and streams K/V tiles, rescaling by e^(m_old − m_new): the table lives only as one tile in registers. The footprint saving is unconditional; traffic savings scale as 2·BM/d before cache effects.</li>
<li>Causal and window masks skip tiles (528 of 1,024 and 252 of 1,024 at N = 2,048); masks go to −∞ before the max.</li>
<li>Heads go in the grid; GQA shares KV heads (4× smaller cache in Llama-3-8B); varlen packs sequences with cu_seqlens.</li>
<li>Decode is memory-bound on the KV cache: int8 halves the bytes, dequantized in registers; warps merge partial (m, l, o) states.</li>
<li>Speculative verification: accept with min(1, p/q), resample from the residual; the output is exactly the target distribution.</li>
<li>Triton writes the same program for a tile, with tl.dot on tensor cores; a layer is eight kernels from L4 to L6.</li></ul>
<p>That completes Lane B. Together with Lane A's P5.8 to P6, you now have every kernel the serving engine of the capstone relies on.</p>`,
    sim: {
      title: "Attention cost and tile skipping",
      intro: "Pick the sequence length, head dimension, tile height and mask. The map shows which (query tile, key tile) pairs the L6 kernel visits (green); gray pairs are skipped. The panels compare the memory the naive kernel needs for its N × N tables with the traffic of the tiled kernel, and give the KV-cache size for your head layout.",
      height: 330,
      controls: [
        { id: "N", label: "sequence length N", type: "select", value: 4096, options: [[1024, "1,024"], [2048, "2,048"], [4096, "4,096"], [8192, "8,192"], [16384, "16,384"], [32768, "32,768"]] },
        { id: "d", label: "head dim d", type: "select", value: 128, options: [[64, "64"], [128, "128"]] },
        { id: "BM", label: "query tile BM", type: "select", value: 64, options: [[32, "32"], [64, "64"], [128, "128"]] },
        { id: "mask", label: "mask", type: "select", value: "causal", options: [["full", "none"], ["causal", "causal"], ["window", "causal + window"]] },
        { id: "w", label: "window w (tokens)", min: 64, max: 8192, step: 64, value: 1024 },
        { id: "H", label: "query heads H", type: "select", value: 32, options: [[8, "8"], [16, "16"], [32, "32"], [64, "64"]] },
        { id: "kv", label: "KV heads (1 = MQA, = H is MHA)", type: "select", value: 8, options: [[1, "1"], [2, "2"], [4, "4"], [8, "8"], [16, "16"], [32, "32"], [64, "64"]] },
        { id: "by", label: "bytes per element", type: "select", value: 4, options: [[2, "2 (fp16 / bf16)"], [4, "4 (fp32)"]] },
      ],
      draw(G, v) {
        const N = v.N, d = v.d, BM = v.BM, BN = 32, by = v.by, H = v.H, kv = Math.min(v.kv, H);
        const causal = v.mask !== "full", win = v.mask === "window" ? v.w : 0;
        const vis = visited(N, BM, BN, causal, win), full = visited(N, BM, BN, false, 0);
        // map: 40 x 40 downsampled
        const R = 40, cs = 7.4, ox = 30, oy = 22;
        for (let a = 0; a < R; a++) for (let b = 0; b < R; b++) {
          const i = (a + 0.5) / R * N, j = (b + 0.5) / R * N;
          let ok = true; if (causal && j > i + N / R) ok = false; if (win > 0 && i - j >= win + N / R) ok = false;
          G.rect(ox + b * cs, oy + a * cs, cs - 0.8, cs - 0.8, { fill: ok ? "ok" : "line", rx: 1, opacity: ok ? 0.85 : 0.3 });
        }
        G.label(ox, oy + R * cs + 16, "rows: query tiles, columns: key tiles (downsampled to 40 × 40)", { size: 11 });
        G.label(ox + R * cs + 14, oy + 20, `${vis.Tq} × ${vis.Tk} tile pairs`, { size: 12, color: "ink" });
        G.label(ox + R * cs + 14, oy + 44, `${F.num(vis.pairs)} visited`, { size: 12, color: "ok" });
        G.label(ox + R * cs + 14, oy + 68, `${(100 * vis.pairs / full.pairs).toFixed(1)}% of all`, { size: 12, color: "ok" });
        const naiveFoot = N * N * by * H, naiveTraf = 4 * N * N * by * H;
        const kvTraf = vis.pairs * BN * d * 2 * by * H + 2 * N * d * by * H;
        const flops = 4 * d * (vis.pairs * BM * BN) * H, naiveFlops = 4 * d * N * N * H;
        const perTok = 2 * kv * d * 2 * 32;   // bf16 cache, 32 layers
        return [
          { title: "Memory", rows: [["naive S table, all heads", F.bytes(naiveFoot)], ["flash extra memory (m, l)", F.bytes(8 * N * H)], ["naive S/P traffic", F.bytes(naiveTraf)], ["flash Q/O + K/V tile loads", F.bytes(kvTraf)]],
            gauge: [[Math.min(1, kvTraf / naiveTraf), "ok"], [Math.max(0, 1 - kvTraf / naiveTraf), "line"]], gaugeText: "tiled traffic as a share of naive",
            chip: [kvTraf < naiveTraf, kvTraf < naiveTraf ? `${(naiveTraf / kvTraf).toFixed(1)}× less traffic (requests, before cache reuse)` : "no traffic saving at this tile size"] },
          { title: "Arithmetic (tile level)", rows: [["FLOPs, tiles visited", `${(flops / 1e9).toFixed(1)} GFLOP`], ["FLOPs, full N × N", `${(naiveFlops / 1e9).toFixed(1)} GFLOP`], ["skipped", `${(100 * (1 - flops / naiveFlops)).toFixed(1)}%`]] },
          { title: "KV cache (bf16, 32 layers)", rows: [["per token", F.bytes(perTok)], [`for ${F.num(N)} tokens`, F.bytes(perTok * N)], ["vs MHA", `${(H / kv).toFixed(0)}× smaller`]],
            html: `<p class="note">Model: per-layer tile counts follow the reference kernel's loop bounds (BN = 32; causal upper bound; window lower bound rounded to a tile). Traffic counts K/V tile requests per query tile; L2 hits make real DRAM traffic lower. Naive traffic assumes write S, read S, write P, read P. 32 layers as in Llama-3-8B.</p>` },
        ];
      },
      note: "Formulas: flash traffic = visited tile pairs · BN · d · 2 · bytes + 2·N·d·bytes (per head); naive = 4 · N² · bytes (per head).",
    },
    practice: {
      intro: `<p>Solve each problem on <a href="https://leetgpu.com/challenges">LeetGPU</a> first (T1, in the browser). We don't reproduce the statements; read them there. Then run the local reference and test (T2, needs an NVIDIA GPU and CUDA ≥ 12.4). Build once from the repository root:</p>
<p><code>cmake -S laneB-cuda -B build/laneB -DCMAKE_BUILD_TYPE=Release &amp;&amp; cmake --build build/laneB -j</code></p>
<p>Nearly every attention solution is the one shared core, <code>${LV}/solutions/_shared/attn_core.cuh</code>, plus a different <code>solve()</code>: read it first. Climb the sequence naive → tiled → FA-1 → FA-2 as in P5.8. <b>Exit check:</b> FA-2 forward is ≥ 5× faster than your naive attention at sequence length 4,096 (fp16; head dimension 64 or 128; state batch × heads), correct against PyTorch SDPA (<code>TODO(run-on: g6.xlarge)</code> until run). <b>Triton pass:</b> rewrite softmax, RMSNorm, GEMM and FA-2 in Triton and compare with your CUDA versions: <code>uv run --extra torch pytest course/P5-cuda-deep-dive/P5.9-triton/exercises -m torch</code> (runs in Triton's CPU interpreter without a GPU). <b>Cloud GPU (T2):</b> use <code>infra/aws/single-node</code> (g6.xlarge, an L4, is its default). It bills per hour while running plus disk while the disk exists (look up current prices on the EC2 pricing page), has an idle-stop alarm and an AWS Budget, no inbound ports (SSM only), and <code>make down</code> deletes everything: tear it down when you finish.</p>`,
      items: [
        P("6-softmax-attention", "Softmax Attention", "medium", "core · exit check", "Naive to tiled to FA-1 to FA-2 forward: keep (m, l, o) per query row and stream K/V tiles."),
        P("53-causal-self-attention", "Causal Self-Attention", "hard", "core", "Mask j &gt; i to −∞ before the max; skip K/V tiles in the future."),
        P("12-multi-head-attention", "Multi-Head Attention", "hard", "core", "Heads in <code>blockIdx.y</code>; strides instead of a transpose kernel."),
        P("80-grouped-query-attention", "Grouped Query Attention", "medium", "core", "Query head h reads KV head <code>h / (H / Hkv)</code>; try one block per head group."),
        P("102-varlen-causal-attention", "Variable-Length Causal Attention", "medium", "core", "Packed sequences with <code>cu_seqlens</code>; blocks past their sequence exit at once."),
        P("96-int8-kv-cache-attention", "INT8 KV-Cache Attention", "medium", "core", "Decode attention with in-register dequantization and a warp merge of online states."),
        P("87-speculative-decoding-verification", "Speculative Decoding Verification", "medium", "core", "Accept with min(1, p/q), resample from the residual, bonus token; randoms passed in."),
        P("59-sliding-window-attention", "Sliding Window Self-Attention", "hard", "practice", "Skip tiles on both ends of the band: cost O(N · w)."),
        P("55-alibi-attention", "Attention with Linear Biases", "medium", "practice", "Add the per-head ALiBi bias inside the score tile."),
        P("112-attention-with-sinks", "Attention with Sinks", "medium", "practice", "One extra softmax term with no value vector."),
        P("26-multi-head-cross-attention", "Multi-Head Cross-Attention", "hard", "practice", "Lq ≠ Lk: the same core with different lengths."),
        P("114-mla-decode", "Multi-Head Latent Attention Decode", "hard", "stretch", "Outline only: decode attention over a latent KV representation.", null),
        P("119-block-sparse-kv-attention", "Block-Sparse KV Selection Attention", "hard", "stretch", "Outline only: attend to selected KV blocks.", null),
        P("111-attention-backward", "Softmax Attention Backward", "medium", "stretch", "Outline only: the FlashAttention backward pass.", null),
        P("56-linear-self-attention", "Linear Self-Attention", "hard", "stretch", "Outline only: attention without the softmax table.", null),
        P("92-decaying-causal-attention", "Decaying Causal Attention", "medium", "stretch", "Outline only: causal attention with decay.", null),
        P("74-gpt2-block", "GPT-2 Transformer Block", "hard", "hard · assemble", "Outline only: LayerNorm, QKV GEMM, causal MHA, projection plus residual, MLP, all from your kernels.", null),
        P("93-llama-block", "Llama Transformer Block", "hard", "hard · assemble", "Outline only: the eight-kernel layer of the last step.", null),
        P("116-dit-block", "Diffusion Transformer Block", "hard", "hard · assemble", "Outline only: a DiT block from your kernels.", null),
      ],
      labs: [
        { label: "Level syllabus, objectives and exit check", path: `${LV}/syllabus.md` },
        { label: "Problem map with links and priorities", path: `${LV}/leetgpu-map.md` },
        { label: "The shared FA-2 style core used by the attention solutions", path: `${LV}/solutions/_shared/attn_core.cuh` },
        { label: "Triton kernels: softmax, rmsnorm, matmul, flash_attention", path: "platform/kernels/triton_kernels/" },
        { label: "Triton exercises and CUDA-vs-Triton bench (P5.9)", path: "course/P5-cuda-deep-dive/P5.9-triton/exercises/README.md" },
        { label: "FlashAttention animation (Lane A P5.8)", path: "animations/p5-flashattention.html" },
        { label: "Online softmax animation (Lane A P5.5)", path: "animations/p5-online-softmax.html" },
        { label: "AWS T2 box with sm_80+ (L4 or A10G): the g6.xlarge guide", path: "infra/aws/single-node" },
      ],
    },
  });
})();
