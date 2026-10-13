/* P5.8 — FlashAttention. Never write the N×N score matrix: tile over K/V with online softmax (P5.5), then fix the loop order (FA-2). */
(function () {
  const F = S2S.fmt;
  const MiB = 1024 * 1024;
  // Example bandwidth and memory from course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/gpu_specs.yaml (UNVERIFIED there).
  const GPUS = { t4: { name: "T4", bw: 320e9, gb: 16 }, l4: { name: "L4", bw: 300e9, gb: 24 }, a100: { name: "A100-80GB", bw: 2039e9, gb: 80 } };
  // Per-head byte counts for fp16 Q/K/V/O (2 B), fp32 scores (4 B), head dim d. Models, see the notes in the lesson.
  const naiveBytes = (N, d) => 4 * N * N * 4 + 4 * N * d * 2;                       // S written, read, P written, read + Q,K,V,O
  const compulsory = (N, d) => 4 * N * d * 2;                                       // Q,K,V read once, O written once
  const fa2NoCache = (N, d, Br) => (2 + (2 * N) / Br) * N * d * 2;                  // Q read, O written, K and V re-read N/Br times
  const fa1Extra = (N, d, Bc) => (N / Bc) * N * (8 * d + 16);                       // O (fp32) and (m, l) read-modify-write per K/V tile
  const logBar = (v, lo, hi, w) => (w * (Math.log10(v) - lo)) / (hi - lo);

  S2S.lesson({
    id: "p5-8", n: "P5.8", title: "FlashAttention",
    subtitle: "CUDA deep dive · first principles · T2, CUDA-core kernels on sm_75+",
    kicker: "Lesson · ≈ 55 min",
    headline: "The matrix that never exists",
    intro: `<p>Attention compares every token with every other token, so it builds an N × N table of scores. At N = 4,096 that table is 1 GiB per 16 heads, and the naive implementation writes it to memory and reads it back four times. <b>FlashAttention</b> computes exactly the same result without ever storing the table. This lesson derives it from two ideas you already have: tiling (P5.6) and the online softmax (P5.5). Then it follows the two versions in the repo (FA-1 and FA-2), counts the memory traffic of each, and ends with the decode-time kernel that serving engines run every step, over a paged KV cache.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["q", "Q (query)"], ["k", "K (key)"], ["v", "V (value) / on chip"], ["ok", "output O"], ["hot", "HBM traffic / skipped"]],
    prev: "p5-7", next: "p5-9",
    steps: [
      { rail: "attention", title: "Attention is three small steps",
        body: `<p>One query vector <b>q</b> (what this token is looking for) is compared with a <b>key</b> k<sub>j</sub> for every earlier token (what each token offers). Each comparison is a dot product, a <b>score</b>. Softmax turns the scores into weights. The output is the weighted average of the <b>values</b> v<sub>j</sub>:</p>
<div class="eq">s_j = q · k_j / √d        (scores)
p   = softmax(s)         (weights, sum to 1)
o   = Σ_j p_j · v_j      (output)

d = 2, q = (1, 1), three keys:
  k = (2,0) (0,2) (1,-1)  → q·k = 2, 2, 0
  s = 2/√2, 2/√2, 0 = 1.414, 1.414, 0
  e^s = 4.113, 4.113, 1   sum = 9.226
  p = 0.446, 0.446, 0.108
  v = (1,0) (0,1) (1,1)
  o = 0.446·(1,0) + 0.446·(0,1) + 0.108·(1,1)
    = (0.554, 0.554)</div>
<p>For N tokens the same thing happens for all N queries at once. Stack the queries as a matrix Q, the keys as K, the values as V (each N × d). Then <code>S = Q·Kᵀ/√d</code> is an N × N matrix, <code>P = softmax(S)</code> row by row, and <code>O = P·V</code>. Two matrix multiplications with a softmax in between: that is attention. In a <b>causal</b> (decoder) model, token i may only look at tokens j ≤ i, so scores above the diagonal are set to -inf before the softmax.</p>`,
        scene(G) {
          G.text(24, 32, "one query, three keys (d = 2)", { size: 13 });
          G.box(24, 56, 120, 34, "q = (1, 1)", { fill: "q", size: 13 });
          const heads = ["key k_j", "score s_j", "weight p_j", "value v_j"];
          [190, 300, 410, 520].forEach((x, i) => G.label(x, 78, heads[i], { size: 12 }));
          const K = ["(2, 0)", "(0, 2)", "(1, -1)"], S = ["1.414", "1.414", "0"], Pw = [0.446, 0.446, 0.108], V = ["(1, 0)", "(0, 1)", "(1, 1)"];
          const all = [];
          for (let j = 0; j < 3; j++) {
            const y = 96 + j * 52;
            G.label(24, y + 24, "token " + j, { size: 12 });
            all.push(G.box(180, y, 86, 38, K[j], { fill: "k", size: 13 }));
            all.push(G.box(290, y, 86, 38, S[j], { stroke: "ink", size: 13 }));
            all.push(G.rect(400, y, 56 + 140 * Pw[j], 38, { fill: "ok", rx: 6, opacity: 0.85 }));
            G.text(406, y + 24, Pw[j].toFixed(3), { size: 13, color: "bg" });
            all.push(G.box(520, y, 86, 38, V[j], { fill: "v", size: 13 }));
            G.arrow(268, y + 19, 288, y + 19, { color: "muted" });
          }
          G.rect(24, 270, 592, 124, { stroke: "line", rx: 8 });
          G.label(40, 296, "softmax(1.414, 1.414, 0) = (0.446, 0.446, 0.108)", { size: 13, color: "ink" });
          G.label(40, 320, "o = Σ p_j · v_j = (0.554, 0.554)", { size: 13, color: "ok" });
          G.label(40, 346, "N queries at once: S = Q·Kᵀ/√d is N × N,  O = softmax(S)·V", { size: 12, color: "ink" });
          G.label(40, 368, "causal: scores with j > i are set to -inf before the softmax", { size: 12 });
          G.from(all, { opacity: 0, stagger: 0.04, duration: 0.25 });
          G.caption("score, softmax, weighted average of values");
        } },

      { rail: "N × N", title: "The score matrix grows with N squared",
        body: `<p>The inputs Q, K, V and the output O each have N × d numbers (d = 64 per head in the course kernels). The score matrix S has N × N. As sequences get longer, N² outruns N × d quickly. Take 16 (batch × head) pairs, N = 4,096, S in fp32 (4 bytes):</p>
<div class="eq">S values  = 16 × 4096 × 4096 = 268,435,456
S bytes   = 268.4M × 4 B  = 1 GiB

Q, K, V, O in fp16, d = 64:
  each 16 × 4096 × 64 × 2 B = 8 MiB
  all four                  = 32 MiB

N =  1024: S =  64 MiB   (Q,K,V,O =  8 MiB)
N =  2048: S = 256 MiB   (Q,K,V,O = 16 MiB)
N =  4096: S =   1 GiB   (Q,K,V,O = 32 MiB)</div>
<p>Doubling N quadruples S but only doubles the inputs. At N = 4,096, S is <b>32 times larger than everything the computation actually needs to read and write</b>. And it is not a small scratch buffer that fits in cache: 1 GiB is far beyond any GPU's on-chip memory, so it lives in DRAM (HBM or GDDR). The next step shows what that costs.</p>`,
        scene(G) {
          const N = [1024, 2048, 4096], side = [44, 88, 176], sMiB = ["64 MiB", "256 MiB", "1 GiB"], io = ["8 MiB", "16 MiB", "32 MiB"];
          G.text(24, 32, "S is N × N, 16 heads, fp32", { size: 13 });
          const sq = [];
          let x = 40;
          N.forEach((n, i) => {
            const bottom = 300;
            sq.push(G.rect(x, bottom - side[i], side[i], side[i], { fill: "hot", rx: 4, opacity: 0.8 }));
            G.text(x + side[i] / 2, bottom - side[i] / 2 + 5, sMiB[i], { anchor: "middle", size: i === 0 ? 11 : 15, color: "bg" });
            G.rect(x, bottom + 14, 8 + i * 3, 8, { fill: "ok", rx: 2 });
            G.label(x, bottom + 42, "N = " + n, { size: 13, color: "ink" });
            G.label(x, bottom + 62, "Q,K,V,O: " + io[i], { size: 11 });
            x += side[i] + 64;
          });
          G.label(24, 396, "green: inputs and output (not to scale); red area: the N×N table", { size: 12 });
          G.from(sq, { opacity: 0, scale: 0.7, transformOrigin: "bottom left", stagger: 0.15, duration: 0.4 });
          G.caption("double N: S grows 4×, the inputs 2×");
        } },

      { rail: "naive traffic", title: "Naive attention is limited by bytes, not FLOPs",
        body: `<p>The straightforward implementation (<code>attn_naive</code> in <code>platform/kernels/include/d4/attention.cuh</code>) launches three kernels and every one of them goes through DRAM:</p>
<ol><li><b>scores:</b> compute S = QKᵀ/√d and <b>write</b> it (1 GiB).</li>
<li><b>softmax:</b> <b>read</b> S, <b>write</b> P (2 GiB).</li>
<li><b>P·V:</b> <b>read</b> P (1 GiB), write O.</li></ol>
<div class="eq">DRAM traffic = 4 passes × 1 GiB + Q,K,V,O
              = 4 GiB + 32 MiB = 4,128 MiB

at 300 GB/s (L4 example value): 4.33e9 B
  → about 14 ms just to move the bytes

FLOPs = 2 matmuls × 2·N²·d × 16 heads
      = 4 × 16.8M × 64 × 16 = 68.7 GFLOP
FLOP per byte = 68.7e9 / 4.33e9 ≈ 16</div>
<p>Sixteen FLOPs per byte is under the ridge point of fp32 CUDA cores (≈ 25 on the T4 example from P5.6) and far under a tensor core's (P5.7). In general the intensity is <code>d / 4</code> whatever N is: the work grows as N² but so does the traffic. The matrix multiplications themselves are fine. The memory round trips of the table between them are the cost, and the softmax in the middle is what forces them: a row cannot be normalised until all of it is known (P5.5). If we could remove that dependency, the table would not need to exist.</p>`,
        check: { q: "Naive attention's FLOPs per DRAM byte come out at about d / 4 for any sequence length N. What does that tell you?",
          options: ["It is compute-bound, so use tensor cores", "Making N larger will not make it more compute-efficient: its speed is set by memory traffic", "Traffic is independent of N"], answer: 1,
          why: "Both the work (two N × N × d matmuls) and the table traffic (four N × N passes) grow as N², so their ratio is constant (about 16 FLOP per byte at d = 64), below the ridge of any modern GPU. More compute would not help; fewer bytes would." },
        scene(G) {
          const x0 = 150, w = 400, tot = 4128;
          G.text(24, 32, "DRAM traffic of naive attention (N=4096, 16 heads)", { size: 13 });
          const rows = [["scores: write S", 1024, "v"], ["softmax: read S", 1024, "k"], ["softmax: write P", 1024, "v"], ["P·V: read P", 1024, "k"], ["Q, K, V, O", 32, "ok"]];
          const bars = rows.map(([name, mib, col], i) => {
            const y = 56 + i * 48;
            G.label(24, y + 20, name, { size: 12, color: "ink" });
            const r = G.rect(x0, y, Math.max(4, (w * mib) / 1100), 28, { fill: col, rx: 3 });
            G.text(x0 + (w * mib) / 1100 + 8, y + 20, mib >= 1024 ? "1 GiB" : mib + " MiB", { size: 13 });
            return r;
          });
          G.rect(24, 310, 592, 80, { stroke: "line", rx: 8 });
          G.label(40, 336, "total ≈ 4,128 MiB: 128× the 32 MiB the computation needs", { size: 13, color: "hot" });
          G.label(40, 360, "FLOP per byte ≈ 16 = d / 4, for every N: memory-bound", { size: 13, color: "ink" });
          G.label(40, 380, "teal = read · amber = write (fp32 table)", { size: 11 });
          G.from(bars, { attr: { width: 0 }, duration: 0.5, stagger: 0.12 });
          G.caption("the table crosses DRAM four times; the useful data is 32 MiB");
        } },

      { rail: "online tiles", title: "Online softmax lets you process keys in tiles",
        body: `<p>From P5.5: a softmax row can be built left to right by keeping a running max <code>m</code> and a running sum <code>l</code>, rescaling <code>l</code> by <code>e^(m_old - m_new)</code> whenever the max grows. The output <code>o = Σ p_j · v_j</code> is a sum of the same kind, so it can use the <i>same</i> rescale. Keep an <b>unnormalised</b> output, and divide by <code>l</code> only at the very end. When a new tile of keys with scores <code>s_j</code> arrives:</p>
<div class="eq">m_new = max(m, max_j s_j)
f     = e^(m - m_new)
l     = l·f + Σ_j e^(s_j - m_new)
o     = o·f + Σ_j e^(s_j - m_new) · v_j
m     = m_new
(at the end)  out = o / l</div>
<p>Redo the example of step 1 with the keys in two tiles, key 2 first, then keys 0 and 1 (so the max changes):</p>
<div class="eq">start      m = -inf  l = 0  o = (0, 0)
tile {2}   s = 0
           m = 0  l = 1  o = 1·(1,1) = (1, 1)
tile {0,1} s = 1.414, 1.414  → m_new = 1.414
           f = e^(0 - 1.414) = 0.243
           l = 1·0.243 + 2 = 2.243
           o = (1,1)·0.243 + (1,0) + (0,1)
             = (1.243, 1.243)
out = o / l = (0.554, 0.554)   ✓ same as step 1</div>
<p>The scores of an earlier tile are never needed again: only the three small things <code>(m, l, o)</code> carry over. That is the whole trick. Memory per query row is d + 2 numbers, however long the sequence.</p>`,
        check: { q: "A new tile raises the running max from m to m_new. Which quantities must be multiplied by e^(m - m_new)?",
          options: ["Only l", "Both l and the unnormalised output o", "Only o"], answer: 1,
          why: "Both l = Σ e^(s - m) and o = Σ e^(s - m) · v were accumulated relative to the old max. When the reference max changes, every term in both sums shrinks by the same factor e^(m - m_new). Forgetting either one is the classic bug (quiz and common mistakes)." },
        scene(G) {
          G.text(24, 32, "state (m, l, o) after each tile of keys", { size: 13 });
          const stages = [
            ["start", "m = -inf", "l = 0", "o = (0, 0)", "line", ""],
            ["tile {key 2}  s = 0", "m = 0", "l = 1", "o = (1, 1)", "k", ""],
            ["tile {keys 0, 1}  s = 1.414, 1.414", "m = 1.414", "l = 2.243", "o = (1.243, 1.243)", "v", "rescale old l, o by e^(0 - 1.414) = 0.243"],
          ];
          const gs = [];
          stages.forEach(([title, m, l, o, col, note], i) => {
            const y = 56 + i * 98;
            G.label(24, y + 4, title, { size: 12, color: "ink" });
            const a = G.box(24, y + 14, 110, 36, m, { fill: "q", size: 13 });
            const b = G.box(144, y + 14, 100, 36, l, { fill: col === "line" ? "w" : "ok", size: 13 });
            const c = G.box(254, y + 14, 170, 36, o, { fill: col === "line" ? "w" : "k", size: 13 });
            if (note) G.label(24, y + 70, note, { size: 12, color: "hot" });
            gs.push(a, b, c);
          });
          G.box(450, 266, 166, 56, "", { stroke: "ok", color: "ok", size: 13 });
          G.text(533, 290, "out = o / l", { anchor: "middle", size: 14, color: "ok" });
          G.text(533, 310, "= (0.554, 0.554)", { anchor: "middle", size: 13, color: "ok" });
          G.label(24, 366, "earlier scores are never revisited: only (m, l, o) carry over", { size: 13, color: "ink" });
          G.from(gs, { opacity: 0, y: 8, stagger: 0.07, duration: 0.25 });
          G.caption("the same answer as step 1, built one tile at a time");
        } },

      { rail: "the algorithm", title: "FlashAttention: a Q tile stays, K/V tiles stream past",
        body: `<p>Put it together. Give a block of threads a <b>tile of query rows</b> (the course kernel: 64 rows, one thread each). Then stream through the keys and values in tiles (32 keys at a time):</p>
<div class="eq">load Q tile (64 × d)         once
(m, l, o) = (-inf, 0, 0)
for each K/V tile (32 keys):
    load K tile, V tile    → shared memory
    S = Q·Kᵀ/√d           (64 × 32, on chip)
    update (m, l, o) as in step 4
write out = o / l          once</div>
<p>The 64 × 32 tile of scores lives in registers: 2,048 fp32 values, 8 KiB, and is overwritten by the next tile. The full N × N table never exists. The matrix multiplications, the softmax and the weighted sum are all <b>fused</b> into one kernel (the same idea as fusing add and RMSNorm in P5.5), so scores never touch DRAM. In <code>attn_fa2</code> each thread owns one query row: <code>q[64]</code>, <code>o[64]</code> and the tile's scores <code>s[32]</code> are all in registers, and the block stages each K/V tile in shared memory once (2 × 32 × 65 floats = 16,640 B, the +1 column avoids bank conflicts). Every thread then reads the same K and V row at the same time, which is a shared-memory broadcast.</p>
<p>This produces the <b>exact</b> same output as naive attention, up to the order of floating-point additions. It is not an approximation. The result for each row depends only on that row's scores, so the work splits cleanly by query tile: each block owns its rows from start to finish and never talks to another block.</p>`,
        scene(G) {
          G.text(24, 32, "S = Q·Kᵀ as tiles: row block = one thread block", { size: 13 });
          const x0 = 70, y0 = 56, cw = 62, ch = 44;
          G.label(x0, y0 - 8, "K/V tiles (32 keys each) →", { size: 11 });
          const cells = [];
          for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) {
            const mine = r === 1;
            const f = !mine ? "line" : c < 4 ? "ok" : c === 4 ? "v" : "w";
            cells.push(G.rect(x0 + c * cw, y0 + r * ch, cw - 4, ch - 4, { fill: f, rx: 4, opacity: mine ? 1 : 0.3 }));
          }
          for (let r = 0; r < 4; r++) G.label(24, y0 + r * ch + 26, "Q " + r, { size: 11, color: r === 1 ? "q" : "muted" });
          G.label(x0, y0 + 4 * ch + 12, "green: done · amber: this tile · grey: still to stream", { size: 11 });
          G.rect(24, 288, 592, 104, { stroke: "line", rx: 8 });
          G.label(40, 312, "one block, 64 query rows: Q tile, m, l, o stay in registers", { size: 13, color: "ink" });
          G.label(40, 334, "each step: K, V tile → shared memory; S tile 64×32 (8 KiB) on chip", { size: 12, color: "v" });
          G.label(40, 356, "the full N × N table is never stored: exact result, no approximation", { size: 12, color: "ok" });
          G.label(40, 378, "rows are independent: a block owns its rows from start to finish", { size: 12 });
          G.from(cells, { opacity: 0, stagger: 0.02, duration: 0.2 });
          G.pulse(cells[12], { repeat: 6 });
          G.caption("a tile of scores exists only briefly, on chip");
        } },

      { rail: "traffic", title: "How many bytes does it move?",
        body: `<p>Count DRAM traffic per head for the FlashAttention loop. Q is read once and O written once, but <b>K and V are read again by every query tile</b>: with Br query rows per tile there are N / Br tiles, so each of K and V is streamed N / Br times.</p>
<div class="eq">per head, fp16, d = 64, N = 4096, Br = 64:
 N·d·2 B = 524,288 B  (one matrix)
 tiles N / Br = 64

 no cache:  Q + O + 64 × (K + V)
          = (2 + 128) × 524,288 B = 65 MiB
 16 heads = 1,040 MiB   (naive: 4,128 MiB)

 perfect reuse (compulsory):
          = Q + K + V + O = 4 × 0.5 MiB
 16 heads = 32 MiB</div>
<p>So the bytes depend on one number you control, <b>Br</b>: double the tile and the K/V re-reads halve (a 128-row tile: 528 MiB). Bigger tiles need more registers or shared memory, the same trade as P5.6 and P5.7. And the reuse is real: one head's K and V together are 1 MiB, and the 64 blocks working on the same head read them at about the same time, so most of those re-reads are served by the <b>L2 cache</b> (48 MB on the L4 example) instead of DRAM. The truth lies between 32 MiB and 1,040 MiB, and the way to find it is to measure: <code>ncu --metrics dram__bytes.sum</code> on naive vs FA-2 (the course's AWS guide does exactly this).</p>
<p>Compare with naive on the same terms. Its table is 64 MiB per head and every pass is a stream over that whole table with no reuse at all, so none of its 4 GiB can hide in cache. Even the pessimistic FlashAttention figure is about 4× lower, and the likely one is much lower.</p>`,
        scene(G) {
          const lo = 1, hi = 4, w = 380, x0 = 200;
          G.text(24, 32, "DRAM traffic, 16 heads, N = 4096 (log scale)", { size: 13 });
          const rows = [["naive", 4128, "hot"], ["FA-2, Br 64, no cache", 1040, "v"], ["FA-2, Br 128, no cache", 528, "k"], ["FA-2, perfect reuse", 32, "ok"]];
          const bars = rows.map(([name, v, col], i) => {
            const y = 60 + i * 58;
            G.label(24, y + 18, name, { size: 12, color: "ink" });
            const r = G.rect(x0, y, logBar(v, lo, hi, w), 28, { fill: col, rx: 3 });
            G.text(x0 + logBar(v, lo, hi, w) + 8, y + 20, F.num(v) + " MiB", { size: 13 });
            return r;
          });
          [10, 100, 1000].forEach((v) => { G.line(x0 + logBar(v, lo, hi, w), 52, x0 + logBar(v, lo, hi, w), 288, { color: "line" }); G.label(x0 + logBar(v, lo, hi, w), 304, String(v), { anchor: "middle", size: 11 }); });
          G.label(24, 340, "FA-2 bytes = (2 + 2N/Br) · N · d · 2 per head: K, V re-read N/Br times", { size: 12, color: "ink" });
          G.label(24, 362, "L2 serves most re-reads (one head's K, V = 1 MiB): measure with ncu", { size: 12 });
          G.label(24, 384, "dram__bytes.sum, naive vs FA-2", { size: 12 });
          G.from(bars, { attr: { width: 0 }, duration: 0.5, stagger: 0.1 });
          G.caption("tile height Br sets the re-reads; the cache decides the rest");
        } },

      { rail: "FA-1 vs FA-2", title: "FA-2 swaps the loops",
        body: `<p>The first FlashAttention (FA-1) had the loops the other way round: the <b>outer</b> loop over K/V tiles, the inner loop over query tiles. That reads each K/V tile exactly once, which sounds good, but now each query row's <code>(m, l, o)</code> is touched once per K/V tile, so it must be kept in memory between visits. The repo's <code>attn_flash1</code> makes this explicit, with one kernel launch per K/V tile: every launch reads and writes the running O (fp32), m and l in global memory.</p>
<div class="eq">FA-1 O round trip, per head, N = 4096, d = 64:
  per K/V tile: 4096 rows × (64 × 4 B) × 2
              = 2 MiB   (read + write)
  K/V tiles: 4096 / 32 = 128
  = 256 MiB per head = 4 GiB for 16 heads</div>
<p>On paper that equals the naive table traffic, with these small 32-key tiles. In practice the 16 MiB accumulator can stay in L2, so the DRAM counters read lower; the point stands that FA-1's structure does a read-modify-write of O that does not need to exist. <b>FA-2</b> swaps the loops: the outer loop is over <b>query tiles, one per thread block</b>, and the inner loop streams K/V. Now (m, l, o) stay in <b>registers</b> for the whole sweep and O is written once.</p>
<table><tr><th></th><th>FA-1</th><th>FA-2</th></tr>
<tr><td>outer loop</td><td>K/V tiles</td><td>Q tiles (one per block)</td></tr>
<tr><td>(m, l, o) between tiles</td><td>global memory</td><td>registers</td></tr>
<tr><td>rescale</td><td>per tile, in memory</td><td>per tile, in registers</td></tr>
<tr><td>parallelism</td><td>batch × heads</td><td>also over query tiles</td></tr>
<tr><td>causal mask</td><td>mask</td><td>mask + skip tiles</td></tr></table>
<p>The extra parallelism matters at small batch: with 1 sequence of 4,096 tokens and 32 heads, FA-2 launches 64 query tiles × 32 heads = 2,048 blocks instead of 32.</p>`,
        check: { q: "What is the main structural change from FA-1 to FA-2?",
          options: ["FA-2 uses tensor cores", "The query tile becomes the outer loop, so (m, l, o) stay in registers and O is written once", "FA-2 stores a smaller S matrix"], answer: 1,
          why: "Making Q tiles the outer loop gives each thread block sole ownership of its output rows for the whole K/V sweep. Their running state fits in registers, so there is no per-tile read-modify-write of O in memory. (The course's FA-2 kernel is CUDA-core; tensor cores are the stretch exercise.)" },
        scene(G) {
          G.text(24, 32, "FA-1: outer = K/V tiles", { size: 13, color: "hot" });
          G.box(24, 46, 120, 34, "HBM: m, l, O", { stroke: "hot", color: "hot", size: 12 });
          const f1 = [0, 1, 2].map((i) => G.box(190 + i * 140, 46, 124, 34, "K/V tile " + i, { fill: "k", size: 12 }));
          [0, 1, 2].forEach((i) => { G.arrow(144, 56, 188 + i * 140, 56, { color: "hot", w: 1.2 }); });
          G.label(190, 102, "every tile: read O, rescale, accumulate, write O back", { size: 12, color: "hot" });
          G.text(24, 160, "FA-2: outer = Q tiles", { size: 13, color: "ok" });
          G.box(24, 174, 120, 34, "regs: m, l, o", { fill: "ok", size: 12 });
          const f2 = [0, 1, 2].map((i) => G.box(190 + i * 140, 174, 124, 34, "K/V tile " + i, { fill: "k", size: 12 }));
          G.arrow(144, 191, 188, 191, { color: "ok", w: 1.5 });
          G.label(190, 230, "(m, l, o) never leave the chip until the end", { size: 12, color: "ok" });
          G.rect(24, 262, 592, 128, { stroke: "line", rx: 8 });
          G.label(40, 288, "FA-1 O round trip: 2 MiB per K/V tile per head × 128 tiles", { size: 12, color: "ink" });
          G.label(40, 310, "= 256 MiB per head, 4 GiB for 16 heads (on paper; L2 may absorb)", { size: 12, color: "hot" });
          G.label(40, 336, "FA-2: O written once; 64 query tiles per head fill the GPU", { size: 12, color: "ok" });
          G.label(40, 358, "1 sequence × 32 heads: 2,048 blocks instead of 32", { size: 12, color: "ink" });
          G.from(f1.concat(f2), { opacity: 0, x: 20, stagger: 0.08, duration: 0.25 });
          G.caption("the state that was in memory between tiles moves into registers");
        } },

      { rail: "causal", title: "Causal masking: skip tiles that are entirely masked",
        body: `<p>In a decoder, query i sees only keys j ≤ i. Scores with j &gt; i are -inf, and e<sup>-inf</sup> = 0 contributes nothing, so a naive kernel computes them and throws them away. FlashAttention works in tiles, so it can do better: a K/V tile that lies entirely to the right of the query tile's last row is <b>fully masked</b> and is skipped without being loaded or computed. Only tiles that cross the diagonal need the per-element mask.</p>
<div class="eq">course kernel: Br = 64 rows, Bc = 32 keys, N = 4096
Q tile q covers rows 64q .. 64q+63
keys needed: 0 .. 64q+63
K/V tiles = (64q + 64) / 32 = 2(q + 1)

total over q = 0..63:
  Σ 2(q+1) = 2 × (64 × 65 / 2) = 4,160
full (no skip): 64 × 128 = 8,192
4,160 / 8,192 = 51%   (about 2× less work)</div>
<p>In code: <code>kv_end = causal ? min(N, i0 + FBR) : N</code>, the loop bound that ends the K/V sweep at the end of the query tile. Two details cause the classic bugs. The mask is <code>j &gt; i</code> (the diagonal itself must stay visible, a query attends to itself). And a row whose scores so far are all -inf has <code>m = -inf</code>, so <code>m - m_new</code> would be <code>-inf - (-inf)</code> = NaN. The kernel guards that case (<code>if (m_new != -INFINITY)</code>), the same guard as <code>md_merge</code> in P5.5.</p>`,
        check: { q: "With N = 4096, a 64-row query tile and 32-key K/V tiles, roughly how much K/V work does causal tile skipping save?",
          options: ["Nothing: every tile crosses the diagonal", "About half (4,160 of 8,192 tile visits)", "About 90%"], answer: 1,
          why: "Query tile q only needs keys 0 to 64q + 63, which is 2(q + 1) tiles of 32. Summed over the 64 query tiles that is 4,160 against 8,192 for the full square: the lower triangle plus the diagonal, a bit over half." },
        scene(G) {
          const x0 = 90, y0 = 60, c = 46, n = 8;
          G.text(24, 32, "causal tiles (8 × 8 sketch)", { size: 13 });
          G.label(x0, y0 - 8, "K/V tiles →", { size: 11 });
          const cells = [];
          for (let r = 0; r < n; r++) for (let k = 0; k < n; k++) {
            const done = k < r, diag = k === r;
            cells.push(G.rect(x0 + k * c, y0 + r * c * 0.5, c - 3, c * 0.5 - 3, { fill: done ? "ok" : diag ? "v" : "line", rx: 3, opacity: done || diag ? 1 : 0.35 }));
          }
          for (let r = 0; r < n; r++) G.label(24, y0 + r * c * 0.5 + 18, "Q " + r, { size: 10 });
          G.label(x0 + 5 * c, y0 + 3 * c * 0.5 + 4, "skipped:", { size: 12, color: "hot" });
          G.label(x0 + 5 * c, y0 + 3 * c * 0.5 + 20, "fully masked", { size: 12, color: "hot" });
          G.label(x0, y0 + n * c * 0.5 + 16, "green: full tile · amber: diagonal tile (per-element mask j > i)", { size: 11 });
          G.rect(24, 292, 592, 100, { stroke: "line", rx: 8 });
          G.label(40, 316, "real kernel (64 rows × 32 keys, N = 4096): 4,160 of 8,192 tile visits = 51%", { size: 12, color: "ink" });
          G.label(40, 338, "loop bound: kv_end = min(N, i0 + 64) for the query tile starting at row i0", { size: 12, color: "ink" });
          G.label(40, 360, "mask is j > i (a token sees itself); guard m = -inf to avoid NaN", { size: 12, color: "hot" });
          G.from(cells, { opacity: 0, stagger: 0.008, duration: 0.2 });
          G.caption("about half the K/V tiles are never loaded");
        } },

      { rail: "decode", title: "Decode: one query, a long history, a paged cache",
        body: `<p>Everything so far is <b>prefill</b>: many queries at once. During <b>decode</b> (P1.2) each sequence adds one token per step, so there is <b>one</b> query row against a long KV history. There is no query tile to amortise the K/V reads over, so this kernel is purely about reading the KV cache at full bandwidth.</p>
<div class="eq">Llama-3-8B, per token, all 32 layers:
  32 × 8 KV heads × 128 × 2 (K,V) × 2 B
  = 131,072 B = 128 KiB
sequence of 4,096 tokens: 512 MiB of KV

one decode step must read all of it:
  512 MiB = 537e6 B ÷ 300e9 B/s ≈ 1.8 ms
  (L4 example bandwidth, lower bound)

per KV element (2 B): 1 FMA per K, 1 per V
  = 1 FLOP/B; with 4 query heads sharing
  each KV head (GQA) = 4 FLOP/B</div>
<p>That is far under every ridge point, so decode attention is memory-bound whatever the arithmetic. The cache is also <b>paged</b> (P2.3): memory is split into fixed-size blocks (16 tokens here), and a per-sequence <b>block table</b> maps logical block i to a physical block anywhere in the pool. The kernel <code>paged_decode_attention</code> reads <code>block_table[b][i]</code> before each block of tokens, one extra indirection per 16 tokens, cheap as long as a whole block is read contiguously.</p>
<p>The work split: one thread block per (sequence, query head), 4 warps. GQA: query head h reads KV head <code>h / (H / Hkv)</code>. Warps walk the sequence's tokens in a strided pattern, each keeping its own online-softmax state <code>(m, l, o)</code>; lanes split the 64 dimensions and a warp shuffle sums the dot product. At the end the four states are merged with the rule from P5.5. This is exercise 4, and what the engine of P6.3 calls every step.</p>`,
        check: { q: "Why is decode attention memory-bound even though FlashAttention-style fusion is used?",
          options: ["The softmax is slow", "Each KV byte read supports only about 1 to 4 FLOPs, and every step must read the whole KV cache", "Block tables are slow to read"], answer: 1,
          why: "With a single query, each K or V element is used for one multiply-add, so the arithmetic per byte is tiny (GQA multiplies it by the group size, still single digits). Time is KV bytes divided by bandwidth; fusion removes extra traffic but cannot reduce the cache that must be read." },
        scene(G) {
          G.text(24, 32, "one sequence, 40 tokens, block size 16", { size: 13 });
          const cw = 14, x0 = 30, wc = ["k", "q", "ok", "v"];
          const toks = [];
          for (let t = 0; t < 40; t++) toks.push(G.rect(x0 + t * (cw + 0.5), 50, cw - 2, 26, { fill: wc[t % 4], rx: 2 }));
          G.label(x0, 96, "token t → warp t % 4: each warp keeps its own (m, l, o)", { size: 12, color: "ink" });
          G.label(x0, 118, "logical block 0", { size: 11 }); G.label(x0 + 16 * (cw + 0.5), 118, "block 1", { size: 11 }); G.label(x0 + 32 * (cw + 0.5), 118, "block 2", { size: 11 });
          G.text(24, 170, "block table  [7, 2, 5]", { size: 13, color: "v" });
          const bt = [7, 2, 5].map((p, i) => G.box(24 + i * 56, 182, 48, 30, String(p), { fill: "v", size: 14 }));
          const px = 250;
          G.text(px, 170, "physical pool (shared by all sequences)", { size: 13 });
          const pool = [];
          for (let p = 0; p < 8; p++) pool.push(G.box(px + (p % 4) * 90, 182 + Math.floor(p / 4) * 44, 82, 36, "phys " + p, { fill: [7, 2, 5].includes(p) ? "k" : "line", color: [7, 2, 5].includes(p) ? "bg" : "muted", size: 12, rx: 6 }));
          G.label(24, 290, "scrambled on purpose: logical order ≠ physical order", { size: 12, color: "ink" });
          G.rect(24, 312, 592, 80, { stroke: "line", rx: 8 });
          G.label(40, 338, "one thread block per (sequence, query head), 4 warps; GQA: kv head = h / (H / Hkv)", { size: 12, color: "ink" });
          G.label(40, 360, "4 warp states merged (P5.5 rule) · 4,096 tokens = 512 MiB KV (Llama-3-8B)", { size: 12, color: "ink" });
          G.label(40, 380, "≈ 1.8 ms per step to stream it at 300 GB/s (example value, lower bound)", { size: 12, color: "ok" });
          G.from(toks, { opacity: 0, stagger: 0.01, duration: 0.2 });
          G.from(bt.concat(pool), { opacity: 0, duration: 0.3, delay: 0.3 });
          G.caption("block table: one lookup per 16 tokens, then a contiguous read");
        } },

      { rail: "check", title: "Test it, place it, know what is missing",
        body: `<p><b>Correctness.</b> The tests compare with a double-precision CPU reference run on the same fp16-rounded inputs. The output is rounded once to fp16 (u = 2<sup>-11</sup> ≈ 4.9e-4 relative, outputs ≤ 1 in magnitude) and the math is fp32 with the fast <code>__expf</code>, so <code>atol = 2e-3</code> covers it with margin (derived like the tolerances in P5.5). Test full and causal, and awkward N that is not a multiple of the tile size, where the masking of out-of-range keys is what breaks.</p>
<p><b>What the repo gives you</b> (<code>platform/kernels/include/d4/attention.cuh</code>): <code>attn_naive</code> (exercise 1), <code>attn_flash1</code> (FA-1 structure, exercise 2), <code>attn_flash2</code> (FA-2, exercise 3) and <code>paged_decode_attention</code> (exercise 4). All are CUDA-core kernels with head_dim 64, fp16 storage and fp32 math. The <b>L6 exit check</b> is FA-2 at ≥ 5× the naive kernel at N = 4096 (<code>attention_bench</code>); it is a ratio of two measurements, so it holds on any GPU.</p>
<p><b>The honest gap.</b> Run <code>vs_flash_attn.py</code> on an L4 and the official flash-attn will be well ahead of your FA-2. Do not conclude the algorithm is slow: the official kernel puts both matmuls on tensor cores (P5.7) and rescales on the accumulator fragments. Closing that gap is the stretch exercise, and the reason "fewer non-matmul FLOPs" is an FA-2 selling point: once the matmuls are fast, the softmax rescale is what is left. FA-3 (Hopper) adds warp specialisation and asynchronous copies; the course covers it as an overview only.</p>
<p>Common mistakes: rescaling <code>o</code> but forgetting <code>l</code> (or the reverse); NaN from <code>-inf - (-inf)</code> on fully masked tiles; masking with <code>j ≥ i</code> instead of <code>j &gt; i</code>; and comparing a CUDA-core kernel with flash-attn.</p>`,
        scene(G) {
          const steps = [["naive", "S in DRAM, 4 passes", "hot"], ["FA-1", "tiles; O round trips", "v"], ["FA-2", "Q outer; registers", "k"], ["+ tensor cores", "stretch (P5.7)", "ok"], ["flash-attn", "official, CUTLASS", "q"]];
          G.text(24, 32, "the attention ladder", { size: 13 });
          const bs = steps.map(([name, note, col], i) => {
            const y = 54 + i * 54;
            const b = G.box(24 + i * 20, y, 200, 38, name, { fill: col, size: 14 });
            G.label(244 + i * 20, y + 24, note, { size: 12, color: "ink" });
            return b;
          });
          G.rect(24, 332, 592, 60, { stroke: "line", rx: 8 });
          G.label(40, 356, "exit check (L6): FA-2 ≥ 5× naive at N = 4096, fp16, same GPU", { size: 12, color: "ink" });
          G.label(40, 376, "tolerance: fp16 output u = 4.9e-4, atol 2e-3; test full, causal, odd N", { size: 12, color: "ink" });
          G.from(bs, { opacity: 0, x: -20, stagger: 0.1, duration: 0.25 });
          G.caption("each rung removes memory traffic or adds hardware");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Attention is S = QKᵀ/√d, a row softmax, then O = P·V. S is N × N: 1 GiB for 16 heads at N = 4,096, against 32 MiB of real inputs and outputs.</li>
<li>Naive attention moves S through DRAM four times (about 4 GiB) at about 16 FLOP per byte (d / 4): memory-bound for every N.</li>
<li>Online softmax makes the output computable in tiles: keep (m, l, o), rescale both l and o by e^(m - m_new) when the max grows, divide by l at the end. The result is exact.</li>
<li>FlashAttention keeps a Q tile on chip and streams K/V tiles past it, so the score tile is 8 KiB on chip instead of a table in DRAM. K and V are re-read N / Br times, mostly from L2: the bytes range from 32 MiB to 1,040 MiB, measurable with ncu.</li>
<li>FA-2 makes the query tile the outer loop (state in registers, O written once, more parallelism) and skips fully masked causal tiles (4,160 of 8,192 tile visits).</li>
<li>Decode has one query, so it is bound by reading the KV cache (512 MiB per 4,096-token Llama-3-8B sequence); a paged kernel adds one block-table lookup per block and merges per-warp softmax states.</li></ul>`,
    sim: {
      title: "Attention traffic calculator",
      intro: "Set the sequence length, number of heads, tile sizes and a GPU's example bandwidth. The chart compares the DRAM bytes of naive attention, the course's FA-1, and FA-2 with and without cache help. These are models of bytes, not measurements: the cache decides where real kernels land between the two FA-2 bars.",
      height: 280,
      controls: [
        { id: "n", label: "sequence length N", type: "select", value: 4096, options: [[512, "512"], [1024, "1,024"], [2048, "2,048"], [4096, "4,096"], [8192, "8,192"], [16384, "16,384"]] },
        { id: "bh", label: "batch × heads", type: "select", value: 16, options: [[1, "1"], [8, "8"], [16, "16"], [32, "32"], [64, "64"]] },
        { id: "d", label: "head dimension d", type: "select", value: 64, options: [[64, "64 (course kernels)"], [128, "128 (Llama 3)"]] },
        { id: "br", label: "Br (query rows per tile, FA-2)", type: "select", value: 64, options: [[32, "32"], [64, "64"], [128, "128"], [256, "256"]] },
        { id: "bc", label: "Bc (keys per tile, FA-1 launches)", type: "select", value: 32, options: [[32, "32"], [64, "64"], [128, "128"], [256, "256"]] },
        { id: "gpu", label: "GPU (example values)", type: "select", value: "l4", options: [["t4", "T4 · 320 GB/s · 16 GB"], ["l4", "L4 · 300 GB/s · 24 GB"], ["a100", "A100-80GB · 2,039 GB/s · 80 GB"]] },
      ],
      draw(G, v) {
        const g = GPUS[v.gpu], N = v.n, d = v.d, BH = v.bh;
        const naive = BH * naiveBytes(N, d), comp = BH * compulsory(N, d), fa2 = BH * fa2NoCache(N, d, v.br);
        const fa1 = fa2 + BH * fa1Extra(N, d, v.bc);
        const rows = [["naive", naive, "hot"], ["FA-1 (d4 launches)", fa1, "v"], ["FA-2, no cache help", fa2, "k"], ["FA-2, perfect reuse", comp, "ok"]];
        const lo = Math.log10(Math.min(...rows.map((r) => r[1]))) - 0.3, hi = Math.log10(Math.max(...rows.map((r) => r[1]))) + 0.15;
        const x0 = 190, W = 330;
        G.text(24, 20, "DRAM bytes (log scale), models", { size: 11, color: "muted" });
        rows.forEach(([name, b, col], i) => {
          const y = 36 + i * 56, w = Math.max(3, (W * (Math.log10(b) - lo)) / (hi - lo));
          G.label(24, y + 18, name, { size: 11, color: "ink" });
          G.rect(x0, y, w, 28, { fill: col, rx: 3 });
          G.text(x0 + w + 8, y + 20, F.bytes(b), { size: 13 });
        });
        G.line(x0, 28, x0, 252, { color: "line" });
        const S = BH * N * N * 4, flops = BH * 4 * N * N * d;
        const t = (b) => b / g.bw;
        return [
          { title: "Score matrix", rows: [["S per head (fp32)", F.bytes(N * N * 4)], ["S for all heads", F.bytes(S)], ["Q, K, V, O (fp16)", F.bytes(comp)], ["S ÷ inputs and outputs", (S / comp).toFixed(1) + "×"], ["GPU memory (example)", g.gb + " GB"]],
            chip: [S < g.gb * 1e9 * 0.8, S < g.gb * 1e9 * 0.8 ? "S fits in GPU memory" : "S alone would not fit in memory"] },
          { title: `Bytes and time lower bound on ${g.name}`, rows: [["naive", F.bytes(naive) + " · " + F.ms(t(naive))], ["FA-1 (d4)", F.bytes(fa1) + " · " + F.ms(t(fa1))], ["FA-2 no cache", F.bytes(fa2) + " · " + F.ms(t(fa2))], ["FA-2 perfect reuse", F.bytes(comp) + " · " + F.ms(t(comp))], ["FLOP per byte (naive)", (flops / naive).toFixed(1)], ["total FLOPs (full)", F.si(flops, 1) + "FLOP"]],
            html: `<p class="note">Byte models, per head and fp16 Q/K/V/O, fp32 scores: naive = 4 table passes + Q, K, V, O; FA-2 = (2 + 2N/Br)·N·d·2 with every K/V re-read from DRAM, or the 4·N·d·2 minimum if the cache serves all re-reads; FA-1 = FA-2 plus an fp32 read-modify-write of O and (m, l) every K/V tile. FA-1's extra bytes may sit in L2 in practice. Times are bytes ÷ example bandwidth (UNVERIFIED, gpu_specs.yaml), a lower bound that ignores compute and caches: measure with <code>ncu dram__bytes.sum</code>.</p>` },
        ];
      },
    },
    practice: {
      intro: `Build once per GPU session (see <code>course/P5-cuda-deep-dive/aws-common.md</code>): <code>cmake -S course/P5-cuda-deep-dive -B build/p5 -DCMAKE_CUDA_ARCHITECTURES=native &amp;&amp; cmake --build build/p5 -j</code>. The CUDA-core kernels and the exit check run on a T4 (<code>g4dn.xlarge</code>); flash-attn needs an L4 (<code>g6.xlarge</code>). Add <code>-DS2S_USE_SOLUTIONS=ON</code> (separate build dir) to test against the reference kernels. Measured results are not yet recorded in the repo (<code>TODO(run-on: g4dn.xlarge)</code>).`,
      items: [
        { title: "Naive attention vs a reference", tier: "T2", goal: "Three kernels: S = QKᵀ/√64 with the causal mask, a row softmax (reuse your P5.5 kernel or d4::softmax), and O = P·V. Count the HBM traffic for N = 4096, B·H = 16 and compare with time × bandwidth.",
          cmd: "ctest --test-dir build/p5 -R p5.8_01-naive --output-on-failure" },
        { title: "FlashAttention-1 structure", tier: "T2", goal: "Annotate flash-attention-minimal (read it, do not copy it), then implement the outer loop over K/V tiles with m, l and the unnormalised O in global workspaces. How much HBM traffic goes on re-reading and re-writing O?",
          cmd: "ctest --test-dir build/p5 -R p5.8_02-fa1 --output-on-failure" },
        { title: "FlashAttention-2 forward + causal (L6 exit check)", tier: "T2", goal: "One block per query tile, (m, l, o) in registers, one rescale per K/V tile, causal tile skipping. attention_bench must show FA-2 at least 5x naive at N = 4096. Stretch: both matmuls on tensor cores with mma.sync.",
          cmd: "ctest --test-dir build/p5 -R p5.8_03-fa2 --output-on-failure && ./build/p5/p5.8_attention_bench | tee results/p5.8-attention.md" },
        { title: "Decode attention over a paged KV cache (hard)", tier: "T2 · hard", goal: "One block per (sequence, query head), 4 warps with their own online-softmax state, GQA head mapping, a block table with scrambled physical blocks. Tested at lengths 1, 15, 16, 17 and with GQA. Feeds P6.3.",
          cmd: "ctest --test-dir build/p5 -R p5.8_04-paged-decode --output-on-failure" },
      ],
      labs: [
        { label: "attention_bench: naive vs FA-1 vs FA-2, full and causal, N = 1k to 4k (the L6 table)", path: "course/P5-cuda-deep-dive/P5.8-flashattention/examples/attention_bench.cu" },
        { label: "PyTorch SDPA and flash-attn v2.8.3.post1 on the same shapes (L4)", path: "course/P5-cuda-deep-dive/P5.8-flashattention/examples/vs_flash_attn.py" },
        { label: "Double-precision CPU reference shared by the tests", path: "course/P5-cuda-deep-dive/P5.8-flashattention/examples/attn_ref.hpp" },
        { label: "D4 kernels: attn_naive, attn_flash1, attn_flash2, paged_decode_attention", path: "platform/kernels/include/d4/attention.cuh" },
        { label: "FlashAttention animation (Q block stationary, K/V streaming, causal skipping)", path: "animations/p5-flashattention.html" },
        { label: "AWS guide (g4dn.xlarge, about 3 instance-hours, ncu dram__bytes.sum)", path: "course/P5-cuda-deep-dive/P5.8-flashattention/aws.md" },
        { label: "Reading: FlashAttention (Dao et al., 2022)", path: "https://arxiv.org/abs/2205.14135" },
        { label: "Reading: FlashAttention-2 (Dao, 2023)", path: "https://arxiv.org/abs/2307.08691" },
      ],
    },
  });
})();
