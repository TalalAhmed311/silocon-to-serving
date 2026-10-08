/* P4.1 — NCCL collectives and bandwidth math. */
(function () {
  const F = S2S.fmt;
  const MOD = "course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives";
  const RC = ["k", "v", "q", "pink"];           // colour of rank 0..3's own data
  const NAMES = ["a", "b", "c", "d"];

  /* time formatter: µs below 1 ms */
  const T = (s) => (s < 1e-3 ? (s * 1e6).toFixed(s < 1e-5 ? 2 : 1) + " µs" : F.ms(s));

  /* ring all-reduce state after k steps (k = 0 … 2(n−1)); sends = the messages of step k */
  function ringState(n, k) {
    const st = [...Array(n)].map((_, r) => [...Array(n)].map(() => [r]));
    let sends = [];
    for (let s = 0; s < Math.min(k, n - 1); s++) {               // reduce-scatter: receiver ADDS
      sends = [...Array(n)].map((_, r) => ({ from: r, to: (r + 1) % n, chunk: (((r - s - 1) % n) + n) % n }));
      const msg = sends.map((m) => [...st[m.from][m.chunk]]);
      sends.forEach((m, i) => { st[m.to][m.chunk] = [...new Set([...st[m.to][m.chunk], ...msg[i]])].sort(); });
    }
    for (let s = 0; s < k - (n - 1); s++) {                       // all-gather: receiver OVERWRITES
      sends = [...Array(n)].map((_, r) => ({ from: r, to: (r + 1) % n, chunk: (((r - s) % n) + n) % n }));
      const msg = sends.map((m) => [...st[m.from][m.chunk]]);
      sends.forEach((m, i) => { st[m.to][m.chunk] = msg[i]; });
    }
    return { st, sends: k ? sends : [] };
  }

  /* draw 4 GPUs × 4 chunks; highlight what arrived in step k */
  function drawRing(G, k, title) {
    const n = 4, { st, sends } = ringState(n, k);
    G.text(24, 34, title, { size: 14 });
    const cells = [];
    for (let r = 0; r < n; r++) {
      const x = 24 + r * 152;
      G.rect(x, 46, 136, 222, { stroke: "line", rx: 10 });
      G.text(x + 68, 66, `GPU ${r}`, { anchor: "middle", size: 13 });
      for (let c = 0; c < n; c++) {
        const who = st[r][c], full = who.length === n, got = sends.some((m) => m.to === r && m.chunk === c);
        const fill = full ? "ok" : who.length > 1 ? "v" : "w";
        cells.push(G.box(x + 10, 78 + c * 46, 116, 38, who.map((w) => NAMES[w] + c).join("+"),
          { fill, size: 11, rx: 6, color: fill === "w" ? "ink" : "bg", stroke: got ? "ink" : undefined, sw: got ? 2.5 : undefined }));
      }
    }
    const arrows = [];
    sends.forEach((m) => {
      const x1 = 92 + m.from * 152, x2 = 92 + m.to * 152;
      if (m.to > m.from) {
        arrows.push(G.arrow(x1 + 34, 296, x2 - 34, 296, { color: "k", w: 2.5 }));
        G.label((x1 + x2) / 2, 288, `chunk ${m.chunk}`, { anchor: "middle", size: 11, color: "k" });
      } else {
        arrows.push(G.path(`M ${x1} 282 C ${x1} 336, ${x2} 336, ${x2} 284`, { color: "k", w: 2.5, arrow: true }));
        G.label((x1 + x2) / 2, 334, `chunk ${m.chunk} (wraps around)`, { anchor: "middle", size: 11, color: "k" });
      }
    });
    return { cells, arrows };
  }

  /* one row of before → after strips for 4 ranks: cell(r, c) → color or null */
  function strips(G, y, name, before, after, sub) {
    G.text(16, y + 16, name, { size: 13 });
    if (sub) G.label(16, y + 32, sub, { size: 10 });
    const out = [];
    const strip = (x0, f) => {
      for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
        const col = f(r, c);
        out.push(G.rect(x0 + r * 56 + c * 12, y, 11, 26, col ? { fill: col, rx: 2 } : { stroke: "line", rx: 2, sw: 1 }));
      }
    };
    strip(160, before); G.arrow(386, y + 13, 412, y + 13, { color: "muted" }); strip(420, after);
    return out;
  }
  const rankHeads = (G, y) => {
    for (let r = 0; r < 4; r++) { G.label(160 + r * 56 + 22, y, `r${r}`, { anchor: "middle", size: 11 }); G.label(420 + r * 56 + 22, y, `r${r}`, { anchor: "middle", size: 11 }); }
    G.label(160, y - 18, "before", { size: 11 }); G.label(420, y - 18, "after", { size: 11 });
  };

  /* log-scale helpers for the ring/tree charts */
  const ringT = (S, n, a, B) => 2 * (n - 1) * a + (2 * (n - 1) / n) * S / B;
  const treeT = (S, n, a, B) => 2 * Math.log2(n) * a + 2 * S / B;
  const cross = (n, a, B) => ((2 * (n - 1) - 2 * Math.log2(n)) * a) / ((2 - (2 * (n - 1)) / n) / B);
  function logChart(G, box, xr, yr) {
    const [x0, y0, w, h] = box;
    const X = (v) => x0 + (w * (Math.log10(v) - xr[0])) / (xr[1] - xr[0]);
    const Y = (v) => y0 + h - (h * (Math.log10(v) - yr[0])) / (yr[1] - yr[0]);
    return { X, Y };
  }
  const sizeLabel = (b) => (b >= 2 ** 30 ? b / 2 ** 30 + " GiB" : b >= 2 ** 20 ? b / 2 ** 20 + " MiB" : b >= 1024 ? b / 1024 + " KiB" : b + " B");
  const timeLabel = (s) => (s >= 1 ? s + " s" : s >= 1e-3 ? s * 1e3 + " ms" : Math.round(s * 1e6) + " µs");

  S2S.lesson({
    id: "p4-1", n: "P4.1", title: "NCCL collectives",
    subtitle: "Multi-GPU and distributed · first principles · T0 simulators, T3 for nccl-tests",
    kicker: "Lesson · ≈ 45 min",
    headline: "Passing notes around a ring",
    intro: `<p>Once a model or a training job spans several GPUs, those GPUs must combine their numbers: add up gradients, sum partial matrix products, hand out weights. This lesson builds the small vocabulary of <b>collective operations</b> that every multi-GPU program uses, then derives, step by step, the ring algorithm that NCCL (NVIDIA's collective communication library) runs and the formula that predicts its time. P4.2 uses that formula to price tensor parallelism.</p>`,
    facts: ["11 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "rank 0's data"], ["v", "rank 1"], ["q", "rank 2"], ["pink", "rank 3"], ["ok", "sum of all ranks"]],
    prev: "p3-9", next: "p4-2",
    steps: [
      { rail: "why talk", title: "Why GPUs have to talk",
        body: `<p>Two situations force GPUs to share numbers.</p>
<ul><li><b>Data parallel training.</b> Four GPUs each hold a full copy of the model and each processes a different slice of the batch. Each computes its own <b>gradient</b> (the direction to nudge every weight). Before anyone updates, all four must agree on the <i>average</i> gradient, or the copies drift apart.</li>
<li><b>Tensor parallelism</b> (P4.2). One layer's matrix is split across GPUs. Each GPU computes a <i>partial</i> result for the same token, and the partial results must be added together.</li></ul>
<p>Both reduce to one operation: every GPU holds a vector of the same length, and every GPU needs the <b>element-wise sum</b> of all of them.</p>
<div class="eq">GPU 0  [1, 2, 0, 3]
GPU 1  [2, 0, 1, 1]
GPU 2  [0, 1, 2, 2]
GPU 3  [1, 1, 1, 0]
sum    [4, 4, 4, 6]   needed on every GPU
mean   [1, 1, 1, 1.5] = sum / 4</div>
<p>For Llama-3-8B in bf16 that vector is not 4 numbers but 8 billion: <b>16 GB</b> of gradient per training step. How fast we can add it up across GPUs decides whether the GPUs compute or wait.</p>`,
        scene(G) {
          const V = [[1, 2, 0, 3], [2, 0, 1, 1], [0, 1, 2, 2], [1, 1, 1, 0]], S = [4, 4, 4, 6];
          const rows = [];
          V.forEach((v, r) => {
            G.text(24, 62 + r * 50, `GPU ${r}`, { size: 13 });
            rows.push(G.row(100, 40 + r * 50, v.map(String), { w: 44, h: 34, gap: 6, fill: RC[r], size: 14 }));
          });
          G.text(330, 62, "different data →", { size: 12, color: "muted" });
          G.text(330, 80, "different gradients", { size: 12, color: "muted" });
          G.line(100, 252, 290, 252, { color: "ink" });
          G.text(24, 290, "sum", { size: 13, color: "ok" });
          const s = G.row(100, 268, S.map(String), { w: 44, h: 34, gap: 6, fill: "ok", size: 14 });
          G.text(330, 290, "must end up on EVERY GPU", { size: 13, color: "ok" });
          G.label(24, 350, "Llama-3-8B: 8.0 B gradients × 2 bytes (bf16) = 16 GB to add up, every step", { size: 13, color: "ink" });
          G.from(rows.flat(), { opacity: 0, y: -10, stagger: 0.04, duration: 0.3 });
          G.from(s, { opacity: 0, scale: 0.6, transformOrigin: "center", stagger: 0.1, delay: 0.8, duration: 0.3 });
          G.caption("each GPU computed different numbers; all of them need the total");
        } },

      { rail: "collectives I", title: "Collectives: who ends up with what",
        body: `<p>A <b>collective</b> is a communication operation that all GPUs in a group call together. Each GPU in the group is a <b>rank</b>, numbered 0 … n−1. A collective is defined only by what each rank holds <i>before</i> and <i>after</i>; the library picks how the bytes actually move.</p>
<table><tr><th>collective</th><th>before</th><th>after</th></tr>
<tr><td>broadcast</td><td>root has x</td><td>every rank has x</td></tr>
<tr><td>reduce</td><td>rank r has x<sub>r</sub></td><td>root has Σ x<sub>r</sub></td></tr>
<tr><td><b>all-reduce</b></td><td>rank r has x<sub>r</sub></td><td><b>every</b> rank has Σ x<sub>r</sub></td></tr></table>
<p>"Reduce" means combine with an operation, usually a sum (max and min also exist). Broadcast sends weights from one rank to the others at start-up. Reduce collects a metric on one rank. <b>All-reduce</b> is the operation from step 1, and the workhorse of both data and tensor parallelism.</p>`,
        scene(G) {
          rankHeads(G, 70);
          const a = strips(G, 90, "broadcast", (r) => (r === 0 ? "k" : null), () => "k", "root r0 → all");
          const b = strips(G, 170, "reduce", (r) => RC[r], (r) => (r === 0 ? "ok" : null), "all → root r0");
          const c = strips(G, 250, "all-reduce", (r) => RC[r], () => "ok", "all → all");
          G.label(160, 340, "each strip = one rank's buffer, in 4 chunks", { size: 12 });
          G.label(160, 360, "colour = whose data it is · green = the sum over all ranks", { size: 12 });
          G.from([...a, ...b, ...c].filter((_, i) => (i % 32) >= 16), { opacity: 0, stagger: 0.008, duration: 0.2 });
          G.caption("a collective is defined by its before and after, not by how bytes move");
        } },

      { rail: "collectives II", title: "Three more: reduce-scatter, all-gather, all-to-all",
        body: `<p>Three more collectives deal with <b>pieces</b> of a buffer. Split every rank's buffer into n equal chunks.</p>
<ul><li><b>Reduce-scatter:</b> rank r ends up with the sum of <i>chunk r only</i>. The total work of an all-reduce, but each rank keeps just its share.</li>
<li><b>All-gather:</b> rank r starts with chunk r; afterwards every rank has all n chunks, in rank order. Nothing is added; pieces are copied.</li>
<li><b>All-to-all:</b> rank r holds n blocks, one addressed to each rank. Afterwards rank r has block r from everyone. Mixture-of-experts models use it to send each token to the GPU holding its expert.</li></ul>
<div class="eq">reduce-scatter  then  all-gather  =  all-reduce</div>
<p>That identity is the key to the next steps: first make each rank responsible for summing one chunk, then share the finished chunks. FSDP (P4.2) uses the two halves separately.</p>`,
        check: { q: "After the collective, rank r holds only chunk r of the element-wise sum. Which collective is it?",
          options: ["all-reduce", "reduce-scatter", "all-gather", "all-to-all"], answer: 1,
          why: "Reduce-scatter sums across ranks (reduce) but leaves each rank one piece (scatter). All-reduce would leave the whole sum on every rank; all-gather and all-to-all copy data without adding." },
        scene(G) {
          rankHeads(G, 70);
          const a = strips(G, 90, "reduce-scatter", (r) => RC[r], (r, c) => (c === r ? "ok" : null), "r keeps Σ chunk r");
          const b = strips(G, 170, "all-gather", (r, c) => (c === r ? RC[r] : null), (r, c) => RC[c], "copy, no adding");
          const c = strips(G, 250, "all-to-all", (r) => RC[r], (r, c) => RC[c], "block r from each");
          G.label(160, 340, "all-to-all: rank r's 4 blocks go to ranks 0, 1, 2, 3", { size: 12 });
          G.label(160, 360, "after: each rank holds one block from every rank", { size: 12 });
          G.from([...a, ...b, ...c].filter((_, i) => (i % 32) >= 16), { opacity: 0, stagger: 0.008, duration: 0.2 });
          G.caption("reduce-scatter followed by all-gather gives exactly the all-reduce result");
        } },

      { rail: "α and β", title: "What one message costs",
        body: `<p>To predict a collective's time, first price a single message of S bytes over one link. Two things cost time:</p>
<ul><li><b>α (latency):</b> a fixed cost per message, whatever its size: launching the transfer, synchronising sender and receiver, crossing the wire. Microseconds.</li>
<li><b>S / B (transfer):</b> the bytes divided by the link's bandwidth B.</li></ul>
<div class="eq">t(S) = α + S / B          (the "α–β model", β = 1/B)

example values: α = 5 µs, B = 100 GB/s
  8 KiB:   5 µs + 0.08 µs  ≈ 5.1 µs   (98% latency)
  256 MiB: 5 µs + 2,684 µs ≈ 2.69 ms  (0.2% latency)
  break-even where α = S/B:  S = α·B = 500 KB</div>
<p>Small messages are <b>latency-bound</b>: making the link faster barely helps. Large messages are <b>bandwidth-bound</b>: α disappears. It is the same idea as the roofline in P1.4, with latency in place of compute. The α and B above are example values; exercise 2 fits real ones from your own measurements.</p>`,
        scene(G) {
          const a = 5e-6, B = 100e9;
          const lane = (y, S, name) => {
            const t = a + S / B, sc = 560 / t;
            G.text(24, y - 10, `${name}: total ${T(t)}`, { size: 13 });
            const r1 = G.rect(24, y, Math.max(3, a * sc), 34, { fill: "hot", rx: 3 });
            const r2 = G.rect(24 + a * sc, y, Math.max(2, (S / B) * sc), 34, { fill: "k", rx: 3 });
            return [r1, r2];
          };
          const s1 = lane(70, 8192, "8 KiB message");
          G.label(24, 124, "almost all α: the bytes take 0.08 µs", { size: 12 });
          const s2 = lane(186, 2 ** 28, "256 MiB message");
          G.label(24, 240, "almost all S/B: α is a sliver at the left edge", { size: 12 });
          G.rect(24, 290, 14, 14, { fill: "hot", rx: 2 }); G.label(44, 302, "α  fixed per message", { size: 12, color: "ink" });
          G.rect(240, 290, 14, 14, { fill: "k", rx: 2 }); G.label(260, 302, "S / B  grows with size", { size: 12, color: "ink" });
          G.label(24, 350, "example values: α = 5 µs, B = 100 GB/s (not a measurement)", { size: 12 });
          G.label(24, 372, "each bar is scaled to its own total time", { size: 12 });
          G.from([...s1, ...s2], { attr: { width: 0 }, stagger: 0.2, duration: 0.5 });
          G.caption("small messages pay latency, big messages pay bandwidth");
        }, legend: [["hot", "latency α"], ["k", "transfer S/B"]] },

      { rail: "naive", title: "The obvious all-reduce has a traffic jam",
        body: `<p>The simplest all-reduce: every rank sends its whole buffer to rank 0, rank 0 adds them up, then sends the sum back to everyone.</p>
<p>Count rank 0's traffic with n = 8 ranks and S = 16 GB (the Llama-3-8B gradient). It receives 7 × 16 = 112 GB and sends 112 GB back. Every other rank sends 16 GB and receives 16 GB, so their links sit idle most of the time.</p>
<div class="eq">rank 0 link:  2 (n−1) S = 224 GB
at B = 100 GB/s (example):  ≈ 2.24 s per step
(1.12 s if send and receive fully overlap)</div>
<p>The problem is not the total bytes; it is that one link carries nearly all of them. A good algorithm spreads the traffic so that <b>every link is busy at the same time</b>.</p>`,
        scene(G) {
          const cx = 190, cy = 210, R = 140, nodes = [], arrows = [];
          for (let r = 1; r < 8; r++) {
            const ang = -Math.PI / 2 + ((r - 1) * 2 * Math.PI) / 7, x = cx + R * Math.cos(ang), y = cy + R * Math.sin(ang);
            arrows.push(G.line(x, y, cx, cy, { color: "hot", w: 3, opacity: 0.8 }));
            nodes.push(G.box(x - 24, y - 16, 48, 32, `r${r}`, { fill: "w", color: "ink", size: 12 }));
          }
          const hub = G.box(cx - 32, cy - 22, 64, 44, "r0", { fill: "hot", size: 15 });
          G.text(372, 56, "bytes through each rank's link", { size: 13 }); G.label(372, 74, "(sent + received)", { size: 11 });
          G.bars(380, 330, [14, 2, 2, 2, 2, 2, 2, 2], { w: 22, gap: 8, h: 220, fill: (i) => (i === 0 ? "hot" : "w") });
          for (let i = 0; i < 8; i++) G.label(391 + i * 30, 348, `r${i}`, { anchor: "middle", size: 11 });
          G.label(380, 98, "r0: 14 × S", { color: "hot" }); G.label(380, 372, "others: 2 × S", { size: 12 });
          G.pulse(hub, { repeat: 6 }); G.from(arrows, { opacity: 0, stagger: 0.08, duration: 0.3 });
          G.caption("one link carries almost everything; the others wait");
        }, legend: [["hot", "the bottleneck rank"], ["w", "mostly idle ranks"]] },

      { rail: "ring: step 1", title: "The ring: everyone sends a little, all at once",
        body: `<p>Arrange the ranks in a <b>ring</b>: rank r only ever sends to rank r+1, and the last rank sends to rank 0. Each link then carries traffic in one direction only, and all n links can work at the same time.</p>
<p>Split every buffer into n chunks. In each <b>step</b>, every rank sends one chunk (S/n bytes) to its right-hand neighbour, and the neighbour <b>adds</b> it to its own copy of that chunk.</p>
<div class="eq">step s (from 0): rank r sends chunk (r − s − 1) mod n
receiver: chunk += incoming</div>
<p>On the stage, letters name the original owner: a3 is rank 0's chunk 3, b3 is rank 1's chunk 3. In step 1 rank 0 sends chunk 3 to rank 1, so rank 1's chunk 3 now holds a3+b3: a partial sum with two contributors. At the same moment rank 1 sends its chunk 0 to rank 2, and so on round the ring. Every link carried one chunk, and nobody waited.</p>`,
        scene(G) {
          const { cells, arrows } = drawRing(G, 1, "reduce-scatter · step 1 of 3 · every link carries 1 chunk");
          G.label(24, 372, "grey = own data only · amber = partial sum · outlined = just received", { size: 12 });
          G.label(24, 394, "chunk size = S / 4, so each rank sent a quarter of its buffer", { size: 12 });
          G.from(arrows, { opacity: 0, duration: 0.5 });
          G.from(cells, { opacity: 0.3, stagger: 0.02, duration: 0.3 });
          G.caption("four sends happen at the same moment, one per link");
        }, legend: [["w", "own data"], ["v", "partial sum"], ["ok", "full sum"]] },

      { rail: "reduce-scatter", title: "After n − 1 steps, each rank owns one finished chunk",
        body: `<p>Keep going. Follow chunk 0: in step 1 rank 2 made b0+c0; in step 2 rank 3 receives it and adds d0; in step 3 rank 0 receives b0+c0+d0 and adds a0. A partial sum grows by one contributor per hop as it travels around the ring.</p>
<p>A chunk needs all n contributors, so it must travel n − 1 hops. After <b>n − 1 steps</b> every rank holds exactly one chunk that is fully summed: with this schedule, rank r owns the finished chunk r.</p>
<div class="eq">n = 4:  3 steps × S/4 bytes = 0.75 S sent per rank
result: rank r holds Σ of chunk r     (reduce-scatter)</div>
<p>That is precisely the reduce-scatter collective from step 3. The other chunks on each rank are stale partial sums; the second phase overwrites them.</p>`,
        scene(G) {
          const { cells, arrows } = drawRing(G, 3, "reduce-scatter · step 3 of 3 · each rank owns one full sum");
          G.label(24, 372, "green = the sum over all 4 ranks · GPU r owns chunk r", { size: 12 });
          G.label(24, 394, "each green chunk travelled 3 hops, gaining one contributor per hop", { size: 12 });
          G.from(arrows, { opacity: 0, duration: 0.5 });
          G.from(cells.filter((_, i) => i % 5 === 0), { scale: 0.7, transformOrigin: "center", duration: 0.5 });
          G.caption("n − 1 steps: every chunk is finished somewhere");
        }, legend: [["w", "own data"], ["v", "partial sum"], ["ok", "full sum"]] },

      { rail: "all-gather", title: "Then pass the finished chunks around",
        body: `<p>Phase two is an <b>all-gather</b>. Each rank sends its finished chunk to the right; the receiver <b>overwrites</b> its stale copy (no adding) and forwards it in the next step. A finished chunk again needs n − 1 hops to reach every rank.</p>
<div class="eq">1 reduce-scatter  n − 1 steps  receiver adds
2 all-gather      n − 1 steps  receiver copies
total             2(n − 1) steps of S/n bytes</div>
<p>For 4 ranks: 6 steps, and every rank sent 6 × S/4 = <b>1.5 S</b>. Compare the naive version, where rank 0 sent 3 S and received 3 S. The repository's simulator <code>examples/01_collectives_numpy.py</code> prints exactly this: <code>4 ranks, 6 steps (2(N-1)), each rank sent 1.50× its buffer</code>.</p>`,
        check: { q: "How many communication steps does a ring all-reduce take on 8 GPUs?",
          options: ["7", "8", "14", "16"], answer: 2,
          why: "n − 1 = 7 steps of reduce-scatter so every chunk collects all 8 contributors, then 7 steps of all-gather so every finished chunk reaches all 8 ranks: 2(n − 1) = 14." },
        scene(G) {
          const { cells, arrows } = drawRing(G, 6, "all-gather · step 3 of 3 (6 of 6 overall) · done");
          G.label(24, 372, "every GPU holds every chunk of the sum: all-reduce complete", { size: 12 });
          G.label(24, 394, "6 steps × S/4 = 1.5 S sent by each rank", { size: 12 });
          G.from(arrows, { opacity: 0, duration: 0.5 });
          G.from(cells, { opacity: 0.2, stagger: 0.03, duration: 0.25 });
          G.caption("2(n − 1) steps in total, every link busy in every step");
        }, legend: [["ok", "full sum"]] },

      { rail: "ring cost", title: "The ring's price: 2(n−1)α + 2(n−1)/n · S/B",
        body: `<p>Now combine the step count with the α–β model. Each of the 2(n−1) steps sends one message of S/n bytes on every link at once, so each step costs α + (S/n)/B.</p>
<div class="eq">T_ring = 2(n−1) · (α + S/(n·B))
       = 2(n−1)·α  +  2(n−1)/n · S/B

n = 8, S = 16 GB, α = 5 µs, B = 100 GB/s (examples)
  latency term:   14 × 5 µs        = 70 µs
  bandwidth term: 1.75 × 0.16 s   = 0.28 s</div>
<p>The naive version took 2(n−1)·S/B = 2.24 s: exactly <b>n = 8 times</b> slower. Look at the factor 2(n−1)/n: 1 for n = 2, 1.5 for 4, 1.75 for 8, 1.97 for 64. It never exceeds 2. Adding GPUs barely changes the bandwidth cost, and no algorithm can do better (each rank must at least send its data out and receive the result), so the ring is <b>bandwidth-optimal</b>.</p>`,
        check: { q: "As the number of ranks n grows, the bytes each rank sends in a ring all-reduce…",
          options: ["grow linearly with n", "approach 2 × S and stop growing", "shrink towards S/n"], answer: 1,
          why: "Each rank sends 2(n−1)/n · S. The fraction (n−1)/n tends to 1, so the total tends to 2S however many GPUs join. Only the latency term 2(n−1)α keeps growing." },
        scene(G) {
          const ns = [2, 4, 8, 16, 32, 64], f = ns.map((n) => (2 * (n - 1)) / n);
          G.text(24, 36, "bytes each rank sends, in units of S", { size: 14 });
          G.axes(60, 60, 540, 200, {});
          G.line(60, 60, 600, 60, { color: "hot", dash: "5 4" }); G.label(600, 36, "dashed: the 2 S limit", { anchor: "end", color: "hot" });
          const b = G.bars(90, 260, f, { w: 54, gap: 32, h: 200, max: 2, fill: "k" });
          ns.forEach((n, i) => { G.label(117 + i * 86, 278, `n = ${n}`, { anchor: "middle", size: 11 }); G.text(117 + i * 86, 250 - (200 * f[i]) / 2, f[i].toFixed(f[i] % 1 ? 3 : 0).replace(/0+$/, "").replace(/\.$/, ""), { anchor: "middle", size: 12, color: "bg" }); });
          G.text(24, 330, "T = 2(n−1)·α  +  2(n−1)/n · S/B", { size: 16, color: "ok" });
          G.label(24, 356, "latency term grows with n · bandwidth term is capped at 2 S/B", { size: 12 });
          G.label(24, 380, "n = 8, 16 GB: ring 0.28 s vs naive 2.24 s (B = 100 GB/s example)", { size: 12, color: "ink" });
          G.from(b, { attr: { height: 0, y: 260 }, stagger: 0.08, duration: 0.4 });
          G.caption("more GPUs, almost the same bytes per GPU");
        }, legend: [["k", "2(n−1)/n"], ["hot", "upper bound 2"]] },

      { rail: "ring vs tree", title: "Small messages: the tree wins",
        body: `<p>The ring's weakness is its 2(n−1) steps: each pays α. A <b>tree</b> all-reduce sums up a binary tree (log₂ n levels) and broadcasts back down, so it pays roughly 2·log₂n·α, but each level sends the whole buffer:</p>
<div class="eq">T_tree ≈ 2·log₂n·α + 2·S/B
T_ring  = 2(n−1)·α + 2(n−1)/n·S/B

n = 8, α = 5 µs, B = 100 GB/s (examples):
  latency: tree 30 µs vs ring 70 µs
  crossover: (14 − 6)·α = (2 − 1.75)·S/B
             S = 32·α·B = 16 MB</div>
<p>Below about 16 MB the tree is faster here; above it the ring is. NCCL measures the topology and picks the algorithm per call (<code>commmodel.crossover_size</code> computes this model's crossover). These are idealised formulas: real NCCL trees are pipelined, so treat the crossover as a direction, not a constant.</p>
<p>Now the case that matters for serving. With tensor parallelism on 4 GPUs, each Llama-3-8B layer does 2 all-reduces of one token's activation: 4,096 × 2 bytes = <b>8 KiB</b>. The bandwidth term is 1.5 × 8,192 / 100 GB/s ≈ 0.12 µs; the ring's latency term is 6α = 30 µs. Across 32 layers that is 64 all-reduces, about <b>1.9 ms per token</b> of almost pure latency.</p>`,
        check: { q: "During decode, a tensor-parallel all-reduce moves 8 KiB. What limits its time?",
          options: ["Link bandwidth: buy faster links", "Latency α: the per-message cost", "GPU arithmetic"], answer: 1,
          why: "8 KiB takes well under a microsecond to transfer at tens to hundreds of GB/s, while each step's α is microseconds. That is why NCCL has low-latency protocols and why engines write custom all-reduce kernels for decode." },
        scene(G) {
          const n = 8, a = 5e-6, B = 100e9;
          const { X, Y } = logChart(G, [80, 50, 520, 280], [3, 9.5], [-5, -0.5]);
          G.axes(80, 50, 520, 280, {});
          [1e-5, 1e-4, 1e-3, 1e-2, 1e-1].forEach((t) => { G.line(80, Y(t), 600, Y(t), { color: "line", w: 0.6 }); G.label(74, Y(t) + 4, timeLabel(t), { anchor: "end", size: 10 }); });
          [1024, 2 ** 20, 2 ** 30].forEach((s) => G.label(X(s), 348, sizeLabel(s), { anchor: "middle", size: 10 }));
          G.label(600, 366, "message size per rank (log) →", { anchor: "end", size: 11 });
          G.label(80, 40, "time (log)", { size: 11 });
          const pts = (f) => { let d = ""; for (let e = 3; e <= 9.5; e += 0.05) { const S = 10 ** e; d += (d ? " L " : "M ") + X(S).toFixed(1) + " " + Y(f(S, n, a, B)).toFixed(1); } return d; };
          const p1 = G.path(pts(ringT), { color: "k", w: 2.5 }), p2 = G.path(pts(treeT), { color: "v", w: 2.5 });
          const c = cross(n, a, B);
          G.line(X(c), 50, X(c), 330, { color: "ink", dash: "4 4" }); G.label(X(c) + 6, 64, `crossover ≈ ${(c / 1e6).toFixed(0)} MB`, { color: "ink" });
          G.line(X(8192), 50, X(8192), 330, { color: "hot", dash: "2 4" }); G.label(X(8192) + 6, 120, "TP decode: 8 KiB", { color: "hot" });
          G.label(X(2e5), 318, "tree: 6α = 30 µs floor", { color: "v", anchor: "middle" }); G.label(X(2e5), 264, "ring: 14α = 70 µs floor", { color: "k", anchor: "middle" });
          G.label(X(c) - 6, 84, "tree faster ←", { anchor: "end", color: "v" }); G.label(X(c) + 6, 84, "→ ring faster", { color: "k" });
          G.from([p1, p2], { opacity: 0, duration: 0.6 });
          G.caption("n = 8 · α = 5 µs · B = 100 GB/s (example values, idealised model)");
        }, legend: [["k", "ring"], ["v", "tree"], ["hot", "TP decode message"]] },

      { rail: "busbw and links", title: "Measuring it: algbw, busbw and the links",
        body: `<p>NVIDIA's <code>nccl-tests</code> (pinned v2.21.1) times each collective over message sizes from 8 bytes to 4 GB and prints two bandwidths:</p>
<div class="eq">algbw = S / t        what your code feels
busbw = algbw × factor, where factor is
  2(n−1)/n   all-reduce
  (n−1)/n    all-gather, reduce-scatter
  1          broadcast, reduce</div>
<p>The factor is the ring's bytes-per-rank from step 9 (nccl-tests <code>doc/PERFORMANCE.md</code>). It converts the result into what each <i>link</i> actually carried, so <b>busbw is the number to compare with the link's peak</b>, whatever n and whatever the collective. Comparing algbw with the peak undercounts all-reduce traffic by almost 2×.</p>
<p>Which link? <code>nvidia-smi topo -m</code> prints the path between each GPU pair: <code>NV#</code> (NVLink), <code>PIX</code>/<code>PHB</code> (PCIe, through a switch or the CPU's host bridge), <code>SYS</code> (across CPU sockets, slowest). <code>NCCL_DEBUG=INFO</code> prints the rings and trees NCCL built. A PCIe 4.0 x16 slot signals 16 GT/s on each of 16 lanes, about 31.5 GB/s per direction; NVLink figures are in the A100/H100 whitepapers. Treat every peak as unverified until you cite the spec for your machine.</p>`,
        check: { q: "nccl-tests reports algbw = 100 GB/s for all_reduce on 8 GPUs. What busbw does it print?",
          options: ["100 GB/s", "87.5 GB/s", "175 GB/s", "800 GB/s"], answer: 2,
          why: "busbw = algbw × 2(n−1)/n = 100 × 14/8 = 175 GB/s. That is the traffic each link carried, and it is the figure to compare with the per-GPU link bandwidth." },
        scene(G) {
          const n = 8, a = 5e-6, B = 100e9;
          const { X } = logChart(G, [80, 60, 520, 230], [3, 9.6], [0, 1]);
          const Yb = (bw) => 290 - (230 * bw) / 110e9;
          G.axes(80, 60, 520, 230, {});
          G.line(80, Yb(B), 600, Yb(B), { color: "hot", dash: "5 4" }); G.label(596, Yb(B) - 6, "link B = 100 GB/s (example)", { anchor: "end", color: "hot" });
          let d1 = "", d2 = "";
          for (let e = 3; e <= 9.6; e += 0.05) {
            const S = 10 ** e, t = ringT(S, n, a, B), alg = S / t, bus = alg * (2 * (n - 1)) / n;
            d1 += (d1 ? " L " : "M ") + X(S).toFixed(1) + " " + Yb(bus).toFixed(1);
            d2 += (d2 ? " L " : "M ") + X(S).toFixed(1) + " " + Yb(alg).toFixed(1);
          }
          const p1 = G.path(d1, { color: "ok", w: 2.5 }), p2 = G.path(d2, { color: "k", w: 2.5, dash: "6 4" });
          [0, 50e9, 100e9].forEach((bw) => G.label(74, Yb(bw) + 4, bw / 1e9 + "", { anchor: "end", size: 10 }));
          G.label(80, 50, "GB/s", { size: 11 });
          [1024, 2 ** 20, 2 ** 30].forEach((s) => G.label(X(s), 308, sizeLabel(s), { anchor: "middle", size: 10 }));
          G.label(600, 326, "message size (log) →", { anchor: "end", size: 11 });
          G.label(600, Yb(100e9) + 36, "busbw → the link peak", { color: "ok", anchor: "end" }); G.label(600, Yb(57e9) + 20, "algbw = busbw ÷ 1.75", { color: "k", anchor: "end" });
          G.label(24, 360, "modelled from T_ring with n = 8, α = 5 µs: a shape, not a measurement", { size: 12 });
          G.label(24, 382, "nvidia-smi topo -m:  NV# = NVLink · PIX/PHB = PCIe · SYS = across sockets", { size: 12, color: "ink" });
          G.from([p1, p2], { opacity: 0, duration: 0.6 });
          G.caption("small sizes: α hides the link · large sizes: busbw approaches the peak");
        }, legend: [["ok", "busbw"], ["k", "algbw"], ["hot", "link peak"]] },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A collective is defined by what each rank holds before and after: broadcast, reduce, all-reduce, reduce-scatter, all-gather, all-to-all.</li>
<li>One message costs α + S/B: small messages are latency-bound, large ones bandwidth-bound.</li>
<li>Ring all-reduce = reduce-scatter (n−1 steps, adding) + all-gather (n−1 steps, copying), with every link busy each step.</li>
<li>Its time is 2(n−1)α + 2(n−1)/n · S/B: bandwidth-optimal, but the latency term grows with n, so trees win for small messages.</li>
<li>busbw = algbw × 2(n−1)/n is what you compare with a link's cited peak.</li></ul>`,
    sim: {
      title: "Ring or tree? Price an all-reduce",
      intro: "Choose the number of GPUs, the per-step latency α, the per-link bandwidth B and a message size. The chart draws both models across all sizes; the dot marks your message. The panels split the ring's time into its latency and bandwidth terms and show what nccl-tests would print if the model were exact.",
      height: 300,
      controls: [
        { id: "n", label: "ranks n", min: 2, max: 64, value: 8 },
        { id: "a", label: "latency α per step, µs (example value: fit yours in exercise 2)", min: 0.5, max: 50, step: 0.5, value: 5, format: (v) => v + " µs" },
        { id: "bw", label: "link bandwidth B, GB/s (example value: check the spec sheet)", min: 5, max: 900, step: 5, value: 100, format: (v) => v + " GB/s" },
        { id: "e", label: "message size per rank", min: 10, max: 32, step: 1, value: 13, format: (v) => sizeLabel(2 ** v) },
        { id: "p", label: "preset", type: "select", value: "none", options: [["none", "free choice"], ["tp", "TP decode: 1 token × 4096 × bf16"], ["dp", "DP gradients: Llama-3-8B bf16 (16 GB)"]] },
      ],
      draw(G, v) {
        const n = v.n, a = v.a * 1e-6, B = v.bw * 1e9;
        const S = v.p === "tp" ? 8192 : v.p === "dp" ? 16e9 : 2 ** v.e;
        const xr = [3, 10.3];
        let lo = Infinity, hi = 0;
        for (let e = xr[0]; e <= xr[1]; e += 0.1) { const s = 10 ** e; lo = Math.min(lo, ringT(s, n, a, B), treeT(s, n, a, B)); hi = Math.max(hi, ringT(s, n, a, B), treeT(s, n, a, B)); }
        const yr = [Math.floor(Math.log10(lo)), Math.ceil(Math.log10(hi))];
        const { X, Y } = logChart(G, [70, 24, 550, 236], xr, yr);
        G.axes(70, 24, 550, 236, {});
        for (let k = yr[0]; k <= yr[1]; k++) { const t = 10 ** k; G.line(70, Y(t), 620, Y(t), { color: "line", w: 0.5 }); G.label(64, Y(t) + 4, timeLabel(+t.toPrecision(1)), { anchor: "end", size: 10 }); }
        [1024, 2 ** 20, 2 ** 30].forEach((s) => G.label(X(s), 276, sizeLabel(s), { anchor: "middle", size: 10 }));
        G.label(620, 292, "message size per rank (log) →", { anchor: "end", size: 11 });
        const curve = (f) => { let d = ""; for (let e = xr[0]; e <= xr[1] + 1e-9; e += 0.05) { const s = 10 ** e; d += (d ? " L " : "M ") + X(s).toFixed(1) + " " + Y(f(s, n, a, B)).toFixed(1); } return d; };
        G.path(curve(ringT), { color: "k", w: 2.5 }); G.path(curve(treeT), { color: "v", w: 2.5 });
        const c = cross(n, a, B);
        if (c > 10 ** xr[0] && c < 10 ** xr[1]) { G.line(X(c), 24, X(c), 260, { color: "muted", dash: "4 4" }); G.label(X(c) + 4, 38, "crossover", { size: 11 }); }
        const tr = ringT(S, n, a, B), tt = treeT(S, n, a, B);
        G.circle(X(S), Y(tr), 5, { fill: "k" }); G.circle(X(S), Y(tt), 5, { fill: "v" });
        G.rect(80, 30, 10, 10, { fill: "k", rx: 2 }); G.label(94, 40, "ring", { size: 11 });
        G.rect(140, 30, 10, 10, { fill: "v", rx: 2 }); G.label(154, 40, "tree", { size: 11 });
        const lat = 2 * (n - 1) * a, bwt = ((2 * (n - 1)) / n) * S / B, alg = S / tr;
        const sz = S >= 1e9 ? (S / 1e9).toFixed(0) + " GB" : sizeLabel(S);
        return [
          { title: `This all-reduce · ${sz} on ${n} GPUs`, rows: [["ring time", T(tr)], ["tree time", T(tt)], ["steps (ring)", F.num(2 * (n - 1))]],
            chip: [tr <= tt, tr <= tt ? "ring is faster here" : "tree is faster here"] },
          { title: "Ring time, split", gauge: [[lat / tr, "hot"], [bwt / tr, "k"]], gaugeText: `latency ${(100 * lat / tr).toFixed(0)}% · bandwidth ${(100 * bwt / tr).toFixed(0)}%`,
            rows: [["latency 2(n−1)α", T(lat)], ["bandwidth 2(n−1)/n·S/B", T(bwt)], ["bytes sent per rank", F.bytes((2 * (n - 1) / n) * S)]] },
          { title: "What nccl-tests would print (model)", rows: [["algbw = S / t", (alg / 1e9).toFixed(2) + " GB/s"], ["busbw = algbw × 2(n−1)/n", ((alg * 2 * (n - 1)) / n / 1e9).toFixed(2) + " GB/s"], ["link peak B", v.bw + " GB/s"]] },
          { title: "Crossover", rows: [["tree beats ring below", F.bytes(c)]],
            html: `<p class="note">Idealised α–β models from <code>commmodel.py</code>. Real NCCL pipelines its trees and has several protocols, so measured crossovers differ. With a preset selected, the size slider is ignored.</p>` },
        ];
      },
    },
    practice: {
      items: [
        { title: "Ring all-reduce simulator", tier: "T0 · medium", goal: "Implement reduce_scatter, all_gather and all_reduce in exercises/ring.py on NumPy \"ranks\"; match sum for n = 1…8 in exactly 2(n−1) steps.",
          cmd: `uv run pytest ${MOD}/exercises/test_ring.py` },
        { title: "Fit the α–β model", tier: "T0 → T3 · medium", goal: "Recover α and B from noisy timings with commmodel.fit_alpha_beta, then fit your own nccl-tests run (small sizes for α, large sizes for B).",
          cmd: `uv run pytest ${MOD}/exercises/test_comm_model.py -k "fit or tree"` },
        { title: "busbw from algbw", tier: "T0 · easy", goal: "Derive the busbw factor of each collective from its ring byte count and check it against nccl-tests' PERFORMANCE.md factors.",
          cmd: `uv run pytest ${MOD}/exercises/test_comm_model.py -k "busbw or parse or rs_plus_ag"` },
        { title: "Reduce-scatter + all-gather = all-reduce", tier: "T0 + T3 · hard", goal: "Show the identity in the simulator (same result, same steps), then compare the timings from nccl-tests and explain where they differ.",
          cmd: `uv run pytest ${MOD}/exercises/test_ring.py::test_rs_then_ag_equals_all_reduce` },
      ],
      labs: [
        { label: "Simulated ring on N ranks (<code>uv run python … --ranks 4</code>)", path: `${MOD}/examples/01_collectives_numpy.py` },
        { label: "Real collectives on CPU processes with gloo (<code>uv run --extra torch python … --world 4</code>)", path: `${MOD}/examples/02_torch_dist_gloo.py` },
        { label: "T3: build and sweep nccl-tests v2.21.1 with NCCL_DEBUG=INFO", path: `${MOD}/examples/03_nccl_tests.sh` },
        { label: "Parse nccl-tests output into size | time | algbw | busbw | % of peak", path: `${MOD}/bench/parse_nccl_tests.py` },
        { label: "The cost model P4.2 imports", path: `${MOD}/commmodel.py` },
        { label: "AWS guide: g6.12xlarge (4 GPUs, PCIe), optional p4d.24xlarge; cost, auto-stop, teardown", path: `${MOD}/aws.md` },
        { label: "Step-by-step ring animation", path: "animations/p4-ring-allreduce.html" },
      ],
    },
  });
})();
