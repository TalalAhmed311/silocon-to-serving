/* P5.7 — Tensor cores. One warp-wide instruction does a small matrix product; the job is feeding it (fragments, ldmatrix, cp.async). */
(function () {
  const F = S2S.fmt;
  // Example values from course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/gpu_specs.yaml (UNVERIFIED there):
  // fp32 and dense fp16 tensor TFLOP/s, memory GB/s. smem = per-SM shared memory in KiB (recalled from the CUDA Programming Guide table, UNVERIFIED).
  const GPUS = {
    t4: { name: "T4", sm: "sm_75", fp32: 8.1, tc: 65, bw: 320, smem: 64 },
    l4: { name: "L4", sm: "sm_89", fp32: 30.3, tc: 121, bw: 300, smem: 100 },
    a100: { name: "A100-80GB", sm: "sm_80", fp32: 19.5, tc: 312, bw: 2039, smem: 164 },
    h100: { name: "H100-SXM", sm: "sm_90", fp32: 67, tc: 989, bw: 3350, smem: 228 },
  };
  // shared-memory halves per pipeline stage in d4/hgemm.cuh: A tile BM×(BK+8) plus B tile BK×(BN+8), fp16
  const stageBytes = (BM, BN, BK) => (BM * (BK + 8) + BK * (BN + 8)) * 2;
  // arithmetic intensity (FLOP per DRAM byte) of a BM×BN fp16 block tile
  const tileAI16 = (BM, BN) => (BM * BN) / (BM + BN);

  S2S.lesson({
    id: "p5-7", n: "P5.7", title: "Tensor cores",
    subtitle: "CUDA deep dive · first principles · T2, WMMA on sm_70+, mma.sync on sm_80+",
    kicker: "Lesson · ≈ 55 min",
    headline: "A matrix product in one instruction",
    intro: `<p>In P5.6 you pushed an fp32 GEMM toward the GPU's FMA peak. Modern GPUs have a second kind of arithmetic unit, the <b>tensor core</b>, that multiplies small matrices in one step and is several times faster on matrix products. This lesson explains what the instruction computes, why its operands must sit in a strange per-lane layout (a <b>fragment</b>), and the three ways to program it: <b>WMMA</b> (the layout is hidden), <b>mma.sync with ldmatrix</b> (you control every register), and <b>CuTe</b> (layouts as algebra). Then comes the real difficulty: a tensor core is so fast that memory cannot keep up unless you pipeline loads with <code>cp.async</code>. It ends with the int8 path used by quantized serving.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "5 exercises"],
    legend: [["q", "A tile"], ["k", "B tile"], ["ok", "C / result"], ["v", "in flight / loading"], ["hot", "conflict / bug"]],
    prev: "p5-6", next: "p5-8",
    steps: [
      { rail: "why", title: "Matrix multiply is what the chip is for",
        body: `<p>An LLM spends nearly all its FLOPs in matrix multiplication (GEMM). The FMA cores of P5.6 do one fused multiply-add per lane per instruction. That is a general-purpose design: the same cores run any arithmetic. But a GEMM does the <i>same shape</i> of work over and over, and hardware that is built for exactly that shape can do far more per clock and per watt.</p>
<p>A <b>tensor core</b> is that hardware: a unit, several per SM, that multiplies a small tile of A by a small tile of B and adds the result to a tile of C. NVIDIA GPUs from Volta (sm_70) have them. Compare the peak fp32 FMA rate with the dense fp16 tensor-core rate on the spec sheet:</p>
<div class="eq">GPU        fp32 FMA   fp16 tensor   ratio
T4          8.1 TF       65 TF        8.0×
L4         30.3 TF      121 TF        4.0×
A100-80GB  19.5 TF      312 TF       16.0×
H100-SXM     67 TF      989 TF       14.8×
(example values from gpu_specs.yaml, UNVERIFIED;
 dense, not the 2× sparse headline)</div>
<p>So a kernel that stays on FMA cores leaves 4× to 16× of the chip unused. The price: the tensor core takes <b>fp16 / bf16 / int8 / fp8 inputs</b> (not fp32 inputs in the same fast mode), and its operands must arrive in a particular register layout. The rest of the lesson is about paying that price without wasting the gain.</p>`,
        scene(G) {
          const rows = [["T4", 8.1, 65], ["L4", 30.3, 121], ["A100", 19.5, 312], ["H100", 67, 989]];
          const wOf = (v) => (330 * Math.log10(v)) / 3;
          G.text(24, 34, "peak TFLOP/s per GPU, log scale (example values)", { size: 13 });
          const bars = [];
          rows.forEach(([name, f32, tc], i) => {
            const y = 60 + i * 84;
            G.text(24, y + 22, name, { size: 14 });
            bars.push(G.rect(100, y, wOf(f32), 22, { fill: "w", rx: 3 }));
            G.label(108 + wOf(f32), y + 16, "fp32 FMA " + f32, { size: 12, color: "ink" });
            bars.push(G.rect(100, y + 28, wOf(tc), 22, { fill: "ok", rx: 3 }));
            G.label(108 + wOf(tc), y + 44, "fp16 tensor " + tc, { size: 12, color: "ok" });
            G.text(616, y + 40, (tc / f32).toFixed(1) + "×", { anchor: "end", size: 16, color: "v" });
          });
          G.label(24, 398, "grey: CUDA-core FMA · green: tensor core, dense", { size: 12 });
          G.from(bars, { attr: { width: 0 }, duration: 0.5, stagger: 0.07 });
          G.caption("UNVERIFIED example values: check each GPU's datasheet before quoting");
        } },

      { rail: "the mma", title: "One instruction, 2,048 multiply-adds",
        body: `<p>The basic tensor-core operation is <code>D = A · B + C</code> on small fixed shapes. For the instruction used in this module, <b>m16n8k16</b>: A is 16×16, B is 16×8, C and D are 16×8 (the name lists M, N, K). One <b>warp</b> issues it, all 32 lanes together.</p>
<div class="eq">D[16×8] = A[16×16] · B[16×8] + C[16×8]

each of the 16 × 8 = 128 outputs is a dot product
of length K = 16:
  D[i][j] = C[i][j] + Σ_k A[i][k] · B[k][j]

multiply-adds = 16 × 8 × 16 = 2,048   (4,096 FLOPs)</div>
<p>Compare the FMA path: one warp-wide <code>FFMA</code> instruction does 32 multiply-adds (one per lane). The tensor instruction does 2,048 in the same issue slot: <b>64× the work per instruction</b>. The spec-sheet gap in step 1 (4× to 16×) is smaller than 64× because the tensor unit takes longer to finish one instruction than the FMA unit does; the exact rate is in each architecture's whitepaper. Either way, to use it you must issue tensor instructions almost every cycle, and each one needs its A, B and C operands <i>already in registers</i>.</p>
<p>Inputs are fp16 here and the accumulator is fp32 (<code>mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32</code>). A product of two fp16 numbers has 11 + 11 = 22 significant bits, which fits fp32's 24 bits, so every product is exact. Only the additions round.</p>`,
        check: { q: "How many multiply-adds does one m16n8k16 mma instruction perform for the warp?",
          options: ["128 (one per output)", "2,048 (16 × 8 × 16)", "4,096"], answer: 1,
          why: "Each of the 16 × 8 = 128 outputs is a length-16 dot product, which is 16 multiply-adds, so 128 × 16 = 2,048 multiply-adds. (4,096 is the FLOP count, counting each multiply-add as 2.) One FFMA instruction gives the warp only 32." },
        scene(G) {
          const u = 10, ax = 24, ay = 74;
          G.text(ax, 34, "mma.m16n8k16 · one warp · one instruction", { size: 13 });
          const A = G.grid(ax, ay, 16, 16, u, u, (r, c) => (r === 5 ? "q" : "line"), { gap: 1, rx: 1 });
          G.text(ax + 8 * u, ay - 14, "A  16×16", { anchor: "middle", size: 12, color: "q" });
          const bx = 224;
          const B = G.grid(bx, ay, 16, 8, u, u, (r, c) => (c === 3 ? "k" : "line"), { gap: 1, rx: 1 });
          G.text(bx + 4 * u, ay - 14, "B  16×8", { anchor: "middle", size: 12, color: "k" });
          const cx = 334;
          const C = G.grid(cx, ay, 16, 8, u, u, (r, c) => (r === 5 && c === 3 ? "ok" : "line"), { gap: 1, rx: 1 });
          G.text(cx + 4 * u, ay - 14, "C → D  16×8", { anchor: "middle", size: 12, color: "ok" });
          G.text(ax + 16 * u + 14, ay + 88, "×", { size: 20, anchor: "middle" });
          G.text(bx + 8 * u + 14, ay + 88, "+", { size: 20, anchor: "middle" });
          G.label(24, 258, "one output = row of A · column of B = 16 multiply-adds", { size: 12, color: "ink" });
          G.rect(24, 278, 592, 112, { stroke: "line", rx: 8 });
          G.label(40, 304, "128 outputs × 16 = 2,048 multiply-adds in one warp instruction", { size: 13, color: "ok" });
          G.label(40, 328, "one FFMA instruction: 32 multiply-adds → 64× less work", { size: 13, color: "ink" });
          G.label(40, 352, "inputs fp16, accumulator fp32: each product is exact", { size: 13, color: "ink" });
          G.label(40, 374, "operands must already be in registers, in a fixed layout", { size: 12, color: "v" });
          G.from(A.concat(B, C), { opacity: 0.2, stagger: 0.001, duration: 0.3 });
          G.caption("a whole 16×16 by 16×8 product per instruction");
        } },

      { rail: "fragments", title: "Operands live spread across the 32 lanes",
        body: `<p>Where do the 256 + 128 + 128 numbers sit? In the <b>registers of the 32 lanes</b>, split by a fixed rule. A lane never holds a whole row or column; it holds a few scattered elements. This slice is a <b>fragment</b>. For the 16×8 accumulator (C and D), each lane holds 4 fp32 values:</p>
<div class="eq">g = lane / 4      t = lane % 4
c0, c1 : row g,     columns 2t, 2t+1
c2, c3 : row g + 8, columns 2t, 2t+1

lane 13: g = 3, t = 1
  → rows 3 and 11, columns 2 and 3

32 lanes × 4 floats = 128 = 16 × 8   ✓</div>
<p>The other operands follow the same logic. A (16×16 fp16) takes 4 registers per lane, each register packing two fp16 values: 32 × 4 × 2 = 256 = 16 × 16. B (16×8) takes 2 registers per lane: 32 × 2 × 2 = 128. The exact maps are in the PTX ISA (figures for <i>mma.m16n8k16</i>), and the course's <code>warp_mma_tile</code> and <code>store_acc</code> in <code>platform/kernels/include/d4/hgemm.cuh</code> encode them.</p>
<p>Why this strange layout? It is what the hardware wires up. You do not choose it; you must produce data in that layout, and that is the whole difficulty of tensor-core programming. A single wrong lane assignment gives a silently permuted result.</p>`,
        check: { q: "In the m16n8 accumulator layout, which elements of the 16×8 tile does lane 13 hold?",
          options: ["Row 13, columns 0 to 3", "Rows 3 and 11, columns 2 and 3", "Rows 1 and 9, columns 3 and 4"], answer: 1,
          why: "g = 13 / 4 = 3 and t = 13 % 4 = 1. The lane holds c0, c1 at row g = 3, columns 2t = 2 and 3, and c2, c3 at row g + 8 = 11, the same columns. 4 values, matching 4 floats per lane." },
        scene(G) {
          const cw = 38, ch = 17, x0 = 40, y0 = 62;
          G.text(24, 34, "who holds which element of the 16×8 accumulator", { size: 13 });
          const cols = ["k", "q", "ok", "v"];
          const cells = [];
          for (let r = 0; r < 16; r++) for (let c = 0; c < 8; c++) {
            const lane = (r % 8) * 4 + Math.floor(c / 2);
            const mine = lane === 13;
            cells.push(G.rect(x0 + c * cw, y0 + r * ch, cw - 2, ch - 2, { fill: cols[Math.floor(c / 2)], rx: 2, opacity: mine ? 1 : 0.35 }));
            G.text(x0 + c * cw + cw / 2 - 1, y0 + r * ch + 12, String(lane), { anchor: "middle", size: 10, color: mine ? "bg" : "ink", weight: mine ? "bold" : undefined });
            if (mine) G.rect(x0 + c * cw, y0 + r * ch, cw - 2, ch - 2, { stroke: "hot", sw: 2, rx: 2 });
          }
          for (let c = 0; c < 8; c++) G.label(x0 + c * cw + cw / 2, y0 - 6, String(c), { anchor: "middle", size: 10 });
          G.label(x0 - 12, y0 + 12, "0", { anchor: "end", size: 10 });
          G.label(x0 - 12, y0 + 3 * ch + 12, "3", { anchor: "end", size: 10, color: "hot" });
          G.label(x0 - 12, y0 + 11 * ch + 12, "11", { anchor: "end", size: 10, color: "hot" });
          G.text(372, 96, "lane 13", { size: 15, color: "hot" });
          G.label(372, 120, "g = 13 / 4 = 3", { size: 13, color: "ink" });
          G.label(372, 140, "t = 13 % 4 = 1", { size: 13, color: "ink" });
          G.label(372, 170, "c0, c1 → row 3, cols 2, 3", { size: 13, color: "ink" });
          G.label(372, 190, "c2, c3 → row 11, cols 2, 3", { size: 13, color: "ink" });
          G.label(372, 230, "each cell: the lane that owns it", { size: 12 });
          G.label(372, 250, "a lane owns 2 adjacent columns", { size: 12 });
          G.label(372, 270, "in 2 rows, 8 apart: 4 floats", { size: 12 });
          G.label(372, 310, "32 lanes × 4 = 128 = 16 × 8", { size: 13, color: "ok" });
          G.from(cells, { opacity: 0, stagger: 0.002, duration: 0.2 });
          G.caption("lane = 4·(row mod 8) + column / 2");
        } },

      { rail: "WMMA", title: "WMMA: use fragments without knowing the layout",
        body: `<p>CUDA's first, friendliest interface is <b>WMMA</b> (<code>nvcuda::wmma</code>, sm_70+). A <code>fragment</code> is an opaque object: you do not know which lane holds what, and you do not need to. Three calls cover everything:</p>
<div class="eq">// all in namespace nvcuda::wmma
fragment&lt;matrix_a, 16,16,16,
         half, row_major&gt; fa;
load_matrix_sync(fa, ptr, ldm);  // smem→regs
mma_sync(acc, fa, fb, acc);      // acc += fa·fb
store_matrix_sync(C, acc, ldm, mem_row_major);</div>
<p>The shape here is 16×16×16 (fp16 in, fp32 accumulator). The course kernel <code>hgemm_wmma</code> uses the same scheme as P5.6's rung 5, with fragments playing the role of register tiles. A block computes a 128×128 tile of C with 8 warps; each warp owns a 64×32 piece, which is <b>4 × 2 = 8 accumulator fragments</b>. For each k step of 16, a warp loads 4 A fragments and 2 B fragments and does 8 <code>mma_sync</code>:</p>
<div class="eq">loads per k16 step : 4 + 2 = 6 fragments
mma_sync per step  : 4 × 2 = 8 (8,192 FLOPs each)
block tile, BK = 32:
  8 warps × 2 steps × 8 mma × 8,192
  = 1,048,576 FLOPs = 2·128·128·32  ✓</div>
<p>Each A fragment feeds 2 multiplies and each B fragment feeds 4: the outer-product reuse of P5.6, now per fragment. The rules that bite: the pointer given to <code>load_matrix_sync</code> must be <b>32-byte aligned</b> and <code>ldm</code> a multiple of 8 halves (16 bytes). That is why the shared tiles in <code>hgemm.cuh</code> are declared <code>__align__(32)</code> and padded by 8 halves.</p>`,
        scene(G) {
          G.text(24, 34, "one warp's 64×32 tile of C = 4 × 2 fragments", { size: 13 });
          const x0 = 150, y0 = 110, w = 110, h = 52;
          const fb = [0, 1].map((j) => G.box(x0 + j * w, y0 - 54, w - 8, 36, "fb[" + j + "]  16×16", { fill: "k", size: 12 }));
          const fa = [0, 1, 2, 3].map((i) => G.box(x0 - 112, y0 + i * h, 98, h - 8, "fa[" + i + "]  16×16", { fill: "q", size: 12 }));
          const acc = [];
          for (let i = 0; i < 4; i++) for (let j = 0; j < 2; j++) acc.push(G.box(x0 + j * w, y0 + i * h, w - 8, h - 8, `acc[${i}][${j}]`, { fill: "ok", size: 12 }));
          G.label(x0 + 4, y0 + 4 * h + 14, "8 fp32 accumulator fragments, kept across all of K", { size: 11 });
          G.label(400, 120, "per k16 step", { size: 14 });
          G.label(400, 144, "6 fragment loads", { size: 13, color: "ink" });
          G.label(400, 164, "8 mma_sync", { size: 13, color: "ink" });
          G.label(400, 184, "each fa → 2 mma, each fb → 4", { size: 13, color: "ink" });
          G.label(400, 224, "fragment pointers must be", { size: 12, color: "hot" });
          G.label(400, 242, "32-byte aligned; ldm % 8 = 0", { size: 12, color: "hot" });
          G.label(400, 282, "128×128 block, 8 warps, BK = 32:", { size: 12 });
          G.label(400, 300, "1,048,576 FLOPs per k-tile", { size: 12 });
          G.from(acc, { opacity: 0, stagger: 0.05, duration: 0.25, delay: 0.3 });
          G.from(fa.concat(fb), { opacity: 0, duration: 0.3 });
          G.caption("the register-blocking of rung 5, with fragments as the registers");
        } },

      { rail: "mma.sync", title: "mma.sync and ldmatrix: you own the layout",
        body: `<p>WMMA hides the layout, and with it some speed and all control. The next level down is the PTX instruction itself, <code>mma.sync.aligned.m16n8k16</code> (sm_80+), with you moving the data into the exact registers. Shared memory to registers needs a special load, <b>ldmatrix</b>: one warp-wide instruction in which each lane supplies one 16-byte row address, and the hardware deals the 8×8 pieces out to the lanes in fragment order.</p>
<ul><li><b>A</b> (16×16, row-major in smem): <code>ldmatrix.x4</code>. Lane <code>l</code> gives the address of row <code>l % 16</code>, column <code>8·(l / 16)</code>.</li>
<li><b>B</b> (16×8) is stored row-major as K×N, but the mma wants it "column" style. <code>ldmatrix.x2.trans</code> transposes each 8×8 piece on the way in, so no separate transpose is needed.</li></ul>
<p><b>Bank conflicts.</b> One ldmatrix reads 8 rows of 16 bytes (4 banks wide each). From P5.2: those 8 reads should land in 8 different bank groups.</p>
<div class="eq">A tile, BK = 32 halves = 16 words per row
no pad: row r starts at word 16·r
  → bank 0 or 16: rows 0,2,4,6 collide
  (4-way conflict on banks 0–3)
pad 8 halves: 20 words per row (80 B)
  start banks 0,20,8,28,16,4,24,12
  each 4 wide → 8 distinct groups ✓</div>
<p>That is why <code>HPAD = 8</code> in <code>d4/hgemm.cuh</code>. The B tile has 128 + 8 = 136 halves = 68 words per row, which is 4 mod 32, and so also spreads the 8 rows over distinct banks. The payoff of all this work: the same block and warp tiles as WMMA, but with explicit loads, so you can later reorder, share and pipeline them. With the same tiles, expect WMMA and <code>mma.sync</code> to perform alike (the compiler emits similar SASS, exercise 2).</p>`,
        check: { q: "Why does the course pad shared-memory tile rows by 8 halves (16 bytes) before using ldmatrix?",
          options: ["ldmatrix needs 80-byte rows", "A power-of-two row stride makes the 8 rows of one ldmatrix hit the same banks; the pad spreads them over distinct banks", "To align the pointer to 32 bytes"], answer: 1,
          why: "A 64-byte row stride means rows 0, 2, 4 and 6 all start at bank 0, a 4-way conflict. With 80-byte rows the 8 row starts fall at banks 0, 20, 8, 28, 16, 4, 24, 12, each 4 banks wide, so all 32 banks are used once." },
        scene(G) {
          const cw = 17, x0 = 40;
          const strip = (y, title, fillOf, textOf, colOf) => {
            G.text(24, y - 10, title, { size: 13 });
            const out = [];
            for (let b = 0; b < 32; b++) {
              const f = fillOf(b);
              out.push(G.rect(x0 + b * cw, y, cw - 2, 30, { fill: f, rx: 2, opacity: f === "line" ? 0.5 : 1 }));
              const t = textOf(b);
              if (t) G.text(x0 + b * cw + cw / 2 - 1, y + 20, t, { anchor: "middle", size: 11, color: "bg" });
            }
            for (let b = 0; b < 32; b += 4) G.label(x0 + b * cw, y + 46, String(b), { size: 10 });
            return out;
          };
          const cnt = (b, stride) => { let n = 0; for (let r = 0; r < 8; r++) { const s = (stride * r) % 32; if (b >= s && b < s + 4) n++; } return n; };
          const s1 = strip(76, "no pad: stride 16 words → bank 0 or 16", (b) => (cnt(b, 16) > 1 ? "hot" : "line"), (b) => (cnt(b, 16) > 1 ? String(cnt(b, 16)) : ""));
          G.label(x0, 142, "numbers = rows hitting that bank: 4-way conflict on banks 0–3 and 16–19", { size: 12, color: "hot" });
          const pal = ["k", "q", "ok", "v", "blue", "pink", "w", "k"];
          const owner = (b) => { for (let r = 0; r < 8; r++) { const s = (20 * r) % 32; if (b >= s && b < s + 4) return r; } return -1; };
          const s2 = strip(206, "pad 8 halves: stride 20 words → 8 distinct groups", (b) => pal[owner(b)], (b) => (b % 4 === 1 ? String(owner(b)) : ""));
          G.label(x0, 272, "digit = which of the 8 ldmatrix rows reads that bank group: all 32 banks, once", { size: 12, color: "ok" });
          G.rect(24, 300, 592, 90, { stroke: "line", rx: 8 });
          G.label(40, 326, "ldmatrix.x4: lane l → row l % 16, column 8·(l / 16) of the A tile", { size: 12, color: "ink" });
          G.label(40, 348, "ldmatrix.x2.trans: row-major B tile → column fragment, no transpose pass", { size: 12, color: "ink" });
          G.label(40, 370, "pad 16 bytes per row: conflict-free reads (HPAD = 8 in d4/hgemm.cuh)", { size: 12, color: "ok" });
          G.from(s1.concat(s2), { opacity: 0, stagger: 0.01, duration: 0.2 });
          G.caption("eight 16-byte row reads per ldmatrix: spread them over the 32 banks");
        } },

      { rail: "feeding", title: "The tensor core is faster than memory",
        body: `<p>Back to the roofline from P5.6. A GEMM is compute-bound when its FLOPs per DRAM byte exceed the <b>ridge point</b> (peak FLOP/s ÷ bandwidth). Tensor cores raise the peak 4× to 16×, so they raise the ridge by the same factor.</p>
<div class="eq">L4 (example values, UNVERIFIED):
  fp32 FMA ridge   30.3e12 / 300e9 ≈ 101 FLOP/B
  fp16 tensor ridge 121e12 / 300e9 ≈ 403 FLOP/B</div>
<p>What does a block tile deliver? Per k step it loads (BM + BN) · BK halves (2 bytes each) and does 2 · BM · BN · BK FLOPs:</p>
<div class="eq">intensity = 2·BM·BN·BK / (2·(BM + BN)·BK)
          = BM·BN / (BM + BN)   FLOP per byte (fp16)

 64× 64 →  32      128×128 →  64
256×128 →  85      256×256 → 128
square T×T tile: T/2 → needs T ≥ 806 to reach 403</div>
<p>Even a 256×256 tile, bigger than shared memory comfortably allows, delivers 128 FLOP per DRAM byte, a third of the ridge. Read that carefully: counted against DRAM alone, a tensor-core GEMM would be memory-bound. Real kernels reach high speed only because neighbouring blocks read the <i>same</i> panels of A and B, and the <b>L2 cache</b> serves most of those re-reads (the L4's L2 is 48 MB in the spec file). So the job changes: it is no longer "make the math fast" but <b>keep the tensor core fed</b>: large tiles, L2-friendly block ordering, vectorized conflict-free shared-memory reads, and loads that overlap the maths. The next step is the last of those.</p>`,
        scene(G) {
          const T = [[64, 64], [128, 128], [256, 128], [256, 256]];
          const ai = T.map(([a, b]) => tileAI16(a, b)), max = 450, base = 330, h = 240;
          G.text(24, 34, "DRAM FLOP per byte of an fp16 block tile (L4 example)", { size: 13 });
          const b = G.bars(60, base, ai, { w: 80, gap: 36, h, max, fill: (i, v) => "k" });
          ai.forEach((v, i) => { G.text(100 + i * 116, base - (h * v) / max - 8, v.toFixed(0), { anchor: "middle", size: 14 }); G.label(100 + i * 116, base + 18, `${T[i][0]}×${T[i][1]}`, { anchor: "middle", size: 12 }); });
          const ry = base - (h * 403) / max;
          G.line(40, ry, 560, ry, { color: "hot", dash: "6 4", w: 2 });
          G.label(300, ry - 8, "fp16 tensor ridge ≈ 403 FLOP/B", { size: 12, color: "hot" });
          const fy = base - (h * 101) / max;
          G.line(40, fy, 560, fy, { color: "muted", dash: "3 4" });
          G.label(44, fy - 6, "fp32 FMA ridge ≈ 101", { size: 12 });
          G.label(24, 372, "all tiles sit left of the ridge: only L2 reuse across blocks", { size: 12, color: "ink" });
          G.label(24, 392, "makes tensor-core GEMMs fast. Feeding is the problem.", { size: 12, color: "ink" });
          G.from(b, { attr: { height: 0, y: base }, duration: 0.5, stagger: 0.08 });
          G.caption("tensor cores move the ridge right; data movement is the whole game");
        } },

      { rail: "cp.async", title: "cp.async: load the next tile while computing this one",
        body: `<p>The synchronous loop of P5.6 (load tile → sync → compute → sync) idles the tensor core during every load. Rung 7 hid it with registers and double buffering. Ampere (sm_80) adds <b><code>cp.async</code></b>: copy 16 bytes from global to shared memory <i>without going through registers</i>, asynchronously. You <b>commit</b> a batch of copies as a group, and later <b>wait</b> until only N groups are still pending.</p>
<p>A pipeline with S stages keeps S shared-memory buffers. The kernel <code>hgemm_mma_async&lt;S&gt;</code>:</p>
<div class="eq">prologue: issue tiles 0..S-2, commit each
for t in 0 .. tiles-1:
    wait_group(S-2)    // tile t landed
    __syncthreads()    // visible; buffer
                       // (t-1)%S now free
    issue tile t+S-1 → buffer (t+S-1)%S
    commit
    compute tile t from buffer t%S</div>
<p>With S = 3 that is 2 tiles always in flight while one is being used. The <code>__syncthreads()</code> after the wait does two jobs: it makes the landed tile visible to every warp, and it proves every warp finished computing tile t-1, so its buffer can be overwritten. Commit even when there is nothing left to load, so the wait counts stay consistent.</p>
<div class="eq">shared memory per stage (128×128, BK = 32, +8 pad):
  A: 128 × 40 halves = 10,240 B
  B:  32 × 136 halves =  8,704 B   → 18,944 B
  S = 2: 37,888 B   S = 3: 56,832 B   S = 4: 75,776 B</div>
<p>More stages hide more latency but cost shared memory. At 3 stages the block needs 56,832 B, above the 48 KiB default, so the host code opts in with <code>cudaFuncSetAttribute</code>, and on an SM with 100 KiB (example value for sm_89, UNVERIFIED) only one such block fits. That is the trade that exercise 3 measures.</p>`,
        check: { q: "In the S-stage pipeline, what does cp.async.wait_group(S-2) guarantee?",
          options: ["Every copy ever issued has finished", "At most S-2 groups are still pending, so the oldest group (tile t) has completed for this thread", "The tensor cores are idle"], answer: 1,
          why: "Groups complete in order. With S - 1 tiles in flight, allowing S - 2 pending means the oldest, tile t, is done. This only covers the thread's own copies, so a __syncthreads() follows to make the tile visible to the whole block." },
        scene(G) {
          const S = 3, iters = 6, x0 = 90, y0 = 90, cw = 86, ch = 48;
          G.text(24, 32, "S = 3 buffers: what each holds at iteration t", { size: 13 });
          for (let t = 0; t < iters; t++) G.label(x0 + t * cw + cw / 2 - 4, y0 - 12, "t = " + t, { anchor: "middle", size: 12 });
          const cells = [];
          for (let b = 0; b < S; b++) {
            G.label(24, y0 + b * ch + 28, "buffer " + b, { size: 12 });
            for (let t = 0; t < iters; t++) {
              const off = (((b - t) % S) + S) % S, tile = t + off;
              const col = off === 0 ? "ok" : off === 1 ? "k" : "v";
              cells.push(G.box(x0 + t * cw, y0 + b * ch, cw - 6, ch - 8, "T" + tile, { fill: col, size: 14, rx: 6 }));
            }
          }
          G.rect(24, 264, 592, 128, { stroke: "line", rx: 8 });
          G.box(40, 278, 20, 14, "", { fill: "ok", rx: 3 }); G.label(68, 290, "computing now (tile t)", { size: 12, color: "ink" });
          G.box(252, 278, 20, 14, "", { fill: "k", rx: 3 }); G.label(280, 290, "in flight (issued t-1)", { size: 12, color: "ink" });
          G.box(486, 278, 20, 14, "", { fill: "v", rx: 3 }); G.label(514, 290, "issued now", { size: 12, color: "ink" });
          G.label(40, 322, "iteration t: wait_group(1) → tile t landed → sync", { size: 12, color: "ink" });
          G.label(40, 342, "then issue tile t+2 into buffer (t+2) % 3 = (t-1) % 3", { size: 12, color: "ink" });
          G.label(40, 362, "(the buffer computed last iteration), then compute tile t", { size: 12, color: "ink" });
          G.label(40, 382, "3 stages × 18,944 B = 56,832 B of shared memory", { size: 12, color: "v" });
          G.from(cells, { opacity: 0, stagger: 0.02, duration: 0.2 });
          G.caption("two tiles always in flight; memory latency hides behind the mma");
        } },

      { rail: "testing", title: "Testing: rounding is tiny, layout bugs are huge",
        body: `<p>How do you know a kernel with hand-placed fragments is right? Compare with cuBLAS, but pick the tolerance by reasoning, not by trial. fp16 inputs convert to fp32 exactly, each product is exact in fp32 (step 2), and only the fp32 <i>additions</i> round, each by at most u = 2<sup>-24</sup> ≈ 6.0e-8 relative. The tensor core adds in an order the documentation does not specify, so you cannot match cuBLAS bit for bit, but you can bound the difference:</p>
<div class="eq">K = 4096, u = 2^-24 ≈ 6.0e-8

worst case (K roundings in one direction):
  K · u        ≈ 2.4e-4  of |C|
typical (errors cancel, random walk):
  sqrt(K) · u  ≈ 3.8e-6  of |C|

test tolerance 2e-3: 8× above the worst case</div>
<p>The tolerance is deliberately generous because the accumulation order is unspecified. But notice what it does <i>not</i> hide: a wrong fragment layout permutes or mixes elements, and the error is of order 1 (100% of |C|), hundreds of times above the tolerance. So a loose rtol is safe here, and any test failure is a real bug, never "just rounding".</p>
<p>Common ways to fail, all silent until the test runs: misaligned WMMA pointers (need 32 B), <code>ldm</code> not a multiple of 8, ldmatrix addresses taken from the wrong lanes (a quietly permuted fragment), forgetting <code>cp.async.commit_group</code> on iterations with no load (the wait counts drift), and comparing an fp16-accumulate kernel with an fp32-accumulate cuBLAS call. Match the compute type.</p>
<p>The exit check of this module (L5) is a tensor-core HGEMM at ≥ 50% of cuBLAS at N = 4096. It is a ratio against a measurement, not a hardware peak, so it is meaningful on whatever GPU you have.</p>`,
        check: { q: "The HGEMM test tolerance is 2e-3, much looser than the expected rounding error. Can it hide a fragment-layout bug?",
          options: ["Yes: any bug under 2e-3 slips through, and layout bugs are small", "No: a layout bug scrambles elements, giving errors of order 1, far above 2e-3", "Only on small matrices"], answer: 1,
          why: "Rounding differences are around 1e-5 or less relative to |C|. A wrong lane mapping puts wrong numbers into outputs, which changes results by amounts comparable to the values themselves (order 1). The gap between 1e-5 and 1 is so wide that 2e-3 is safe on both sides." },
        scene(G) {
          const x0 = 150, w = 440;
          const lx = (v) => x0 + (w * (Math.log10(v) + 7)) / 7;
          G.text(24, 34, "error relative to |C| (log scale), K = 4096", { size: 13 });
          const rows = [["typical", 3.8e-6, "ok", "3.8e-6"], ["worst case", 2.4e-4, "v", "2.4e-4"], ["test tolerance", 2e-3, "k", "2e-3"], ["layout bug", 1, "hot", "≈ 1"]];
          const bars = rows.map(([name, v, col, lbl], i) => {
            const y = 70 + i * 62;
            G.label(24, y + 18, name, { size: 12, color: "ink" });
            const r = G.rect(x0, y, lx(v) - x0, 26, { fill: col, rx: 3 });
            G.text(lx(v) + 6, y + 19, lbl, { size: 13 });
            return r;
          });
          [1e-7, 1e-5, 1e-3, 1e-1].forEach((v) => { G.line(lx(v), 62, lx(v), 318, { color: "line" }); G.label(lx(v), 334, "1e" + Math.log10(v), { anchor: "middle", size: 11 }); });
          G.label(24, 372, "a loose tolerance is safe: bugs are orders of magnitude above it", { size: 12, color: "ink" });
          G.label(24, 392, "u = 2^-24 ≈ 6.0e-8 per fp32 add; tensor-core add order is unspecified", { size: 12 });
          G.from(bars, { attr: { width: 0 }, duration: 0.5, stagger: 0.1 });
          G.caption("rounding and a real bug are orders of magnitude apart");
        } },

      { rail: "CuTe", title: "CuTe: a layout is a function from coordinates to offsets",
        body: `<p>Look back at the index arithmetic in the kernels: <code>row * K + col</code>, padded strides, the ldmatrix lane maps. Most tensor-core bugs live in such expressions. <b>CuTe</b> (part of CUTLASS, pinned <code>v4.8.0</code> in SOURCES.md) turns them into an object. A <b>layout</b> is a pair, <code>shape : stride</code>, and it is a function from a logical coordinate to a memory offset:</p>
<div class="eq">offset(coord) = Σ coord_i × stride_i

shape (4,4), stride (4,1)  row-major:
  (2,3) → 2·4 + 3·1 = 11
shape (4,4), stride (1,4)  column-major:
  (2,3) → 2·1 + 3·4 = 14
shape (4,4), stride (5,1)  rows padded to 5:
  (2,3) → 2·5 + 3·1 = 13</div>
<p>Shapes and strides can nest, so one layout can describe "a 128×128 tile of 16×8 mma tiles of 2 values per register". On top of layouts CuTe defines <b>algebra</b>: composing two layouts, dividing a tensor into tiles, partitioning a tile among threads. The MMA atom (for example <code>SM80_16x8x16_F32F16F16F32_TN</code>) bundles the instruction with the layouts of its fragments, and the copy atoms (<code>SM80_CP_ASYNC_CACHEGLOBAL</code>, <code>ldmatrix</code>) do the same for loads. You state what you want (tile sizes, atoms) and the library derives the indexing that you wrote by hand in step 5.</p>
<p>The course does not vendor CUTLASS: exercise 4 (hard) has you clone it at the pinned tag, read <code>01_layout.md</code> to <code>0x_gemm_tutorial.md</code>, and rewrite the HGEMM so that each CuTe abstraction replaces about ten lines of hand-written <code>mma.sync</code> code. The exit check is again ≥ 50% of cuBLAS.</p>`,
        scene(G) {
          const cs = 38, gy = 90;
          const lay = [["row-major", "(4,4):(4,1)", (r, c) => r * 4 + c, 24], ["column-major", "(4,4):(1,4)", (r, c) => r + c * 4, 224], ["padded rows", "(4,4):(5,1)", (r, c) => r * 5 + c, 424]];
          G.text(24, 32, "same 4×4 coordinates, different offsets", { size: 13 });
          const all = [];
          lay.forEach(([name, spec, f, x]) => {
            G.text(x, gy - 38, name, { size: 13, color: "ink" });
            G.label(x, gy - 18, spec, { size: 12 });
            for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
              const hit = r === 2 && c === 3;
              all.push(G.box(x + c * cs, gy + r * cs, cs - 4, cs - 4, String(f(r, c)), { fill: hit ? "ok" : "w", size: 13, rx: 4 }));
            }
          });
          G.label(24, 290, "highlighted: coordinate (2,3) → 11, 14, 13", { size: 13, color: "ok" });
          G.rect(24, 312, 592, 78, { stroke: "line", rx: 8 });
          G.label(40, 338, "layout = shape : stride, offset = Σ coord × stride", { size: 13, color: "ink" });
          G.label(40, 360, "nest shapes → tiles of tiles; algebra → tile, partition, compose", { size: 12, color: "ink" });
          G.label(40, 380, "MMA / copy atoms carry the fragment layouts for you", { size: 12, color: "k" });
          G.from(all, { opacity: 0, stagger: 0.012, duration: 0.2 });
          G.caption("index arithmetic as an object you can compose");
        } },

      { rail: "int8", title: "int8: four multiplies per lane, then tensor cores",
        body: `<p>Quantized serving (P2.5) stores weights, and sometimes activations, as 8-bit integers to move fewer bytes and to use faster integer units. Two hardware paths do the int8 multiply:</p>
<ul><li><b><code>__dp4a</code></b> (sm_61+, regular CUDA cores): one instruction takes two 32-bit registers, each packing four int8 values, multiplies the four pairs and adds all four plus an int32 accumulator. 4 multiply-adds per lane instead of 1: a warp does 32 × 4 = 128.</li>
<li><b>int8 tensor cores</b> (<code>mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32</code>, sm_80+): 16 × 8 × 32 = 4,096 multiply-adds per instruction, twice the fp16 shape, matching the spec sheet's 2× (L4: 242 int8 TOPS vs 121 fp16 TFLOP/s dense, example values, UNVERIFIED). Its fragment layouts differ from fp16's, so look them up in the PTX ISA.</li></ul>
<div class="eq">|int8| ≤ 128 → |product| ≤ 16,384 = 2^14
K = 4096: |sum| ≤ 2^14 · 2^12 = 2^26
2^26 = 67,108,864 &lt; 2^31 (int32 max) ✓</div>
<p>int32 accumulation is therefore safe for K up to 2<sup>16</sup> = 65,536 (a worst-case sum of 2<sup>30</sup>). The course kernel <code>igemm_dp4a</code> stores B <b>transposed</b> (N×K), as quantized weights usually come, so both operands are contiguous along k and <code>dp4a</code> can read 4 packed bytes at a time.</p>
<p><b>Where does the dequantization happen?</b> That is the real design question. In W8A8 (int8 weights and activations) the integer GEMM runs end to end and the int32 result is multiplied by the two scales once, afterwards: <code>C ≈ s_a · s_w · C_int32</code>. In W4A16 (4-bit weights, fp16 activations) there is no int4 × fp16 instruction, so weights are expanded to fp16 in registers just before an ordinary fp16 mma. Exercise 5 asks you to map each serving path to its instruction.</p>`,
        scene(G) {
          const items = [["FFMA (fp32)", 32, "w"], ["dp4a (int8)", 128, "k"], ["mma fp16 m16n8k16", 2048, "ok"], ["mma int8 m16n8k32", 4096, "v"]];
          const wOf = (v) => (360 * Math.log10(v)) / Math.log10(4096);
          G.text(24, 34, "multiply-adds per warp instruction (log scale)", { size: 13 });
          const bars = items.map(([name, v, col], i) => {
            const y = 56 + i * 52;
            G.label(24, y + 14, name, { size: 12, color: "ink" });
            const r = G.rect(196, y, wOf(v), 24, { fill: col, rx: 3 });
            G.text(196 + wOf(v) + 8, y + 18, F.num(v), { size: 13 });
            return r;
          });
          G.text(24, 290, "W8A8: dequantize AFTER the integer GEMM", { size: 13 });
          const f1 = G.row(24, 304, ["int8 A, B", "int32 acc", "× s_a·s_w", "fp16 out"], { w: 132, h: 34, gap: 14, fill: (i) => ["k", "v", "q", "ok"][i], size: 12 });
          G.label(24, 366, "W4A16: dequantize BEFORE: int4 weights → fp16 in registers → fp16 mma", { size: 12, color: "ink" });
          G.label(24, 388, "int32 never overflows: 4096 × 2^14 = 2^26 < 2^31", { size: 12 });
          G.from(bars, { attr: { width: 0 }, duration: 0.5, stagger: 0.08 });
          G.from(f1, { opacity: 0, x: -14, stagger: 0.1, duration: 0.25, delay: 0.4 });
          G.caption("int8 doubles the tensor-core work per instruction; scales apply after");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A tensor core multiplies small tiles in one warp instruction: m16n8k16 is 2,048 multiply-adds, against 32 for an FFMA. Spec-sheet gain over fp32 FMA: about 4× to 16× (example values, UNVERIFIED).</li>
<li>Operands live in per-lane fragments: lane l holds accumulator rows g and g+8 (g = l/4), columns 2t and 2t+1 (t = l%4).</li>
<li>WMMA hides the layout (fragments, load/mma/store, 32-byte alignment). mma.sync + ldmatrix exposes it; 16-byte row padding keeps ldmatrix conflict-free.</li>
<li>Tensor cores raise the roofline ridge (about 403 FLOP/B on the L4 example) so feeding them is the problem: big tiles, L2 reuse, and cp.async pipelines whose stages trade shared memory for hidden latency.</li>
<li>Tolerances can be loose (2e-3) because layout bugs are order 1; CuTe layouts (shape : stride) make index arithmetic composable; int8 uses dp4a or m16n8k32 with int32 accumulation and dequantization afterwards.</li></ul>`,
    sim: {
      title: "Tile and pipeline budget",
      intro: "Choose a GPU's example numbers, a block tile and a pipeline depth. The chart compares the fp32 FMA peak, the fp16 tensor peak and what the tile could sustain if every tile load came from DRAM. The panels show the shared memory the pipeline needs and how many blocks fit on an SM.",
      height: 260,
      controls: [
        { id: "gpu", label: "GPU (example values)", type: "select", value: "l4", options: [["t4", "T4 (sm_75)"], ["l4", "L4 (sm_89)"], ["a100", "A100-80GB (sm_80)"], ["h100", "H100-SXM (sm_90)"]] },
        { id: "bm", label: "BM (block tile rows)", type: "select", value: 128, options: [[64, "64"], [128, "128"], [256, "256"]] },
        { id: "bn", label: "BN (block tile columns)", type: "select", value: 128, options: [[64, "64"], [128, "128"], [256, "256"]] },
        { id: "bk", label: "BK (k depth per stage)", type: "select", value: 32, options: [[16, "16"], [32, "32"], [64, "64"]] },
        { id: "st", label: "pipeline stages (cp.async)", type: "select", value: 3, options: [[1, "1 (synchronous loads)"], [2, "2"], [3, "3"], [4, "4"]] },
      ],
      draw(G, v) {
        const g = GPUS[v.gpu], BM = v.bm, BN = v.bn, BK = v.bk, S = v.st;
        const ai = tileAI16(BM, BN), ridge = (g.tc * 1e12) / (g.bw * 1e9);
        const dram = Math.min(g.tc, (ai * g.bw * 1e9) / 1e12);
        const vals = [["fp32 FMA peak", g.fp32, "w"], ["fp16 tensor peak", g.tc, "ok"], ["tile, DRAM only", dram, ai >= ridge ? "ok" : "hot"]];
        const max = Math.max(...vals.map((x) => x[1])), W = 380, x0 = 170;
        G.text(24, 22, "TFLOP/s (linear)", { size: 11, color: "muted" });
        vals.forEach(([name, val, col], i) => {
          const y = 44 + i * 64, w = Math.max(2, (W * val) / max);
          G.label(24, y + 18, name, { size: 11, color: "ink" });
          G.rect(x0, y, w, 28, { fill: col, rx: 3 });
          G.text(x0 + w + 8, y + 20, val.toFixed(1), { size: 14 });
        });
        G.line(x0, 36, x0, 232, { color: "line" });
        G.label(24, 252, `tile: ${BM}×${BN}×${BK} · ${ai.toFixed(0)} FLOP/B vs ridge ${ridge.toFixed(0)}`, { size: 12, color: "ink" });
        const per = stageBytes(BM, BN, BK), smem = per * S, cap = g.smem * 1024;
        const blocks = Math.floor(cap / smem);
        const asyncOk = g.sm !== "sm_75", stagesOk = S === 1 || asyncOk;
        return [
          { title: "Shared memory", rows: [["bytes per stage (+8 pad)", F.num(per) + " B"], ["stages", String(S)], ["per block", F.num(smem) + " B (" + (smem / 1024).toFixed(1) + " KiB)"], ["static limit 48 KiB", smem > 49152 ? "over: needs opt-in attribute" : "within"], ["SM capacity (example)", g.smem + " KiB"], ["blocks per SM by smem", String(blocks)]],
            gauge: [[Math.min(1, smem / cap), smem > cap ? "hot" : "v"]], gaugeText: `${(smem / 1024).toFixed(1)} of ${g.smem} KiB per SM`,
            chip: [blocks >= 1 && stagesOk, !stagesOk ? "cp.async needs sm_80+ (T4: WMMA only)" : blocks < 1 ? "does not fit in one SM" : blocks === 1 ? "fits: 1 block per SM" : `fits: ${blocks} blocks per SM`] },
          { title: "Tile and roofline", rows: [["intensity BM·BN/(BM+BN)", ai.toFixed(1) + " FLOP/B"], ["fp16 tensor ridge", ridge.toFixed(0) + " FLOP/B"], ["DRAM-only attainable", dram.toFixed(1) + " TFLOP/s"], ["share of tensor peak", ((dram / g.tc) * 100).toFixed(0) + "%"], ["warp tile (8 warps)", `${BM * BN / 8} outputs per warp`], ["tensor peak ÷ fp32 peak", (g.tc / g.fp32).toFixed(1) + "×"]],
            html: `<p class="note">A model with two honest limits. It counts every tile load as a DRAM read; real kernels also draw on the L2 cache, because neighbouring blocks re-read the same panels, so measured speed can be well above the DRAM-only line. It also says nothing about how well the stages hide latency: that is a measurement (exercise 3, ncu long-scoreboard stalls). All peaks and bandwidths are example values from gpu_specs.yaml, UNVERIFIED; the per-SM shared memory sizes are recalled from the CUDA Programming Guide table, UNVERIFIED.</p>` },
        ];
      },
    },
    practice: {
      intro: `Build once per GPU session (see <code>course/P5-cuda-deep-dive/aws-common.md</code>): <code>cmake -S course/P5-cuda-deep-dive -B build/p5 -DCMAKE_CUDA_ARCHITECTURES=native &amp;&amp; cmake --build build/p5 -j</code>. WMMA runs on a T4 (<code>g4dn.xlarge</code>); <code>mma.sync</code>, <code>ldmatrix</code> and <code>cp.async</code> need sm_80+, so use an L4 (<code>g6.xlarge</code>). Add <code>-DS2S_USE_SOLUTIONS=ON</code> (separate build dir) to test against the reference kernels. Measured results are not yet recorded in the repo (<code>TODO(run-on: g6.xlarge)</code>).`,
      items: [
        { title: "WMMA HGEMM", tier: "T2", goal: "fp16 in, fp32 accumulate, 128×128 block tile, 8 warps, each warp 4×2 fragments of 16×16, 16-byte staging loads with 8-half padding. Derive the tolerance. --bench prints % of cuBLAS at N = 4096 (L5 exit check: at least 50%).",
          cmd: "ctest --test-dir build/p5 -R p5.7_01-wmma --output-on-failure" },
        { title: "ldmatrix + mma.sync", tier: "T2 · sm_80+", goal: "Write the fragment loop (ldmatrix.x4 for A, ldmatrix.x2.trans for B) and the accumulator store with g = lane/4, t = lane%4. Compare with WMMA at the same tiles and inspect SASS with cuobjdump -sass.",
          cmd: "ctest --test-dir build/p5 -R p5.7_02-mma-sync --output-on-failure" },
        { title: "2-stage to 3-stage pipeline", tier: "T2 · sm_80+", goal: "Trace which tile each buffer holds per iteration on paper, then bench mma.sync vs 2 and 3 stages at N = 2048, 4096, 8192, and compare the ncu long-scoreboard stall metric. Try 4 stages.",
          cmd: "./build/p5/p5.7_hgemm_bench | tee results/p5.7-hgemm.md" },
        { title: "CuTe HGEMM (hard)", tier: "T2 · sm_80+ · hard", goal: "Clone CUTLASS at the pinned tag, read the CuTe docs and tutorials, then write 04_cute_hgemm.cu for row-major A and B, fp32 C, at least 50% of cuBLAS. Don't commit the clone.",
          cmd: "git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass.git" },
        { title: "INT8 GEMM with dp4a", tier: "T2", goal: "Compare igemm_dp4a with fp32 SGEMM rung 6 at N = 4096, register-block it, then (stretch, sm_80+) use the int8 tensor-core mma. Map W8A8 and W4A16 serving paths to the instructions they use.",
          cmd: "./build/p5/p5.7_int8_gemm && ctest --test-dir build/d4 -R hgemm --output-on-failure" },
      ],
      labs: [
        { label: "Four tensor-core HGEMMs vs cuBLAS GemmEx (WMMA, mma.sync, cp.async x2, x3)", path: "course/P5-cuda-deep-dive/P5.7-tensor-cores/examples/hgemm_bench.cu" },
        { label: "int8 GEMM with __dp4a vs fp32 SGEMM", path: "course/P5-cuda-deep-dive/P5.7-tensor-cores/examples/int8_gemm.cu" },
        { label: "The guided CuTe path (exercise 4)", path: "course/P5-cuda-deep-dive/P5.7-tensor-cores/examples/cute_hgemm.md" },
        { label: "D4 kernels: hgemm_wmma, hgemm_mma, hgemm_mma_async, igemm_dp4a (read the fragment comments next to warp_mma_tile and store_acc)", path: "platform/kernels/include/d4/hgemm.cuh" },
        { label: "AWS guide (g6.xlarge, about 3 instance-hours, ncu on the pipeline variants)", path: "course/P5-cuda-deep-dive/P5.7-tensor-cores/aws.md" },
        { label: "Reading: PTX ISA, warp-level matrix multiply-accumulate (mma.m16n8k16 fragment figures), ldmatrix, cp.async", path: "https://docs.nvidia.com/cuda/parallel-thread-execution/" },
        { label: "Reading: CUDA Programming Guide, Warp Matrix Functions and Asynchronous Data Copies", path: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/" },
      ],
    },
  });
})();
