/* P5.2 — Coalescing and shared memory: 32-byte sectors, banks, the transpose ladder. */
(function () {
  const F = S2S.fmt;
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  const PAL = ["k", "v", "q", "blue", "pink", "ok"];

  // sectors touched by one warp: lane i reads element (off + i*stride) of eb bytes
  function warpSectors(stride, off, eb) {
    const s = new Map();
    for (let i = 0; i < 32; i++) {
      const sec = Math.floor(((off + i * stride) * eb) / 32);
      if (!s.has(sec)) s.set(sec, s.size);
    }
    return s;
  }

  // Lanes above, 64 floats of memory below (8 per sector). Draws the lane -> address lines and the sectors touched.
  // Returns the number of sectors the whole warp touches.
  function mapping(G, y, off, stride, title, note, noteColor) {
    const x0 = 32, cw = 9, lw = 18;
    const secs = warpSectors(stride, off, 4);
    G.text(24, y, title, { size: 13 });
    const dots = [];
    for (let i = 0; i < 32; i++) {
      const idx = off + i * stride;
      const cx = x0 + (i + 0.5) * lw;
      const col = idx < 64 ? "k" : "line";
      dots.push(G.circle(cx, y + 16, 5, { fill: col }));
      if (idx < 64) G.line(cx, y + 22, x0 + (idx + 0.5) * cw, y + 44, { color: "k", w: 1 });
    }
    const touched = (c) => secs.has(Math.floor(c / 8));
    const read = (c) => (c - off) >= 0 && (c - off) % stride === 0 && (c - off) / stride < 32;
    G.grid(x0, y + 44, 1, 64, cw, 20, (r, c) => (read(c) ? "v" : touched(c) ? "blue" : "line"), { gap: 0 });
    for (let s = 0; s <= 8; s++) G.line(x0 + s * 8 * cw, y + 42, x0 + s * 8 * cw, y + 68, { color: "muted", w: 1 });
    G.label(24, y + 82, note, { size: 12, color: noteColor || "muted" });
    return secs.size;
  }

  S2S.lesson({
    id: "p5-2", n: "P5.2", title: "Coalescing and shared memory",
    subtitle: "CUDA deep dive · first principles · T2, one NVIDIA GPU (sm_75+)",
    kicker: "Lesson · ≈ 50 min",
    headline: "Memory is sold by the sector, not the byte",
    intro: `<p>Most kernels you will write are limited by memory, not arithmetic. This lesson explains how a GPU actually fetches data: in 32-byte sectors, one warp request at a time. You will count sectors by hand, see why a strided access wastes bandwidth, and use the small on-chip <b>shared memory</b> to turn a bad access pattern into a good one. Shared memory has its own trap, <b>bank conflicts</b>, and the lesson ends with the transpose ladder that puts everything together.</p>`,
    facts: ["10 steps", "4 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "lane / used"], ["v", "element read"], ["blue", "sector fetched"], ["line", "untouched"], ["hot", "conflict / waste"], ["ok", "good"]],
    prev: "p5-1", next: "p5-3",
    steps: [
      { rail: "memory-bound", title: "Why memory comes first: most kernels wait on bytes",
        body: `<p>Take the vector add from P5.1: <code>c[i] = a[i] + b[i]</code>. Each thread reads two 4-byte floats and writes one, so it moves <b>12 bytes to do 1 floating-point operation</b>. A GPU can do thousands of operations in the time it takes to move a few bytes from DRAM, so the arithmetic units sit idle waiting for data.</p>
<p>Worked example with <b>example values</b> (T4: 320 GB/s memory bandwidth and 8.1 TFLOP/s fp32 from <code>P1.4 gpu_specs.yaml</code>, status UNVERIFIED: check the datasheet):</p>
<div class="eq">memory : 320e9 B/s / 12 B  = 26.7 G adds/s
compute: 8.1e12 FLOP/s     = 8100 G adds/s
ratio  : 8100 / 26.7       = 304x</div>
<p>So the speed of such a kernel is set by <b>how efficiently it uses memory bandwidth</b>. "Efficiently" has a precise meaning on a GPU, and the next steps build it: what the hardware really fetches when a warp asks for data, and how much of that you actually use.</p>
<p>(P0.4 and P1.4 introduced this as the roofline. Here we look inside the memory system to see what controls the left-hand slope.)</p>`,
        scene(G) {
          G.text(24, 40, "c[i] = a[i] + b[i]  moves 12 B for 1 FLOP", { size: 14 });
          G.box(24, 60, 90, 34, "read a[i]", { fill: "k", size: 12, rx: 5 });
          G.box(122, 60, 90, 34, "read b[i]", { fill: "k", size: 12, rx: 5 });
          G.box(220, 60, 60, 34, "add", { fill: "ok", size: 12, rx: 5 });
          G.box(288, 60, 100, 34, "write c[i]", { fill: "v", size: 12, rx: 5 });
          G.label(400, 82, "4 + 4 + 4 = 12 bytes", { size: 12 });
          G.text(24, 150, "time to move the data vs time to do the maths", { size: 13 });
          G.label(24, 176, "(example values: 320 GB/s, 8.1 TFLOP/s, UNVERIFIED)", { size: 11 });
          G.label(24, 214, "memory", { size: 12 });
          G.rect(90, 200, 520, 22, { fill: "hot", rx: 4 });
          G.label(24, 262, "compute", { size: 12 });
          G.rect(90, 248, 2, 22, { fill: "ok", rx: 1 });
          G.label(100, 264, "1/304 of the memory time", { size: 12, color: "ok" });
          G.text(24, 330, "so: the goal is to use every fetched byte", { size: 14, color: "v" });
          G.label(24, 358, "next: what the hardware really fetches", { size: 12 });
          G.caption("a memory-bound kernel is as fast as its bandwidth use");
        } },

      { rail: "sectors", title: "A warp load fetches whole 32-byte sectors",
        body: `<p>DRAM does not deliver single bytes. The memory system moves data in <b>32-byte sectors</b>, and groups 4 sectors into a <b>128-byte cache line</b>. When a warp executes a load, the hardware looks at the 32 addresses of its 32 lanes, works out which sectors they fall in, and fetches each of those sectors <b>once</b>. Combining the lanes' requests like this is called <b>coalescing</b>.</p>
<p>The best case: 32 lanes read 32 consecutive floats, and the first address is a multiple of 32 bytes (aligned).</p>
<div class="eq">warp bytes needed = 32 lanes x 4 B = 128 B
sector size       = 32 B  (8 floats)
sectors fetched   = 128 / 32 = 4
useful / fetched  = 128 / 128 = 100%</div>
<p>Lane <i>i</i> reads float <i>i</i>; lanes 0-7 share sector 0, lanes 8-15 share sector 1, and so on. No byte is fetched in vain.</p>
<p>This is why the first rule of GPU memory is: <b>neighbouring threads should touch neighbouring addresses</b>. In <code>c[i] = a[i] + b[i]</code> with <code>i = blockIdx.x * blockDim.x + threadIdx.x</code> that happens automatically.</p>`,
        scene(G) {
          mapping(G, 40, 0, 1, "stride 1, aligned: lane i reads float i", "4 sectors x 32 B = 128 B fetched, 128 B useful: 100%", "ok");
          G.text(24, 190, "memory system units", { size: 13 });
          G.box(24, 204, 70, 34, "sector", { fill: "blue", size: 12, rx: 4 });
          G.label(100, 226, "32 B = 8 floats = the smallest fetch", { size: 12 });
          G.rect(24, 252, 4 * 36, 34, { stroke: "muted", rx: 4 });
          for (let s = 0; s < 4; s++) G.box(26 + s * 36, 254, 34, 30, "", { fill: "blue", rx: 3 });
          G.label(24 + 4 * 36 + 8, 274, "cache line = 4 sectors = 128 B", { size: 12 });
          G.label(24, 330, "a warp = 32 lanes x 4 B = 128 B = exactly one line", { size: 12, color: "ink" });
          G.label(24, 354, "(64 floats shown; the other lanes of wider patterns fall to the right)", { size: 11 });
          G.caption("32 lanes, 4 sectors: every fetched byte is used");

        } },

      { rail: "patterns", title: "Count the sectors: offsets and strides",
        body: `<p>Now change the access. The recipe never changes: <b>find the sector of each lane's address, count the distinct sectors, multiply by 32 bytes</b>.</p>
<div class="eq">useful bytes  = 32 lanes x 4 B = 128 B
fetched bytes = (distinct sectors) x 32 B
efficiency    = useful / fetched</div>
<table><tr><th>lane i reads</th><th>sectors</th><th>efficiency</th></tr>
<tr><td><code>in[i]</code>, aligned</td><td>4</td><td>128/128 = 100%</td></tr>
<tr><td><code>in[i + 1]</code> (offset by one float)</td><td>5</td><td>128/160 = 80%</td></tr>
<tr><td><code>in[2*i]</code></td><td>8</td><td>128/256 = 50%</td></tr>
<tr><td><code>in[4*i]</code></td><td>16</td><td>128/512 = 25%</td></tr>
<tr><td><code>in[8*i]</code> or wider</td><td>32</td><td>128/1024 = 12.5%</td></tr></table>
<p>Why 5 for the offset case? The 128 bytes start 4 bytes into a sector, so they spill into a fifth one. Why 32 at stride 8? The lanes are 8 floats = 32 bytes apart, so every lane lands in its own sector, and 28 of the 32 bytes it pulled in are thrown away.</p>
<p>The stride never costs more than 32 sectors per request, because there are only 32 lanes.</p>`,
        scene(G) {
          mapping(G, 34, 1, 1, "offset 1: in[i + 1]", "5 sectors: 128 of 160 B useful = 80%", "v");
          mapping(G, 140, 0, 2, "stride 2: in[2 * i]", "8 sectors: 128 of 256 B useful = 50%", "v");
          mapping(G, 246, 0, 4, "stride 4: in[4 * i]  (lanes 0-15 shown)", "16 sectors: 128 of 512 B = 25%  (stride 8: 32 sectors, 12.5%)", "hot");
          G.label(24, 366, "yellow = element a lane reads; blue = rest of the fetched sectors (wasted)", { size: 12 });
          G.caption("each sector is paid for in full, whether you use 4 bytes of it or 32");
        } },

      { rail: "bandwidth", title: "Sector waste is bandwidth waste",
        body: `<p>Memory moves at a fixed rate in <i>fetched</i> bytes. If only a fraction of the fetched bytes are useful, only that fraction of the peak reaches your program:</p>
<div class="eq">useful GB/s = peak GB/s x (useful / fetched)

example peak 320 GB/s (T4, UNVERIFIED):
  stride 1 : 320 x 1.000 = 320 GB/s
  stride 2 : 320 x 0.500 = 160 GB/s
  stride 4 : 320 x 0.250 =  80 GB/s
  stride 8 : 320 x 0.125 =  40 GB/s</div>
<p>This is an <b>upper bound</b> for a kernel limited by DRAM: caches, the 128-byte line behaviour and other lanes' reuse can soften it, but the direction is firm. <code>examples/01_strided_read.cu</code> measures it: useful GB/s about halves per doubling of the stride until stride 8, then stays flat, because beyond 8 each lane already costs a whole sector.</p>
<p>Vector loads help the other way round. A <code>float4</code> load gives each lane 16 consecutive bytes, so a warp requests 512 bytes (16 sectors) in one instruction: the same 100% efficiency with fewer instructions. It requires a 16-byte aligned address.</p>`,
        check: { q: "A warp loads 32 floats at stride 4 (lane i reads in[4*i]). How many sectors, and what share of the fetched bytes is useful?",
          options: ["16 sectors, 25%", "8 sectors, 50%", "32 sectors, 12.5%"], answer: 0,
          why: "Lanes are 4 floats = 16 bytes apart, so two lanes share each 32-byte sector: 32 lanes / 2 = 16 sectors = 512 bytes fetched. Useful bytes are 32 x 4 = 128, and 128/512 = 25%." },
        scene(G) {
          G.text(24, 36, "useful GB/s from a 320 GB/s peak (example value)", { size: 13 });
          const vals = [320, 160, 80, 40];
          const names = ["stride 1", "stride 2", "stride 4", "stride 8+"];
          G.axes(80, 60, 520, 230, {});
          vals.forEach((v, i) => {
            const h = (v / 320) * 210, x = 110 + i * 120;
            const b = G.rect(x, 290 - h, 70, h, { fill: i === 0 ? "ok" : i < 3 ? "v" : "hot", rx: 4 });
            G.text(x + 35, 284 - h, String(v), { size: 13, anchor: "middle" });
            G.label(x + 35, 310, names[i], { anchor: "middle", size: 12 });
            G.from(b, { scaleY: 0, transformOrigin: "50% 100%", duration: 0.5, delay: i * 0.1 });
          });
          G.label(24, 348, "efficiency: 100% / 50% / 25% / 12.5%", { size: 12 });
          G.label(24, 372, "an upper bound: caches and reuse can soften it, never beat it", { size: 12 });
          G.caption("halving the useful bytes per sector halves the bandwidth you get");
        } },

      { rail: "transpose", title: "A case you cannot avoid: the matrix transpose",
        body: `<p>Sometimes the data layout forces a bad pattern. A <b>transpose</b> writes <code>out[c][r] = in[r][c]</code> for a row-major matrix (each row is contiguous in memory). The natural kernel gives thread (r, c) one element:</p>
<div class="eq">out[c * rows + r] = in[r * cols + c];</div>
<p>Neighbouring threads have consecutive <code>c</code> (they are laid out along x). So the <b>read</b> <code>in[r*cols + c]</code> is contiguous: 4 sectors, 100%. But the <b>write</b> <code>out[c*rows + r]</code> puts consecutive threads <code>rows</code> floats apart. For an 8192 x 8192 matrix that is 32 KiB between neighbours: every lane writes its own sector, 32 sectors per request, 12.5% efficient.</p>
<div class="eq">8192 x 8192 floats = 256 MiB each way
copy moves     : 256 read + 256 write   =  512 MiB
naive transpose: 256 read + 8 x 256 write = 2304 MiB
worst-case traffic ratio: 2304 / 512 = 4.5x</div>
<p>That 4.5x is a pessimistic model, because the L2 cache merges some partial writes; you measure the real number in exercise 2. The fix cannot be "write the other way": whichever side you make contiguous, the other is strided. We need a way to <b>change the order of data between the read and the write</b>. That is what shared memory is for.</p>`,
        scene(G) {
          const cw = 24, n = 8;
          G.text(40, 36, "in (row-major)", { size: 13 }); G.text(350, 36, "out = transpose", { size: 13 });
          G.grid(40, 50, n, n, cw, cw, (r, c) => (r === 2 ? "k" : "line"), { gap: 2 });
          G.grid(350, 50, n, n, cw, cw, (r, c) => (c === 2 ? "hot" : "line"), { gap: 2 });
          G.arrow(40 + n * (cw + 2) + 10, 50 + 2 * (cw + 2) + 12, 340, 50 + 2 * (cw + 2) + 12, { color: "muted", dash: "4 3" });
          G.label(40, 290, "READ row r of in: consecutive addresses", { color: "k", size: 12 });
          G.label(40, 308, "4 sectors per warp request: 100%", { color: "ok", size: 12 });
          G.label(350, 290, "WRITE column r of out: `rows` apart", { color: "hot", size: 12 });
          G.label(350, 308, "32 sectors per request: 12.5%", { color: "hot", size: 12 });
          G.text(24, 350, "rows of 8192 floats: neighbours 32 KiB apart", { size: 13 });
          G.label(24, 376, "one side is always strided unless the data is reordered on chip", { size: 12 });
          G.caption("a row read becomes a column write: contiguous in, scattered out");
        } },

      { rail: "shared memory", title: "Shared memory: a programmer-managed scratchpad",
        body: `<p>Each SM has a small, fast on-chip memory called <b>shared memory</b>, split among the blocks resident on it. Unlike a cache you control it explicitly: you declare an array with <code>__shared__</code>, threads of <b>one block</b> write into it and read from it, and it lives only as long as the block. It is roughly an order of magnitude lower latency than DRAM, and reading it in any order costs the same as long as you avoid bank conflicts (step 7).</p>
<p>That is the tool for the transpose. A block handles one <b>32 x 32 tile</b>:</p>
<ol><li>Read the tile from <code>in</code> with coalesced loads (warp reads one row of 32 floats) and store it into shared memory.</li>
<li><code>__syncthreads()</code>: wait until <i>every</i> thread of the block has finished storing.</li>
<li>Read the tile back <b>column-wise</b> from shared memory and write it to <code>out</code> with coalesced stores (warp writes one row of the output tile).</li></ol>
<div class="eq">__shared__ float tile[32][32];      // 4 KiB
tile[ty][tx] = in[(by*32+ty)*cols + bx*32+tx];
__syncthreads();
out[(bx*32+ty)*rows + by*32+tx] = tile[tx][ty];</div>
<p>The reordering happens <i>inside</i> the SM, where scattered access is cheap, so both global accesses are 100% efficient. The barrier is essential: step 3 reads elements that <i>other</i> threads wrote in step 1.</p>`,
        scene(G) {
          G.text(24, 34, "one block, one 32 x 32 tile (8 x 8 drawn)", { size: 13 });
          const cw = 16;
          G.label(24, 62, "global in", { size: 12 });
          G.grid(24, 70, 8, 8, cw, cw, (r) => (r === 3 ? "k" : "line"), { gap: 1 });
          G.arrow(172, 130, 230, 130, { color: "k", w: 2 });
          G.label(176, 118, "1 load", { size: 11, color: "k" });
          G.label(236, 62, "shared tile", { size: 12 });
          const tl = G.grid(236, 70, 8, 8, cw, cw, (r, c) => (r === 3 || c === 3 ? "q" : "line"), { gap: 1 });
          G.box(236, 220, 140, 30, "__syncthreads()", { stroke: "v", color: "v", size: 12, rx: 5 });
          G.label(236, 270, "2 wait for the whole block", { size: 11, color: "v" });
          G.arrow(384, 130, 442, 130, { color: "q", w: 2 });
          G.label(392, 118, "3 store", { size: 11, color: "q" });
          G.label(448, 62, "global out", { size: 12 });
          G.grid(448, 70, 8, 8, cw, cw, (r) => (r === 3 ? "q" : "line"), { gap: 1 });
          G.label(24, 330, "row 3 goes in as a row; the tile's column 3 comes out as a row", { size: 12, color: "ink" });
          G.label(24, 354, "the transposition happens in shared memory, not in DRAM", { size: 12 });
          G.from(tl, { opacity: 0, stagger: 0.01, duration: 0.2 });
          G.caption("shared memory reorders data so both global accesses are contiguous");
        } },

      { rail: "banks", title: "Shared memory has 32 banks, and lanes can collide",
        body: `<p>Shared memory is fast because it is built from <b>32 independent banks</b>, each 4 bytes wide. Consecutive 4-byte words go to consecutive banks:</p>
<div class="eq">bank of word w = w mod 32
(a float at index w of a __shared__ float array)</div>
<p>In one cycle each bank can serve <b>one word</b>. A warp's 32 accesses finish in a single pass if they land in 32 different banks. If lanes need <i>different</i> words in the same bank, the hardware serves them one after another: a <b>k-way bank conflict</b> takes k passes. (If several lanes read the <i>same</i> word, that is a broadcast and costs one pass.)</p>
<p>Now look at the column read from the last step, <code>tile[tx][k]</code> with <code>tile[32][32]</code>. Lane <code>tx</code> reads word <code>32*tx + k</code>:</p>
<div class="eq">bank = (32 * tx + k) mod 32 = k   for every lane
-> all 32 lanes in bank k, 32 different words
-> 32-way conflict: 32 passes instead of 1</div>
<p>Row reads (<code>tile[k][tx]</code>) are fine: words <code>32k + tx</code> hit banks 0-31. The tile transpose has one access of each kind, and the column one is the slow one.</p>`,
        check: { q: "All 32 lanes of a warp read the same shared-memory word. What does it cost?",
          options: ["A 32-way bank conflict: 32 passes", "One pass: it is a broadcast", "It is undefined behaviour"], answer: 1,
          why: "Conflicts come from different words in the same bank. Many lanes reading one word is served by a single read and broadcast to all of them, so it costs one pass." },
        scene(G) {
          G.text(24, 34, "tile[32][32], lane tx reads tile[tx][k]  (k = 5)", { size: 13 });
          const bx = 40, bw = 17, base = 250, hmax = 190;
          G.axes(bx - 6, 60, 32 * bw + 16, base - 60, {});
          for (let b = 0; b < 32; b++) {
            const n = b === 5 ? 32 : 0;
            if (n) {
              const r = G.rect(bx + b * bw, base - n * 5.6, bw - 3, n * 5.6, { fill: "hot", rx: 2 });
              G.from(r, { scaleY: 0, transformOrigin: "50% 100%", duration: 0.6 });
              G.text(bx + b * bw + 8, base - n * 5.6 - 8, "32 lanes", { size: 12, anchor: "middle", color: "hot" });
            } else G.rect(bx + b * bw, base - 2, bw - 3, 2, { fill: "line" });
          }
          for (let b = 0; b < 32; b += 4) G.label(bx + b * bw + 7, base + 16, String(b), { size: 10, anchor: "middle" });
          G.label(bx, base + 34, "bank number", { size: 11 });
          G.label(430, 74, "lanes per bank", { size: 11 });
          G.text(24, 330, "bank = (32 * tx + 5) mod 32 = 5 for every lane", { size: 13, color: "hot" });
          G.label(24, 356, "32 different words, one bank: 32 passes instead of 1", { size: 12 });
          G.caption("every lane lands in the same bank: the access is serialized");
        } },

      { rail: "padding", title: "The padding trick: [32][33]",
        body: `<p>The conflict comes from the row length (32 words) being a multiple of the bank count (32). Make the row <b>one word longer</b>:</p>
<div class="eq">__shared__ float tile[32][33];   // one unused column

word index of (row i, col k) = 33*i + k
bank = (33*i + k) mod 32 = (i + k) mod 32</div>
<p>For a column read (fixed <i>k</i>, lane <i>i</i> = row), the banks <code>(i + k) mod 32</code> are all different: 32 lanes, 32 banks, one pass. Row reads still work (fixed <i>i</i>, varying <i>k</i> also gives 32 different banks). The cost is 32 extra floats (128 bytes) per tile.</p>
<p>The general rule: if the row pitch is <i>p</i> words and lanes read down a column, the number of conflicting lanes is <code>gcd(p, 32)</code>. p = 32 gives 32, p = 33 gives 1, p = 34 gives 2, p = 48 gives 16. Odd pitches are always conflict-free.</p>
<p>Padding is not the only cure. You can also change <i>which</i> thread reads what (a different index mapping) or XOR-swizzle the column index; exercise 3 asks you to find one without padding. And if a tile is only ever read row-wise, padding fixes nothing and just wastes memory.</p>`,
        check: { q: "A tile is declared float tile[32][34] and a warp reads down a column (lane i reads row i). How many lanes share a bank?",
          options: ["1 (no conflict)", "2 (2-way conflict)", "32 (32-way conflict)"], answer: 1,
          why: "Row pitch p = 34 words, so lane i is in bank (34*i + k) mod 32 = (2*i + k) mod 32. Lanes i and i + 16 give the same bank, so there are 16 banks used twice: gcd(34, 32) = 2 lanes per bank, a 2-way conflict. Only an odd pitch such as 33 avoids it." },
        scene(G) {
          G.text(24, 34, "tile[32][33]: bank = (i + k) mod 32, k = 5", { size: 13 });
          const bx = 40, bw = 17, base = 250;
          G.axes(bx - 6, 60, 32 * bw + 16, base - 60, {});
          const bars = [];
          for (let b = 0; b < 32; b++) bars.push(G.rect(bx + b * bw, base - 5.6, bw - 3, 5.6, { fill: "ok", rx: 2 }));
          for (let b = 0; b < 32; b += 4) G.label(bx + b * bw + 7, base + 16, String(b), { size: 10, anchor: "middle" });
          G.text(bx + 16 * bw, base - 24, "1 lane in every bank", { size: 12, anchor: "middle", color: "ok" });
          G.label(bx, base + 34, "bank number", { size: 11 });
          G.text(24, 316, "pitch p  ->  conflict degree gcd(p, 32)", { size: 13 });
          [[32, 32, "hot"], [33, 1, "ok"], [34, 2, "v"], [48, 16, "v"]].forEach(([p, d, c], i) => {
            G.label(24 + i * 150, 342, `p = ${p}: ${d}-way`, { size: 12, color: c });
          });
          G.label(24, 372, "cost of the fix: 32 extra floats = 128 B per tile", { size: 12 });
          G.from(bars, { scaleY: 0, transformOrigin: "50% 100%", stagger: 0.02, duration: 0.3 });
          G.caption("a one-word shift spreads a column across all 32 banks");
        } },

      { rail: "the ladder", title: "The transpose ladder, rung by rung",
        body: `<p>Put the pieces together. The repo's <code>platform/kernels/include/d4/transpose.cuh</code> has the three rungs; all use a 32 x 32 tile with a (32, 8) block, so each thread moves 4 elements.</p>
<table><tr><th>rung</th><th>global read</th><th>global write</th><th>shared-memory access</th><th>limiter</th></tr>
<tr><td>1 naive</td><td>4 sectors</td><td>32 sectors</td><td>none</td><td>write sectors</td></tr>
<tr><td>2 tile</td><td>4 sectors</td><td>4 sectors</td><td>column read: 32-way conflict</td><td>bank conflicts</td></tr>
<tr><td>3 tile + pad</td><td>4 sectors</td><td>4 sectors</td><td>conflict-free</td><td>about copy bandwidth</td></tr></table>
<p>The <b>exit check</b> for this module: rung 3 reaches at least 80% of the measured copy bandwidth on an 8192 x 8192 matrix. A transpose cannot beat a plain copy (same bytes, plus a rearrangement), so "percent of copy" is the honest yardstick. I have no measured numbers to show here: run it yourself and record them.</p>
<p>About <code>__syncthreads()</code>: it is a barrier for the whole block. Put it between filling and reading the tile. Two classic mistakes: leaving it out (results are often <i>almost always</i> right, which is the worst case, because the bug hides), and calling it inside a branch some threads skip (deadlock or undefined behaviour).</p>`,
        check: { q: "Which statement about __syncthreads() in the shared-tile transpose is correct?",
          options: ["It can be omitted if the block has 32 threads or fewer", "It is needed because a thread writes out elements that other threads loaded", "It makes global memory writes coalesced"], answer: 1,
          why: "After the load phase thread (tx, ty) reads tile[tx][ty + k], an element loaded by a different thread. Without the barrier it may read a slot not yet written. Here the block is 32 x 8 = 256 threads (8 warps), so warps run independently. The barrier orders memory access; coalescing comes from the index mapping." },
        scene(G) {
          G.text(24, 34, "the three rungs: sectors per warp request", { size: 13 });
          const rungs = [["1 naive", [4, 32], 0], ["2 tile", [4, 4], 1], ["3 tile + pad", [4, 4], 2]];
          G.label(220, 62, "read", { size: 11 }); G.label(310, 62, "write", { size: 11 }); G.label(420, 62, "smem column read", { size: 11 });
          rungs.forEach(([nm, [rd, wr], k], i) => {
            const y = 76 + i * 76;
            G.text(24, y + 26, nm, { size: 13 });
            G.box(210, y, 70, 38, rd + " sec", { fill: "ok", size: 12, rx: 5 });
            G.box(300, y, 70, 38, wr + " sec", { fill: wr > 4 ? "hot" : "ok", size: 12, rx: 5 });
            G.box(420, y, 160, 38, k === 0 ? "not used" : k === 1 ? "32-way conflict" : "1 pass", { fill: k === 0 ? "line" : k === 1 ? "hot" : "ok", size: 12, rx: 5 });
          });
          G.label(24, 330, "exit check: rung 3 >= 80% of measured copy GB/s at 8192 x 8192", { size: 12, color: "v" });
          G.label(24, 354, "(copy is the ceiling; a transpose moves the same bytes)", { size: 12 });
          G.label(24, 378, "d4/transpose.cuh: transpose_naive, transpose_smem<0>, <1>", { size: 12, color: "k" });
          G.caption("fix the global pattern first, then the shared-memory pattern");
        } },

      { rail: "more levers", title: "Smaller levers: restrict, __ldg, float4",
        body: `<p>Three more tools, each narrow:</p>
<ul><li><b><code>__restrict__</code></b> promises the compiler that pointers do not alias (no two of them point at overlapping memory). The compiler may then keep values in registers instead of re-loading after every store, and may use the read-only data path. The transpose kernels use it on <code>in</code> and <code>out</code>.</li>
<li><b><code>__ldg(p)</code></b> loads through the read-only data cache explicitly. It helps for gathered, reused data (the vector <code>x</code> in a sparse matrix-vector product), and is mostly implied by <code>const __restrict__</code> now.</li>
<li><b><code>float4</code> loads</b> move 16 bytes per instruction: 4x fewer load instructions for the same bytes. They need a 16-byte aligned address (a <code>cudaMalloc</code> base is aligned, but <code>p + 1</code> is not). They help when the kernel is limited by instruction issue, not already at the bandwidth limit.</li></ul>
<p>Last, a measurement trap: if your test buffer fits in the L2 cache (example values: T4 4 MB, L4 48 MB, UNVERIFIED: check <code>deviceQuery</code>), you are timing L2, not DRAM. Use buffers many times larger than L2, as the examples do (64 MB of useful floats, 8192 x 8192 matrices).</p>
<p>To confirm a sector count instead of guessing, ask the profiler (P5.3 goes deeper): the ratio of <code>l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum</code> to <code>l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum</code> is sectors per request. Your hand count should match it.</p>`,
        scene(G) {
          G.text(24, 34, "float4: 16 B per lane", { size: 14 });
          G.label(24, 58, "scalar: 32 lanes x 4 B, 1 instruction per 128 B", { size: 12 });
          G.grid(24, 66, 1, 32, 17, 20, () => "k", { gap: 1 });
          G.label(24, 118, "float4: 32 lanes x 16 B = 512 B in one instruction (16 sectors)", { size: 12 });
          const f4 = G.grid(24, 126, 1, 32, 17, 20, (r, c) => PAL[Math.floor(c / 8) % 4], { gap: 1 });
          G.label(24, 170, "needs address % 16 == 0", { size: 12, color: "v" });
          G.text(24, 226, "__restrict__ : \"these pointers never overlap\"", { size: 13 });
          G.label(24, 250, "-> compiler keeps values in registers, can skip re-loads", { size: 12 });
          G.text(24, 300, "measure with a buffer >> L2", { size: 13, color: "hot" });
          G.label(24, 324, "small buffers sit in cache and look faster than DRAM really is", { size: 12 });
          G.text(24, 372, "confirm sectors: l1tex__t_sectors / l1tex__t_requests", { size: 12, color: "ok" });
          G.from(f4, { opacity: 0, stagger: 0.01, duration: 0.2 });
          G.caption("fewer instructions for the same bytes, if aligned");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Global memory is fetched in 32-byte sectors (4 per 128-byte line). A warp load costs the number of distinct sectors its 32 addresses touch; efficiency is useful bytes over fetched bytes.</li>
<li>Stride 1 aligned is 4 sectors (100%), offset 1 is 5 (80%), stride 2 is 8 (50%), stride 4 is 16 (25%), stride 8 and wider is 32 (12.5%).</li>
<li>Shared memory reorders data on chip so both global accesses can be contiguous. Put <code>__syncthreads()</code> between the fill and the use of a tile.</li>
<li>Shared memory has 32 banks of 4 bytes. k different words in one bank cost k passes; one word read by many lanes is a free broadcast. A column read of a 32-word-pitch tile is a 32-way conflict, and an odd pitch such as 33 removes it.</li>
<li>The transpose ladder (naive, tile, padded tile) targets at least 80% of copy bandwidth. Confirm patterns with ncu sector and bank-conflict counters.</li></ul>`,
    sim: {
      title: "Sectors and banks calculator",
      intro: "Pick how lanes address global memory (element size, stride, starting offset) and how a shared-memory tile is laid out. The sector count is exact for aligned elements of 2, 4, 8 or 16 bytes. Bandwidth is the upper-bound model from the lesson: peak x useful/fetched. The peak is an example value (T4: 320 GB/s, gpu_specs.yaml, UNVERIFIED), so enter your GPU's number.",
      height: 330,
      controls: [
        { id: "eb", label: "bytes per element", type: "select", value: 4, options: [[2, "2 (half)"], [4, "4 (float)"], [8, "8 (double)"], [16, "16 (float4)"]] },
        { id: "st", label: "stride (elements between lanes)", min: 1, max: 40, step: 1, value: 1 },
        { id: "off", label: "start offset (elements)", min: 0, max: 7, step: 1, value: 0 },
        { id: "peak", label: "peak bandwidth, GB/s (example)", min: 50, max: 1000, step: 10, value: 320 },
        { id: "pitch", label: "tile row pitch, words (column read)", min: 32, max: 48, step: 1, value: 32 },
      ],
      draw(G, v) {
        const secs = warpSectors(v.st, v.off, v.eb);
        const n = secs.size, useful = 32 * v.eb, fetched = n * 32;
        const eff = Math.min(1, useful / fetched), ideal = Math.ceil(useful / 32);
        G.label(10, 16, "the 32 lanes, coloured by the sector they land in", { size: 11 });
        const lw = 18;
        G.grid(10, 24, 1, 32, lw, 26, (r, c) => PAL[secs.get(Math.floor(((v.off + c * v.st) * v.eb) / 32)) % PAL.length], { gap: 1 });
        G.label(10, 74, `distinct sectors: ${n}  (ideal for ${useful} useful bytes: ${ideal})`, { size: 11 });
        const sc = 580 / 32;
        G.rect(10, 84, Math.max(2, ideal * sc), 14, { fill: "ok", rx: 3 });
        G.rect(10, 102, Math.max(2, Math.min(32, n) * sc), 14, { fill: n > ideal ? "hot" : "ok", rx: 3 });
        G.label(Math.max(2, ideal * sc) + 16, 95, "ideal", { size: 10 });
        G.label(Math.max(2, Math.min(32, n) * sc) + 16, 113, "fetched", { size: 10 });
        // banks: lane i reads word pitch*i + 5
        const cnt = new Array(32).fill(0);
        for (let i = 0; i < 32; i++) cnt[(v.pitch * i + 5) % 32]++;
        const deg = Math.max(...cnt);
        G.label(10, 146, `shared-memory column read, pitch ${v.pitch}: lanes per bank`, { size: 11 });
        G.axes(10, 156, 600, 130, {});
        const bw = 600 / 32;
        for (let b = 0; b < 32; b++) {
          const h = cnt[b] * (110 / 32 * (deg > 8 ? 1 : 3));
          if (cnt[b]) G.rect(10 + b * bw + 2, 286 - Math.min(h, 120), bw - 4, Math.min(h, 120), { fill: deg > 1 ? "hot" : "ok", rx: 2 });
        }
        G.label(10, 306, "bank 0 ... 31", { size: 10 });
        G.label(300, 306, `${deg}-way conflict = ${deg} pass${deg > 1 ? "es" : ""}`, { size: 11, color: deg > 1 ? "hot" : "ok" });
        return [
          { title: "Global load", rows: [["sectors per request", String(n)], ["fetched bytes", String(fetched)], ["useful bytes", String(useful)], ["efficiency", F.num(eff * 100, 1) + "%"]],
            gauge: [[eff, eff >= 0.99 ? "ok" : eff >= 0.5 ? "v" : "hot"]], gaugeText: "useful / fetched" },
          { title: "Bandwidth bound", rows: [["peak", F.num(v.peak) + " GB/s"], ["useful GB/s <=", F.num(v.peak * eff, 1)]],
            html: `<p class="note">Upper bound for a DRAM-bound kernel. Measured values (examples/01_strided_read.cu) will differ: caches, line fills and reuse.</p>` },
          { title: "Shared memory", rows: [["pitch", String(v.pitch) + " words"], ["gcd(pitch, 32)", String(gcd(v.pitch, 32))], ["passes per column read", String(deg)]],
            chip: [deg === 1, deg === 1 ? "conflict-free" : `${deg}-way bank conflict`] },
        ];
      },
      note: "Model: warp of 32 lanes, aligned elements, 32-byte sectors, 32 banks of 4 bytes, lane i reads word pitch*i + 5. Real GPUs add details (wider shared-memory modes, L1 line handling); confirm with ncu.",
    },
    practice: {
      intro: "Build once from the repository root (see the shared P5 page): <code>cmake -S course/P5-cuda-deep-dive -B build/p5 -DCMAKE_CUDA_ARCHITECTURES=native &amp;&amp; cmake --build build/p5 -j</code>. Starter tests fail until you implement them. Recommended order: run the three examples, then exercises 1-3, then the hard one.",
      items: [
        { title: "Predict sectors per request, then confirm with ncu", tier: "T2 · easy", goal: "Fill the table for in[i], in[i+1], in[2*i], in[32*i], then measure sectors / requests with ncu and explain the GB/s column of <b>01_strided_read</b> from your counts. Brief: <b>exercises/01-sectors.md</b>.",
          cmd: "sudo $(which ncu) --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./build/p5/p5.2_01_strided_read" },
        { title: "Transpose at 80% of copy bandwidth", tier: "T2 · medium", goal: "Climb the ladder: shared tile with <b>__syncthreads()</b>, then pad to [32][33]. Test is exact on square, tall, wide and non-multiple-of-32 shapes; <b>--bench</b> prints the exit check at 8192 x 8192.",
          cmd: "ctest --test-dir build/p5 -R p5.2_02-transpose --output-on-failure && ./build/p5/p5.2_02-transpose --bench" },
        { title: "Remove the bank conflicts from a planted kernel", tier: "T2 · medium", goal: "Measure the bank-conflict counter, fix it without changing results (padding, then a re-mapping with no padding), and re-measure: it should be about 0.",
          cmd: "sudo $(which ncu) --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./build/p5/p5.2_03-bank-conflicts --bench" },
        { title: "2D convolution with halo tiles", tier: "T2 · hard", goal: "K x K filter (odd K up to 7), zero padding. Naive first, then a (16 + K - 1)^2 shared tile with halo and the filter in <b>__constant__</b>; count global reads per output at K = 7.",
          cmd: "ctest --test-dir build/p5 -R p5.2_04-conv2d-halo --output-on-failure && ./build/p5/p5.2_04-conv2d-halo --bench" },
      ],
      labs: [
        { label: "Examples: strided read, transpose ladder, bank conflicts", path: "course/P5-cuda-deep-dive/P5.2-memory-coalescing-and-shared-memory/examples/" },
        { label: "AWS walkthrough (g4dn.xlarge, about 2 instance-hours; cost, auto-stop and teardown on the shared page)", path: "course/P5-cuda-deep-dive/P5.2-memory-coalescing-and-shared-memory/aws.md" },
        { label: "Shared P5 setup: build, Nsight counter access, teardown", path: "course/P5-cuda-deep-dive/aws-common.md" },
        { label: "D4 transpose ladder (the three rungs)", path: "platform/kernels/include/d4/transpose.cuh" },
        { label: "Standalone animations: coalescing, shared-memory tiling", path: "animations/p5-coalescing.html" },
        { label: "Standalone animation: shared-memory tiling and banks", path: "animations/p5-smem-tiling.html" },
      ],
    },
  });
})();
