/* L2 — Memory and shared memory (Lane B). Concepts behind the L2 LeetGPU problems: the memory hierarchy, coalescing and
   32-byte sectors, tiling through shared memory, barriers, bank conflicts and padding, halos, atomics and privatization,
   occupancy. */
(function () {
  const F = S2S.fmt;
  const LV = "laneB-cuda/L2-memory-and-shared-memory";
  const mono = (h) => h.replace(/<code>(.*?)<\/code>/g, '<span style="font-family:var(--f-mono);font-size:13px">$1</span>');
  const P = (slug, title, diff, prio, goal) => ({
    title: `#${slug.split("-")[0]} ${title}`, tier: `${diff} · ${prio}`,
    goal: mono(`${goal} Hints and reference solution: <code>${LV}/solutions/${slug}/</code>`),
    cmd: `ctest --test-dir build/laneB -R '^L2_${slug}$' --output-on-failure`,
  });
  const cells = (G, x, y, n, w, h, fill, gap = 3) => Array.from({ length: n }, (_, i) => G.rect(x + i * (w + gap), y, w, h, { fill: typeof fill === "function" ? fill(i) : fill, rx: 3 }));

  S2S.lesson({
    id: "l2", n: "L2", title: "Memory and shared memory",
    subtitle: "Lane B · CUDA on LeetGPU · T1 browser or T2 harness",
    kicker: "Lesson · ≈ 55 min",
    headline: "Read it once, use it many times",
    intro: `<p>L1 ended with a fact: simple kernels are <b>memory-bound</b>. Their speed is set by how many bytes move, not by arithmetic. L2 is about moving fewer bytes, and moving them in the pattern the hardware likes.</p>
<p>The dark panel builds this up: the memory hierarchy, why a warp's addresses matter (coalescing), staging data in <b>shared memory</b>, the barrier that makes that safe, bank conflicts and the padding trick, halos for stencils, atomics for counting, and occupancy. The Simulate tab lets you pick a stride and see both the global-memory and the shared-memory cost.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "6 core problems"],
    legend: [["k", "thread / data"], ["v", "highlighted"], ["hot", "wasted or conflicting"], ["ok", "efficient"]],
    prev: "l1", next: "l3",
    steps: [
      { rail: "hierarchy", title: "Memory comes in layers: small and fast, big and slow",
        body: `<p>A GPU's arithmetic units are very fast compared with the memory that feeds them. Hardware designers cannot make everything both big and fast, so they build layers:</p>
<ul><li><b>Registers</b>: private to one thread. The fastest storage. Every local variable lives here when it can.</li>
<li><b>Shared memory</b>: a small on-chip scratchpad that all threads of <i>one block</i> can read and write. You manage it yourself, with <code>__shared__</code>.</li>
<li><b>L2 cache</b>: on the chip, shared by every block. Managed by the hardware.</li>
<li><b>Global memory</b> (DRAM): the big memory your <code>cudaMalloc</code> pointers point into. Largest, slowest, and the only layer the host can copy to.</li></ul>
<p>Going down a layer costs hundreds of clock cycles of latency for global memory versus a few tens for shared memory (order of magnitude; exact figures depend on the GPU, so measure rather than quote). Capacities also differ by GPU and compute capability: read them from the CUDA Programming Guide table or <code>cudaGetDeviceProperties</code>.</p>
<p>The strategy for the whole lesson follows from that picture: <b>fetch each value from global memory once, keep it somewhere fast, and reuse it</b>. Transpose, convolution, tiled matmul and histograms are four ways to do this.</p>
<div class="analogy"><b>Picture it</b>A cook with a small chopping board (shared memory) and a walk-in freezer across the building (global memory). Carrying one carrot per trip is hopeless. Carry a tray, work on the board, go back when you are done.</div>`,
        scene(G) {
          const L = [["registers", "private to one thread", "k", 100], ["shared memory", "one block, you manage it", "v", 170], ["L2 cache", "all blocks, hardware managed", "q", 240], ["global (DRAM)", "everything; host can copy here", "w", 310]];
          const bars = L.map(([n, d, c, w], i) => {
            const y = 56 + i * 76;
            const r = G.rect(24, y, w, 52, { fill: c, rx: 8, opacity: 0.9 });
            G.text(34, y + 24, n, { size: 14, color: "bg", weight: 600 });
            G.label(24 + w + 12, y + 30, d, { size: 12, color: "ink" });
            return r;
          });
          G.arrow(612, 80, 612, 360, { color: "muted", w: 2 });
          G.label(530, 60, "slower, bigger", { size: 11 });
          G.label(24, 384, "(sizes not to scale)", { size: 11 });
          G.text(24, 408, "fetch once → keep it close → reuse it", { size: 14, color: "v" });
          G.from(bars, { scaleX: 0, transformOrigin: "left center", stagger: 0.2, duration: 0.4 });
          G.caption("four layers: each step down is bigger and slower");
        } },

      { rail: "coalescing", title: "A warp reads memory in 32-byte sectors",
        body: `<p>Global memory is not read one float at a time. The memory system hands out data in fixed <b>32-byte sectors</b> (CUDA C++ Best Practices Guide). When a warp executes a load, the hardware looks at the 32 addresses and fetches every sector that any lane touches.</p>
<p>Count it for a warp of 32 lanes each reading one 4-byte float. The warp needs 32 × 4 = 128 useful bytes.</p>
<div class="eq">lane l reads float (l * stride), aligned

stride 1:  bytes 0-127 :  4 sectors,  128 B, 100% used
stride 2:  bytes 0-251 :  8 sectors,  256 B,  50% used
stride 4:  bytes 0-499 : 16 sectors,  512 B,  25% used
stride 8+: 32 sectors, 1024 B,        12.5% used</div>
<p>When consecutive lanes read consecutive addresses the accesses are <b>coalesced</b>: the warp's loads merge into the minimum number of sectors. A stride-8 pattern moves eight times the bytes for the same useful data, and since the kernel is memory-bound, it runs about eight times slower. An address that starts mid-sector costs extra too: 32 consecutive floats shifted by one float straddle 5 sectors instead of 4.</p>
<p>In L1 you learned to map <code>threadIdx.x</code> to the fastest-moving array index. This is the reason.</p>`,
        check: { q: "Each lane of a warp reads one float at index 4 × lane (aligned start). How many 32-byte sectors does the warp fetch, and what fraction of the fetched bytes is used?",
          options: ["4 sectors, 100%", "16 sectors, 25%", "32 sectors, 12.5%"], answer: 1,
          why: "Lane l reads bytes 16·l … 16·l + 3, so the warp spans 31 · 16 + 4 = 500 bytes, which is 16 sectors (512 bytes). It uses 128 of those 512 bytes: 25%. Two lanes share each sector." },
        scene(G) {
          const cases = [["stride 1", 4, 1], ["stride 2", 8, 0.5], ["stride 4", 16, 0.25]];
          G.text(24, 34, "memory as 32-byte sectors (16 shown = 512 B)", { size: 14 });
          const all = [];
          cases.forEach(([nm, n, frac], r) => {
            const y = 70 + r * 96;
            G.label(24, y - 8, `${nm}: ${n} sectors fetched, ${Math.round(frac * 100)}% of fetched bytes used`, { color: "ink", size: 12 });
            for (let s = 0; s < 16; s++) {
              const hit = s < n;
              G.rect(24 + s * 37, y, 34, 50, { stroke: hit ? "k" : "line", rx: 4 });
              if (hit) { all.push(G.rect(24 + s * 37, y, 34, 50, { fill: "hot", rx: 4, opacity: 0.35 })); all.push(G.rect(24 + s * 37, y + 50 * (1 - frac), 34, 50 * frac, { fill: "ok", rx: 4 })); }
            }
          });
          G.rect(24, 370, 12, 12, { fill: "ok", rx: 2 }); G.label(42, 381, "bytes the warp needs", { size: 11 });
          G.rect(200, 370, 12, 12, { fill: "hot", rx: 2, opacity: 0.5 }); G.label(218, 381, "fetched but unused", { size: 11 });
          G.text(24, 412, "stride 8 or more: 32 sectors, 12.5% used", { size: 14, color: "hot" });
          G.from(all, { opacity: 0, stagger: 0.015, duration: 0.2 });
          G.caption("the hardware fetches whole sectors, so scattered lanes waste bandwidth");
        } },

      { rail: "naive transpose", title: "Transpose: the read is coalesced, the write is not",
        body: `<p>Transposing a matrix swaps rows and columns: <code>out[c][r] = in[r][c]</code>. It does no arithmetic at all, so its speed is pure memory behaviour, which makes it the classic test.</p>
<p>Take the obvious kernel, with <code>threadIdx.x</code> mapped to the column <code>c</code> (as L1 taught):</p>
<div class="eq">c = blockIdx.x * blockDim.x + threadIdx.x;
r = blockIdx.y * blockDim.y + threadIdx.y;
out[c * rows + r] = in[r * cols + c];</div>
<p>In one warp the row <code>r</code> is fixed and <code>c</code> runs over 32 consecutive values.</p>
<ul><li><b>Read</b> <code>in[r * cols + c]</code>: 32 consecutive floats, 4 sectors. Perfect.</li>
<li><b>Write</b> <code>out[c * rows + r]</code>: consecutive lanes write addresses <code>rows</code> floats apart. For a 4,096 × 4,096 matrix that is 16,384 bytes apart, so each lane lands in its own sector: 32 sectors, each carrying only 4 useful bytes.</li></ul>
<p>Per warp that is 4 + 32 = 36 sector transactions where 4 + 4 = 8 would do: 4.5 times the memory traffic. Swapping the thread mapping only moves the problem: then the write is coalesced and the read is strided. With a direct mapping, one side is always bad.</p>
<p>The fix is to let the threads of a block <i>exchange</i> data on chip, so that both global accesses are coalesced. That is what shared memory is for.</p>`,
        scene(G) {
          const N = 8, cw = 30;
          G.text(24, 34, "naive: warp (8 lanes drawn) reads a row, writes a column", { size: 13 });
          G.label(24, 62, "input, row-major", { color: "ink" });
          G.label(344, 62, "output, row-major", { color: "ink" });
          G.grid(24, 72, N, N, cw, cw, (r, c) => (r === 2 ? "k" : "line"), { gap: 3, rx: 3 });
          G.grid(344, 72, N, N, cw, cw, (r, c) => (c === 2 ? "hot" : "line"), { gap: 3, rx: 3 });
          G.arrow(278, 150, 336, 150, { color: "v", w: 2 });
          G.label(268, 134, "out[c][r]", { size: 11, color: "v" });
          G.label(24, 320, "read: 8 lanes, 1 row: coalesced", { color: "k", size: 12 });
          G.label(344, 320, "write: 8 lanes, 8 rows: 8 sectors", { color: "hot", size: 12 });
          G.text(24, 392, "per warp: 4 + 32 = 36 sectors, ideal 4 + 4 = 8", { size: 14, color: "v" });
          G.text(24, 416, "4.5× the traffic, from one strided store", { size: 13, color: "hot" });
          G.caption("a transpose always has one strided side, unless it stops in the middle");
        } },

      { rail: "shared tile", title: "Stage a tile in shared memory, then flip it",
        body: `<p>Shared memory is the on-chip scratchpad from step 1. Declare it with <code>__shared__</code> and every thread of the block sees the same array. Use it as a stopover:</p>
<ol><li>The block loads a 32 × 32 tile of the <i>input</i> into shared memory. Threads read along rows, so the global reads are coalesced.</li>
<li>All threads wait until the whole tile has arrived (the barrier, next step).</li>
<li>Each thread reads from the tile <i>transposed</i> and writes along a row of the <i>output</i>. The global writes are coalesced too.</li></ol>
<div class="eq">__shared__ float tile[32][33];
tile[ty][tx] = in[(by*32+ty)*cols + bx*32+tx];
__syncthreads();
out[(bx*32+ty)*rows + by*32+tx] = tile[tx][ty];</div>
<p>Look at the indices. The load puts element (ty, tx) of the input tile at <code>tile[ty][tx]</code>. The store reads <code>tile[tx][ty]</code>: the transposed element. Which output position is that? The block at (bx, by) writes to where block (by, bx) would sit in the transposed matrix, and within it, lane <code>tx</code> writes along the row. That is the whole trick: the shuffle happens in shared memory, where scattered access is cheap, not in DRAM, where it is not.</p>
<p>The reference solution uses 32 × 8 threads per block, so each thread moves 4 elements (a loop over 4 row offsets). A tile of 32 × 33 floats takes 32 · 33 · 4 = 4,224 bytes of shared memory per block. Why 33 and not 32 is step 6.</p>`,
        scene(G) {
          const N = 4, cw = 34, lab = (r, c) => String(r * N + c);
          const panels = [[24, "input tile", "k"], [226, "shared tile", "v"], [428, "output tile", "ok"]];
          panels.forEach(([x, t, col], p) => {
            G.label(x, 56, t, { color: "ink", size: 12 });
            for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
              const v = p === 2 ? c * N + r : r * N + c;       // output tile holds the transpose
              G.box(x + c * (cw + 3), 66 + r * (cw + 3), cw, cw, String(v), { fill: col, size: 12, boxOpacity: 0.85 });
            }
          });
          G.arrow(172, 130, 218, 130, { color: "k", w: 2 }); G.arrow(374, 130, 420, 130, { color: "ok", w: 2 });
          G.label(160, 160, "1 load", { size: 11, color: "k" }); G.label(364, 160, "3 store", { size: 11, color: "ok" });
          G.rect(226, 232, 150, 26, { fill: "hot", rx: 6, opacity: 0.9 });
          G.text(236, 250, "2 __syncthreads()", { size: 13, color: "bg", weight: 600 });
          G.arrow(300, 230, 300, 214, { color: "hot", w: 2 });
          G.text(24, 300, "global read:  in[row][col]   (4 sectors / warp)", { size: 13, color: "k" });
          G.text(24, 326, "shared:       tile[ty][tx] in, tile[tx][ty] out", { size: 13, color: "v" });
          G.text(24, 352, "global write: out[col'][row'] (4 sectors / warp)", { size: 13, color: "ok" });
          G.text(24, 396, "the transpose happens on chip", { size: 15, color: "ink" });
          G.caption("both global accesses are now row-wise, so both coalesce");
        } },

      { rail: "barrier", title: "__syncthreads: wait until everyone has written",
        body: `<p>Threads of a block do not run in lock-step. Warps are scheduled independently, so warp 0 can be several steps ahead of warp 3. In step 4, a thread reads <code>tile[tx][ty]</code>, which was written by <i>a different thread</i>, probably in a different warp. If the reader gets there first, it reads stale data. That is a <b>race</b>.</p>
<p><code>__syncthreads()</code> is a <b>barrier</b>: no thread of the block continues past it until every thread of the block has reached it. It also guarantees that shared-memory writes made before the barrier are visible after it.</p>
<p>Three rules that cause most L2 bugs:</p>
<ul><li><b>Every thread must reach the barrier.</b> A <code>__syncthreads()</code> inside <code>if (i &lt; N)</code> deadlocks or misbehaves for the threads that skip it. Load with a guard, then synchronize <i>outside</i> the guard, as in the transpose kernel above.</li>
<li><b>Barrier after writing, before reading.</b> Tile fully loaded, then used.</li>
<li><b>Barrier after reading, before overwriting.</b> A loop that reuses the same tile needs a second barrier, or a fast warp starts loading the next tile while a slow one is still reading the old one.</li></ul>
<p>The tiled matmul in <code>${LV}/solutions/2-matrix-multiplication-tiled/kernel.cu</code> has exactly that shape: load tile, <code>__syncthreads()</code>, 16 multiply-adds from shared memory, <code>__syncthreads()</code>, next tile.</p>`,
        check: { q: "In a tiled matmul loop that reloads the shared tiles on every iteration, what goes wrong if you delete the SECOND __syncthreads() (the one after the multiply-adds)?",
          options: ["Nothing: the first barrier is enough", "A fast warp can overwrite the tile for iteration t+1 while a slow warp is still reading iteration t's values", "The kernel deadlocks"], answer: 1,
          why: "The first barrier makes the loads visible before use. The second protects the opposite direction: nobody may start storing the next tile until everyone has finished reading the current one. Without it, results are wrong only sometimes, depending on warp timing, which makes the bug hard to find. It does not deadlock, because every thread still reaches the barrier that remains." },
        scene(G) {
          G.text(24, 34, "4 warps load a tile at different speeds", { size: 14 });
          const lens = [150, 260, 110, 200], x0 = 70, bx = 360;
          lens.forEach((w, i) => {
            const y = 60 + i * 50;
            G.label(24, y + 18, `warp ${i}`, { size: 11 });
            G.rect(x0, y, w, 26, { fill: "k", rx: 4 }); G.label(x0 + 8, y + 17, "load", { size: 11, color: "bg" });
            G.rect(x0 + w, y, bx - x0 - w, 26, { stroke: "muted", dash: "3 3", rx: 4 }); if (bx - x0 - w > 60) G.label(x0 + w + 8, y + 17, "wait", { size: 11 });
            G.rect(bx + 6, y, 120, 26, { fill: "ok", rx: 4 }); G.label(bx + 14, y + 17, "read tile", { size: 11, color: "bg" });
          });
          const line = G.line(bx, 52, bx, 264, { color: "hot", w: 3 });
          G.label(bx - 56, 288, "__syncthreads()", { color: "hot", size: 12 });
          G.text(24, 330, "barrier = every warp has finished its writes", { size: 13, color: "ink" });
          G.text(24, 356, "without it, warp 2 could read before warp 1 stored", { size: 13, color: "hot" });
          G.text(24, 394, "loop: load → sync → compute → sync → next tile", { size: 13, color: "v" });
          G.pulse(line, { repeat: 6 });
          G.caption("the barrier line is where the slowest warp sets the pace");
        } },

      { rail: "bank conflicts", title: "Shared memory has 32 banks, and a column read hits one",
        body: `<p>To serve a whole warp per cycle, shared memory is split into <b>32 banks</b>. Consecutive 4-byte words go to consecutive banks: word w lives in bank <code>w mod 32</code>. Each bank can serve one word per cycle. If the 32 lanes of a warp touch 32 different banks, all are served at once. If k lanes touch <i>different words in the same bank</i>, the hardware serves them one after another: a <b>k-way bank conflict</b>, k times slower. (Lanes reading the <i>same</i> word are fine: it is broadcast.)</p>
<p>Now the transpose store reads <code>tile[tx][ty]</code>. For a fixed <code>ty</code>, lane <code>tx</code> reads row <code>tx</code>: a column of the tile.</p>
<div class="eq">tile[32][32]: word = tx * 32 + ty
   bank = (tx * 32 + ty) mod 32 = ty
        (the same bank for every lane)
   → 32-way conflict, 32 passes

tile[32][33]: word = tx * 33 + ty
   bank = (tx + ty) mod 32   (32 different values)
   → conflict-free, 1 pass</div>
<p>Adding one unused column shifts each row by one bank, so a column now walks across all banks. The price is 32 wasted words per tile (4,224 B instead of 4,096 B). This is why the reference solution declares <code>tile[TILE][TILE + PAD]</code> and the bench prints the unpadded variant next to it.</p>`,
        check: { q: "A warp reads one column of __shared__ float t[32][32]: lane l reads t[l][5]. How many passes does the access take, and what does padding to t[32][33] change?",
          options: ["1 pass either way: shared memory is always fast", "32 passes; with padding, 1 pass", "2 passes; with padding, 1 pass"], answer: 1,
          why: "Lane l reads word 32·l + 5, and (32·l + 5) mod 32 = 5 for every lane: all in bank 5, all different words, so 32 serialized passes. With 33 columns the word is 33·l + 5 and the bank is (l + 5) mod 32, which is different for each lane: one pass." },
        scene(G) {
          const bx = 40, bw = 16;
          const lane = (x, y) => Array.from({ length: 32 }, (_, i) => G.circle(x + i * 18 + 8, y, 5, { fill: "k" }));
          const bank = (y, cf) => Array.from({ length: 32 }, (_, b) => G.rect(bx + b * 18, y, bw, 22, { fill: cf(b), rx: 3 }));
          G.text(24, 32, "tile[32][32]: lane l reads tile[l][5]", { size: 13, color: "hot" });
          lane(bx, 56);
          bank(160, (b) => (b === 5 ? "hot" : "line"));
          const A = Array.from({ length: 32 }, (_, i) => G.line(bx + i * 18 + 8, 62, bx + 5 * 18 + 8, 160, { color: "hot", w: 1 }));
          G.label(bx + 5 * 18 - 8, 200, "bank 5: 32-way", { color: "hot", size: 12 });
          G.text(24, 244, "tile[32][33]: bank = (l + 5) mod 32", { size: 13, color: "ok" });
          lane(bx, 268);
          bank(372, (b) => "ok");
          const B = Array.from({ length: 32 }, (_, i) => G.line(bx + i * 18 + 8, 274, bx + ((i + 5) % 32) * 18 + 8, 372, { color: "ok", w: 1 }));
          G.label(bx, 410, "32 lanes → 32 banks: one pass", { color: "ok", size: 12 });
          [0, 8, 16, 24, 31].forEach((b) => { G.label(bx + b * 18 + 2, 150, String(b), { size: 9 }); G.label(bx + b * 18 + 2, 366, String(b), { size: 9 }); });
          G.from(A, { opacity: 0, stagger: 0.02, duration: 0.2 }); G.from(B, { opacity: 0, stagger: 0.02, duration: 0.2, delay: 0.3 });
          G.caption("padding by one word spreads a column across all 32 banks");
        } },

      { rail: "halo", title: "Stencils and convolution: load the halo too",
        body: `<p>A 1D convolution (correlation, as LeetGPU #9 uses it) slides a filter of K weights along the input: <code>out[i] = Σⱼ in[i + j] · k[j]</code> for j = 0 … K − 1. Each output reads K inputs, and neighbouring outputs read almost the same ones. That is reuse, so it is a job for shared memory.</p>
<p>A block that computes 256 outputs, <code>out[base … base + 255]</code>, needs inputs <code>in[base … base + 255 + K − 1]</code>. The extra K − 1 elements past the block's own range are the <b>halo</b>. The block loads this whole span once into shared memory, synchronizes, and then each thread does its K multiply-adds from the tile.</p>
<div class="eq">K = 9, BLOCK = 256
tile loads    = 256 + 9 - 1 = 264
no-tile reads = 256 * 9     = 2304    (8.7x more)
per output: 264/256 = 1.03 loads, versus 9</div>
<p>In 2D the halo goes around all four sides. A 16 × 16 output tile with a 5 × 5 filter needs a (16 + 4) × (16 + 4) = 400-element input tile. Without a tile, 256 outputs each read 25 inputs: 6,400. That is 16 times more requests than 400.</p>
<p>The filter weights are the same for every thread, and every lane reads the same weight at the same time, which is the perfect case for <code>__constant__</code> memory: one value broadcast to the whole warp. The reference solution uses it (<code>${LV}/solutions/9-1d-convolution/kernel.cu</code>), and falls back to plain global reads when K is too large for the constant buffer.</p>
<p>These counts are <i>requests</i>. The L2 and L1 caches absorb part of the repeated reads in practice, so measure before claiming a speed-up.</p>`,
        scene(G) {
          const K = 5, B = 8, cw = 36, x0 = 24;
          G.text(24, 34, "block of 8 outputs, filter K = 5 → halo of 4", { size: 14 });
          G.label(x0, 62, "input array (index below)", { size: 11 });
          const total = B + K - 1;
          const inp = Array.from({ length: total }, (_, i) => G.box(x0 + i * cw * 0.95, 72, cw * 0.9, 36, String(i), { fill: i < B ? "k" : i < total ? "v" : "line", size: 12, boxOpacity: i < total ? 0.9 : 0.4 }));
          G.label(x0, 130, "cyan: the block's own 8 inputs", { size: 11, color: "k" });
          G.label(x0, 146, "amber: the K − 1 = 4 halo inputs, loaded too", { size: 11, color: "v" });
          G.rect(x0 - 3, 68, total * cw * 0.95 + 1, 44, { stroke: "v", dash: "4 3", rx: 6 });
          G.text(24, 190, "shared tile = 8 + 5 − 1 = 12 loads", { size: 14, color: "v" });
          G.label(x0, 226, "outputs of this block", { size: 11 });
          const out = Array.from({ length: B }, (_, i) => G.box(x0 + i * cw * 0.95, 238, cw * 0.9, 36, `o${i}`, { fill: "ok", size: 12 }));
          G.rect(x0 + 3 * cw * 0.95 - 2, 70, K * cw * 0.95 - 2, 40, { stroke: "ok", sw: 2, rx: 4 });
          G.label(x0 + 3 * cw * 0.95 - 2, 290, "o3 reads in[3 … 7] (5 inputs)", { size: 11, color: "ok" });
          G.text(24, 340, "without tile: 8 · 5 = 40 reads", { size: 14, color: "hot" });
          G.text(24, 368, "with tile: 12 loads, then K reads from shared", { size: 14, color: "ok" });
          G.text(24, 404, "with K = 9, 256 outputs: 2304 → 264", { size: 13, color: "ink" });
          G.from(inp, { opacity: 0, y: -8, stagger: 0.04, duration: 0.2 });
          G.caption("neighbouring outputs share inputs: load the span once");
        } },

      { rail: "tiled matmul", title: "Tiled matmul: each tile is loaded once, used 16 times",
        body: `<p>Matrix multiplication is the largest reuse opportunity in this lesson. To compute <code>C = A · B</code> with A of size M × N and B of size N × K, each output element is a dot product of one row of A and one column of B: N multiply-adds. In L1's naive kernel, each thread reads 2N values from global memory for those N multiply-adds, and nearly all of them are also read by other threads.</p>
<p>Tiling fixes this. A block of TS × TS threads owns a TS × TS tile of C. It walks along the shared dimension N in steps of TS:</p>
<ol><li>Each thread loads <i>one</i> element of an A tile and <i>one</i> of a B tile into shared memory (coalesced).</li>
<li><code>__syncthreads()</code></li>
<li>Each thread does TS multiply-adds, reading both tiles from shared memory.</li>
<li><code>__syncthreads()</code>, then the next step.</li></ol>
<div class="eq">TS = 16: per step, a block loads 2 * 256 = 512 values
        and does 256 threads * 16 = 4096 multiply-adds
loads per MAC: 512/4096 = 1/8   (naive: 2)

M = N = K = 1024:
  naive: 2 * 1024^3   = 2,147,483,648 loads
  tiled: that / 16    =   134,217,728 loads</div>
<p>The saving is a factor of TS. A bigger tile saves more, but two 16 × 16 float tiles already take 2 · 256 · 4 = 2,048 bytes of shared memory per block, and a 32 × 32 block would hit the 1,024-thread limit. Real GEMMs add register tiling, vector loads and tensor cores on top: that is L5.</p>
<p>Out-of-range elements (when the shape is not a multiple of TS) are loaded as 0, so the inner loop needs no branches. See <code>${LV}/solutions/2-matrix-multiplication-tiled/</code>.</p>`,
        scene(G) {
          const T = 4, cs = 40, ys = 70;
          const mat = (x, name, hl) => {
            G.label(x, ys - 10, name, { color: "ink", size: 12 });
            for (let r = 0; r < T; r++) for (let c = 0; c < T; c++) G.box(x + c * (cs + 2), ys + r * (cs + 2), cs, cs, "", { fill: hl(r, c) ? hl(r, c) : "line", boxOpacity: hl(r, c) ? 0.9 : 0.35 });
          };
          mat(24, "A (tiles)", (r) => (r === 1 ? "k" : null));
          mat(224, "B (tiles)", (r, c) => (c === 2 ? "v" : null));
          mat(424, "C (tiles)", (r, c) => (r === 1 && c === 2 ? "ok" : null));
          G.text(196, 160, "×", { size: 24, color: "ink" }); G.text(396, 160, "=", { size: 24, color: "ink" });
          for (let t = 0; t < T; t++) {
            G.label(24 + t * (cs + 2) + 14, ys + 1 * (cs + 2) + 24, String(t), { size: 12, color: "bg" });
            G.label(224 + 2 * (cs + 2) + 14, ys + t * (cs + 2) + 24, String(t), { size: 12, color: "bg" });
          }
          G.label(24, 270, "step t = 0, 1, 2, 3: load A tile (1, t) and B tile (t, 2)", { size: 12, color: "ink" });
          G.text(24, 306, "one tile of C = Σ over t of A(1, t) · B(t, 2)", { size: 14, color: "ok" });
          G.text(24, 342, "global loads: naive 2.15 billion", { size: 14, color: "hot" });
          G.text(24, 366, "tiled, TS = 16: 134 million (1024³, requests)", { size: 14, color: "ok" });
          G.text(24, 400, "reuse factor = TS: every loaded value is used TS times", { size: 13, color: "v" });
          G.caption("a block marches along one row of A tiles and one column of B tiles");
        } },

      { rail: "atomics", title: "Atomics count safely, but one counter is a queue",
        body: `<p>A histogram counts how many inputs fall in each bin: <code>hist[x]++</code>. With one thread per input, many threads may update the same bin at the same moment. A plain <code>hist[x]++</code> is read, add, write: two threads can both read 5 and both write 6. One count is lost.</p>
<p>An <b>atomic</b> operation, <code>atomicAdd(&amp;hist[x], 1)</code>, performs the read-modify-write as one indivisible step. It is always correct. The cost: updates to the <i>same address</i> are served one at a time. If a million threads add to one bin, they queue up.</p>
<p>The fix is <b>privatization</b>: give each block its own copy of the histogram in shared memory.</p>
<ol><li>Zero the block's shared histogram, <code>__syncthreads()</code>.</li>
<li>Each thread loops over its inputs (grid-stride) and does <code>atomicAdd(&amp;hs[x], 1)</code> on the <i>shared</i> copy. Shared atomics stay on chip and only compete with the block's own threads.</li>
<li><code>__syncthreads()</code>, then merge: one <code>atomicAdd(&amp;hist[b], hs[b])</code> per bin.</li></ol>
<div class="eq">N = 2^24 = 16,777,216 inputs, 256 bins, 160 blocks
  naive, global atomics:  up to 16,777,216
  privatized:  at most 160 * 256 = 40,960
  ratio: 16,777,216 / 40,960 = 409.6</div>
<p>The kernel launches 4 blocks per SM (160 blocks assumes 40 SMs, as on a T4: example value, check your GPU). Skewed data, where almost all inputs hit one bin, still serializes inside a block; the bench measures both uniform and skewed inputs.</p>
<p><b>Count Array Element</b> (#43) is the one-bin case. Each thread counts matches in a register, a warp adds them with <code>__shfl_down_sync</code> (L3 covers this), and only lane 0 of each warp does one global <code>atomicAdd</code>: with 320 blocks (the reference launches 8 per SM, so 40 SMs gives 320) × 256 threads = 2,560 warps, at most 2,560 atomics for any N.</p>`,
        check: { q: "A histogram has 256 bins; the input has 1,000,000 elements; the kernel runs 100 blocks, each with a shared-memory histogram that is merged once at the end. At most how many global atomicAdd operations happen?",
          options: ["1,000,000", "25,600", "100"], answer: 1,
          why: "Each block does at most one global atomic per bin: 100 blocks × 256 bins = 25,600. The 1,000,000 per-element atomics all hit shared memory instead. The naive version would issue up to 1,000,000 global atomics. It is 100 only if there was a single bin." },
        scene(G) {
          G.text(24, 32, "naive: every thread hits global hist[b]", { size: 13, color: "hot" });
          const th = Array.from({ length: 8 }, (_, i) => G.circle(40 + i * 28, 68, 9, { fill: "k" }));
          G.box(440, 54, 160, 40, "hist[b] (global)", { fill: "hot", size: 12 });
          th.forEach((t, i) => G.line(40 + i * 28 + 8, 74, 440, 78, { color: "hot", w: 1, arrow: true }));
          G.label(236, 108, "N updates queue on the same address", { size: 11, color: "hot" });
          G.text(24, 166, "privatized: shared copy per block, merge once", { size: 13, color: "ok" });
          [0, 1].forEach((b) => {
            const x = 24 + b * 214;
            G.rect(x, 184, 196, 92, { stroke: "k", rx: 8 }); G.label(x + 8, 202, `block ${b}: shared hs[ ]`, { size: 11, color: "k" });
            for (let i = 0; i < 4; i++) G.box(x + 10 + i * 44, 214, 40, 28, String([3, 7, 1, 5][(i + b) % 4]), { fill: "ok", size: 12, boxOpacity: 0.85 });
            for (let i = 0; i < 4; i++) G.circle(x + 30 + i * 44, 256, 5, { fill: "k" });
            
          });
          G.arrow(130, 278, 330, 318, { color: "ok", w: 2 }); G.arrow(344, 278, 340, 318, { color: "ok", w: 2 });
          for (let i = 0; i < 4; i++) G.box(240 + i * 54, 320, 50, 30, `hist[${i}]`, { fill: "w", size: 10 });
          G.label(24, 340, "global", { size: 11 }); G.label(24, 300, "cyan dots: on-chip atomics", { size: 10 });
          G.text(24, 388, "global atomics ≤ blocks × bins", { size: 15, color: "ok" });
          G.text(24, 414, "2^24 inputs: 16.8 M → 40,960 (160 blocks, 256 bins)", { size: 13, color: "v" });
          G.caption("the contention moves from DRAM to on-chip shared memory");
        } },

      { rail: "occupancy", title: "Occupancy: how many warps an SM keeps in flight",
        body: `<p>A global load takes hundreds of cycles to return. The SM hides that wait by switching to another warp that is ready. More resident warps means more chances to find one that is ready. <b>Occupancy</b> is the ratio of warps resident on an SM to the maximum that SM supports.</p>
<p>Resident blocks are limited by whichever resource runs out first: threads per SM, registers per SM, shared memory per SM, or a cap on blocks per SM. Worked example with example limits (sm_75 values for illustration; read your GPU's row in the CUDA Programming Guide compute-capability table): 1,024 threads and 65,536 registers per SM, 64 KB shared memory.</p>
<div class="eq">block = 256 threads
threads: 1024 / 256 = 4 blocks
regs 64:  64 * 256 = 16,384 per block
          65,536 / 16,384 = 4 blocks → 100%
regs 128: 128 * 256 = 32,768 per block
          65,536 / 32,768 = 2 blocks → 50%
smem 24 KB per block: 64 / 24 → 2 blocks → 50%</div>
<p>(Real allocation rounds registers and shared memory up in fixed chunks, so the true answer can be slightly lower. Use the API instead of hand maths: <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code>, and read <i>achieved occupancy</i> in Nsight Compute.)</p>
<p>Higher is not always better. A kernel that keeps many values in registers per thread (register tiling, L5) may be faster at 50% occupancy than a leaner one at 100%. Occupancy is a reason to check, not a target. For these problems: pick a block size that is a multiple of 32 (128 or 256), look at the register count in <code>nvcc --resource-usage</code>, and keep shared memory per block modest.</p>`,
        check: { q: "Example limits: 1,024 threads and 65,536 registers per SM. A kernel uses 128 registers per thread with 256-thread blocks. How many blocks fit on an SM, and what is the occupancy?",
          options: ["4 blocks, 100%", "2 blocks, 50%", "1 block, 25%"], answer: 1,
          why: "One block needs 128 × 256 = 32,768 registers. 65,536 / 32,768 = 2 blocks, which is 512 threads out of 1,024: 50%. The thread limit alone would allow 4 blocks, but registers run out first." },
        scene(G) {
          const rows = [["64 regs/thread", 4, "4 blocks: 100%"], ["128 regs/thread", 2, "2 blocks: 50%"], ["24 KB smem/block", 2, "2 blocks: 50%"]];
          G.text(24, 32, "one SM, 256-thread blocks, example limits", { size: 14 });
          rows.forEach(([nm, n, res], r) => {
            const y = 56 + r * 108;
            G.label(24, y + 8, nm, { color: "ink", size: 12 });
            G.rect(24, y + 16, 410, 60, { stroke: "line", rx: 8 });
            for (let s = 0; s < 4; s++) {
              const on = s < n;
              G.rect(32 + s * 100, y + 24, 92, 44, on ? { fill: "k", rx: 6, opacity: 0.85 } : { stroke: "hot", dash: "4 3", rx: 6 });
              G.label(40 + s * 100, y + 50, on ? `block ${s}` : "no room", { size: 11, color: on ? "bg" : "hot" });
            }
            G.label(446, y + 50, res, { size: 11, color: n === 4 ? "ok" : "v" });
          });
          G.text(24, 392, "the scarcest resource sets the limit", { size: 15, color: "v" });
          G.label(24, 414, "limits: 1,024 threads, 65,536 registers, 64 KB smem (example values)", { size: 11 });
          G.caption("registers or shared memory can cap the blocks well below the thread limit");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Memory is layered; fetch from global memory once and reuse from registers or shared memory.</li>
<li>A warp's load is served in 32-byte sectors. Consecutive addresses coalesce (4 sectors for 128 B); a stride of 8 floats or more costs 32 sectors, 8 times the traffic.</li>
<li>Stage a tile in shared memory so that both global accesses are row-wise. Synchronize after writing and again before overwriting; never inside a divergent branch.</li>
<li>Shared memory has 32 banks. A column read of a 32-wide tile is a 32-way conflict; padding a row to 33 words fixes it.</li>
<li>Stencils load tile plus halo once (264 loads instead of 2,304 for K = 9). Tiled matmul cuts global loads by the tile size.</li>
<li>Privatize atomics per block. Check occupancy with the API; don't chase 100%.</li></ul>
<p>Next, L3 combines many values into one: reductions and scans, where the shared-memory and warp tools from this lesson do the heavy lifting.</p>`,
    sim: {
      title: "Stride explorer: sectors and banks",
      intro: "Lane l of a warp touches element number offset + l × stride of a float array. The same number is used twice. For global memory it is the element stride between lanes, so the chart shows how many 32-byte sectors the warp fetches. For a shared-memory tile it is the row length in words (reading a column), so the bars show how many lanes land on each of the 32 banks. Try stride 1, 2, 8, 32, 33.",
      height: 300,
      controls: [
        { id: "s", label: "stride (floats between lanes / tile row length)", min: 1, max: 64, step: 1, value: 32 },
        { id: "o", label: "start offset (floats)", min: 0, max: 31, step: 1, value: 0 },
      ],
      draw(G, v) {
        const x0 = 32, cw = 18, s = v.s, off = v.o;
        const sec = [], bankWords = Array.from({ length: 32 }, () => new Set());
        for (let l = 0; l < 32; l++) { sec.push(Math.floor(((off + l * s) * 4) / 32)); bankWords[(off + l * s) % 32].add(off + l * s); }
        const uniq = [...new Set(sec)].sort((a, b) => a - b), n = uniq.length;
        const deg = Math.max(...bankWords.map((w) => w.size));
        G.label(x0, 12, "32 lanes of one warp", { size: 11 });
        for (let l = 0; l < 32; l++) G.rect(x0 + l * cw, 18, cw - 3, 16, { fill: "k", rx: 3 });
        G.label(x0, 140, `${n} distinct 32-byte sector${n > 1 ? "s" : ""} fetched (one slot each)`, { size: 11 });
        for (let l = 0; l < 32; l++) { const slot = uniq.indexOf(sec[l]); G.line(x0 + l * cw + 7, 34, x0 + slot * cw + 7, 104, { color: n > 4 ? "hot" : "ok", w: 1 }); }
        for (let i = 0; i < 32; i++) G.rect(x0 + i * cw, 104, cw - 3, 16, i < n ? { fill: n > 4 ? "hot" : "ok", rx: 3, opacity: 0.9 } : { stroke: "line", rx: 3 });
        G.label(x0, 172, "shared memory banks: lanes (different words) per bank", { size: 11 });
        for (let b = 0; b < 32; b++) {
          const c = bankWords[b].size, h = Math.max(c, 0) * 3;
          if (c) G.rect(x0 + b * cw, 280 - h, cw - 3, h, { fill: c > 1 ? "hot" : "ok", rx: 2 });
          else G.rect(x0 + b * cw, 277, cw - 3, 3, { fill: "line", rx: 1 });
          if (b % 4 === 0) G.label(x0 + b * cw + 2, 294, String(b), { size: 9 });
        }
        G.line(x0 - 4, 280, x0 + 32 * cw, 280, { color: "muted", w: 1 });
        const fetched = n * 32, eff = 128 / fetched;
        return [
          { title: "Global memory (float loads)", rows: [["useful bytes", "128"], ["sectors fetched", String(n)], ["bytes fetched", F.num(fetched)], ["efficiency", `${(100 * eff).toFixed(1)}%`]],
            gauge: [[Math.min(1, eff), "ok"], [Math.max(0, 1 - eff), "hot"]], gaugeText: "useful vs fetched",
            chip: [n <= 5, n <= 4 ? "fully coalesced" : n <= 5 ? "misaligned: one extra sector" : `${(n / 4).toFixed(1)}× the ideal traffic`] },
          { title: "Shared memory (same pattern)", rows: [["banks used", String(bankWords.filter((w) => w.size).length)], ["worst bank: lanes", String(deg)], ["passes", String(deg)]],
            chip: [deg === 1, deg === 1 ? "conflict-free" : `${deg}-way bank conflict`],
            html: `<p class="note">Model: 32 banks of 4-byte words, bank = word mod 32. Distinct words in one bank serialize; identical words are broadcast (not modelled here since lanes never share a word at stride ≥ 1). Sector model: aligned 32-byte sectors, 4-byte loads.</p>` },
        ];
      },
      note: "Stride 32 vs 33 shows the transpose padding trick. Stride 1 vs 8 shows coalescing. Offset 1 at stride 1 shows misalignment.",
    },
    practice: {
      intro: `<p>Solve each problem on <a href="https://leetgpu.com/challenges">LeetGPU</a> first (T1, in the browser). We don't reproduce the statements; read them there. Then run the local reference and test (T2, needs an NVIDIA GPU and CUDA ≥ 12.4). Build once from the repository root:</p>
<p><code>cmake -S laneB-cuda -B build/laneB -DCMAKE_BUILD_TYPE=Release &amp;&amp; cmake --build build/laneB -j</code></p>
<p><b>Predict first.</b> Before benchmarking transpose, write down the sectors per warp for the naive write (step 3) and your predicted GB/s ratio to a plain copy. Then run <code>./build/laneB/L2_3-matrix-transpose --bench</code>: it prints naive, tiled without padding, and tiled with padding, each next to a measured device copy. <b>Exit check:</b> padded transpose reaches ≥ 80% of the copy bandwidth; report the table and the <code>ncu</code> sectors/request metric (a <code>TODO(run-on: g4dn.xlarge)</code> in the repo until someone runs it).</p>`,
      items: [
        P("3-matrix-transpose", "Matrix Transpose", "easy", "core · exit check", "Coalesced read and write through a padded shared tile, <code>[32][33]</code>, with a 32 × 8 block."),
        P("9-1d-convolution", "1D Convolution", "easy", "core", "Load tile plus halo into shared memory; filter weights in <code>__constant__</code>."),
        P("10-2d-convolution", "2D Convolution", "medium", "core", "2D tile with a halo on all four sides."),
        P("13-histogramming", "Histogramming", "medium", "core", "Per-block shared histograms with atomics, merged once into the global histogram."),
        P("43-count-array-element", "Count Array Element", "medium", "core", "Count in registers, reduce within the warp, one atomic per warp."),
        P("2-matrix-multiplication-tiled", "Matrix Multiplication (tiled)", "easy", "core", "Revisit L1's naive matmul with two shared tiles and two barriers per step."),
        P("28-gaussian-blur", "Gaussian Blur", "medium", "practice", "Separable versus direct 2D filter: compare the loads."),
        P("42-2d-max-pooling", "2D Max Pooling", "medium", "practice", "A windowed reduction over a 2D tile."),
        P("69-2d-jacobi-stencil", "2D Jacobi Stencil", "medium", "practice", "A four-neighbour stencil with shared-memory reuse."),
        P("44-count-2d-array-element", "Count 2D Array Element", "medium", "practice", "The 2D variant of counting; flatten with row-major indices."),
        P("45-count-3d-array-element", "Count 3D Array Element", "medium", "practice", "The 3D variant; one more stride."),
        P("11-3d-convolution", "3D Convolution", "medium", "stretch", "3D tiling; reuse along z in registers."),
        P("37-matrix-power", "Matrix Power", "medium", "stretch", "Repeated tiled matmul; watch launch overhead."),
      ],
      labs: [
        { label: "Level syllabus, objectives and exit check", path: `${LV}/syllabus.md` },
        { label: "Problem map with links and priorities", path: `${LV}/leetgpu-map.md` },
        { label: "The harness: CUDA_CHECK, DeviceBuffer, time_gpu, measure_copy_gbs", path: "laneB-cuda/harness/" },
        { label: "Coalescing animation (Lane A P5.2)", path: "animations/p5-coalescing.html" },
        { label: "Shared-memory tiling animation (Lane A P5.2)", path: "animations/p5-smem-tiling.html" },
        { label: "AWS T2 box for L1–L4 (T4, sm_75): instance_type = \"g4dn.xlarge\", then make down", path: "infra/aws/single-node" },
      ],
    },
  });
})();
