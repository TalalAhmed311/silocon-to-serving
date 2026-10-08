/* P0.1 — C and C++ for systems work. Memory as bytes, layout, strides, ownership, and the tools that check them. */
(function () {
  const F = S2S.fmt;
  // a run of byte cells: spans = [[nBytes, color, label?], ...], cw px per byte
  const bytes = (G, x, y, spans, cw, h = 34) => {
    const out = [], labs = []; let b = 0;
    spans.forEach(([n, col, lab]) => {
      for (let i = 0; i < n; i++) out.push(G.rect(x + (b + i) * cw, y, cw - 2, h, { fill: col, rx: 3, opacity: col === "hot" ? 0.35 : 0.9 }));
      if (lab) labs.push([x + (b + n / 2) * cw - 1, lab, col, x + b * cw, n]);
      b += n;
    });
    labs.forEach(([lx, lab, col, sx, n]) => G.rect(sx, y, n * cw - 2, h, { fill: col, rx: 3, opacity: col === "hot" ? 0.35 : 0.9 }) && G.text(lx, y + h / 2 + 4, lab, { anchor: "middle", size: 11, weight: 700, color: col === "hot" ? "ink" : "bg" }));
    return out;
  };
  // layout of RequestRecord (exercise 1): as written and sorted by alignment
  const REQ_BAD = [[1, "k"], [7, "hot"], [8, "k"], [2, "k"], [6, "hot"], [8, "k"], [1, "k"], [3, "hot"], [8, "k"], [4, "hot"]];
  const REQ_GOOD = [[8, "k"], [8, "k"], [4, "k"], [4, "k"], [2, "k"], [1, "k"], [1, "k"], [4, "hot"]];

  S2S.lesson({
    id: "p0-1", n: "P0.1", title: "C and C++ for systems work",
    subtitle: "Systems primer · first principles · T0, any laptop",
    kicker: "Lesson · ≈ 45 min",
    headline: "Where exactly is element (i, j)?",
    intro: `<p>A language model is, to a computer, a few gigabytes of numbers laid out in flat arrays. Everything in this course, from a CPU engine to a GPU kernel to a serving system's cache, is a different way of slicing those arrays. This lesson builds the four ideas you need to reason about them: memory as a line of bytes, how the compiler lays out a struct, how one buffer becomes many matrices through <b>strides</b>, and how C++ makes sure every buffer is freed exactly once.</p>
<p>You only need to know how to write a loop and a function in some language. Scroll: the dark panel redraws for each step.</p>`,
    facts: ["11 steps", "5 checkpoints", "1 simulator", "5 exercises"],
    legend: [["k", "data"], ["v", "neighbour"], ["q", "pointer / char"], ["hot", "padding"], ["ok", "allowed"]],
    next: "p0-2",
    steps: [
      { rail: "bytes", title: "Memory is one long line of bytes",
        body: `<p>To the CPU, memory is a numbered line of bytes. Byte 0, byte 1, byte 2, and so on for billions of bytes. The number of a byte is its <b>address</b>. Every variable you create is a run of consecutive bytes somewhere on that line. A <code>float</code> takes 4 bytes; an array of 4 floats takes 16 consecutive bytes.</p>
<p>A <b>pointer</b> is an address plus a type. The type tells the compiler how far to step when you add 1:</p>
<div class="eq">float a[4] = {1, 2, 3, 4};
float *p = a;      // p holds the address of a[0]
p + 1              // address + sizeof(float) = +4 bytes
*(p + 2) == a[2]   // indexing IS pointer arithmetic</div>
<p>If <code>a</code> starts at address <code>0x1000</code>, then <code>a[2]</code> lives at <code>0x1000 + 2 × 4 = 0x1008</code>. In C, <code>a[i]</code> is defined as <code>*(a + i)</code>: there is no hidden table, only arithmetic on addresses.</p>`,
        scene(G) {
          G.label(24, 40, "16 bytes of memory, starting at address 0x1000", { size: 13 });
          const cols = ["k", "v", "k", "v"];
          const cells = [];
          for (let i = 0; i < 16; i++) cells.push(G.rect(24 + i * 37, 90, 34, 40, { fill: cols[i >> 2], rx: 4, opacity: 0.85 }));
          for (let e = 0; e < 4; e++) {
            G.text(24 + e * 148 + 72, 76, `a[${e}] = ${e + 1}.0`, { anchor: "middle", size: 13 });
            G.label(24 + e * 148, 150, "0x" + (0x1000 + e * 4).toString(16), { size: 11 });
          }
          G.label(24, 400, "each colored block of 4 bytes = one float", { size: 12 });
          const ptrs = [["p", 0], ["p + 1", 1], ["p + 2", 2]];
          ptrs.forEach(([s, e], i) => {
            const x = 24 + e * 148 + 18;
            G.box(x - 4, 250 + i * 0, 76, 32, s, { stroke: "q", color: "q", size: 13 });
            G.arrow(x + 34, 250, x + 34, 136, { color: "q", w: 2 });
          });
          G.text(24, 330, "p + 1 = 0x1000 + 1 × sizeof(float) = 0x1004", { size: 14 });
          G.text(24, 360, "*(p + 2) reads the 4 bytes at 0x1008 → 3.0", { size: 14 });
          G.from(cells, { opacity: 0, stagger: 0.03, duration: 0.2 });
          G.caption("a pointer is an address; its type sets the step size");
        } },

      { rail: "alignment", title: "Values like to start at round addresses",
        body: `<p>Memory does not travel to the CPU one byte at a time. It moves in fixed blocks called <b>cache lines</b>, 64 bytes on mainstream x86 and ARM CPUs (Apple's M-series uses 128-byte lines). A line always starts at an address that is a multiple of its size.</p>
<p>Now picture an 8-byte <code>double</code> stored at address 60. Its bytes 60–63 sit in one line and 64–67 in the next, so one read becomes two. Some CPUs refuse such accesses outright.</p>
<p>So every type has an <b>alignment</b>: its address must be a multiple of that number. On the 64-bit ABIs you will meet (x86-64 Linux and macOS, arm64, Windows x64):</p>
<table><tr><th>type</th><th>size</th><th>alignment</th></tr>
<tr><td><code>char</code>, <code>bool</code>, <code>int8_t</code></td><td>1</td><td>1</td></tr>
<tr><td><code>short</code>, <code>uint16_t</code></td><td>2</td><td>2</td></tr>
<tr><td><code>int</code>, <code>float</code>, <code>uint32_t</code></td><td>4</td><td>4</td></tr>
<tr><td><code>double</code>, <code>int64_t</code>, any pointer</td><td>8</td><td>8</td></tr></table>
<p>An aligned value of size ≤ 8 can never straddle a 64-byte line, because 64 is a multiple of its alignment.</p>`,
        scene(G) {
          const px = 4.5;
          G.label(24, 34, "two 64-byte cache lines", { size: 13 });
          G.rect(24, 50, 64 * px - 2, 40, { stroke: "muted", rx: 4 }); G.rect(24 + 64 * px, 50, 64 * px - 2, 40, { stroke: "muted", rx: 4 });
          G.label(28, 106, "line 0: bytes 0–63", { size: 11 }); G.label(28 + 64 * px, 106, "line 1: bytes 64–127", { size: 11 });
          const bad = G.rect(24 + 60 * px, 54, 8 * px, 32, { fill: "hot", rx: 3 });
          const good = G.rect(24 + 88 * px, 54, 8 * px, 32, { fill: "ok", rx: 3 });
          G.text(24 + 64 * px, 132, "double at 60: split over 2 lines", { anchor: "middle", color: "hot", size: 12 });
          G.text(24 + 92 * px, 150, "double at 88: inside one line", { anchor: "middle", color: "ok", size: 12 });
          const rows = [["char", 1], ["short", 2], ["int/float", 4], ["double/ptr", 8]];
          G.label(140, 196, "allowed start addresses 0 … 23 (green)", { size: 12 });
          const cells = [];
          rows.forEach(([n, a], r) => {
            G.text(24, 232 + r * 44, n, { size: 13 });
            for (let b = 0; b < 24; b++) cells.push(G.rect(140 + b * 20, 214 + r * 44, 18, 26, { fill: b % a === 0 ? "ok" : "line", rx: 3, opacity: b % a === 0 ? 0.85 : 0.6 }));
          });
          for (let b = 0; b < 24; b += 4) G.label(140 + b * 20 + 9, 400, String(b), { anchor: "middle", size: 11 });
          G.pulse(bad, { repeat: 5 });
          G.from(cells, { opacity: 0, stagger: 0.004, duration: 0.2 });
          G.caption("alignment keeps every value inside a single cache line");
        } },

      { rail: "padding", title: "The compiler pads structs to keep fields aligned",
        body: `<p>A <code>struct</code> stores its fields in the order you wrote them. If a field would land on a bad address, the compiler inserts unused <b>padding</b> bytes before it.</p>
<div class="eq">struct Bad  { char tag; double value; char flag; };
//  tag  @0, 7 pad, value @8, flag @16, 7 pad → 24 B

struct Good { double value; char tag; char flag; };
//  value @0, tag @8, flag @9, 6 pad          → 16 B</div>
<p>Why the 7 bytes <i>after</i> <code>flag</code> in <code>Bad</code>? Because of arrays. In <code>Bad arr[2]</code>, <code>arr[1]</code> starts at <code>sizeof(Bad)</code>. For its <code>value</code> to be 8-aligned, the size must be a multiple of 8, the struct's largest alignment. That is <b>tail padding</b>.</p>
<p>Same data, 50% more bytes in the bad order. The rule of thumb: <b>order fields from largest alignment to smallest</b>, and check with <code>offsetof</code> and <code>static_assert</code> rather than assuming. <code>examples/01_layout.c</code> prints both layouts.</p>`,
        check: { q: "What is sizeof(struct { char a; int b; char c; }) on x86-64 Linux?",
          options: ["6", "8", "12", "16"], answer: 2,
          why: "a sits at 0, then 3 pad bytes so the 4-aligned int starts at 4, c at 8, then 3 bytes of tail padding round 9 up to 12 (a multiple of 4). Ordering int, char, char gives 8." },
        scene(G) {
          G.text(24, 50, "struct Bad   { char tag; double value; char flag; }", { size: 13 });
          const a = bytes(G, 24, 70, [[1, "q", ""], [7, "hot", "pad"], [8, "k", "value"], [1, "v", ""], [7, "hot", "pad"]], 24);
          [["0", 0], ["8", 8], ["16", 16], ["24", 24]].forEach(([s, b]) => G.label(24 + b * 24, 124, s, { size: 11 }));
          G.text(24, 152, "sizeof = 24   (10 useful bytes)", { color: "hot", size: 14 });
          G.text(24, 220, "struct Good  { double value; char tag; char flag; }", { size: 13 });
          const b = bytes(G, 24, 240, [[8, "k", "value"], [1, "q", ""], [1, "v", ""], [6, "hot", "pad"]], 24);
          [["0", 0], ["8", 8], ["16", 16]].forEach(([s, x]) => G.label(24 + x * 24, 294, s, { size: 11 }));
          G.text(24, 322, "sizeof = 16   (same 10 useful bytes)", { color: "ok", size: 14 });
          G.label(24, 360, "violet = tag (char) · amber = flag (char) · red = padding", { size: 12 });
          G.from(a.concat(b), { opacity: 0, stagger: 0.012, duration: 0.15 });
          G.caption("largest alignment first: the padding shrinks from 14 bytes to 6");
        } },

      { rail: "paid N times", title: "Padding is paid once per element",
        body: `<p>One wasted byte does not matter. A million of them, read every step, does. A serving system keeps a small record for every in-flight request and scans those records constantly. Exercise 1 gives you this one:</p>
<div class="eq">struct RequestRecord {
  bool     streaming;       // 1   @0   (+7 pad)
  uint64_t request_id;      // 8   @8
  uint16_t priority;        // 2   @16  (+6 pad)
  double   arrival_time_s;  // 8   @24
  uint8_t  state;           // 1   @32  (+3 pad)
  uint32_t prompt_tokens;   // 4   @36
  uint32_t output_tokens;   // 4   @40  (+4 tail)
};                          // sizeof = 48</div>
<p>The fields hold 1 + 8 + 2 + 8 + 1 + 4 + 4 = 28 bytes, but the struct takes 48. Sorted by alignment (8, 8, 4, 4, 2, 1, 1) the fields pack into 28 bytes, and tail padding to a multiple of 8 gives <b>32</b>.</p>
<p>Count cache lines: four records as written take 4 × 48 = 192 bytes = 3 lines; sorted, 128 bytes = 2 lines. For a million records that is 48 MB versus 32 MB of memory traffic per full scan.</p>`,
        scene(G) {
          const px = 3, x0 = 24;
          const spans = (x, y, sp, h) => { let b = 0; return sp.map(([n, col]) => { const r = G.rect(x + b * px, y, n * px - 1, h, { fill: col, rx: 2, opacity: col === "hot" ? 0.45 : 0.9 }); b += n; return r; }); };
          G.text(x0, 44, "4 × RequestRecord as written (48 B each)", { size: 13 });
          for (let l = 0; l < 3; l++) G.label(x0 + l * 64 * px + 4, 66, `cache line ${l}`, { size: 11 });
          const r1 = []; for (let r = 0; r < 4; r++) r1.push(...spans(x0 + r * 48 * px, 74, REQ_BAD, 40));
          for (let r = 0; r < 4; r++) G.label(x0 + r * 48 * px + 4, 132, `rec ${r}`, { size: 11 });
          G.text(x0, 180, "4 × RequestRecord sorted (32 B each)", { size: 13 });
          const r2 = []; for (let r = 0; r < 4; r++) r2.push(...spans(x0 + r * 32 * px, 194, REQ_GOOD, 40));
          for (let r = 0; r < 4; r++) G.label(x0 + r * 32 * px + 4, 252, `rec ${r}`, { size: 11 });
          for (let l = 0; l <= 3; l++) { G.line(x0 + l * 64 * px - 1, 70, x0 + l * 64 * px - 1, 118, { color: "ink", dash: "4 4" }); G.line(x0 + l * 64 * px - 1, 190, x0 + l * 64 * px - 1, 238, { color: "ink", dash: "4 4" }); }
          for (let r = 1; r < 4; r++) { G.line(x0 + r * 48 * px - 1, 74, x0 + r * 48 * px - 1, 114, { color: "bg", w: 3 }); G.line(x0 + r * 32 * px - 1, 194, x0 + r * 32 * px - 1, 234, { color: "bg", w: 3 }); }
          G.text(x0, 310, "as written: 192 B → 3 cache lines (20 of every 48 B are padding)", { color: "hot", size: 13 });
          G.text(x0, 336, "sorted:     128 B → 2 cache lines (4 of every 32 B are padding)", { color: "ok", size: 13 });
          G.text(x0, 372, "1,000,000 records: 48 MB vs 32 MB per full scan", { size: 14 });
          G.from(r2, { opacity: 0, stagger: 0.004, duration: 0.15 });
          G.caption("dashed lines = 64-byte cache line boundaries");
        } },

      { rail: "flat matrix", title: "A matrix is a flat buffer plus a rule",
        body: `<p>Memory has one dimension; a matrix has two. To store a matrix with <code>R</code> rows and <code>C</code> columns we lay it out in a line, and use a rule to find element <code>(i, j)</code>.</p>
<p><b>Row-major</b> (C, C++, NumPy, PyTorch default): write row 0, then row 1, and so on. Element <code>(i, j)</code> is at</p>
<div class="eq">offset(i, j) = i × C + j

3 × 4 matrix:  (2, 1) → 2 × 4 + 1 = 9
byte address = base + 9 × sizeof(float) = base + 36</div>
<p><b>Column-major</b> (Fortran, BLAS, MATLAB) writes column 0 first, so <code>(i, j)</code> is at <code>i + j × R</code>. The data is the same; only the rule differs. Llama-3-8B's embedding table is one such buffer: 128,256 rows × 4,096 columns of bf16, and looking up token 42 means reading the 4,096 numbers that start at offset <code>42 × 4096</code>.</p>`,
        scene(G) {
          const cols = ["k", "v", "q"];
          G.label(60, 50, "the matrix you think about (3 × 4)", { size: 13 });
          const cells = [];
          for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) {
            const hit = i === 2 && j === 1;
            cells.push(G.box(60 + j * 64, 66 + i * 50, 58, 44, `(${i},${j})`, { fill: cols[i], size: 12, stroke: hit ? "ink" : undefined, sw: hit ? 3 : 1 }));
          }
          G.label(360, 90, "row-major:", { size: 13, color: "ink" }); G.label(360, 112, "offset = i × 4 + j", { size: 13, color: "ink" });
          G.label(360, 146, "(2,1) → 2 × 4 + 1 = 9", { size: 13, color: "ok" });
          G.label(24, 344, "the buffer the machine stores (offsets 0 … 11)", { size: 13 });
          const flat = [];
          for (let k = 0; k < 12; k++) {
            const hit = k === 9;
            flat.push(G.box(24 + k * 49, 270, 45, 44, String(k), { fill: cols[Math.floor(k / 4)], size: 13, stroke: hit ? "ink" : undefined, sw: hit ? 3 : 1 }));
          }
          G.arrow(60 + 64 + 29, 218, 24 + 9 * 49 + 22, 266, { color: "ink", dash: "4 3" });
          G.label(24, 368, "row 0 | row 1 | row 2, back to back", { size: 12 });
          G.from(flat, { opacity: 0, x: -10, stagger: 0.05, duration: 0.25 });
          G.caption("two dimensions are a convention; memory is one line");
        } },

      { rail: "strides", title: "Strides: one buffer, many matrices",
        body: `<p>Both rules are special cases of one formula with two numbers called <b>strides</b>: how far to move in the buffer when <code>i</code> or <code>j</code> goes up by one.</p>
<div class="eq">offset(i, j) = i × stride0 + j × stride1</div>
<table><tr><th>view</th><th>shape</th><th>stride0</th><th>stride1</th></tr>
<tr><td>row-major</td><td>(R, C)</td><td>C</td><td>1</td></tr>
<tr><td>column-major</td><td>(R, C)</td><td>1</td><td>R</td></tr>
<tr><td>transpose of row-major</td><td>(C, R)</td><td>1</td><td>C</td></tr>
<tr><td>every other column</td><td>(R, C/2)</td><td>C</td><td>2</td></tr></table>
<p>So a <b>transpose is free</b>: keep the buffer, swap the shape and the strides. Zero bytes move. This is exactly what a PyTorch tensor is: a pointer, a shape and strides. <code>t.T</code> swaps strides; <code>t.contiguous()</code> pays to copy the elements into a fresh row-major buffer. Exercise 2 asks you to write that copy, <code>strided_copy</code>.</p>`,
        check: { q: "You transpose a 4096 × 4096 row-major float matrix by swapping its strides. How many bytes are copied?",
          options: ["0", "64 MiB (the whole matrix)", "16 KiB (one row)"], answer: 0,
          why: "Swapping strides only changes how offsets are computed. The buffer is untouched. Bytes move only when someone makes the view contiguous, which costs a full 64 MiB read and 64 MiB write." },
        scene(G) {
          G.label(24, 36, "one buffer", { size: 13 });
          const flat = [];
          for (let k = 0; k < 12; k++) flat.push(G.box(24 + k * 49, 46, 45, 36, String(k), { fill: ["k", "v", "q"][Math.floor(k / 4)], size: 12 }));
          // walk order of the transposed view's row 0: 0, 4, 8
          G.path("M 46 86 C 70 126, 220 126, 242 86", { color: "ink", arrow: true, w: 2 });
          G.path("M 242 86 C 266 126, 416 126, 438 86", { color: "ink", arrow: true, w: 2 });
          G.label(250, 140, "transposed row 0 jumps by 4", { size: 12, color: "ink" });
          G.text(24, 190, "view A: (3, 4), strides (4, 1)", { size: 13 });
          for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) G.box(24 + j * 52, 204 + i * 46, 46, 40, String(i * 4 + j), { fill: ["k", "v", "q"][i], size: 12 });
          G.text(330, 190, "view B = Aᵀ: (4, 3), strides (1, 4)", { size: 13 });
          const B = [];
          for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) { const off = i * 1 + j * 4; B.push(G.box(330 + j * 52, 204 + i * 46, 46, 40, String(off), { fill: ["k", "v", "q"][Math.floor(off / 4)], size: 12 })); }
          G.text(500, 260, "0 bytes", { size: 16, color: "ok" }); G.text(500, 282, "moved", { size: 16, color: "ok" });
          G.from(B, { opacity: 0, scale: 0.6, transformOrigin: "center", stagger: 0.05, duration: 0.25 });
          G.caption("numbers in the views are buffer offsets: same data, new strides");
        } },

      { rail: "cache lines", title: "Why the walking order decides the speed",
        body: `<p>Free views have a hidden cost. When the CPU needs one float, it fetches the whole 64-byte line around it: 16 floats. If the next 15 accesses use those floats, the fetch paid for itself. If the next access is far away, 60 of the 64 bytes were moved for nothing.</p>
<p>Take the naive transpose of a 4096 × 4096 float matrix:</p>
<div class="eq">for i in 0..R:  for j in 0..C:
    dst[j * R + i] = src[i * C + j]</div>
<p>Reads walk along <code>src</code>: consecutive, every byte of each line used. Writes walk down a column of <code>dst</code>: each write is <code>4096 × 4 = 16 KiB</code> after the previous one, so every write lands on a different line and uses 4 of its 64 bytes. One column of writes touches 4,096 lines = 256 KiB, more than a typical L1 data cache holds (often 32–48 KiB, example values: check your CPU), so lines are thrown out before the next column could reuse them.</p>
<p>The fix is <b>blocking</b> (exercise 4): transpose in small square tiles. With 16 × 16 float tiles, a tile's source and destination together touch 32 lines = 2 KiB, which stays in L1 until every byte of every line has been used.</p>`,
        check: { q: "The naive transpose reads and writes exactly the same number of bytes as a plain copy. Why is it much slower?",
          options: ["Writes are slower than reads on every CPU", "Each write lands on a different cache line and uses only 4 of its 64 bytes, so the hardware moves far more bytes than the payload", "The compiler can't optimize two nested loops"], answer: 1,
          why: "Memory moves in 64-byte lines. A stride of 16 KiB means each line fetched for writing contributes 4 useful bytes before it is evicted. Blocking keeps a tile's lines in L1 until they are fully used." },
        scene(G) {
          const cw = 13, ch = 22;
          const draw = (x0, title, hotFn) => {
            G.text(x0, 44, title, { size: 13 });
            const cells = [];
            for (let r = 0; r < 8; r++) {
              for (let c = 0; c < 16; c++) cells.push(G.rect(x0 + c * cw, 60 + r * (ch + 6), cw - 2, ch, { fill: hotFn(r, c) || "line", rx: 2, opacity: hotFn(r, c) ? 0.95 : 0.5 }));
              G.rect(x0 - 3, 58 + r * (ch + 6), 16 * cw + 4, ch + 4, { stroke: "muted", rx: 4, sw: 1 });
            }
            return cells;
          };
          const a = draw(24, "naive: one column of dst", (r, c) => (c === 0 ? "hot" : null));
          const b = draw(330, "blocked: one 8 × 8 tile of dst", (r, c) => (c < 8 ? "ok" : null));
          G.label(24, 300, "each outlined row = one 64-byte line (16 floats)", { size: 12 });
          G.text(24, 330, "8 lines fetched, 4 B used in each", { color: "hot", size: 13 });
          G.text(24, 352, "useful: 4 / 64 = 6.25%", { color: "hot", size: 13 });
          G.text(330, 330, "the same 8 lines, reused 8 times", { color: "ok", size: 13 });
          G.text(330, 352, "8 of 16 floats used; 16 × 16 tiles use all", { color: "ok", size: 13 });
          G.text(24, 392, "4096 × 4096 floats: consecutive naive writes are 16 KiB apart", { size: 13 });
          G.from(a.filter((_, i) => i % 16 === 0), { opacity: 0, stagger: 0.12, duration: 0.2 });
          G.caption("memory moves in lines: use the whole line while you have it");
        } },

      { rail: "ownership", title: "Every buffer needs exactly one owner",
        body: `<p>Memory you allocate must be freed exactly once. Free it zero times and the process leaks until it runs out. Free it twice (a <b>double free</b>) and the allocator's bookkeeping is corrupted. Use it after freeing (<b>use-after-free</b>) and you read whatever moved in. In C you track this by hand, on every return path.</p>
<p>C++ ties the release to an object's lifetime. This is <b>RAII</b> (Resource Acquisition Is Initialization): the constructor acquires, the destructor releases, and the destructor runs automatically when the object goes out of scope, on a normal return or an exception.</p>
<div class="eq">class Buffer {
  float* p_;
 public:
  explicit Buffer(size_t n) : p_(new float[n]) {}
  ~Buffer() { delete[] p_; }
};</div>
<p>This class has a trap. Writing <code>Buffer b = a;</code> makes C++ generate a copy that copies the <i>pointer</i>. Now two objects own one block, and both destructors free it. If you write a destructor, you must also decide what copying means (the "rule of three/five").</p>`,
        scene(G) {
          G.label(24, 40, "stack (objects)", { size: 13 }); G.label(400, 40, "heap (the allocation)", { size: 13 });
          G.box(24, 70, 200, 52, "Buffer a { p_ }", { stroke: "k", color: "k" });
          const b = G.box(24, 190, 200, 52, "Buffer b = a  { p_ }", { stroke: "v", color: "v" });
          G.rect(400, 110, 220, 90, { fill: "w", opacity: 0.6, rx: 8 });
          G.text(510, 160, "float[n]", { anchor: "middle", size: 14 });
          G.arrow(226, 96, 396, 138, { color: "k", w: 2 });
          const ar = G.arrow(226, 216, 396, 172, { color: "v", w: 2 });
          G.text(24, 300, "end of scope:", { size: 13 });
          G.text(24, 326, "~b()  delete[] p_   → block freed", { size: 13, color: "ink" });
          const x = G.text(24, 352, "~a()  delete[] p_   → freed AGAIN: double free", { size: 13, color: "hot" });
          G.text(24, 392, "fix: delete copying, allow moving (next step)", { size: 13, color: "ok" });
          G.from([b, ar], { opacity: 0, duration: 0.5, delay: 0.3 }); G.pulse(x, { repeat: 5 });
          G.caption("a copied pointer means two owners: one block, two frees");
        } },

      { rail: "move", title: "Move: hand over ownership instead of copying",
        body: `<p>There are three honest answers to "what does copying a Buffer mean?":</p>
<ol><li><b>Forbid it</b>: <code>Buffer(const Buffer&amp;) = delete;</code></li>
<li><b>Deep copy</b>: allocate a new block and copy every byte. Correct, but it can be expensive by accident.</li>
<li><b>Move</b>: the new object takes the pointer, and the old one is set to <code>nullptr</code> so its destructor frees nothing.</li></ol>
<div class="eq">Buffer(Buffer&amp;&amp; o) noexcept
  : n_(std::exchange(o.n_, 0)),
    p_(std::exchange(o.p_, nullptr)) {}

Buffer b = std::move(a);   // a is now empty</div>
<p>Systems code picks 1 + 3: <b>move-only</b> types. That is what <code>std::unique_ptr</code> is. <code>examples/04_move_vs_copy.cpp</code> passes a 64 MB tensor (16M floats) through 4 stages: by value that copies 4 × 64 MB = 256 MB; with <code>std::move</code> it hands over a pointer four times. This is why inference engines pass tensors by handle, never by value.</p>
<p>Mark moves <code>noexcept</code>: <code>std::vector</code> only moves elements when it grows if the move cannot throw; otherwise it copies them.</p>`,
        check: { q: "After Buffer b = std::move(a); with the move constructor above, what happens when a and b go out of scope?",
          options: ["Both free the block: double free", "b frees the block; a holds nullptr, and delete[] nullptr does nothing", "Neither frees it: leak"], answer: 1,
          why: "The move constructor stole the pointer and left nullptr behind. Deleting a null pointer is defined to do nothing, so the block is freed exactly once, by b." },
        scene(G) {
          G.text(24, 40, "copy pipeline: each stage copies 64 MB", { size: 13, color: "hot" });
          const cp = [];
          for (let s = 0; s < 4; s++) { cp.push(G.box(24 + s * 150, 56, 130, 60, "64 MB", { fill: "hot", size: 13 })); if (s < 3) G.arrow(156 + s * 150, 86, 172 + s * 150, 86, { color: "hot" }); }
          G.label(24, 136, "4 allocations · 256 MB copied · 4 frees", { size: 12 });
          G.text(24, 200, "move pipeline: one block, the pointer is handed on", { size: 13, color: "ok" });
          G.rect(24, 300, 580, 60, { fill: "w", opacity: 0.6, rx: 8 }); G.text(314, 336, "64 MB (allocated once)", { anchor: "middle", size: 13 });
          const pts = [];
          for (let s = 0; s < 4; s++) {
            const own = s === 3;
            pts.push(G.box(24 + s * 150, 216, 130, 40, own ? "stage 4: p_" : `stage ${s + 1}: null`, { stroke: own ? "ok" : "muted", color: own ? "ok" : "muted", size: 12 }));
          }
          G.arrow(540, 258, 500, 296, { color: "ok", w: 2 });
          G.label(24, 390, "8-byte pointer moved 4 times · 1 free at the end", { size: 12 });
          G.from(pts, { opacity: 0.2, stagger: 0.3, duration: 0.3 });
          G.caption("a move costs a few bytes; a copy costs the whole buffer");
        } },

      { rail: "templates · span", title: "Write once for every type; borrow without owning",
        body: `<p>A kernel library needs the same transpose for <code>float</code>, <code>int8_t</code>, bf16. A <b>function template</b> is written once and compiled once <i>per type you use</i>: <code>transpose&lt;float&gt;</code> and <code>transpose&lt;int8_t&gt;</code> become two separate, fully optimized functions, with no runtime <code>switch</code> on the type.</p>
<p>A <b>concept</b> states what a template parameter must be (<code>concept Element = std::is_arithmetic_v&lt;T&gt;;</code>), so misuse gives one readable error. A <b><code>constexpr</code></b> function runs at compile time when its inputs are constants. <code>examples/05_templates.cpp</code> picks a transpose tile so two tiles fit in 32 KiB:</p>
<div class="eq">float: 2 tiles × 64 × 64 × 4 B  = 32 KiB → tile 64
int8:  2 tiles × 128 × 128 × 1 B = 32 KiB → tile 128
static_assert(tile_for_l1&lt;float&gt;() == 64);</div>
<p>Finally, <b><code>std::span&lt;T&gt;</code></b> is a (pointer, length) pair that owns nothing. Functions take spans; owners (<code>std::vector</code>, <code>unique_ptr</code>, <code>s2s::aligned_buffer</code>) live outside and must outlive every span into them. The same split, owner versus view, reappears in every tensor library.</p>`,
        scene(G) {
          G.box(24, 50, 250, 50, "template<Element T> transpose", { stroke: "q", color: "q", size: 12 });
          G.box(24, 150, 120, 44, "<float>", { fill: "k", size: 13 }); G.box(154, 150, 120, 44, "<int8_t>", { fill: "v", size: 13 });
          G.arrow(100, 102, 84, 146, { color: "q" }); G.arrow(200, 102, 214, 146, { color: "q" });
          G.label(24, 218, "two separate compiled functions", { size: 12 });
          G.label(24, 236, "tiles 64 / 128 set at compile time", { size: 12 });
          G.text(330, 72, "owner: aligned_buffer<float>", { size: 13 });
          const own = []; for (let k = 0; k < 16; k++) own.push(G.rect(330 + k * 18, 84, 16, 34, { fill: "w", rx: 2, opacity: 0.7 }));
          const sp = []; for (let k = 4; k < 10; k++) sp.push(G.rect(330 + k * 18, 84, 16, 34, { fill: "k", rx: 2 }));
          G.box(372, 170, 160, 40, "span { ptr, len = 6 }", { stroke: "k", color: "k", size: 12 });
          G.arrow(420, 168, 404, 122, { color: "k", w: 2 });
          G.label(330, 236, "a view: frees nothing, copies nothing", { size: 12 });
          G.text(24, 300, "owner frees · views borrow · the owner must outlive the view", { size: 14 });
          G.text(24, 330, "same pattern later: torch.Tensor storage vs view", { size: 13, color: "muted" });
          G.from(sp, { opacity: 0, stagger: 0.06, duration: 0.2 });
          G.caption("templates multiply code at compile time; spans share data at run time");
        } },

      { rail: "build · check", title: "Build with targets, check with sanitizers",
        body: `<p>C and C++ won't stop you from writing past the end of a buffer. The program might crash, or silently corrupt a neighbour, or work by luck. Two tools turn luck into certainty.</p>
<p><b>CMake</b> describes the build as targets with their own flags, so the same project builds an optimized benchmark and a checked debug test:</p>
<div class="eq">add_executable(app main.cpp)
target_compile_options(app PRIVATE -O3 -march=native)
target_link_libraries(app PRIVATE tensor)</div>
<p><b>Sanitizers</b> are compiler options that add checks. AddressSanitizer (<code>-fsanitize=address</code>) surrounds every allocation with poisoned "red zones" and checks each memory access, so an out-of-bounds write stops the program on the exact line. UndefinedBehaviorSanitizer (<code>-fsanitize=undefined</code>) catches things like shifting a 64-bit value by 64.</p>
<div class="eq">for (int i = 0; i &lt;= cap_; ++i) buf_[i] = 0;   // BUG
ERROR: AddressSanitizer: heap-buffer-overflow
  ... 0 bytes after 16-byte region</div>
<p>Exercise 5 plants three such bugs in a token ring buffer. <code>examples/CMakeLists.txt</code> has an <code>S2S_SANITIZE</code> option for every target. Never benchmark a Debug or sanitized build: always <code>-DCMAKE_BUILD_TYPE=Release</code>.</p>`,
        check: { q: "ASan reports heap-buffer-overflow, \"0 bytes after 16-byte region\", for an int32 buffer of capacity 4. What is the most likely bug?",
          options: ["The buffer was freed twice", "A loop bound uses &lt;= where it should use &lt;, so it writes element 4", "The buffer isn't 64-byte aligned"], answer: 1,
          why: "16 bytes is 4 int32s. \"0 bytes after\" means the bad access starts exactly at the end: index 4 of a 4-element buffer, the classic off-by-one." },
        scene(G) {
          G.label(24, 50, "int32 buffer, capacity 4 (16 bytes), with ASan red zones", { size: 13 });
          G.rect(24, 80, 120, 50, { fill: "hot", opacity: 0.35, rx: 4 }); G.text(84, 110, "red zone", { anchor: "middle", size: 12 });
          const sl = [];
          for (let i = 0; i < 4; i++) sl.push(G.box(150 + i * 80, 80, 74, 50, `buf[${i}]`, { fill: "k", size: 12 }));
          G.rect(470, 80, 150, 50, { fill: "hot", opacity: 0.35, rx: 4 }); G.text(545, 110, "red zone", { anchor: "middle", size: 12 });
          const bad = G.box(476, 150, 74, 40, "buf[4]", { stroke: "hot", color: "hot", size: 12, dash: "4 3" });
          G.arrow(513, 150, 513, 134, { color: "hot", w: 2 });
          G.text(24, 236, "for (int i = 0; i <= cap_; ++i) buf_[i] = 0;", { size: 13 });
          G.text(24, 270, "→ heap-buffer-overflow … 0 bytes after 16-byte region", { size: 13, color: "hot" });
          G.text(24, 320, "without ASan: may pass, may crash, may corrupt", { size: 13, color: "muted" });
          G.text(24, 346, "with ASan: stops on this line, every time", { size: 13, color: "ok" });
          G.from(sl, { opacity: 0, stagger: 0.12, duration: 0.2 }); G.pulse(bad, { repeat: 6 });
          G.caption("sanitizers turn silent memory bugs into exact error reports");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Memory is a line of bytes; a pointer is an address whose type sets the step size, and <code>a[i]</code> is <code>*(a + i)</code>.</li>
<li>Every type has an alignment, so the compiler pads structs. Order fields largest-alignment first and check with <code>offsetof</code>; padding is paid once per element.</li>
<li>A matrix is a buffer plus strides: <code>offset = i × stride0 + j × stride1</code>. Transposes are free views; <code>.contiguous()</code> is a copy.</li>
<li>Memory moves in 64-byte cache lines, so the walking order decides how many fetched bytes are useful. Blocking keeps lines hot until they are used up.</li>
<li>RAII ties freeing to scope; move-only types make "exactly one owner" a compile-time fact; sanitizers catch what slips through.</li></ul>
<p>Next, P0.2 asks what an address really is: the numbers in this lesson are <i>virtual</i>, and the operating system translates every one of them.</p>`,
    sim: {
      title: "Walk a matrix through a tiny cache",
      intro: "A float matrix sits in row-major memory. Pick the order in which a loop visits its elements and the size of a small cache. Each access either hits a line already in the cache (green) or fetches a new 64-byte line (red). Press Run to watch the walk. The panels count how many bytes the walk moved compared with the bytes it actually used.",
      height: 300,
      controls: [
        { id: "order", label: "walking order", type: "select", value: "col", options: [["row", "row by row"], ["col", "column by column"], ["tile", "T × T tiles"]] },
        { id: "n", label: "matrix size N (N × N floats)", min: 16, max: 64, step: 16, value: 32 },
        { id: "lines", label: "cache capacity, lines of 64 B", min: 2, max: 64, step: 2, value: 8 },
        { id: "tile", label: "tile size T (tiles order only)", min: 2, max: 32, step: 2, value: 8 },
        { id: "es", label: "element size", type: "select", value: 4, options: [[4, "4 bytes (float)"], [2, "2 bytes (bf16)"], [8, "8 bytes (double)"]] },
      ],
      run: { label: "Run the walk", frames: 48, ms: 55 },
      draw(G, v, t) {
        const N = v.n, es = v.es, per = 64 / es, cap = v.lines, T = Math.min(v.tile, N);
        // build the access order
        const seq = [];
        if (v.order === "row") { for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) seq.push([i, j]); }
        else if (v.order === "col") { for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) seq.push([i, j]); }
        else { for (let bi = 0; bi < N; bi += T) for (let bj = 0; bj < N; bj += T) for (let j = bj; j < Math.min(bj + T, N); j++) for (let i = bi; i < Math.min(bi + T, N); i++) seq.push([i, j]); }
        // fully associative LRU cache of `cap` lines
        const lru = new Map(); const res = new Array(seq.length); let miss = 0;
        seq.forEach(([i, j], k) => {
          const line = Math.floor((i * N + j) / per);
          if (lru.has(line)) { lru.delete(line); res[k] = 1; } else { miss++; res[k] = 0; if (lru.size >= cap) lru.delete(lru.keys().next().value); }
          lru.set(line, 1);
        });
        const shown = Math.max(1, Math.round(seq.length * t));
        const S = 270 / N, st = new Map();
        for (let k = 0; k < shown; k++) st.set(seq[k][0] * N + seq[k][1], res[k]);
        G.label(16, 14, `${N} × ${N} matrix, row-major · one cell = one element`, { size: 11 });
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
          const s = st.get(i * N + j);
          G.rect(16 + j * S, 24 + i * S, Math.max(1, S - 1), Math.max(1, S - 1), { fill: s === undefined ? "line" : s ? "ok" : "hot", rx: 0, opacity: s === undefined ? 0.5 : 0.9 });
        }
        if (shown < seq.length || t < 1) { const [ci, cj] = seq[shown - 1]; G.rect(16 + cj * S - 1, 24 + ci * S - 1, S + 1, S + 1, { stroke: "ink", rx: 0, sw: 2 }); }
        let missSoFar = 0; for (let k = 0; k < shown; k++) if (!res[k]) missSoFar++;
        const moved = miss * 64, useful = N * N * es;
        // bar chart: moved vs useful
        G.label(320, 40, "bytes moved from memory vs bytes used", { size: 12 });
        const mx = Math.max(moved, useful), bw = 290;
        G.rect(320, 56, (bw * missSoFar * 64) / mx, 26, { fill: "hot", rx: 3 }); G.label(320, 100, `moved so far: ${F.bytes(missSoFar * 64)}`, { size: 12, color: "ink" });
        G.rect(320, 116, (bw * useful) / mx, 26, { fill: "k", rx: 3 }); G.label(320, 160, `useful (each element once): ${F.bytes(useful)}`, { size: 12, color: "ink" });
        G.label(320, 200, `line = 64 B = ${per} elements`, { size: 12 });
        G.label(320, 220, `cache = ${cap} lines = ${F.bytes(cap * 64)}`, { size: 12 });
        G.label(320, 240, `one row = ${N * es} B = ${(N * es) / 64} line(s)`, { size: 12 });
        const eff = useful / moved;
        return [
          { title: "Result of the full walk", rows: [["accesses", F.num(seq.length)], ["cache misses (lines fetched)", F.num(miss)], ["hits", F.num(seq.length - miss)], ["bytes moved", F.bytes(moved)], ["useful bytes", F.bytes(useful)]],
            gauge: [[Math.min(1, eff), "ok"], [Math.max(0, 1 - eff), "hot"]], gaugeText: `efficiency ${(100 * eff).toFixed(1)}%`, chip: [eff > 0.9, eff > 0.9 ? "every fetched line fully used" : "lines evicted before they were used up"] },
          { title: "Why", html: `<p class="note">The best possible is one miss per line: ${F.num((N * N) / per)} misses. Column order fetches a new line per access unless the cache can hold one line per row (${N} lines here). Tiles need about T lines (T × T floats span T rows) to stay hot.</p>` },
          { title: "Model", html: `<p class="note">Fully associative LRU cache, one matrix, no hardware prefetcher. Real CPUs have set-associative caches of hundreds of lines and prefetchers that hide some strided misses, so measure with <code>bench/transpose_bench</code>.</p>` },
        ];
      },
    },
    practice: {
      intro: `All five exercises build as one CMake project. Run from the module folder <code>course/P0-systems-primer/P0.1-c-cpp-for-systems</code>. Add <code>-DS2S_USE_SOLUTIONS=ON</code> to the configure step to build the reference solutions instead and confirm the tests pass.`,
      items: [
        { title: "Shrink a struct", tier: "T0 · easy", goal: "Reorder the fields of RequestRecord, without removing or retyping any, so that sizeof drops from 48 to 32.",
          cmd: "cmake -S exercises -B build/ex && cmake --build build/ex -j && ctest --test-dir build/ex -R 01-shrink-struct --output-on-failure" },
        { title: "strided_copy = .contiguous()", tier: "T0 · easy", goal: "Copy any strided 2-D view (row-major, column-major, transposed, sliced, flipped) into a contiguous row-major buffer.",
          cmd: "ctest --test-dir build/ex -R 02-strided-copy --output-on-failure" },
        { title: "The Tensor RAII class", tier: "T0 · medium", goal: "A move-only tensor owning 64-byte-aligned storage, with an explicit clone() and a live-allocation counter that never goes negative.",
          cmd: "ctest --test-dir build/ex -R 03-tensor-raii --output-on-failure" },
        { title: "Blocked transpose<T, Block>", tier: "T0 · medium", goal: "Transpose in Block × Block tiles for float and int8_t, any shape. The test checks correctness; the bench shows the speed-up.",
          cmd: "ctest --test-dir build/ex -R 04-blocked-transpose --output-on-failure" },
        { title: "Sanitizer hunt", tier: "T0 · hard", goal: "Find and fix an off-by-one write, a use-after-free and an over-wide shift in a token ring buffer, guided by ASan and UBSan reports.",
          cmd: "ctest --test-dir build/ex -R 05-sanitizer-hunt --output-on-failure" },
      ],
      labs: [
        { label: "Examples: layout, strides, RAII counters, move vs copy, templates (build with cmake -S examples -B build/examples -DCMAKE_BUILD_TYPE=Release)", path: "course/P0-systems-primer/P0.1-c-cpp-for-systems/examples/" },
        { label: "Benchmark: copy vs naive vs your blocked transpose, GB/s and % of copy", path: "course/P0-systems-primer/P0.1-c-cpp-for-systems/bench/transpose_bench.cpp" },
        { label: "The aligned owner used across the course", path: "course/common/include/s2s/aligned.hpp" },
        { label: "Animation: memory layout explorer", path: "animations/p0-memory-layout.html" },
        { label: "Self-check questions", path: "course/P0-systems-primer/P0.1-c-cpp-for-systems/quiz.md" },
      ],
    },
  });
})();
