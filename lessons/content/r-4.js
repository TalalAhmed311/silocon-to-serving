/* R.4 — Python and NumPy for this course. A Readiness lesson: teaches and self-checks. */
(function () {
  const F = S2S.fmt;
  const DT = { float32: ["float32", 4], bfloat16: ["bfloat16", 2], float16: ["float16", 2], int8: ["int8", 1] };

  // A flat buffer of n cells, each 4 bytes (float32), with byte offsets underneath.
  const buffer = (G, x, y, n, cw, o = {}) => {
    const cells = [];
    for (let i = 0; i < n; i++) {
      cells.push(G.box(x + i * cw, y, cw - 4, 38, o.labels ? o.labels[i] : String(i), { fill: o.fill ? o.fill(i) : "k", size: 13 }));
      if (o.offsets) G.label(x + i * cw + 4, y + 56, String(i * 4), { size: 11 });
    }
    return cells;
  };

  // Broadcast two shapes (arrays of dims, left to right). Returns {ok, out, cols:[{a,b,kind}]} aligned from the right over 3 slots.
  function bcast(a, b) {
    const cols = [], out = [];
    let ok = true;
    for (let p = 0; p < 3; p++) {
      const ia = a.length - 3 + p, ib = b.length - 3 + p;
      const da = ia >= 0 ? a[ia] : null, db = ib >= 0 ? b[ib] : null;
      const va = da === null ? 1 : da, vb = db === null ? 1 : db;
      let kind = "same";
      if (da === null && db === null) kind = "none";
      else if (va === vb) kind = "same";
      else if (va === 1 || vb === 1) kind = "stretch";
      else { kind = "clash"; ok = false; }
      cols.push({ da, db, kind });
      out.push(kind === "clash" ? null : Math.max(va, vb));
    }
    return { ok, cols, out };
  }

  S2S.lesson({
    id: "r-4", n: "R.4", title: "Python and NumPy for this course",
    subtitle: "Readiness · first principles · T0, any laptop",
    kicker: "Lesson · ≈ 35 min",
    headline: "A flat buffer wearing a shape",
    intro: `<p><b>Already comfortable with Python and NumPy? Answer the checkpoints. If you get them all right, skip to R.5.</b></p>
<p>The course writes its fast engines in C++ and CUDA and keeps a short, trusted Python twin of each piece. Every test compares the two. Almost every number in the course (model sizes, FLOPs, cache sizes) is shape-times-dtype arithmetic on NumPy arrays. This lesson teaches the ten ideas you need: the Python the repo uses, the ndarray as a buffer plus a shape, dtypes and bytes, views, strides, broadcasting, vectorising, matrix multiply, and tests with tolerances.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "data"], ["v", "view / stretched"], ["ok", "match / ok"], ["hot", "copy / clash"], ["q", "second array"]],
    prev: "r-3", next: "r-5",
    steps: [
      { rail: "why python", title: "Why Python sits next to the C++",
        body: `<p>A fast kernel can be wrong in ways that still run and still look plausible. So each engine piece has a short <b>reference implementation</b> in Python: slow, simple, and easy to trust. A <b>test</b> feeds both the same input and checks that they agree.</p>
<p>The biggest example is <code>platform/engine/v0/reference/llama_numpy.py</code>: a whole Llama forward pass in NumPy, with functions such as <code>rmsnorm</code>, <code>softmax</code> and <code>silu</code> that are a few lines each. If you can read NumPy shapes, you can read that file.</p>
<p>You do not need to be a Python expert. You need to read it and write small array functions.</p>`,
        scene(G) {
          G.box(24, 40, 170, 48, "same input", { stroke: "ink", size: 13 });
          G.arrow(110, 90, 110, 128, { color: "ink", w: 2 }); G.arrow(110, 90, 420, 128, { color: "ink", w: 2 });
          const a = G.box(24, 130, 200, 70, "NumPy reference", { fill: "k", size: 13 });
          const b = G.box(330, 130, 220, 70, "C++ / CUDA engine", { fill: "v", size: 13 });
          G.label(24, 216, "slow, simple, trusted", { size: 11 }); G.label(330, 216, "fast, must match", { size: 11 });
          G.arrow(124, 224, 250, 250, { color: "ink", w: 2 }); G.arrow(440, 224, 330, 250, { color: "ink", w: 2 });
          G.box(180, 252, 250, 52, "assert_allclose(...)", { stroke: "ok", color: "ok", size: 14 });
          G.arrow(305, 306, 305, 336, { color: "ok", w: 2 });
          G.text(305, 366, "pytest: pass or fail", { size: 14, anchor: "middle", color: "ok" });
          G.text(24, 408, "example: platform/engine/v0/reference/llama_numpy.py", { size: 12, color: "muted" });
          G.from([a, b], { opacity: 0, y: 10, stagger: 0.2, duration: 0.4 });
          G.caption("the reference is the answer key for the fast code");
        } },

      { rail: "python basics", title: "The Python you will read",
        body: `<p>Five pieces cover most of the repo's Python:</p>
<div class="eq">@dataclass
class Config:              # fields, no boilerplate
    layers: int = 32
    kv_heads: int = 8

sizes = {n: len(n) for n in ["q", "k", "v"]}
squares = [i * i for i in range(5)]</div>
<ul><li><b>Functions</b> and <b>dicts</b> (<code>{"k": 3}</code>), indexed by key.</li>
<li><b>Comprehensions</b> build a list or dict in one line: <code>[i * i for i in range(5)]</code> is <code>[0, 1, 4, 9, 16]</code>.</li>
<li><b>Dataclasses</b> are classes that are mostly named fields.</li>
<li><b>Type hints</b> (<code>layers: int</code>) document what is expected. Python does not enforce them when it runs.</li></ul>`,
        scene(G) {
          G.label(24, 36, "[i * i for i in range(5)]", { size: 13 });
          const a = [], b = [];
          for (let i = 0; i < 5; i++) {
            a.push(G.box(24 + i * 70, 56, 56, 38, String(i), { stroke: "ink", size: 14 }));
            G.arrow(52 + i * 70, 96, 52 + i * 70, 130, { color: "muted", w: 1.5 });
            b.push(G.box(24 + i * 70, 132, 56, 38, String(i * i), { fill: "k", size: 14 }));
          }
          G.label(390, 80, "range(5)", { size: 12 }); G.label(390, 152, "the new list", { size: 12 });
          G.label(24, 212, "@dataclass Config", { size: 13 });
          G.box(24, 228, 150, 38, "layers = 32", { fill: "v", size: 12 }); G.box(184, 228, 150, 38, "kv_heads = 8", { fill: "v", size: 12 });
          G.label(24, 304, "dict: key → value", { size: 13 });
          [["q", "1"], ["k", "1"], ["v", "1"]].forEach(([k, v], i) => { G.box(24 + i * 160, 320, 50, 36, `"${k}"`, { stroke: "ink", size: 12 }); G.arrow(76 + i * 160, 338, 100 + i * 160, 338, { color: "muted", w: 1.5 }); G.box(102 + i * 160, 320, 40, 36, v, { fill: "q", size: 12 }); });
          G.from(b, { opacity: 0, y: -10, stagger: 0.12, duration: 0.3 });
          G.caption("comprehension, dataclass, dict: the repo's everyday shapes");
        } },

      { rail: "the ndarray", title: "An array is a flat buffer plus a shape",
        body: `<p>NumPy does not store a grid. It stores the numbers in <b>one flat run of memory</b> (a buffer) and remembers three facts: the <b>dtype</b> (what each number is), the <b>shape</b> (how to read it as a grid) and the <b>strides</b> (how many bytes to jump per step along each axis).</p>
<div class="eq">a = np.arange(6, dtype=np.float32).reshape(2, 3)
a.shape    # (2, 3)
a.strides  # (12, 4)</div>
<p>A float32 is 4 bytes. To move one column, jump 4 bytes. To move one row, jump 3 columns: 12 bytes. Element <code>(i, j)</code> is at byte <code>i × 12 + j × 4</code>. This is the same rule as P0.1's <code>offset = i × stride0 + j × stride1</code>.</p>`,
        scene(G) {
          G.label(24, 34, "what you see: shape (2, 3)", { size: 13 });
          const grid = [];
          for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) grid.push(G.box(24 + c * 66, 50 + r * 46, 62, 40, String(r * 3 + c), { fill: r ? "q" : "k", size: 14 }));
          G.label(240, 76, "one row = 12 bytes", { size: 12 }); G.label(240, 122, "one column = 4 bytes", { size: 12 });
          G.label(24, 200, "what memory holds: 6 float32 in a row", { size: 13 });
          const cells = buffer(G, 24, 216, 6, 98, { offsets: true, fill: (i) => (i < 3 ? "k" : "q") });
          G.label(24, 300, "byte offsets under each cell", { size: 11 });
          G.text(24, 346, "(1, 2) → 1 × 12 + 2 × 4 = byte 20 → cell 5", { size: 14, color: "ok" });
          G.text(24, 376, "reshape does not move the numbers", { size: 13 });
          G.from(cells, { opacity: 0, y: 8, stagger: 0.08, duration: 0.25 });
          G.caption("a shape is a way of reading the buffer, not a different storage");
        } },

      { rail: "dtype bytes", title: "The dtype decides the bytes",
        body: `<p>Size in bytes is the number of elements times the bytes per element. <code>a.nbytes</code> reports it.</p>
<table><tr><th>dtype</th><th>bytes each</th></tr><tr><td>float32</td><td>4</td></tr><tr><td>float16, bfloat16</td><td>2</td></tr><tr><td>int8</td><td>1</td></tr></table>
<p>A real example: one Llama-3-8B layer's <code>q_proj</code> weight is 4096 × 4096 = 16,777,216 numbers.</p>
<div class="eq">float32 : 16,777,216 × 4 = 67,108,864 B = 64 MiB
bfloat16: 16,777,216 × 2 = 33,554,432 B = 32 MiB
int8    : 16,777,216 × 1 = 16,777,216 B = 16 MiB</div>
<p>(1 MiB = 1,048,576 bytes. NumPy has no bfloat16 type; the course counts it as 2 bytes.) Halving the bytes halves the memory and the time to read it. That is why serving engines quantise.</p>`,
        check: { q: "A (4096, 4096) matrix is stored in bfloat16 (2 bytes). How big is it?",
          options: ["16 MiB", "32 MiB", "64 MiB"], answer: 1,
          why: "4096 × 4096 = 16,777,216 elements × 2 bytes = 33,554,432 bytes = 32 MiB. float32 would be 64 MiB and int8 16 MiB." },
        scene(G) {
          G.text(24, 36, "the same 4096 × 4096 matrix", { size: 14 });
          [["float32", 64, "k"], ["bfloat16", 32, "v"], ["int8", 16, "ok"]].forEach(([n, mib, c], i) => {
            const y = 70 + i * 90, w = mib * 8;
            G.text(24, y + 20, n, { size: 14 });
            const r = G.rect(24, y + 32, w, 34, { fill: c, rx: 4 });
            G.text(32 + w, y + 55, `${mib} MiB`, { size: 14 });
            G.from(r, { scaleX: 0, transformOrigin: "left center", duration: 0.5 });
          });
          G.label(24, 360, "16,777,216 elements × bytes per element", { size: 12 });
          G.label(24, 380, "4 bytes → 64 MiB · 2 bytes → 32 MiB · 1 byte → 16 MiB", { size: 12 });
          G.caption("bytes scale directly with the dtype");
        } },

      { rail: "views", title: "A slice is a view; a copy is a new buffer",
        body: `<p><code>a[1]</code>, <code>a[:, 0]</code> and <code>a[::2]</code> do not copy. They return a <b>view</b>: a new shape and strides over the <i>same</i> buffer. Writing through a view changes the original.</p>
<div class="eq">b = a[:, 0]        # view: cells 0 and 3
b[0] = 99          # a[0, 0] is now 99
c = a[:, 0].copy() # new buffer, independent</div>
<p>A view is free. A copy costs time and memory. Fancy indexing (<code>a[[0, 2]]</code>, boolean masks) always copies. Check with <code>np.shares_memory(a, b)</code>.</p>`,
        check: { q: "<code>b = a[:, 0]; b[0] = 99</code>. What happens to <code>a</code>?",
          options: ["Nothing: b is a copy", "a[0, 0] becomes 99: b is a view", "NumPy raises an error"], answer: 1,
          why: "A basic slice returns a view over the same buffer, so the write lands in a. Use a[:, 0].copy() when you need independence." },
        scene(G) {
          G.label(24, 30, "buffer of a (2 × 3)", { size: 12 });
          const buf = buffer(G, 24, 44, 6, 98, { fill: (i) => (i === 0 || i === 3 ? "v" : "k") });
          G.label(24, 212, "view b = a[:, 0]  (cells 0 and 3: no new numbers)", { size: 12, color: "v" });
          G.box(24, 150, 90, 38, "b[0]", { stroke: "v", color: "v", size: 13 }); G.box(122, 150, 90, 38, "b[1]", { stroke: "v", color: "v", size: 13 });
          G.arrow(69, 150, 69, 84, { color: "v", w: 2 }); G.arrow(167, 150, 320, 84, { color: "v", w: 2 });
          G.text(240, 176, "write to b → a changes", { size: 13, color: "v" });
          G.label(24, 250, "copy c = a[:, 0].copy()  (its own buffer)", { size: 12, color: "hot" });
          G.box(24, 266, 90, 38, "0", { fill: "hot", size: 13 }); G.box(122, 266, 90, 38, "3", { fill: "hot", size: 13 });
          G.text(240, 292, "write to c → a unchanged", { size: 13, color: "hot" });
          G.text(24, 356, "np.shares_memory(a, b)  → True", { size: 13, color: "v" });
          G.text(24, 382, "np.shares_memory(a, c)  → False", { size: 13, color: "hot" });
          G.from(buf, { opacity: 0, stagger: 0.06, duration: 0.2 });
          G.caption("the view points into a's buffer; the copy owns new bytes");
        } },

      { rail: "reshape · transpose", title: "Transpose changes strides, not data",
        body: `<p><code>a.T</code> does not move a single number. It returns a view with the strides swapped.</p>
<div class="eq">a.shape, a.strides      # (2, 3) (12, 4)
a.T.shape, a.T.strides  # (3, 2) (4, 12)</div>
<p>Same buffer, new reading: the column of <code>a</code> becomes a row of <code>a.T</code>. The cost shows up later. Walking <code>a.T</code> row by row jumps 12 bytes at a time, which is the cache problem from P0.1. <code>np.ascontiguousarray(a.T)</code> pays for one real copy so later walks are smooth.</p>
<p><code>reshape</code> works the same way when the layout allows it (a view), and copies when it does not.</p>`,
        scene(G) {
          G.label(24, 30, "a  shape (2, 3)  strides (12, 4)", { size: 12 });
          const A = [[0, 1, 2], [3, 4, 5]];
          for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) G.box(24 + c * 58, 44 + r * 44, 54, 38, String(A[r][c]), { fill: r ? "q" : "k", size: 14 });
          G.label(330, 30, "a.T  shape (3, 2)  strides (4, 12)", { size: 12 });
          for (let r = 0; r < 3; r++) for (let c = 0; c < 2; c++) { const v = A[c][r]; G.box(330 + c * 58, 44 + r * 44, 54, 38, String(v), { fill: c ? "q" : "k", size: 14 }); }
          G.label(24, 214, "both read the same buffer:", { size: 13 });
          const cells = buffer(G, 24, 232, 6, 98, { offsets: true, fill: (i) => (i < 3 ? "k" : "q") });
          G.text(24, 340, "a  : next column = +4 B, next row = +12 B", { size: 13 });
          G.text(24, 366, "a.T: next column = +12 B, next row = +4 B", { size: 13, color: "v" });
          G.from(cells, { opacity: 0, stagger: 0.06, duration: 0.2 });
          G.caption("swap the strides and the grid reads transposed; no bytes move");
        } },

      { rail: "broadcasting", title: "Broadcasting stretches size-1 dimensions",
        body: `<p>To combine arrays of different shapes, NumPy lines the shapes up <b>from the right</b>. For each pair of dimensions the sizes must be <b>equal</b>, or one must be <b>1</b> (that one is stretched to match). A missing dimension counts as 1.</p>
<div class="eq">x (4, 3)  -  m (3,)    -> (4, 3)   ok
x (4, 3)  -  m (4,)    -> error    (3 vs 4)
x (4, 3)  -  m (4, 1)  -> (4, 3)   ok</div>
<p>Nothing is copied: the stretched axis gets a stride of 0, so the same numbers are read again. To subtract a per-row mean, keep that axis: <code>x.mean(axis=1, keepdims=True)</code> has shape <code>(4, 1)</code>.</p>`,
        check: { q: "<code>x</code> has shape (8, 3). <code>x.mean(axis=1)</code> has shape (8,). What does <code>x - x.mean(axis=1)</code> do?",
          options: ["Subtracts each row's mean from that row", "Raises an error: 3 does not match 8", "Subtracts the mean from every column"], answer: 1,
          why: "Shapes align from the right: (8, 3) against (8,) compares 3 with 8. They are neither equal nor 1, so NumPy raises an error. Use keepdims=True to get shape (8, 1)." },
        scene(G) {
          G.label(24, 30, "x (4, 3)  -  m (3,)", { size: 13 });
          const x = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 3; c++) x.push(G.box(24 + c * 52, 44 + r * 34, 48, 30, "x", { fill: "k", size: 12 }));
          G.text(190, 100, "-", { size: 22, color: "ink" });
          for (let c = 0; c < 3; c++) G.box(224 + c * 52, 76, 48, 30, "m" + c, { fill: "q", size: 12 });
          const ghost = [];
          for (let r = 1; r < 4; r++) for (let c = 0; c < 3; c++) ghost.push(G.box(224 + c * 52, 76 + r * 34, 48, 30, "m" + c, { stroke: "v", dash: "3 3", color: "v", size: 11 }));
          G.label(224, 232, "the one row m is read 4 times (stretched)", { size: 12, color: "v" });
          G.text(24, 270, "align from the right:", { size: 13 });
          [["x", "4", "3", "k"], ["m", "·", "3", "q"], ["=", "4", "3", "ok"]].forEach(([n, a, b, c], i) => {
            G.text(24, 306 + i * 30, n, { size: 14 });
            G.box(56, 288 + i * 30, 50, 26, a, { fill: a === "·" ? null : c, stroke: "line", size: 12, dash: a === "·" ? "3 3" : undefined });
            G.box(110, 288 + i * 30, 50, 26, b, { fill: c, size: 12 });
          });
          G.text(190, 330, "(4,3) vs (4,): 3 ≠ 4 → error", { size: 13, color: "hot" });
          G.from(ghost, { opacity: 0, stagger: 0.04, duration: 0.2 });
          G.caption("sizes must match, or be 1, reading from the right");
        } },

      { rail: "vectorise", title: "One array expression replaces a Python loop",
        body: `<p>Python runs a loop one element at a time, paying interpreter overhead on every step. A NumPy expression hands the whole array to compiled code that does the loop for you.</p>
<div class="eq">out = np.empty(n)
for i in range(n):         # n trips through Python
    out[i] = a[i] * 2 + b[i]

out = a * 2 + b            # one call, a compiled loop</div>
<p>How much faster? It depends on the array size, the dtype and your machine, so <b>measure it</b>:</p>
<div class="eq">t = time.perf_counter()
f()
print(time.perf_counter() - t, "s")</div>
<p>Vectorising often builds temporary arrays. A broadcast of <code>(n, 1, d)</code> against <code>(1, m, d)</code> makes an <code>(n, m, d)</code> intermediate, so it trades memory for speed. Exercise 4 has you do exactly that.</p>`,
        scene(G) {
          G.text(24, 36, "loop: Python visits each element", { size: 14 });
          const L = []; for (let i = 0; i < 10; i++) L.push(G.box(24 + i * 60, 52, 54, 36, String(i), { fill: "k", size: 13 }));
          for (let i = 0; i < 10; i++) G.arrow(51 + i * 60, 128, 51 + i * 60, 92, { color: "hot", w: 1.5 });
          G.label(24, 148, "10 trips through the interpreter (each arrow = one)", { size: 12, color: "hot" });
          G.text(24, 212, "vectorised: one call over the whole array", { size: 14 });
          const V = G.box(24, 228, 594, 44, "a * 2 + b", { fill: "ok", size: 16 });
          G.arrow(320, 330, 320, 276, { color: "ok", w: 2 });
          G.label(24, 330, "the loop runs inside compiled code", { size: 12, color: "ok" });
          G.text(24, 388, "how much faster? measure: time.perf_counter()", { size: 13, color: "v" });
          G.from(L, { opacity: 0, stagger: 0.07, duration: 0.2 }); G.from(V, { opacity: 0, duration: 0.5, delay: 0.5 });
          G.caption("fewer trips through Python, more work per call");
        } },

      { rail: "matmul", title: "Matrix multiply: shapes and 2·m·k·n FLOPs",
        body: `<p><code>A @ B</code> multiplies <code>(m, k)</code> by <code>(k, n)</code> and gives <code>(m, n)</code>. The inner sizes must match and disappear. Each output cell is a <b>dot product</b> of one row of A with one column of B: <code>k</code> multiplies and about <code>k</code> adds, counted as <code>2k</code> <b>FLOPs</b> (floating-point operations).</p>
<div class="eq">FLOPs = 2k per cell × (m × n) cells = 2 · m · k · n

(2, 3) @ (3, 4) -> (2, 4):  2 · 2 · 3 · 4 = 48 FLOPs</div>
<p>One token through <code>q_proj</code> is <code>(1, 4096) @ (4096, 4096)</code>: 2 × 1 × 4096 × 4096 = 33,554,432 FLOPs. Exercise 1 asks you to code this formula.</p>`,
        check: { q: "How many FLOPs is <code>(32, 4096) @ (4096, 4096)</code>?",
          options: ["134,217,728", "536,870,912", "1,073,741,824"], answer: 2,
          why: "2 × 32 × 4096 × 4096 = 64 × 16,777,216 = 1,073,741,824 (2^30). The other options are half and one eighth of that." },
        scene(G) {
          const cs = 40, gap = 3;
          const drawM = (x, y, rows, cols, f, lab) => { G.label(x, y - 8, lab, { size: 12 }); for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) G.rect(x + c * (cs + gap), y + r * (cs + gap), cs, cs, { fill: f(r, c), rx: 4, opacity: 0.9 }); };
          drawM(24, 60, 2, 3, (r) => (r === 0 ? "k" : "w"), "A  (2, 3)");
          G.text(158, 112, "@", { size: 22 });
          drawM(190, 60, 3, 4, (r, c) => (c === 1 ? "q" : "w"), "B  (3, 4)");
          G.text(386, 112, "=", { size: 22 });
          drawM(416, 60, 2, 4, (r, c) => (r === 0 && c === 1 ? "ok" : "w"), "C  (2, 4)");
          G.text(24, 232, "C[0, 1] = row 0 of A · column 1 of B", { size: 14 });
          G.text(24, 262, "= 3 multiplies + 3 adds ≈ 2 × 3 = 6 FLOPs", { size: 14 });
          G.text(24, 292, "C has 2 × 4 = 8 cells", { size: 14 });
          G.text(24, 330, "6 × 8 = 48 = 2 · m · k · n  (2·2·3·4)", { size: 15, color: "ok" });
          G.label(24, 380, "inner size k = 3 must match, and vanishes", { size: 12 });
          G.caption("each output cell costs 2k FLOPs; there are m × n of them");
        } },

      { rail: "pytest", title: "Floats need tolerances: assert_allclose",
        body: `<p>A <b>test</b> is a function named <code>test_...</code> that uses <code>assert</code>. Run it with <code>uv run pytest path</code> (R.3). But floats are not exact:</p>
<div class="eq">0.1 + 0.2 == 0.3          # False
0.1 + 0.2                 # 0.30000000000000004</div>
<p>Two correct computations that add numbers in a different order differ in the last bits. So compare with a tolerance:</p>
<div class="eq">np.testing.assert_allclose(actual, expected,
                           rtol=1e-6, atol=0)</div>
<p>It passes when <code>|actual − expected| ≤ atol + rtol × |expected|</code>. <code>rtol</code> is the relative error you allow. Add a small <code>atol</code> when the expected value can be 0, since a relative error against 0 is meaningless. float32 has about 7 decimal digits, so <code>rtol=1e-6</code> is tight for float32 and loose for float64.</p>`,
        check: { q: "Which check is right for comparing two float arrays produced by different code paths?",
          options: ["assert fast == reference", "np.testing.assert_allclose(fast, reference, rtol=1e-5, atol=1e-8)", "assert fast.shape == reference.shape"], answer: 1,
          why: "Valid computations can differ in the last bits, so exact == is fragile. assert_allclose with tolerances compares values; shape equality alone says nothing about the numbers." },
        scene(G) {
          G.text(24, 36, "0.1 + 0.2 == 0.3  →  False", { size: 15, color: "hot" });
          G.label(24, 58, "0.1 + 0.2 = 0.30000000000000004", { size: 12 });
          G.text(24, 120, "allclose: |actual − expected| ≤ atol + rtol·|expected|", { size: 13 });
          G.line(40, 220, 600, 220, { color: "muted", w: 2 });
          G.rect(260, 190, 120, 60, { fill: "ok", opacity: 0.25, rx: 6 });
          G.line(320, 180, 320, 260, { color: "ink", w: 2 });
          G.label(290, 172, "expected", { size: 12 }); G.label(290, 280, "band: ± rtol·|expected|", { size: 12 });
          G.circle(333, 220, 7, { fill: "ok" }); G.label(180, 304, "green: inside, pass", { size: 12, color: "ok" });
          G.circle(470, 220, 7, { fill: "hot" }); G.label(450, 304, "red: outside, fail", { size: 12, color: "hot" });
          G.text(24, 340, "assert_allclose(actual, expected, rtol=1e-6)", { size: 13 });
          G.text(24, 368, "never assert a == b on computed floats", { size: 13, color: "v" });
          G.caption("the shaded band is the tolerance");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>The course keeps a slow, trusted NumPy reference beside each fast engine piece, and tests compare them.</li>
<li>An ndarray is a flat typed buffer plus a shape plus strides. Bytes = elements × dtype size: a 4096 × 4096 matrix is 64 MiB in float32 and 32 MiB in bfloat16.</li>
<li>Slices and transposes are views (free). Fancy indexing and <code>.copy()</code> allocate.</li>
<li>Broadcasting aligns shapes from the right: equal or 1. <code>keepdims=True</code> keeps an axis broadcastable.</li>
<li>A matrix multiply of <code>(m, k) @ (k, n)</code> costs 2·m·k·n FLOPs. Compare floats with <code>assert_allclose</code>.</li></ul>
<h2>Go further</h2><ul>
<li>The Python tutorial: <a href="https://docs.python.org/3/tutorial/">docs.python.org/3/tutorial</a></li>
<li>NumPy, the absolute basics for beginners: <a href="https://numpy.org/doc/stable/user/absolute_beginners.html">numpy.org absolute beginners</a></li></ul>
<p>These links are unverified from the build environment. Next, R.5 covers the maths and ML that P1.1 assumes.</p>`,
    sim: {
      title: "Broadcasting checker",
      intro: "Choose two shapes of up to three dimensions each (leave a dimension at \"none\" to shorten the shape). The checker right-aligns them, marks each column as equal, stretched or clashing, and gives the result shape, the number of elements and the bytes for the dtype you pick.",
      height: 300,
      controls: [
        ...["a0", "a1", "a2"].map((id, i) => ({ id, label: `A, dimension ${i + 1}`, type: "select", value: [8, 1, 6][i], options: [[0, "none"], ...[1, 2, 3, 4, 5, 6, 7, 8].map((n) => [n, String(n)])] })),
        ...["b0", "b1", "b2"].map((id, i) => ({ id, label: `B, dimension ${i + 1}`, type: "select", value: [0, 7, 1][i], options: [[0, "none"], ...[1, 2, 3, 4, 5, 6, 7, 8].map((n) => [n, String(n)])] })),
        { id: "dt", label: "dtype of the result", type: "select", value: "float32", options: [["float32", "float32 (4 bytes)"], ["bfloat16", "bfloat16 (2 bytes)"], ["float16", "float16 (2 bytes)"], ["int8", "int8 (1 byte)"]] },
      ],
      draw(G, v) {
        const a = [v.a0, v.a1, v.a2].filter((n) => n > 0), b = [v.b0, v.b1, v.b2].filter((n) => n > 0);
        const r = bcast(a, b);
        const cx = (p) => 170 + p * 130;
        const KC = { same: "ok", stretch: "v", clash: "hot", none: "line" };
        G.label(16, 24, "shapes aligned from the right (3 slots)", { size: 11 });
        [["A", a, 70], ["B", b, 130]].forEach(([n, s, y]) => {
          G.text(24, y + 26, n, { size: 16, weight: 700 });
          for (let p = 0; p < 3; p++) {
            const i = s.length - 3 + p;
            const present = i >= 0;
            const col = r.cols[p];
            G.box(cx(p), y, 100, 44, present ? String(s[i]) : "·", { fill: present ? (col.kind === "stretch" && s[i] !== 1 ? "ok" : KC[col.kind]) : null, stroke: "line", size: 16, dash: present ? undefined : "4 3" });
            if (!present) G.label(cx(p) + 30, y + 62, "(counts as 1)", { size: 10 });
          }
        });
        G.line(16, 206, 624, 206, { color: "line", w: 1 });
        G.text(24, 246, "out", { size: 16, weight: 700 });
        for (let p = 0; p < 3; p++) {
          const o = r.out[p], k = r.cols[p].kind;
          G.box(cx(p), 222, 100, 44, k === "none" ? "" : o === null ? "x" : String(o), { fill: k === "none" ? null : o === null ? "hot" : "ok", stroke: "line", size: 16, dash: k === "none" ? "4 3" : undefined });
          G.label(cx(p) + 6, 286, k === "none" ? "" : k === "same" ? "equal" : k === "stretch" ? "stretch the 1" : "CLASH", { size: 11, color: k === "clash" ? "hot" : k === "stretch" ? "v" : "ok" });
        }
        const outShape = r.out.filter((x, p) => r.cols[p].kind !== "none");
        const n = r.ok ? outShape.reduce((s, x) => s * x, 1) : 0;
        const [dn, bytes] = DT[v.dt];
        const shapeStr = (s) => (s.length ? "(" + s.join(", ") + (s.length === 1 ? "," : "") + ")" : "()");
        return [
          { title: "Result", rows: [["shape of A", shapeStr(a)], ["shape of B", shapeStr(b)], ["result shape", r.ok ? shapeStr(outShape) : "error"]],
            chip: [r.ok, r.ok ? "broadcast works" : "ValueError: shapes do not broadcast"] },
          { title: `Size as ${dn}`, rows: [["elements", r.ok ? F.num(n) : "-"], ["bytes per element", String(bytes)], ["total", r.ok ? F.bytes(n * bytes) : "-"]],
            html: `<p class="note">The result is a new array. Broadcasting itself copies nothing: the inputs are read with stride 0 along stretched axes, but the output has to be stored.</p>` },
        ];
      },
    },
    practice: {
      intro: `Run from the repository root. Starters fail with <code>NotImplementedError</code> until you write them. Add <code>S2S_SOLUTIONS=1</code> in front to check the reference solutions.`,
      items: [
        { title: "Shapes and bytes", tier: "T0 · easy", goal: "Write nbytes(shape, dtype) and matmul_flops(m, k, n); check against the q_proj sizes.",
          cmd: "uv run pytest course/R-readiness/R.4-python-and-numpy/exercises/01-shapes-and-bytes" },
        { title: "Views, not copies", tier: "T0 · easy", goal: "Return views for a row, a column, a transpose and every other row. The test checks np.shares_memory and strides.",
          cmd: "uv run pytest course/R-readiness/R.4-python-and-numpy/exercises/02-views" },
        { title: "Broadcasting", tier: "T0 · medium", goal: "normalize_rows(x) and center_columns(x) with no loops, using keepdims and right-aligned shapes.",
          cmd: "uv run pytest course/R-readiness/R.4-python-and-numpy/exercises/03-broadcast" },
        { title: "Vectorise a loop", tier: "T0 · medium", goal: "Turn a double-loop pairwise-distance function into one broadcast expression that matches within rtol=1e-6, then time both.",
          cmd: "uv run pytest course/R-readiness/R.4-python-and-numpy/exercises/04-vectorize" },
      ],
      labs: [
        { label: "Module README: all ten sections", path: "course/R-readiness/R.4-python-and-numpy/README.md" },
        { label: "Self-check questions", path: "course/R-readiness/R.4-python-and-numpy/quiz.md" },
        { label: "A whole Llama forward pass in NumPy (read it once you finish)", path: "platform/engine/v0/reference/llama_numpy.py" },
      ],
    },
  });
})();
