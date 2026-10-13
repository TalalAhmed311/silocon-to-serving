/* R.5 — The maths and ML you need. Vectors, matrix products, softmax, layers, and the path from text to a next token. */
(function () {
  const F = S2S.fmt;
  const softmax = (z) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
  const silu = (x) => x / (1 + Math.exp(-x));
  // a small matrix drawn as numbered cells
  const mat = (G, x, y, M, cw, ch, fill, o = {}) => {
    M.forEach((row, r) => row.forEach((v, c) => {
      G.rect(x + c * cw, y + r * ch, cw - 3, ch - 3, { fill: typeof fill === "function" ? fill(r, c) : fill, rx: 4, opacity: 0.9 });
      G.text(x + c * cw + (cw - 3) / 2, y + r * ch + (ch - 3) / 2 + 4, String(v), { anchor: "middle", size: o.size ?? 13, weight: 700, color: "bg" });
    }));
  };

  S2S.lesson({
    id: "r-5", n: "R.5", title: "The maths and ML you need",
    subtitle: "Readiness · first principles · T0, any laptop",
    kicker: "Lesson · ≈ 40 min",
    headline: "Two multiplications and a curve",
    intro: `<p><b>Already comfortable with vectors, matrix products, softmax and how a language model turns text into a next token? Answer the checkpoints. If you get them all right, skip to R.6.</b></p>
<p>A language model looks mysterious until you see what it is made of. It is made of three things: multiplying a matrix by a vector, bending the result with a simple curve, and turning a list of scores into probabilities. This lesson teaches those three things and then follows a piece of text through a model, from words to a chosen next word. It does not teach the transformer itself. That is P1.1, and after this lesson you can read it.</p>
<p>You need school maths and the ability to read a loop. Scroll: the dark panel redraws for each step.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "input / first"], ["v", "second"], ["q", "third"], ["w", "weights"], ["ok", "result"], ["hot", "problem"]],
    prev: "r-4", next: "r-6",
    steps: [
      { rail: "dot product", title: "A dot product says how aligned two lists are",
        body: `<p>A <b>vector</b> is a list of numbers. Draw <code>[3, 1]</code> as an arrow: 3 steps right, 1 step up. A language model stores everything as vectors: a word, a position, a guess.</p>
<p>To compare two vectors, multiply them position by position and add. That is the <b>dot product</b>:</p>
<div class="eq">[3, 1] · [2, 2] = 3·2 + 1·2 = 8</div>
<p>The result is large and positive when the arrows point the same way. It is <b>0</b> when they are at right angles. It is <b>negative</b> when they point apart. Attention, the heart of a transformer, is a dot product between a "question" vector and many "label" vectors: the biggest result is the best match.</p>`,
        scene(G) {
          const ox = 150, oy = 290, s = 40, P = (x, y) => [ox + x * s, oy - y * s];
          G.line(30, oy, 310, oy, { color: "line" }); G.line(ox, oy + 40, ox, 100, { color: "line" });
          const arr = (v, col, lab, dx, dy) => { const [x, y] = P(...v); const a = G.arrow(ox, oy, x, y, { color: col, w: 3 }); G.text(x + dx, y + dy, lab, { size: 13, color: col, weight: 700 }); return a; };
          const a1 = arr([3, 1], "k", "a", 8, 4);
          const a2 = arr([2, 2], "v", "b", 8, -2);
          const a3 = arr([-1, 3], "q", "c", -16, -4);
          const a4 = arr([-3, -1], "hot", "−a", -8, 18);
          G.text(350, 100, "a = [3, 1]", { size: 14, weight: 700, color: "k" });
          G.text(350, 124, "b = [2, 2]    c = [−1, 3]", { size: 13 });
          const rows = [["a · b = 3·2 + 1·2 = 8", "similar direction: big and positive", "ok"], ["a · c = 3·(−1) + 1·3 = 0", "right angle: no match", "muted"], ["a · (−a) = −9 − 1 = −10", "opposite: negative", "hot"]];
          rows.forEach(([eq, why, col], i) => { G.text(350, 180 + i * 62, eq, { size: 14 }); G.text(350, 200 + i * 62, why, { size: 12, color: col }); });
          G.from([a1, a2, a3, a4], { opacity: 0, stagger: 0.25, duration: 0.3 });
          G.caption("dot product = how much two arrows agree");
        } },

      { rail: "matrix × vector", title: "A matrix times a vector is one dot product per row",
        body: `<p>A <b>matrix</b> is a grid of numbers: a stack of row vectors. Multiplying a matrix <code>W</code> (3 rows, 4 columns) by a vector <code>x</code> (4 numbers) means: dot each row with <code>x</code>. You get one number per row, so a vector of 3.</p>
<div class="eq">row 1 · x = 3·2 + 0·1 + 1·3 + 2·1
        = 6 + 0 + 3 + 2 = 11</div>
<p>How much work is that? Each multiply-and-add pair counts as <b>2 FLOPs</b> (floating-point operations). A row has 4 pairs, so 8 FLOPs. Three rows: 24. In general an <code>m × n</code> matrix times a vector costs <code>2·m·n</code> FLOPs, which is two FLOPs for every number in the matrix. Hold on to that: it is why a model with 8 billion numbers needs about 16 billion FLOPs per token.</p>`,
        check: { q: "A 1000 × 500 matrix is multiplied by a vector of 500 numbers. How many FLOPs is that?",
          options: ["500,000", "1,000,000", "1,500", "250,000"], answer: 1,
          why: "There are 1000 rows. Each row is a dot product of length 500, which is 500 multiply-adds, or 1,000 FLOPs. 1000 rows × 1,000 FLOPs = 1,000,000. That is 2·m·n = 2 × 1000 × 500." },
        scene(G) {
          const W = [[1, 2, 0, -1], [3, 0, 1, 2], [0, 1, 1, 1]], x = [2, 1, 3, 1], y = [3, 11, 5];
          G.label(24, 96, "W  (3 × 4)", { size: 12 });
          mat(G, 24, 106, W, 46, 42, (r) => (r === 1 ? "k" : "w"));
          G.text(222, 168, "×", { size: 22, anchor: "middle" });
          G.label(246, 96, "x  (4)", { size: 12 });
          mat(G, 246, 106, x.map((v) => [v]), 46, 42, "v");
          G.text(316, 168, "=", { size: 22, anchor: "middle" });
          G.label(340, 96, "y  (3)", { size: 12 });
          mat(G, 340, 106, y.map((v) => [v]), 46, 42, (r) => (r === 1 ? "ok" : "w"));
          const hl = G.rect(20, 148, 190, 42, { stroke: "ink", sw: 2, rx: 6 });
          G.pulse(hl, { lo: 0.3, repeat: 6 });
          G.text(410, 128, "row 1 · x", { size: 13, weight: 700 });
          G.text(410, 150, "3·2 + 0·1 + 1·3 + 2·1", { size: 13 });
          G.text(410, 172, "= 6 + 0 + 3 + 2 = 11", { size: 13, color: "ok" });
          G.rect(20, 296, 600, 108, { stroke: "line", rx: 8 });
          G.text(36, 324, "cost: 3 rows × 4 multiply-adds × 2 FLOPs = 24 FLOPs", { size: 14 });
          G.text(36, 350, "general rule: 2 · m · n  (2 FLOPs per number in the matrix)", { size: 14 });
          G.text(36, 380, "one 4096 × 4096 matrix: 2 · 4096 · 4096 = 33,554,432 FLOPs", { size: 14, color: "k" });
          G.caption("a matrix-vector product is many dot products");
        } },

      { rail: "matrix × matrix", title: "A matrix times a matrix does that for many vectors at once",
        body: `<p>To process several vectors together, stack them as the rows of a matrix. A product of shapes <code>(m, k) @ (k, n)</code> gives shape <code>(m, n)</code>. The two <code>k</code>s must match, and they disappear. Output cell <code>(i, j)</code> is row <code>i</code> of the left matrix dotted with column <code>j</code> of the right one.</p>
<p>There are <code>m·n</code> output cells. Each is a dot product of length <code>k</code>, which is <code>2k</code> FLOPs. Total:</p>
<div class="eq">FLOPs = 2 · m · k · n
example: 2 · 2 · 3 · 4 = 48</div>
<p>This is the whole difference between reading a prompt and writing an answer. Reading a 512-token prompt through one 4096 × 4096 matrix is <code>(512, 4096) @ (4096, 4096)</code>: 17,179,869,184 FLOPs, about 17.2 billion. Writing one token is the matrix-vector case: 33,554,432. The first does 512 times the arithmetic on the same weights.</p>`,
        check: { q: "What is the shape of (64, 128) @ (128, 32), and how many FLOPs does it cost?",
          options: ["(64, 32), 524,288 FLOPs", "(64, 32), 262,144 FLOPs", "(128, 128), 524,288 FLOPs", "(64, 128), 1,048,576 FLOPs"], answer: 0,
          why: "The inner 128s match and vanish, leaving (64, 32). Cost 2·m·k·n = 2 × 64 × 128 × 32 = 524,288. The option with 262,144 forgot the factor of 2." },
        scene(G) {
          const A = [[1, 2, 0], [0, 3, 1]], B = [[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1]];
          const C = [[0, 0, 0, 0], [0, 0, 0, 0]];
          for (let i = 0; i < 2; i++) for (let j = 0; j < 4; j++) for (let t = 0; t < 3; t++) C[i][j] += A[i][t] * B[t][j];
          G.label(24, 74, "A  (m=2, k=3)", { size: 12 }); mat(G, 24, 84, A, 36, 34, (r) => (r === 1 ? "k" : "w"), { size: 12 });
          G.text(148, 128, "@", { size: 20, anchor: "middle" });
          G.label(172, 54, "B  (k=3, n=4)", { size: 12 }); mat(G, 172, 64, B, 36, 34, (r, c) => (c === 2 ? "v" : "w"), { size: 12 });
          G.text(334, 128, "=", { size: 20, anchor: "middle" });
          G.label(358, 74, "C  (m=2, n=4)", { size: 12 }); mat(G, 358, 84, C, 36, 34, (r, c) => (r === 1 && c === 2 ? "ok" : "w"), { size: 12 });
          G.text(24, 188, "C[1][2] = row 1 of A · column 2 of B = 0·2 + 3·1 + 1·0 = 3", { size: 13, color: "ok" });
          G.text(24, 226, "(m, k) @ (k, n) → (m, n)      cost 2·m·k·n", { size: 14, weight: 700 });
          G.text(24, 252, "here: 2 · 2 · 3 · 4 = 48 FLOPs", { size: 13, color: "muted" });
          G.rect(20, 286, 600, 126, { stroke: "line", rx: 8 });
          G.text(36, 312, "same weights, 4096 × 4096:", { size: 13, color: "muted" });
          G.text(36, 340, "write 1 token   (1, 4096) @ (4096, 4096) =  33.6 million FLOPs", { size: 13, color: "k" });
          G.text(36, 366, "read 512 tokens (512, 4096) @ (4096, 4096) = 17.2 billion FLOPs", { size: 13, color: "v" });
          G.text(36, 394, "512 times the arithmetic for one pass over the weights", { size: 13 });
          G.caption("shapes tell you the cost before you run anything");
        } },

      { rail: "exp and log", title: "exp turns any score into a positive weight",
        body: `<p>Scores can be negative or huge. Before we turn them into shares of a whole, we need them all positive. The function <code>exp(x)</code> (that is <code>e^x</code>, with <code>e ≈ 2.718</code>) does that. It is positive for every input, it never reaches zero, and it grows fast: raising the score by 1 multiplies the weight by 2.718.</p>
<table><tr><th>x</th><th>−1</th><th>0</th><th>1</th><th>2</th><th>3</th></tr><tr><td>exp(x)</td><td>0.368</td><td>1</td><td>2.718</td><td>7.389</td><td>20.09</td></tr></table>
<p><code>log</code> (the natural log, <code>ln</code>) undoes <code>exp</code>. Its useful trick: it turns products into sums. <code>ln(2 × 3) = ln 2 + ln 3 = 0.693 + 1.099 = 1.792</code>. Probabilities of many events multiply and shrink toward zero; their logs add and stay readable. That is why models report scores in log form.</p>`,
        scene(G) {
          const px = (x) => 320 + x * 90, py = (v) => 330 - v * 14;
          G.line(40, 330, 600, 330, { color: "line" }); G.line(320, 330, 320, 40, { color: "line" });
          G.text(600, 350, "x", { size: 12, color: "muted", anchor: "end" }); G.text(330, 48, "exp(x)", { size: 12, color: "muted" });
          let d = "";
          for (let i = 0; i <= 60; i++) { const x = -3 + i * 0.1; d += (i ? "L" : "M") + px(x).toFixed(1) + " " + py(Math.exp(x)).toFixed(1); }
          const curve = G.path(d, { color: "k", w: 3 });
          G.line(px(-3), py(1), px(3), py(1), { color: "line", dash: "4 4" });
          [[-1, 0.368], [0, 1], [1, 2.718], [2, 7.389], [3, 20.09]].forEach(([x, v]) => {
            G.circle(px(x), py(v), 5, { fill: "v" });
            G.text(px(x), 346, String(x), { size: 11, anchor: "middle", color: "muted" });
            G.text(px(x) + (x >= 1 ? -10 : 10), py(v) - 8, String(v), { size: 12, color: "ink", anchor: x >= 1 ? "end" : "start" });
          });
          G.text(46, 316, "never below 0:", { size: 12, color: "ok" }); G.text(46, 300, "positive for every x", { size: 12, color: "ok" });
          G.text(24, 392, "ln(2 × 3) = ln 2 + ln 3 = 0.693 + 1.099 = 1.792", { size: 13 });
          G.text(24, 414, "exp: scores → positive weights.   ln: products → sums.", { size: 12, color: "muted" });
          G.from(curve, { opacity: 0, duration: 0.5 });
          G.caption("each +1 on the score multiplies the weight by 2.718");
        } },

      { rail: "softmax", title: "Softmax: exp, then divide by the total",
        body: `<p>A model's final output is a list of scores, one per possible next word. We want <b>probabilities</b>: positive numbers that add to 1. <b>Softmax</b> does it in two moves. Take <code>exp</code> of each score. Divide each by the sum.</p>
<div class="eq">scores        [2, 1, 0]
exp           [7.389, 2.718, 1]      sum = 11.107
probabilities [0.665, 0.245, 0.090]  sum = 1.000</div>
<p>There is a trap. <code>exp</code> overflows a float32 near <code>exp(89)</code> (the largest float32 is about 3.4e38). A score of 1000 gives <code>inf</code>, and <code>inf / inf</code> is <code>nan</code>. The fix is a free lunch: subtract the largest score from every score first. The ratios do not change, because the same factor cancels on top and bottom, but now every exponent is 0 or below, so nothing can overflow. Every real implementation does this.</p>`,
        check: { q: "Softmax subtracts the maximum score before taking exp. What does that do to the output probabilities?",
          options: ["Changes them slightly, but avoids overflow", "Nothing: it only prevents overflow", "Makes them add up to 1", "Sets the largest one to 0"], answer: 1,
          why: "exp(x − m) = exp(x) · exp(−m). The factor exp(−m) is the same for every score, so it cancels in the division. The probabilities are identical. They already sum to 1 either way. What changes is that no exponent can be positive, so exp cannot overflow." },
        scene(G) {
          const s = [2, 1, 0], e = s.map(Math.exp), tot = e.reduce((a, b) => a + b), p = e.map((v) => v / tot);
          const grp = (x, vals, max, col, lab, fmt) => {
            const b = G.bars(x, 220, vals, { w: 34, gap: 10, h: 110, max, fill: (i) => col[i] });
            vals.forEach((v, i) => G.text(x + i * 44 + 17, 214 - (110 * v) / max, fmt(v), { size: 12, anchor: "middle", color: "ink" }));
            ["A", "B", "C"].forEach((n, i) => G.label(x + i * 44 + 17, 240, n, { anchor: "middle", size: 12 }));
            G.text(x + 66, 270, lab, { size: 13, anchor: "middle", weight: 700 });
            return b;
          };
          const g1 = grp(30, s.map((v) => Math.max(v, 0.001)), 2, ["k", "v", "q"], "scores", (v) => String(Math.round(v)));
          const g2 = grp(236, e, e[0], ["k", "v", "q"], "exp(score)", (v) => v.toFixed(3));
          const g3 = grp(442, p, 1, ["k", "v", "q"], "÷ 11.107 → probability", (v) => v.toFixed(3));
          G.arrow(176, 170, 226, 170, { color: "muted", w: 2 }); G.arrow(382, 170, 432, 170, { color: "muted", w: 2 });
          G.rect(20, 296, 600, 116, { stroke: "line", rx: 8 });
          G.text(36, 322, "stable version: subtract the max (2) first", { size: 13, weight: 700 });
          G.text(36, 346, "scores − 2 = [0, −1, −2]  →  exp = [1, 0.368, 0.135]  sum 1.503", { size: 13 });
          G.text(36, 370, "÷ 1.503 = [0.665, 0.245, 0.090]   the same answer", { size: 13, color: "ok" });
          G.text(36, 396, "scores [1000, 1001]: naive exp(1000) = inf → nan.  Stable: [0.269, 0.731]", { size: 12, color: "hot" });
          G.from([...g1, ...g2, ...g3], { opacity: 0, stagger: 0.07, duration: 0.25 });
          G.caption("softmax keeps the order of the scores and makes them add to 1");
        } },

      { rail: "layers", title: "A layer: multiply, add, bend",
        body: `<p>One layer of a neural network does three things to its input vector <code>x</code>:</p>
<div class="eq">y = activation( W x + b )</div>
<ol><li><b>Multiply</b> by a weight matrix <code>W</code> (step 2).</li><li><b>Add</b> a bias vector <code>b</code>, a fixed shift for each output.</li><li><b>Bend</b> every number with an <b>activation</b>, a simple curve.</li></ol>
<p>Why bend? Without it, two stacked layers are just one bigger matrix, since a matrix times a matrix is a matrix. The curve is what lets a stack learn more than straight-line rules. <b>ReLU</b> is <code>max(0, x)</code>: negatives become 0. <b>SiLU</b> is <code>x · sigmoid(x) = x / (1 + e^−x)</code>: a smooth version that dips just below zero before rising. Llama-style models use SiLU.</p>
<p>Worked example: <code>x = [1, 2]</code>, <code>w = [0.5, −1]</code>, <code>b = 0.5</code>. Then <code>z = 0.5·1 + (−1)·2 + 0.5 = −1</code>, so ReLU gives 0 and SiLU gives <code>−1 / (1 + 2.718) = −0.269</code>.</p>`,
        scene(G) {
          const px = (x) => 205 + x * 43.75, py = (v) => 310 - v * 60;
          G.line(30, 310, 380, 310, { color: "line" }); G.line(205, 360, 205, 60, { color: "line" });
          G.text(380, 330, "x", { size: 12, color: "muted", anchor: "end" });
          G.path(`M${px(-4)} ${py(0)} L${px(0)} ${py(0)} L${px(4)} ${py(4)}`, { color: "k", w: 3 });
          let d = ""; for (let i = 0; i <= 80; i++) { const x = -4 + i * 0.1; d += (i ? "L" : "M") + px(x).toFixed(1) + " " + py(silu(x)).toFixed(1); }
          G.path(d, { color: "v", w: 3 });
          [-4, -2, 2, 4].forEach((x) => G.text(px(x), 328, String(x), { size: 11, anchor: "middle", color: "muted" }));
          G.circle(px(-1), py(silu(-1)), 5, { fill: "hot" });
          G.text(px(-1) - 8, py(silu(-1)) + 22, "−0.269", { size: 12, anchor: "end", color: "hot" });
          G.rect(40, 96, 10, 10, { fill: "k", rx: 2 }); G.text(56, 106, "ReLU = max(0, x)", { size: 12 });
          G.rect(40, 118, 10, 10, { fill: "v", rx: 2 }); G.text(56, 128, "SiLU = x / (1 + e^−x)", { size: 12 });
          G.text(410, 100, "One neuron", { size: 14, weight: 700 });
          [["x = [1, 2]", "ink"], ["w = [0.5, −1]   b = 0.5", "ink"], ["z = 0.5 − 2 + 0.5 = −1", "ink"], ["ReLU(−1) = 0", "k"], ["SiLU(−1) = −0.269", "v"]].forEach(([t, c], i) => G.text(410, 130 + i * 28, t, { size: 13, color: c }));
          G.text(410, 300, "A layer is m of these", { size: 13, weight: 700 });
          G.text(410, 322, "side by side: one per row", { size: 13 });
          G.text(410, 344, "of W. W x does them all.", { size: 13 });
          G.caption("the curve is what stops stacked layers collapsing into one matrix");
        } },

      { rail: "parameters", title: "A network is layers stacked; the parameters are the numbers inside",
        body: `<p>Feed the output of one layer into the next and you have a <b>network</b>. The <b>parameters</b> (also called weights) are every number inside every <code>W</code> and <code>b</code>. Training sets them. Inference reads them.</p>
<p>Count them for a small two-layer network that reads a 784-number input, has 128 hidden numbers and gives 10 outputs:</p>
<div class="eq">layer 1: W 128×784 = 100,352    b = 128
layer 2: W  10×128 =   1,280    b =  10
total = 100,352 + 128 + 1,280 + 10 = 101,770</div>
<p>The input and the in-between vectors are <i>activations</i>, not parameters: they change with every input. Llama-3-8B is the same recipe on a larger scale. It has about 8.03 billion parameters, which at 2 bytes each is about 16 GB of weights that must be read to produce each token.</p>`,
        check: { q: "A network has layers 10 → 20 → 5, each with a bias. How many parameters does it have?",
          options: ["300", "320", "325", "1,300"], answer: 2,
          why: "Layer 1: 20×10 weights + 20 biases = 220. Layer 2: 5×20 weights + 5 biases = 105. Total 325. The 320 forgets the second layer's biases; the 300 forgets all biases." },
        scene(G) {
          G.box(24, 100, 72, 130, "784", { fill: "k", size: 18 });
          G.box(256, 130, 72, 70, "128", { fill: "v", size: 18 });
          G.box(480, 155, 72, 20, "", { fill: "ok" }); G.text(516, 192, "10", { size: 18, anchor: "middle", weight: 700, color: "ok" });
          G.label(24, 90, "input", { size: 12 }); G.label(292, 222, "hidden (after SiLU)", { size: 12, anchor: "middle" }); G.label(480, 145, "output", { size: 12 });
          G.arrow(100, 165, 252, 165, { color: "muted", w: 2 }); G.arrow(332, 165, 476, 165, { color: "muted", w: 2 });
          G.text(176, 120, "W1: 128 × 784", { size: 13, anchor: "middle", weight: 700 }); G.text(176, 142, "= 100,352", { size: 13, anchor: "middle" }); G.text(176, 196, "+ b1: 128", { size: 13, anchor: "middle", color: "muted" });
          G.text(404, 115, "W2: 10 × 128", { size: 13, anchor: "middle", weight: 700 }); G.text(404, 137, "= 1,280", { size: 13, anchor: "middle" }); G.text(404, 196, "+ b2: 10", { size: 13, anchor: "middle", color: "muted" });
          G.rect(20, 262, 600, 150, { stroke: "line", rx: 8 });
          G.text(36, 292, "100,352 + 128 + 1,280 + 10 = 101,770 parameters", { size: 15, weight: 700 });
          G.text(36, 324, "at 2 bytes each (bf16): about 204 KB", { size: 13 });
          G.text(36, 356, "Llama-3-8B: ≈ 8.03 billion parameters → ≈ 16 GB in bf16", { size: 13, color: "k" });
          G.text(36, 388, "same recipe: multiply, add, bend, repeat 32 times", { size: 13, color: "muted" });
          G.caption("parameters = numbers in W and b, not the data flowing through");
        } },

      { rail: "train vs infer", title: "Training changes the weights. Inference only reads them.",
        body: `<p>Where do the weights come from? From <b>training</b>. Show the network an example, run it forward, measure how wrong the output is (the <b>loss</b>), run backward to find which weights to nudge, nudge them, and repeat for trillions of tokens. The weights change at every step. It costs months of GPU time and is done once by the lab that makes the model.</p>
<p><b>Inference</b> is what happens afterwards. The weights are a fixed file. For each request we run only the forward pass and read the result. Nothing is learned or updated.</p>
<p>This course is about inference. Everything we optimise is about running a forward pass with fixed weights as cheaply and quickly as possible: how to store the weights, how to move them, and how to share one copy among many users.</p>`,
        scene(G) {
          G.text(170, 44, "Training", { size: 16, weight: 700, anchor: "middle" });
          G.text(470, 44, "Inference", { size: 16, weight: 700, anchor: "middle" });
          G.line(320, 30, 320, 330, { color: "line", dash: "4 4" });
          const tr = ["example", "forward pass", "loss: how wrong?", "backward: who to blame?", "update weights"];
          tr.forEach((t, i) => { G.box(80, 66 + i * 54, 180, 36, t, { fill: i === 4 ? "hot" : "w", size: 12 }); if (i) G.arrow(170, 66 + i * 54 - 14, 170, 66 + i * 54, { color: "muted" }); });
          G.path("M262 302 Q300 302 300 186 Q300 84 264 84", { color: "muted", w: 2, arrow: true, dash: "5 4" });
          G.label(38, 350, "repeat for trillions of tokens", { size: 12 });
          const inf = ["prompt", "forward pass", "next token"];
          inf.forEach((t, i) => { G.box(380, 66 + i * 54, 180, 36, t, { fill: i === 1 ? "w" : "k", size: 12 }); if (i) G.arrow(470, 66 + i * 54 - 14, 470, 66 + i * 54, { color: "muted" }); });
          G.path("M562 192 Q600 192 600 130 Q600 84 564 84", { color: "muted", w: 2, arrow: true, dash: "5 4" });
          G.label(380, 244, "repeat per generated token", { size: 12 });
          G.box(60, 372, 220, 36, "weights: change every step", { stroke: "hot", color: "hot", size: 12 });
          G.box(360, 372, 220, 36, "weights: fixed, read-only", { stroke: "ok", color: "ok", size: 12 });
          G.caption("the course starts after training has finished");
        } },

      { rail: "text to token", title: "From text to a next token",
        body: `<p>Now put the pieces in a row. A model only knows numbers, so text goes through six steps:</p>
<ol><li><b>Tokenise.</b> Cut the text into <b>tokens</b>: words or word pieces.</li>
<li><b>Look up ids.</b> Each token has an integer <b>id</b> in a fixed list called the <b>vocabulary</b>. Llama-3-8B's has <b>128,256</b> entries.</li>
<li><b>Embed.</b> The id picks one row of a big table (128,256 rows × 4,096 columns, about 525 million numbers). The token is now a vector of 4,096 numbers.</li>
<li><b>Layers.</b> 32 layers refine the vectors. P1.1 opens them up.</li>
<li><b>Scores.</b> One last matrix turns the final vector into 128,256 numbers, one per vocabulary entry. They are called <b>logits</b>.</li>
<li><b>Softmax</b> turns logits into probabilities, and one token is chosen.</li></ol>
<p>That token is added to the text and the whole thing runs again for the next one. Writing an answer of 500 tokens means 500 trips through this pipeline.</p>`,
        scene(G) {
          const top = [["text", "\"The cat sat\""], ["tokens", "The | cat | sat"], ["ids", "3 integers"], ["embed", "3 × 4,096"]];
          const bot = [["32 layers", "same shape each"], ["logits", "128,256 scores"], ["softmax", "probabilities"], ["pick", "\" on\""]];
          const draw = (row, y, cols) => row.map(([t, sub], i) => {
            const x = 44 + i * 144;
            const b = G.box(x, y, 120, 50, t, { fill: cols[i], size: 14 });
            G.label(x + 60, y + 70, sub, { anchor: "middle", size: 12, color: "ink" });
            if (i) G.arrow(x - 22, y + 25, x - 2, y + 25, { color: "muted", w: 2 });
            return b;
          });
          const a = draw(top, 60, ["k", "k", "v", "q"]);
          const b = draw(bot, 230, ["w", "v", "ok", "ok"]);
          G.path("M584 160 L584 190 L104 190 L104 224", { color: "muted", w: 2, arrow: true });
          G.label(44, 340, "ids are made up for the picture; the vocabulary size is real", { size: 12 });
          G.label(44, 362, "the logits for the LAST position decide the next token", { size: 12 });
          G.label(44, 392, "128,256 × 4,096 = 525,336,576 numbers in the embedding table", { size: 12, color: "ink" });
          G.from([...a, ...b], { opacity: 0, stagger: 0.12, duration: 0.25 });
          G.caption("text → numbers → vectors → scores → one token");
        } },

      { rail: "sampling", title: "Picking the token: greedy, sampling and temperature",
        body: `<p>Softmax gives a probability for every token. How do we pick one? Two basic ways:</p>
<ul><li><b>Greedy:</b> always take the most likely token. It is predictable, and it can repeat itself.</li>
<li><b>Sampling:</b> draw at random, in proportion to the probabilities. A token with 0.24 is chosen about 24% of the time.</li></ul>
<p><b>Temperature</b> <code>T</code> controls how adventurous sampling is. Divide the logits by <code>T</code> before softmax. With logits <code>[3, 2, 1, 0]</code>:</p>
<div class="eq">T = 0.5   [0.865, 0.117, 0.016, 0.002]   sharper
T = 1     [0.644, 0.237, 0.087, 0.032]
T = 2     [0.455, 0.276, 0.167, 0.102]   flatter</div>
<p>As <code>T</code> goes to 0 the top token takes everything and sampling becomes greedy. High <code>T</code> flattens the bars and surprises appear. Serving systems expose <code>temperature</code> as a request field (R.6 shows the JSON). You write a sampler in P0.5, and P2 covers the server options.</p>`,
        check: { q: "The logits are [2, 1, 0]. The temperature is lowered from 1 to 0.5. What happens to the top token's probability?",
          options: ["It falls", "It rises, from about 0.665 to about 0.867", "It stays at 0.665", "It becomes exactly 1"], answer: 1,
          why: "Dividing by 0.5 doubles the logits to [4, 2, 0]. Softmax of that is [0.867, 0.117, 0.016], so the gap between the top token and the rest gets wider. It never becomes exactly 1 for a finite T, though it gets very close as T shrinks." },
        scene(G) {
          const lg = [3, 2, 1, 0], names = ["on", "in", "at", "the"];
          [[0.5, "T = 0.5  sharper", 30], [1, "T = 1", 240], [2, "T = 2  flatter", 450]].forEach(([T, lab, x]) => {
            const p = softmax(lg.map((v) => v / T));
            G.bars(x, 300, p, { w: 30, gap: 10, h: 190, max: 1, fill: (i) => (i === 0 ? "ok" : "k") });
            p.forEach((v, i) => { G.text(x + i * 40 + 15, 294 - 190 * v, v.toFixed(3), { size: 11, anchor: "middle" }); G.label(x + i * 40 + 15, 320, names[i], { anchor: "middle", size: 12 }); });
            G.text(x + 75, 60, lab, { size: 14, weight: 700, anchor: "middle" });
          });
          G.text(24, 364, "greedy: always the tallest bar (\"on\")", { size: 13 });
          G.text(24, 388, "sampling: draw in proportion to the bars", { size: 13 });
          G.text(24, 412, "temperature: divide the logits by T first", { size: 13, color: "muted" });
          G.caption("lower T makes the tallest bar win more often");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A dot product measures agreement. A matrix times a vector is one dot product per row and costs <code>2·m·n</code> FLOPs; a matrix times a matrix costs <code>2·m·k·n</code>.</li>
<li><code>exp</code> makes any score positive. Softmax divides the exponentials by their sum, and subtracts the maximum first so it cannot overflow.</li>
<li>A layer is <code>activation(W x + b)</code>. A network is layers stacked, and its parameters are the numbers in all the <code>W</code> and <code>b</code>.</li>
<li>Training changes the weights; inference only reads them. This course is about inference.</li>
<li>Text becomes token ids, then embedding vectors, then (after the layers) one logit per vocabulary entry; softmax and a sampling rule pick the next token.</li></ul>
<p><b>Next:</b> R.6 (HTTP), then R.7 (a safe cloud account), then the course proper. P1.1 uses everything on this page.</p>
<h2>Go further</h2><ul>
<li><a href="https://www.3blue1brown.com/topics/linear-algebra">3Blue1Brown, Essence of linear algebra</a>: the pictures behind vectors and matrices.</li>
<li><a href="https://www.3blue1brown.com/topics/neural-networks">3Blue1Brown, Neural networks</a>: layers, weights and training, animated.</li>
<li><a href="https://karpathy.ai/zero-to-hero.html">Karpathy, Neural Networks: Zero to Hero</a>: build a language model from scratch, in code.</li>
<li><a href="https://numpy.org/doc/stable/user/absolute_beginners.html">NumPy: the absolute basics for beginners</a>: the array tool used in the exercises.</li></ul>`,
    sim: {
      title: "Softmax and temperature",
      intro: "Set five logits, a temperature and an optional shift added to every logit. The bars are the probabilities. The shift changes nothing in the answer, because softmax ignores a constant added to all scores, but it makes the naive formula blow up in float32. The panels compare the naive and the stable computation.",
      height: 270,
      controls: [
        { id: "a", label: "logit A (\"the\")", min: -20, max: 20, step: 0.5, value: 3 },
        { id: "b", label: "logit B (\"cat\")", min: -20, max: 20, step: 0.5, value: 2 },
        { id: "c", label: "logit C (\"sat\")", min: -20, max: 20, step: 0.5, value: 1 },
        { id: "d", label: "logit D (\"mat\")", min: -20, max: 20, step: 0.5, value: 0 },
        { id: "e", label: "logit E (\"end\")", min: -20, max: 20, step: 0.5, value: -1 },
        { id: "t", label: "temperature T", min: 0.1, max: 3, step: 0.05, value: 1, format: (v) => v.toFixed(2) },
        { id: "s", label: "shift added to every logit", min: 0, max: 200, step: 5, value: 0 },
      ],
      draw(G, v) {
        const names = ["the", "cat", "sat", "mat", "end"], raw = [v.a, v.b, v.c, v.d, v.e];
        const z = raw.map((x) => (x + v.s) / v.t), p = softmax(z);
        const zmax = Math.max(...z), FMAX = 3.4028235e38;
        const naiveE = z.map(Math.exp), naiveSum = naiveE.reduce((a, b) => a + b, 0);
        const overflow = naiveE.some((x) => x > FMAX) || naiveSum > FMAX, underflow = !overflow && naiveSum === 0;
        const H = -p.reduce((a, x) => a + (x > 0 ? x * Math.log2(x) : 0), 0), Hmax = Math.log2(5);
        G.label(16, 18, "probability of each next token", { size: 11 });
        G.axes(16, 28, 380, 190, {});
        const cols = p.map((x) => (x === Math.max(...p) ? "ok" : "k"));
        G.bars(36, 218, p, { w: 52, gap: 18, h: 170, max: 1, fill: (i) => cols[i] });
        p.forEach((x, i) => { G.text(36 + i * 70 + 26, 212 - 170 * x, (100 * x).toFixed(1) + "%", { size: 11, anchor: "middle" }); G.label(36 + i * 70 + 26, 236, names[i], { anchor: "middle", size: 12 }); });
        G.label(420, 40, "z = (logit + shift) / T", { size: 11 });
        z.forEach((x, i) => G.label(420, 62 + i * 20, `${names[i]}: ${x.toFixed(1)}`, { size: 12, color: "ink" }));
        G.label(420, 176, `largest exp(z) = ${zmax > 88.7 ? "exp(" + zmax.toFixed(0) + ")" : Math.exp(zmax).toExponential(2)}`, { size: 11 });
        G.label(420, 196, `float32 max = exp(88.72)`, { size: 11 });
        G.label(420, 216, `stable: z − ${zmax.toFixed(1)} ≤ 0`, { size: 11, color: "ok" });
        const naiveTxt = overflow ? "exp overflows → inf / inf = nan" : underflow ? "all exp round to 0 → 0 / 0 = nan" : "fine here";
        return [
          { title: "Probabilities", rows: names.map((n, i) => [n, (100 * p[i]).toFixed(2) + "%"]) },
          { title: "How spread out?", rows: [["entropy", H.toFixed(2) + " bits"], ["most uncertain possible", Hmax.toFixed(2) + " bits"]],
            gauge: [[H / Hmax, "k"], [1 - H / Hmax, "line"]], gaugeText: H < 0.3 ? "nearly certain (greedy-like)" : H > 2 ? "almost uniform" : "a clear favourite, some spread" },
          { title: "Naive vs stable (float32)", rows: [["naive exp(z) / sum", naiveTxt], ["stable (subtract max)", "always fine"]], chip: [!(overflow || underflow), overflow || underflow ? "naive softmax breaks here" : "both agree"],
            html: `<p class="note">Entropy is −Σ p·log2 p. Overflow check: a float32 exp overflows above about 88.72, or when the sum passes 3.4e38. Underflow (all terms round to 0) happens below about −104. The model here is arithmetic only; the exercise uses numpy.</p>` },
        ];
      },
    },
    practice: {
      intro: `Run these from the repository root. Each exercise has a starter file and a test; <code>S2S_SOLUTIONS=1</code> runs the same test against the reference solution. Run all four with <code>uv run pytest course/R-readiness/R.5-maths-and-ml-basics/exercises</code>.`,
      items: [
        { title: "Dot product and matrix-vector product", tier: "T0 · easy", goal: "Write dot, matvec and matvec_flops with plain loops, and check them against numpy's @.",
          cmd: "uv run pytest course/R-readiness/R.5-maths-and-ml-basics/exercises/01-dot-and-matvec" },
        { title: "Stable softmax", tier: "T0 · easy", goal: "Softmax over any axis that survives logits like [1000, 1001] without overflow.",
          cmd: "uv run pytest course/R-readiness/R.5-maths-and-ml-basics/exercises/02-softmax" },
        { title: "A 2-layer MLP", tier: "T0 · medium", goal: "SiLU, the forward pass y = W2 silu(W1 x + b1) + b2, and param_count checked by hand arithmetic.",
          cmd: "uv run pytest course/R-readiness/R.5-maths-and-ml-basics/exercises/03-mlp-forward" },
        { title: "From logits to a next token", tier: "T0 · medium", goal: "Pick the greedy token and the top-k tokens with probabilities, with a temperature.",
          cmd: "uv run pytest course/R-readiness/R.5-maths-and-ml-basics/exercises/04-next-token" },
      ],
      labs: [
        { label: "Module README: the ten ideas in one page", path: "course/R-readiness/R.5-maths-and-ml-basics/README.md" },
        { label: "Self-check questions", path: "course/R-readiness/R.5-maths-and-ml-basics/quiz.md" },
        { label: "A real reference implementation you can read after P1.1", path: "platform/engine/v0/reference/llama_numpy.py" },
      ],
    },
  });
})();
