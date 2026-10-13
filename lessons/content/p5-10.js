/* P5.10 — A custom op inside a serving engine (#12). Register, fake, opcheck, dispatch, plug into vLLM, measure end to end. */
(function () {
  const F = S2S.fmt;
  // a small vector as a row of boxes
  const vec = (G, x, y, vals, color, o = {}) => vals.map((v, i) => G.box(x + i * (o.w ?? 56), y, (o.w ?? 56) - 4, o.h ?? 28, String(v), { fill: color, size: o.size ?? 12, rx: 4 }));
  // a graph node
  const node = (G, x, y, w, t, color, sub) => {
    const g = G.box(x, y, w, sub ? 44 : 32, "", { stroke: color || "ink", rx: 8 });
    G.text(x + w / 2, y + (sub ? 19 : 21), t, { anchor: "middle", size: 12, color: color || "ink", weight: 700, parent: g });
    if (sub) G.label(x + w / 2, y + 36, sub, { anchor: "middle", size: 10, parent: g });
    return g;
  };

  S2S.lesson({
    id: "p5-10", n: "P5.10", title: "A custom op inside a serving engine",
    subtitle: "CUDA deep dive · first principles · T0 for registration and tests, T2 for the end-to-end bench",
    kicker: "Lesson · ≈ 40 min",
    headline: "A fast kernel is not a faster server until the engine calls it",
    intro: `<p>You can now write a kernel that beats PyTorch eager in a micro-benchmark. This lesson is about everything that stands between that kernel and a user's latency: telling PyTorch's compiler that the kernel exists and what it changes, making the engine call it, and then measuring honestly. The vehicle is project <b>#12</b>: a fused residual-add + RMSNorm (from P5.5) running inside vLLM behind a flag. You will also learn why the expected end-to-end gain is small, and why that is the lesson.</p>`,
    facts: ["9 steps", "4 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "our code"], ["w", "framework / engine"], ["ok", "checked"], ["hot", "breaks silently"], ["v", "measured"]],
    prev: "p5-9", next: "p6-1",
    steps: [
      { rail: "the gap", title: "A kernel in a benchmark is not a kernel in a server",
        body: `<p>A micro-benchmark calls your kernel in a loop on a fresh tensor. A serving engine does something very different: it runs the model under <b>torch.compile</b> (a compiler that traces the model into a graph, then fuses and reorders it) and inside <b>CUDA graphs</b> (recorded GPU work that is replayed with one launch, P6.6). Both need to know, for every operation: what goes in, what comes out, and what is changed in place.</p>
<p>A kernel that does not tell them this either breaks the graph or, worse, gets reordered or dropped and returns wrong answers without any error.</p>
<p>So the work has four layers, and a gap at each:</p>
<ol><li><b>Register</b> the kernel as a PyTorch operator with a declared schema.</li><li><b>Test</b> the registration itself, not just the arithmetic.</li><li><b>Dispatch</b>: make vLLM call it instead of its own kernel.</li><li><b>Measure</b> end to end, where a kernel can be 2× faster and the server 1.5% faster.</li></ol>`,
        scene(G) {
          const L = [["user latency (p90 TTFT, ITL)", "ok", "what #4 measures"], ["engine step", "w", "scheduler + runner (P6)"], ["model forward under torch.compile + CUDA graph", "w", "needs schema + fake"], ["vllm._custom_ops.fused_add_rms_norm", "w", "the call site we redirect"], ["torch.ops.s2s.fused_add_rms_norm", "k", "registered op"], ["CUDA / Triton kernel", "k", "your P5.5 work"]];
          const nodes = L.map(([t, c, s], i) => {
            const y = 36 + i * 62;
            const g = G.box(40, y, 400, 46, "", { stroke: c, rx: 8 });
            G.text(54, y + 20, t, { size: 13, color: c, weight: 700, parent: g });
            G.label(54, y + 37, s, { size: 11, parent: g });
            if (i) G.arrow(240, y - 2, 240, y - 14, { color: "muted" });
            return g;
          });
          ["4 · measure end to end", "", "1-2 · register + test", "3 · dispatch (plugin)", "1 · register", ""].forEach((t, i) => G.label(452, 64 + i * 62, t, { size: 12, color: "ink" }));
          G.from(nodes, { opacity: 0, y: 12, stagger: 0.12, duration: 0.3 });
          G.caption("the kernel is the bottom box; the user only sees the top one");
        } },

      { rail: "the op", title: "The op we ship: residual add and RMSNorm in one pass",
        body: `<p>Every transformer layer adds its output back onto a running <b>residual</b> stream, then normalises. RMSNorm divides each row by its root-mean-square and scales by a learned weight. vLLM fuses "add, then norm" into one kernel with this contract, and so does ours (<code>s2s::fused_add_rms_norm</code>):</p>
<div class="eq">residual ← x + residual      (rounded to its dtype)
x ← rmsnorm(residual) · weight   (fp32 math)
rmsnorm(s) = s / sqrt(mean(s²) + eps)

x = [1, -1, 2, 0]    residual = [1, 3, 0, 0]
w = [1, .5, 2, 1]
s = x + residual = [2, 2, 2, 0]
mean(s²) = (4+4+4+0)/4 = 3,  1/sqrt(3) = 0.5774
x_new = s · 0.5774 · w
      = [1.155, 0.577, 2.309, 0]</div>
<p>Both tensors are changed <b>in place</b>: the op returns nothing. That detail drives the whole registration. Why fuse? Eager PyTorch (see <code>eager()</code> in <code>bench_op.py</code>) runs about ten small kernels for this, each reading and writing the whole tensor. The fused kernel reads <code>x</code> and <code>residual</code> once and writes both once: 4 passes over T × H values (the byte count <code>bench_op.py</code> divides by).</p>`,
        scene(G) {
          G.label(24, 30, "worked example (hidden size 4 for the picture; Llama-3-8B uses 4096)", { size: 12 });
          const labs = [["x", 56], ["residual", 92], ["s = x + residual", 128], ["s · 1/√3 · w", 200]];
          G.text(24, 62, "x", { size: 12 }); vec(G, 150, 46, [1, -1, 2, 0], "k");
          G.text(24, 100, "residual", { size: 12 }); vec(G, 150, 84, [1, 3, 0, 0], "v");
          G.arrow(262, 114, 262, 138, { color: "ink" });
          G.text(24, 160, "s (written back to residual)", { size: 12 }); const s = vec(G, 270, 144, [2, 2, 2, 0], "v");
          G.text(24, 196, "mean(s²) = 3 → ×0.5774", { size: 12 });
          G.arrow(380, 176, 380, 200, { color: "ink" });
          G.text(24, 226, "x_new (written back to x)", { size: 12 }); const o = vec(G, 270, 210, ["1.155", "0.577", "2.309", 0], "k", { w: 62 });
          G.text(24, 290, "memory traffic over a T × H tensor", { size: 13, weight: 700 });
          G.label(24, 312, "eager: about ten kernels, each reads and writes the tensor", { size: 11, color: "hot" });
          G.rect(24, 318, 560, 14, { stroke: "hot", rx: 3 });
          G.label(24, 354, "fused: 1 kernel, 4 passes (read x, residual · write residual, x)", { size: 11, color: "ok" });
          G.rect(24, 360, 120, 14, { fill: "ok", rx: 3 });
          G.label(24, 396, "T = tokens in the step, H = hidden size · bar lengths are schematic", { size: 11 });
          G.from(o, { opacity: 0, stagger: 0.1, duration: 0.3, delay: 0.3 });
          G.caption("two tensors updated in place, one pass over the data");
        } },

      { rail: "register", title: "custom_op: give the compiler a schema",
        body: `<p>Why not call the kernel from a plain Python function? Because <code>torch.compile</code> would then trace <i>into</i> the function and see its internals (or fail on the extension call, a graph break). A <b>registered operator</b> is opaque: the compiler sees one node with a typed signature and promises to treat it as a black box.</p>
<p>In <code>platform/kernels/torch_ext/ops.py</code>:</p>
<div class="eq">@torch.library.custom_op(
    "s2s::fused_add_rms_norm",
    mutates_args=("x", "residual"))
def fused_add_rms_norm(
        x, residual, weight, eps) -> None:
    ...   # PyTorch reference, any device</div>
<p>The schema says: four inputs, no outputs, and <code>x</code> and <code>residual</code> are <b>mutated</b>. The compiler relies on that list. It may reorder ops, delete ops whose result is unused, or convert in-place ops into functional ones. An op that writes to a tensor without admitting it looks, to the compiler, like an op that does nothing: dead code.</p>`,
        check: { q: "You forget to list residual in mutates_args. The op still runs correctly in eager mode. What goes wrong under torch.compile?",
          options: ["Nothing: the compiler inspects the kernel and finds the write", "The compiler may treat the residual as unchanged, reorder or drop the op, and later layers read stale values with no error", "Python raises a TypeError when the op is registered"], answer: 1,
          why: "The schema is all the compiler knows about an opaque op. It cannot look inside a CUDA kernel. An undeclared write is invisible to functionalization, reordering and dead-code elimination, so the result can be silently wrong. The reference implementation in eager mode still looks fine, which is why exercise 1 makes you break it on purpose." },
        scene(G) {
          G.text(24, 34, "plain function: the compiler traces in", { size: 13, weight: 700, color: "hot" });
          const a = [["x + res", 24], ["copy_", 120], ["float", 200], ["pow", 276], ["mean", 340], ["rsqrt", 410], ["mul", 490], ["copy_", 556]];
          const an = a.map(([t, x], i) => { const g = G.box(x, 52, t.length * 8 + 22, 28, t, { stroke: "hot", size: 11, rx: 4 }); return g; });
          G.label(24, 100, "many nodes, each free to be reordered or fused; a C++ call may even break the graph", { size: 11 });
          G.text(24, 160, "registered op: one opaque node with a schema", { size: 13, weight: 700, color: "ok" });
          const n = node(G, 24, 180, 260, "s2s::fused_add_rms_norm", "ok", "(x!, residual!, weight, eps) -> ()");
          G.label(300, 196, "! = mutated: x, residual", { size: 12, color: "ok" });
          G.label(300, 214, "read-only: weight, eps", { size: 12 });
          G.text(24, 290, "what the compiler may do with the schema", { size: 13, weight: 700 });
          ["keep the op, in order, between its readers and writers", "functionalize: pass copies, copy results back", "never delete it as dead code", "never run it before its inputs are final"].forEach((t, i) => G.label(34, 314 + i * 22, "· " + t, { size: 12, color: "ink" }));
          G.from(an, { opacity: 0, x: -8, stagger: 0.05, duration: 0.2 }); G.from(n, { opacity: 0, delay: 0.5, duration: 0.4 });
          G.caption("the schema is the only thing the compiler knows about your kernel");
        } },

      { rail: "fake", title: "The fake implementation: shapes without running",
        body: `<p>To plan memory and fuse neighbours, the compiler traces the model with <b>fake tensors</b>: objects that carry a shape, dtype, device and strides but no data. Every op must answer "given inputs like these, what are the outputs like?" without computing anything. That answer is the <b>fake</b> (or meta) implementation:</p>
<div class="eq">@fused_add_rms_norm.register_fake
def _fake(x, residual, weight, eps):
    return None     # only mutates: no new tensor</div>
<p>For this op the fake is trivial. For an op that returns a tensor it would be <code>torch.empty_like(x)</code> or a new shape, and getting it wrong (wrong shape, wrong dtype, aliasing an input by accident) means the compiler plans the wrong memory.</p>
<p>The cost of skipping the work is large: 33 tokens × 4096 hidden is 135,168 numbers per tensor, and a real trace has 65 such calls per decode step (two norms in each of 32 layers, plus the final one). A fake call moves none of them.</p>`,
        check: { q: "What does the fake implementation of a custom op do, and when is it called?",
          options: ["It runs a slow CPU copy of the kernel to validate the GPU result at runtime", "It reports output shapes, dtypes and strides without computing, and torch.compile calls it while tracing", "It is called on every serving request to allocate the output"], answer: 1,
          why: "Tracing must not execute kernels, so the compiler substitutes fake tensors and calls the fake implementation to learn what comes out. The real kernel only runs later, when the compiled graph executes. opcheck compares the fake's metadata against the real output to catch disagreements." },
        scene(G) {
          G.text(24, 34, "real run", { size: 13, weight: 700 });
          const real = G.grid(24, 46, 6, 24, 17, 12, () => "k", { gap: 2, rx: 2 });
          G.label(24, 134, "data moves: 33 × 4096 values per tensor, GPU does work", { size: 11 });
          G.text(24, 176, "fake run (tracing)", { size: 13, weight: 700 });
          const f = G.box(24, 190, 380, 56, "", { stroke: "q", dash: "4 3", rx: 8 });
          G.text(36, 212, "FakeTensor  shape (33, 4096)", { size: 12, color: "q", parent: f });
          G.text(36, 232, "bf16 · cuda:0 · stride (4096, 1) · no data", { size: 12, color: "q", parent: f });
          G.label(24, 268, "metadata only: nothing allocated, no kernel launched", { size: 11 });
          G.arrow(408, 218, 436, 218, { color: "q" });
          const fk = node(G, 440, 198, 176, "fake(x, res, w, eps)", "q", "returns None");
          G.text(24, 322, "the compiler learns:", { size: 13, weight: 700 });
          ["op has no outputs → graph needs no new buffer", "x, residual keep their shapes → plan memory around them", "all 65 norm calls in a decode step traced in milliseconds"].forEach((t, i) => G.label(34, 346 + i * 22, "· " + t, { size: 12, color: "ink" }));
          G.from(real, { opacity: 0, stagger: 0.002, duration: 0.1 }); G.from(f, { opacity: 0, delay: 0.5, duration: 0.4 });
          G.caption("tracing asks what comes out, never runs the kernel");
        } },

      { rail: "opcheck", title: "opcheck: test the registration, not just the maths",
        body: `<p>A kernel can be numerically perfect and still be registered wrongly. <code>torch.library.opcheck</code> checks the registration itself. It runs the op on sample inputs and tests, among other things (names from the PyTorch docs; re-check them for your version):</p>
<ul><li><b>schema</b>: does the declared mutation list match what the op really changes?</li><li><b>fake tensor</b>: does the fake implementation return the same metadata as the real one?</li><li><b>aot dispatch</b>: does the op survive being traced and functionalised as the compiler will do?</li><li><b>autograd registration</b>: is the op correct about gradients (here: none needed)?</li></ul>
<div class="eq">uv run --extra torch pytest \\
  platform/kernels/tests_py -m torch \\
  -k "semantics or opcheck"     # T0: laptop CPU</div>
<p>On a laptop this uses the PyTorch reference as the implementation; the same check runs on the GPU variants (<code>test_cuda_impls</code>). Exercise 1 asks you to <b>break the op on purpose</b>, three ways, and read each error: drop <code>"x"</code> from <code>mutates_args</code>; make the fake return <code>torch.empty_like(x)</code>; make the reference write into a new tensor instead of in place. Reading what opcheck says is how you learn to recognise these mistakes in real pull requests.</p>`,
        scene(G) {
          const tests = [["schema", "declared mutation = actual mutation?", "ok"], ["fake tensor", "fake output metadata = real output?", "q"], ["aot dispatch", "survives tracing + functionalisation?", "k"], ["autograd", "gradient registration consistent?", "w"]];
          const cards = tests.map(([t, d, c], i) => {
            const y = 40 + i * 54;
            const g = G.box(24, y, 340, 42, "", { stroke: c, rx: 8 });
            G.text(38, y + 18, t, { size: 13, color: c, weight: 700, parent: g }); G.label(38, y + 34, d, { size: 11, parent: g });
            return g;
          });
          G.text(390, 56, "break it on purpose", { size: 13, weight: 700, color: "hot" });
          [["remove \"x\" from mutates_args", "schema"], ["fake returns empty_like(x)", "fake"], ["write to a new tensor", "schema"]].forEach(([b, w], i) => {
            const y = 72 + i * 56;
            G.box(386, y, 230, 34, b, { stroke: "hot", color: "hot", size: 11, rx: 6 });
            G.arrow(503, y + 36, 503, y + 50, { color: "hot" }); G.label(514, y + 48, "opcheck: " + w, { size: 10, color: "hot", anchor: "start" });
          });
          G.rect(24, 270, 592, 120, { stroke: "line", rx: 8 });
          G.text(40, 296, "tests_py/test_custom_op.py", { size: 13, weight: 700 });
          ["test_semantics_cpu : matches the reference maths", "test_opcheck_cpu : the four checks above (T0)", "test_torch_compile_fullgraph_cpu : step 9's idea, T0", "test_cuda_impls : cuda and triton, fp32 / bf16 / fp16 (GPU)"].forEach((t, i) => G.label(40, 320 + i * 20, t, { size: 12, color: "ink" }));
          G.from(cards, { opacity: 0, x: -16, stagger: 0.12, duration: 0.3 });
          G.caption("opcheck tests the contract between your op and the compiler");
        } },

      { rail: "dispatch", title: "One op name, a kernel per device",
        body: `<p>A registered op is a name plus a schema. The <i>code</i> that runs is chosen by <b>dispatch</b> on the tensors' device:</p>
<ul><li>the function you decorated is the default, which runs anywhere: the PyTorch reference. That is what lets the T0 tests run on a CPU;</li><li><code>@fused_add_rms_norm.register_kernel("cuda")</code> overrides it for CUDA tensors. Ours reads <code>S2S_RMSNORM_IMPL</code>: <code>cuda</code> (default) calls the JIT-built extension <code>csrc/s2s_ext.cu</code>, <code>triton</code> calls <code>triton_kernels/rmsnorm.py</code>.</li></ul>
<p>The CUDA kernel is the P5.5 design: one block per row. For Llama-3-8B (H = 4096): 512 threads, so each thread handles 4096 / 512 = 8 values. It adds in fp32, stores the rounded sum back to <code>residual</code>, keeps the row in a shared-memory buffer (4096 × 4 B = 16 KiB), reduces the sum of squares with warp shuffles, and writes <code>x</code>. The buffer is why the kernel checks <code>cols ≤ 12288</code>: 12288 × 4 B = 48 KiB, the default shared-memory limit.</p>
<p>Why can the <b>residual</b> match the reference exactly while <code>x</code> only matches within a tolerance? The residual is one fp32 add then one rounding, the same in every implementation. <code>x</code> needs a sum of squares, and floating-point addition is not associative: a different summation order changes the last bits.</p>`,
        scene(G) {
          const call = node(G, 24, 40, 260, "torch.ops.s2s.fused_add_rms_norm", "k", "called with CUDA tensors");
          const cpu = node(G, 24, 150, 170, "CPU / default", "w", "PyTorch reference");
          const cu = node(G, 230, 150, 170, "cuda · impl=cuda", "ok", "csrc/s2s_ext.cu (JIT)");
          const tr = node(G, 440, 150, 176, "cuda · impl=triton", "q", "triton_kernels/rmsnorm.py");
          G.arrow(100, 86, 100, 146, { color: "muted", dash: "3 3" }); G.label(106, 124, "device = cpu", { size: 11 });
          G.arrow(190, 86, 300, 146, { color: "ok" }); G.arrow(210, 86, 510, 146, { color: "q" });
          G.label(300, 58, "device = cuda: S2S_RMSNORM_IMPL picks", { size: 11, color: "ink" });
          G.text(24, 244, "CUDA kernel, hidden 4096, one block per row", { size: 13, weight: 700 });
          const th = G.grid(24, 258, 1, 32, 17, 22, () => "k", { gap: 2, rx: 2 });
          G.label(24, 296, "512 threads × 8 values = 4096 (32 of 512 shown as one box per 16)", { size: 11 });
          G.box(24, 312, 200, 32, "shared row buffer 16 KiB", { stroke: "v", color: "v", size: 12 });
          G.box(236, 312, 190, 32, "warp-shuffle Σ s²", { stroke: "ok", color: "ok", size: 12 });
          G.box(438, 312, 178, 32, "write x · w · rsqrt", { stroke: "k", color: "k", size: 12 });
          G.label(24, 380, "residual: bit-exact in all impls · x: tolerance (sum order differs)", { size: 12, color: "ink" });
          G.from([cpu, cu, tr], { opacity: 0, y: 10, stagger: 0.15, duration: 0.3 }); G.from(th, { opacity: 0, stagger: 0.01, duration: 0.1 });
          G.caption("the op is a name; the device and a flag choose the code");
        } },

      { rail: "into vLLM", title: "Making vLLM call it: a plugin, in every process",
        body: `<p>Registering the op does not make vLLM use it. In the pinned vLLM (<code>v0.31.0</code>; re-verify the paths after any upgrade), the <code>RMSNorm</code> layer in <code>model_executor/layers/layernorm.py</code> calls <code>ops.fused_add_rms_norm(x, residual, weight, eps)</code>, where <code>ops</code> is <code>vllm._custom_ops</code>, which forwards to the compiled <code>torch.ops._C.fused_add_rms_norm</code> (<code>csrc/layernorm_kernels.cu</code>).</p>
<p>The layer looks the function up on the module <b>at call time</b>. So replacing the attribute redirects every call, with no fork of vLLM:</p>
<div class="eq">def register():   # a vllm.general_plugins entry
    if os.environ.get("S2S_RMSNORM") != "1":
        return
    import torch_ext.ops     # registers s2s ops
    from vllm import _custom_ops as vops
    vops.fused_add_rms_norm = ours   # calls our op</div>
<p>Where it runs matters. vLLM does not run the model in the process that serves HTTP: the API server, the engine core and the GPU workers are separate processes. A patch applied in your launch script reaches only that one process. vLLM loads <code>vllm.general_plugins</code> entry points in each of its processes, so the plugin patches the one that actually runs the model. The flag <code>S2S_RMSNORM</code> defaults to off, so installing the plugin changes nothing until you opt in.</p>`,
        check: { q: "You set vops.fused_add_rms_norm = ours in your launch script, then start vLLM. The log never shows your kernel. Most likely cause?",
          options: ["The kernel has a bug and silently falls back", "The model runs in a different process (engine core / worker) that never saw the patch; use a plugin entry point", "bf16 is not supported by custom ops"], answer: 1,
          why: "Module attributes are per-process. A patch in the launcher changes the launcher. The plugin entry point is loaded by vLLM in every process it starts, including the ones that execute the model. This is one of the module's listed common mistakes." },
        scene(G) {
          const procs = [["launch script", 24, "w", "patch here: no effect"], ["API server", 24, "w", "HTTP, tokenise"], ["engine core", 236, "w", "scheduler"], ["GPU worker", 430, "w", "model + kernels"]];
          G.box(24, 40, 592, 28, "S2S_RMSNORM=1 vllm serve …", { stroke: "line", size: 12, rx: 6 });
          const bx = [24, 24, 236, 430];
          const pr = [
            node(G, 24, 100, 170, "API server", "w", "HTTP · tokenise"),
            node(G, 236, 100, 170, "engine core", "w", "scheduler · KV manager"),
            node(G, 430, 100, 186, "GPU worker", "w", "model forward · kernels")];
          G.arrow(110, 70, 110, 96, { color: "muted" }); G.arrow(200, 120, 232, 120, { color: "muted" }); G.arrow(410, 120, 426, 120, { color: "muted" });
          const pl = [24, 236, 430].map((x, i) => G.box(x, 160, i === 2 ? 186 : 170, 28, "plugin: register()", { fill: "ok", size: 11, rx: 5 }));
          G.label(24, 212, "vllm.general_plugins entry point: loaded in each process", { size: 12, color: "ok" });
          G.text(24, 264, "the call", { size: 13, weight: 700 });
          G.box(24, 276, 590, 34, "", { stroke: "ink", rx: 6 });
          G.text(36, 298, "RMSNorm.forward → ops.fused_add_rms_norm(x, res, w, eps)", { size: 12 });
          G.arrow(320, 312, 320, 336, { color: "k" });
          G.box(150, 340, 340, 34, "vops.fused_add_rms_norm is now ours()", { stroke: "k", color: "k", size: 12 });
          G.label(24, 396, "attribute looked up at call time → the swap takes effect", { size: 11 });
          G.label(24, 412, "the plugin logs: s2s plugin: fused_add_rms_norm → …", { size: 11 });
          G.from(pr, { opacity: 0, y: 10, stagger: 0.15, duration: 0.3 }); G.from(pl, { opacity: 0, delay: 0.6, stagger: 0.1, duration: 0.3 });
          G.caption("the patch must exist where the model runs");
        } },

      { rail: "end to end", title: "A 2× faster kernel on a 3% path is a 1.5% win",
        body: `<p>Now measure, and predict first (the README asks you to write the prediction down). RMSNorm is memory-bound and light: in an nsys trace of one decode step (P5.3) it is a few percent of GPU time next to the big GEMMs and attention. Call that share <i>f</i>. If your kernel is <i>s</i> times faster, the step time becomes:</p>
<div class="eq">new   = (1 - f) + f / s      (old step = 1)
saved = f · (1 - 1/s)

f = 3%, s = 2:
  new = 0.97 + 0.015 = 0.985 → 1.5% faster
f = 3%, s = ∞:   saved = 3%   (the ceiling)
f = 3%, s = 1.1: saved = 0.27%</div>
<p>This is Amdahl's law. The 3% is the module's illustrative figure; <b>your trace</b> supplies the real one. The ceiling is the lesson: no kernel can save more than the time the old one used.</p>
<p>The second force is noise. Repeated runs of the same load test vary from run to run (scheduling, clocks, arrival jitter). A 1.5% effect can sit inside that spread, so you run each configuration three times with the same seed, as in the AWS guide, and report the spread. The Simulate tab lets you play with exactly this.</p>`,
        check: { q: "Your kernel is 2x faster than vLLM's and RMSNorm is 3% of a decode step. What end-to-end ITL change do you expect?",
          options: ["About 50% lower", "About 1.5% lower, possibly inside run-to-run noise", "About 3% lower, always visible", "No change at all, ever"], answer: 1,
          why: "Saved time = 3% × (1 − 1/2) = 1.5% of the step. That is the best case if nothing else changes, and it is small enough that #4's run-to-run variation can hide it. Report several runs and the spread instead of one number." },
        scene(G) {
          const f = 0.03, s = 2, H = 230, x0 = 80;
          G.label(24, 30, "one decode step (old = 100%), illustrative shares", { size: 12 });
          const bars = [["old kernel", 1, 0.97, f], ["2× faster", 0.985, 0.97, f / s]];
          bars.forEach(([name, tot, rest, nrm], i) => {
            const x = x0 + i * 150, base = 330;
            G.rect(x, base - H * rest, 90, H * rest, { fill: "w", rx: 3 });
            G.rect(x, base - H * rest - H * nrm, 90, Math.max(2, H * nrm), { fill: "k", rx: 2 });
            G.text(x + 45, base + 20, name, { anchor: "middle", size: 12 });
            G.label(x + 45, base - H * tot - 6, i ? "98.5%" : "100%", { anchor: "middle", size: 11, color: "ink" });
          });
          G.line(x0 - 6, 330 - H * 0.985, x0 + 250, 330 - H * 0.985, { color: "v", dash: "3 3" });
          G.label(x0 + 256, 330 - H * 0.985 + 4, "1.5% saved", { size: 12, color: "v" });
          G.label(x0 + 256, 330 - H * 0.985 + 20, "cyan: RMSNorm, 3% → 1.5%", { size: 11, color: "k" });
          G.text(24, 380, "noise between runs of #4 can be as big as this gap", { size: 13, color: "hot" });
          G.label(24, 402, "grey: GEMMs, attention, everything else (97%)", { size: 11 });
          G.from(G.svg.querySelectorAll("rect"), { opacity: 0, duration: 0.4, stagger: 0.05 });
          G.caption("the ceiling is the old kernel's share of the step");
        } },

      { rail: "prove it", title: "Prove it ran, then report the spread",
        body: `<p>Before you believe a number, show the patched path actually executed. Exercise 3 asks for three kinds of evidence:</p>
<ol><li><b>The swap happened</b>: the vLLM log contains the plugin's line (<code>grep "s2s plugin" results/p5.10-vllm-1.log</code>), and a trace (nsys, with #13's tracing on) shows your kernel's name where vLLM's <code>fused_add_rms_norm_kernel</code> used to be.</li><li><b>The answers did not change</b>: greedy sampling (<code>temperature=0</code>) on the same 20 prompts gives the same tokens, or differs only where bf16 rounding ties go the other way. Explain any difference.</li><li><b>The speed</b>: a micro table (tokens 1, 32, 256, 4096 × hidden 4096) for eager, vLLM's kernel and yours, and an end-to-end table: three #4 runs each with <code>S2S_RMSNORM=0</code> and <code>=1</code>, same seed, with knee, p90 TTFT and p50 ITL <b>and the spread</b>.</li></ol>
<div class="eq">bytes per call = 4 · T · H · 2 B   (bf16)
T = 1:    4 · 4096 · 2         = 32 KiB
T = 4096: 4 · 4096 · 4096 · 2  = 128 MiB</div>
<p>The shape of the micro table follows from those sizes: at one token the kernel moves 32 KiB, so launch overhead decides the time; at 4,096 tokens it moves 128 MiB, so bandwidth does. A good kernel's GB/s should approach vLLM's at large T. Last, exercise 4 has you write the pull-request description you would send upstream: the problem with numbers, design, correctness, performance <i>including the cases where you are slower</i>, risks (hidden size above 12,288, non-contiguous inputs, ROCm) and alternatives. Read two merged kernel PRs first.</p>`,
        scene(G) {
          const ev = [["1  log line + kernel name in nsys", "swap happened", "ok"], ["2  greedy tokens identical on 20 prompts", "numerics preserved", "k"], ["3  micro table at T = 1, 32, 256, 4096", "kernel is competitive", "w"], ["4  three #4 runs each, flag 0 vs 1", "end-to-end effect + spread", "v"]];
          const cards = ev.map(([a, b, c], i) => {
            const y = 36 + i * 52;
            const g = G.box(24, y, 380, 42, "", { stroke: c, rx: 8 });
            G.text(36, y + 18, a, { size: 12, color: c, weight: 700, parent: g }); G.label(36, y + 34, "shows: " + b, { size: 11, parent: g });
            return g;
          });
          // illustrative spread plot
          G.text(24, 270, "what to plot for row 4 (illustrative, not measured)", { size: 12, color: "muted" });
          G.axes(40, 284, 360, 110);
          const dots = [];
          [[0, [0.46, 0.5, 0.54]], [1, [0.44, 0.48, 0.53]]].forEach(([c, vals]) => {
            vals.forEach((v, j) => dots.push(G.circle(110 + c * 160, 394 - v * 180, 5, { fill: c ? "k" : "w" })));
            G.text(110 + c * 160, 408, c ? "flag = 1" : "flag = 0", { anchor: "middle", size: 11 });
          });
          G.label(430, 330, "the three dots of each", { size: 11 }); G.label(430, 346, "flag overlap: the", { size: 11 }); G.label(430, 362, "honest answer is", { size: 11 }); G.label(430, 378, "\"within noise\"", { size: 11, color: "ink" });
          G.from(cards, { opacity: 0, x: -14, stagger: 0.12, duration: 0.3 }); G.from(dots, { opacity: 0, delay: 0.5, stagger: 0.06, duration: 0.2 });
          G.caption("report the spread, not a single run");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A kernel only helps a server once it is registered (schema, fake), tested (opcheck), dispatched (plugin in every process) and measured end to end.</li>
<li><code>mutates_args</code> tells the compiler which tensors change; leaving one out can make it drop or reorder the op with no error.</li>
<li>The fake implementation answers shape and dtype questions during tracing, without running anything.</li>
<li>One op name, many implementations: the device and <code>S2S_RMSNORM_IMPL</code> choose. The residual matches exactly; the normalised output within tolerance.</li>
<li>vLLM runs the model in other processes than the launcher, so the swap goes in a <code>vllm.general_plugins</code> entry point.</li>
<li>Amdahl's law: saved = f · (1 − 1/s). A 2× faster kernel on a 3% path saves 1.5%, and run-to-run noise can be as large.</li></ul>`,
    sim: {
      title: "How much does a faster kernel buy end to end?",
      intro: "Set the share of a decode step that your kernel's old version used (f), how many times faster the new kernel is (s), and how noisy the load test is. The simulator applies Amdahl's law and compares the saving with the noise on the mean of n runs. The share f is an input: take it from your own nsys trace (P5.3). The bandwidth is an example value; check the GPU's spec sheet.",
      height: 290,
      controls: [
        { id: "f", label: "share of the step in this kernel f (%)", min: 0.5, max: 30, step: 0.5, value: 3 },
        { id: "s", label: "speedup of the kernel s (×)", min: 1, max: 10, step: 0.1, value: 2 },
        { id: "sd", label: "run-to-run noise, std dev of a run (% of ITL)", min: 0, max: 5, step: 0.1, value: 1.5 },
        { id: "n", label: "runs per configuration", min: 1, max: 10, step: 1, value: 3 },
        { id: "T", label: "tokens in the call (T)", type: "select", value: 32, options: [[1, "1"], [32, "32"], [256, "256"], [4096, "4096"]] },
        { id: "bw", label: "memory bandwidth, GB/s (example value, check the spec sheet)", min: 100, max: 3000, step: 50, value: 300 },
      ],
      draw(G, v) {
        const f = v.f / 100, saved = f * (1 - 1 / v.s) * 100, ceil = f * 100;
        const se = v.sd / Math.sqrt(v.n), diffSe = se * Math.SQRT2;
        const z = diffSe > 0 ? saved / diffSe : Infinity;
        // left: bars of the step
        G.label(10, 16, "decode step, old = 100%", { size: 11 });
        const H = 190, base = 240;
        [["old", 1], ["new", 1 - f * (1 - 1 / v.s)]].forEach(([n, tot], i) => {
          const x = 40 + i * 110, rest = H * (1 - f), nrm = H * (n === "old" ? f : f / v.s);
          G.rect(x, base - rest, 70, rest, { fill: "w", rx: 3 });
          G.rect(x, base - rest - nrm, 70, Math.max(1.5, nrm), { fill: "k", rx: 2 });
          G.text(x + 35, base + 18, n, { anchor: "middle", size: 12 });
          G.text(x + 35, base - H * tot - 6, (tot * 100).toFixed(1) + "%", { anchor: "middle", size: 11, color: "ink" });
        });
        // right: saving vs speedup curve
        const cx = 300, cw = 320, ch = 170, cy = 70;
        G.axes(cx, cy, cw, ch);
        G.label(cx, cy - 10, "saving (% of step) vs kernel speedup s", { size: 11 });
        const X = (s) => cx + (cw * (s - 1)) / 9, Y = (p) => cy + ch - (ch * p) / Math.max(ceil * 1.15, 0.5);
        let d = "";
        for (let s = 1; s <= 10.01; s += 0.25) d += `${d ? " L" : "M"} ${X(s)} ${Y(f * (1 - 1 / s) * 100)}`;
        G.path(d, { color: "k", w: 2.5 });
        G.line(cx, Y(ceil), cx + cw, Y(ceil), { color: "hot", dash: "4 3" }); G.label(cx + cw, Y(ceil) - 5, "ceiling f", { size: 10, anchor: "end", color: "hot" });
        G.line(cx, Y(se * 2), cx + cw, Y(se * 2), { color: "v", dash: "2 3" }); G.label(cx + 4, Y(se * 2) - 4, "≈ 2 × noise on the mean", { size: 10, color: "v" });
        G.circle(X(v.s), Y(saved), 5, { fill: "ink" });
        [1, 2, 4, 6, 8, 10].forEach((s) => G.label(X(s), cy + ch + 14, String(s), { size: 10, anchor: "middle" }));
        G.label(cx + cw, cy + ch + 30, "kernel speedup s", { size: 10, anchor: "end" });
        // kernel regime
        const bytes = 4 * v.T * 4096 * 2, us = (bytes / (v.bw * 1e9)) * 1e6;
        G.label(10, 280, `one call at T = ${v.T}, H = 4096, bf16 moves ${F.bytes(bytes)}`, { size: 11 });
        return [
          { title: "End-to-end saving", rows: [["step time saved", saved.toFixed(2) + " %"], ["ceiling (s = ∞)", ceil.toFixed(1) + " %"], ["new step time", (100 - saved).toFixed(2) + " % of old"]] },
          { title: "Is it visible?", rows: [["noise on mean of n runs", se.toFixed(2) + " %"], ["noise on the difference", diffSe.toFixed(2) + " %"], ["saving ÷ noise", isFinite(z) ? z.toFixed(1) + " ×" : "∞"]],
            chip: [z >= 2, z >= 2 ? "saving stands out of the noise" : "within noise: add runs or report the spread"] },
          { title: "One call: memory floor", rows: [["bytes moved (4·T·H·2)", F.bytes(bytes)], ["time at the example bandwidth", us.toFixed(2) + " µs"]],
            html: `<p class="note">A kernel launch costs on the order of microseconds of CPU/driver time (example scale), so for small T the call is launch-bound, not bandwidth-bound. CUDA graphs (P6.6) shrink that part.</p>` },
        ];
      },
      note: "Amdahl's law with Gaussian run noise (the difference of two means has std dev √2 · σ/√n). A model, not a measurement: f must come from your own trace, and real noise is rarely Gaussian.",
    },
    practice: {
      intro: "Exercises 1, 2 and 4 are T0 (a laptop CPU is enough for the tests; install torch with --extra torch). Exercise 3 is T2: one g6.xlarge GPU, about 3 instance-hours; the server stays on 127.0.0.1 behind an API key, reached through the shared P5 setup, and you tear it down afterwards (see aws.md and the shared P5 page for cost, auto-stop and teardown).",
      items: [
        { title: "Register the op and pass opcheck", tier: "T0 → T2 · medium", goal: "Explain each decorator in ops.py, make semantics and opcheck pass on CPU, then break the op three ways and read opcheck's errors. On the GPU: the residual must match exactly.",
          cmd: "uv run --extra torch pytest platform/kernels/tests_py -m torch -k \"semantics or opcheck\"" },
        { title: "Make it torch.compile-safe", tier: "T0 · medium", goal: "Compile a small block that calls the op with fullgraph=True, inspect the graph (the op is one node), then replace the op by a plain function and see what changes.",
          cmd: "TORCH_LOGS=graph_code uv run --extra torch pytest platform/kernels/tests_py -m torch -k compile -s" },
        { title: "#12 end to end: micro + #4 + #13", tier: "T2 · hard", goal: "Micro table, identical greedy outputs, three #4 runs per flag with the spread, an nsys trace per configuration, and a verdict in 5-8 sentences.",
          cmd: "PYTHONPATH=platform:platform/kernels uv run python platform/kernels/torch_ext/bench_op.py | tee results/p5.10-micro.md" },
        { title: "An upstream-quality PR description", tier: "T0 · hard", goal: "Write what you would attach to a vLLM PR: what and why, design, correctness, performance (including slower cases), risk, alternatives. Then compare with two merged kernel PRs.",
          cmd: "cat course/P5-cuda-deep-dive/P5.10-custom-op-into-serving/exercises/04-pr.md" },
      ],
      labs: [
        { label: "The registered op: reference, CUDA and Triton dispatch, fake", path: "platform/kernels/torch_ext/ops.py" },
        { label: "The CUDA extension (JIT-built)", path: "platform/kernels/torch_ext/csrc/s2s_ext.cu" },
        { label: "Micro-benchmark vs vLLM's kernel and eager", path: "platform/kernels/torch_ext/bench_op.py" },
        { label: "The vLLM plugin behind S2S_RMSNORM=1", path: "platform/kernels/vllm_plugin/s2s_vllm_plugin/__init__.py" },
        { label: "Opcheck, compile and GPU tests", path: "platform/kernels/tests_py/test_custom_op.py" },
        { label: "AWS guide: three instance-hours, g6.xlarge, SSM only, teardown", path: "course/P5-cuda-deep-dive/P5.10-custom-op-into-serving/aws.md" },
        { label: "Quiz", path: "course/P5-cuda-deep-dive/P5.10-custom-op-into-serving/quiz.md" },
      ],
    },
  });
})();
