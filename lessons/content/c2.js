/* C2 — The public benchmark teardown. Teaches reproducibility: provenance, missing data, what must accompany a
   number, run-to-run variance, fair comparison and cost per token, report structure. Code: platform/report. */
(function () {
  const F = S2S.fmt;
  // two-sided 95% Student t critical values by degrees of freedom (standard table)
  const TT = [[1, 12.706], [2, 4.303], [3, 3.182], [4, 2.776], [5, 2.571], [6, 2.447], [7, 2.365], [8, 2.306], [9, 2.262], [10, 2.228], [12, 2.179], [14, 2.145], [16, 2.12], [18, 2.101], [20, 2.086], [30, 2.042], [40, 2.021], [60, 2.0], [120, 1.98]];
  const tcrit = (df) => { let v = 1.96; if (df < 1) return 12.706; for (const [d, t] of TT) if (d <= df) v = t; return df > 120 ? 1.96 : v; };

  S2S.lesson({
    id: "c2", n: "C2", title: "The public benchmark teardown",
    subtitle: "Capstone · first principles · T0 builder, T2/T3 numbers",
    kicker: "Lesson · ≈ 45 min",
    headline: "A number nobody can reproduce is an opinion",
    intro: `<p>You have spent the course building a serving platform and measuring it. The capstone asks you to publish what you found as a <b>teardown</b>: a report where a stranger, given your repository, the stated hardware and a credit card, can reproduce every number.</p>
<p>The lesson covers what makes that possible: tracing each number back to raw data, refusing to invent missing ones, listing everything that must travel with a number, telling signal from run-to-run noise, comparing fairly, and structuring the write-up so the failures are in it too. The code is <code>platform/report</code>.</p>`,
    facts: ["8 steps", "4 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "generated / raw data"], ["v", "hand-written"], ["ok", "reproduces"], ["hot", "missing or wrong"], ["q", "noise / spread"]],
    prev: "c1", next: "l1",
    steps: [
      { rail: "stranger test", title: "The test: can a stranger get the same number?",
        body: `<p>A benchmark report is a claim: "this engine does X tokens a second on this hardware". A claim is only worth something if someone else can check it. So set the standard first. A stranger with <b>your repository, the stated hardware and a credit card</b> must be able to reproduce every number.</p>
<p>Three habits break that standard, and each is easy to fall into:</p>
<ul><li><b>Numbers typed by hand.</b> A copy-paste slip, or a "rounded" figure, and the report no longer matches any run.</li>
<li><b>Numbers without their configuration.</b> "1,200 tokens a second" means nothing without the model, the precision, the batch limits and the workload.</li>
<li><b>Leaving out what failed.</b> If only the wins are shown, readers cannot tell which results are typical.</li></ul>
<p>The rest of the lesson turns each habit into a rule that the tooling can enforce.</p>`,
        scene(G) {
          G.circle(54, 110, 20, { fill: "w" }); G.rect(34, 134, 40, 52, { fill: "w", rx: 12 }); G.label(54, 206, "a stranger", { anchor: "middle", size: 12 });
          const src = [["your repo @ commit", 60], ["stated hardware", 112], ["a credit card", 164]];
          src.forEach(([t, y]) => { G.box(130, y - 18, 180, 40, t, { stroke: "k", size: 12 }); G.arrow(74, 150, 128, y + 2, { color: "muted" }); G.arrow(310, y + 2, 372, 112, { color: "muted" }); });
          G.box(374, 76, 240, 70, "reproduce the table", { fill: "ok", size: 14 });
          G.text(494, 172, "same numbers?", { anchor: "middle", size: 13, color: "ok" });
          G.text(24, 252, "three habits that make it impossible", { size: 12, color: "muted" });
          const bad = ["numbers typed by hand", "numbers without config", "failures left out"];
          const o = bad.map((t, i) => G.box(24 + i * 200, 268, 188, 46, "✗ " + t, { stroke: "hot", dash: "5 4", color: "hot", size: 12 }));
          const fix = ["numbers link to JSON", "config travels along", "failures are reported"];
          const f = fix.map((t, i) => G.box(24 + i * 200, 344, 188, 46, "✓ " + t, { fill: "ok", size: 12 }));
          for (let i = 0; i < 3; i++) G.arrow(118 + i * 200, 314, 118 + i * 200, 342, { color: "muted" });
          G.from(o, { opacity: 0, y: -8, stagger: 0.1, duration: 0.3 }); G.from(f, { opacity: 0, y: 8, stagger: 0.1, delay: 0.4, duration: 0.3 });
          G.caption("each bad habit has a rule the builder can enforce");
        } },

      { rail: "provenance", title: "Every number links to the file that produced it",
        body: `<p>The first rule removes hand-typing. Benchmarks write raw results as JSON files under <code>results/</code>. The report is then <i>generated</i> from those files, never edited by hand where numbers appear.</p>
<p><code>platform/report/build.py</code> reads a <b>manifest</b> (<code>manifest.yaml</code>): a list of sections, each naming the result files to read (globs relative to the repo root), the hardware, and how to render them as a table (<code>loadgen</code>, <code>table</code> with columns, or <code>kv</code>). For each section it:</p>
<ol><li>turns every number into a Markdown link to its source JSON;</li>
<li>appends an <b>appendix</b> listing every input file with its <b>sha256</b>, a fingerprint of the file's bytes, so a reader can check that their reproduction produced the same raw data;</li>
<li>stamps the git <b>commit</b> in the header.</li></ol>
<div class="eq">PYTHONPATH=platform uv run python -m report.build \\
    --root . --out reports/teardown.md</div>
<p>Because <code>results/</code> is git-ignored, you commit exactly the files the report cites with <code>git add -f</code>. A link that points at nothing is a broken promise, so check that every cited file is in the repository.</p>`,
        scene(G) {
          G.rect(20, 50, 130, 60, { stroke: "k", rx: 8 }); G.text(85, 76, "benchmarks", { anchor: "middle", size: 13 }); G.text(85, 94, "write JSON", { anchor: "middle", size: 13 });
          G.arrow(150, 80, 188, 80, { color: "ink" });
          G.rect(190, 50, 130, 60, { fill: "k", rx: 8 }); G.text(255, 76, "build.py", { anchor: "middle", size: 13, color: "bg" }); G.text(255, 94, "+ manifest.yaml", { anchor: "middle", size: 12, color: "bg" });
          G.arrow(320, 80, 358, 80, { color: "ink" });
          G.rect(360, 50, 130, 60, { stroke: "ok", rx: 8 }); G.text(425, 76, "reports/", { anchor: "middle", size: 13 }); G.text(425, 94, "teardown.md", { anchor: "middle", size: 13 });
          G.label(20, 36, "results/…/*.json", { size: 11 });
          G.text(24, 160, "inside the generated table", { size: 12, color: "muted" });
          G.box(24, 172, 130, 38, "vllm", { stroke: "line", size: 12 });
          const num = G.box(164, 172, 260, 38, "[512](../results/p6.7/vllm.json)", { fill: "v", size: 10 });
          G.path("M 294 172 C 294 140, 85 150, 85 112", { color: "v", dash: "4 4", arrow: true });
          G.text(24, 252, "appendix: every input file with its sha256", { size: 12, color: "muted" });
          const files = ["results/p6.7/vllm.json    9f2c41ab…", "results/p6.7/s2s.json     0be77d10…", "results/cost/g6.json      c43a9e55…"];
          files.forEach((t, i) => G.text(24, 278 + i * 24, t, { size: 12 }));
          G.text(24, 372, "header: built from commit <sha>", { size: 12 });
          G.label(24, 392, "(file names and hashes above are placeholders)", { size: 10 });
          G.from(num, { opacity: 0, scale: 0.8, transformOrigin: "center", duration: 0.4 });
          G.caption("click a number, land on the raw data");
        } },

      { rail: "missing data", title: "Missing data must look missing",
        body: `<p>What should the builder do when a section has no result file yet? The tempting answer is to fill in an estimate so the table looks complete. Do not. Once a table is screenshotted and shared, an estimate is indistinguishable from a measurement.</p>
<p>So <code>build.py</code> renders a section without files as a visible marker, never a number:</p>
<div class="eq">TODO(run-on: g6.xlarge) - no results matching
results/bakeoff/*.json</div>
<p>This is a <i>design constraint</i>, not a courtesy: the builder is simply unable to invent a number because it only copies values out of JSON. The tests in <code>platform/report/tests/test_build.py</code> check it: with no results, every section must render TODO; with two result files present, exactly the four other sections still show TODO.</p>
<p>The same principle covers one missing field inside a row: it shows as an em dash, not as a guess.</p>`,
        check: { q: "Why does the builder render TODO instead of an estimate when a result is missing?",
          options: ["Estimates take longer to compute", "Once screenshotted, an estimate in a table cannot be told from a measurement", "Markdown cannot show estimated values"], answer: 1,
          why: "A number in a table carries the authority of a measurement. If the builder can only copy values out of raw JSON, a missing result stays visibly missing." },
        scene(G) {
          const secs = [["serving-knee", "[512](…/vllm.json)"], ["bakeoff", 0], ["quant", 0], ["cost", "[0.278](…/g6.json)"], ["kernels", 0], ["failover", 0]];
          G.text(24, 36, "manifest.yaml sections (two have data)", { size: 12, color: "muted" });
          const out = [];
          secs.forEach(([n, has], i) => {
            const y = 52 + i * 52;
            out.push(G.box(24, y, 150, 40, n, { stroke: "ink", size: 13 }));
            if (has) { out.push(G.box(190, y, 250, 40, has, { fill: "k", size: 12 })); out.push(G.box(456, y, 160, 40, "sha256 listed", { fill: "ok", size: 12 })); }
            else out.push(G.box(190, y, 426, 40, "TODO(run-on: g6.xlarge)", { stroke: "hot", dash: "5 4", color: "hot", size: 13 }));
          });
          G.label(24, 380, "no file → TODO, never a made-up number", { size: 13, color: "ink" });
          G.from(out, { opacity: 0, x: -12, stagger: 0.06, duration: 0.3 });
          G.caption("a gap in the data is shown as a gap in the table");
        } },

      { rail: "what travels", title: "A number needs its context",
        body: `<p>"2,400 tokens a second" tells a stranger nothing they can act on. For another person to get the same result, the report must state everything that changes the answer:</p>
<ul><li><b>Hardware:</b> instance type, GPU, driver and CUDA versions.</li>
<li><b>Model and precision:</b> which weights, and the dtype (bf16, fp8, int4).</li>
<li><b>Engine and flags:</b> the version or commit (pinned in <code>SOURCES.md</code>) plus batch limits, context length and any scheduler options.</li>
<li><b>Workload:</b> the arrival process, prompt and output lengths, the random seed (the loadgen arguments).</li>
<li><b>SLOs:</b> the latency targets used to count a request as "good", since goodput depends on them.</li>
<li><b>Runs and spread:</b> how many times you ran it and how much the results varied (next step).</li></ul>
<p>In the repo the <b>Method</b> section of the teardown holds these, and each generated section prints its <code>Hardware:</code> line. Spec-sheet figures (peak TFLOPS, GB/s) are not measurements: cite them as specs with their source, and show your measured fraction of them beside.</p>`,
        check: { q: "Which of these is NOT needed for a stranger to reproduce a throughput number?",
          options: ["The engine version and its batch and context-length flags", "Your opinion of which engine is better", "The workload definition: arrival process, lengths and seed"], answer: 1,
          why: "Reproduction needs the inputs that change the number: hardware, model and dtype, engine version and flags, the workload, the SLOs and the run count with spread. An opinion is a conclusion, not an input." },
        scene(G) {
          const cx = 320, cy = 210;
          const items = [["hardware", 30, 64], ["model + dtype", 30, 150], ["engine + flags", 30, 236], ["workload + seed", 470, 64], ["SLOs", 470, 150], ["runs + spread", 470, 236]];
          const bx = [];
          items.forEach(([t, x, y], i) => { bx.push(G.box(x, y, 140, 44, t, { fill: "k", size: 12 })); const ex = x < 300 ? x + 140 : x; G.line(ex, y + 22, ex < 300 ? cx - 56 : cx + 56, cy, { color: "muted", dash: "3 4" }); });
          G.rect(cx - 56, cy - 40, 112, 80, { fill: "bg", rx: 12 }); G.rect(cx - 56, cy - 40, 112, 80, { stroke: "v", sw: 2, rx: 12 });
          G.text(cx, cy - 2, "2,400", { anchor: "middle", size: 20, color: "v", weight: 700 }); G.text(cx, cy + 20, "tok/s (example)", { anchor: "middle", size: 11, color: "muted" });
          G.label(24, 330, "spec-sheet values are specs: cite the source,", { size: 12, color: "ink" });
          G.label(24, 350, "show measured as a fraction of the spec", { size: 12, color: "ink" });
          G.label(24, 392, "the 2,400 above is a placeholder, not a result", { size: 11 });
          G.from(bx, { opacity: 0, scale: 0.8, transformOrigin: "center", stagger: 0.08, duration: 0.3 });
          G.caption("strip any of the six and the number cannot be repeated");
        } },

      { rail: "variance", title: "Is the gap real, or is it noise?",
        body: `<p>Run the same benchmark twice and you get two different numbers: scheduling, clocks, neighbouring workloads and caches all vary. This <b>run-to-run spread</b> sets how large a difference you are allowed to claim.</p>
<p>Take a made-up practice case. Your engine measures 94, 100, 106 tokens/s over 3 runs (mean 100). vLLM measures 102, 108, 114 (mean 108). The gap is 8%. The spread inside each group is ±6.</p>
<div class="eq">sd (each group)      = 6   (deviations 0, −6, +6)
std error of a mean  = 6 / √3 = 3.46
std error of the gap = √(3.46² + 3.46²) = 4.90
t = 8 / 4.90 = 1.63
needed ≈ 2.78 (4 degrees of freedom)</div>
<p>The 95% interval for the gap is 8 ± 2.776 × 4.90 = −5.6 to +21.6. It includes zero, so you <b>cannot claim</b> vLLM is faster. You can claim only that the two are within noise on this setup.</p>
<p>More runs shrink the interval, because the standard error falls as 1/√n. With 6 runs per engine and the same ±6 spread, the standard error of the gap is 6 × √(2/6) = 3.46, t = 2.31 against a needed 2.23, and the interval is 8 ± 7.7 = 0.3 to 15.7: it just excludes zero. <b>Report mean and spread, and say how many runs.</b></p>
<details><summary>Go deeper: why t and not 2</summary><p>With few runs the standard deviation is itself uncertain, so the bar is higher than the familiar 1.96. Student's t distribution gives the right value for each degrees-of-freedom count (here 2n − 2 for two equal-sized groups). The simulator uses a standard t table. This simplification assumes similar spread in both groups; a Welch test relaxes that.</p></details>`,
        check: { q: "Your engine is 8% slower than vLLM, but run-to-run spread is ±6% and you have 3 runs each. What can you claim?",
          options: ["vLLM is 8% faster", "They are within noise of each other here, unless more runs tighten the interval", "Your engine has a bug"], answer: 1,
          why: "The 95% interval for the gap runs from about −5.6 to +21.6, which includes zero. Claiming 8% would report noise as a result." },
        scene(G) {
          const X = (v) => 60 + ((v - 85) / 35) * 520;
          G.text(24, 34, "runs (tok/s)", { size: 12, color: "muted" });
          G.line(60, 100, 580, 100, { color: "line" });
          [90, 100, 110, 120].forEach((v) => { G.line(X(v), 96, X(v), 104, { color: "line" }); G.label(X(v), 118, String(v), { anchor: "middle", size: 10 }); });
          [94, 100, 106].forEach((v) => G.circle(X(v), 80, 7, { fill: "k" })); [102, 108, 114].forEach((v) => G.circle(X(v), 60, 7, { fill: "v" }));
          G.text(X(94) - 14, 84, "A", { anchor: "end", size: 12, color: "k" }); G.text(X(102) - 14, 64, "B", { anchor: "end", size: 12, color: "v" });
          G.text(24, 146, "95% interval for the gap (B − A), in tok/s", { size: 12, color: "muted" });
          const D = (v) => 60 + ((v + 10) / 35) * 520;
          G.line(D(0), 176, D(0), 340, { color: "hot", w: 2 }); G.text(D(0), 170, "0 = no difference", { anchor: "middle", size: 11, color: "hot" });
          [-10, 0, 10, 20].forEach((v) => { G.label(D(v), 356, String(v), { anchor: "middle", size: 10 }); });
          G.line(D(-5.6), 214, D(21.6), 214, { color: "q", w: 6 }); G.circle(D(8), 214, 6, { fill: "ink" }); G.text(D(0) + 12, 240, "3 runs: −5.6 to 21.6", { size: 12, color: "q" }); G.text(D(0) + 12, 258, "includes 0: cannot claim", { size: 11, color: "hot" });
          G.line(D(0.28), 292, D(15.72), 292, { color: "ok", w: 6 }); G.circle(D(8), 292, 6, { fill: "ink" }); G.text(D(0.28), 318, "6 runs: 0.3 to 15.7", { size: 12, color: "ok" });
          G.text(D(0.28) + 190, 318, "excludes 0", { size: 11, color: "ok" });
          G.from(G.svg.querySelectorAll("line[stroke-width='6']"), { opacity: 0, duration: 0.5, stagger: 0.2 });
          G.caption("same 8% gap: only the interval that excludes 0 is a claim");
        } },

      { rail: "fair comparison", title: "Compare like with like, and price it honestly",
        body: `<p>Three comparison traps, all avoidable:</p>
<ul><li><b>Best run versus typical run.</b> Comparing your best result with someone else's typical one is cherry-picking. Same hardware, same limits, same number of runs, and report each side the same way.</li>
<li><b>Peak versus measured.</b> A GPU's spec-sheet TFLOPS is a ceiling, not something you measured. Quote the spec with its source, then your measurement as a fraction of it.</li>
<li><b>Cost at 100% utilisation.</b> Cost per million tokens is price divided by throughput, and real fleets rarely run at the knee all day:</li></ul>
<div class="eq">usd_per_mtok = price_per_hour / (tok_s × 3600) × 10⁶
             / utilisation</div>
<p>Practice values (a placeholder price, not a real one, and a made-up throughput): $1.00 per hour and 1,000 tok/s at 100% utilisation gives 1.00 / (1,000 × 3,600) × 10⁶ = $0.278 per million tokens. At 40% utilisation the same machine costs 0.278 / 0.4 = $0.694: two and a half times more. State the utilisation you assume, and give the price's source, region and date (on-demand or spot): the report's <code>cost</code> section has columns for <code>price_source</code> for this reason.</p>`,
        check: { q: "A GPU serves 1,000 tok/s at its knee. Your cost table assumes it always runs at the knee, but real traffic keeps it 40% busy. What happens to the true cost per token?",
          options: ["It stays the same", "It rises by 1 / 0.4 = 2.5 times", "It falls, because idle time is free"], answer: 1,
          why: "You pay for every hour whether or not the GPU is busy, but only get 40% of the knee throughput on average. Cost per token scales as 1 / utilisation: 0.278 → 0.694 dollars per million tokens in the practice case." },
        scene(G) {
          const u = [1, 0.7, 0.4, 0.2], base = 0.2778;
          const vals = u.map((x) => base / x);
          const b = G.bars(80, 340, vals, { w: 80, gap: 36, h: 230, max: 1.45, fill: (i) => (i === 0 ? "ok" : i === 3 ? "hot" : "v") });
          vals.forEach((v, i) => { G.text(120 + i * 116, 340 - (230 * v) / 1.45 - 10, "$" + v.toFixed(3), { anchor: "middle", size: 13 }); G.text(120 + i * 116, 362, Math.round(u[i] * 100) + "% busy", { anchor: "middle", size: 12 }); });
          G.line(70, 340, 580, 340, { color: "line" });
          G.text(24, 36, "$ per million tokens: $1.00/h, 1,000 tok/s at the knee", { size: 12, color: "muted" });
          G.label(24, 392, "placeholder price and throughput: use your measured values", { size: 11 });
          G.from(b, { attr: { height: 0, y: 340 }, stagger: 0.1, duration: 0.5 });
          G.caption("idle hours still bill: cost per token scales as 1 / utilisation");
        } },

      { rail: "the report", title: "Shape the report so it can be audited",
        body: `<p>The generated tables are the evidence. The report around them has a fixed shape so a reader can find any answer fast:</p>
<ol><li><b>Summary:</b> three sentences and the one table people will screenshot (#0 v1 versus vLLM, cost per million tokens).</li>
<li><b>Architecture:</b> the platform diagram (gateway, engines, observability, autoscaling, tenancy) in Mermaid, plus a versions table: every component's version or commit from <code>SOURCES.md</code>.</li>
<li><b>Method:</b> hardware, model and dtype, engine flags, the loadgen arguments, SLOs, how many runs and how you measured spread.</li>
<li><b>Results:</b> the <i>generated</i> sections: serving knee, bake-off, quantization, cost, custom kernels, failover.</li>
<li><b>Reproduce:</b> per section, the exact commands and the expected run time and cost.</li>
<li><b>What didn't work:</b> dead ends, regressions, numbers that did not reproduce, and what you would do next.</li></ol>
<p>The last section is hand-written (the builder only leaves a placeholder for it) and it is the most-read part of any good teardown. It is also what shows the results above were not cherry-picked. The acceptance bar is at least three entries. One more rule: no digit appears in your prose that is not in a generated table, so a number can never drift from its data.</p>`,
        scene(G) {
          const secs = [["1  Summary", "v", "hand-written + the screenshot table"], ["2  Architecture", "v", "Mermaid + versions from SOURCES.md"], ["3  Method", "v", "hardware, flags, workload, SLOs, runs"], ["4  Results", "k", "generated from results/*.json"], ["5  Reproduce", "v", "commands, run time, cost"], ["6  What didn't work", "v", "at least three entries"]];
          const out = [];
          secs.forEach(([t, c, d], i) => { const y = 40 + i * 56; out.push(G.box(24, y, 190, 44, t, { fill: c, size: 13 })); out.push(G.text(228, y + 27, d, { size: 12 })); });
          G.rect(18, 40 + 3 * 56 - 5, 604, 54, { stroke: "k", dash: "5 3", rx: 8 });
          G.label(24, 392, "teal-blue = generated · amber = written by a person", { size: 12, color: "ink" });
          G.from(out, { opacity: 0, x: -14, stagger: 0.08, duration: 0.3 });
          G.caption("only section 4 is generated; the others need judgement");
        } },

      { rail: "reproduce", title: "Prove it from a clean clone",
        body: `<p>A report is finished when someone other than you has reproduced it. Two kinds of number have two kinds of check:</p>
<ul><li><b>T0 numbers</b> (the tests, calculators and simulators that run on a laptop) must reproduce <b>exactly</b>: same inputs, same code, same output.</li>
<li><b>T2 and T3 numbers</b> (one GPU, several GPUs or two regions) must land <b>within the variance you stated</b>, on the stated hardware. They will never match to the last digit, and pretending otherwise is the failure.</li></ul>
<p>The acceptance checklist, from the module:</p>
<ul><li>A clean clone plus the README reproduces the T0 numbers exactly (<code>uv run pytest</code>).</li>
<li>Every T2/T3 number links to a committed raw JSON, and its section states hardware and variance.</li>
<li>No number in the prose is missing from a generated table.</li>
<li>Prices quote their source and date.</li>
<li>"What didn't work" has at least three entries.</li></ul>
<p>Then do the humbling step (exercise 3): follow your own instructions from a fresh clone on fresh instances and fix every place they were wrong. Tear the instances down afterwards; the AWS guides state cost, auto-stop and teardown.</p>`,
        scene(G) {
          G.box(24, 50, 130, 50, "fresh clone", { stroke: "ink", size: 13 }); G.arrow(154, 75, 196, 75, { color: "ink" });
          G.box(198, 50, 180, 50, "follow the README", { stroke: "ink", size: 13 });
          G.arrow(378, 75, 420, 56, { color: "ink" }); G.arrow(378, 75, 420, 100, { color: "ink" });
          G.box(422, 36, 196, 38, "T0: exact match", { fill: "ok", size: 13 }); G.box(422, 84, 196, 38, "T2/T3: inside the band", { fill: "q", size: 13 });
          G.text(24, 176, "T2/T3: reported 100 tok/s, spread ±6", { size: 12, color: "muted" });
          G.rect(60, 196, 520, 40, { stroke: "line", rx: 4 });
          const X = (v) => 60 + ((v - 80) / 40) * 520;
          G.rect(X(94), 196, X(106) - X(94), 40, { fill: "q", opacity: 0.35, rx: 0 });
          [[96, "ok"], [99, "ok"], [103, "ok"], [112, "hot"]].forEach(([v, c]) => G.circle(X(v), 216, 7, { fill: c }));
          [80, 90, 100, 110, 120].forEach((v) => G.label(X(v), 256, String(v), { anchor: "middle", size: 10 }));
          G.label(X(112), 190, "outside: investigate", { anchor: "middle", size: 11, color: "hot" });
          G.label(24, 300, "outside the band means one of two things:", { size: 12, color: "ink" });
          G.label(24, 322, "the instructions missed a setting, or the spread was understated", { size: 12 });
          G.label(24, 360, "T0 is a unit test; T2/T3 is a measurement with an error bar", { size: 12, color: "ink" });
          G.caption("exact where you can be, honest about the band where you can't");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>The standard: a stranger with your repo, the stated hardware and a credit card reproduces every number.</li>
<li>Generate the report from raw JSON; every number links to its file, and an appendix lists each input with its sha256 and the commit.</li>
<li>Missing data renders TODO. The builder only copies values, so it cannot invent one.</li>
<li>Ship each number with hardware, model and dtype, engine version and flags, workload, SLOs, and runs with spread.</li>
<li>A gap is a claim only if the interval for the difference excludes zero. More runs shrink it as 1/√n.</li>
<li>Compare like with like, quote specs as specs, and state the utilisation behind a cost per token.</li>
<li>Include a real "what didn't work" section, and check the report from a clean clone.</li></ul>
<p>That is the end of the core course. Lane B (L1 to L6) keeps building CUDA skill alongside it.</p>`,
    sim: {
      title: "Can you claim the gap?",
      intro: "Enter the two engines' mean throughput, the run-to-run spread and the number of runs per engine. The chart shows each engine's run-to-run spread (wide, faint) and the uncertainty of its mean (narrow). The panel tests whether the gap is distinguishable from noise and how many runs it would take.",
      height: 280,
      controls: [
        { id: "a", label: "engine A mean (tok/s)", min: 50, max: 200, step: 1, value: 100 },
        { id: "b", label: "engine B mean (tok/s)", min: 50, max: 200, step: 1, value: 108 },
        { id: "sd", label: "run-to-run spread (% of A, one sd)", min: 1, max: 15, step: 0.5, value: 6 },
        { id: "n", label: "runs per engine", min: 2, max: 30, step: 1, value: 3 },
      ],
      draw(G, v) {
        const A = v.a, B = v.b, s = (v.sd / 100) * A, n = v.n;
        const seDiff = s * Math.sqrt(2 / n), diff = B - A, t = diff / seDiff, tc = tcrit(2 * n - 2);
        const sig = Math.abs(t) > tc;
        let need = null; for (let m = 2; m <= 500; m++) { if (Math.abs(diff) / (s * Math.sqrt(2 / m)) > tcrit(2 * m - 2)) { need = m; break; } }
        const lo = Math.min(A, B) - 3.2 * s, hi = Math.max(A, B) + 3.2 * s, X = (x) => 30 + ((x - lo) / (hi - lo)) * 580, base = 220;
        const curve = (mu, sd, amp, color, op) => {
          let d = ""; for (let i = 0; i <= 120; i++) { const x = lo + ((hi - lo) * i) / 120, y = Math.exp(-0.5 * ((x - mu) / sd) ** 2); d += (i ? " L " : "M ") + X(x).toFixed(1) + " " + (base - amp * y).toFixed(1); }
          G.path(d + ` L ${X(hi)} ${base} L ${X(lo)} ${base} Z`, { fill: color, color, w: 1.5, opacity: op });
        };
        curve(A, s, 70, "k", 0.18); curve(B, s, 70, "v", 0.18);
        const se = s / Math.sqrt(n);
        curve(A, se, 150, "k", 0.65); curve(B, se, 150, "v", 0.65);
        G.line(30, base, 610, base, { color: "line" });
        for (let i = 0; i <= 4; i++) { const x = lo + ((hi - lo) * i) / 4; G.label(X(x), base + 16, x.toFixed(0), { anchor: "middle", size: 10 }); }
        G.text(X(A), 36, "A", { anchor: "middle", size: 14, color: "k" }); G.text(X(B), 36, "B", { anchor: "middle", size: 14, color: "v" });
        G.label(30, 258, "faint: spread of single runs · solid: uncertainty of each engine's mean (spread / √n)", { size: 11 });
        const ci = tc * seDiff;
        return [
          { title: "The gap", rows: [["B − A", `${diff.toFixed(1)} tok/s (${((100 * diff) / A).toFixed(1)}%)`], ["std error of the gap", seDiff.toFixed(2)], ["t statistic", t.toFixed(2)], ["needed (95%, df " + (2 * n - 2) + ")", tc.toFixed(2)], ["95% interval for the gap", `${(diff - ci).toFixed(1)} to ${(diff + ci).toFixed(1)}`]],
            chip: [sig, sig ? "the gap is distinguishable from noise" : "within noise: do not claim a winner"] },
          { title: "Runs needed", rows: [["now", `${n} per engine`], ["to clear the bar", need ? `${need} per engine` : "more than 500 (gap too small)"]],
            html: `<p class="note">t table values, equal spread in both groups, runs assumed independent. Real runs from one instance can be correlated: spread them over several instances (exercise 4).</p>` },
        ];
      },
      note: "Teaching model with made-up default numbers. It tests a difference of means; it does not replace looking at the raw runs.",
    },
    practice: {
      intro: `<p>The report builder and its tests are T0 and run on a laptop. Exercises 3 and 4 need GPU instances (T2/T3) and cost money: follow the AWS guides for the stated cost, auto-stop and teardown, and keep every endpoint private. The report itself is built with:</p>
<p><code>PYTHONPATH=platform uv run python -m report.build --root . --out reports/teardown.md</code></p>`,
      items: [
        { title: "A new section kind", tier: "T0 · medium", goal: "Add a section kind to the builder, for example compare, that computes percent-of-baseline from two result files, links both, and has a test.",
          cmd: "uv run pytest platform/report -q" },
        { title: "Game-day JSON into the report", tier: "T0 · easy", goal: "Give failover.gameday a flag that writes its table as JSON, then point the manifest's failover section (results/c1/gameday.json) at it.",
          cmd: "uv run pytest platform/failover platform/report -q" },
        { title: "Reproduce it yourself", tier: "T2/T3 · costs money", goal: "From a fresh clone on fresh instances, follow your own instructions. Record every place they were wrong, and fix them.",
          cmd: "uv run pytest -q   # T0 numbers must match exactly; then re-run each T2/T3 section on its stated hardware" },
        { title: "Variance (hard)", tier: "T2 · hard · costs money", goal: "Rerun the serving knee three times on three instances of the same type. Report mean, spread, and whether your engine-versus-vLLM conclusion survives it.",
          cmd: "PYTHONPATH=platform uv run python -m loadgen.cli --url http://127.0.0.1:8000 --rates 1 2 4 8 16 32 --duration 60 --ttft-slo 2" },
      ],
      labs: [
        { label: "Module README, quiz and acceptance checklist", path: "course/capstone/C2-public-benchmark-teardown/" },
        { label: "The builder and its manifest", path: "platform/report/" },
        { label: "Load generator used for the knee (P2.4)", path: "platform/loadgen/" },
        { label: "Pinned versions for the versions table", path: "SOURCES.md" },
        { label: "Game-day analysis feeding the failover section", path: "platform/failover/gameday.py" },
      ],
    },
  });
})();
