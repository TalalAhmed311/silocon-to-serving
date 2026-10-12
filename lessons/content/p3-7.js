/* P3.7 — Weight delivery: hops, safetensors, parallel S3, streaming loaders, an immutable registry and a node-local LRU cache (#5). */
(function () {
  const F = S2S.fmt;

  /* time-to-weights-on-GPU for the three delivery methods, from per-hop rates in GB/s.
     A cold: download to disk (S3 and disk write), then read back and copy to the GPU (disk read and PCIe overlap per tensor).
     A warm: only the read-back and copy.   B stream: S3 into pinned host buffers, copied to the GPU as ranges land. */
  function methods(v) {
    const G = v.gb, S = Math.min(v.c * v.r, v.net);
    const dl = G / Math.min(S, v.dw), ld = G / Math.min(v.dr, v.pcie), st = G / Math.min(S, v.pcie);
    const hopOf = (pairs) => pairs.reduce((a, b) => (b[1] < a[1] ? b : a))[0];
    return {
      S, dl, ld, st,
      cold: dl + ld, warm: ld,
      hopCold: hopOf([["S3 streams", v.c * v.r], ["network", v.net], ["disk write", v.dw]]),
      hopLoad: hopOf([["disk read", v.dr], ["PCIe", v.pcie]]),
      hopStream: hopOf([["S3 streams", v.c * v.r], ["network", v.net], ["PCIe", v.pcie]]),
    };
  }
  const EX = { gb: 16, r: 0.1, c: 16, net: 1.25, dw: 1.0, dr: 2.0, pcie: 12 };   // example rates, GB/s

  S2S.lesson({
    id: "p3-7", n: "P3.7", title: "Weight delivery",
    subtitle: "Deployment and infra · first principles · T0 for manifest and cache, T2/T3 for S3 and GPU runs",
    kicker: "Lesson · ≈ 50 min",
    headline: "Sixteen gigabytes that every new GPU must receive before it is useful",
    intro: `<p>P3.6 showed that a cold start has four phases, and one of them is "weight load". For an 8B model that is about 16 GB of numbers that must travel from storage into GPU memory every time a replica starts. This lesson follows those bytes hop by hop, learns the file format that makes lazy loading possible, finds out why one S3 connection is slow and sixteen are not, compares download-then-load with streaming, and builds project #5: a registry of immutable, checksummed model versions and a node-local cache that evicts the least recently used.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "bytes in flight"], ["v", "disk"], ["q", "GPU"], ["hot", "bottleneck"]],
    prev: "p3-6", next: "p3-8",
    steps: [
      /* 1 */
      { rail: "the problem", title: "The weights are most of the bytes, and every replica needs them",
        body: `<p>A model's <b>weights</b> are its learned numbers. Llama-3-8B has about 8.0 billion of them. In bf16 each takes 2 bytes, so the file is about <b>16 GB</b> (8.0 × 10⁹ × 2). A 70B model is about <b>141 GB</b> (70.6 × 10⁹ × 2). Compare that with the container image from P3.1, an example 9 GB compressed.</p>
<p>A server cannot answer anything until all of its weights are in GPU memory. So every time a replica starts (a scale-up in P3.6, a rolling update in P3.4, a restart) those bytes must arrive again. The time is always the same division:</p>
<div class="eq">load time = bytes ÷ effective rate
16 GB ÷ 1 GB/s   =  16 s
16 GB ÷ 0.1 GB/s = 160 s
141 GB ÷ 1 GB/s  = 141 s</div>
<p>"Effective rate" is the hard part: the bytes pass through several hops, each with its own speed, and the rate you get depends on how you chain them. This lesson is about that chain. Notice also what is <i>not</i> a good idea: baking the weights into the image (P3.1) makes every image pull carry them, and every model update rebuilds the image.</p>`,
        scene(G) {
          const sc = 560 / 141;
          const rows = [["container image (example, P3.1)", 9, "w"], ["Llama-3-8B, bf16", 16, "k"], ["Llama-3-70B, bf16", 141, "q"]];
          G.text(24, 34, "bytes a new replica must fetch (to scale)", { size: 13 });
          const bs = rows.map(([l, g, c], i) => { const y = 66 + i * 68; G.label(24, y - 6, l, { size: 12, color: "ink" }); const r = G.rect(24, y, g * sc, 32, { fill: c, rx: 4 }); G.text(24 + g * sc + 8, y + 22, g + " GB", { size: 13 }); return r; });
          G.rect(24, 276, 592, 124, { stroke: "line", rx: 10 });
          G.text(40, 302, "load time = bytes ÷ effective rate", { size: 13, color: "ink" });
          G.text(40, 330, "16 GB at 1 GB/s = 16 s     at 0.1 GB/s = 160 s", { size: 13 });
          G.text(40, 356, "141 GB at 1 GB/s = 141 s", { size: 13 });
          G.label(40, 384, "paid on every start, scale-up and rolling update", {});
          G.from(bs, { attr: { width: 0 }, stagger: 0.25, duration: 0.5 });
          G.caption("the weights dwarf the image, and move every time");
        } },

      /* 2 */
      { rail: "the hops", title: "The bytes cross six hops, and the slowest one sets the pace",
        body: `<p>From the registry to GPU memory the bytes take this path:</p>
<ol><li><b>S3 to the instance</b> over the network. Each connection delivers a modest rate; many connections add up, until the instance's network bandwidth.</li>
<li><b>Network to local disk</b>: an NVMe drive (fast) or an EBS gp3 volume (limited by the throughput you provisioned).</li>
<li><b>Disk to the page cache</b>: the operating system's RAM copy of file pages (P0.2). A cold read comes from the disk; a warm one from RAM.</li>
<li><b>Page cache to host tensors</b>: a memory copy, or nothing at all if the file is <code>mmap</code>ped.</li>
<li><b>Host to GPU</b> over PCIe. Pinned (page-locked) memory copies faster than pageable memory (P0.2).</li></ol>
<p>How the hops combine matters more than any one speed. Done <b>one after another</b> (finish a hop for all 16 GB, then start the next) the times <b>add</b>. <b>Pipelined</b> (a byte moves on as soon as it arrives) the whole chain runs at the speed of the <b>slowest hop</b>, plus a little start-up. Example rates (<b>example values; measure yours with <code>weights.bench</code></b>): network 1.25 GB/s, disk write 1.0, disk read 2.0, PCIe 12.</p>
<div class="eq">one after another:
  16/1.0 + 16/2.0 + 16/12
  = 16 + 8 + 1.33 = 25.3 s
pipelined, slowest hop (disk write 1.0):
  16/1.0 = 16 s</div>
<p>The levers are therefore: find the slowest hop, and overlap the rest behind it.</p>`,
        scene(G) {
          const hops = [["S3", "network", 1.25], ["disk", "write", 1.0], ["disk", "read", 2.0], ["host", "RAM", null], ["GPU", "PCIe", 12]];
          G.text(24, 34, "one 16 GB model, example hop rates (GB/s)", { size: 13 });
          const names = ["S3 → NIC", "→ disk", "disk → cache", "→ host RAM", "host → GPU"];
          const rate = [1.25, 1.0, 2.0, null, 12];
          const boxes = names.map((n, i) => G.box(24 + i * 120, 60, 108, 56, n, { fill: i === 1 ? "hot" : "k", size: 11 }));
          [0, 1, 2, 3].forEach((i) => G.arrow(132 + i * 120, 88, 144 + i * 120, 88, { color: "muted", w: 2 }));
          rate.forEach((r, i) => { if (r) { G.label(78 + i * 120, 140, r + " GB/s", { anchor: "middle", size: 11 }); G.rect(24 + i * 120 + 24, 262 - Math.min(r * 22, 110), 60, Math.min(r * 22, 110), { fill: i === 1 ? "hot" : "k", rx: 3, opacity: 0.9 }); } else G.label(78 + i * 120, 140, "memcpy / mmap", { anchor: "middle", size: 11 }); });
          G.line(24, 262, 616, 262, { color: "line" });
          G.rect(24, 292, 592, 108, { stroke: "line", rx: 10 });
          G.text(40, 318, "one after another: 16 + 8 + 1.33 = 25.3 s", { size: 13 });
          G.text(40, 344, "pipelined: the slowest hop sets it, 16 ÷ 1.0 = 16 s", { size: 13, color: "ok" });
          G.label(40, 374, "bar height = hop rate (the 12 GB/s PCIe bar is cut short)", {});
          G.label(40, 392, "all rates are example values: measure yours", {});
          G.from(boxes, { opacity: 0, x: -14, stagger: 0.12, duration: 0.3 });
          G.caption("overlap the hops, then speed up the slowest one");
        } },

      /* 3 */
      { rail: "safetensors", title: "A file you can point into without reading it",
        body: `<p>Why can a loader start using tensor 1 before tensor 500 has arrived? Because of the file format. <b>safetensors</b> is built for exactly this:</p>
<ol><li>8 bytes: the length <i>N</i> of the header.</li>
<li>The <b>header</b>, <i>N</i> bytes of JSON: for each tensor its name, dtype, shape and <code>data_offsets</code> (start and end byte, relative to the data area).</li>
<li>The raw bytes of all tensors, back to back, no padding tricks.</li></ol>
<p>Everything needed to find a tensor is in the header. Example: <code>q_proj</code> of one Llama-3-8B layer is a 4096 × 4096 bf16 matrix. Its size is 4096 × 4096 × 2 bytes = 33,554,432 bytes (32 MiB), so its entry looks like <code>"dtype": "BF16", "shape": [4096, 4096], "data_offsets": [0, 33554432]</code> and the next tensor starts at 33,554,432.</p>
<p>Because offsets are explicit, a loader can <b>mmap</b> the file (map it into memory without reading it) and hand out tensors that are views at those offsets. The OS reads a page only when something touches it (a <i>page fault</i>, P0.2). Nothing is parsed or executed, so loading runs no code.</p>
<p>Contrast <b>pickle</b>, the older PyTorch checkpoint format: it is a program that rebuilds Python objects. To load it you must run it, so it cannot be lazy and it can run arbitrary code from whoever made the file.</p>`,
        check: { q: "Why can a safetensors file be memory-mapped and read lazily, while a pickle checkpoint cannot?",
          options: ["safetensors files are compressed", "safetensors has explicit byte offsets in a JSON header, so a tensor is just a view into the file; pickle must be executed to rebuild its objects", "pickle files are always larger than memory"], answer: 1,
          why: "With explicit dtype, shape and offsets, finding a tensor needs only the header and no execution. Pickle encodes a sequence of instructions that build objects, so you have to run it from the start, and doing so can execute arbitrary code." },
        scene(G) {
          G.text(24, 34, "model.safetensors", { size: 13 });
          G.box(24, 56, 56, 56, "8 B", { fill: "w", size: 12 }); G.label(24, 128, "len N", { size: 11 });
          G.box(86, 56, 200, 56, "JSON header (N bytes)", { fill: "v", size: 12 }); G.label(92, 128, "dtype · shape · offsets", { size: 11 });
          const w = [90, 70, 70, 40];
          let x = 292; const ts = ["q_proj", "k_proj", "v_proj", "…"].map((n, i) => { const b = G.box(x, 56, w[i], 56, n, { fill: "k", size: 12 }); x += w[i] + 4; return b; });
          G.label(292, 128, "raw tensor bytes", { size: 11 });
          G.rect(24, 164, 592, 92, { stroke: "v", rx: 8, dash: "5 4" });
          ['"q_proj": {', '  "dtype": "BF16", "shape": [4096, 4096],', '  "data_offsets": [0, 33554432] }'].forEach((t, i) => G.text(40, 188 + i * 22, t, { size: 12 }));
          G.arrow(250, 112, 340, 160, { color: "v", w: 1.5, dash: "4 3" });
          G.text(24, 292, "4096 × 4096 × 2 bytes = 33,554,432 B (32 MiB)", { size: 13, color: "ok" });
          G.label(24, 322, "mmap: map the file, touch a page, the OS reads that page only", { color: "ink" });
          G.label(24, 346, "no pickle: nothing runs when the file loads", { color: "ink" });
          G.label(24, 370, "pickle must be executed to rebuild its objects: not lazy, not safe", { color: "hot" });
          G.from(ts, { opacity: 0, y: 14, stagger: 0.12, duration: 0.3 });
          G.caption("the header is a map; the data is untouched until needed");
        } },

      /* 4 */
      { rail: "shards", title: "Splitting into shards lets many hands carry the model",
        body: `<p>One 16 GB file is awkward: one writer, one reader, one connection. Large checkpoints are therefore split into <b>shards</b>, several safetensors files of a few GB each. (Llama-3-8B in bf16 is typically published as four shards of about 4 GB.)</p>
<p>A small file ties them together, <code>model.safetensors.index.json</code>. Its <code>weight_map</code> says which shard holds each tensor:</p>
<div class="eq">"weight_map": {
  "model.embed_tokens.weight":
                    "model-00001-of-00004.safetensors",
  ...
  "lm_head.weight":
                    "model-00004-of-00004.safetensors" }</div>
<p>The loader reads the index, then reads shards <b>in parallel</b>. That matters twice: four independent files can be fetched over four S3 connections at once (next step), and with several workers on different shards a tensor-parallel setup (P4) can load only the shards it needs. <code>weights.manifest</code> reads this file in <code>safetensors_shards()</code>.</p>
<p>The index is also a checklist: if a shard listed in the <code>weight_map</code> is missing or truncated, the load fails <i>for a clear reason</i>, provided you check. Step 8 builds that check into the registry.</p>`,
        scene(G) {
          G.box(24, 56, 592, 44, "model.safetensors.index.json  →  weight_map: tensor name → shard", { stroke: "v", color: "v", size: 12 });
          const sh = [0, 1, 2, 3].map((i) => {
            const x = 24 + i * 150, g = G.group();
            G.rect(x, 150, 136, 100, { stroke: "k", rx: 8, parent: g });
            G.text(x + 68, 176, "shard " + (i + 1), { anchor: "middle", size: 13, color: "k", parent: g });
            G.label(x + 68, 198, "model-0000" + (i + 1), { anchor: "middle", size: 10, parent: g }); G.label(x + 68, 214, "-of-00004", { anchor: "middle", size: 10, parent: g });
            G.text(x + 68, 238, "≈ 4 GB", { anchor: "middle", size: 12, parent: g });
            return g;
          });
          const ar = [0, 1, 2, 3].map((i) => G.arrow(92 + i * 150, 104, 92 + i * 150, 146, { color: "v", w: 1.5, dash: "4 3" }));
          const gets = [0, 1, 2, 3].map((i) => G.arrow(92 + i * 150, 330, 92 + i * 150, 254, { color: "ok", w: 2 }));
          G.box(24, 332, 592, 40, "S3: four ranged GETs in parallel, one per shard", { stroke: "ok", color: "ok", size: 12 });
          G.label(24, 396, "4 shards × ≈ 4 GB ≈ 16 GB · each can be read, verified and loaded on its own", { color: "ink" });
          G.from(sh, { opacity: 0, y: 14, stagger: 0.15, duration: 0.3 }); G.from(gets, { opacity: 0, stagger: 0.1, duration: 0.3, delay: 0.6 });
          G.caption("the index maps names to shards; shards travel in parallel");
        } },

      /* 5 */
      { rail: "cold vs warm", title: "Cold reads come from the disk. Warm reads come from RAM",
        body: `<p>The operating system keeps recently read file pages in unused RAM, the <b>page cache</b> (P0.2). The first read of a file (<b>cold</b>) goes to the disk. A second read (<b>warm</b>) is served from RAM and is many times faster. That is wonderful for serving and a trap for benchmarking.</p>
<p>The trap: you time "reading the model from disk" twice, and the second run reports an astonishing 20 GB/s. You did not measure a disk, you measured memory. A gp3 volume cannot do that. To measure a cold read: drop the caches as root (<code>echo 3 &gt; /proc/sys/vm/drop_caches</code>), or read a file bigger than RAM, or use direct I/O.</p>
<div class="eq">effective rate = model GB ÷ seconds
16 GB in 8 s = 2 GB/s     (plausible: a fast NVMe)
16 GB in 0.8 s = 20 GB/s  (that is the page cache)</div>
<p>The same effect has a good use. A replica that restarts on a node where the model was recently read finds it still warm. In <code>weights.bench</code> the rows <i>read-cold</i>, <i>read-warm</i>, <i>safetensors</i> and <i>mmap</i> exist to show the <b>ratios</b> between these, which are the point; absolute numbers depend on the machine.</p>`,
        check: { q: "Your \"disk read\" benchmark of a 16 GB file on a gp3 volume reports 20 GB/s. What most likely happened?",
          options: ["gp3 is that fast", "The file was already in the page cache, so you timed RAM", "The model was compressed"], answer: 1,
          why: "A network-attached gp3 volume delivers a fraction of that, and the number is the speed of memory. Drop the caches (as root), or read a file larger than RAM, then measure again. Always say whether a read was cold or warm." },
        scene(G) {
          G.text(24, 34, "reading the same 16 GB file twice (shape only, not to scale)", { size: 13 });
          G.box(24, 60, 120, 56, "disk", { fill: "v", size: 13 }); G.box(256, 60, 128, 56, "page cache (RAM)", { fill: "k", size: 12 }); G.box(496, 60, 120, 56, "your process", { fill: "q", size: 13 });
          G.arrow(146, 78, 252, 78, { color: "hot", w: 3 }); G.label(160, 70, "cold: slow", { color: "hot", size: 11 });
          G.arrow(386, 88, 492, 88, { color: "ok", w: 3 }); G.label(396, 80, "warm: fast", { color: "ok", size: 11 });
          G.line(24, 150, 616, 150, { color: "line", dash: "4 4" });
          const bc = G.rect(24, 180, 480, 34, { fill: "hot", rx: 4 }); G.label(24, 172, "first read (cold): 8 s", { color: "ink" }); G.text(512, 202, "2 GB/s", { size: 13 });
          const bw = G.rect(24, 250, 48, 34, { fill: "ok", rx: 4 }); G.label(24, 242, "second read (warm): 0.8 s", { color: "ink" }); G.text(80, 272, "20 GB/s: that is RAM", { size: 13, color: "ok" });
          G.rect(24, 316, 592, 84, { stroke: "hot", rx: 8, dash: "5 4" });
          G.text(40, 342, "benchmark rule: say whether the read was cold or warm", { size: 13, color: "hot" });
          G.label(40, 366, "drop caches (root), or read a file larger than RAM", {});
          G.label(40, 386, "a restarted replica gets warm reads for free: that is a feature", {});
          G.from([bc, bw], { attr: { width: 0 }, stagger: 0.4, duration: 0.6 });
          G.caption("cold measures the disk; warm measures memory");
        } },

      /* 6 */
      { rail: "parallel S3", title: "One S3 connection is slow. Sixteen are not",
        body: `<p>S3 is object storage reached over HTTP. A single connection to it delivers a modest rate, far below what the instance's network card can carry, because each request has latency and S3 serves one stream from limited resources. The fix is to ask for <b>many byte ranges at once</b> (<b>ranged GETs</b>: "send me bytes 0 to 64 MiB of this object"), several shards or several chunks of one shard in parallel.</p>
<p>Model it simply. Let one stream deliver <i>r</i> GB/s and let the destination (network card, or the disk you write to) cap out at <i>B</i> GB/s:</p>
<div class="eq">throughput(c) = min( c × r , B )
example: r = 0.1 GB/s, B = 1.25 GB/s
   c =  1 → 0.10 GB/s   16 GB takes 160 s
   c =  4 → 0.40 GB/s   16 GB takes  40 s
   c =  8 → 0.80 GB/s   16 GB takes  20 s
   c = 16 → min(1.6, 1.25) = 1.25 GB/s   12.8 s
   c = 64 → 1.25 GB/s    12.8 s  (no gain)</div>
<p>Two regions: below the <b>knee</b> (here c = B ÷ r = 12.5, so 13) each added connection helps; above it nothing helps, because some other hop is now the limit. That plateau tells you <i>which</i> hop limits you: compare it with the instance's network bandwidth, then with the disk's throughput (<code>iostat -x 1</code> during the download), and prove it by writing to <code>/dev/shm</code> instead. The numbers above are <b>example values</b>; exercise 3 measures yours.</p>
<p>The classic mistake is to run one stream, see 100 MB/s, and conclude "S3 is slow".</p>`,
        check: { q: "With r = 0.1 GB/s per connection and a 1.25 GB/s cap, what is the smallest number of connections that reaches the cap?",
          options: ["4", "13", "64"], answer: 1,
          why: "The cap is reached when c × 0.1 ≥ 1.25, so c ≥ 12.5, which means 13 whole connections. Fewer leave bandwidth unused. More than that gains nothing because another hop (network card or disk) is the limit." },
        scene(G) {
          const x0 = 70, x1 = 600, yb = 300, yt = 70, ymax = 1.5, cs = [1, 2, 4, 8, 16, 32, 64];
          const X = (i) => x0 + 30 + i * 74, Y = (v) => yb - ((yb - yt) * v) / ymax;
          G.axes(x0, yt, x1 - x0, yb - yt, {});
          G.label(x0, yt - 12, "GB/s from S3, by number of concurrent streams c", { size: 11 });
          [0, 0.5, 1, 1.5].forEach((v) => G.label(x0 - 6, Y(v) + 4, String(v), { anchor: "end", size: 10 }));
          const vals = cs.map((c) => Math.min(c * 0.1, 1.25));
          const bars = G.bars(x0 + 30 - 22, yb, vals, { w: 44, gap: 30, h: yb - yt, max: ymax, fill: (i, v) => (v >= 1.25 ? "ok" : "k") });
          cs.forEach((c, i) => { G.label(X(i), yb + 16, String(c), { anchor: "middle", size: 11 }); G.text(X(i), Y(vals[i]) - 6, vals[i].toFixed(2), { anchor: "middle", size: 11 }); });
          G.line(x0, Y(1.25), x1, Y(1.25), { color: "hot", dash: "5 4", w: 2 }); G.label(x0 + 6, Y(1.25) - 6, "cap B = 1.25 GB/s (network or disk)", { color: "hot", size: 11 });
          G.text(24, 354, "16 GB: c=1 → 160 s · c=4 → 40 s · c=16 → 12.8 s", { size: 13, color: "ink" });
          G.label(24, 380, "knee = B ÷ r = 12.5 → 13 streams · values are examples", {});
          G.label(24, 402, "beyond the knee another hop is the limit: find which one", {});
          G.from(bars, { attr: { height: 0 }, stagger: 0.1, duration: 0.4 });
          G.caption("throughput climbs with streams, then a different hop caps it");
        } },

      /* 7 */
      { rail: "streaming", title: "Do not land the model on disk if you only need it on the GPU",
        body: `<p><b>Download-then-load</b> (A) runs the hops in sequence: pull all 16 GB from S3 onto the local disk, <i>then</i> read it back and copy it to the GPU. The disk is touched twice (written, read), and the GPU waits for the whole download.</p>
<p>A <b>streaming loader</b> (B) overlaps the hops. It issues many concurrent ranged GETs, writes the bytes straight into pinned host buffers, and copies each tensor to the GPU as soon as its range has landed. No disk in the path, and the GPU starts filling while the download is still going. vLLM integrates the Run:ai Model Streamer (pinned 0.16.1):</p>
<div class="eq">vllm serve s3://bucket/models/llama3-8b/REV \\
  --load-format runai_streamer \\
  --model-loader-extra-config '{"concurrency": 16}'</div>
<p>(Flag and key names come from vLLM's <code>docs/models/extensions/runai_model_streamer.md</code> at the pinned SHA: re-check them if you change versions.) With the example rates from step 2 and 16 streams (S3 side 1.25 GB/s):</p>
<div class="eq">A cold, download:  16/min(1.25, 1.0) = 16.0 s
A cold, load:      16/min(2.0, 12)   =  8.0 s
A cold, total                        = 24.0 s
A warm (on NVMe):  16/min(2.0, 12)   =  8.0 s
B stream:          16/min(1.25, 12)  = 12.8 s</div>
<p>So streaming beats a cold download-then-load (12.8 s vs 24.0 s), but a <b>warm</b> local copy beats streaming (8 s vs 12.8 s). Which would you choose for a scale-from-zero node with an empty disk? Streaming. For the tenth restart on the same node? The cache. The simulator lets you change every rate.</p>`,
        scene(G) {
          const m = methods(EX), sc = 560 / 26;
          G.text(24, 34, "time to weights on GPU, 16 GB (example rates, to scale)", { size: 13 });
          const row = (y, label, segs, tot, col) => {
            G.label(24, y - 6, label, { size: 12, color: "ink" });
            let x = 24; const rs = segs.map(([t, c]) => { const r = G.rect(x, y, t * sc, 34, { fill: c, rx: 3 }); x += t * sc; return r; });
            G.text(x + 8, y + 23, tot, { size: 13, color: col || "ink" }); return rs;
          };
          const a = row(70, "A cold: download to disk, then read back + copy", [[m.dl, "v"], [m.ld, "q"]], m.cold.toFixed(1) + " s");
          const w = row(150, "A warm: the file is already on NVMe", [[m.ld, "q"]], m.warm.toFixed(1) + " s", "ok");
          const s = row(230, "B stream: S3 → pinned RAM → GPU, overlapped", [[m.st, "k"]], m.st.toFixed(1) + " s", "ok");
          G.rect(24, 296, 592, 106, { stroke: "line", rx: 10 });
          G.rect(40, 312, 12, 12, { fill: "v", rx: 2 }); G.label(58, 323, "download to disk (S3, disk write)", { size: 11 });
          G.rect(40, 334, 12, 12, { fill: "q", rx: 2 }); G.label(58, 345, "read back and copy to GPU (disk read, PCIe)", { size: 11 });
          G.rect(40, 356, 12, 12, { fill: "k", rx: 2 }); G.label(58, 367, "stream straight to the GPU (S3 side is the limit)", { size: 11 });
          G.label(40, 392, "cold: stream wins · repeated restarts on one node: cache wins", { size: 11, color: "ink" });
          G.from(a.concat(w, s), { attr: { width: 0 }, stagger: 0.2, duration: 0.5 });
          G.caption("streaming skips the disk; a warm cache skips the network");
        } },

      /* 8 */
      { rail: "the registry", title: "A registry of versions that never change",
        body: `<p>Project #5 builds the place weights live. The contract (<code>platform/weights/manifest.py</code>) is a folder layout in S3 or any directory:</p>
<div class="eq">models/&lt;model&gt;/&lt;version&gt;/manifest.json
models/&lt;model&gt;/&lt;version&gt;/model-00001-of-00004.safetensors
models/&lt;model&gt;/&lt;version&gt;/model.safetensors.index.json
models/&lt;model&gt;/&lt;version&gt;/config.json  tokenizer…</div>
<p><code>manifest.json</code> lists every file with its size and <b>SHA-256</b> (a fingerprint of the bytes), plus <code>total_bytes</code>. Two rules make it safe.</p>
<ul><li><b>Versions are immutable.</b> A new upload is a new version; you never overwrite one. Otherwise two replicas can hold different weights under the same name, and "roll back to the previous version" means nothing. A rollout points replicas at a new version; a rollback points them back. Enforce it with S3 bucket versioning and a deny on <code>s3:PutObject</code> to existing version prefixes. "Latest" is a pointer file, never an overwrite.</li>
<li><b>Verify at install, not at every load.</b> A half-downloaded shard otherwise shows up an hour later as a confusing CUDA or shape error. Check <i>size first</i> (free), then the hash. A size check alone misses a flipped bit, and hashing 16 GB costs real time, so do it once when the version lands on the node, not on every start. That assumes the local disk does not silently corrupt files; re-verify after a node problem.</li></ul>
<p>Exercise 1 implements <code>verify(root, manifest)</code>. It reports <code>missing:</code>, <code>size:</code>, <code>sha256:</code> and <code>unexpected:</code> problems, one string each.</p>`,
        check: { q: "Why must a model version in the registry never be overwritten in place?",
          options: ["S3 forbids overwriting objects", "Replicas could serve different weights under the same name, and rollback would be impossible", "Overwriting is slower than uploading a new version"], answer: 1,
          why: "A version name is a promise about exact bytes. If it can change, two replicas that fetched at different times disagree, caches go stale unnoticed, and there is nothing to roll back to. A new version name makes every change explicit and reversible." },
        scene(G) {
          G.text(24, 34, "s3://registry/models/", { size: 13 });
          const tree = [[0, "llama3-8b/", "ink"], [1, "9f3a…  (version v1: frozen)", "ok"], [2, "manifest.json", "v"], [2, "model-0000N-of-00004.safetensors ×4", "k"], [1, "c07d…  (version v2: new upload)", "ok"], [2, "manifest.json", "v"], [2, "model-0000N-of-00004.safetensors ×4", "k"]];
          tree.forEach(([d, t, c], i) => G.text(30 + d * 22, 66 + i * 26, (d ? "└ " : "") + t, { size: 12, color: c }));
          G.rect(24, 262, 330, 138, { stroke: "v", rx: 8 });
          G.text(36, 284, "manifest.json", { size: 12, color: "v" });
          ['{ "model": "llama3-8b", "version": "9f3a…",', '  "total_bytes": 16060000000,', '  "files": [ { "path": "model-00001…",', '     "size": …, "sha256": "…" }, … ] }'].forEach((t, i) => G.text(36, 306 + i * 20, t, { size: 11 }));
          G.rect(374, 262, 242, 138, { stroke: "line", rx: 8 });
          G.text(386, 284, "install checks", { size: 12, color: "ink" });
          ["1  every file present", "2  size matches (free)", "3  sha256 matches", "4  nothing unexpected"].forEach((t, i) => G.text(386, 308 + i * 22, t, { size: 11, color: i === 2 ? "ok" : "ink" }));
          G.label(24, 418, "new bytes = new version; rollback = point back", { size: 11 });
          G.caption("a version names exact bytes, forever");
        } },

      /* 9 */
      { rail: "the cache", title: "A node-local cache that never shows a half-copied model",
        body: `<p>Every node asking S3 for the same 16 GB wastes the slow hop. <code>platform/weights/cache.py</code> keeps versions on the node's NVMe (or a <code>hostPath</code>) under three rules:</p>
<ol><li><b>Atomic install.</b> Fetch into a temp directory (<code>.tmp/&lt;random&gt;</code>), verify against the manifest, then <code>os.replace()</code> it to its final name. A rename within one filesystem is atomic, so readers see either no version or a whole, verified one; a crash leaves only garbage in <code>.tmp/</code>.</li>
<li><b>A size limit with LRU eviction.</b> When a new version does not fit, evict the <b>least recently used</b> first, and stop as soon as it fits. <code>get()</code> updates recency, saved in <code>.lru.json</code> so it survives pod restarts.</li>
<li><b>Pins.</b> A version a running replica is reading is <i>pinned</i> and is never evicted.</li></ol>
<p>The eviction decision is a pure function, <code>plan_evictions(lru, sizes, pinned, need, max_bytes)</code> (exercise 2). Worked example, cache limit 100 GB, with A (40 GB, last used t=1), B (30 GB, t=5), C (20 GB, t=2, pinned): used = 90 GB.</p>
<div class="eq">need 35: 90 + 35 = 125 > 100, oldest is A
  evict A: used 50, 50 + 35 = 85 ≤ 100
  plan [A]
need 60: evict A: 50 + 60 = 110 > 100
  evict B too: used 20, 20 + 60 = 80
  plan [A, B]
need 85: A and B gone: 20 + 85 = 105 > 100
  C is pinned: CacheFull, evict nothing</div>
<p>Note <i>plan first, then act</i>: a failed plan must evict nothing, otherwise you throw away cache and still cannot fit. One gap remains: two pods on the same node calling <code>get()</code> for the same new version at the same time. The in-process lock does not span pods. Fix it with a file lock (<code>fcntl.flock</code>) or make one prefetch DaemonSet the only writer (exercise 2's design question).</p>`,
        check: { q: "Why does the cache fetch into a temp directory and rename, instead of downloading straight into the final path?",
          options: ["Rename makes the download faster", "The rename is atomic, so a reader or a crash never leaves a partial version at the final path", "S3 requires a temp directory"], answer: 1,
          why: "Downloads take minutes. Written in place, a partial version would be visible (and maybe loaded) while it is still incomplete. Verify in the temp dir, then one atomic rename publishes it whole; a crash leaves garbage only under .tmp/." },
        scene(G) {
          const sc = 5.4, x0 = 40;
          G.text(24, 34, "cache limit 100 GB: A 40 (t=1) · B 30 (t=5) · C 20 (t=2, pinned)", { size: 12 });
          const bar = (y, parts, label) => {
            G.label(24, y - 6, label, { size: 12, color: "ink" });
            G.rect(x0, y, 100 * sc, 36, { stroke: "line", rx: 4 });
            let x = x0; parts.forEach(([g, c, l, o]) => { G.rect(x, y, g * sc, 36, Object.assign({ fill: c, rx: 3 }, o || {})); G.text(x + g * sc / 2, y + 23, l, { anchor: "middle", size: 12, color: "bg" }); x += g * sc; });
          };
          bar(76, [[40, "k", "A 40"], [30, "v", "B 30"], [20, "w", "C 20 pinned"]], "now: 90 GB used");
          bar(150, [[30, "v", "B 30"], [20, "w", "C 20"], [35, "ok", "new 35"]], "need 35: evict A → 85 GB");
          bar(224, [[20, "w", "C 20"], [60, "ok", "new 60"]], "need 60: evict A, B → 80 GB");
          G.rect(24, 290, 592, 110, { stroke: "hot", rx: 8, dash: "5 4" });
          G.text(40, 316, "need 85: C is pinned and A, B together free only 70 GB", { size: 12, color: "hot" });
          G.text(40, 340, "20 + 85 = 105 > 100 → CacheFull; plan first, evict nothing", { size: 12, color: "hot" });
          G.label(40, 368, "install = fetch to .tmp/ → verify → os.replace (atomic)", { size: 11, color: "ink" });
          G.label(40, 388, "pinned = a running replica reads it: never evicted", { size: 11, color: "ink" });
          G.caption("evict least-recently-used, never a pinned version");
        } },

      /* 10 */
      { rail: "prefetch", title: "Put it together: warm the node before the pod arrives",
        body: `<p>The cache only helps if it is warm when the pod starts. <code>platform/weights/daemonset-prefetch.yaml</code> is a <b>DaemonSet</b> (one pod per node) that calls the cache's <code>get()</code> for the models you serve, so every GPU node pulls from S3 once, in the background. Then a P3.6 scale-up on that node skips the S3 hop and reads from NVMe: 8 s, not 24 s, in the example rates.</p>
<p>Choosing a method by situation:</p>
<table><tr><th>situation</th><th>best path</th><th>why</th></tr>
<tr><td>brand-new node, empty disk, scale from zero</td><td>streamer (B)</td><td>no disk write, starts using the GPU early</td></tr>
<tr><td>node that served the model before (restart, rolling update)</td><td>local cache (A warm)</td><td>skips the network entirely</td></tr>
<tr><td>many nodes, same model</td><td>DaemonSet prefetch + cache</td><td>each node pays the S3 cost once, before it is needed</td></tr></table>
<p>The costs to watch: a streamer makes many S3 requests per load (request charges grow with concurrency and shrinking chunk size), and an unbounded <code>hostPath</code> cache fills the node's disk until the kubelet starts evicting pods. That is why the cache has a size limit.</p>
<p><b>AWS safety (aws.md, T2).</b> Cost: <code>g6.xlarge</code> on-demand for about 2 instance-hours plus gp3 storage for as long as the volume exists (look up the $/h for your region; spot is fine because every step is rerunnable). Auto-stop: an idle watchdog stops the instance after 30 idle minutes and a CloudWatch backstop stops it after 3 hours of near-zero CPU. Teardown: <code>make down</code> then <code>../scripts/cost.sh</code>, and delete the registry bucket objects, because S3 storage bills monthly. No inbound ports: servers bind to <code>127.0.0.1</code> with an API key and you reach them through SSM (<code>make ssm</code>, <code>make forward</code>). The instance role is read-only on the registry bucket.</p>`,
        scene(G) {
          const nodes = [0, 1, 2].map((i) => {
            const x = 24 + i * 200, g = G.group();
            G.rect(x, 130, 184, 130, { stroke: "k", rx: 10, parent: g }); G.text(x + 92, 154, "GPU node " + (i + 1), { anchor: "middle", size: 13, color: "k", parent: g });
            G.box(x + 12, 168, 160, 34, "prefetch DaemonSet pod", { stroke: "v", color: "v", size: 11, parent: g });
            G.box(x + 12, 212, 160, 36, "NVMe cache: model v1", { fill: "q", size: 11, parent: g });
            return g;
          });
          G.box(180, 40, 280, 48, "S3 registry (once per node)", { stroke: "ok", color: "ok", size: 13 });
          const ar = [0, 1, 2].map((i) => G.arrow(320, 90, 116 + i * 200, 128, { color: "ok", w: 2, dash: "4 3" }));
          G.rect(24, 290, 592, 112, { stroke: "line", rx: 10 });
          G.text(40, 316, "scale-up on a warmed node: read NVMe, 8 s (example rates)", { size: 13, color: "ok" });
          G.text(40, 342, "scale-up on an empty node: stream from S3, 12.8 s", { size: 13 });
          G.text(40, 368, "no cache, no stream: download then load, 24 s", { size: 13, color: "hot" });
          G.label(40, 390, "all numbers: example rates from the simulator defaults", {});
          G.from(nodes, { opacity: 0, y: 14, stagger: 0.2, duration: 0.3 }); G.from(ar, { opacity: 0, stagger: 0.2, duration: 0.3, delay: 0.5 });
          G.caption("pay the S3 hop once per node, before the pod needs it");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Weights are about 16 GB for an 8B model in bf16 and arrive through hops: S3, network, disk, page cache, host, PCIe, GPU. Done in sequence the times add; pipelined, the slowest hop sets the pace.</li>
<li>safetensors is a header with explicit offsets plus raw bytes, so it can be mmapped and read lazily; shards plus an index let many connections and workers share the load.</li>
<li>Cold reads measure the disk, warm reads measure RAM. Always say which.</li>
<li>Throughput from S3 is min(streams × per-stream rate, cap). One stream proves nothing; the plateau shows which hop is the limit.</li>
<li>Streaming skips the disk and wins on a fresh node; a warm local cache wins on repeats; a prefetch DaemonSet makes the cache warm in advance.</li>
<li>The registry is immutable, checksummed versions. The cache installs atomically, evicts least-recently-used, and never evicts a pinned version.</li></ul>`,
    sim: {
      title: "How long until the weights are on the GPU?",
      intro: "Set the model size and the rate of each hop (example values: measure yours with <code>weights.bench</code>). The three bars compare download-then-load with a cold cache, a warm local cache, and a streaming loader. The label under each bar names the hop that limits it. Press Run to watch the bytes move.",
      height: 260,
      controls: [
        { id: "gb", label: "model size (GB)", type: "select", value: 16, options: [[16, "8B bf16 · 16 GB"], [141, "70B bf16 · 141 GB"], [8, "8B 8-bit · 8 GB"]] },
        { id: "r", label: "one S3 stream (GB/s)", min: 0.02, max: 0.5, step: 0.01, value: 0.1, format: (x) => x.toFixed(2) },
        { id: "c", label: "concurrent streams", min: 1, max: 64, step: 1, value: 16 },
        { id: "net", label: "network (GB/s)", min: 0.25, max: 6, step: 0.05, value: 1.25, format: (x) => x.toFixed(2) },
        { id: "dw", label: "disk write (GB/s)", min: 0.1, max: 4, step: 0.1, value: 1.0, format: (x) => x.toFixed(1) },
        { id: "dr", label: "disk read, cold (GB/s)", min: 0.1, max: 8, step: 0.1, value: 2.0, format: (x) => x.toFixed(1) },
        { id: "pcie", label: "host to GPU (GB/s)", min: 2, max: 32, step: 1, value: 12 },
      ],
      run: { label: "Move the bytes", frames: 40, ms: 50 },
      draw(G, v, t) {
        const m = methods(v), max = Math.max(m.cold, 1), sc = 470 / max, now = t * max, x0 = 80;
        const rows = [
          ["A cold", [[m.dl, "v"], [m.ld, "q"]]],
          ["A warm", [[m.ld, "q"]]],
          ["B stream", [[m.st, "k"]]],
        ];
        G.label(10, 16, "seconds until weights are on the GPU", { size: 11 });
        G.axes(x0 - 6, 24, 550, 150, {});
        rows.forEach(([name, segs], i) => {
          const y = 40 + i * 46; let x = x0, done = 0;
          G.text(6, y + 21, name, { size: 12 });
          segs.forEach(([d, c]) => {
            G.rect(x, y, d * sc, 32, { fill: "line", rx: 3 });
            const fill = Math.max(0, Math.min(d, now - done)); G.rect(x, y, fill * sc, 32, { fill: c, rx: 3 });
            x += d * sc; done += d;
          });
          G.text(x + 6, y + 21, F.num(done, 1) + " s", { size: 12 });
        });
        G.line(x0 + now * sc, 28, x0 + now * sc, 176, { color: "hot", dash: "3 3" });
        [0, 0.5, 1].forEach((f) => G.label(x0 + f * 470, 192, F.num(f * max, 0) + " s", { size: 10, anchor: f === 1 ? "end" : f === 0 ? "start" : "middle" }));
        G.rect(x0, 214, 12, 12, { fill: "v", rx: 2 }); G.label(x0 + 16, 225, "download to disk", { size: 10 });
        G.rect(x0 + 130, 214, 12, 12, { fill: "q", rx: 2 }); G.label(x0 + 146, 225, "read back + copy to GPU", { size: 10 });
        G.rect(x0 + 290, 214, 12, 12, { fill: "k", rx: 2 }); G.label(x0 + 306, 225, "stream to GPU", { size: 10 });
        const best = Math.min(m.cold, m.warm, m.st);
        return [
          { title: "Time to weights on GPU", rows: [["A cold (download, then load)", F.num(m.cold, 1) + " s"], ["A warm (cache hit)", F.num(m.warm, 1) + " s"], ["B stream", F.num(m.st, 1) + " s"], ["streaming vs cold", F.num(m.cold / m.st, 2) + "× faster"]],
            chip: [m.st < m.cold, m.st < m.cold ? "streaming beats a cold download" : "streaming is limited by a slow hop: check below"] },
          { title: "Effective rate", rows: [["A cold", F.num(v.gb / m.cold, 2) + " GB/s"], ["A warm", F.num(v.gb / m.warm, 2) + " GB/s"], ["B stream", F.num(v.gb / m.st, 2) + " GB/s"]],
            gauge: [[Math.min(1, m.S / v.net), "k"]], gaugeText: `S3 side: ${F.num(m.S, 2)} GB/s of ${v.net} GB/s network` },
          { title: "What limits each", rows: [["A cold: download", m.hopCold], ["A: read back + copy", m.hopLoad], ["B stream", m.hopStream]],
            chip: [best === m.warm, best === m.warm ? "a warm cache is the fastest path" : "no cache yet: warm it with the prefetch DaemonSet"] },
          { title: "Model", html: `<p class="note">Each hop is a constant rate. S3 delivers min(streams × per-stream rate, network). A cold download is limited by the slower of S3 and disk write; loading overlaps disk read and the PCIe copy per tensor; streaming overlaps S3 and the PCIe copy, with no disk. Real runs add start-up time, request overhead and contention. Rates are example values.</p>` },
        ];
      },
    },
    practice: {
      intro: "Exercises 1 and 2 are T0: pure pytest on a laptop. Exercise 3 is T2/T3 on a <code>g6.xlarge</code> (infra/aws/single-node, about 2 instance-hours); exercise 4 needs a GPU. Start every cloud session with <code>make up</code> and finish with <code>make down</code> and <code>../scripts/cost.sh</code>; the idle watchdog stops the instance after 30 minutes and a CloudWatch alarm after 3 hours. There are no inbound ports.",
      items: [
        { title: "Manifest and checksum verifier", tier: "T0", goal: "Implement verify(root, manifest) in verify_impl.py: report missing, size, sha256 and unexpected files, checking size before hashing. Think: why verify once at install and not on every load?",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.7-weight-delivery/exercises/test_verifier.py" },
        { title: "LRU NVMe cache with a size limit", tier: "T0", goal: "(test_cache.py then tests the whole cache.) Implement plan_evictions in evict_impl.py: least recently used first, never pinned, raise CacheFull and evict nothing if it cannot fit. Then design a fix for two pods fetching the same version at once.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.7-weight-delivery/exercises/test_lru.py" },
        { title: "S3 throughput versus concurrency", tier: "T2/T3", goal: "Run weights.bench with concurrency 1 to 64, then check the curve: at least 2x gain at the top, the knee, and the fraction of your instance's network used. Say which hop is the plateau and prove it with a second run to /dev/shm.",
          cmd: "python course/P3-deployment-and-infra/P3.7-weight-delivery/exercises/check_s3_curve.py results/p3.7-loader.json --instance-gbps <your instance's network Gbps>" },
        { title: "Stream shards to the GPU vs download-then-load", tier: "T2 · hard", goal: "Time download-then-load (cold and warm) against the runai_streamer path and fill in the comparison table. When does the warm cache beat streaming, and what do the S3 request charges look like?",
          cmd: "vllm serve $REG/llama3-8b/<sha> --load-format runai_streamer --host 127.0.0.1 --api-key $S2S_API_KEY" },
      ],
      labs: [
        { label: "Run all T0 exercises", path: "uv run pytest course/P3-deployment-and-infra/P3.7-weight-delivery/exercises" },
        { label: "Manifest, cache, bench and the prefetch DaemonSet (#5)", path: "platform/weights/" },
        { label: "Tiny T0 model: <code>uv run python platform/engine/v0/tools/make_tiny_llama.py /tmp/tiny</code>", path: "platform/engine/v0/tools/make_tiny_llama.py" },
        { label: "Local bench: <code>PYTHONPATH=platform uv run python -m weights.bench --dir /tmp/tiny</code>", path: "platform/weights/bench.py" },
        { label: "AWS guide (cost, quota, auto-stop, teardown)", path: "course/P3-deployment-and-infra/P3.7-weight-delivery/aws.md" },
        { label: "Single-node Terraform (SSM only, no inbound ports)", path: "infra/aws/single-node/README.md" },
      ],
    },
  });
})();
