/* P3.1 — GPU containers. From "what is a container" (namespaces, cgroups, layers) to the NVIDIA toolkit and cold-start cost. */
(function () {
  const F = S2S.fmt;
  // a stacked layer bar: x, y, w, label, color
  const layer = (G, x, y, w, h, label, o = {}) => G.box(x, y, w, h, label, Object.assign({ size: 12, rx: 5 }, o));

  S2S.lesson({
    id: "p3-1", n: "P3.1", title: "GPU containers",
    subtitle: "Deployment and infra · first principles · T0 to build, T2 to run on a GPU",
    kicker: "Lesson · ≈ 40 min",
    headline: "Shipping the whole machine, except the driver",
    intro: `<p>P2 ran vLLM on one machine that you set up by hand. To run it on a fleet, every node must get exactly the same files. This lesson builds that idea from the ground up: what a container really is (a normal Linux process with a restricted view and a budget), how an image is stored as layers, why the GPU driver is the one thing a container cannot bring with it, and why image size turns into seconds of cold start.</p>`,
    facts: ["11 steps", "4 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "container / image"], ["w", "host"], ["ok", "fine"], ["hot", "problem"]],
    prev: "p2-7", next: "p3-2",
    steps: [
      { rail: "the problem", title: "A program is more than its code",
        body: `<p>Your vLLM server is a few hundred lines of your own code. To run, it also needs a Python interpreter of the right version, hundreds of Python packages (torch, vLLM, their dependencies), system libraries such as <code>glibc</code> and <code>libgomp</code>, and the CUDA libraries that torch was built against.</p>
<p>Install these by hand on a second machine and some version will differ. Sometimes it fails loudly (<code>ImportError</code>). Sometimes it runs and behaves differently. With 20 GPU nodes that come and go every hour, hand setup is not an option.</p>
<p>The fix: package the program <b>together with every file it needs</b> into one artifact, an <b>image</b>, and run that image the same way everywhere. A running image is a <b>container</b>.</p>
<div class="analogy"><b>Picture it</b>Instead of mailing a recipe and hoping every kitchen has the same ingredients, you mail the sealed meal kit.</div>`,
        scene(G) {
          G.text(24, 36, "same code, two machines", { size: 14 });
          const L = [["Python 3.12", "ok"], ["torch 2.x + CUDA 12 libs", "ok"], ["vLLM 0.31.0", "ok"], ["glibc 2.39", "ok"]];
          const R = [["Python 3.10", "hot"], ["torch missing", "hot"], ["vLLM missing", "hot"], ["glibc 2.31", "hot"]];
          G.rect(24, 52, 280, 200, { stroke: "ink", rx: 10 }); G.label(36, 74, "your laptop: works", { color: "ok" });
          G.rect(336, 52, 280, 200, { stroke: "ink", rx: 10 }); G.label(348, 74, "GPU server: ImportError", { color: "hot" });
          const a = L.map(([t, c], i) => layer(G, 40, 88 + i * 38, 248, 30, t, { stroke: c }));
          const b = R.map(([t, c], i) => layer(G, 352, 88 + i * 38, 248, 30, t, { stroke: c, color: c }));
          const img = G.box(170, 280, 300, 46, "image = app + every file it needs", { fill: "k", size: 13 });
          G.arrow(250, 330, 150, 372, { color: "k", w: 2 }); G.arrow(390, 330, 490, 372, { color: "k", w: 2 });
          G.label(96, 392, "laptop: runs the same", { color: "ok" }); G.label(420, 392, "server: runs the same", { color: "ok" });
          G.from(b, { opacity: 0, x: 20, stagger: 0.1, duration: 0.3 }); G.from(img, { opacity: 0, y: 20, delay: 0.6, duration: 0.5 });
          G.caption("ship the files with the program, not a list of instructions");
        } },

      { rail: "namespaces", title: "A container is a process with a private view",
        body: `<p>A container is <b>not</b> a virtual machine. There is no second operating system. It is an ordinary Linux process, started by the same kernel as every other process, with two extra restrictions. The first is <b>namespaces</b>.</p>
<p>A namespace gives a process its own private copy of one kind of system resource. The kernel keeps one table per namespace and shows each process only its own:</p>
<ul><li><b>mount</b>: its own filesystem tree. Its <code>/</code> is the image's files, not the host's.</li>
<li><b>PID</b>: its own process numbers. Inside, vLLM is PID 1; on the host it might be PID 4127.</li>
<li><b>network</b>: its own interfaces, IP address and ports.</li>
<li><b>UTS</b> (hostname), <b>IPC</b>, <b>user</b> (user ids) and <b>cgroup</b>.</li></ul>
<p>Because every container shares the host's kernel, starting one takes about as long as starting a process, not booting a machine. Remember the shared kernel: it decides where the GPU driver must live (step 7).</p>`,
        scene(G) {
          G.rect(24, 360, 592, 44, { fill: "w", opacity: 0.8, rx: 8 }); G.text(320, 387, "one Linux kernel (the host's)", { anchor: "middle", size: 14 });
          const C = [["container A", "PID 1 = vllm", "/ = image A files", "eth0 10.0.1.7"], ["container B", "PID 1 = mockllm", "/ = image B files", "eth0 10.0.1.8"]];
          const g = C.map(([n, a, b, c], i) => {
            const x = 24 + i * 200;
            const gg = G.group();
            G.rect(x, 60, 184, 170, { stroke: "k", rx: 10, parent: gg }); G.text(x + 12, 84, n, { color: "k", size: 13, parent: gg });
            [a, b, c].forEach((t, j) => G.text(x + 12, 118 + j * 32, t, { size: 12, parent: gg }));
            G.arrow(x + 92, 232, x + 92, 356, { color: "k", dash: "4 3" });
            return gg;
          });
          G.rect(432, 60, 184, 170, { stroke: "muted", rx: 10 }); G.text(444, 84, "host view (ps)", { color: "muted", size: 13 });
          ["PID 4127 vllm", "PID 4290 mockllm", "PID 1 systemd", "/ = host files"].forEach((t, j) => G.text(444, 118 + j * 28, t, { size: 12, color: j < 2 ? "ink" : "muted" }));
          G.label(24, 270, "same processes, two views: namespaces decide what each one can see", { color: "ink" });
          G.label(24, 300, "no guest OS, no boot: a container starts like a process", {});
          G.from(g, { opacity: 0, y: -16, stagger: 0.2, duration: 0.4 });
          G.caption("inside, vLLM thinks it is PID 1 on its own machine");
        } },

      { rail: "cgroups", title: "And a budget it cannot exceed",
        body: `<p>Namespaces limit what a process can <i>see</i>. <b>Control groups (cgroups)</b> limit what it can <i>use</i>. The kernel counts the CPU time and memory of every process in the group and enforces limits you write into small files:</p>
<div class="eq">memory.max = 8589934592     # 8 GiB hard limit
cpu.max    = 200000 100000  # 200 ms CPU per
                            # 100 ms = 2 CPUs</div>
<p>CPU over the limit is <b>throttled</b>: the process just waits. Memory cannot be throttled, so when the group tries to go past <code>memory.max</code> the kernel's out-of-memory killer ends a process with <code>SIGKILL</code>. You see exit code <b>137</b> (= 128 + signal 9) and, in Kubernetes, the reason <code>OOMKilled</code>.</p>
<p>Note what cgroups do <i>not</i> count: GPU memory. That lives on the GPU, managed by the NVIDIA driver, which is why Kubernetes hands out whole GPUs (P3.3) instead of GPU megabytes.</p>`,
        scene(G) {
          G.axes(40, 60, 560, 250, { xlabel: "time →", ylabel: "container memory" });
          const lim = 110; G.line(40, lim, 600, lim, { color: "hot", dash: "6 4", w: 2 }); G.label(60, lim - 8, "memory.max = 8 GiB", { color: "hot" });
          const pts = [[40, 300], [120, 270], [200, 240], [280, 200], [360, 170], [440, 140], [500, 112]];
          G.path("M " + pts.map((p) => p.join(" ")).join(" L "), { color: "k", w: 3 });
          const x = G.circle(500, 112, 12, { stroke: "hot", sw: 3 }); G.text(518, 150, "OOM kill", { color: "hot", size: 14 });
          G.text(518, 170, "exit 137", { color: "hot", size: 13 });
          G.label(40, 350, "CPU over cpu.max → throttled (waits)", { color: "ink" });
          G.label(40, 374, "memory over memory.max → killed (cannot wait for RAM)", { color: "ink" });
          G.label(40, 398, "GPU memory is not counted by cgroups", {});
          G.pulse(x, { repeat: 6 });
          G.caption("a container is a process with a view (namespaces) and a budget (cgroups)");
        } },

      { rail: "layers", title: "An image is a stack of read-only layers",
        body: `<p>Where does the container's private <code>/</code> come from? From the <b>image</b>: a stack of <b>layers</b>. Each layer is a tar archive of the files one build step added or changed. A Dockerfile instruction that changes files (<code>RUN</code>, <code>COPY</code>) makes one layer; instructions such as <code>ENV</code> and <code>USER</code> only change the image's settings.</p>
<p>At run time, an <b>overlay filesystem</b> stacks the layers so the process sees one merged tree: when two layers contain the same path, the upper one wins. On top sits one thin <b>writable layer</b> owned by this container. Writing to a file copies it up into the writable layer first (<b>copy-on-write</b>); the image layers are never modified, and the writable layer is thrown away when the container is removed.</p>
<p>Each layer is named by the <b>SHA-256 hash</b> of its contents. Two images that start <code>FROM python:3.12-slim</code> share that base layer on disk and in the registry, and a node that already has it does not download it again.</p>`,
        scene(G) {
          const Ls = [["FROM python:3.12-slim  (base OS + Python)", "sha256:9a1f…"], ["RUN apt-get install libgomp1 curl", "sha256:4c07…"], ["COPY --from=build /opt/venv", "sha256:e2b8…"]];
          const g = Ls.map(([t, h], i) => { const y = 300 - i * 62; const b = layer(G, 24, y, 400, 48, t, { fill: "k", size: 12 }); G.label(436, y + 29, h + "  read-only", {}); return b; });
          const wr = G.box(24, 300 - 3 * 62, 400, 48, "writable layer (this container only)", { stroke: "v", dash: "6 4", color: "v", size: 12 });
          G.label(24, 384, "overlay: upper layers win · writes copy the file up first", { color: "ink" });
          G.label(24, 406, "ENV / USER / EXPOSE change settings only, no layer", {});
          G.label(24, 96, "what the process sees as / = all layers merged", { color: "ink" });
          G.from(g, { opacity: 0, y: 30, stagger: 0.25, duration: 0.4 }); G.from(wr, { opacity: 0, delay: 0.9, duration: 0.4 });
          G.caption("layers stack bottom-up; only the top one is writable");
        } },

      { rail: "append-only", title: "Layers only ever add",
        body: `<p>A layer records a change; it is never edited afterwards. Two practical consequences follow.</p>
<p><b>Deleting does not remove.</b> If step 3 copies an HF token into the image and step 4 runs <code>rm /token</code>, step 4's layer just records "this path is deleted" (a <i>whiteout</i> marker). The token is still inside layer 3, and anyone who can pull the image can extract it. Secrets go in at build time with <code>RUN --mount=type=secret</code> (never written to a layer) or at run time as environment variables or files.</p>
<p><b>The build cache works top-down.</b> Docker reuses a cached layer only if the instruction and everything below it are unchanged. Change one line and every layer above it is rebuilt. So put what rarely changes (dependencies) low, and what changes often (your code) high:</p>
<div class="eq">COPY requirements.txt .      # changes rarely
RUN  pip install -r requirements.txt
COPY . .                     # changes every commit</div>
<p>With <code>COPY . .</code> first, every code edit reinstalls every package.</p>`,
        check: { q: "A Dockerfile copies <code>secret.txt</code> in one step and runs <code>rm secret.txt</code> in the next. Can someone with the image read the secret?",
          options: ["No, the final filesystem does not contain it", "Yes, it is still stored in the earlier layer", "Only if the container is running as root"], answer: 1,
          why: "Each layer is an immutable tar of changes. The rm adds a whiteout in a new layer that hides the path in the merged view, but the earlier layer still contains the file and can be pulled and unpacked on its own." },
        scene(G) {
          const Ls = [["1  FROM python:3.12-slim", "k"], ["2  COPY requirements.txt + pip install", "k"], ["3  COPY secret.txt", "hot"], ["4  RUN rm secret.txt  (whiteout)", "muted"]];
          const g = Ls.map(([t, c], i) => layer(G, 24, 330 - i * 56, 340, 44, t, c === "muted" ? { stroke: "muted", color: "muted" } : { fill: c }));
          G.arrow(380, 218, 470, 218, { color: "hot", w: 2 }); G.text(480, 214, "still in", { color: "hot", size: 13 }); G.text(480, 232, "layer 3", { color: "hot", size: 13 });
          G.text(380, 162, "merged view: file hidden", { color: "muted", size: 12 });
          G.label(24, 80, "cache: a change in layer N rebuilds N, N+1, …", { color: "ink" });
          G.label(24, 104, "→ dependencies low, your code high", { color: "ok" });
          G.from(g, { opacity: 0, y: 20, stagger: 0.2, duration: 0.35 }); G.pulse(g[2], { repeat: 6, lo: 0.4 });
          G.caption("a later rm hides a file; it never removes it from the image");
        } },

      { rail: "tag vs digest", title: "Tags move, digests do not",
        body: `<p>An image has a <b>manifest</b>: a small JSON document that lists its layers' hashes and its settings. The SHA-256 of the manifest is the image's <b>digest</b>. Change one byte in any layer and the digest changes. A digest therefore names exactly one image, forever.</p>
<p>A <b>tag</b> such as <code>python:3.12-slim</code> is just a pointer that the publisher can move. When Debian ships a security fix next week, <code>3.12-slim</code> points to a new digest, and the "same" Dockerfile builds a different image.</p>
<div class="eq">FROM python:3.12-slim                    # moves
FROM python:3.12-slim@sha256:&lt;64 hex&gt;   # fixed</div>
<p>Pin by digest so a build is reproducible and auditable, and bump the digest deliberately (a pull request you can review and roll back). This course's policy linter flags tag-only bases in <code>--strict</code> mode: that is exercise 1.</p>`,
        check: { q: "Why pin <code>FROM</code> by digest rather than by a specific tag like <code>3.12-slim</code>?",
          options: ["Digests download faster", "A tag can be moved to different content; a digest is the hash of the content itself", "Tags are not allowed in multi-stage builds"], answer: 1,
          why: "A tag is a mutable name in the registry. A digest is computed from the manifest, which lists every layer's hash, so the same digest always means byte-for-byte the same image." },
        scene(G) {
          G.box(24, 70, 200, 44, "python:3.12-slim", { stroke: "v", color: "v" }); G.label(24, 60, "tag (a movable pointer)", {});
          const d1 = G.box(330, 50, 286, 40, "sha256:1b7e…  (week 1)", { stroke: "muted", color: "muted", size: 12 });
          const d2 = G.box(330, 110, 286, 40, "sha256:c4a9…  (week 2)", { fill: "k", size: 12 });
          G.line(226, 92, 326, 70, { color: "muted", dash: "4 4" }); const a = G.arrow(226, 92, 326, 130, { color: "v", w: 2 });
          G.text(24, 200, "manifest → digest", { size: 14 });
          G.rect(24, 216, 380, 120, { stroke: "line", rx: 8 });
          [[0, '{ "config": "sha256:77d0…",'], [16, '"layers": [ "sha256:9a1f…",'], [117, '"sha256:4c07…",'], [117, '"sha256:e2b8…" ] }']].forEach(([dx, t], i) => G.text(36 + dx, 244 + i * 24, t, { size: 12 }));
          G.arrow(408, 276, 470, 276, { color: "ok", w: 2 }); G.text(478, 270, "sha256(manifest)", { color: "ok", size: 12 }); G.text(478, 290, "= the digest", { color: "ok", size: 12 });
          G.label(24, 370, "one byte changes in any layer → different digest", { color: "ink" });
          G.from(a, { opacity: 0, duration: 0.6, delay: 0.3 }); G.from(d2, { opacity: 0, x: 20, delay: 0.3, duration: 0.5 });
          G.caption("the tag moved overnight; the digest you pinned did not");
        } },

      { rail: "the driver", title: "The one thing a container cannot bring: the driver",
        body: `<p>Now add a GPU. Software reaches the GPU through a stack:</p>
<ol><li><b>Kernel driver</b> (<code>nvidia.ko</code>): a kernel module that owns the hardware.</li>
<li><b>User-space driver</b> (<code>libcuda.so</code>, plus <code>libnvidia-ml.so</code> for <code>nvidia-smi</code>): talks to the kernel module. It must match the kernel module's version exactly.</li>
<li><b>CUDA runtime and libraries</b> (<code>libcudart</code>, cuBLAS, cuDNN, NCCL): what torch and vLLM link against.</li>
<li>torch, vLLM, your code.</li></ol>
<p>Step 2 said every container shares the host's kernel. So the kernel module is the host's, and <code>libcuda.so</code> must be the host's too (it has to match). Layers 1–2 come from the host; layers 3–4 ship in the image. <b>Never install a driver inside an image.</b> The vLLM wheel already bundles the CUDA runtime libraries it needs, which is why <code>env/Dockerfile.serving</code> can start from plain <code>python:3.12-slim</code>.</p>`,
        scene(G) {
          const S = [["your app · vLLM · torch", "k"], ["CUDA runtime: libcudart, cuBLAS, NCCL", "k"], ["libcuda.so  (user-space driver)", "w"], ["nvidia.ko  (kernel driver)", "w"], ["GPU", "muted"]];
          const g = S.map(([t, c], i) => layer(G, 140, 50 + i * 66, 360, 50, t, { fill: c, size: 13 }));
          G.line(24, 50 + 2 * 66 - 8, 616, 50 + 2 * 66 - 8, { color: "v", dash: "6 4", w: 2 });
          G.text(512, 50 + 2 * 66 - 14, "container boundary", { color: "v", size: 11 });
          G.label(512, 80, "in the image", { color: "k" }); G.label(512, 100, "(you control)", {});
          G.label(512, 222, "from the host", { color: "ink" }); G.label(512, 242, "(must match", {}); G.label(512, 258, "each other)", {});
          G.from(g, { opacity: 0, x: -20, stagger: 0.12, duration: 0.35 });
          G.caption("the kernel is shared, so the driver is the host's");
        } },

      { rail: "compatibility", title: "Which image runs on which driver?",
        body: `<p>Since the driver comes from the host and the CUDA runtime from the image, they can disagree. Each driver release supports CUDA runtimes <b>up to</b> a certain CUDA version. Older runtimes keep working on newer drivers. A runtime <b>newer</b> than the driver supports may fail at start with an error like "CUDA driver version is insufficient for CUDA runtime version". NVIDIA documents partial exceptions (minor-version compatibility within one major version, and a forward-compatibility package for data-centre GPUs); check its CUDA compatibility guide for your exact versions. <i>UNVERIFIED here: the build environment could not reach docs.nvidia.com.</i></p>
<p>Two commands tell you each side:</p>
<div class="eq">nvidia-smi
  # "CUDA Version: X.Y" = the HOST DRIVER's max
python -c "import torch; print(torch.version.cuda)"
  # the CUDA runtime the IMAGE was built with</div>
<p><code>examples/01_check_cuda_compat.sh</code> prints both, side by side, for an image on a GPU host.</p>`,
        check: { q: "Inside a container, <code>nvidia-smi</code> reports \"CUDA Version: 12.4\". What does that number tell you?",
          options: ["The CUDA toolkit installed in the image", "The highest CUDA version the host's driver supports", "The CUDA version torch was compiled with"], answer: 1,
          why: "nvidia-smi is the host's binary talking to the host's driver (both injected at start). It reports the driver's maximum supported CUDA version, not anything inside the image; torch.version.cuda tells you the image's side." },
        scene(G) {
          const v = ["11.8", "12.1", "12.4", "12.6", "12.8"], x0 = 60, dx = 130;
          G.line(40, 200, 600, 200, { color: "line", w: 2 });
          v.forEach((s, i) => { G.line(x0 + i * dx, 192, x0 + i * dx, 208, { color: "line", w: 2 }); G.text(x0 + i * dx, 232, s, { anchor: "middle", size: 13 }); });
          G.text(40, 260, "CUDA version (example values)", { size: 11, color: "muted" });
          const drv = x0 + 2 * dx; G.line(drv, 110, drv, 200, { color: "v", w: 3 }); G.box(drv - 90, 70, 180, 36, "driver max = 12.4", { stroke: "v", color: "v", size: 12 });
          const imgs = [[0, "ok", "image A"], [1, "ok", "image B"], [4, "hot", "image C"]];
          const g = imgs.map(([i, c, n]) => { G.circle(x0 + i * dx, 200, 9, { fill: c }); return G.box(x0 + i * dx - 50, 290, 100, 34, n, { fill: c, size: 12 }); });
          G.label(40, 360, "A, B: runtime ≤ driver's version → runs", { color: "ok" });
          G.label(40, 384, "C: runtime newer than the driver → may fail at start", { color: "hot" });
          G.label(40, 408, "(exceptions exist: check NVIDIA's compatibility guide)", {});
          G.from(g, { opacity: 0, y: 20, stagger: 0.2, duration: 0.35 });
          G.caption("older runtime on newer driver: fine; the reverse: not guaranteed");
        } },

      { rail: "toolkit", title: "What --gpus all actually does",
        body: `<p>A plain container cannot see the GPU: its mount namespace has no <code>/dev/nvidia*</code> device files, and no <code>libcuda.so</code>. The <b>NVIDIA Container Toolkit</b> fixes that when the container starts. With <code>docker run --gpus all</code> (or <code>NVIDIA_VISIBLE_DEVICES=all</code>), it:</p>
<ol><li>adds the device nodes <code>/dev/nvidia0</code> (one per GPU), <code>/dev/nvidiactl</code> and <code>/dev/nvidia-uvm</code>, and lets the container's cgroup open them;</li>
<li>bind-mounts the host's driver libraries (<code>libcuda.so</code>, <code>libnvidia-ml.so</code>, …) and binaries such as <code>nvidia-smi</code> into the container;</li>
<li>exposes only the GPUs you asked for (<code>--gpus '"device=0"'</code> gives one).</li></ol>
<p>Newer setups describe the same injection with <b>CDI</b> (Container Device Interface), a spec file generated by <code>nvidia-ctk cdi generate</code> that any runtime can read. In Kubernetes the GPU Operator installs and configures this toolkit on every node (P3.3).</p>`,
        scene(G) {
          G.rect(24, 50, 250, 300, { fill: "w", opacity: 0.3, rx: 10 }); G.text(36, 74, "host", { size: 14 });
          G.rect(366, 50, 250, 300, { stroke: "k", rx: 10 }); G.text(378, 74, "container", { color: "k", size: 14 });
          const items = ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm", "libcuda.so", "libnvidia-ml.so", "nvidia-smi"];
          const arr = [];
          items.forEach((t, i) => {
            const y = 96 + i * 40;
            G.box(36, y, 200, 30, t, { fill: i < 3 ? "v" : "w", size: 12 });
            arr.push(G.arrow(240, y + 15, 374, y + 15, { color: "ok", dash: "4 3" }));
            arr.push(G.box(380, y, 200, 30, t, { stroke: i < 3 ? "v" : "ink", color: i < 3 ? "v" : "ink", size: 12 }));
          });
          G.label(24, 380, "docker run --gpus all  →  toolkit hook / CDI spec", { color: "ink", size: 13 });
          G.label(24, 404, "device files + host driver libs appear inside at start", {});
          G.from(arr, { opacity: 0, stagger: 0.07, duration: 0.25 });
          G.caption("the image never contained these; they are lent by the host");
        } },

      { rail: "slim images", title: "A good serving image: built in two stages",
        body: `<p>Building Python packages needs compilers, headers and download caches. Running them does not. A <b>multi-stage build</b> uses one throwaway stage to build a virtual environment and copies only the result into a clean final stage. This is exactly <code>env/Dockerfile.serving</code>:</p>
<div class="eq">FROM python:3.12-slim AS build
RUN uv venv /opt/venv &amp;&amp; uv pip install \\
      --no-cache -p /opt/venv -r /tmp/req.txt

FROM python:3.12-slim
COPY --from=build /opt/venv /opt/venv
USER 10001
ENTRYPOINT ["vllm", "serve"]</div>
<p>The other rules from the module, each with its reason: run as a <b>non-root</b> user with a read-only root filesystem (a hijacked server cannot rewrite itself); <b>no weights in the image</b> (mount them at <code>/models</code> or stream them at run time, P3.7); a <b>long start period</b> on health checks (<code>--start-period=600s</code>: vLLM spends minutes loading weights); one process per container.</p>`,
        scene(G) {
          G.rect(24, 50, 270, 300, { stroke: "muted", dash: "6 4", rx: 10 }); G.text(36, 74, "stage 1: build (thrown away)", { color: "muted", size: 12 });
          [["python:3.12-slim", "w"], ["uv, pip caches", "muted"], ["build tools, headers", "muted"], ["/opt/venv", "k"]].forEach(([t, c], i) => layer(G, 40, 300 - i * 52, 238, 40, t, c === "muted" ? { stroke: "muted", color: "muted" } : { fill: c }));
          G.rect(346, 50, 270, 300, { stroke: "k", rx: 10 }); G.text(358, 74, "stage 2: what ships", { color: "k", size: 12 });
          const v = layer(G, 362, 248, 238, 40, "/opt/venv", { fill: "k" });
          layer(G, 362, 300, 238, 40, "python:3.12-slim", { fill: "w" });
          layer(G, 362, 196, 238, 40, "USER 10001 · no weights", { stroke: "ok", color: "ok" });
          const a = G.path("M 280 164 C 330 164, 320 268, 358 268", { color: "k", w: 2, arrow: true }); G.label(362, 120, "COPY --from=build /opt/venv", { color: "k", size: 11 });
          G.label(24, 384, "models mounted at run time: -v /opt/models:/models:ro", { color: "ink" });
          G.label(24, 408, "fewer files → smaller pulls and fewer CVEs to patch", {});
          G.from([v, a], { opacity: 0, x: -30, duration: 0.6 });
          G.caption("only the venv crosses into the final image");
        } },

      { rail: "size = cold start", title: "Image size is cold-start time",
        body: `<p>A new GPU node has none of your layers. Before Python even starts, it must <b>download</b> every compressed layer and <b>extract</b> it to disk. Each costs time proportional to the bytes.</p>
<p>The module's worked example (example rates, measure your own): a 9 GB compressed image, a node that downloads at 400 MB/s and decompresses 200 MB/s of compressed input.</p>
<div class="eq">download  9,000 MB ÷ 400 MB/s = 22.5 s
extract   9,000 MB ÷ 200 MB/s = 45.0 s
total, one after the other    ≈ 67.5 s</div>
<p>If the two overlap (extract layer 1 while layer 2 downloads) the best case is the slower of the two, 45 s. Either way it is paid on <b>every new node</b>, and only then do weight loading (P3.7) and CUDA graph capture (P2.6) begin. Bake 16 GB of Llama-3-8B weights into the image and the same arithmetic adds about 120 s more at those rates (16,000 ÷ 400 + 16,000 ÷ 200), on every pull, for every model update. P3.6 measures the whole cold start on a fresh node.</p>`,
        check: { q: "A 4 GB compressed image, download 400 MB/s, extract 200 MB/s, no overlap. Roughly how long until the container can start?",
          options: ["10 s", "20 s", "30 s", "60 s"], answer: 2,
          why: "4,000 ÷ 400 = 10 s to download plus 4,000 ÷ 200 = 20 s to extract: 30 s in total, before any weights are loaded." },
        scene(G) {
          const sc = 3.1, y = 100;
          G.text(24, 50, "9 GB image on a fresh node (example rates, to scale)", { size: 13 });
          const d = G.rect(24, y, 22.5 * sc, 40, { fill: "k", rx: 4 });
          const e = G.rect(24 + 22.5 * sc, y, 45 * sc, 40, { fill: "v", rx: 4 });
          G.label(24, y + 60, "download 22.5 s", { color: "k" }); G.label(24 + 22.5 * sc + 8, y + 60 + 16, "extract 45 s", { color: "v" });
          G.text(24 + 67.5 * sc + 8, y + 26, "= 67.5 s", { size: 13 });
          G.text(24, 230, "same image + 16 GB of weights baked in", { size: 13, color: "hot" });
          G.rect(24, 250, 22.5 * sc, 40, { fill: "k", rx: 4 }); G.rect(24 + 22.5 * sc, 250, 45 * sc, 40, { fill: "v", rx: 4 });
          const extra = G.rect(24 + 67.5 * sc, 250, 120 * sc, 40, { fill: "hot", rx: 4 });
          G.text(24 + 67.5 * sc + 8, 275, "+40 s + 80 s = +120 s", { size: 12, color: "bg" });
          G.text(24, 316, "= 187.5 s", { size: 13 });
          G.label(24, 360, "then: weight load (P3.7) → CUDA graphs (P2.6) → ready", { color: "ink" });
          G.label(24, 384, "paid again on every new node and every image change", {});
          G.from([d, e], { attr: { width: 0 }, stagger: 0.4, duration: 0.6 }); G.from(extra, { attr: { width: 0 }, delay: 1, duration: 0.6 });
          G.caption("bytes in the image become seconds before the first token");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A container is a normal process with namespaces (a private view) and cgroups (a resource budget), sharing the host kernel.</li>
<li>An image is a stack of content-addressed, read-only layers merged by an overlay filesystem. Layers only add, so deleted secrets survive and layer order decides cache reuse.</li>
<li>Tags move; digests are hashes of content. Pin by digest.</li>
<li>The GPU driver (kernel module and <code>libcuda.so</code>) comes from the host, injected by the NVIDIA Container Toolkit; the CUDA runtime ships in the image and must not be newer than the driver supports.</li>
<li>Slim, multi-stage, non-root images without weights pull faster, and pull time is part of every cold start.</li></ul>`,
    sim: {
      title: "How long before the container starts?",
      intro: "Set the image size and the node's download and extract rates (example values: measure your own with <code>time docker pull</code> on a fresh node). Layers already cached on the node are skipped. Press Run to watch a fresh node pull the image.",
      controls: [
        { id: "img", label: "image size, compressed (GB)", min: 0.5, max: 20, step: 0.5, value: 9, format: (v) => v.toFixed(1) },
        { id: "net", label: "download rate, MB/s (example value)", min: 50, max: 2000, step: 25, value: 400 },
        { id: "ext", label: "extract rate, MB/s of compressed input (example value)", min: 50, max: 1000, step: 25, value: 200 },
        { id: "cache", label: "share of layers already on the node (%)", min: 0, max: 95, step: 5, value: 0 },
        { id: "wts", label: "weights baked into the image", type: "select", value: 0, options: [[0, "none"], [16, "+16 GB (8B bf16)"], [141, "+141 GB (70B)"]] },
        { id: "mode", label: "download and extract", type: "select", value: "seq", options: [["seq", "one after another"], ["pipe", "overlapped"]] },
      ],
      run: { label: "Pull on a fresh node", frames: 40, ms: 50 },
      draw(G, v, t) {
        const MB = (v.img + v.wts) * 1000 * (1 - v.cache / 100);
        const d = MB / v.net, e = MB / v.ext, tot = v.mode === "seq" ? d + e : Math.max(d, e);
        const base = (v.img * 1000) / v.net + (v.img * 1000) / v.ext;
        const max = Math.max(60, Math.ceil(tot / 60) * 60), sc = 560 / max, now = t * tot;
        G.label(10, 18, "time on a fresh node, seconds", { size: 11 });
        G.axes(40, 30, 580, 170);
        const y1 = 60, y2 = 120;
        G.text(44, y1 - 8, "download", { size: 11, color: "muted" }); G.text(44, y2 - 8, "extract", { size: 11, color: "muted" });
        const ex0 = v.mode === "seq" ? d : Math.max(0, e > d ? 0 : d - e);
        G.rect(44, y1, d * sc, 34, { fill: "line", rx: 3 }); G.rect(44, y1, Math.min(d, now) * sc, 34, { fill: "k", rx: 3 });
        G.rect(44 + ex0 * sc, y2, e * sc, 34, { fill: "line", rx: 3 }); G.rect(44 + ex0 * sc, y2, Math.max(0, Math.min(e, now - ex0)) * sc, 34, { fill: "v", rx: 3 });
        G.line(44 + now * sc, 34, 44 + now * sc, 200, { color: "hot", dash: "3 3" });
        [0, 0.25, 0.5, 0.75, 1].forEach((f) => G.text(44 + f * 560, 218, (f * max).toFixed(0) + " s", { size: 10, color: "muted", anchor: f === 1 ? "end" : "middle" }));
        G.text(44, 248, t >= 1 ? `container can start after ≈ ${tot.toFixed(1)} s` : `t = ${now.toFixed(1)} s`, { size: 13 });
        return [
          { title: "Pull time", rows: [["bytes to fetch", F.num(MB / 1000, 2) + " GB"], ["download", F.num(d, 1) + " s"], ["extract", F.num(e, 1) + " s"], ["total", F.num(tot, 1) + " s"]],
            chip: [tot < 60, tot < 60 ? "under a minute" : "over a minute per new node"] },
          { title: "Cost of baked weights", rows: [["image only, sequential", F.num(base, 1) + " s"], ["weights added", v.wts ? v.wts + " GB" : "none"]],
            html: `<p class="note">Weights still have to be loaded either way; keeping them out of the image lets you cache and stream them separately (P3.7) and update models without rebuilding.</p>` },
          { title: "Model", html: `<p class="note">Assumes a constant download and extract rate and that safetensors weights barely compress. Real pulls fetch a few layers in parallel and extract them in order, so the truth lies between the two modes.</p>` },
        ];
      },
    },
    practice: {
      intro: "Run these from the repository root. Exercise 1 is a pytest over the policy linter; 2–4 need Docker on your laptop (T0) and record evidence for a checker.",
      items: [
        { title: "Pin every base image by digest", tier: "T0 · easy", goal: "Rewrite each FROM in env/Dockerfile.* as image:tag@sha256:… so the strict policy passes.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.1-gpu-containers/exercises/test_policy.py" },
        { title: "Shrink the serving image by 30%", tier: "T0 · medium", goal: "Find the biggest layers with docker history, cut them, and record before/after sizes in results/p31.json.",
          cmd: "uv run python course/P3-deployment-and-infra/P3.1-gpu-containers/exercises/check_p31.py results/p31.json" },
        { title: "Non-root and read-only root filesystem", tier: "T0 · medium", goal: "Serve from the platform image with no writable root, no capabilities and a non-root user; then write the matching K8s securityContext.",
          cmd: "docker run --rm --read-only --tmpfs /tmp --user 10001 --cap-drop ALL --security-opt no-new-privileges -p 8001:8001 s2s-platform:dev" },
        { title: "Reproducible build", tier: "T0 · hard", goal: "Build env/Dockerfile.platform twice from a clean cache with SOURCE_DATE_EPOCH and get the same digest; write down every source of nondeterminism you fixed.",
          cmd: "SOURCE_DATE_EPOCH=$(git log -1 --format=%ct) docker buildx build --build-arg SOURCE_DATE_EPOCH --output type=image,rewrite-timestamp=true -f env/Dockerfile.platform ." },
      ],
      labs: [
        { label: "Dockerfile policy linter (rules P1–P5)", path: "course/P3-deployment-and-infra/P3.1-gpu-containers/examples/02_dockerfile_policy.py" },
        { label: "Host driver vs image CUDA runtime table (T2, GPU host)", path: "course/P3-deployment-and-infra/P3.1-gpu-containers/examples/01_check_cuda_compat.sh" },
        { label: "The course images: cpu, cuda-dev, platform, serving", path: "env/Dockerfile.*" },
        { label: "Build and run the mock server image: <code>docker build -f env/Dockerfile.platform -t s2s-platform:dev .</code>", path: "env/Dockerfile.platform" },
      ],
    },
  });
})();
