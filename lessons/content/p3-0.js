/* P3.0 — Containers and Kubernetes from zero. Images, layers, Dockerfiles, isolation, the desired-state loop, Pods, Deployments, Services. */
(function () {
  const F = S2S.fmt;
  const MODULE = "course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics";
  // labelled box that also accepts "\n" for several lines of text
  const mb = (G, x, y, w, h, label, o = {}) => {
    const g = G.box(x, y, w, h, "", o), lines = String(label).split("\n"), size = o.size ?? 13, lh = size * 1.3;
    lines.forEach((l, i) => G.text(x + w / 2, y + h / 2 - ((lines.length - 1) * lh) / 2 + i * lh + size * 0.36, l, { anchor: "middle", size, color: o.color ?? (o.fill ? "bg" : "ink"), weight: o.weight, parent: g }));
    return g;
  };

  // simulator data: Dockerfile in the good order and in the "copy code first" order. Sizes are EXAMPLES.
  const GOOD = [
    ["FROM python:3.12-slim", 45, "base"], ["RUN apt-get install ...", 80, "apt"], ["WORKDIR /app", 0, ""], ["COPY pyproject.toml uv.lock ./", 1, "deps"],
    ["RUN uv sync --frozen", 900, ""], ["COPY . .", 5, "code"], ["USER 10001", 0, ""], ["CMD [\"python\", \"-m\", \"app\"]", 0, ""],
  ];
  const BAD = [
    ["FROM python:3.12-slim", 45, "base"], ["RUN apt-get install ...", 80, "apt"], ["WORKDIR /app", 0, ""], ["COPY . .", 5, "code"],
    ["COPY pyproject.toml uv.lock ./", 1, "deps"], ["RUN uv sync --frozen", 900, ""], ["USER 10001", 0, ""], ["CMD [\"python\", \"-m\", \"app\"]", 0, ""],
  ];

  S2S.lesson({
    id: "p3-0", n: "P3.0", title: "Containers and Kubernetes from zero",
    subtitle: "Deployment and infra · first principles · T0, Docker and kind on a laptop",
    kicker: "Lesson · ≈ 30 min",
    headline: "Say what you want; let a loop make it true",
    intro: `<p><b>Already know Docker and the basics of Kubernetes (Pods, Deployments, Services, kubectl apply)? Answer the checkpoints. If you get them all right, skip to P3.1.</b></p>
<p>P3.1 puts vLLM in a container, and P3.3 runs it on Kubernetes with GPUs. Both assume you know what a layer is and what a Deployment does. This lesson teaches only that, from the problem up, with no GPU and no cloud. The first half packages one program into an image. The second half asks what to do when one machine is not enough.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "3 exercises"],
    legend: [["k", "image / container"], ["ok", "cached / ready"], ["hot", "rebuilt / problem"], ["v", "Kubernetes"], ["w", "host"]],
    prev: "p2-7", next: "p3-1",
    steps: [
      { rail: "my machine", title: "'It works on my machine'",
        body: `<p>A program is more than its code. It needs an interpreter of the right version, libraries, and system packages. Two machines differ in some of those, and the same code behaves differently, or fails to start.</p>
<p>The fix is to ship the program together with every file it needs, as one package, and run that package the same way everywhere. That package is an <b>image</b>. A running image is a <b>container</b>.</p>
<div class="analogy"><b>Picture it</b>Instead of mailing a recipe and hoping every kitchen has the same ingredients, you mail a sealed meal kit.</div>`,
        scene(G) {
          G.text(24, 36, "same code, two machines", { size: 14 });
          G.rect(24, 52, 280, 150, { stroke: "ok", rx: 10 }); G.label(36, 74, "laptop: works", { color: "ok" });
          G.rect(336, 52, 280, 150, { stroke: "hot", rx: 10 }); G.label(348, 74, "server: crashes", { color: "hot" });
          [["app code", "k"], ["Python 3.12", "ok"], ["libs v2", "ok"]].forEach(([t, c], i) => mb(G, 40, 86 + i * 36, 248, 28, t, { stroke: c, color: c, size: 12 }));
          [["app code", "k"], ["Python 3.9", "hot"], ["libs v1", "hot"]].forEach(([t, c], i) => mb(G, 352, 86 + i * 36, 248, 28, t, { stroke: c, color: c, size: 12 }));
          const img = mb(G, 140, 250, 360, 56, "image = app + Python + libs\n(everything but the kernel)", { fill: "k", size: 13 });
          G.arrow(170, 308, 170, 360, { color: "k", w: 2 }); G.arrow(470, 308, 470, 360, { color: "k", w: 2 });
          mb(G, 90, 362, 160, 40, "container: same", { stroke: "ok", color: "ok", size: 12 }); mb(G, 390, 362, 160, 40, "container: same", { stroke: "ok", color: "ok", size: 12 });
          G.from(img, { opacity: 0, y: -14, duration: 0.5 });
          G.caption("ship the user space with the program");
        } },

      { rail: "layers", title: "An image is a stack of read-only layers",
        body: `<p>An image is not one big file. It is a stack of <b>layers</b>. Each layer records a set of file changes: "add these files", "install these packages". The layers are read-only.</p>
<p>A <b>container</b> is a running process that sees all the layers stacked together, plus one thin <b>writable layer</b> on top for anything it writes. Remove the container and the writable layer disappears. The image is untouched.</p>
<p>Layers are shared. Start ten containers from one image and the layers are stored once, with ten small writable layers on top.</p>
<div class="eq">docker pull python:3.12-slim     # fetch layers
docker history python:3.12-slim  # list layers + sizes
docker run --rm python:3.12-slim \
  python -c "print(1)"           # start a container</div>`,
        scene(G) {
          const layers = [["base OS files", "w"], ["Python 3.12", "w"], ["your packages", "blue"], ["your code", "k"]];
          const shared = layers.map(([t, c], i) => mb(G, 60, 330 - i * 52, 220, 44, t + "  (read-only)", { fill: c, size: 12 }));
          G.label(60, 350 + 36, "image layers: stored once", { size: 12 });
          const wa = mb(G, 360, 160, 110, 40, "writable A", { stroke: "hot", color: "hot", size: 12 });
          const wb = mb(G, 490, 160, 110, 40, "writable B", { stroke: "hot", color: "hot", size: 12 });
          G.rect(360, 212, 240, 130, { stroke: "w", rx: 6, dash: "5 4" }); G.label(372, 236, "the same layers,", { size: 12 }); G.label(372, 254, "seen by both", { size: 12 });
          G.arrow(280, 290, 360, 290, { color: "muted" });
          G.text(360, 130, "container A", { size: 12, color: "hot" }); G.text(490, 130, "container B", { size: 12, color: "hot" });
          G.from(shared, { opacity: 0, x: -20, stagger: 0.12, duration: 0.3 }); G.from([wa, wb], { opacity: 0, y: -12, delay: 0.5, stagger: 0.15, duration: 0.3 });
          G.caption("many containers, one copy of the layers");
        } },

      { rail: "Dockerfile", title: "A Dockerfile is the recipe, and order matters",
        body: `<p>A <b>Dockerfile</b> lists the steps that build an image:</p>
<div class="eq">FROM python:3.12-slim      base layers
WORKDIR /app
COPY requirements.txt .   one layer
RUN pip install -r requirements.txt
COPY . .                  your code
CMD ["python", "-m", "app"]</div>
<p>Each <code>RUN</code> and <code>COPY</code> makes a layer. Docker <b>caches</b> a layer when its instruction and inputs are unchanged. The key rule: <b>when one layer changes, every layer after it is rebuilt.</b> So put what changes rarely first, and your own code last.</p>
<p>If <code>COPY . .</code> comes before <code>pip install</code>, any edit to any file reinstalls every package. Copy only the dependency list first, install, then copy the code.</p>`,
        check: { q: "You change one line of application code. Which Dockerfile order rebuilds less?",
          options: ["COPY . . then pip install", "COPY requirements.txt, pip install, then COPY . .", "They are the same"], answer: 1,
          why: "With the dependency list copied and installed first, those layers are unchanged and cached. Only the last COPY layer rebuilds. With COPY . . first, the changed layer invalidates the install after it." },
        scene(G) {
          const names = ["FROM base", "COPY requirements", "RUN pip install", "COPY . .", "CMD"];
          G.text(24, 34, "you edited app.py", { size: 13, color: "hot" });
          const cols = [["good order", 40, [0, 0, 0, 1, 1]], ["bad order", 340, [0, 1, 1, 1, 1]]];
          const bad = ["FROM base", "COPY . .", "COPY requirements", "RUN pip install", "CMD"];
          cols.forEach(([title, x, st], c) => {
            G.text(x, 70, title, { size: 13, color: c ? "hot" : "ok" });
            const list = c ? bad : names, boxes = [];
            list.forEach((n, i) => {
              const hit = c ? i >= 1 : i >= 3;
              boxes.push(mb(G, x, 86 + i * 58, 240, 46, n + (hit ? "  rebuilt" : "  cached"), { fill: hit ? "hot" : "ok", size: 12 }));
            });
            G.from(boxes, { opacity: 0, y: 8, stagger: 0.08, duration: 0.25 });
          });
          G.label(40, 392, "good: only the last layer rebuilds", { size: 12, color: "ok" });
          G.label(340, 392, "bad: the edit invalidates pip install", { size: 12, color: "hot" });
          G.caption("a changed layer rebuilds everything below it");
        } },

      { rail: "not a VM", title: "A container is a process, not a machine",
        body: `<p>A <b>virtual machine</b> runs a whole second operating system, with its own kernel, on emulated hardware. A container does not. It is an ordinary process on the <b>host's kernel</b>, with two Linux features that make it look separate:</p>
<ul><li><b>namespaces</b> limit what the process can <i>see</i>: its own process list, its own network, its own file tree, its own hostname;</li>
<li><b>cgroups</b> limit what it can <i>use</i>: a ceiling on memory, CPU and more.</li></ul>
<p>That is why a container starts in a second and is small: there is no second kernel to boot. It is also why container isolation is weaker than a VM's: they all share one kernel. P3.8 returns to this.</p>`,
        check: { q: "Which is true of a container?",
          options: ["It boots its own kernel", "It shares the host's kernel and is restricted by namespaces and cgroups", "It needs a hypervisor"], answer: 1,
          why: "Containers are processes on the host kernel. Namespaces limit what they see and cgroups limit what they use. A VM is the one with its own kernel and a hypervisor." },
        scene(G) {
          G.text(24, 34, "virtual machines", { size: 13, color: "hot" }); G.text(344, 34, "containers", { size: 13, color: "ok" });
          [0, 1].forEach((i) => { mb(G, 24 + i * 140, 48, 126, 40, "app", { stroke: "k", color: "k", size: 12 }); mb(G, 24 + i * 140, 94, 126, 40, "guest kernel", { fill: "hot", size: 12 }); });
          mb(G, 24, 140, 266, 36, "hypervisor", { fill: "w", size: 12 }); mb(G, 24, 182, 266, 36, "host kernel", { fill: "w", size: 12 });
          [0, 1, 2].forEach((i) => mb(G, 344 + i * 90, 48, 80, 70, "app\n+ libs", { stroke: "k", color: "k", size: 12 }));
          G.rect(338, 40, 272, 86, { stroke: "ok", rx: 8, dash: "5 4" }); G.label(344, 142, "namespaces: what each can see", { size: 11, color: "ok" });
          G.label(344, 160, "cgroups: what each can use", { size: 11, color: "ok" });
          mb(G, 344, 182, 266, 36, "ONE host kernel (shared)", { fill: "w", size: 12 });
          G.label(24, 254, "own kernel per VM:", { size: 12 }); G.label(24, 272, "slower to start, stronger wall", { size: 12 });
          G.label(344, 254, "shared kernel:", { size: 12 }); G.label(344, 272, "fast and small, weaker wall", { size: 12 });
          mb(G, 24, 300, 586, 80, "host hardware", { stroke: "line", size: 13 });
          G.caption("the difference is the second kernel");
        } },

      { rail: "ports, volumes", title: "Connecting a container to the outside",
        body: `<p>A container starts isolated: its own network, its own files, an empty environment. Three flags connect it:</p>
<div class="eq">docker run --rm \\
  -p 8000:8000 \\
  -v $PWD/data:/data \\
  -e MODE=dev  myimage</div>
<ul><li><code>-p 8000:8000</code> <b>publishes</b> a port: host port first, container port second. Without it the server runs but nothing outside can reach it.</li>
<li><code>-v host:container</code> <b>mounts</b> a directory, so files survive the container.</li>
<li><code>-e NAME=value</code> sets an <b>environment variable</b>, the usual way to configure a container. Do not put secrets in an image.</li></ul>`,
        scene(G) {
          G.rect(24, 50, 120, 300, { stroke: "w", rx: 10 }); G.label(36, 72, "host", { color: "muted" });
          G.rect(260, 50, 360, 300, { stroke: "k", rx: 10 }); G.label(272, 72, "container (isolated)", { color: "k" });
          mb(G, 40, 100, 90, 40, "port 8000", { stroke: "v", color: "v", size: 12 }); mb(G, 280, 100, 130, 40, "server :8000", { stroke: "k", color: "k", size: 12 });
          const a = G.arrow(130, 120, 280, 120, { color: "v", w: 2 }); G.label(160, 108, "-p 8000:8000", { size: 11, color: "v" });
          mb(G, 40, 190, 90, 40, "./data", { stroke: "ok", color: "ok", size: 12 }); mb(G, 280, 190, 130, 40, "/data", { stroke: "ok", color: "ok", size: 12 });
          G.arrow(130, 210, 280, 210, { color: "ok", w: 2 }); G.arrow(280, 210, 130, 210, { color: "ok", w: 2 }); G.label(176, 198, "-v", { size: 11, color: "ok" });
          mb(G, 280, 280, 200, 40, "MODE=dev (env)", { stroke: "hot", color: "hot", size: 12 }); G.label(500, 304, "-e", { size: 12, color: "hot" });
          G.from(a, { opacity: 0, duration: 0.6 });
          G.caption("each flag opens one controlled connection");
        } },

      { rail: "tags, digests", title: "A tag can move; a digest cannot",
        body: `<p>A <b>registry</b> stores images. An image name such as <code>python:3.12-slim</code> is a repository plus a <b>tag</b>. A tag is a label its owner can move to a different image whenever they like. The same Dockerfile can therefore build something different next month.</p>
<p>A <b>digest</b> (<code>@sha256:...</code>) is a hash of the image's exact content. It cannot move. Pin by digest where repeatability matters, and keep the tag next to it for readability:</p>
<div class="eq">FROM python:3.12-slim@sha256:&lt;64 hex digits&gt;</div>
<p>The course's <code>env/</code> Dockerfiles follow this rule, and P3.1 has you enforce it.</p>`,
        check: { q: "Why pin a base image by digest rather than only by tag?",
          options: ["Digests download faster", "A tag can be re-pointed to a different image; a digest always names the same content", "Tags are not allowed in Dockerfiles"], answer: 1,
          why: "Tags are mutable labels. A digest is a content hash, so the build gets the same bytes every time." },
        scene(G) {
          G.text(24, 36, "python:3.12-slim (a tag)", { size: 13, color: "v" });
          const imgs = [["image A\nsha256:aaaa…", 24], ["image B\nsha256:bbbb…", 224], ["image C\nsha256:cccc…", 424]];
          imgs.forEach(([t, x]) => mb(G, x, 150, 170, 60, t, { stroke: "k", color: "k", size: 12 }));
                    mb(G, 44, 64, 130, 34, "tag (month 1)", { stroke: "v", color: "v", size: 11, rx: 17, dash: "4 3" }); G.arrow(109, 100, 109, 146, { color: "v", dash: "4 3" });
          mb(G, 244, 64, 130, 34, "tag (month 3)", { stroke: "v", color: "v", size: 11, rx: 17, dash: "4 3" }); G.arrow(309, 100, 309, 146, { color: "v", dash: "4 3" });
          const tag = mb(G, 444, 64, 130, 34, "tag (month 6)", { fill: "v", size: 11, rx: 17 }); G.arrow(509, 100, 509, 146, { color: "v", w: 2 });
          G.label(206, 252, "the tag moved: same Dockerfile,", { size: 12, color: "hot" }); G.label(206, 270, "different image", { size: 12, color: "hot" });
          const dg = mb(G, 24, 290, 250, 40, "@sha256:aaaa…  (a digest)", { fill: "ok", size: 12, rx: 20 });
          G.arrow(80, 288, 80, 214, { color: "ok", w: 2 });
          G.label(24, 358, "the digest always names image A", { size: 12, color: "ok" });
          G.from(tag, { opacity: 0, x: -200, duration: 1.0, delay: 0.3 });
          G.caption("tags are labels; digests are fingerprints");
        } },

      { rail: "one machine", title: "One machine is not enough",
        body: `<p>A container runs on one machine. One machine can crash, run out of memory or GPUs, or need a reboot. Real services run many copies across many machines and must survive any one failing.</p>
<p><b>Kubernetes</b> does this with one idea: a <b>desired-state loop</b>. You do not issue commands such as "start a container on server 3". You <i>declare</i> the end state: "three copies of this app". A <b>controller</b> then repeats, forever:</p>
<ol><li>read the desired state (3);</li><li>look at the actual state (say 2 running);</li><li>act to close the gap (start one more).</li></ol>
<p>A pod dies, a node vanishes: the loop notices a gap and fixes it. You never wrote restart logic.</p>`,
        check: { q: "You declared 3 replicas. A node fails and 1 replica is lost. What restores it?",
          options: ["You must run a command", "The controller sees 2 actual vs 3 desired and starts another", "Nothing, Kubernetes only starts apps once"], answer: 1,
          why: "The controller compares actual with desired state continuously and acts on any difference. Declaring the end state is the whole interface." },
        scene(G) {
          mb(G, 24, 50, 180, 60, "desired state\nreplicas: 3", { stroke: "v", color: "v", size: 14 });
          mb(G, 24, 150, 180, 60, "actual state\nrunning: 2", { stroke: "hot", color: "hot", size: 14 });
          const ctl = mb(G, 250, 100, 140, 70, "controller\n(a loop)", { fill: "v", size: 14 });
          G.arrow(204, 80, 250, 120, { color: "v", w: 2 }); G.arrow(204, 180, 250, 150, { color: "hot", w: 2 });
          G.text(250, 200, "gap = 1  →  act", { size: 12, color: "hot" });
          [0, 1].forEach((i) => mb(G, 440 + i * 90, 70, 80, 50, "pod " + (i + 1), { fill: "ok", size: 12 }));
          const np = mb(G, 530, 150, 80, 50, "pod 3", { stroke: "hot", color: "hot", size: 12, dash: "4 3" });
          const dead = mb(G, 440, 150, 80, 50, "pod ✕", { stroke: "line", color: "muted", size: 12, dash: "4 3" });
          G.arrow(390, 140, 440, 175, { color: "muted", dash: "4 3" }); G.arrow(390, 140, 530, 175, { color: "v", w: 2 });
          G.label(24, 270, "forever: read desired · read actual · close the gap", { size: 13, color: "ink" });
          G.label(24, 296, "self-healing is just this loop", { size: 12 });
          G.from(np, { opacity: 0, scale: 0.4, transformOrigin: "center", delay: 0.6, duration: 0.5 }); G.pulse(ctl, { repeat: 5 });
          G.caption("declare the end state; the loop does the rest");
        } },

      { rail: "pods, services", title: "Pods run it; a Deployment keeps it; a Service names it",
        body: `<p>Three objects cover most of what you need.</p>
<ul><li>A <b>Pod</b> is the smallest unit: one or more containers that run together on one node. Pods are disposable.</li>
<li>A <b>Deployment</b> declares "keep N identical Pods of this template" and handles rolling updates. It is the controller from the last step.</li>
<li>A <b>Service</b> gives a set of Pods one stable address, because individual Pod IPs change every time a Pod is replaced.</li></ul>
<p>The glue is <b>labels</b>: key/value tags on objects, like <code>app: echo</code>. A <b>selector</b> picks objects by label. The Deployment finds its Pods by selector, and the Service finds the same Pods by selector. If the selector and the Pod labels disagree, you get no Pods, or a Service with no endpoints.</p>`,
        check: { q: "A Service selects app: echo, but the Pods are labelled app: ecko. What happens?",
          options: ["Traffic is spread anyway", "The Service has no endpoints and requests fail", "Kubernetes fixes the label"], answer: 1,
          why: "A Service sends traffic only to Pods whose labels match its selector. No match means an empty endpoint list. Kubernetes does not guess." },
        scene(G) {
          mb(G, 24, 50, 130, 56, "client\ncurl echo:80", { stroke: "ink", size: 12 });
          mb(G, 200, 50, 410, 56, "Service echo   (stable address)\nselector app=echo  ·  port 80 → 5678", { fill: "ok", size: 12 });
          G.arrow(154, 78, 200, 78, { color: "ink", w: 2 });
          const pods = [0, 1].map((i) => mb(G, 230 + i * 210, 170, 160, 70, `Pod\napp=echo\n10.0.0.${i + 4}`, { fill: "k", size: 12 }));
          G.arrow(310, 108, 310, 168, { color: "ok", w: 2 }); G.arrow(520, 108, 520, 168, { color: "ok", w: 2 });
          mb(G, 200, 300, 410, 70, "Deployment echo\nreplicas: 2  ·  selector app=echo", { stroke: "v", color: "v", size: 12 });
          G.arrow(310, 298, 310, 242, { color: "v", w: 2 }); G.arrow(520, 298, 520, 242, { color: "v", w: 2 });
          G.label(330, 276, "creates and replaces", { size: 11, color: "v" });
          G.label(24, 180, "Pod IPs change;", { size: 11 }); G.label(24, 198, "the Service name", { size: 11 }); G.label(24, 216, "does not.", { size: 11 });
          G.label(24, 396, "labels connect all three: app=echo", { size: 12, color: "ok" });
          G.from(pods, { opacity: 0, y: 14, stagger: 0.2, duration: 0.4 });
          G.caption("the selector is the link");
        } },

      { rail: "kubectl, kind", title: "Four commands and a cluster on your laptop",
        body: `<p>You talk to Kubernetes with <code>kubectl</code>. Four commands do most of the work:</p>
<div class="eq">kubectl apply -f FILE     declare it
kubectl get pods           what exists now
kubectl describe pod NAME  events: why Pending?
kubectl logs NAME          container output</div>
<p><b>kind</b> runs a Kubernetes cluster inside Docker on your laptop: each "node" is a container. It costs nothing and deletes cleanly. The repo's config is <code>platform/deploy/kind/kind-config.yaml</code>: one control plane and two workers. Exercise 3 walks through it. Those commands are <code>TODO(run)</code> here because the build environment has no Docker.</p>`,
        scene(G) {
          G.rect(24, 40, 360, 240, { stroke: "w", rx: 12 }); G.label(36, 62, "your laptop · Docker", { color: "muted" });
          ["control-plane", "worker (system)", "worker (gpu pool)"].forEach((n, i) => mb(G, 44 + (i % 2) * 170, 82 + Math.floor(i / 2) * 90, 150, 70, "kind node\n" + n, { stroke: "k", color: "k", size: 11 }));
          mb(G, 214, 172, 150, 70, "pods run\ninside nodes", { stroke: "ok", color: "ok", size: 11, dash: "4 3" });
          mb(G, 430, 90, 186, 50, "kubectl", { fill: "v", size: 14 });
          G.arrow(430, 115, 384, 115, { color: "v", w: 2 });
          ["apply", "get", "describe", "logs"].forEach((c, i) => G.text(440, 176 + i * 28, "kubectl " + c, { size: 12, color: "ink" }));
          G.label(24, 314, "kind create cluster --name s2s --config platform/deploy/kind/kind-config.yaml", { size: 10 });
          G.label(24, 336, "kind delete cluster --name s2s", { size: 11 });
          G.caption("a whole cluster as three containers");
        } },

      { rail: "requests, GPU", title: "Requests, limits, and a GPU as a countable resource",
        body: `<p>A container says how much it needs:</p>
<div class="eq">resources:
  requests: { cpu: 100m, memory: 128Mi }
  limits:   { cpu: 500m, memory: 256Mi }</div>
<p>The <b>request</b> is what the scheduler reserves when it chooses a node: a Pod goes only where the unreserved capacity covers it. The <b>limit</b> is a ceiling: over its memory limit, a container is killed. (<code>100m</code> means a tenth of a CPU core. <code>Mi</code> is mebibytes.)</p>
<p>A GPU is <i>just another resource</i>. A node advertises <code>nvidia.com/gpu: 1</code>, and a Pod asks for it under <code>limits</code>. The scheduler counts whole GPUs like any other quantity. How a node learns it has one is the job of the GPU Operator, in P3.3.</p>`,
        scene(G) {
          G.rect(24, 40, 330, 300, { stroke: "w", rx: 12 }); G.label(36, 62, "node capacity (example)", { color: "muted" });
          const rows = [["CPU", 4, 2.5, "cores"], ["memory", 16, 10, "GiB"], ["nvidia.com/gpu", 1, 1, "GPU"]];
          rows.forEach(([n, cap, used, u], i) => {
            const y = 90 + i * 76; G.label(40, y, n + "  (capacity " + cap + " " + u + ")", { size: 12, color: "ink" });
            G.rect(40, y + 10, 290, 24, { stroke: "line", rx: 5 }); G.rect(40, y + 10, (used / cap) * 290, 24, { fill: i === 2 ? "hot" : "k", rx: 5, opacity: 0.85 });
            G.text(44 + (used / cap) * 290 - 6, y + 27, "reserved " + used, { anchor: "end", size: 11, color: "bg", weight: 700 });
          });
          G.label(40, 326, "free GPU: 0  → next GPU Pod waits (Pending)", { size: 11, color: "hot" });
          mb(G, 400, 70, 216, 120, "Pod\nrequests cpu 500m\nmemory 1Gi\nlimits nvidia.com/gpu: 1", { stroke: "k", color: "k", size: 11 });
          G.arrow(400, 130, 356, 160, { color: "k", w: 2 });
          mb(G, 400, 230, 216, 60, "scheduler: picks a node\nwith enough unreserved", { fill: "v", size: 11 });
          G.caption("a GPU is a counted resource, whole units only");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>An image is read-only layers; a container is a process on the host's kernel with a writable layer, limited by namespaces and cgroups. It is not a VM.</li>
<li>A changed Dockerfile layer rebuilds everything after it, so copy dependency files first and your code last.</li>
<li><code>-p</code>, <code>-v</code> and <code>-e</code> connect a container to ports, files and configuration. Pin images by digest.</li>
<li>Kubernetes is a desired-state loop. A Deployment keeps N Pods, a Service names them, and labels with selectors link the two.</li>
<li><code>kubectl apply/get/describe/logs</code> run the loop on a kind cluster. Requests reserve, limits cap, and a GPU is a resource named <code>nvidia.com/gpu</code>.</li></ul>
<p>Next, P3.1 puts a GPU server in a container and explains the one thing images cannot carry: the host driver.</p>
<h2>Go further</h2><ul>
<li><a href="https://docs.docker.com/get-started/">Docker: Get started</a></li>
<li><a href="https://kubernetes.io/docs/tutorials/kubernetes-basics/">Kubernetes basics tutorial</a></li>
<li><a href="https://kind.sigs.k8s.io/docs/user/quick-start/">kind quick start</a></li></ul>
<p>These links are UNVERIFIED from the build environment. Check them.</p>`,
    sim: {
      title: "Which layers rebuild?",
      intro: "An 8-line Dockerfile. Choose what you changed. Every layer from the first changed one downwards is rebuilt (red); layers above it come from the cache (green). Switch the order to see why copying the code before the dependencies is expensive. Sizes are example values, only there to show proportions.",
      height: 360,
      controls: [
        { id: "chg", label: "what changed", type: "select", value: "code", options: [["base", "base image tag bumped"], ["apt", "system package list"], ["deps", "dependency file (uv.lock)"], ["code", "application code"]] },
        { id: "ord", label: "Dockerfile order", type: "select", value: "good", options: [["good", "dependencies first, code last"], ["bad", "copy code before dependencies"]] },
      ],
      draw(G, v) {
        const L = v.ord === "good" ? GOOD : BAD;
        const first = L.findIndex((l) => l[2] === v.chg);
        let rebuilt = 0, mb_ = 0, total = 0;
        L.forEach((l, i) => {
          total += l[1];
          const hit = i >= first; if (hit) { rebuilt++; mb_ += l[1]; }
          const y = 12 + i * 41;
          G.rect(10, y, 330, 34, { fill: hit ? "hot" : "ok", rx: 6, opacity: 0.9 });
          G.text(20, y + 22, `${i + 1}  ${l[0]}`, { size: 11, color: "bg", weight: 700 });
          const w = Math.max(2, (l[1] / 900) * 190);
          G.rect(360, y + 6, w, 22, { fill: hit ? "hot" : "ok", rx: 3, opacity: 0.9 });
          G.label(366 + w, y + 22, l[1] + " MB", { size: 10 });
        });
        G.label(360, 350, "bar width = example layer size (MB)", { size: 10 });
        return [
          { title: "This build", rows: [["changed line", first + 1], ["layers rebuilt", `${rebuilt} of 8`], ["layers from cache", `${first} of 8`], ["example size rebuilt", `${F.num(mb_)} MB of ${F.num(total)} MB`]],
            gauge: [[total ? mb_ / total : 0, mb_ / total > 0.5 ? "hot" : "ok"]], gaugeText: `${F.num(total ? (mb_ / total) * 100 : 0)}% of the image rebuilt`,
            chip: [mb_ / total < 0.2, mb_ / total < 0.2 ? "cheap rebuild" : "expensive rebuild"] },
          { title: "Rule", html: `<p class="note">A changed layer invalidates every layer after it. Put rarely-changing steps first and the code last. Sizes here are examples, not measurements.</p>` },
        ];
      },
      note: "Try 'application code' in both orders. Then try 'dependency file': in the good order the install is rebuilt for a real reason.",
    },
    practice: {
      intro: `Exercises 1 and 2 need no Docker. Exercise 3 needs Docker, kind and kubectl and is marked <code>TODO(run)</code>, because this build environment cannot run them. Run commands from the repository root.`,
      items: [
        { title: "Dockerfile linter", tier: "T0 · easy", goal: "Find unpinned or latest base images, COPY . . before the install, and a missing non-root USER. This is the beginner linter; P3.1 has the stricter one.",
          cmd: `uv run pytest ${MODULE}/exercises/01-dockerfile-lint` },
        { title: "Deployment and Service YAML", tier: "T0 · medium", goal: "Write a 2-replica echo Deployment with a readiness probe, requests and limits, plus a Service. The test checks that the labels and selectors match.",
          cmd: `uv run pytest ${MODULE}/exercises/02-deployment-yaml` },
        { title: "kind walkthrough", tier: "T0 · Docker, kind, kubectl · TODO(run)", goal: "Create a cluster, apply your YAML, port-forward and curl it, break the selector on purpose, then delete the cluster.",
          cmd: `cat ${MODULE}/exercises/03-kind-walkthrough/README.md` },
        { title: "All reference solutions", tier: "T0 · check", goal: "Confirm the solutions pass.", cmd: `S2S_SOLUTIONS=1 uv run pytest ${MODULE}` },
      ],
      labs: [
        { label: "The kind cluster config", path: "platform/deploy/kind/kind-config.yaml" },
        { label: "The course's Dockerfiles (read them with your linter)", path: "env/" },
        { label: "P3.1's stricter Dockerfile policy check", path: "course/P3-deployment-and-infra/P3.1-gpu-containers/examples/02_dockerfile_policy.py" },
        { label: "Self-check questions", path: `${MODULE}/quiz.md` },
      ],
    },
  });
})();
