/* P3.3 — Kubernetes and the GPU Operator. Kubernetes from first principles (desired state, pods, services,
   requests and limits, taints), how a pod gets a GPU, what each Operator component does, and the pod spec lines
   that make a GPU server survive load times, rollouts and shutdowns. Manifests: platform/deploy/base/vllm.yaml. */
(function () {
  const F = S2S.fmt;

  /* ---------- simulator: what happens to streams when a pod is deleted (platform/deploy/base/vllm.yaml) ---------- */
  function drainModel(v) {
    const S = v.sleep, G = v.grace, P = v.prop, L = v.len, r = v.rate;
    const lastAccepted = Math.min(S, P);                 // new requests still arrive until endpoints update; refused once SIGTERM has stopped intake
    const refused = P > S ? r * (P - S) : 0;
    const drainEnd = Math.max(S, lastAccepted + L);      // last accepted stream finishes lastAccepted + L; SIGTERM only at S
    const cut = drainEnd > G ? r * Math.max(0, lastAccepted - (G - L)) : 0;   // streams still running when SIGKILL lands (steady arrivals, fixed length L)
    return { lastAccepted, refused, drainEnd, cut, killed: drainEnd > G };
  }

  S2S.lesson({
    id: "p3-3", n: "P3.3", title: "Kubernetes and the GPU Operator",
    subtitle: "Deployment and infra · first principles · T0 on kind, T3 on EKS with GPUs",
    kicker: "Lesson · ≈ 50 min",
    headline: "Telling a cluster what you want, and giving it a GPU",
    intro: `<p>One GPU server that you start by hand is a lab. A service needs many of them to start, stay up, get replaced when they die, be updated without dropping users, and land on the right machines. <b>Kubernetes</b> does that by a simple idea: you declare the state you want, and a set of controllers keep making reality match it. This lesson builds that idea first, then follows one request for a GPU from a pod spec down to the device, explains the NVIDIA <b>GPU Operator</b> pieces that make a node ready, and reads <code>platform/deploy/base/vllm.yaml</code> line by line: probes that survive a slow model load, drains that do not cut streams, and a spec that passes a security policy.</p>`,
    facts: ["11 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "pod / workload"], ["ok", "ready / allowed"], ["hot", "rejected / killed"], ["v", "waiting / event"], ["w", "node / infra"]],
    prev: "p3-2", next: "p3-4",
    steps: [
      { rail: "desired state", title: "Declare what you want; a loop makes it so",
        body: `<p>Suppose you run 3 copies of a server across several machines. Machines fail, processes crash, you roll out new versions. Writing scripts for each case ("if X died, start a replacement on a machine with free memory…") never ends. Kubernetes replaces the scripts with <b>desired state</b>: you write down <i>what should be true</i>, and software keeps checking and repairing.</p>
<p>The loop, run by <b>controllers</b>, is the same everywhere in Kubernetes:</p>
<ol><li><b>Observe</b> what actually exists.</li><li><b>Compare</b> it with what you declared.</li><li><b>Act</b> to close the gap, then repeat forever.</li></ol>
<div class="eq">declared:  replicas: 3
observed:  2 pods running, 1 crashed
action:    create 1 pod  (the scheduler picks a node)
observed:  3 running → nothing to do</div>
<p>You never tell Kubernetes <i>how</i> to restart the crashed pod. You say "3", and the Deployment controller does the arithmetic every few seconds. Everything in this lesson is a declaration of this kind, written as YAML and applied with <code>kubectl apply -k platform/deploy/overlays/…</code> (<code>-k</code> uses kustomize to combine a base with an overlay for kind or EKS).</p>
<p>A Kubernetes <b>cluster</b> is a control plane (the API and the controllers; on EKS, run by AWS, P3.2) plus <b>nodes</b>, the machines that run containers. The GPU node group from P3.2 is the set of nodes this lesson cares about.</p>`,
        scene(G) {
          G.text(24, 32, "the reconcile loop", { size: 13 });
          G.box(24, 64, 150, 56, "declared", { fill: "k", size: 14 }); G.label(30, 150, "replicas: 3", { size: 12, color: "ink" });
          G.box(246, 64, 150, 56, "controller", { stroke: "ok", color: "ok", size: 14 });
          G.arrow(176, 92, 244, 92, { color: "muted", w: 2 }); G.label(178, 84, "compare", { size: 10 });
          G.box(468, 64, 150, 56, "observed", { fill: "w", size: 14 }); G.label(474, 150, "what is running now", { size: 12, color: "ink" });
          G.arrow(466, 92, 398, 92, { color: "muted", w: 2 }); G.label(404, 84, "observe", { size: 10 });
          G.path("M 321 122 C 321 170, 543 170, 543 126", { color: "v", w: 2, dash: "5 4", arrow: true });
          G.label(380, 186, "act: create, delete, restart", { size: 12, color: "v" });
          G.text(24, 240, "example", { size: 13 });
          const pods = [["running", "ok"], ["crashed", "hot"], ["running", "ok"]].map(([t, c], i) => G.box(24 + i * 100, 258, 88, 36, t, { fill: c, size: 12 }));
          G.label(24, 312, "2 of 3: the gap is 1", { size: 12, color: "hot" });
          G.arrow(330, 276, 370, 276, { color: "v", w: 2 });
          const repl = G.box(380, 258, 88, 36, "new pod", { stroke: "v", color: "v", size: 12, dash: "4 3" });
          G.label(380, 312, "3 of 3: nothing to do", { size: 12, color: "ok" });
          G.text(24, 372, "you declare WHAT; the loop works out HOW", { size: 14, color: "ink" });
          G.from(pods.concat([repl]), { opacity: 0, stagger: 0.1, duration: 0.25 }); G.pulse(repl, { repeat: 4 });
          G.caption("declare the end state; controllers close the gap forever");
        } },

      { rail: "pods & services", title: "Pods, Deployments and Services",
        body: `<p>Three objects cover most of what a serving stack needs.</p>
<ul><li>A <b>Pod</b> is the smallest unit: one or more containers that share a network address and run on the same node. In <code>vllm.yaml</code> a pod has one container, the vLLM server. Pods are disposable: when one dies it is replaced by a <i>new</i> pod with a new IP.</li><li>A <b>Deployment</b> declares "keep N identical pods of this template running" and manages updates by creating a new ReplicaSet and shifting pods across (next steps cover how).</li><li>A <b>Service</b> gives a changing set of pods one stable name and address (<code>vllm:8000</code>). It picks pods by <b>label</b>: pods carry labels (<code>app: vllm</code>), the Service has a <i>selector</i> (<code>app: vllm</code>), and it routes only to the pods that match <i>and are Ready</i>.</li></ul>
<div class="eq">Deployment vllm: pods labelled app: vllm
Service vllm:     selector app: vllm, port 8000
client → http://vllm.s2s:8000 (stable name)
       → one of the Ready pods</div>
<p>The last words matter for GPU servers: the Service lists a pod among its <b>endpoints</b> only while its readiness probe passes. A pod that is still loading 16 GB of weights receives no traffic. This is how the gateway from P2.7 can use one stable URL while pods come and go behind it. The namespace <code>s2s</code> groups all of the course's objects, so they can be listed or deleted together.</p>`,
        scene(G) {
          G.text(24, 32, "Deployment → pods; Service → Ready pods by label", { size: 13 });
          G.box(24, 56, 170, 44, "Deployment vllm", { fill: "k", size: 13 }); G.label(28, 116, "replicas: 3 (example)", { size: 11 });
          const pods = [["Ready", "ok"], ["Ready", "ok"], ["loading…", "v"]].map(([t, c], i) => { const y = 62 + i * 70; G.arrow(196, 78, 246, y + 20, { color: "muted", w: 1.5 }); G.box(248, y, 150, 48, "pod · app: vllm", { stroke: c, color: c, size: 12 }); G.label(256, y + 62, t, { size: 11, color: c }); return null; });
          G.box(470, 100, 146, 60, "Service vllm", { fill: "w", size: 13 }); G.label(474, 176, "selector app: vllm", { size: 11 });
          G.line(400, 86, 468, 120, { color: "ok", w: 2 }); G.line(400, 156, 468, 134, { color: "ok", w: 2 });
          G.line(400, 226, 468, 148, { color: "v", w: 2, dash: "4 3" }); G.label(404, 262, "not Ready: no traffic yet", { size: 11, color: "v" });
          G.text(24, 316, "client → vllm:8000 → one of the Ready pods", { size: 13, color: "ink" });
          G.label(24, 340, "pods get new IPs when replaced; the Service name does not change", { size: 12 });
          G.label(24, 362, "endpoints = pods matching the selector AND passing readiness", { size: 12, color: "ok" });
          G.caption("a stable name over a changing set of pods");
        } },

      { rail: "requests & limits", title: "Requests, limits, and the GPU as a resource",
        body: `<p>The scheduler (the controller that picks a node for each new pod) must know how much a pod needs. Each container declares two numbers per resource:</p>
<ul><li><b>requests</b>: what the scheduler <i>reserves</i>. A pod is placed only on a node whose remaining <b>allocatable</b> amount covers it.</li><li><b>limits</b>: the cap enforced at run time. Exceed a memory limit and the container is killed (<code>OOMKilled</code>); CPU over the limit is merely throttled.</li></ul>
<p><code>vllm.yaml</code> asks for <code>cpu: 3, memory: 12Gi, nvidia.com/gpu: 1</code> and limits <code>memory: 14Gi, nvidia.com/gpu: 1</code> (no CPU limit, so no throttling of the tokenizer threads). Place it on a node with, as an example, 3.9 CPU, 14 GiB and 1 GPU allocatable (<code>kubectl describe node</code> shows the real numbers; some of a node's capacity is kept for the system):</p>
<div class="eq">first pod:   cpu 3 ≤ 3.9   memory 12 ≤ 14   gpu 1 ≤ 1   ✓
second pod:  cpu 6 &gt; 3.9   memory 24 &gt; 14   gpu 2 &gt; 1   ✗
             → Pending, unschedulable on this node</div>
<p>The GPU is an <b>extended resource</b>: a countable thing the node advertises under a name (<code>nvidia.com/gpu</code>), with one rule that surprises people. It cannot be shared fractionally or overcommitted, it must be named in <code>limits</code>, and if you also give a <code>request</code> it must equal the limit (giving only a limit is enough). Whole GPUs only; sharing a GPU needs the time-slicing or MIG setups of P4.4.</p>`,
        check: { q: "How do you ask Kubernetes for one GPU?",
          options: ["Put nvidia.com/gpu: 1 under requests only", "Put nvidia.com/gpu: 1 in limits (a request, if given, must equal it)", "Set a CPU request; the GPU follows automatically"], answer: 1,
          why: "An extended resource such as nvidia.com/gpu is set in limits; if requests are also given they must be identical, because it cannot be overcommitted. Requests alone is rejected, and CPU settings have nothing to do with it." },
        scene(G) {
          G.text(24, 32, "one node (example allocatable), one vLLM pod fits", { size: 13 });
          const rows = [["CPU", 3.9, 3, "3 of 3.9"], ["memory (GiB)", 14, 12, "12 of 14"], ["GPU", 1, 1, "1 of 1"]];
          const bars = [];
          rows.forEach(([n, cap, use, lab], i) => {
            const y = 66 + i * 58, w = 360;
            G.text(24, y + 22, n, { size: 13 });
            G.rect(150, y, w, 32, { stroke: "line", rx: 4 });
            bars.push(G.rect(150, y, (w * use) / cap, 32, { fill: "k", rx: 4 }));
            G.label(150 + w + 10, y + 21, lab, { size: 12, color: "ink" });
          });
          G.text(24, 262, "second identical pod:", { size: 13, color: "hot" });
          rows.forEach(([n, cap, use], i) => { G.box(24 + i * 200, 280, 180, 36, `${n.split(" ")[0]} ${use * 2} > ${cap}`, { stroke: "hot", color: "hot", size: 12 }); });
          G.text(24, 352, "→ Pending: no node has room", { size: 14, color: "hot" });
          G.label(24, 378, "node numbers are an example: kubectl describe node shows the real allocatable", { size: 12 });
          G.from(bars, { attr: { width: 0 }, stagger: 0.12, duration: 0.5 });
          G.caption("requests are reserved at scheduling time; a GPU is all or nothing");
        } },

      { rail: "taints & selectors", title: "Keeping the right pods on the right nodes",
        body: `<p>A GPU node costs many times a CPU node (P3.2), and the GPU pool scales to zero when idle. Two failure modes to prevent: an ordinary pod landing on a GPU node and keeping it alive, and a GPU pod landing somewhere without a GPU. Kubernetes has three tools, and each does one job:</p>
<ul><li>A <b>taint</b> on a node <i>repels</i>: <code>nvidia.com/gpu=present:NoSchedule</code> means "no pod may be scheduled here unless it tolerates this".</li><li>A <b>toleration</b> on a pod <i>permits</i>: it lets the pod ignore the matching taint. It does not attract the pod to the node.</li><li>A <b>nodeSelector</b> (or <b>affinity</b>) <i>attracts</i> by labels: <code>s2s/pool: gpu</code> restricts the pod to nodes carrying that label.</li></ul>
<div class="eq">GPU node:  taint nvidia.com/gpu=present:NoSchedule
           label s2s/pool=gpu
vllm pod:  toleration for nvidia.com/gpu (may enter)
           nodeSelector s2s/pool: gpu    (must go)
web pod:   no toleration                 (kept out)</div>
<p>You need both halves for a dedicated pool: taint plus toleration alone lets the GPU pod run <i>anywhere</i> that tolerates, including non-GPU nodes; a selector alone does not keep strangers out. The EKS stack of P3.2 applies exactly the taint and label to the GPU node group, and <code>vllm.yaml</code> carries the matching toleration and selector. On kind (T0), the worker labelled <code>s2s/pool: gpu</code> plays the GPU pool and mockllm pods tolerate the same taint, so the scheduling rules are testable on a laptop.</p>`,
        check: { q: "A pod tolerates the nvidia.com/gpu taint but has no nodeSelector or affinity. Where may it run?",
          options: ["Only on GPU nodes", "On GPU nodes and on any untainted node: a toleration permits, it does not attract", "Nowhere until it requests a GPU"], answer: 1,
          why: "A toleration only removes the taint's repulsion. To confine a pod to the GPU pool you also need a nodeSelector or affinity on a label that only GPU nodes have." },
        scene(G) {
          G.text(24, 32, "taint repels · toleration permits · selector attracts", { size: 13 });
          G.rect(24, 52, 250, 180, { stroke: "w", rx: 8 }); G.text(34, 72, "system node (no taint)", { size: 12, color: "w" });
          G.rect(300, 52, 316, 180, { stroke: "k", rx: 8 }); G.text(310, 72, "GPU node · taint nvidia.com/gpu", { size: 12, color: "k" });
          G.label(310, 88, "label s2s/pool=gpu", { size: 11 });
          G.box(40, 100, 100, 40, "web pod", { stroke: "ok", color: "ok", size: 12 }); G.label(40, 156, "runs here", { size: 11, color: "ok" });
          G.box(40, 168, 160, 40, "vllm pod", { stroke: "hot", color: "hot", size: 12 }); G.line(40, 168, 200, 208, { color: "hot", w: 2 }); G.line(40, 208, 200, 168, { color: "hot", w: 2 }); G.label(40, 224, "selector says s2s/pool=gpu", { size: 11, color: "hot" });
          G.box(330, 100, 120, 40, "vllm pod", { fill: "k", size: 12 }); G.label(330, 156, "tolerates + selects", { size: 11, color: "k" });
          G.box(480, 100, 120, 40, "web pod", { stroke: "hot", color: "hot", size: 12 }); G.line(480, 100, 600, 140, { color: "hot", w: 2 }); G.line(480, 140, 600, 100, { color: "hot", w: 2 });
          G.label(480, 156, "repelled", { size: 11, color: "hot" });
          G.text(24, 276, "taint alone: others stay out", { size: 13, color: "hot" });
          G.text(24, 302, "toleration alone: the pod may wander anywhere", { size: 13, color: "v" });
          G.text(24, 328, "taint + toleration + selector: a dedicated pool", { size: 13, color: "ok" });
          G.label(24, 366, "kind labels a worker s2s/pool=gpu so this is testable on a laptop (T0)", { size: 12 });
          G.caption("repel strangers, permit your pods, attract them to the pool");
        } },

      { rail: "device plugin", title: "How a pod actually gets a GPU",
        body: `<p>Kubernetes itself knows nothing about GPUs. A small program on each GPU node, the <b>device plugin</b>, teaches it. The path of one request, from your YAML to a running container that sees a GPU:</p>
<ol><li>The pod says <code>limits: nvidia.com/gpu: 1</code> and tolerates the taint.</li><li>The device plugin, started on the node, has already told the <b>kubelet</b> (the node's agent) "this node has N <code>nvidia.com/gpu</code>". The node then advertises <code>allocatable nvidia.com/gpu: N</code>.</li><li>The <b>scheduler</b> considers only nodes with at least 1 allocatable GPU (and tolerated taints, and matching selectors) and picks one.</li><li>The kubelet on that node asks the device plugin <i>which</i> GPU to give; the plugin answers with a specific one (a UUID).</li><li>The container runtime, with the <b>NVIDIA container toolkit</b> (P3.1), injects that GPU's device files and the driver libraries into the container.</li><li>The container sees exactly that GPU (<code>CUDA_VISIBLE_DEVICES</code> is set to match).</li></ol>
<div class="eq">limits nvidia.com/gpu: 1
 → scheduler: nodes with allocatable ≥ 1
 → kubelet asks device plugin: which GPU?
 → plugin answers with one GPU's UUID
 → toolkit injects /dev/nvidia0 + libs
 → container sees one GPU</div>
<p>Until the device plugin is running, the node advertises <b>zero</b> GPUs, so a GPU pod stays Pending even though the hardware and driver are fine. This is why a fresh GPU node takes about a minute beyond joining the cluster before pods can use it (step 11).</p>`,
        scene(G) {
          G.text(24, 32, "one pod → one GPU", { size: 13 });
          const steps = [["pod spec", "limits nvidia.com/gpu: 1", "k"], ["scheduler", "node has allocatable ≥ 1", "v"], ["kubelet + device plugin", "which GPU? → GPU-UUID", "ok"], ["container toolkit", "inject /dev/nvidia0 + libs", "q"], ["container", "sees exactly that GPU", "blue"]];
          const bx = steps.map(([t, d, c], i) => {
            const y = 54 + i * 62;
            const b = G.box(24, y, 220, 44, t, { fill: c, size: 13 });
            G.text(262, y + 26, d, { size: 13, color: c });
            if (i < 4) G.arrow(134, y + 46, 134, y + 60, { color: "muted", w: 2 });
            return b;
          });
          G.label(24, 376, "node advertises allocatable nvidia.com/gpu only after the device plugin is up", { size: 12, color: "ink" });
          G.label(24, 398, "no plugin → 0 GPUs advertised → GPU pods stay Pending", { size: 12, color: "hot" });
          G.from(bx, { opacity: 0, stagger: 0.12, duration: 0.25 });
          G.caption("Kubernetes counts GPUs because the device plugin says so");
        } },

      { rail: "GPU Operator", title: "The GPU Operator: what makes a node GPU-ready",
        body: `<p>Someone has to install the driver, configure the container runtime, run the device plugin, label the nodes, and export metrics. The NVIDIA <b>GPU Operator</b> automates all of it: a set of DaemonSets (one pod per GPU node), installed once with Helm. Its parts:</p>
<table><tr><th>Component</th><th>Job</th><th>On the EKS AL2023 NVIDIA AMI</th></tr>
<tr><td>driver</td><td>installs the kernel driver on each GPU node</td><td><b>disabled</b>: the AMI has it</td></tr>
<tr><td>container toolkit</td><td>configures containerd to inject GPUs</td><td><b>disabled</b>: the AMI has it</td></tr>
<tr><td><b>device plugin</b></td><td>advertises and allocates <code>nvidia.com/gpu</code></td><td>on</td></tr>
<tr><td><b>GFD</b> (GPU Feature Discovery)</td><td>labels nodes with GPU properties</td><td>on</td></tr>
<tr><td><b>DCGM exporter</b></td><td>GPU metrics for Prometheus (P3.5)</td><td>on</td></tr>
<tr><td>MIG manager</td><td>partitions A100/H100 GPUs (P4.4)</td><td>off for now</td></tr></table>
<p>P3.2's EKS node group uses the <code>AL2023_x86_64_NVIDIA</code> AMI type, which already contains the driver and the toolkit, so installing them again would be redundant and risky. <code>platform/deploy/gpu-operator-values.yaml</code> records this: <code>driver.enabled: false</code>, <code>toolkit.enabled: false</code>, the rest on, and the Operator's own pods tolerate the GPU taint so they can run on the tainted pool. The pinned version is Operator <code>v26.7.1</code> with device plugin <code>v0.20.1</code> and dcgm-exporter <code>4.8.4</code>. The value keys are <b>UNVERIFIED</b> against the chart at that version; check <code>deployments/gpu-operator/values.yaml</code> in the pinned repository.</p>`,
        check: { q: "On EKS with the AL2023 NVIDIA AMI, which GPU Operator components are turned off, and why?",
          options: ["The device plugin and GFD, since EKS handles them", "The driver and the container toolkit, because the AMI already ships both", "None: the Operator must always install everything"], answer: 1,
          why: "The AMI type used in P3.2 comes with the NVIDIA driver and container toolkit preinstalled. The Operator still runs the device plugin, GPU Feature Discovery and the DCGM exporter, which the AMI does not provide." },
        scene(G) {
          G.text(24, 32, "operator DaemonSets on one GPU node (EKS AL2023 NVIDIA)", { size: 13 });
          G.rect(16, 48, 608, 290, { stroke: "k", rx: 10, dash: "5 4" });
          const parts = [["driver", "in the AMI", false], ["container toolkit", "in the AMI", false], ["device plugin", "nvidia.com/gpu", true], ["GFD", "node labels", true], ["DCGM exporter", "GPU metrics", true], ["MIG manager", "later, P4.4", false]];
          const out = parts.map(([t, d, on], i) => {
            const x = 32 + (i % 3) * 196, y = 70 + Math.floor(i / 3) * 120;
            const b = G.box(x, y, 180, 56, t, on ? { fill: "ok", size: 13 } : { stroke: "muted", color: "muted", size: 13, dash: "4 3" });
            G.label(x + 4, y + 74, on ? "on · " + d : "off · " + d, { size: 11, color: on ? "ok" : "muted" });
            return b;
          });
          G.text(24, 372, "Helm values: driver off, toolkit off, the rest on", { size: 13, color: "ink" });
          G.label(24, 396, "platform/deploy/gpu-operator-values.yaml (keys UNVERIFIED vs chart v26.7.1)", { size: 12 });
          G.from(out, { opacity: 0, stagger: 0.08, duration: 0.25 });
          G.caption("install only what the AMI does not already provide");
        } },

      { rail: "GPU properties", title: "Scheduling on what the GPU is, not what the node is called",
        body: `<p>A pod rarely cares that a node is a <code>g6.xlarge</code>; it cares that the GPU has enough memory. One 8B model in bf16 needs about 16 GB for weights (P1.1), plus KV cache. If you select by instance name, you must edit every manifest when the fleet changes. <b>GPU Feature Discovery</b> (GFD) solves this: it runs on each GPU node and adds labels that describe the hardware, such as <code>nvidia.com/gpu.product</code>, <code>nvidia.com/gpu.memory</code> and <code>nvidia.com/gpu.compute.major</code>.</p>
<p>A pod can then require properties with <b>node affinity</b>, an expression over those labels:</p>
<div class="eq">affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: nvidia.com/gpu.memory
          operator: Gt
          values: ["23000"]      # MiB</div>
<p>This reads: schedule only on nodes whose GPU-memory label is greater than 23,000 MiB, which keeps out a 16 GB card and admits a 24 GB one. <b>"Required"</b> is a hard rule (otherwise the pod stays Pending); a <i>preferred</i> variant would only bias the choice. The words <i>IgnoredDuringExecution</i> mean a pod already running is not evicted if labels change later.</p>
<p>Two cautions. The label <b>names and units</b> are <b>UNVERIFIED</b> here: read the real ones from a node with <code>kubectl get node &lt;gpu-node&gt; -o json</code> (exercise 4 has you record them). And a 24 GB card does not report exactly 24,000: the label is the card's actual memory, so choose the threshold from what you see.</p>`,
        scene(G) {
          G.text(24, 32, "affinity: gpu.memory Gt 23000 (illustrative labels)", { size: 13 });
          const nodes = [["node A", "L4-class", "≈ 24,000 MiB", true], ["node B", "T4-class", "≈ 15,000 MiB", false]];
          const bx = [];
          nodes.forEach(([n, p, m, ok], i) => {
            const x = 24 + i * 300;
            G.rect(x, 56, 270, 130, { stroke: ok ? "ok" : "hot", rx: 8 });
            G.text(x + 12, 80, n, { size: 14, color: ok ? "ok" : "hot" });
            G.label(x + 12, 104, "nvidia.com/gpu.product = " + p, { size: 11 });
            G.label(x + 12, 124, "nvidia.com/gpu.memory = " + m, { size: 11, color: "ink" });
            G.box(x + 12, 140, 246, 32, ok ? "vllm pod: schedulable" : "vllm pod: rejected (memory)", ok ? { fill: "ok", size: 12 } : { stroke: "hot", color: "hot", size: 12 });
          });
          G.text(24, 230, "how the labels got there", { size: 13 });
          const flow = [["GFD", "ok"], ["node labels", "w"], ["pod affinity", "k"], ["scheduler", "v"]];
          flow.forEach(([t, c], i) => { G.box(24 + i * 150, 248, 128, 40, t, { stroke: c, color: c, size: 13 }); if (i < 3) G.arrow(154 + i * 150, 268, 172 + i * 150, 268, { color: "muted", w: 2 }); });
          G.label(24, 322, "required...IgnoredDuringExecution: hard rule at scheduling time only", { size: 12 });
          G.label(24, 344, "label names and units: verify on your node with kubectl (UNVERIFIED here)", { size: 12, color: "v" });
          G.label(24, 366, "select by property, so manifests survive a change of instance type", { size: 12, color: "ok" });
          G.caption("GFD turns hardware into labels the scheduler can match");
        } },

      { rail: "probes", title: "Probes: surviving a model load that takes minutes",
        body: `<p>Kubernetes checks a container's health with <b>probes</b>, each answering a different question:</p>
<ul><li><b>startupProbe</b>: "has it finished starting?" Until it succeeds, the other two are <i>switched off</i>.</li><li><b>readinessProbe</b>: "may it receive traffic now?" Failing removes the pod from the Service's endpoints, nothing is restarted.</li><li><b>livenessProbe</b>: "is it stuck?" Failing enough times makes the kubelet <b>kill and restart</b> the container.</li></ul>
<p>A vLLM pod is not healthy at second 0: it loads weights, then captures CUDA graphs (P2.6), which takes minutes. Look at <code>vllm.yaml</code>:</p>
<div class="eq">startupProbe: /health every 10 s, 90 failures allowed
  → budget = 90 × 10 s = 900 s = 15 min
livenessProbe: /health every 20 s, 6 failures
  → killed after 6 × 20 s = 120 s of failures</div>
<p>Without the startup probe, liveness would begin immediately. If loading takes 4 minutes (an example; measure yours, P3.7), the first 120 s of failures kill the container, it restarts, starts loading again, and is killed again: <b>a restart loop that never becomes ready</b>. With the startup probe, liveness and readiness do not run until <code>/health</code> first answers; then readiness lets traffic in and liveness guards against a later hang (a 2-minute window).</p>
<p>The 15-minute budget is deliberately generous. Size it from your measured load time (P3.7), with margin. Too small and a slow cold start is killed; too large and a truly broken pod takes that long to be declared failed.</p>`,
        check: { q: "A vLLM pod needs 4 minutes to load. It has a liveness probe (6 failures at 20 s apart) but no startupProbe. What happens?",
          options: ["It starts normally; liveness only runs after it is Ready", "Liveness fails for about 120 s and the kubelet restarts the container, repeatedly: a restart loop", "Kubernetes waits for the model automatically"], answer: 1,
          why: "Without a startupProbe the liveness probe runs from the start. /health fails during the load, 6 × 20 s = 120 s later the container is killed, and the next attempt hits the same wall. The startup probe suspends liveness until the first success." },
        scene(G) {
          const x0 = 40, sc = 1.45, X = (t) => x0 + t * sc;
          G.text(24, 30, "one pod, model load = 240 s (example)", { size: 13 });
          [0, 60, 120, 180, 240, 300].forEach((t) => G.label(X(t), 392, t + " s", { anchor: "middle", size: 10 }));
          G.line(x0, 380, X(330), 380, { color: "line" });
          G.text(24, 66, "liveness only", { size: 13, color: "hot" });
          [0, 120, 240].forEach((t, i) => { if (t < 330) { G.rect(X(t), 80, 120 * sc - 2, 30, { fill: "v", rx: 3 }); G.label(X(t) + 4, 99, "loading", { size: 10, color: "bg" }); G.line(X(t + 120), 74, X(t + 120), 116, { color: "hot", w: 3 }); } });
          G.label(X(0), 134, "killed at ~120 s, restarts, loads again, killed again: never ready", { size: 12, color: "hot" });
          G.text(24, 206, "startupProbe + liveness + readiness", { size: 13, color: "ok" });
          G.rect(X(0), 222, 240 * sc, 30, { fill: "v", rx: 3 }); G.label(X(0) + 4, 241, "loading weights + capturing graphs", { size: 10, color: "bg" });
          G.rect(X(240), 222, 60 * sc, 30, { fill: "ok", rx: 3 }); G.label(X(240) + 4, 241, "Ready", { size: 10, color: "bg" });
          G.line(X(240), 214, X(240), 262, { color: "ok", w: 3 });
          G.label(X(0), 276, "startup probe: 10 s × 90 = 15 min budget; liveness + readiness wait for its first success", { size: 11, color: "ok" });
          G.label(X(240) + 8, 214, "first /health OK", { size: 11, color: "ok" });
          G.text(24, 330, "startup: starting?   readiness: traffic now?   liveness: stuck?", { size: 13, color: "ink" });
          G.label(24, 356, "load time of 240 s is an example: measure yours in P3.7", { size: 11 });
          G.caption("the startup probe holds liveness off until the model is loaded");
        } },

      { rail: "rollouts & drains", title: "Rollouts and shutdowns that don't cut streams",
        body: `<p>Two moments put live streams at risk: a rollout (a new version replaces the old pod) and any pod deletion. The settings in <code>vllm.yaml</code> handle both.</p>
<p><b>Rollout.</b> <code>maxUnavailable: 0, maxSurge: 1</code> means: first create <i>one extra</i> new pod, wait until it is Ready, only then remove an old one, so serving capacity never dips. A catch for GPUs: with one GPU per node, the extra pod needs a <i>second</i> GPU, so on a scale-from-zero pool the surge pod waits for a brand-new node (minutes, step 11). Plan the rollout time and cost accordingly.</p>
<p><b>Deletion.</b> When a pod is deleted, two things start at once, with no ordering between them: (1) the control plane <b>removes the pod from the Service endpoints</b>, a change that takes time to reach every proxy and load balancer; and (2) the kubelet runs the container's <b>preStop</b> hook, then sends <code>SIGTERM</code>. If SIGTERM came first, the server would begin shutting down while requests were still being routed to it. The fix is a deliberate pause:</p>
<div class="eq">t = 0     pod deleted: endpoints update starts,
          preStop "sleep 20" starts (still serving)
t = 20    preStop ends → SIGTERM → vLLM stops taking
          new requests and finishes in-flight streams
t = 120   terminationGracePeriodSeconds: SIGKILL</div>
<p>The grace period counts from the deletion, <i>including</i> the preStop time, so streams get 120 − 20 = 100 s to finish after SIGTERM. A <b>PodDisruptionBudget</b> (<code>maxUnavailable: 1</code>) limits how many pods a voluntary drain (such as upgrading a node) may take down at once; with a single replica it does not stop the drain, it starts to matter with several. Exercise 3 measures dropped streams through a rollout with and without the preStop.</p>`,
        check: { q: "Why does vllm.yaml run preStop: sleep 20 before the server gets SIGTERM?",
          options: ["To give the GPU time to cool down", "Endpoint removal and shutdown start at the same time; the pause keeps the pod serving until proxies stop sending it new requests", "Kubernetes requires a 20-second delay"], answer: 1,
          why: "Removing the pod from the Service endpoints is asynchronous and takes time to propagate. A short sleep keeps the process fully available until traffic has stopped arriving, and only then does SIGTERM let it finish in-flight streams." },
        scene(G) {
          const x0 = 30, sc = 4.2, X = (t) => x0 + t * sc;
          G.text(24, 30, "what happens after kubectl delete pod (vllm.yaml values)", { size: 13 });
          [0, 20, 60, 100, 120].forEach((t) => G.label(X(t), 366, t + " s", { anchor: "middle", size: 10 }));
          G.line(x0, 354, X(125), 354, { color: "line" });
          G.rect(X(0), 60, X(5) - X(0), 28, { fill: "v", rx: 3 }); G.label(X(5) + 6, 79, "endpoints update reaches proxies (assume a few seconds)", { size: 11, color: "v" });
          G.rect(X(0), 110, X(20) - X(0), 28, { fill: "k", rx: 3 }); G.label(X(20) + 8, 129, "preStop sleep 20: still serving", { size: 11, color: "k" });
          G.rect(X(20), 160, X(100) - X(20), 28, { fill: "ok", rx: 3 }); G.label(X(20) + 6, 179, "SIGTERM: finish in-flight streams (up to 100 s)", { size: 11, color: "bg" });
          G.line(X(20), 100, X(20), 200, { color: "ink", dash: "3 3", w: 1 }); G.label(X(20) + 4, 214, "SIGTERM", { size: 11 });
          G.line(X(120), 56, X(120), 354, { color: "hot", w: 3 }); G.label(X(120) - 6, 228, "SIGKILL", { size: 12, color: "hot", anchor: "end" });
          G.label(X(120) - 6, 244, "grace period 120 s", { size: 11, color: "hot", anchor: "end" });
          G.text(24, 290, "without the sleep: SIGTERM at 0 while requests still arrive", { size: 13, color: "hot" });
          G.label(24, 314, "grace counts from deletion, preStop included", { size: 12, color: "ink" });
          G.label(24, 336, "rollout: maxUnavailable 0, maxSurge 1 → new pod Ready first", { size: 12 });
          G.caption("keep serving until traffic stops, then drain, then exit");
        } },

      { rail: "hardening", title: "A pod spec that passes the security policy",
        body: `<p>A GPU server is also a network service, so its pod is locked down. Each line in <code>vllm.yaml</code> does a job, and <code>platform/deploy/policy.py</code> (T0, no cluster needed) fails the build if one is missing:</p>
<ul><li><b><code>runAsNonRoot: true</code>, <code>runAsUser: 10001</code></b>: a compromised process is not root inside the container.</li><li><b><code>allowPrivilegeEscalation: false</code></b> and <b><code>capabilities: drop: [ALL]</code></b>: no extra Linux privileges, no way to gain them.</li><li><b><code>readOnlyRootFilesystem: true</code></b>: the image's files cannot be modified; writable places are explicit volumes (<code>/tmp</code>, and <code>/dev/shm</code>).</li><li><b><code>seccompProfile: RuntimeDefault</code></b>: the runtime's default syscall filter. Together with the lines above, this satisfies the <b>restricted</b> Pod Security Standard that the namespace enforces.</li><li><b>The API key from a Secret</b> (<code>secretKeyRef</code>), never an inline value, and created outside git (P3.8).</li><li><b><code>/dev/shm</code> as a memory-backed emptyDir</b> (<code>sizeLimit: 2Gi</code>): PyTorch and NCCL use shared memory between processes, and the container default of 64 MiB breaks multi-process serving with cryptic errors.</li><li><b>Pinned image tag</b> (<code>0.31.0</code>, never <code>:latest</code>): the same bytes every time.</li></ul>
<p>The policy also checks GPU-specific rules: a GPU pod must tolerate the GPU taint, and must have a <code>startupProbe</code>. <code>test_manifests.py</code> runs it over every Deployment, and a second test feeds it a deliberately bad pod to prove the checks fire. Exercise 1 adds one more rule (every container sets a CPU <i>request</i>, or the scheduler may overpack a node and starve the tokenizer threads) with a test.</p>`,
        scene(G) {
          G.text(24, 32, "vllm.yaml spec lines → policy.py check", { size: 13 });
          const rows = [["runAsNonRoot, runAsUser 10001", "runAsNonRoot"], ["allowPrivilegeEscalation: false", "no escalation"], ["capabilities drop [ALL]", "drop ALL"], ["readOnlyRootFilesystem: true", "read-only root FS"], ["API key via secretKeyRef", "no literal *KEY / *TOKEN"], ["image tag 0.31.0", "no :latest"], ["startupProbe + readinessProbe", "probes present"], ["toleration nvidia.com/gpu", "GPU pods tolerate taint"]];
          const out = [];
          rows.forEach(([a, b], i) => {
            const y = 54 + i * 40;
            out.push(G.box(24, y, 290, 30, a, { stroke: "k", color: "k", size: 11 }));
            G.arrow(316, y + 15, 346, y + 15, { color: "muted" });
            out.push(G.box(348, y, 200, 30, b, { stroke: "ok", color: "ok", size: 11 }));
            G.text(566, y + 20, "✓", { size: 16, color: "ok" });
          });
          G.label(24, 392, "uv run python platform/deploy/policy.py: no cluster needed", { size: 12, color: "ink" });
          G.from(out, { opacity: 0, stagger: 0.05, duration: 0.2 });
          G.caption("every line is a rule a script can check before apply");
        } },

      { rail: "cold start", title: "Putting it together: the timeline of a cold GPU pod",
        body: `<p>Now you can predict what happens when you apply <code>vllm.yaml</code> to a cluster whose GPU pool is at 0 nodes. In order:</p>
<ol><li><b>Pending, unschedulable.</b> No node has an allocatable GPU. The pod waits; this event is what triggers scale-up.</li><li><b>The autoscaler adds a node.</b> It sees the pending pod needs a GPU node that would be allowed by the taint and selector, and asks AWS to start one (P3.6 builds the autoscaler). Roughly 1 to 3 minutes.</li><li><b>The node joins</b> and the Operator's pods start; the device plugin advertises <code>nvidia.com/gpu</code> and GFD labels the node. Roughly another minute.</li><li><b>The pod is scheduled</b> and the <b>image is pulled</b> (P3.1: size over network).</li><li><b>Weights load</b> (P3.7) from the volume or S3.</li><li><b>CUDA graph capture</b> (P2.6) and warm-up. The startup probe is still failing the whole time.</li><li><b>Ready.</b> <code>/health</code> answers, the Service adds the pod to its endpoints, traffic flows.</li></ol>
<p>The first two ranges come from the module README as rough expectations. Steps 4 to 6 depend on your image, weights and model, and are exactly what you measure: <code>kubectl get events -n s2s --sort-by=.lastTimestamp</code>. Keep your timeline. P3.6 (autoscaling and cold starts) starts from it, and it is the number your latency SLO has to absorb the first time a user hits a scaled-to-zero model.</p>
<p>On a laptop (T0) the same machinery runs on kind with mockllm: <code>kubectl apply -k platform/deploy/overlays/kind</code> puts two mockllm pods on the node labelled <code>s2s/pool=gpu</code>, with the same probes and taint, and exercise 2 makes the mock take 30 s to become healthy so you can watch the endpoint appear only when it is ready.</p>`,
        scene(G) {
          G.text(24, 30, "apply → Ready, GPU pool at 0 (durations from the README are rough)", { size: 13 });
          const ev = [["Pending: no GPU node", "v", "the trigger"], ["autoscaler adds a node", "w", "≈ 1–3 min"], ["node joins; device plugin advertises GPU", "w", "≈ 1 min"], ["image pull", "k", "measure (P3.1)"], ["weights load", "k", "measure (P3.7)"], ["graph capture + warm-up", "k", "measure"], ["Ready: endpoints updated", "ok", "traffic flows"]];
          const bx = ev.map(([t, c, d], i) => {
            const y = 50 + i * 48;
            G.text(24, y + 18, String(i + 1), { size: 14, color: c });
            const b = G.box(46, y, 360, 34, t, { fill: i === 6 ? "ok" : undefined, stroke: i === 6 ? undefined : c, color: i === 6 ? "bg" : c, size: 12 });
            G.text(424, y + 22, d, { size: 12, color: c });
            if (i < 6) G.arrow(226, y + 36, 226, y + 47, { color: "muted", w: 1.5 });
            return b;
          });
          G.label(24, 396, "kubectl get events -n s2s --sort-by=.lastTimestamp: record your own", { size: 12, color: "ink" });
          G.from(bx, { opacity: 0, stagger: 0.1, duration: 0.25 });
          G.caption("seven events; only your measurements fill in the minutes");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Kubernetes keeps declared state true with reconcile loops. Pods are disposable, Deployments keep N of them, and Services give them a stable name over Ready pods chosen by label.</li>
<li>Requests reserve capacity at scheduling time, limits cap at run time. A GPU is an extended resource set in limits, whole units only.</li>
<li>Taints repel, tolerations permit, selectors attract: a dedicated GPU pool needs all three. GFD labels plus node affinity select by GPU properties.</li>
<li>The device plugin makes nodes advertise GPUs; the Operator installs and runs its parts, and on the EKS NVIDIA AMI only the driver and toolkit are off.</li>
<li>Startup, readiness and liveness probes answer three questions. A startup probe protects a model load from the liveness probe. preStop plus a grace period lets streams drain.</li>
<li>A cold GPU pod goes Pending, node added, device plugin up, image pull, weights, graphs, Ready. Measure each step.</li></ul>`,
    sim: {
      title: "Does a pod delete cut streams?",
      intro: "A pod is deleted while it serves streaming requests. Endpoint removal and the preStop hook start together; SIGTERM follows the preStop; the container is killed when the grace period ends. Set the preStop sleep, the grace period, how long endpoint removal takes to reach the proxies (an assumption: measure yours), and the stream length. The model assumes fixed-length streams arriving at a steady rate (Little's law).",
      controls: [
        { id: "sleep", label: "preStop sleep, seconds", min: 0, max: 60, value: 20 },
        { id: "grace", label: "grace period, seconds", min: 1, max: 300, value: 120 },
        { id: "prop", label: "endpoint removal reaches proxies after, seconds (assumption)", min: 0, max: 40, value: 5 },
        { id: "len", label: "stream duration, seconds", min: 1, max: 200, value: 30 },
        { id: "rate", label: "requests per second to this pod", min: 0, max: 20, value: 5 },
      ],
      height: 250,
      draw(G, v) {
        const m = drainModel(v), T = Math.max(v.grace * 1.15, m.drainEnd * 1.05, 60), x0 = 30, W = 590, X = (t) => x0 + (W * t) / T;
        G.label(10, 14, "seconds after deletion", { size: 11 });
        G.rect(X(0), 30, X(v.prop) - X(0), 28, { fill: "v", rx: 3 }); G.label(X(v.prop) + 6, 49, "traffic still routed to this pod", { size: 10, color: "v" });
        G.rect(X(0), 70, Math.max(1, X(v.sleep) - X(0)), 28, { fill: "k", rx: 3 }); if (v.sleep > 0) G.label(X(v.sleep) + 6, 89, "preStop sleep (still serving)", { size: 10, color: "k" });
        const endShown = Math.min(m.drainEnd, T), killed = m.drainEnd > v.grace;
        if (m.drainEnd > v.sleep) {
          G.rect(X(v.sleep), 110, X(Math.min(m.drainEnd, v.grace)) - X(v.sleep), 28, { fill: "ok", rx: 3 }); G.label(X(Math.min(m.drainEnd, v.grace)) + 6, 129, "SIGTERM: draining", { size: 10, color: "ok" });
          if (killed) G.rect(X(v.grace), 110, X(endShown) - X(v.grace), 28, { fill: "hot", rx: 3, opacity: 0.55 });
        }
        G.line(X(v.grace), 24, X(v.grace), 160, { color: "hot", w: 3 }); G.label(X(v.grace) - 4, 176, "SIGKILL", { size: 11, color: "hot", anchor: "end" });
        G.line(X(v.sleep), 64, X(v.sleep), 142, { color: "ink", dash: "3 3", w: 1 }); G.label(X(v.sleep) + 4, 156, "SIGTERM", { size: 10 });
        if (v.prop > v.sleep) { G.rect(X(v.sleep), 30, X(v.prop) - X(v.sleep), 28, { stroke: "hot", rx: 3, sw: 2 }); G.label(X(v.sleep), 22, "requests refused here", { size: 10, color: "hot" }); }
        [0, Math.round(T / 2), Math.round(T)].forEach((t) => G.label(X(t), 214, t + " s", { anchor: "middle", size: 10 }));
        G.line(x0, 202, x0 + W, 202, { color: "line" });
        const ok = !m.killed && m.refused === 0;
        return [
          { title: "Timeline", rows: [["last request accepted at", F.num(m.lastAccepted) + " s"], ["drain finishes at", F.num(m.drainEnd) + " s"], ["killed at", v.grace + " s"]],
            chip: [ok, ok ? "clean shutdown" : m.killed ? "streams cut by SIGKILL" : "requests refused after SIGTERM"] },
          { title: "Damage (approximate)", rows: [["requests refused (SIGTERM before routing stops)", m.refused.toFixed(0)], ["streams cut by SIGKILL", m.cut.toFixed(0)]],
            html: `<p class="note">Rules: new requests keep arriving until endpoint removal reaches the proxies. SIGTERM starts at the end of the preStop sleep and stops new intake; requests arriving between SIGTERM and endpoint removal are refused. The drain ends when the last accepted stream finishes. The grace period counts from deletion and includes the preStop. The vllm.yaml values are sleep 20, grace 120. A model of the sequence, not a measurement.</p>` },
        ];
      },
    },
    practice: {
      intro: "Exercises 1–3 are T0: kind (Docker) with mockllm standing in for the GPU backend, no cloud account. Exercise 4 needs an EKS cluster with a real GPU node, brought up from P3.2 (make up && make addons) and torn down with make down: it costs money, so read the cost and teardown notes in infra/aws/eks first. No endpoint is exposed publicly, and the API key lives in a Secret, never in the repo.",
      items: [
        { title: "Manifests pass policy (and kubeconform)", tier: "T0 · easy", goal: "Keep policy.py green over every Deployment, run kubeconform over the kustomize output for both overlays, then add one rule with a test: every container sets a CPU request.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.3-kubernetes-and-gpu-operator/exercises/test_manifests.py" },
        { title: "Readiness gates on warm-up", tier: "T0 · medium", goal: "Add MOCKLLM_STARTUP_DELAY_S so /health returns 503 until N seconds after start, deploy with 30 s, and watch the pod IP appear in the endpoints only when ready. Then remove the readinessProbe and explain the errors.",
          cmd: "kubectl -n s2s get endpoints mockllm -w" },
        { title: "preStop drain: zero dropped streams during a rollout", tier: "T0 · medium", goal: "Keep 16 streams open for 60 s through kubectl rollout restart, from a client inside the cluster. Expect 0 errors with the shipped preStop and a grace period; then remove both and count and explain each error kind.",
          cmd: "uv run python course/P3-deployment-and-infra/P3.3-kubernetes-and-gpu-operator/exercises/drain_test.py --url http://mockllm.s2s:8001" },
        { title: "Schedule only on GPUs with 24 GB or more", tier: "T3 · medium", goal: "Dump a GPU node's nvidia.com labels and record the real names and units. Add a nodeAffinity in an overlay using the memory label, then show the pod refuses to land on a node with a smaller GPU.",
          cmd: "kubectl get node <gpu-node> -o json | jq '.metadata.labels | with_entries(select(.key|startswith(\"nvidia.com\")))'" },
      ],
      labs: [
        { label: "Read the vLLM Deployment, Service and PDB with this lesson beside it", path: "platform/deploy/base/vllm.yaml" },
        { label: "The mockllm twin used on kind, and the 3-node kind config with a labelled pool", path: "platform/deploy/base/mockllm.yaml, platform/deploy/kind/kind-config.yaml" },
        { label: "GPU Operator Helm values for EKS AL2023 NVIDIA nodes (keys UNVERIFIED)", path: "platform/deploy/gpu-operator-values.yaml" },
        { label: "The policy checks that run over every Deployment", path: "platform/deploy/policy.py" },
        { label: "Walkthrough: kind up, apply the kind overlay, curl mockllm, run policy.py", path: "course/P3-deployment-and-infra/P3.3-kubernetes-and-gpu-operator/README.md" },
        { label: "Animation: request path, cold vs warm", path: "animations/p3-request-path.html" },
        { label: "Pinned: GPU Operator v26.7.1, device plugin v0.20.1, dcgm-exporter 4.8.4", path: "SOURCES.md" },
      ],
    },
  });
})();
