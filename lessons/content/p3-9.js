/* P3.9 — A second cloud (GKE): the mapping, what stays the same, quota, spot, and comparing clouds on $/1M tokens at the SLO. */
(function () {
  const F = S2S.fmt;

  /* $ per 1M output tokens = ($/GPU-h after discount) ÷ (tokens/h) × 1e6, then ÷ utilisation */
  const perM = (price, disc, knee, util) => (price * (1 - disc / 100)) / (knee * 3600) * 1e6 / util;

  S2S.lesson({
    id: "p3-9", n: "P3.9", title: "A second cloud",
    subtitle: "Deployment and infra · first principles · T0 for Terraform validate and overlays, T3 to deploy on GKE",
    kicker: "Lesson · ≈ 40 min",
    headline: "Same platform, different cloud: change the floor, not the house",
    intro: `<p>Everything in P3 so far ran on AWS. This lesson rebuilds the same platform on a second provider, Google Cloud's GKE, and asks what really has to change. The answer is small if you were disciplined: the cloud-specific parts live in <code>infra/</code> and one Kustomize overlay, and everything above them (manifests, gateway, observability, autoscaler, tenancy) stays identical. You will map each AWS building block to its GCP counterpart and find the places where the <i>semantics</i> differ, not just the names, plan for the GPU quota that is often the slowest step, see why a shorter spot notice changes your grace period, and learn to compare clouds on <b>dollars per million tokens at your SLO</b> instead of on the list price of a GPU hour.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "3 exercises"],
    legend: [["v", "AWS"], ["k", "GCP"], ["ok", "same on both"], ["hot", "differs"]],
    prev: "p3-8", next: "p4-1",
    steps: [
      /* 1 */
      { rail: "why a second", title: "Why run on two clouds, and what must not change",
        body: `<p>A second cloud costs effort, so it needs a reason. The usual ones:</p>
<ul><li><b>Capacity.</b> GPUs are scarce. When one provider has no L4s in your region, another might.</li>
<li><b>Price.</b> The same work can be cheaper elsewhere, but only if you compare correctly (step 9).</li>
<li><b>Resilience.</b> A regional outage, or a provider quota freeze, should not stop you.</li>
<li><b>Customers.</b> Some require a specific cloud or region.</li></ul>
<p>The risk is the opposite: two copies of the platform that drift apart, double the maintenance. The cure is a rule about <b>where</b> cloud knowledge may live. Picture the platform as layers. Everything above Kubernetes (the vLLM Deployment, the gateway, the observability rules, the autoscaler, the tenancy policies) speaks only Kubernetes. Only the thin layer below it, <b>infra</b> (cluster, nodes, identity, networking, quota), knows which cloud it is. Plus one <b>overlay</b> for the few values a manifest cannot avoid, such as the image registry.</p>
<p>The rule to hold on to: if you catch yourself writing <code>if cloud == "gcp"</code> in platform code, stop. It belongs in an overlay or in <code>infra/</code>.</p>`,
        scene(G) {
          G.text(24, 34, "the platform is identical; only the floor changes", { size: 13 });
          const top = ["vLLM Deployment · gateway", "observability · autoscaler", "tenancy policies · base manifests"];
          const g = top.map((t, i) => G.box(24, 56 + i * 52, 592, 42, t, { fill: "ok", size: 13 }));
          G.text(24, 232, "same Kubernetes API", { size: 12, color: "ok" });
          G.box(24, 244, 592, 36, "one Kustomize overlay per cloud: image registry, and little else", { stroke: "hot", color: "hot", size: 12 });
          const a = G.box(24, 300, 286, 78, "", { stroke: "v" }); G.text(167, 326, "infra/aws/", { anchor: "middle", size: 14, color: "v" }); G.label(167, 348, "EKS · node groups · IRSA · NAT", { anchor: "middle", size: 11 });
          const b = G.box(330, 300, 286, 78, "", { stroke: "k" }); G.text(473, 326, "infra/gcp/", { anchor: "middle", size: 14, color: "k" }); G.label(473, 348, "GKE · node pools · Workload ID · NAT", { anchor: "middle", size: 11 });
          G.label(24, 402, "cloud knowledge lives in infra/ and one overlay, nowhere else", { color: "ink" });
          G.from(g, { opacity: 0, y: -10, stagger: 0.12, duration: 0.3 }); G.from([a, b], { opacity: 0, y: 14, stagger: 0.15, duration: 0.3, delay: 0.5 });
          G.caption("two floors, one house");
        } },

      /* 2 */
      { rail: "the mapping", title: "Every AWS building block has a GCP counterpart",
        body: `<p>The first job is translation. Most concepts map one to one; the names differ. (GKE shapes and field names here are <b>UNVERIFIED</b>, checked against the <code>hashicorp/google</code> provider docs only when you run <code>terraform validate</code>.)</p>
<table><tr><th>concern</th><th>AWS (P3.2)</th><th>GCP (this lesson)</th></tr>
<tr><td>Kubernetes</td><td>EKS + managed node groups</td><td>GKE Standard + node pools</td></tr>
<tr><td>GPU node</td><td><code>g6.xlarge</code> (1× L4)</td><td><code>g2-standard-4</code> (1× L4)</td></tr>
<tr><td>GPU software</td><td>driver from the AMI, GPU Operator (P3.3)</td><td>GKE installs driver and device plugin</td></tr>
<tr><td>pod to cloud identity</td><td>IRSA / EKS Pod Identity</td><td>Workload Identity Federation for GKE</td></tr>
<tr><td>node metadata protection</td><td>IMDSv2, hop limit 1</td><td><code>GKE_METADATA</code> mode</td></tr>
<tr><td>object storage</td><td>S3</td><td>GCS</td></tr>
<tr><td>private path to storage</td><td>S3 gateway endpoint</td><td>Private Google Access</td></tr>
<tr><td>internet egress</td><td>NAT gateway</td><td>Cloud NAT</td></tr>
<tr><td>spot notice</td><td>about 2 minutes</td><td>about 30 seconds (UNVERIFIED)</td></tr></table>
<p>The next steps look at the rows where the <i>behaviour</i> differs, because those are the ones that cause bugs.</p>`,
        scene(G) {
          const rows = [["EKS + node groups", "GKE + node pools"], ["g6.xlarge (1× L4)", "g2-standard-4 (1× L4)"], ["GPU Operator / AMI driver", "GKE-managed driver"], ["IRSA / Pod Identity", "Workload Identity"], ["IMDSv2, hop limit 1", "GKE_METADATA"], ["S3 + gateway endpoint", "GCS + Private Google Access"], ["NAT gateway", "Cloud NAT"], ["Spot: ~2 min notice", "Spot VMs: ~30 s (UNVERIFIED)"]];
          G.text(24, 36, "AWS", { size: 14, color: "v" }); G.text(366, 36, "GCP", { size: 14, color: "k" });
          const g = rows.map(([a, b], i) => {
            const y = 48 + i * 42, gg = G.group(), diff = i === 2 || i === 7;
            G.box(24, y, 240, 36, a, { stroke: "v", color: "ink", size: 12, parent: gg });
            G.arrow(268, y + 18, 358, y + 18, { color: diff ? "hot" : "muted", w: 2, parent: gg });
            G.box(362, y, 254, 36, b, { stroke: diff ? "hot" : "k", color: "ink", size: 12, parent: gg });
            return gg;
          });
          G.label(24, 402, "red arrows: the behaviour differs, not just the name (next steps)", { size: 11, color: "hot" });
          G.from(g, { opacity: 0, x: -12, stagger: 0.08, duration: 0.25 });
          G.caption("same concepts, different names, a few different behaviours");
        } },

      /* 3 */
      { rail: "GPU software", title: "On GKE the cloud installs the GPU driver, so skip the GPU Operator",
        body: `<p>In P3.3 the GPU Operator installed the container toolkit, the device plugin and the monitoring exporter on every GPU node, and drivers came with the node image. A GPU pod then asked for <code>nvidia.com/gpu: 1</code> and the scheduler found a node that advertised it.</p>
<p>GKE does that part itself. When you create a GPU node pool you set <code>gpu_driver_installation_config</code> and GKE installs the NVIDIA driver and its own device plugin. The node advertises <code>nvidia.com/gpu</code> and the same pod spec works. So <b>the GPU Operator is not installed on GKE</b>. If you install it anyway, two driver stacks fight over the same node. DCGM metrics then come from GKE's managed DCGM or from your own exporter.</p>
<p>Two things stay compatible on purpose. GKE adds the <code>nvidia.com/gpu</code> taint to GPU nodes automatically; on EKS you add it yourself in Terraform. It is the <b>same key</b>, so the same toleration in <code>platform/deploy/base/vllm.yaml</code> works on both. And both Terraform roots label the GPU pool <code>s2s/pool: gpu</code>, which is what the manifests select on, so the manifests never need to know which cloud they are on.</p>
<p>One thing does not carry over. Node shapes differ in vCPU and memory per GPU (<code>g2-standard-4</code> versus <code>g6.xlarge</code>), so re-check that the pod's CPU and memory requests (3 CPU, 12 GiB in <code>vllm.yaml</code>) still fit on the new shape. These shapes are UNVERIFIED here.</p>`,
        check: { q: "Why does the GKE path not install the NVIDIA GPU Operator?",
          options: ["The GPU Operator does not support Kubernetes on GCP", "GKE installs the driver and device plugin itself, and a second driver stack would conflict", "GKE nodes have no GPUs"], answer: 1,
          why: "A GKE GPU node pool with gpu_driver_installation_config gets the NVIDIA driver and device plugin from GKE. Adding the Operator on top installs a competing set, which breaks the node. The pod spec stays the same because the resource name and taint key match." },
        scene(G) {
          const col = (x, title, c, layers) => {
            G.text(x + 8, 40, title, { size: 14, color: c });
            return layers.map(([t, k, own], i) => G.box(x, 56 + i * 56, 280, 46, t, own ? { fill: k, size: 12 } : { stroke: k, color: "ink", size: 12, dash: "5 4" }));
          };
          col(24, "EKS", "v", [["your pod: nvidia.com/gpu: 1", "ok", true], ["GPU Operator: toolkit + plugin", "v", true], ["driver (from the node image)", "w", true], ["GPU", "muted", false]]);
          col(336, "GKE", "k", [["your pod: nvidia.com/gpu: 1", "ok", true], ["GKE device plugin (managed)", "k", true], ["driver (installed by GKE)", "k", true], ["GPU", "muted", false]]);
          G.box(336, 288, 280, 30, "no GPU Operator here", { stroke: "hot", color: "hot", size: 12 });
          G.rect(24, 336, 592, 70, { stroke: "line", rx: 10 });
          G.text(40, 362, "same: nvidia.com/gpu resource, taint key, s2s/pool: gpu label", { size: 12, color: "ok" });
          G.label(40, 386, "differs: who installs the driver · re-check CPU/RAM per GPU", { size: 11 });
          G.caption("the pod spec is portable because the contract is");
        } },

      /* 4 */
      { rail: "identity", title: "Pod identity and metadata: the same idea with different wiring",
        body: `<p>P3.8 gave each workload its own least-privilege cloud role and blocked the node's metadata service. Both clouds offer both, under other names.</p>
<ul><li><b>Pod to cloud identity.</b> On EKS a Kubernetes ServiceAccount is tied to an IAM role (IRSA or Pod Identity). On GKE, <b>Workload Identity Federation</b> ties the ServiceAccount to an IAM principal, either by annotating it with a Google service account or by granting IAM directly to the ServiceAccount's principal. The concept is identical: one workload, one narrow role, scoped to one bucket or one secret. The wiring differs.</li>
<li><b>Protecting node credentials.</b> The cloud's metadata service hands the <i>node's</i> credentials to any process that can reach it, including a compromised pod. AWS answers with IMDSv2 and a hop limit of 1, which stops a pod (one network hop further away) from getting a token. GKE answers with <code>GKE_METADATA</code> mode on each node pool, which serves pods the Workload Identity view instead of the node's. Both stop credential theft; the Terraform setting is different. <code>test_gke_tf.py</code> checks that every GKE pool has it.</li></ul>
<p>Practically: the NetworkPolicy from P3.8 that blocks <code>169.254.169.254</code> is a second layer on both clouds, because the metadata server is reached at that address on GCP too.</p>`,
        scene(G) {
          G.text(24, 34, "a pod needs a cloud credential: one narrow role, never the node's", { size: 12 });
          const row = (y, title, c, a, b, d) => {
            G.text(24, y, title, { size: 14, color: c });
            G.box(24, y + 12, 140, 44, "pod / SA", { stroke: c, color: "ink", size: 12 });
            G.arrow(166, y + 34, 232, y + 34, { color: c, w: 2 });
            G.box(234, y + 12, 170, 44, a, { stroke: c, color: "ink", size: 11 });
            G.arrow(406, y + 34, 472, y + 34, { color: c, w: 2 });
            G.box(474, y + 12, 142, 44, b, { stroke: "ok", color: "ok", size: 11 });
            G.label(24, y + 76, d, { size: 11 });
          };
          row(66, "AWS", "v", "IRSA → IAM role", "one S3 prefix", "node credentials: IMDSv2 with hop limit 1 blocks pods");
          row(196, "GCP", "k", "Workload ID → IAM", "one GCS bucket", "node credentials: GKE_METADATA serves pods their own identity");
          G.rect(24, 322, 592, 84, { stroke: "line", rx: 10 });
          G.text(40, 348, "same concept: one workload, one narrow role", { size: 13, color: "ok" });
          G.label(40, 372, "different wiring: annotate the SA / grant IAM to its principal", {});
          G.label(40, 392, "keep the NetworkPolicy that blocks 169.254.169.254 on both clouds", {});
          G.caption("identical principle, different knobs");
        } },

      /* 5 */
      { rail: "network & cost", title: "Private paths to storage, NAT meters, and what this costs",
        body: `<p>Weights live in object storage (P3.7) and nodes sit in private subnets without public addresses. How do the bytes get from S3 or GCS to the node? Two paths:</p>
<ul><li>Through a <b>NAT</b> (AWS NAT gateway, GCP Cloud NAT), which gives private nodes access to the internet. Both bill <b>per hour and per GB processed</b>.</li>
<li>Through a <b>private path to the storage service</b>: an S3 gateway endpoint on AWS, Private Google Access on GCP. The traffic stays on the provider's network and skips NAT processing charges.</li></ul>
<p>Worked example with a made-up round price, not a quote (look yours up): if NAT processing cost <b>$0.05 per GB</b> and 100 node starts a month each pull 16 GB of weights, that is 1,600 GB × $0.05 = <b>$80 a month</b> for moving the same bytes you could have moved for no processing fee. That is why exercise 2 in P3.2 added the S3 endpoint.</p>
<p><b>What the GKE module costs (infra/gcp/gke).</b> A cluster management fee (a zonal cluster may be covered by a free tier, UNVERIFIED), two <code>e2-standard-4</code> system nodes, <b>GPU node-hours</b> while the L4 pool is scaled up, and Cloud NAT. Look all of them up in your region. <b>Auto-stop:</b> the GPU pool is min 0 and scales to zero when no GPU pod is pending. The system pool, NAT and cluster fee keep billing until <code>make down</code>, and the billing budget alerts at 50, 80 and 100% (budgets <b>alert, they do not stop spending</b>). <b>Teardown:</b> <code>make down</code>. <b>Access:</b> nodes are private, the control plane is open only to your <code>master_authorized_cidrs</code> (never <code>0.0.0.0/0</code>, which <code>test_gke_tf.py</code> checks), and Services stay ClusterIP, so there is no unauthenticated public endpoint.</p>`,
        scene(G) {
          G.text(24, 34, "weights from object storage to a private node", { size: 13 });
          G.box(24, 60, 140, 50, "S3 / GCS", { stroke: "ok", color: "ok", size: 13 }); G.box(476, 60, 140, 50, "private node", { stroke: "k", color: "k", size: 13 });
          G.box(244, 60, 160, 50, "NAT: per hour + per GB", { stroke: "hot", color: "hot", size: 11 });
          G.arrow(166, 85, 240, 85, { color: "hot", w: 2 }); G.arrow(406, 85, 472, 85, { color: "hot", w: 2 });
          G.label(24, 134, "via NAT: every GB is metered", { color: "hot", size: 11 });
          G.box(24, 170, 140, 50, "S3 / GCS", { stroke: "ok", color: "ok", size: 13 }); G.box(476, 170, 140, 50, "private node", { stroke: "k", color: "k", size: 13 });
          G.arrow(166, 195, 472, 195, { color: "ok", w: 3 }); G.text(320, 186, "gateway endpoint / Private Google Access", { anchor: "middle", size: 11, color: "ok" });
          G.label(24, 244, "private path: stays on the provider network, no NAT processing", { color: "ok", size: 11 });
          G.rect(24, 270, 592, 134, { stroke: "line", rx: 10 });
          G.text(40, 296, "example: 100 starts × 16 GB = 1,600 GB × $0.05/GB = $80/month", { size: 12 });
          G.label(40, 318, "(made-up round price: look up yours)", {});
          G.text(40, 348, "billing until make down: control plane, system nodes, NAT", { size: 12, color: "hot" });
          G.text(40, 372, "GPU pool min 0 stops its own cost · budgets alert, they do not stop", { size: 12, color: "ok" });
          G.caption("a private storage path removes a per-GB tax on every cold start");
        } },

      /* 6 */
      { rail: "quota", title: "The quota you do not have yet is often the slowest step",
        body: `<p>On a new account the number of GPUs you may run is often <b>zero</b>, and raising it is a request that a human approves, sometimes taking days. Plan it first, before any Terraform.</p>
<p>The two clouds count it differently:</p>
<ul><li><b>AWS</b> counts <b>vCPUs</b> in a family: "Running On-Demand G and VT instances" (and a separate Spot variant). A <code>g6.xlarge</code> has 4 vCPUs, so <i>gpu_max</i> nodes need 4 × <i>gpu_max</i> vCPUs of quota.</li>
<li><b>GCP</b> counts <b>GPUs</b> per model per region (<code>NVIDIA_L4_GPUS</code>) <i>and</i> a global limit (<code>GPUS_ALL_REGIONS</code>). Both must be at least <i>gpu_max</i>. The global limit starts at 0 on many new projects.</li></ul>
<div class="eq">gpu_max = 2 L4 nodes
AWS: 2 × 4 vCPU = 8 vCPU in "G and VT" (on-demand,
     plus the Spot quota if you use spot)
GCP: NVIDIA_L4_GPUS ≥ 2 in the region
     and GPUS_ALL_REGIONS ≥ 2</div>
<p>The common mistake is to write the Terraform, run <code>make up</code>, and learn from a quota error that the real work is a support ticket. Request quota days ahead, request a little above what you need, and note how long it took: that number belongs in your cross-cloud notes just like a price does.</p>`,
        check: { q: "You plan up to 3 L4 nodes on AWS (g6.xlarge, 4 vCPUs each) and on GCP. What quota do you request?",
          options: ["AWS: 3 vCPUs. GCP: 3 vCPUs", "AWS: 12 vCPUs in the G and VT family. GCP: 3 NVIDIA_L4_GPUS regionally and GPUS_ALL_REGIONS at least 3", "3 on both clouds, counted in nodes"], answer: 1,
          why: "AWS counts vCPUs in an instance family, so 3 nodes × 4 vCPU = 12. GCP counts GPUs, per model per region and globally, so 3 in each. Mixing the units is the usual reason a request is too small and the first make up fails." },
        scene(G) {
          G.text(24, 34, "gpu_max = 2 L4 nodes: what to request", { size: 13 });
          G.rect(24, 56, 280, 190, { stroke: "v", rx: 10 }); G.text(36, 80, "AWS: count vCPUs", { size: 14, color: "v" });
          G.box(40, 96, 248, 40, "G and VT on-demand quota", { stroke: "v", color: "ink", size: 12 });
          G.text(40, 168, "2 nodes × 4 vCPU = 8 vCPU", { size: 14, color: "v" }); G.label(40, 192, "g6.xlarge = 4 vCPU, 1 L4", {}); G.label(40, 212, "+ Spot variant if you use spot", {});
          G.rect(336, 56, 280, 190, { stroke: "k", rx: 10 }); G.text(348, 80, "GCP: count GPUs", { size: 14, color: "k" });
          G.box(352, 96, 248, 40, "NVIDIA_L4_GPUS (regional) ≥ 2", { stroke: "k", color: "ink", size: 12 });
          G.box(352, 144, 248, 40, "GPUS_ALL_REGIONS (global) ≥ 2", { stroke: "k", color: "ink", size: 12 });
          G.label(352, 212, "global quota often starts at 0", { color: "hot" });
          G.rect(24, 270, 592, 134, { stroke: "hot", rx: 10, dash: "5 4" });
          G.text(40, 296, "approval can take days: file the request before you write Terraform", { size: 13, color: "hot" });
          G.label(40, 324, "ask for a little more than gpu_max", {}); G.label(40, 346, "note how long approval took: it is part of the comparison", {});
          G.label(40, 368, "the cluster is free of GPUs until quota exists: min-0 pool won't start", {});
          G.caption("quota is a calendar dependency, not a code one");
        } },

      /* 7 */
      { rail: "spot notice", title: "A shorter spot notice changes your grace period",
        body: `<p>P3.6 taught the spot recipe: on a reclaim notice, stop taking new requests, let in-flight streams finish within <code>terminationGracePeriodSeconds</code>, and retry the rest. That recipe was sized to AWS's two-minute notice (a grace period of about 90 s, leaving margin).</p>
<p>GCP Spot VMs give a much shorter warning, about <b>30 seconds</b> (<b>UNVERIFIED</b>: check the GKE and Compute Engine docs for your version). The grace period cannot exceed the notice, so it must shrink, say to 20 s, and the arithmetic of P3.6 changes. With responses of length <i>R</i> seconds the fraction of in-flight streams that get cut is about:</p>
<div class="eq">fraction cut ≈ max(0, 1 − grace ÷ R)

          AWS (grace 90)   GKE (grace 20)
R = 30 s         0               0.33
R = 60 s         0               0.67
R = 180 s        0.50            0.89</div>
<p>(Check: 1 − 20 ÷ 30 = 0.33; 1 − 20 ÷ 60 = 0.67; 1 − 90 ÷ 180 = 0.5; 1 − 20 ÷ 180 = 0.89.) So on GKE spot you expect <b>more mid-stream failures</b>. The mitigations: lower <code>terminationGracePeriodSeconds</code> and the <code>preStop</code> sleep to fit inside the notice; cap <code>max_tokens</code> on spot replicas so responses are short; and keep a larger on-demand share for streaming-heavy traffic.</p>
<p>The scheduler also needs to <i>prefer</i> spot without Karpenter-style weights. On GKE you add a second GPU pool with <code>spot = true</code> and a higher autoscaling max, then use <code>preferredDuringScheduling</code> node affinity on the spot node label (<b>UNVERIFIED</b> label name <code>cloud.google.com/gke-spot</code>). That is exercise 1's task.</p>`,
        check: { q: "GCP's spot notice is much shorter than AWS's. What do you change on the GKE spot pool?",
          options: ["Nothing: the AWS settings carry over", "A shorter grace period and preStop, a lower max response length on spot replicas, and more on-demand share for streaming traffic", "Disable the readiness probe so pods start faster"], answer: 1,
          why: "The grace period cannot outlast the notice. With a shorter one, long streams get cut more often (1 − grace ÷ R), so you shorten responses on spot replicas and keep more on-demand capacity for the traffic that cannot tolerate cuts." },
        scene(G) {
          const Rs = [30, 60, 180];
          G.text(24, 34, "share of in-flight streams cut when the notice lands", { size: 13 });
          const x0 = 70, base = 280, H = 190;
          G.axes(x0 - 10, base - H, 520, H, {});
          [0, 0.5, 1].forEach((f) => G.label(x0 - 16, base - H * f + 4, Math.round(f * 100) + "%", { anchor: "end", size: 10 }));
          const bars = [];
          Rs.forEach((R, i) => {
            const a = Math.max(0, 1 - 90 / R), b = Math.max(0, 1 - 20 / R), x = x0 + 20 + i * 170;
            bars.push(G.rect(x, base - H * Math.max(a, 0.004), 56, H * Math.max(a, 0.004), { fill: "v", rx: 3 })); bars.push(G.rect(x + 66, base - H * b, 56, H * b, { fill: "k", rx: 3 }));
            G.text(x + 28, base - H * a - 6, a.toFixed(2), { anchor: "middle", size: 12 }); G.text(x + 94, base - H * b - 6, b.toFixed(2), { anchor: "middle", size: 12 });
            G.label(x + 61, base + 18, "R = " + R + " s", { anchor: "middle", size: 12 });
          });
          G.rect(x0 + 8, 56, 12, 12, { fill: "v", rx: 2 }); G.label(x0 + 26, 67, "AWS, grace 90 s (notice ~2 min)", { size: 11 });
          G.rect(x0 + 270, 56, 12, 12, { fill: "k", rx: 2 }); G.label(x0 + 288, 67, "GKE, grace 20 s (notice ~30 s)", { size: 11 });
          G.rect(24, 330, 592, 76, { stroke: "line", rx: 10 });
          G.text(40, 356, "cut ≈ max(0, 1 − grace ÷ R)", { size: 13, color: "ink" }); G.label(40, 380, "notice lengths and the 30 s figure are UNVERIFIED: check the docs", {});
          G.from(bars, { attr: { height: 0 }, stagger: 0.1, duration: 0.4 });
          G.caption("a shorter notice cuts more long streams: shorten responses on spot");
        } },

      /* 8 */
      { rail: "overlays", title: "Keep the platform identical: one overlay, enforced by a test",
        body: `<p>How do you stop the two clouds drifting apart? With a <b>test</b>. Kustomize overlays (P3.3) let one base serve many environments. <code>platform/deploy/overlays/gke/kustomization.yaml</code> and <code>overlays/eks/kustomization.yaml</code> list the same <code>resources</code> and differ in one place, <code>images</code>: the registry the serving image comes from (an ECR repository versus an Artifact Registry path).</p>
<p><code>exercises/test_overlays.py</code> enforces it. It checks that both overlays have the <b>same resources</b>, the <b>same gateway config</b> (except one comment line), and <b>no <code>patches</code></b>, so the only difference is <code>images</code>. Try adding a GKE-only patch that changes the vLLM <code>nodeSelector</code> and watch it fail. That failure is the right outcome: the GPU pool label <code>s2s/pool: gpu</code> is set in <i>both</i> Terraform roots precisely so the manifests need not know the cloud.</p>
<p>Some differences are real, for example the StorageClass for the weights volume (gp3 on EKS, pd-balanced on GKE). The test-friendly home for those is a small <code>cloud.yaml</code> resource in each overlay, and the test is extended to allow exactly that one file to differ. Anything else belongs in <code>infra/</code>.</p>
<p>Likewise for infrastructure: <code>test_gke_tf.py</code> is a static policy check for <code>infra/gcp/gke</code> (private nodes, an authorised-networks list that is never <code>0.0.0.0/0</code>, a min-0 GPU pool, <code>GKE_METADATA</code> on every pool, a billing budget, <code>deletion_protection = false</code> so <code>make down</code> works) and <code>bash tools/ci_terraform.sh</code> runs <code>fmt</code> and <code>validate</code> with no credentials.</p>`,
        check: { q: "Where should cloud-specific differences live in this repository?",
          options: ["In the gateway and base manifests, behind if-statements", "In infra/<cloud>/ and one Kustomize overlay, never in platform code or base manifests", "In a separate fork of the whole repository per cloud"], answer: 1,
          why: "Confining differences to the infra layer and one small overlay keeps everything above it identical and testable. Branching on the cloud inside platform code or forking the repo makes the clouds drift, and every fix has to be made twice." },
        scene(G) {
          G.text(24, 34, "overlays/eks  vs  overlays/gke", { size: 13 });
          const rows = [["resources", "base, gateway.yaml, vllm.yaml", "same", "ok"], ["configMapGenerator", "gateway-config from gateway.yaml", "same", "ok"], ["patches", "none in either overlay", "same", "ok"], ["images", "ECR path  vs  Artifact Registry path", "differs", "hot"]];
          const g = rows.map(([k, v, t, c], i) => {
            const y = 56 + i * 62, gg = G.group();
            G.rect(24, y, 592, 50, { stroke: c, rx: 8, parent: gg });
            G.text(40, y + 22, k, { size: 13, color: c, parent: gg }); G.label(40, y + 40, v, { size: 11, parent: gg });
            G.box(520, y + 10, 84, 30, t, { fill: c, size: 12, parent: gg });
            return gg;
          });
          G.rect(24, 318, 592, 88, { stroke: "line", rx: 10 });
          G.text(40, 344, "test_overlays.py fails if anything but images differs", { size: 13, color: "ink" });
          G.label(40, 368, "real extras (a StorageClass) go in a cloud.yaml the test allows", {});
          G.label(40, 388, "same Terraform label s2s/pool: gpu keeps selectors cloud-free", {});
          G.from(g, { opacity: 0, x: -14, stagger: 0.15, duration: 0.3 });
          G.caption("a test turns \"keep it the same\" into a rule");
        } },

      /* 9 */
      { rail: "compare clouds", title: "Compare on dollars per million tokens at your SLO",
        body: `<p>The tempting comparison is the price of a GPU hour. It is the wrong one, because what you buy is <b>tokens</b>. Two things between the hourly price and the token price differ by cloud: the <b>throughput</b> the same GPU delivers (the knee you measured in #4: CPU per GPU, PCIe topology and driver version all matter), and the <b>utilisation</b> you achieve (P3.5, P3.6). The P3.5 identity gives the token price:</p>
<div class="eq">$ per 1M = ($/GPU-h × (1 − discount))
           ÷ (knee tokens/s × 3,600) × 1,000,000
           ÷ utilisation</div>
<p>Worked example with <b>round example numbers, not prices or measurements</b>. Cloud A: $1.00 per hour, knee 1,000 tokens/s. Cloud B: $0.90 per hour, knee 800 tokens/s.</p>
<div class="eq">A: 1.00 ÷ 3.6M × 1M = $0.278 per 1M at 100%
B: 0.90 ÷ 2.88M × 1M = $0.3125 per 1M at 100%
   B is 10% cheaper per hour, 12.5% dearer per token
at 30% utilisation both scale by 3.33:
   A $0.926   B $1.042
B with a 15% committed-use discount:
   0.3125 × 0.85 = $0.2656 → now cheaper than A</div>
<p>Three lessons: the list price ranking can be backwards; utilisation moves both but not their ratio; and a discount can flip the winner. <code>exercises/compare_clouds.py</code> builds the table from <b>your</b> prices (it refuses prices without a date and a source) and your #4 results. The simulator does the same sums.</p>`,
        check: { q: "Cloud B's GPU hour is 10% cheaper than cloud A's, but its measured knee is 20% lower. Which is cheaper per token?",
          options: ["B, because the hourly price is lower", "A: 0.9 ÷ 0.8 = 1.125, so B costs 12.5% more per token", "They cost the same"], answer: 1,
          why: "Cost per token is price ÷ throughput. B pays 0.9 of A's price for 0.8 of A's throughput: 0.9 ÷ 0.8 = 1.125 times A's cost per token. The cheaper-looking hour is the dearer token." },
        scene(G) {
          G.text(24, 34, "A $1.00/h, 1,000 tok/s  ·  B $0.90/h, 800 tok/s (example numbers)", { size: 12 });
          const pair = (x, title, a, b, fmt, max) => {
            G.text(x, 62, title, { size: 13, color: "ink" });
            const ba = G.rect(x, 280 - 160 * a / max, 100, 160 * a / max, { fill: "v", rx: 3 });
            const bb = G.rect(x + 120, 280 - 160 * b / max, 100, 160 * b / max, { fill: b > a ? "hot" : "ok", rx: 3 });
            G.text(x + 50, 280 - 160 * a / max - 6, fmt(a), { anchor: "middle", size: 13 }); G.text(x + 170, 280 - 160 * b / max - 6, fmt(b), { anchor: "middle", size: 13 });
            G.label(x + 50, 300, "A", { anchor: "middle", size: 12 }); G.label(x + 170, 300, "B", { anchor: "middle", size: 12 });
            return [ba, bb];
          };
          const p1 = pair(40, "$ per GPU-hour", 1.0, 0.9, (v) => "$" + v.toFixed(2), 1.0);
          const p2 = pair(340, "$ per 1M tokens, 100% busy", 0.2778, 0.3125, (v) => "$" + v.toFixed(3), 0.3125);
          G.line(30, 280, 600, 280, { color: "line" });
          G.rect(24, 322, 592, 90, { stroke: "line", rx: 10 });
          G.text(40, 348, "B: cheaper hour, dearer token (0.9 ÷ 0.8 = 1.125×)", { size: 13, color: "hot" });
          G.label(40, 372, "with a 15% commitment B drops to $0.2656 and wins again", { color: "ok" });
          G.label(40, 392, "list price ≠ cost per token: measure the knee and the utilisation", {});
          G.from(p1.concat(p2), { attr: { height: 0 }, stagger: 0.1, duration: 0.4 });
          G.caption("the cheaper hour can be the dearer token");
        } },

      /* 10 */
      { rail: "alternatives", title: "If you don't want a second Kubernetes cloud",
        body: `<p>GKE is the default because it is the closest analogue to EKS, so almost nothing in the platform changes. Two alternatives trade infrastructure code for something else:</p>
<table><tr><th>option</th><th>less of</th><th>more of</th><th>fits when</th></tr>
<tr><td><b>Modal</b> (serverless GPUs)</td><td>infra code, cluster operations</td><td>per-second pricing at a premium, platform-specific APIs</td><td>bursty or batch workloads, small teams</td></tr>
<tr><td><b>Lambda / CoreWeave</b> (GPU clouds)</td><td>price per GPU-hour</td><td>less managed glue; you run your own Kubernetes or Slurm</td><td>sustained large-GPU training or serving</td></tr></table>
<p>All of this is <b>UNVERIFIED</b> against their current docs; check before you rely on it. Whichever you choose, the deliverable is the same <b>cross-cloud table</b>: price per GPU-hour with its date and source, the #4 knee in tokens per second, p90 TTFT at the knee, $/1M tokens at 100% and at 30% utilisation, and cold start seconds (P3.6). Different clouds also change your autoscaling policy: a platform with a longer cold start needs a larger buffer, one with per-second serverless billing may let you scale to zero more aggressively.</p>
<p>In the write-up (exercise 3) answer: which cloud wins at 100% utilisation, and does that hold at 30%? Did the knees match for the same GPU, and if not, why (CPU per GPU for tokenisation, PCIe topology, driver version)? What would a 15% committed-use discount do to the ranking? This closes P3: you can package, deploy, observe, scale and secure the platform, and move it between clouds. Next is P4: more than one GPU.</p>`,
        scene(G) {
          G.text(24, 34, "pick by what you want to stop doing", { size: 13 });
          const opts = [["GKE (default)", "k", "closest to EKS: same manifests, same tools", "you still run a cluster"], ["Modal", "q", "no cluster, per-second billing", "premium price, own APIs"], ["Lambda / CoreWeave", "v", "lower price per GPU-hour", "less managed glue"]];
          const g = opts.map(([n, c, a, b], i) => {
            const y = 56 + i * 82, gg = G.group();
            G.box(24, y, 150, 66, n, { fill: c, size: 13, parent: gg }); G.rect(184, y, 432, 66, { stroke: "line", rx: 8, parent: gg });
            G.text(198, y + 28, "+ " + a, { size: 12, color: "ink", parent: gg }); G.text(198, y + 52, "– " + b, { size: 12, color: "muted", parent: gg });
            return gg;
          });
          G.rect(24, 312, 592, 94, { stroke: "ok", rx: 10 });
          G.text(40, 338, "same deliverable everywhere: the cross-cloud table", { size: 13, color: "ok" });
          G.label(40, 362, "$/GPU-h (dated) · knee · p90 TTFT · $/1M at 100%, 30% · cold start", {});
          G.label(40, 384, "all alternatives: UNVERIFIED against current docs", {});
          G.from(g, { opacity: 0, x: -14, stagger: 0.15, duration: 0.3 });
          G.caption("whatever you pick, compare tokens, not hours");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A second cloud is worth it for capacity, price, resilience or customers, provided cloud knowledge stays in <code>infra/</code> and one overlay.</li>
<li>Most AWS pieces map one to one; the differences that bite are: GKE installs the GPU driver itself (no GPU Operator), identity and metadata protection are wired differently, and GCP spot gives a much shorter notice.</li>
<li>Quota is a calendar dependency: AWS counts vCPUs in a family, GCP counts GPUs per model and globally. Request it first.</li>
<li>A private path to object storage avoids per-GB NAT charges on every weight pull; budgets alert but do not stop spending, so tear down with <code>make down</code>.</li>
<li>Tests keep the clouds from drifting: <code>test_overlays.py</code> and <code>test_gke_tf.py</code>.</li>
<li>Compare clouds on $/1M tokens at the SLO. The cheaper GPU hour can be the dearer token.</li></ul>`,
    sim: {
      title: "Which cloud is cheaper per token?",
      intro: "Enter each cloud's price per GPU-hour and the knee tokens/s you measured in #4 (example values: use your own dated prices). Cloud B can have a committed-use discount. The simulator shows cost per million tokens at your utilisation, and the knee B would need to tie with A.",
      height: 250,
      controls: [
        { id: "pa", label: "A: $/GPU-h (example, AWS g6.xlarge)", min: 0.3, max: 3, step: 0.05, value: 1.0, format: (v) => v.toFixed(2) },
        { id: "ka", label: "A: knee tokens/s (from #4)", min: 200, max: 3000, step: 50, value: 1000 },
        { id: "pb", label: "B: $/GPU-h (example, GCP g2-standard-4)", min: 0.3, max: 3, step: 0.05, value: 0.9, format: (v) => v.toFixed(2) },
        { id: "kb", label: "B: knee tokens/s (from #4)", min: 200, max: 3000, step: 50, value: 800 },
        { id: "disc", label: "B: committed-use discount (%)", min: 0, max: 40, step: 1, value: 0 },
        { id: "util", label: "utilisation (%)", min: 10, max: 100, step: 5, value: 30 },
      ],
      draw(G, v) {
        const u = v.util / 100, A = perM(v.pa, 0, v.ka, u), B = perM(v.pb, v.disc, v.kb, u);
        const hourA = v.pa, hourB = v.pb * (1 - v.disc / 100), mx = Math.max(A, B), mh = Math.max(hourA, hourB);
        const need = (v.ka * hourB) / hourA;
        const bar = (x, title, a, b, max, fmt, lowWins) => {
          G.text(x, 38, title, { size: 12 });
          const hA = 130 * a / max, hB = 130 * b / max, win = lowWins ? (a <= b ? "A" : "B") : "";
          G.rect(x, 190 - hA, 80, hA, { fill: win === "A" ? "ok" : "v", rx: 3 }); G.rect(x + 100, 190 - hB, 80, hB, { fill: win === "B" ? "ok" : "k", rx: 3 });
          G.text(x + 40, 190 - hA - 6, fmt(a), { anchor: "middle", size: 12 }); G.text(x + 140, 190 - hB - 6, fmt(b), { anchor: "middle", size: 12 });
          G.label(x + 40, 208, "A", { anchor: "middle", size: 12 }); G.label(x + 140, 208, "B", { anchor: "middle", size: 12 });
        };
        G.line(30, 190, 630, 190, { color: "line" });
        bar(40, "$ per GPU-hour (list, after discount)", hourA, hourB, mh * 1.15, (x) => "$" + x.toFixed(2), true);
        bar(340, `$ per 1M tokens at ${v.util}% busy`, A, B, mx * 1.15, (x) => "$" + x.toFixed(3), true);
        G.text(40, 236, A <= B ? "A is cheaper per token" : "B is cheaper per token", { size: 13, color: "ok" });
        const hourWin = hourA <= hourB ? "A" : "B", tokWin = A <= B ? "A" : "B";
        return [
          { title: "Cost per 1M tokens", rows: [["A", "$" + F.num(A, 3)], ["B", "$" + F.num(B, 3)], ["B ÷ A", F.num(B / A, 3) + "×"]], chip: [hourWin === tokWin, hourWin === tokWin ? "the cheaper hour is also the cheaper token" : "the cheaper GPU hour is the dearer token"] },
          { title: "Same sums at other loads", rows: [["A at 100%", "$" + F.num(perM(v.pa, 0, v.ka, 1), 3)], ["B at 100%", "$" + F.num(perM(v.pb, v.disc, v.kb, 1), 3)], ["A at 30%", "$" + F.num(perM(v.pa, 0, v.ka, 0.3), 3)], ["B at 30%", "$" + F.num(perM(v.pb, v.disc, v.kb, 0.3), 3)]] },
          { title: "To tie with A", rows: [["B needs a knee of", F.num(need, 0) + " tok/s"], ["B's measured knee", F.num(v.kb) + " tok/s"], ["gap", F.num(100 * (v.kb / need - 1), 0) + "%"]] },
          { title: "Model", html: `<p class="note">$ per 1M = hourly price × (1 − discount) ÷ (knee tokens/s × 3,600) × 1,000,000 ÷ utilisation, the P3.5 cost identity. Utilisation scales both clouds equally, so it changes the dollars but not the ranking; price, discount and knee change the ranking. Prices here are example values; <code>compare_clouds.py</code> requires yours, dated and sourced. Spot, NAT and cold start are not included.</p>` },
        ];
      },
    },
    practice: {
      intro: "Exercises 1 to 3 start at T0 (a laptop: <code>terraform validate</code>, pytest). Deploying to GKE is T3: you need a GCP project with billing and L4 quota. Start with <code>make up</code> in <code>infra/gcp/gke</code> and end with <code>make down</code>. Cost: the cluster fee, two <code>e2-standard-4</code> system nodes, GPU node-hours, and Cloud NAT (look each up in your region). The GPU pool scales to zero when idle; the system pool and NAT bill until <code>make down</code>; the budget alerts at 50, 80 and 100% but does not stop spending. Nodes are private, the control plane is open only to your <code>master_authorized_cidrs</code>, and Services stay ClusterIP: no unauthenticated public endpoint.",
      items: [
        { title: "Terraform for a GKE GPU node pool", tier: "T0 → T3", goal: "Name the AWS counterpart of every resource in infra/gcp/gke/main.tf, run fmt and validate with no credentials, break one policy check on purpose, then add a second spot GPU pool (min 0) and a check that every GPU pool has min_node_count = 0.",
          cmd: "bash tools/ci_terraform.sh && uv run pytest course/P3-deployment-and-infra/P3.9-second-cloud/exercises/test_gke_tf.py" },
        { title: "Same manifests, only an overlay changes", tier: "T0 → T3", goal: "Make test_overlays.py pass, add a GKE-only nodeSelector patch and watch it fail, then allow exactly one cloud.yaml file (a StorageClass) to differ.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.9-second-cloud/exercises/test_overlays.py" },
        { title: "Cross-cloud latency and cost table", tier: "T0 with your numbers / T3", goal: "Write results/clouds.yaml with your own dated, sourced prices and #4 results, generate the table, and write 5 to 10 sentences: who wins at 100% and at 30%, whether the knees matched, and what a 15% discount does.",
          cmd: "PYTHONPATH=platform uv run python course/P3-deployment-and-infra/P3.9-second-cloud/exercises/compare_clouds.py results/clouds.yaml" },
      ],
      labs: [
        { label: "Run all T0 exercises", path: "uv run pytest course/P3-deployment-and-infra/P3.9-second-cloud/exercises" },
        { label: "GKE Terraform: private nodes, min-0 L4 pool, Workload Identity, budget", path: "infra/gcp/gke/" },
        { label: "GKE overlay (the only platform-side difference)", path: "platform/deploy/overlays/gke/kustomization.yaml" },
        { label: "EKS overlay, for comparison", path: "platform/deploy/overlays/eks/kustomization.yaml" },
        { label: "The cost model behind the comparison (P3.5)", path: "platform/cost/model.py" },
        { label: "Extension: implement GCSStore next to S3Store in the P3.7 cache", path: "platform/weights/cache.py" },
      ],
    },
  });
})();
