/* P3.2 — Terraform on AWS. Infrastructure as code from first principles: blocks, state, plan/apply, then the two
   stacks the course uses (infra/aws/single-node and infra/aws/eks) read guardrail by guardrail. */
(function () {
  const F = S2S.fmt;

  /* ---------- simulator: the idle watchdog (infra/aws/scripts/idle_decider.py) plus the CloudWatch backstop ---------- */
  function decide(state, util, sessions, limit) {          // the same logic as idle_decider.decide
    const n = util < 5 && sessions === 0 ? state + 1 : 0;
    return [n, n >= limit];
  }
  function runLab(v) {
    const T = 600, b1 = Math.round(v.b1 * 60), gap = Math.round(v.gap * 60), b2 = 60;
    const busy = (m) => m < b1 || (m >= b1 + gap && m < b1 + gap + b2);
    const sess = (m) => (busy(m) ? 1 : (v.ssm && m >= b1 && m < b1 + gap ? 1 : 0));   // you stay connected while you work (and maybe while idle)
    const running = [], idleRun = [];
    let on = true, n = 0, idleCpu = 0, stops = 0;
    for (let m = 0; m < T; m++) {
      if (!on && busy(m)) { on = true; n = 0; idleCpu = 0; }   // you restart it to work
      running.push(on);
      if (!on) { idleRun.push(false); continue; }
      idleRun.push(!busy(m));
      if (v.dog) { const [nn, stop] = decide(n, busy(m) ? 90 : 0, sess(m), v.limit); n = nn; if (stop) { on = false; stops++; n = 0; } }
      idleCpu = busy(m) ? 0 : idleCpu + 1;
      if (on && idleCpu >= 180) { on = false; stops++; idleCpu = 0; }   // backstop: 12 x 15 min of CPU < 2%
    }
    const mins = running.filter(Boolean).length, wasted = idleRun.filter(Boolean).length;
    return { T, busy, running, idleRun, mins, wasted, stops, work: b1 + b2 };
  }

  S2S.lesson({
    id: "p3-2", n: "P3.2", title: "Terraform on AWS",
    subtitle: "Deployment and infra · first principles · T0 to write and test, T2/T3 to apply",
    kicker: "Lesson · ≈ 45 min",
    headline: "Cloud resources you can read, review and delete",
    intro: `<p>Every T2 and T3 lab in this course needs a GPU machine or a cluster that exists for a few hours and then vanishes. Doing that by clicking in a console is how bills explode and setups stop being reproducible. <b>Terraform</b> lets you describe the machines in text, see exactly what will change before it changes, and delete everything with one command. This lesson builds the idea from the ground up (blocks, state, plan and apply), then reads the course's two stacks, <code>infra/aws/single-node</code> and <code>infra/aws/eks</code>, one guardrail at a time: no inbound ports, least-privilege roles, cost limits that still work when a script fails, and quotas.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "your code / request"], ["ok", "allowed / safe"], ["hot", "denied / risk"], ["v", "cost / warning"], ["w", "state"]],
    prev: "p3-1", next: "p3-3",
    steps: [
      { rail: "why code", title: "Why describe infrastructure in text",
        body: `<p>Suppose a lab needs one GPU server with a certain disk, network rules, an IAM role and a budget alarm. You can create each piece by clicking in the AWS console. Three problems follow:</p>
<ul><li><b>Not repeatable.</b> Next month you cannot remember which boxes you ticked.</li><li><b>Not reviewable.</b> A teammate cannot read "I opened port 22 for a minute".</li><li><b>Not deletable.</b> Forgotten pieces (a disk, a NAT gateway, a load balancer) keep billing.</li></ul>
<p><b>Infrastructure as code</b> fixes all three: the setup is a text file in git. <b>Terraform</b> is a tool that reads such files and makes the real cloud match them. The files are <b>declarative</b>: you state what should exist ("one instance of type g6.xlarge with these rules"), not the steps to create it. Terraform works out the steps by comparing your description to what exists.</p>
<p>The whole workflow is three commands, wrapped in each stack's <code>Makefile</code>:</p>
<div class="eq">make up     init + apply  create what the code says
make down   destroy       delete everything it made
make check  fmt + validate + no-inbound grep
            (no AWS calls)</div>
<p>Every lab in the course is one <code>make up</code> and one <code>make down</code> away. Read the stacks as code: each guardrail the AWS guides promise (no public endpoint, auto-stop, budget) is a few lines you can review.</p>`,
        scene(G) {
          G.text(24, 32, "from text to machines, and back", { size: 13 });
          const boxes = [["*.tf files", "in git", "k"], ["terraform plan", "shows the diff", "v"], ["terraform apply", "makes it real", "ok"], ["AWS resources", "tagged, billed", "w"]];
          const bx = boxes.map(([a, b, c], i) => { G.box(24 + i * 154, 70, 134, 50, a, { fill: c, size: 13 }); G.label(30 + i * 154, 140, b, { size: 12 }); return null; });
          for (let i = 0; i < 3; i++) G.arrow(158 + i * 154, 95, 176 + i * 154, 95, { color: "muted", w: 2 });
          G.path("M 560 152 C 560 206, 90 206, 90 156", { color: "hot", w: 2, dash: "5 4", arrow: true });
          G.text(190, 222, "terraform destroy: delete it all", { size: 13, color: "hot" });
          G.text(24, 262, "by hand, in a console", { size: 13, color: "hot" });
          ["not repeatable", "not reviewable", "not deletable"].forEach((t, i) => { G.box(24 + i * 154, 276, 134, 34, t, { stroke: "hot", color: "hot", size: 12 }); });
          G.text(24, 340, "as code", { size: 13, color: "ok" });
          ["repeatable", "reviewable in git", "make down"].forEach((t, i) => { G.box(24 + i * 154, 356, 134, 34, t, { stroke: "ok", color: "ok", size: 12 }); });
          G.caption("describe what should exist; let the tool make it so");
        } },

      { rail: "building blocks", title: "The five kinds of block",
        body: `<p>A Terraform file is a list of <b>blocks</b>. The language (HCL) has a few kinds, and you meet all of them in <code>infra/aws/</code>:</p>
<div class="eq">provider "aws" { region = "us-east-1" }

variable "owner" { type = string }

resource "aws_instance" "gpu" {
  instance_type = var.instance_type
  ...
}
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "6.0.1"
}
output "instance_id" { value = aws_instance.gpu.id }</div>
<ul><li><b>provider</b>: which cloud API to talk to, and in which region. Credentials come from your environment, never from the file.</li><li><b>variable</b>: an input, with a type and optional default, filled from <code>terraform.tfvars</code> (<code>owner</code>, <code>instance_type</code>, <code>budget_email</code>).</li><li><b>resource</b>: one thing that should exist (an instance, a security group, an alarm). Its address is <code>type.name</code>, like <code>aws_instance.gpu</code>.</li><li><b>module</b>: a reusable bundle of resources with its own inputs. The EKS stack uses community modules for the VPC and the cluster instead of hand-writing hundreds of lines; <code>infra/aws/guardrails</code> is the course's own module (the budget).</li><li><b>output</b>: a value to print or hand on, such as the instance id the Makefile uses for <code>make ssm</code>.</li></ul>
<p>References like <code>aws_instance.gpu.id</code> are also how Terraform learns order: the alarm refers to the instance, so the instance is created first. You never write the order yourself.</p>`,
        scene(G) {
          G.text(24, 32, "one stack, five block kinds", { size: 13 });
          const code = [["provider \"aws\" { region = ... }", "which API, which region", "k"], ["variable \"owner\" { type = string }", "an input (from tfvars)", "v"], ["resource \"aws_instance\" \"gpu\" { ... }", "a thing that exists", "ok"], ["module \"vpc\" { version = \"6.0.1\" }", "a reusable bundle, pinned", "q"], ["output \"instance_id\" { ... }", "a value handed on", "blue"]];
          G.rect(20, 54, 370, 266, { stroke: "line", rx: 8 });
          const items = [];
          code.forEach(([c, d, col], i) => {
            const y = 74 + i * 50;
            items.push(G.rect(30, y - 8, 4, 36, { fill: col, rx: 1 }));
            G.text(42, y + 14, c, { size: 11 });
            G.arrow(392, y + 10, 424, y + 10, { color: col, w: 1.5 });
            G.text(430, y + 14, d, { size: 12, color: col });
          });
          G.text(24, 356, "references order the work:", { size: 13 });
          G.label(24, 378, "alarm → refers to aws_instance.gpu.id → the instance is built first", { size: 12 });
          G.from(items, { opacity: 0, stagger: 0.1, duration: 0.25 });
          G.caption("blocks declare what exists; references imply the order");
        } },

      { rail: "state", title: "State: how Terraform remembers what it built",
        body: `<p>Your code says "an instance called <code>gpu</code>". AWS knows it as <code>i-0abc123…</code>. Something has to connect the two, or Terraform could not tell "create" from "change" from "delete". That something is the <b>state file</b>: a record, in a file, of <i>this code address → this real resource id and its settings</i>.</p>
<p>Every <code>plan</code> compares three things:</p>
<ol><li>the <b>code</b> (what you want),</li><li>the <b>state</b> (what Terraform last built),</li><li>the <b>real cloud</b> (it refreshes the state from the API).</li></ol>
<div class="eq">code:  instance_type = "g6.xlarge"
state: instance_type = "g4dn.xlarge" (id i-0abc…)
plan:  ~ aws_instance.gpu
         instance_type: g4dn → g6
         # forces replacement: -/+ destroy, create</div>
<p>Because state is the memory, three rules follow:</p>
<ul><li><b>Keep it remote and locked.</b> Store it in an S3 bucket with locking (native S3 locking or a DynamoDB table, depending on your Terraform version), so two people or two laptops never apply at the same time and corrupt it.</li><li><b>Never commit <code>terraform.tfstate</code>.</b> It can contain secrets in plain text.</li><li><b>Do not edit it by hand.</b> Use <code>terraform</code> commands.</li></ul>
<p>Lose the state and Terraform forgets the machines exist: it would try to create duplicates, while the old ones keep billing.</p>`,
        check: { q: "Why must Terraform state be stored remotely with locking rather than on each person's laptop?",
          options: ["It makes terraform plan faster", "So everyone plans against the same record of what exists, and two applies cannot run at once and corrupt it", "AWS requires it"], answer: 1,
          why: "State is Terraform's only link between code and real resources. Two copies would disagree (one forgets machines the other made), and simultaneous applies could overwrite each other. It also may hold secrets, so it must not be in git." },
        scene(G) {
          G.text(24, 32, "plan = compare three views", { size: 13 });
          const cols = [["code", "what you want", "k", "g6.xlarge"], ["state", "what was built", "w", "g4dn.xlarge"], ["real AWS", "refreshed from API", "ok", "g4dn.xlarge"]];
          cols.forEach(([t, d, c, val], i) => {
            const x = 24 + i * 204;
            G.box(x, 60, 180, 40, t, { fill: c, size: 14 }); G.label(x + 4, 118, d, { size: 11 });
            G.box(x, 130, 180, 36, val, { stroke: c, color: c, size: 12 });
            G.label(x + 4, 184, "instance_type", { size: 11 });
          });
          G.arrow(204, 148, 226, 148, { color: "hot", w: 2 });
          G.arrow(408, 148, 430, 148, { color: "ok", w: 2 });
          G.text(24, 232, "plan output:", { size: 13 });
          G.rect(24, 244, 592, 70, { stroke: "line", rx: 6 });
          G.text(36, 266, "~ aws_instance.gpu", { size: 12, color: "v" });
          G.text(36, 286, "    instance_type: \"g4dn.xlarge\" → \"g6.xlarge\"  # forces replacement", { size: 12, color: "hot" });
          G.text(36, 306, "Plan: 1 to add, 0 to change, 1 to destroy.", { size: 12, color: "ink" });
          G.label(24, 346, "state lives in S3 with locking: one writer at a time", { size: 12, color: "ok" });
          G.label(24, 368, "never commit terraform.tfstate: it can contain secrets", { size: 12, color: "hot" });
          G.label(24, 390, "lose it and Terraform forgets the machines, which keep billing", { size: 12 });
          G.caption("the state links code names to real resource ids");
        } },

      { rail: "plan, apply", title: "Plan, apply, destroy, and pinned versions",
        body: `<p>The loop is: <b>init</b> (download providers and modules), <b>plan</b> (show the diff, change nothing), <b>apply</b> (make it so), <b>destroy</b> (remove everything in the state). Always read the plan. Its symbols:</p>
<div class="eq">+  create
-  destroy
~  update in place
-/+ destroy and create (replacement)

Plan: 3 to add, 1 to change, 1 to destroy.</div>
<p>"1 to destroy" on a database, or a "-/+" on something that holds data, is the line to stop at. A plan is cheap; a surprise destroy is not.</p>
<p><b>Pin versions.</b> Modules and providers change. The EKS stack pins <code>terraform-aws-modules/eks</code> to exactly <code>21.26.0</code> and Terraform to ≥ 1.6, so a teammate's <code>init</code> gets the same code you tested. The provider range in the guardrails module is loose on purpose (<code>&gt;= 5.70</code>) because two roots use it with different provider major versions (the comment in its <code>versions.tf</code> says so). Upgrading is a deliberate act: <code>terraform init -upgrade</code>, then read the plan.</p>
<p>Before any AWS call, the stacks can be checked on a laptop (T0): <code>make check</code> runs <code>terraform fmt -check</code> and <code>validate</code> (is the code well-formed and are the module inputs real?), and the repo-wide <code>bash tools/ci_terraform.sh</code> adds a policy check on inbound rules. That is exercise 1, along with <code>tflint</code> and <code>checkov</code>.</p>`,
        scene(G) {
          G.text(24, 32, "the loop", { size: 13 });
          const loop = [["init", "k"], ["plan", "v"], ["apply", "ok"], ["destroy", "hot"]];
          loop.forEach(([t, c], i) => { G.box(24 + i * 154, 54, 130, 44, t, { fill: c, size: 15 }); if (i < 3) G.arrow(156 + i * 154, 76, 176 + i * 154, 76, { color: "muted", w: 2 }); });
          G.label(24, 116, "download providers", { size: 11 }); G.label(178, 116, "show the diff only", { size: 11 }); G.label(332, 116, "make it so", { size: 11 }); G.label(486, 116, "delete all in state", { size: 11 });
          G.text(24, 160, "reading a plan", { size: 13 });
          [["+", "create", "ok"], ["-", "destroy", "hot"], ["~", "update in place", "v"], ["-/+", "replace (destroy then create)", "hot"]].forEach(([s, d, c], i) => { G.box(24, 176 + i * 36, 46, 28, s, { stroke: c, color: c, size: 14 }); G.text(84, 196 + i * 36, d, { size: 13, color: c }); });
          G.text(330, 196, "Plan: 3 to add,", { size: 13 }); G.text(330, 216, "1 to change,", { size: 13 }); G.text(330, 236, "1 to destroy.", { size: 13, color: "hot" });
          G.label(330, 262, "stop and read before apply", { size: 12, color: "hot" });
          G.text(24, 346, "pin: module version = \"21.26.0\", Terraform ≥ 1.6", { size: 13, color: "ink" });
          G.label(24, 370, "terraform init -upgrade is a deliberate act, then read the plan", { size: 12 });
          G.caption("a plan changes nothing: read it every time");
        } },

      { rail: "no inbound", title: "Single-node: no inbound ports, and IMDSv2",
        body: `<p><code>infra/aws/single-node</code> creates one GPU instance for T2 labs. Its first guardrail: <b>no inbound rule at all</b>. The security group (<code>main.tf</code>) has only <code>egress</code>, because the box needs to reach the internet (packages, Hugging Face, SSM) but nothing needs to reach it. There is no port 22 and no SSH key. You get a shell through <b>SSM Session Manager</b>: an agent on the instance connects <i>out</i> to AWS, and you authenticate with your own AWS identity. <code>make ssm</code> opens the shell; <code>make forward</code> tunnels the server's port 8000 to your laptop, so the model server binds to <code>127.0.0.1</code> and is never public.</p>
<p>The second guardrail is <b>IMDSv2</b>. Every EC2 instance can ask a link-local address (169.254.169.254, the metadata service) for its own IAM role's temporary credentials. Version 1 answers a plain GET. If an application on the box can be tricked into fetching a URL an attacker chooses (<b>SSRF</b>, a classic bug in services that fetch URLs), the attacker aims it at that address and steals the role's credentials. Version 2 requires a <b>session token</b> that you obtain with a PUT request first. A simple forged GET cannot produce one.</p>
<div class="eq">metadata_options {
  http_tokens = "required"        # IMDSv2 only
  http_put_response_hop_limit = 2
}
root_block_device { encrypted = true }</div>
<p>The hop limit of 2 lets containers on the host (one extra network hop through the docker bridge) still reach the metadata service. The root disk is encrypted. Two more details: <code>instance_initiated_shutdown_behavior = "stop"</code> so the idle watchdog's shutdown <i>stops</i> the instance (you keep the disk and the model) instead of terminating it, and exercise 3's test (<code>test_policy_tf.py</code>) fails if anyone adds an <code>ingress</code> block, turns off IMDSv2 or the encryption.</p>`,
        check: { q: "What does requiring IMDSv2 (http_tokens = required) protect against?",
          options: ["Anyone on the internet logging in over SSH", "A tricked application leaking the instance role's credentials: fetching them now needs a PUT-obtained token that a simple forged GET cannot supply", "Disk theft"], answer: 1,
          why: "IMDSv1 answered any GET, so an SSRF bug could be pointed at the metadata address to read role credentials. IMDSv2 demands a session token obtained with a PUT, which simple request forgery does not do. SSH exposure is handled separately, by having no inbound rules." },
        scene(G) {
          G.text(24, 30, "how you reach the box, and what it can reach", { size: 13 });
          G.text(24, 62, "usual: open a port", { size: 12, color: "hot" });
          G.box(24, 72, 100, 36, "internet", { stroke: "hot", color: "hot", size: 12 });
          G.arrow(126, 90, 190, 90, { color: "hot", w: 2 }); G.label(130, 82, "22 open", { size: 11, color: "hot" });
          G.box(192, 72, 120, 36, "instance", { stroke: "hot", color: "hot", size: 12 });
          G.text(340, 62, "here: no inbound at all", { size: 12, color: "ok" });
          G.box(340, 72, 100, 36, "you + IAM", { stroke: "ok", color: "ok", size: 12 });
          G.box(500, 72, 116, 36, "SSM service", { fill: "ok", size: 12 });
          G.arrow(442, 90, 498, 90, { color: "ok", w: 2 });
          G.box(500, 126, 116, 36, "instance", { stroke: "ok", color: "ok", size: 12 });
          G.arrow(558, 126, 558, 110, { color: "ok", w: 2 });
          G.label(340, 140, "agent calls OUT", { size: 11, color: "ok" });
          G.label(340, 156, "egress only", { size: 11, color: "ok" });
          G.text(24, 214, "metadata service 169.254.169.254", { size: 13 });
          G.box(24, 232, 180, 38, "app tricked by a URL", { stroke: "hot", color: "hot", size: 12 });
          G.arrow(206, 251, 250, 251, { color: "hot", w: 2 }); G.label(208, 242, "GET", { size: 11, color: "hot" });
          G.box(252, 232, 160, 38, "IMDS", { stroke: "line", size: 12 });
          G.text(430, 244, "v1: credentials leak", { size: 12, color: "hot" });
          G.text(430, 264, "v2: needs PUT token ✕", { size: 12, color: "ok" });
          G.label(24, 310, "hop limit 2: containers on the host can still use it", { size: 12 });
          G.label(24, 332, "stop (not terminate) on idle shutdown: the disk and model survive", { size: 12 });
          G.label(24, 354, "test_policy_tf.py fails if ingress or IMDSv1 appear", { size: 12, color: "k" });
          G.caption("nothing listens to the internet; stolen URLs can't steal credentials");
        } },

      { rail: "least privilege", title: "Least privilege: what the instance's role may do",
        body: `<p>Code running on the instance (including the model server and anything it downloads) acts with the instance's <b>IAM role</b>. If that role can do everything, one compromised process owns the account. The rule is <b>least privilege</b>: grant exactly what the lab needs, scoped as narrowly as the API allows. <code>iam.tf</code> in the single-node stack attaches four things:</p>
<ul><li><b>AmazonSSMManagedInstanceCore</b>: lets the SSM agent register, so <code>make ssm</code> works.</li><li><b>CloudWatchAgentServerPolicy</b>: lets the agent publish metrics (GPU utilization, for the idle checks).</li><li><b>S3 read on one prefix</b> (only if you set <code>weights_bucket</code>): <code>s3:ListBucket</code> restricted by a <code>s3:prefix</code> condition to <code>models/*</code>, and <code>s3:GetObject</code> on <code>bucket/models/*</code>. No write, no other buckets.</li><li><b>One SSM parameter</b>: <code>ssm:GetParameter</code> on <code>/s2s/hf_token</code> only.</li></ul>
<div class="eq">Allowed:  s3:GetObject  on  my-bucket/models/llama/…
Denied:   s3:GetObject  on  my-bucket/billing/…
Denied:   s3:PutObject  anywhere
Allowed:  ssm:GetParameter  on  /s2s/hf_token
Denied:   ssm:GetParameter  on  /prod/db_password</div>
<p>Secrets follow the same logic: the Hugging Face token is created once with <code>aws ssm put-parameter --type SecureString</code> and read at run time. It is never in the repo, never in <code>user_data</code>, and the role can read that one parameter and no other. A stolen role, or an SSRF that gets past IMDSv2, can then reach only model files and one token.</p>`,
        scene(G) {
          G.text(24, 32, "instance role: only what the lab needs", { size: 13 });
          G.box(24, 150, 130, 60, "instance role", { fill: "k", size: 13 });
          const g = [["SSM core", "shell via Session Manager", "ok", 54], ["CloudWatch agent", "publish GPU / CPU metrics", "ok", 100], ["S3 read: models/* only", "ListBucket + GetObject", "ok", 146], ["SSM parameter /s2s/hf_token", "one secret, read only", "ok", 192]];
          g.forEach(([t, d, c, y], i) => { G.arrow(156, 180, 218, y + 16, { color: "ok", w: 1.5 }); G.box(220, y, 236, 34, t, { stroke: c, color: c, size: 12 }); G.label(464, y + 21, d, { size: 11 }); });
          G.text(24, 270, "not granted", { size: 13, color: "hot" });
          const d = [["s3:PutObject / other buckets", 288], ["other SSM parameters", 328], ["*:*  (admin)", 368]];
          d.forEach(([t, y]) => { G.box(24, y, 280, 30, t, { stroke: "hot", color: "hot", size: 12, dash: "4 3" }); G.line(24, y + 15, 304, y + 15, { color: "hot", w: 1.5, opacity: 0.6 }); });
          G.label(330, 308, "a leaked role can read model files", { size: 12 }); G.label(330, 326, "and one token. Nothing else.", { size: 12 });
          G.caption("grant the narrowest action on the narrowest resource");
        } },

      { rail: "cost guardrails", title: "Two cost guardrails that work even if one fails",
        body: `<p>A GPU instance bills every hour it is running, whether or not you are using it. The most common lab mistake is walking away. <code>single-node</code> has three layers, built so the second does not depend on the first.</p>
<p><b>1. An idle watchdog</b> (<code>user_data.sh.tftpl</code>, run every minute by cron). It counts consecutive minutes in which GPU utilization is under 5% <i>and</i> no SSM session is open, and shuts the instance down at <code>idle_minutes</code> (default 30). A shutdown from inside <i>stops</i> the instance, keeping the disk. The decision is mirrored as a pure Python function so it can be tested (<code>infra/aws/scripts/idle_decider.py</code>):</p>
<div class="eq">n = n + 1  if util &lt; 5 and sessions == 0
n = 0      otherwise
stop when n ≥ idle_limit

limit 30; busy through minute 100, then idle:
  n = 1 at minute 101 … n = 30 at minute 130 → stop
one busy minute at 120 resets n → stop at 150
SSM session open: n stays 0, never stops
nvidia-smi failing counts as idle
(a dead GPU should stop billing)</div>
<p><b>2. A CloudWatch backstop</b> (<code>guardrails.tf</code>): an alarm on the instance's CPU, 15-minute periods, 12 of them below 2% (12 × 15 min = 3 h), whose action is <code>ec2:stop</code>. It lives outside the instance, so it catches a dead or broken watchdog.</p>
<p><b>3. An AWS Budget</b> (the <code>guardrails</code> module): a monthly limit (default $100) that emails you at 80% of the <i>forecast</i> and at 100% of the <i>actual</i> spend. Budgets alert; they do not stop anything. Only <code>make down</code> deletes the disk, which still bills while the instance is stopped.</p>`,
        check: { q: "The idle watchdog is configured with idle_minutes = 30. The GPU is busy until minute 100, idle at minute 101 onward, but you leave an SSM session open. When does the instance stop by the watchdog?",
          options: ["Minute 130", "Never by the watchdog: an open SSM session keeps the idle count at 0 (the 3-hour CPU backstop would still stop it)", "Minute 101"], answer: 1,
          why: "The watchdog only counts a minute as idle when GPU utilization is under 5% and no SSM session is open. With a session open the count stays 0. That is why the CloudWatch backstop, which ignores sessions, exists." },
        scene(G) {
          const x0 = 40, sc = 7.2, m0 = 96, X = (m) => x0 + (m - m0) * sc;
          G.text(24, 30, "idle watchdog (limit 30): counting idle minutes", { size: 13 });
          const lane = (y, label, busyUntil, blip, stopAt) => {
            G.text(24, y - 8, label, { size: 12 });
            G.rect(X(m0), y, X(busyUntil) - X(m0), 26, { fill: "k", rx: 3 });
            const e = X(stopAt);
            G.rect(X(busyUntil), y, e - X(busyUntil), 26, { fill: "v", rx: 3, opacity: 0.55 });
            if (blip) G.rect(X(blip), y, sc * 1, 26, { fill: "k", rx: 0 });
            G.line(e, y - 6, e, y + 32, { color: "hot", w: 3 });
            G.label(e + 5, y + 18, "stop", { size: 12, color: "hot" });
          };
          G.line(X(m0), 50, X(m0 + 60), 50, { color: "line" });
          [100, 110, 120, 130, 140, 150].forEach((m) => G.label(X(m), 68, String(m) + " min", { anchor: "middle", size: 10 }));
          lane(124, "A: idle from minute 101 → n reaches 30 at minute 130", 100, null, 130);
          lane(200, "B: one busy minute at 120 resets n → stop at 150", 100, 120, 150);
          G.label(X(100) + 4, 170, "n counts 1 … 30", { size: 11, color: "v" });
          G.label(24, 248, "cyan = busy · amber = idle, counting toward the limit", { size: 11 });
          G.text(24, 276, "SSM session open: n stays 0, so the watchdog never fires", { size: 13, color: "ink" });
          G.text(24, 306, "backstop (CloudWatch): CPU < 2% for 12 × 15 min = 3 h → ec2:stop", { size: 13, color: "v" });
          G.label(24, 330, "lives outside the instance, so it still works if the watchdog is broken", { size: 12 });
          G.label(24, 354, "AWS Budget: email at 80% forecast and 100% actual (alerts only)", { size: 12 });
          G.label(24, 376, "a stopped instance still bills for its disk until make down", { size: 12, color: "hot" });
          G.caption("a script, a cloud alarm, and a budget: three independent layers");
        } },

      { rail: "EKS stack", title: "The EKS stack: VPC, cluster, and a GPU pool that scales to zero",
        body: `<p>Kubernetes (P3.3) needs a cluster. <code>infra/aws/eks</code> builds one from three pinned pieces.</p>
<p><b>The network (VPC).</b> A private address space, <code>10.42.0.0/16</code>, across 2 availability zones. The number after the slash is how many leading bits are fixed, so a <code>/19</code> private subnet has 2<sup>32−19</sup> = 8,192 addresses and a <code>/22</code> public subnet has 2<sup>32−22</sup> = 1,024 (AWS reserves a few in each). <b>Private</b> subnets hold the nodes (no public IP, no direct inbound). <b>Public</b> subnets hold load balancers and one <b>NAT gateway</b>, which lets private nodes make outbound connections. One NAT, not one per zone, is a cost compromise (not highly available; the code says so).</p>
<p><b>The cluster.</b> <code>terraform-aws-modules/eks</code> at exactly 21.26.0. The control plane is run by AWS; its API endpoint is public but restricted to <i>your</i> IP by <code>endpoint_public_access_cidrs = var.allowed_api_cidrs</code>. <code>make check</code> refuses <code>0.0.0.0/0</code> there.</p>
<p><b>Two node groups.</b></p>
<ul><li><code>system</code>: 1× m6i.large for cluster add-ons.</li><li><code>gpu</code>: <code>g6.xlarge</code> by default, with the <code>AL2023_x86_64_NVIDIA</code> AMI type so GPU drivers are preinstalled, <code>min_size = 0</code> so you pay for no GPU while nothing needs one, and a <b>taint</b> <code>nvidia.com/gpu=present:NoSchedule</code>. A taint repels every pod that does not explicitly tolerate it, so only GPU workloads land on the expensive nodes, and a stray CPU pod cannot keep the pool from scaling back to zero. P3.6's autoscaler drives the pool size.</li></ul>`,
        check: { q: "Why does the GPU node group have both min_size = 0 and a nvidia.com/gpu taint?",
          options: ["Zero nodes are required by AWS, and the taint is cosmetic", "min_size = 0 means no GPU is billed while idle; the taint stops ordinary pods from landing on GPU nodes and keeping them alive", "The taint makes the nodes faster"], answer: 1,
          why: "Scale-to-zero only saves money if nothing keeps a GPU node busy. A pod without a toleration cannot schedule onto tainted nodes, so only real GPU workloads trigger (and hold) a scale-up." },
        scene(G) {
          G.text(24, 28, "infra/aws/eks (2 availability zones)", { size: 13 });
          G.rect(16, 40, 440, 330, { stroke: "k", rx: 10, dash: "5 4" }); G.label(24, 58, "VPC 10.42.0.0/16", { size: 11, color: "k" });
          G.rect(28, 66, 416, 62, { stroke: "ok", rx: 6 }); G.label(34, 82, "public subnets (/22 = 1,024 addresses)", { size: 11, color: "ok" });
          G.box(40, 88, 90, 32, "NAT gateway", { stroke: "v", color: "v", size: 11 }); G.box(142, 88, 110, 32, "load balancers", { stroke: "ok", color: "ok", size: 11 });
          G.rect(28, 140, 416, 160, { stroke: "muted", rx: 6 }); G.label(34, 156, "private subnets (/19 = 8,192 addresses)", { size: 11 });
          G.box(40, 166, 150, 44, "system · m6i.large", { fill: "w", size: 11 });
          G.box(210, 166, 220, 44, "gpu pool · g6.xlarge", { fill: "k", size: 11 });
          G.label(214, 226, "min 0 · taint nvidia.com/gpu", { size: 10, color: "k" });
          G.label(40, 226, "kube-system add-ons", { size: 10 });
          G.box(40, 244, 390, 40, "pods: GPU workloads tolerate the taint, others stay on system", { stroke: "line", size: 11 });
          G.box(40, 318, 180, 36, "S3 gateway endpoint", { stroke: "ok", color: "ok", size: 11 });
          G.arrow(130, 316, 130, 288, { color: "ok" });
          G.box(480, 70, 144, 60, "control plane", { fill: "q", size: 12 }); G.label(486, 146, "run by AWS", { size: 10 });
          G.label(472, 170, "API endpoint public,", { size: 10 }); G.label(472, 184, "only from your IP", { size: 10, color: "ok" });
          G.arrow(478, 108, 446, 190, { color: "q", w: 1.5 });
          G.label(472, 232, "make check refuses", { size: 10, color: "hot" }); G.label(472, 246, "0.0.0.0/0 here", { size: 10, color: "hot" });
          G.caption("nodes private, API locked to you, GPUs only when needed");
        } },

      { rail: "S3 endpoint", title: "The S3 gateway endpoint: don't pay the NAT for model downloads",
        body: `<p>When a GPU node starts, it pulls model weights, often from S3 (P3.7). The node is in a private subnet, so by default its traffic to S3 leaves through the <b>NAT gateway</b>, which bills <b>per GB processed</b> on top of its hourly charge. Model files are large, and every new node pulls them again.</p>
<div class="eq">16 GB model × 10 nodes = 160 GB through the NAT
cost = 160 × (NAT $/GB in your region)
       (look it up on the EC2 / VPC pricing page)</div>
<p>An <b>S3 gateway endpoint</b> (<code>aws_vpc_endpoint.s3</code> in <code>eks/main.tf</code>) fixes it: it adds an entry to the private route tables that sends S3-bound traffic straight to S3 over AWS's network, not through the NAT. Gateway endpoints for S3 carry no hourly charge and no per-GB data fee, so those 160 GB cost the NAT nothing (confirm on the pricing page). It is also faster, since the transfer skips a shared bottleneck.</p>
<p>One detail matters: the endpoint must be attached to the <b>private</b> route tables, because that is where the nodes' traffic is routed (<code>route_table_ids = module.vpc.private_route_table_ids</code>). Exercise 2 has you read that line, then (on a T3 cluster) measure the NAT's <code>BytesOutToDestination</code> metric while pulling 1 GB with and without the endpoint, and write the per-GB arithmetic for your region.</p>`,
        check: { q: "A node in a private subnet pulls a 16 GB model from S3. With no S3 gateway endpoint, where does that traffic go, and what does it cost?",
          options: ["Straight to S3, free", "Through the NAT gateway, which bills per GB processed (16 GB here, per node)", "It is blocked"], answer: 1,
          why: "Private subnets reach the internet and S3's public address only via the NAT gateway, which charges per GB it processes. The gateway endpoint adds a route that bypasses it for S3." },
        scene(G) {
          G.text(24, 30, "a private node pulling model weights from S3", { size: 13 });
          G.box(24, 150, 110, 56, "GPU node", { fill: "k", size: 13 });
          G.box(500, 150, 110, 56, "S3 bucket", { fill: "w", size: 13 });
          G.text(190, 74, "without the endpoint", { size: 12, color: "hot" });
          G.arrow(136, 166, 224, 98, { color: "hot", w: 2 });
          G.box(226, 78, 140, 40, "NAT gateway", { stroke: "v", color: "v", size: 13 });
          G.arrow(368, 100, 498, 160, { color: "hot", w: 2 });
          G.label(240, 134, "$ per GB processed", { size: 12, color: "v" });
          G.text(190, 224, "with the gateway endpoint", { size: 12, color: "ok" });
          G.arrow(136, 192, 224, 252, { color: "ok", w: 2 });
          G.box(226, 232, 140, 40, "route table entry", { stroke: "ok", color: "ok", size: 12 });
          G.arrow(368, 252, 498, 196, { color: "ok", w: 2 });
          G.label(240, 292, "no NAT hop, no per-GB NAT charge", { size: 12, color: "ok" });
          G.text(24, 340, "16 GB × 10 nodes = 160 GB through the NAT", { size: 14, color: "ink" });
          G.label(24, 364, "cost = 160 × the NAT's $/GB: look up your region's price", { size: 12 });
          G.label(24, 386, "attach the endpoint to the PRIVATE route tables", { size: 12, color: "k" });
          G.caption("one route table entry keeps bulk downloads off the NAT");
        } },

      { rail: "quotas & cost", title: "Quotas, always-on costs and teardown",
        body: `<p>Two surprises hit almost everyone, so treat them as requirements, not afterthoughts.</p>
<p><b>Quotas.</b> AWS limits how many GPU vCPUs an account may run. New accounts often have a <b>0 vCPU</b> quota for "Running On-Demand G and VT instances" (the Spot variant is separate; P instances for P4 are another quota). The EKS stack's GPU group has <code>min_size = 0</code>, so <code>make up</code> succeeds without any quota, and then the first scale-up fails. A <code>g6.xlarge</code> needs a few vCPUs (see its EC2 instance page); request at least that much in Service Quotas (EC2) <i>before</i> the labs. Approval can take days.</p>
<p><b>Costs that run while the GPU pool is at 0:</b></p>
<ul><li>the <b>EKS control plane</b>, hourly;</li><li>the <b>NAT gateway</b>, hourly plus per GB;</li><li>the <b>system node</b> (m6i.large);</li><li><b>EBS volumes</b> and any <b>load balancers</b>.</li></ul>
<div class="eq">6-hour lab, GPU node up for 3 hours:
total = 6 × (control plane + NAT + m6i.large)
      + 3 × g6.xlarge
      + NAT GB × $/GB + storage
(look up each price for your region, compute it BEFORE
make up, compare with cost.sh the next day)</div>
<p>Cost Explorer lags by up to a day; <code>infra/aws/scripts/cost.sh</code> lists what is still tagged <code>Project=silicon-to-serving</code>. For most people the NAT gateway is the line that surprises.</p>
<p><b>Teardown order matters.</b> Kubernetes creates AWS resources Terraform does not know about: load balancers and their network interfaces. They hold on to subnets and security groups, so a plain <code>terraform destroy</code> hangs on the VPC. <code>make down</code> in <code>infra/aws/eks</code> first deletes LoadBalancer services, uninstalls the Helm releases, and only then runs <code>terraform destroy</code>.</p>`,
        scene(G) {
          G.text(24, 30, "what bills during a 6-hour lab, GPU node up for 3 hours", { size: 13 });
          const x0 = 150, u = 74, rows = [["control plane", 6, "v"], ["NAT gateway", 6, "v"], ["m6i.large", 6, "w"], ["g6.xlarge", 3, "k"]];
          const bars = rows.map(([n, h, c], i) => { const y = 60 + i * 44; G.text(24, y + 22, n, { size: 13 }); G.label(x0 + h * u + 8, y + 22, h + " h", { size: 12, color: "ink" }); return G.rect(x0, y, h * u, 30, { fill: c, rx: 4 }); });
          [0, 1, 2, 3, 4, 5, 6].forEach((h) => G.label(x0 + h * u, 246, String(h), { anchor: "middle", size: 10 }));
          G.label(x0 + 3 * u, 262, "hours →", { anchor: "middle", size: 10 });
          G.text(24, 300, "still billing when the GPU pool is at 0:", { size: 13, color: "v" });
          G.label(24, 322, "control plane · NAT (hourly + per GB) · system node · disks · load balancers", { size: 12 });
          G.text(24, 362, "make down: delete LoadBalancers first, then terraform destroy", { size: 13, color: "ok" });
          G.label(24, 386, "otherwise the VPC destroy hangs on ENIs Terraform doesn't know about", { size: 12 });
          G.from(bars, { attr: { width: 0 }, stagger: 0.12, duration: 0.5 });
          G.caption("most of the bill is the cluster, not the GPU hours");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Terraform describes infrastructure as blocks (provider, variable, resource, module, output); references set the order, and a plan shows the diff before anything changes.</li>
<li>State links code to real resource ids: keep it remote and locked, never commit it. Pin module and provider versions; upgrade deliberately.</li>
<li>single-node: no inbound rules (SSM only), IMDSv2, an encrypted disk, a role limited to one S3 prefix and one secret.</li>
<li>Cost guardrails in three independent layers: an idle watchdog (decision logic testable in Python), a CloudWatch backstop, and a budget.</li>
<li>EKS: private nodes, an API locked to your IP, a tainted GPU pool that scales to zero, and an S3 gateway endpoint to keep downloads off the NAT.</li>
<li>Quotas can be zero; the control plane and NAT bill while the GPU pool is idle; teardown must remove load balancers first.</li></ul>`,
    sim: {
      title: "How much does an idle GPU cost?",
      intro: "A 10-hour day on a GPU instance, simulated minute by minute with the real idle-watchdog rule (infra/aws/scripts/idle_decider.py) and the 3-hour CloudWatch backstop. You work in two sessions with a gap between them; the instance is restarted for the second session if it stopped. The hourly price is a placeholder: replace it with your instance's price from the EC2 pricing page.",
      controls: [
        { id: "b1", label: "first work session, hours", min: 0.5, max: 4, step: 0.5, value: 2, format: (x) => x.toFixed(1) },
        { id: "gap", label: "gap before the second 1-hour session, hours", min: 0, max: 6, step: 0.5, value: 2, format: (x) => x.toFixed(1) },
        { id: "limit", label: "idle_minutes (watchdog limit)", min: 5, max: 120, step: 5, value: 30 },
        { id: "dog", label: "idle watchdog", type: "select", value: 1, options: [[1, "working"], [0, "broken or absent"]] },
        { id: "ssm", label: "during the gap you leave an SSM session open", type: "select", value: 0, options: [[0, "no (disconnected)"], [1, "yes"]] },
        { id: "price", label: "instance price, $/h (PLACEHOLDER: use your own)", min: 0.5, max: 5, step: 0.05, value: 1, format: (x) => x.toFixed(2) },
      ],
      height: 250,
      draw(G, v) {
        const r = runLab(v), x0 = 130, w = 490, X = (m) => x0 + (w * m) / r.T;
        G.label(10, 14, "10-hour day, one pixel row per layer", { size: 11 });
        G.text(10, 52, "you work", { size: 12 }); G.text(10, 108, "instance on", { size: 12 }); G.text(10, 164, "idle but billing", { size: 12, color: "hot" });
        const seg = (arr, y, col) => { let s = -1; for (let m = 0; m <= r.T; m++) { const on = m < r.T && arr(m); if (on && s < 0) s = m; if (!on && s >= 0) { G.rect(X(s), y, X(m) - X(s), 30, { fill: col, rx: 0 }); s = -1; } } };
        seg((m) => r.busy(m), 34, "k"); seg((m) => r.running[m], 90, "ok"); seg((m) => r.idleRun[m], 146, "hot");
        [0, 2, 4, 6, 8, 10].forEach((h) => { G.line(X(h * 60), 28, X(h * 60), 186, { color: "line", dash: "2 4", w: 1 }); G.label(X(h * 60), 204, h + " h", { anchor: "middle", size: 10 }); });
        const alwaysOn = (r.T / 60) * v.price, cost = (r.mins / 60) * v.price;
        return [
          { title: "Running time", rows: [["you worked", F.num(r.work / 60) + " h"], ["instance on", (r.mins / 60).toFixed(2) + " h"], ["idle while on", (r.wasted / 60).toFixed(2) + " h"], ["times it stopped itself", String(r.stops)]],
            gauge: [[r.mins / r.T, "ok"]], gaugeText: `${(100 * r.mins / r.T).toFixed(0)}% of the day on` },
          { title: "Cost at the placeholder price", rows: [["with guardrails", "$" + cost.toFixed(2)], ["left running all day", "$" + alwaysOn.toFixed(2)], ["waste (idle while on)", "$" + ((r.wasted / 60) * v.price).toFixed(2)]],
            chip: [r.wasted <= v.limit * 2 + 5, r.wasted <= v.limit * 2 + 5 ? "idle waste is bounded by the limit" : "waste is bounded only by the 3 h backstop"] },
          { title: "Model", rows: [], html: `<p class="note">Rule from idle_decider.py: a minute counts as idle if GPU utilization is under 5% and no SSM session is open; stop when the count reaches idle_minutes. The backstop ignores sessions: 3 hours of near-zero CPU stops the instance (CloudWatch alarm, 12 × 15 min). You are assumed connected via SSM while working. Price is a placeholder; a stopped instance still bills for its disk. A model of the rules, not a bill.</p>` },
        ];
      },
    },
    practice: {
      intro: "Exercises 1, 3 and 4 are T0: they run on a laptop with no AWS credentials. Exercise 2 is T0 to read and T3 to measure. Anything that applies a stack costs money: each AWS guide states the cost, an auto-stop, and the teardown (make down), exposes no public endpoint, and never puts a secret in the repo.",
      items: [
        { title: "Static checks: fmt, validate, tflint, checkov", tier: "T0 · easy", goal: "Run the whole-repo check, fix any module input names validate rejects at EKS module v21.26.0, then triage every checkov finding: fix it or write one line saying why it is acceptable for a lab (infra/aws/CHECKOV.md).",
          cmd: "bash tools/ci_terraform.sh && tflint --recursive infra/ && checkov -d infra/" },
        { title: "The S3 gateway endpoint: why, and prove it", tier: "T0 + T3 · medium", goal: "Confirm aws_vpc_endpoint.s3 is attached to the private route tables. On a live cluster, measure the NAT's BytesOutToDestination for a 1 GB pull with and without it, and write the per-GB arithmetic for your region.",
          cmd: "cd infra/aws/eks && terraform plan -var-file=terraform.tfvars   # read the route_table_ids of aws_vpc_endpoint.s3" },
        { title: "Policy test: no inbound, no public API", tier: "T0 · medium", goal: "Run the policy tests, break each rule once on purpose and see a readable failure, implement the skipped test, then sketch the same rules over terraform show -json plan output.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.2-terraform-on-aws/exercises/test_policy_tf.py" },
        { title: "Idle auto-stop, tested with synthetic metrics", tier: "T0 · hard", goal: "Drive the watchdog decision with synthetic minute-by-minute traces (busy then idle, an SSM session, nvidia-smi failing, a single busy minute). Then add a 15-minute boot grace period to both the Python mirror and the bash template.",
          cmd: "uv run pytest course/P3-deployment-and-infra/P3.2-terraform-on-aws/exercises/test_idle.py" },
      ],
      labs: [
        { label: "single-node stack: main.tf, iam.tf, guardrails.tf, user_data.sh.tftpl (T2)", path: "infra/aws/single-node/" },
        { label: "EKS stack: VPC, cluster, GPU node group, S3 endpoint (T3)", path: "infra/aws/eks/" },
        { label: "Shared budget module", path: "infra/aws/guardrails/" },
        { label: "The idle decision as a testable function; cost listing", path: "infra/aws/scripts/idle_decider.py, infra/aws/scripts/cost.sh" },
        { label: "Local checks: fmt, validate and policy for every stack", path: "tools/ci_terraform.sh" },
        { label: "Sources: terraform-aws-eks v21.26.0, data-on-eks blueprints, AWS docs (UNVERIFIED from the build environment)", path: "SOURCES.md" },
      ],
    },
  });
})();
