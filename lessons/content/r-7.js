/* R.7 — A safe cloud account. Cost, identity, quotas, budgets, hidden costs, SSM, Terraform, the stack lifecycle, teardown, secrets. */
(function () {
  const F = S2S.fmt;
  const MODULE = "course/R-readiness/R.7-cloud-account-and-cost-safety";
  // labelled box that also accepts "\n" for several lines of text
  const mb = (G, x, y, w, h, label, o = {}) => {
    const g = G.box(x, y, w, h, "", o), lines = String(label).split("\n"), size = o.size ?? 13, lh = size * 1.3;
    lines.forEach((l, i) => G.text(x + w / 2, y + h / 2 - ((lines.length - 1) * lh) / 2 + i * lh + size * 0.36, l, { anchor: "middle", size, color: o.color ?? (o.fill ? "bg" : "ink"), weight: o.weight, parent: g }));
    return g;
  };
  const money = (x) => "$" + F.num(x, x < 100 ? 2 : 0);

  S2S.lesson({
    id: "r-7", n: "R.7", title: "A safe cloud account",
    subtitle: "Readiness · first principles · T0 to read, free to set up, required before P2.1",
    kicker: "Lesson · ≈ 30 min",
    headline: "The meter that runs while you sleep",
    intro: `<p><b>Already comfortable with AWS accounts, budgets, SSO and Terraform? Answer the checkpoints. If you get them all right, skip to P0.1</b> (and keep the account checklist for P2.1).</p>
<p>From P2.1 on, the course rents GPU machines by the hour. A rented GPU bills every hour it exists, used or not. This lesson sets up an account where the damage from a mistake is small, then walks through the exact stack that <code>P2.1 aws.md</code> runs: <code>make up</code>, <code>make ssm</code>, <code>make down</code>. Every price in this lesson is an example value. Look up the real one for your region.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "3 exercises"],
    legend: [["hot", "money out"], ["ok", "safe"], ["k", "you"], ["v", "AWS"], ["w", "idle"]],
    prev: "r-6", next: "p0-1",
    steps: [
      { rail: "the bill", title: "On rented hardware, time is the bill",
        body: `<p>A data-centre GPU costs a great deal to buy and needs a server around it. Renting one by the hour costs nothing until you start it. The trade: once it starts, a meter runs.</p>
<div class="eq">cost = hours running × hourly price</div>
<p>The hourly price depends on the instance type and the region. This course lists no prices. Look up the on-demand price for your instance in your region on the EC2 pricing page, and write it in your notes. Below, we use a made-up <b>example value of 2.00 per hour</b>.</p>
<p>A 3-hour lab costs 3 × 2.00 = 6.00. Forget the instance from Friday evening to Monday morning, about 60 hours, and the bill is 120.00. The lab was 5% of it.</p>`,
        scene(G) {
          G.text(24, 36, "one 3 h lab vs one forgotten weekend (example price 2.00 per hour)", { size: 13 });
          G.axes(40, 60, 570, 160, { xlabel: "hours since Friday 18:00 →" });
          const x0 = 44, per = 560 / 60;
          const lab = G.rect(x0, 190, 3 * per, 26, { fill: "ok", rx: 3 });
          const lost = G.rect(x0 + 3 * per, 190, 57 * per, 26, { fill: "hot", rx: 3, opacity: 0.85 });
          G.text(x0 + 3 * per + 6, 172, "lab", { size: 12, color: "ok" }); G.line(x0 + 1.5 * per, 176, x0 + 1.5 * per, 188, { color: "ok" });
          G.text(x0 + 3 * per + 190, 207, "still running", { size: 13, color: "bg", anchor: "middle", weight: 700 });
          mb(G, 40, 250, 180, 70, "lab: 3 h\n6.00", { stroke: "ok", color: "ok", size: 14 });
          mb(G, 240, 250, 180, 70, "weekend: 57 h\n114.00", { stroke: "hot", color: "hot", size: 14 });
          mb(G, 440, 250, 170, 70, "total\n120.00", { stroke: "ink", size: 14 });
          G.label(40, 350, "the formula is the same; only forgetting makes it large", { size: 12 });
          G.label(40, 372, "prices here are example values: check the price page", { size: 12 });
          G.from(lost, { attr: { width: 0 }, duration: 1.2 }); G.from(lab, { opacity: 0, duration: 0.3 });
          G.caption("the dangerous bill is the forgotten instance, not the experiment");
        } },

      { rail: "root vs you", title: "Lock the master key; work as someone smaller",
        body: `<p>A new AWS account has one login: the <b>root user</b>. It can do anything, including close the account and change the bill. If its password leaks, nothing limits the damage.</p>
<ol><li>Sign in as root once. Turn on <b>MFA</b> (a second proof, usually a code from an authenticator app). Make sure root has no access keys. Then stop using it.</li>
<li>Turn on <b>IAM Identity Center</b> (single sign-on) and create a user for yourself with only the permissions the course needs.</li>
<li>Sign in to the AWS access portal as that user. The CLI gets <b>short-lived credentials</b> from it with <code>aws configure sso</code>.</li></ol>
<p>Short-lived matters. A leaked key that expires in a few hours is a small problem. A key that never expires is a large one.</p>`,
        check: { q: "Why use an Identity Center user for daily work instead of the root user?",
          options: ["Root is slower", "Root cannot be limited, so a leak is total; the SSO user has narrow permissions and short-lived credentials", "Root cannot use the CLI"], answer: 1,
          why: "Root has unlimited power and no expiry. A narrowly scoped identity with expiring credentials limits both what a leak can do and for how long." },
        scene(G) {
          mb(G, 30, 60, 200, 110, "root user\nemail + password", { stroke: "hot", color: "hot", size: 14 });
          mb(G, 60, 188, 140, 30, "MFA on · no keys", { fill: "hot", size: 12, rx: 6 });
          G.rect(30, 60, 200, 110, { stroke: "hot", dash: "6 4", rx: 8, opacity: 0.8 });
          G.text(130, 244, "locked, almost never used", { anchor: "middle", size: 12, color: "hot" });
          const sso = mb(G, 300, 60, 150, 110, "IAM Identity\nCenter user\n(you)", { stroke: "ok", color: "ok", size: 14 });
          const arr = G.arrow(450, 115, 520, 115, { color: "ok", w: 2 });
          mb(G, 520, 85, 96, 60, "console\nCLI", { stroke: "ink", size: 13 });
          mb(G, 300, 188, 150, 30, "narrow permissions", { fill: "ok", size: 12, rx: 6 });
          G.text(470, 205, "expires in hours", { size: 12, color: "ok" });
          G.line(30, 280, 616, 280, { color: "line" });
          G.text(30, 312, "leaked root:  everything, forever", { size: 14, color: "hot" });
          G.text(30, 344, "leaked SSO session:  only what you allowed, until it expires", { size: 14, color: "ok" });
          G.from(sso, { opacity: 0, x: -20, duration: 0.5 }); G.from(arr, { opacity: 0, duration: 0.5, delay: 0.3 });
          G.caption("daily work uses the small identity");
        } },

      { rail: "regions, quotas", title: "Pick a region, then ask for a GPU",
        body: `<p>A <b>region</b> is a geographic cluster of data centres, such as <code>us-east-1</code>. Each has several <b>availability zones</b>: separate buildings in the region. Instance types, prices and limits differ by region. Choose one region, write it down, and use it for the whole course.</p>
<p>AWS also sets a <b>quota</b> on how many vCPUs you may run in each instance family. In a new account, the quota for GPU instances ("Running On-Demand G and VT instances") often starts at <b>0</b>. Your first <code>make up</code> then fails with a limit error. A <code>g6.xlarge</code> has 4 vCPUs, so request at least 4 in Service Quotas. Approval can take minutes or days, so do it first.</p>`,
        check: { q: "Your first GPU launch fails on a new account with a vCPU limit error. What is the fix?",
          options: ["Wait an hour and retry", "Request a quota increase for G and VT instances (at least 4 vCPUs) in Service Quotas", "Switch to the root user"], answer: 1,
          why: "The quota starts at 0 for GPU families in many new accounts. Retrying changes nothing; you must ask for the increase, and approval can take time, so request it early." },
        scene(G) {
          G.rect(20, 40, 380, 190, { stroke: "v", rx: 12, dash: "6 4" }); G.label(34, 62, "region (example: us-east-1)", { color: "v" });
          ["zone a", "zone b", "zone c"].forEach((z, i) => mb(G, 36 + i * 122, 90, 108, 110, z + "\ndata centre", { stroke: "ink", size: 12 }));
          G.text(420, 80, "quota check", { size: 13 });
          G.rect(420, 92, 196, 28, { stroke: "line", rx: 5 });
          const q0 = G.rect(422, 94, 6, 24, { fill: "hot", rx: 2 });
          G.label(420, 138, "G and VT vCPU limit: 0", { color: "hot", size: 12 });
          mb(G, 420, 156, 196, 44, "g6.xlarge needs 4 vCPUs", { stroke: "hot", color: "hot", size: 12 });
          G.arrow(518, 206, 518, 252, { color: "k", w: 2 });
          mb(G, 420, 256, 196, 40, "request increase ≥ 4", { fill: "k", size: 13 });
          const q4 = G.rect(430, 330, 180, 24, { fill: "ok", rx: 4 });
          G.label(420, 322, "approved: limit 4 or more", { color: "ok", size: 12 });
          G.label(24, 270, "approval: minutes to days", { size: 13, color: "ink" });
          G.label(24, 294, "so request it on day one", { size: 13, color: "ink" });
          G.from(q4, { attr: { width: 0 }, duration: 0.8, delay: 0.3 });
          G.caption("a new account starts at zero GPU vCPUs");
        } },

      { rail: "budget", title: "A budget is an alarm, not a brake",
        body: `<p>An <b>AWS Budget</b> tracks your spend and emails you when it crosses a threshold. It does not stop anything. Billing data also arrives with a delay of hours, so the alarm can be late.</p>
<p>The repo's <code>infra/aws/guardrails</code> stack creates a monthly budget with two emails: one when the <i>forecast</i> passes 80% of the limit, one when the <i>actual</i> spend passes 100%. The single-node stack makes its own as well. Set the limit to an amount you could afford to lose, and use an email address you read.</p>
<div class="analogy"><b>Picture it</b>A smoke alarm. It tells you about the fire; it does not put it out. The idle auto-stop (step 8) is the sprinkler.</div>`,
        scene(G) {
          G.axes(40, 50, 570, 250, { xlabel: "days in the month →", ylabel: "spend" });
          const lim = 130;
          G.line(44, lim, 610, lim, { color: "hot", dash: "5 4" }); G.label(48, lim - 6, "limit (100%)", { color: "hot", size: 11 });
          G.line(44, lim + 52, 610, lim + 52, { color: "v", dash: "5 4" }); G.label(48, lim + 46, "80% of limit", { color: "v", size: 11 });
          G.path("M 44 300 L 200 290 L 300 270 L 380 220 L 450 150 L 520 90", { color: "k", w: 3 });
          G.circle(380, 220, 6, { fill: "v" }); G.circle(450, 150, 6, { fill: "v" });
          const m1 = mb(G, 300, 322, 170, 34, "email: forecast over 80%", { fill: "v", size: 11, rx: 6 });
          const m2 = mb(G, 430, 368, 180, 34, "email: actual over 100%", { fill: "hot", size: 11, rx: 6 });
          G.line(380, 226, 380, 320, { color: "v", dash: "3 3" }); G.line(520, 98, 520, 366, { color: "hot", dash: "3 3" });
          G.label(60, 330, "the instance keeps running", { size: 12 }); G.label(60, 348, "unless something stops it", { size: 12 });
          G.from([m1, m2], { opacity: 0, y: 10, stagger: 0.5, duration: 0.4 });
          G.caption("the alarm tells you; it does not stop the meter");
        } },

      { rail: "idle costs", title: "'Off' is not free",
        body: `<p>An instance has a life cycle, and each state bills differently.</p>
<table><tr><th>State</th><th>Still billed</th></tr>
<tr><td>running</td><td>instance + disk + public IP</td></tr>
<tr><td><b>stopped</b></td><td><b>disk (EBS)</b> and snapshots</td></tr>
<tr><td>terminated (<code>make down</code>)</td><td>nothing from the stack</td></tr>
<tr><td>any state</td><td>NAT gateways, allocated Elastic IPs, load balancers, snapshots you took by hand</td></tr></table>
<p>"Stop" is a pause. Only <code>make down</code> deletes the disk. A stopped instance with a 200 GB disk and an example disk price of 0.10 per GB-month costs 200 × 0.10 = 20.00 a month while it sits there.</p>`,
        check: { q: "You stop (not terminate) a GPU instance and go on holiday. What do you pay for while it is stopped?",
          options: ["Nothing", "The EBS disk and any snapshots", "The full hourly instance price"], answer: 1,
          why: "A stopped instance is not billed as compute, but its disk still exists and is billed per GB-month. Terminating (make down) is what removes it." },
        scene(G) {
          const states = [["running", "ok"], ["stopped", "w"], ["terminated", "blue"]];
          states.forEach(([s, c], i) => mb(G, 30 + i * 205, 40, 170, 40, s, { fill: c, size: 14, rx: 8 }));
          G.arrow(200, 60, 235, 60, { color: "muted" }); G.arrow(405, 60, 440, 60, { color: "muted" });
          const rows = [["instance (compute)", [1, 0, 0]], ["EBS disk", [1, 1, 0]], ["snapshots", [1, 1, 1]], ["NAT gateway", [1, 1, 1]], ["load balancer", [1, 1, 1]], ["allocated Elastic IP", [1, 1, 1]]];
          const dots = [];
          rows.forEach(([name, on], r) => {
            const y = 118 + r * 44; G.label(30, y + 6, name, { size: 12, color: "ink" });
            on.forEach((v, i) => { const cx = 360 + i * 100 - 80 + 40; if (v) dots.push(G.circle(cx + 20, y, 11, { fill: "hot" })); else G.circle(cx + 20, y, 11, { stroke: "line" }); });
          });
          G.label(30, 384, "red = still billing. Snapshots, NAT, load balancers and IPs", { size: 11 });
          G.label(30, 400, "bill until you delete them, even after the instance is gone", { size: 11 });
          G.from(dots, { scale: 0, transformOrigin: "center", stagger: 0.05, duration: 0.25 });
          G.caption("stopped still pays for the disk; terminated pays for what you left behind");
        } },

      { rail: "no open port", title: "Reach the machine without opening a door",
        body: `<p>The classic way into a server is <b>SSH</b> on port 22, open to the internet and guarded by a key. Automated scanners probe port 22 on every address, all day.</p>
<p><b>Session Manager</b> (part of AWS Systems Manager, "SSM") reverses the direction. A small agent on the instance makes an <i>outbound</i> connection to AWS. You sign in to AWS with your SSO identity, and AWS joins you to that connection. The instance's security group has <b>no inbound rules at all</b>. Servers on the instance bind to <code>127.0.0.1</code>, and <code>make forward</code> tunnels <code>localhost:8000</code> on your laptop to the instance's port 8000.</p>
<p>That is why the course never opens port 22 and never exposes a public inference endpoint.</p>`,
        scene(G) {
          G.text(24, 34, "SSH: a door in the wall", { size: 13, color: "hot" });
          mb(G, 24, 50, 110, 44, "internet", { stroke: "hot", color: "hot", size: 12 });
          G.arrow(134, 72, 214, 72, { color: "hot", w: 2 });
          mb(G, 214, 50, 130, 44, "instance\nport 22 open", { stroke: "hot", color: "hot", size: 11 });
          G.text(400, 78, "scanned by bots", { size: 12, color: "hot" });
          G.line(24, 130, 616, 130, { color: "line" });
          G.text(24, 160, "SSM: the instance dials out", { size: 13, color: "ok" });
          mb(G, 24, 190, 110, 56, "you\n(SSO login)", { stroke: "k", color: "k", size: 12 });
          mb(G, 250, 190, 120, 56, "AWS\nSession Manager", { stroke: "v", color: "v", size: 12 });
          const inst = mb(G, 486, 190, 130, 56, "instance\nno inbound rules", { stroke: "ok", color: "ok", size: 11 });
          G.arrow(134, 218, 250, 218, { color: "k", w: 2 }); G.arrow(486, 226, 370, 226, { color: "ok", w: 2, dash: "5 4" });
          G.label(380, 262, "outbound only", { color: "ok", size: 11 });
          G.rect(486, 282, 130, 40, { stroke: "ok", rx: 6 }); G.text(551, 307, "127.0.0.1:8000", { anchor: "middle", size: 12, color: "ok" });
          G.path("M 74 252 C 74 340, 300 340, 486 302", { color: "k", dash: "5 4", arrow: true });
          G.label(130, 352, "make forward: localhost:8000 → instance 127.0.0.1:8000", { size: 12, color: "k" });
          G.from(inst, { opacity: 0, duration: 0.4 });
          G.caption("a closed door cannot be knocked on");
        } },

      { rail: "terraform", title: "Describe it, preview it, apply it, destroy it",
        body: `<p>Clicking in a console makes things you forget. <b>Terraform</b> lets you write the resources you want in files and then run three verbs:</p>
<ul><li><code>terraform plan</code>: compare the files with reality and print the difference. Nothing changes.</li>
<li><code>terraform apply</code>: change the cloud to match the files.</li>
<li><code>terraform destroy</code>: delete everything Terraform created.</li></ul>
<p>Terraform keeps a <b>state</b>: its record of which real resources belong to your files. <code>terraform state list</code> prints them. After a clean destroy it prints nothing. That empty list is your evidence that nothing is left.</p>
<p>P3.2 teaches Terraform properly. Here you only run a stack someone else wrote, and you can read the plan before you apply.</p>`,
        scene(G) {
          mb(G, 24, 60, 150, 120, ".tf files\ninstance\ndisk · role\nbudget", { stroke: "k", color: "k", size: 13 });
          mb(G, 262, 60, 130, 50, "plan", { fill: "v", size: 14, rx: 8 }); G.label(262, 128, "preview the diff", { size: 11 });
          mb(G, 262, 150, 130, 50, "apply", { fill: "ok", size: 14, rx: 8 }); G.label(262, 218, "make it real", { size: 11 });
          mb(G, 262, 240, 130, 50, "destroy", { fill: "hot", size: 14, rx: 8 }); G.label(262, 308, "remove it all", { size: 11 });
          G.arrow(174, 120, 262, 85, { color: "muted" }); G.arrow(174, 120, 262, 175, { color: "muted" }); G.arrow(174, 120, 262, 265, { color: "muted" });
          const cloud = mb(G, 470, 60, 146, 150, "cloud\nresources", { stroke: "v", color: "v", size: 14 });
          G.arrow(392, 175, 470, 150, { color: "ok", w: 2 }); G.arrow(470, 245, 392, 265, { color: "hot", w: 2 });
          const st = mb(G, 470, 250, 146, 70, "state\nwhat Terraform owns", { stroke: "ink", size: 11, dash: "5 3" });
          G.text(24, 380, "terraform state list  →  empty after destroy", { size: 13, color: "ok" });
          G.from(cloud, { opacity: 0, duration: 0.5 }); G.from(st, { opacity: 0, duration: 0.5, delay: 0.3 });
          G.caption("plan shows, apply builds, destroy removes; state is the record");
        } },

      { rail: "the loop", title: "The lab loop, with the meter on only in the middle",
        body: `<p>Every lab uses the same four commands from <code>infra/aws/single-node</code>. This is exactly the flow in <code>P2.1 aws.md</code>.</p>
<div class="eq">make up        build: instance, disk, role, budget
make ssm       a shell on the instance (no SSH)
make forward   localhost:8000 -> the instance
make down      terraform destroy: delete it all</div>
<p><code>make up</code> runs <code>terraform apply</code>. <code>make forward</code> goes in a second terminal.</p>
<p>The meter runs between <code>make up</code> and <code>make down</code>. Two guards help if you forget: an <b>idle watchdog</b> stops the instance after 30 minutes of GPU under 5% with no SSM session, and a CloudWatch alarm stops it after 3 hours of near-zero CPU. Both only <i>stop</i>. The disk lives on until <code>make down</code>.</p>`,
        check: { q: "The idle watchdog stopped your instance overnight. Are you done paying?",
          options: ["Yes, stopped means free", "No, the disk still bills until make down", "No, the full hourly rate continues"], answer: 1,
          why: "The watchdog only stops the instance. Compute billing ends, but the EBS disk remains and bills per GB-month until make down destroys the stack." },
        scene(G) {
          const cx = 190, cy = 220, r = 120;
          const pts = [["make up", 0], ["make ssm", 1], ["make forward", 2], ["make down", 3]];
          const pos = (i) => { const a = -Math.PI / 2 + (i * Math.PI * 2) / 4; return [cx + Math.cos(a) * r, cy + Math.sin(a) * r]; };
          const ring = G.circle(cx, cy, r, { stroke: "line", fill: "none" });
          pts.forEach(([t, i]) => { const [x, y] = pos(i); mb(G, x - 62, y - 18, 124, 36, t, { fill: i === 3 ? "hot" : i === 0 ? "ok" : "k", size: 12, rx: 8 }); });
          G.text(cx, cy - 4, "lab loop", { anchor: "middle", size: 14 }); G.label(cx - 38, cy + 16, "work here", { size: 11 });
          G.rect(380, 70, 236, 30, { stroke: "line", rx: 4 }); G.label(580, 120, "time →", { size: 11 });
          const meter = G.rect(406, 70, 150, 30, { fill: "hot", rx: 4, opacity: 0.85 });
          G.line(406, 108, 406, 150, { color: "ok", dash: "3 3" }); G.text(406, 166, "make up", { anchor: "middle", size: 11, color: "ok" });
          G.line(556, 108, 556, 150, { color: "hot", dash: "3 3" }); G.text(556, 166, "make down", { anchor: "middle", size: 11, color: "hot" });
          G.text(480, 60, "cost meter runs", { anchor: "middle", size: 12, color: "hot" });
          mb(G, 380, 210, 236, 56, "idle watchdog: stop after\n30 idle minutes", { stroke: "v", color: "v", size: 11 });
          mb(G, 380, 282, 236, 56, "CloudWatch backstop: stop after\n3 h of near-zero CPU", { stroke: "v", color: "v", size: 11 });
          G.label(24, 392, "stop pauses compute; the disk waits for make down", { size: 12 });
          G.from(meter, { attr: { width: 0 }, duration: 1.2 });
          G.caption("the loop ends at make down, not at closing the laptop");
        } },

      { rail: "teardown", title: "Prove it is gone",
        body: `<p>"I think I deleted it" is not evidence. After every session, check:</p>
<ol><li><code>make down</code> ended with <code>Destroy complete</code>.</li>
<li><code>terraform state list</code> prints nothing.</li>
<li><code>../scripts/cost.sh</code> shows no running instance tagged <code>Project=silicon-to-serving</code>.</li>
<li>The console's EC2 pages (instances, volumes, snapshots, Elastic IPs) are empty in your region.</li>
<li>The billing page shows nothing growing that you do not recognise. It lags, so look again the next day.</li></ol>
<p>Write the result in your lab notes. It takes two minutes and is the cheapest insurance in this course.</p>`,
        scene(G) {
          const items = ["make down: Destroy complete", "terraform state list: (empty)", "cost.sh: no running instances", "console: no volumes, snapshots, IPs", "billing page: no new lines (check tomorrow)"];
          const ticks = [];
          items.forEach((t, i) => {
            const y = 60 + i * 66;
            G.rect(30, y, 580, 50, { stroke: "line", rx: 8 });
            G.rect(44, y + 12, 26, 26, { stroke: "ok", rx: 5 });
            ticks.push(G.path(`M 48 ${y + 26} L 55 ${y + 34} L 68 ${y + 16}`, { color: "ok", w: 3 }));
            G.text(90, y + 31, t, { size: 14 });
          });
          G.from(ticks, { opacity: 0, scale: 0.4, transformOrigin: "center", stagger: 0.25, duration: 0.3 });
          G.caption("five checks, two minutes, every session");
        } },

      { rail: "secrets", title: "Keys never enter the repo",
        body: `<p>The most common cloud disaster is a key pushed to a public repository. Bots find it within minutes and start machines. Rules:</p>
<ul><li>No keys, tokens or passwords in code, tests, notebooks or <code>terraform.tfvars</code>.</li>
<li>Prefer SSO and short-lived credentials (<code>aws configure sso</code>, <code>aws sso login</code>). If you must use environment variables, set them in your shell for the session only.</li>
<li>Commit only <code>terraform.tfvars.example</code>. Put <code>terraform.tfvars</code>, <code>*.tfstate*</code> and <code>.terraform/</code> in <code>.gitignore</code>.</li>
<li>Pass the Hugging Face token through SSM Parameter Store, as the single-node README does.</li></ul>
<p>If a key leaks, deactivate it in the console first. Removing it from git history comes second, because anyone may already have copied it. Exercise 3 writes the checks.</p>`,
        check: { q: "You committed a file containing an access key and pushed it. What do you do first?",
          options: ["Delete the file and push again", "Deactivate or delete the key in the console, then clean the history", "Nothing, the repo is private"], answer: 1,
          why: "Git history keeps the old commit, and bots watch for pushed keys. The key must be made useless immediately. History cleanup is secondary." },
        scene(G) {
          G.rect(24, 50, 270, 280, { stroke: "k", rx: 10 }); G.label(38, 74, "your repo (public or shared)", { color: "k" });
          mb(G, 44, 92, 230, 34, "terraform.tfvars.example", { stroke: "ok", color: "ok", size: 12 });
          mb(G, 44, 138, 230, 34, "main.tf", { stroke: "ok", color: "ok", size: 12 });
          mb(G, 44, 184, 230, 34, ".gitignore", { stroke: "ok", color: "ok", size: 12 });
          G.rect(330, 50, 286, 280, { stroke: "w", rx: 10, dash: "6 4" }); G.label(344, 74, "stays on your laptop only", { color: "muted" });
          const a = mb(G, 350, 92, 246, 34, "terraform.tfvars", { stroke: "hot", color: "hot", size: 12 });
          const b = mb(G, 350, 138, 246, 34, "terraform.tfstate", { stroke: "hot", color: "hot", size: 12 });
          const c = mb(G, 350, 184, 246, 34, "keys, tokens, passwords", { stroke: "hot", color: "hot", size: 12 });
          const arrow = G.arrow(350, 110, 276, 110, { color: "hot", w: 2, dash: "4 3" });
          G.text(313, 100, "✕", { anchor: "middle", size: 22, color: "hot", weight: 700 });
          G.text(44, 256, ".gitignore lists:", { size: 12 }); G.text(44, 278, "terraform.tfvars", { size: 12, color: "ok" }); G.text(44, 298, "*.tfstate*   .terraform/", { size: 12, color: "ok" });
          G.label(24, 366, "commit the example file; keep the real one local", { size: 12 });
          G.from([a, b, c], { opacity: 0, x: 10, stagger: 0.15, duration: 0.3 });
          G.caption("examples go in git; secrets and state do not");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Cloud cost is <code>hours × hourly price</code>. The price is an input you look up. The danger is the forgotten instance.</li>
<li>Lock root with MFA. Work through IAM Identity Center with short-lived credentials.</li>
<li>Choose one region, and request the GPU vCPU quota on day one.</li>
<li>A budget emails you. It does not stop anything. A stopped instance still pays for its disk, and some resources bill with no instance at all.</li>
<li>Session Manager gives a shell with no open port. Terraform gives plan, apply and destroy, and a state list that proves teardown.</li>
<li>The lab loop is <code>make up</code>, <code>make ssm</code>, <code>make down</code>. Check the teardown list every time. Keep secrets out of git.</li></ul>
<p>You can now run the <code>P2.1 aws.md</code> flow safely. Next is P0.1. Tick every box of the account checklist before you reach P2.1.</p>
<h2>Go further</h2><ul>
<li><a href="https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html">AWS Budgets user guide</a></li>
<li><a href="https://docs.aws.amazon.com/singlesignon/latest/userguide/what-is.html">IAM Identity Center user guide</a></li>
<li><a href="https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html">Systems Manager Session Manager</a></li>
<li><a href="https://developer.hashicorp.com/terraform/tutorials/aws-get-started">HashiCorp: Get started with Terraform on AWS</a></li></ul>
<p>These links are UNVERIFIED from the build environment. Check them.</p>`,
    sim: {
      title: "What will this cost?",
      intro: "Type in the price you looked up for your region (the starting value is a made-up example, not a real price), then describe your plan. The bar splits the total into lab time, the disk between sessions, and the forgotten days. The gauge compares the total with your budget. This is a model: it ignores taxes, data transfer and anything you create by hand.",
      height: 230,
      controls: [
        { id: "price", label: "hourly price (example value: enter the current price for your region)", min: 0.25, max: 20, step: 0.05, value: 2, format: (v) => v.toFixed(2) },
        { id: "hours", label: "hours per session", min: 1, max: 8, step: 0.5, value: 3 },
        { id: "per", label: "sessions per week", min: 1, max: 7, value: 3 },
        { id: "weeks", label: "weeks", min: 1, max: 12, value: 4 },
        { id: "gb", label: "disk size, GB", min: 50, max: 500, step: 10, value: 200 },
        { id: "gbp", label: "disk price per GB-month (example value: check the price page)", min: 0.02, max: 0.3, step: 0.01, value: 0.1, format: (v) => v.toFixed(2) },
        { id: "keep", label: "between sessions", type: "select", value: "down", options: [["down", "make down each time"], ["stop", "only stop (disk kept)"]] },
        { id: "forgot", label: "forgot to tear down for N days", min: 0, max: 14, value: 2 },
        { id: "budget", label: "your budget", min: 20, max: 1000, step: 10, value: 200 },
      ],
      draw(G, v) {
        const lab = v.price * v.hours * v.per * v.weeks;
        const sessionDays = (v.hours * v.per * v.weeks) / 24;
        const diskDays = v.keep === "stop" ? v.weeks * 7 : sessionDays;
        const disk = v.gb * v.gbp * (diskDays / 30);
        const lost = v.price * 24 * v.forgot;
        const total = lab + disk + lost;
        const scale = Math.max(total, v.budget) * 1.05 || 1, W = 570;
        G.label(10, 18, "where the money goes", { size: 12 });
        let x = 24;
        [[lab, "k", "lab time"], [disk, "w", "disk"], [lost, "hot", "forgotten"]].forEach(([val, col, name], i) => {
          const w = (val / scale) * W;
          if (w > 0.5) G.rect(x, 40, w, 46, { fill: col, rx: 3, opacity: 0.9 });
          if (w > 70) G.text(x + w / 2, 68, name, { anchor: "middle", size: 12, color: "bg", weight: 700 });
          x += w;
        });
        const bx = 24 + (v.budget / scale) * W;
        G.line(bx, 30, bx, 98, { color: "ink", w: 2, dash: "4 3" }); G.label(bx - 2, 112, "budget " + money(v.budget), { size: 11, anchor: "middle" });
        G.rect(24, 140, 12, 12, { fill: "k", rx: 2 }); G.label(42, 151, "lab time " + money(lab), { size: 11 });
        G.rect(190, 140, 12, 12, { fill: "w", rx: 2 }); G.label(208, 151, "disk " + money(disk), { size: 11 });
        G.rect(330, 140, 12, 12, { fill: "hot", rx: 2 }); G.label(348, 151, "forgotten " + money(lost), { size: 11 });
        G.label(10, 200, "model, not a bill: example prices, no tax, no data transfer", { size: 11 });
        return [
          { title: "Total", rows: [["lab time", money(lab)], ["disk", money(disk)], ["forgotten days", money(lost)], ["total", money(total)]] },
          { title: "From forgetting", rows: [["share of total", total > 0 ? F.num((lost / total) * 100, 0) + "%" : "0%"], ["one forgotten day costs", money(v.price * 24)], ["= sessions of this length", F.num(24 / v.hours, 1)]] },
          { title: "Against your budget", gauge: [[Math.min(1, total / v.budget), total > v.budget ? "hot" : total > v.budget * 0.8 ? "v" : "ok"]], gaugeText: `${money(total)} of ${money(v.budget)}`,
            chip: [total <= v.budget * 0.8, total <= v.budget * 0.8 ? "within budget, 20% spare" : total <= v.budget ? "inside budget, no spare" : "over budget"] },
        ];
      },
      note: "Prices are example values you type in. This is the same arithmetic as exercise 1.",
    },
    practice: {
      intro: `Run these from the repository root. The account checklist is the real deliverable: do it before P2.1. Cloud steps are marked <code>TODO(run)</code> because they need your own AWS account.`,
      items: [
        { title: "Cost calculator", tier: "T0 · easy", goal: "Write session_cost, idle_cost, forgotten_cost and budget_ok. Then run them with the real price for your region.",
          cmd: `uv run pytest ${MODULE}/exercises/01-cost-calc` },
        { title: "Account checklist", tier: "T0 · do before P2.1", goal: "Root MFA, SSO user, budget alarm, region, GPU quota, sts get-caller-identity via SSO, guardrails applied. Tick each box with evidence.",
          cmd: `cat ${MODULE}/exercises/02-account-checklist/README.md` },
        { title: "tfvars hygiene", tier: "T0 · medium", goal: "Flag secret-looking values in a tfvars file, and check that a .gitignore really covers terraform.tfvars but not the example file.",
          cmd: `uv run pytest ${MODULE}/exercises/03-tfvars-hygiene` },
        { title: "All reference solutions", tier: "T0 · check", goal: "Confirm the solutions pass.",
          cmd: `S2S_SOLUTIONS=1 uv run pytest ${MODULE}` },
      ],
      labs: [
        { label: "The flow you will run in P2.1", path: "course/P2-serving-engines/P2.1-first-serve-with-vllm/aws.md" },
        { label: "Single-node stack: README, Makefile, tfvars example", path: "infra/aws/single-node/README.md" },
        { label: "Budget guardrail (Terraform)", path: "infra/aws/guardrails/main.tf" },
        { label: "Cost and running-instance check script", path: "infra/aws/scripts/cost.sh" },
        { label: "Self-check questions", path: `${MODULE}/quiz.md` },
      ],
    },
  });
})();
