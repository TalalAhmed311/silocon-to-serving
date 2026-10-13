/* P4.3 — Distributed jobs, checkpointing and fault tolerance (project #9). */
(function () {
  const F = S2S.fmt;
  const MOD = "course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing";
  const RC = ["k", "v", "q", "pink"];

  /* Young / Daly and the first-order waste model, as in platform/training/faults.py */
  const young = (C, M) => Math.sqrt(2 * C * M);
  const daly = (C, M) => (C >= 2 * M ? M : Math.sqrt(2 * C * M) * (1 + Math.sqrt(C / (2 * M)) / 3 + C / (18 * M)) - C);
  const waste = (t, C, M, R = 0) => C / (t + C) + (t + C) / (2 * M) + R / M;
  const dur = (s) => (s >= 7200 ? (s / 3600).toFixed(1) + " h" : s >= 120 ? (s / 60).toFixed(1) + " min" : s.toFixed(0) + " s");
  const nice = (x) => { const e = 10 ** Math.floor(Math.log10(x)); return [1, 2, 5, 10].map((k) => k * e).find((k) => k >= x); };

  S2S.lesson({
    id: "p4-3", n: "P4.3", title: "Distributed jobs and checkpointing",
    subtitle: "Multi-GPU and distributed · first principles · T0 on CPU, T3 for the EKS run",
    kicker: "Lesson · ≈ 45 min",
    headline: "Surviving the day a GPU dies",
    intro: `<p>A training run on many machines lasts days. Somewhere in that time, a process crashes or a machine disappears. This lesson builds the standard defence from scratch: save the job's state to disk now and then (a <b>checkpoint</b>), make the save safe against crashes, restart automatically, and choose how often to save with a short piece of maths. You will meet the pieces that project #9 (<code>platform/training/</code>) assembles.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "5 exercises"],
    legend: [["k", "rank 0"], ["v", "rank 1"], ["q", "rank 2"], ["pink", "rank 3"], ["ok", "committed / good"], ["hot", "failure"]],
    prev: "p4-2", next: "p4-4",
    steps: [
      { rail: "failures", title: "Big jobs fail often, even if each machine rarely does",
        body: `<p>Suppose one machine fails, on average, once every 720 hours (30 days). That is an <b>example value</b> for illustration; real fleets differ, and exercise 3 asks you to look up published numbers. The average time between failures is the <b>MTBF</b> (mean time between failures).</p>
<p>A training job needs <i>all</i> its machines. If any one fails, the job stops. So failures arrive N times as often as for one machine:</p>
<div class="eq">job MTBF  M = node MTBF / N

N = 1:    720 h
N = 2:    360 h
N = 16:    45 h
N = 128:  5.6 h
(N = 128 is a 1,024-GPU job at 8 GPUs per node)</div>
<p>A 5.6-hour interval between failures means a multi-day run is interrupted dozens of times. Waiting for perfect hardware does not work. The answer is to make failure cheap: write the job's state to storage regularly, and on any failure restart from the last copy. The rest of the lesson is how to do that correctly.</p>`,
        scene(G) {
          const Ns = [1, 2, 16, 128], M = Ns.map((n) => 720 / n), x0 = 120, W = 440;
          const X = (h) => x0 + (W * Math.log10(h)) / 3;
          G.text(24, 34, "time between failures of the whole job (hours, log scale)", { size: 13 });
          [1, 10, 100, 1000].forEach((t) => { G.line(X(t), 56, X(t), 290, { color: "line", w: 0.7 }); G.label(X(t), 310, t + " h", { anchor: "middle", size: 11 }); });
          const bars = Ns.map((n, i) => {
            const y = 70 + i * 54;
            G.text(x0 - 10, y + 24, `N = ${n}`, { anchor: "end", size: 13 });
            const b = G.rect(x0, y, X(M[i]) - x0, 36, { fill: i === 3 ? "hot" : "k", rx: 4 });
            G.text(X(M[i]) + 8, y + 24, (M[i] >= 100 ? M[i].toFixed(0) : M[i].toFixed(1).replace(/\.0$/, "")) + " h", { size: 13 });
            return b;
          });
          G.label(24, 354, "example: one node fails every 720 h on average (placeholder, not measured)", { size: 12 });
          G.label(24, 376, "the job needs every node, so its failure rate is N times higher", { size: 12, color: "ink" });
          G.from(bars, { attr: { width: 0 }, stagger: 0.12, duration: 0.5 });
          G.caption("more machines, shorter time between failures");
        }, legend: [["k", "job MTBF"], ["hot", "128 nodes"]] },

      { rail: "what to save", title: "A checkpoint is everything needed to continue, not only the weights",
        body: `<p>Restarting must put the job back <i>exactly</i> where it was. What is that "where"? Training repeats: read a batch, compute gradients, let the optimizer update the weights. The next step depends on more than the weights:</p>
<ul><li><b>Model weights.</b> The thing being trained.</li>
<li><b>Optimizer state.</b> Adam keeps two running averages per weight, m and v (P4.2), which scale every update. Lose them and the first updates after the restart are the wrong size, so the run no longer follows the same path.</li>
<li><b>Step counter and learning-rate schedule position.</b> Warm-up and decay depend on the step.</li>
<li><b>Data position (and random seeds).</b> Otherwise the restart repeats batches or skips some.</li></ul>
<p>How big is it? Mixed-precision Adam needs up to 16 bytes per parameter (P4.2): weights, gradients and optimizer state. Gradients are recomputed, so a checkpoint is a bit smaller, but for an 8.0 B model it is still on the order of 100 GB, at most 8.0 B × 16 B = 128 GB. Saving that takes real time, which is why how often we save matters.</p>
<p>In the repo, <code>platform/training/train.py</code> saves the model <i>and</i> AdamW state together; exercise 1 checks them for bitwise equality after a reload.</p>`,
        scene(G) {
          const rows = [["model weights", "k", "the thing being trained"], ["optimizer state m, v", "q", "scales every update"], ["step + LR schedule", "v", "warm-up and decay position"], ["data position, seeds", "pink", "same batches, no repeats"]];
          G.text(24, 34, "what the checkpoint holds", { size: 14 });
          const bx = rows.map(([t, c, why], i) => {
            const y = 54 + i * 64;
            const b = G.box(24, y, 230, 48, t, { fill: c, size: 13 });
            G.arrow(258, y + 24, 286, y + 24, { color: "muted" });
            G.text(294, y + 18, why, { size: 12 });
            return b;
          });
          G.label(294, 96, "without it: the model itself is gone", { size: 11 });
          G.label(294, 160, "without it: first updates are the wrong size", { size: 11 });
          G.label(294, 224, "without it: LR restarts from the wrong place", { size: 11 });
          G.label(294, 288, "without it: batches repeat or are skipped", { size: 11 });
          G.line(24, 322, 616, 322, { color: "line" });
          G.text(24, 346, "size: up to 16 B per parameter (P4.2)", { size: 13, color: "ink" });
          G.label(24, 368, "8.0 B parameters × 16 B = 128 GB at most; about 100 GB in practice", { size: 12 });
          G.from(bx, { opacity: 0, x: -20, stagger: 0.12, duration: 0.4 });
          G.caption("miss one piece and the resumed run is a different run");
        } },

      { rail: "sharded save", title: "Each rank writes its own shard, in parallel",
        body: `<p>With FSDP (P4.2) every rank holds only a shard of the weights and optimizer state. The naive save gathers everything on rank 0 and writes one file. That needs rank 0 to hold the whole 128 GB, and one writer moves all of the data.</p>
<p>PyTorch's <b>Distributed Checkpoint (DCP)</b> does better. Each rank writes <b>its own shard</b> straight to storage, all at the same time, plus one small <b>.metadata</b> file that records, for every tensor, its full shape and which file holds which slice.</p>
<div class="eq">step_000019/
  __0_0.distcp   rank 0's shard
  __1_0.distcp   rank 1's shard
  ...
  .metadata      global shapes, slice locations
dcp.save(state, checkpoint_id=dir)
(every rank calls it)</div>
<p>Why it matters, with <b>example numbers</b> (storage that scales with writers; real storage may not): 128 GB through one writer at 1 GB/s takes 128 s. Sixteen ranks each writing 8 GB at 1 GB/s take 8 s. Training is paused for that time, so a faster save is a cheaper checkpoint (cost C in step 9). The file names above follow the example in the module README (<code>__0_0.distcp</code>); other ranks follow the same pattern. See <code>platform/training/checkpoint.py</code> and <code>examples/02_dcp_save_load.py</code>.</p>`,
        scene(G) {
          const xs = (r) => 20 + r * 150;
          const items = [];
          for (let r = 0; r < 4; r++) {
            items.push(G.box(xs(r), 50, 130, 50, `rank ${r} shard`, { fill: RC[r], size: 13 }));
            items.push(G.arrow(xs(r) + 65, 104, xs(r) + 65, 150, { color: RC[r], w: 2.5 }));
            items.push(G.box(xs(r), 154, 130, 40, `__${r}_0.distcp`, { stroke: RC[r], size: 12, color: "ink", sw: 2 }));
          }
          G.text(24, 36, "4 ranks write at the same time", { size: 13 });
          G.box(20, 224, 580, 44, ".metadata: global shape of every tensor + which file holds which slice", { stroke: "ink", size: 12, color: "ink" });
          G.text(24, 306, "naive: gather all on rank 0, one writer", { size: 13, color: "hot" });
          G.label(24, 326, "128 GB at 1 GB/s = 128 s   (example values)", { size: 12 });
          G.text(24, 356, "sharded: 16 writers, 8 GB each", { size: 13, color: "ok" });
          G.label(24, 376, "8 GB at 1 GB/s = 8 s   (if the storage keeps up)", { size: 12 });
          G.from(items, { opacity: 0, y: -8, stagger: 0.06, duration: 0.3 });
          G.caption("no rank ever holds the whole state; many writers share the work");
        } },

      { rail: "resharding", title: "Save on 2 ranks, resume on 4",
        body: `<p>After a node loss you may have fewer GPUs; after scaling up, more. The new job has a different <b>world size</b> (number of ranks), so each rank needs a different slice than the one any old file contains. Does the checkpoint have to be saved again?</p>
<p>No, because <code>.metadata</code> stores the global layout. At load time each new rank asks: "which rows of this tensor are mine?", looks up in the metadata which files hold them, and reads only those byte ranges. This is <b>resharding</b>.</p>
<p>Take a tensor with 8 rows saved by 2 ranks. File 0 holds rows 0–3, file 1 rows 4–7. Four new ranks need 2 rows each:</p>
<div class="eq">new rank 0: rows 0–1  ← file 0, first half
new rank 1: rows 2–3  ← file 0, second half
new rank 2: rows 4–5  ← file 1, first half
new rank 3: rows 6–7  ← file 1, second half</div>
<p>Exercise 2 does exactly this at small scale: it trains on 2 ranks, resumes on 4, and requires the final loss to match an uninterrupted run to within 0.2%.</p>`,
        check: { q: "A checkpoint was saved by 2 ranks. How can 4 ranks load it?",
          options: ["It cannot: the world size must match", "Rank 0 loads everything and broadcasts it", "The metadata maps every slice of every tensor to a file, so each new rank reads just the byte ranges it needs", "The checkpoint is re-saved as 4 files first"], answer: 2,
          why: "DCP records each tensor's global shape and where each saved slice lives. A rank in the new layout works out which global slices it owns and reads exactly those ranges, so no rewrite and no single-rank bottleneck are needed." },
        scene(G) {
          G.text(24, 34, "saved by 2 ranks (8 rows)", { size: 13 });
          const old = [0, 1].map((f) => G.box(40 + f * 270, 50, 250, 44, `file ${f}: rows ${f * 4}–${f * 4 + 3}`, { fill: RC[f], size: 13 }));
                    const nw = [], ar = [];
          G.text(24, 322, "loaded by 4 ranks (2 rows each)", { size: 13 });
          for (let r = 0; r < 4; r++) {
            const f = r < 2 ? 0 : 1, x = 40 + r * 140;
            nw.push(G.box(x, 250, 124, 44, `rank ${r}: rows ${r * 2}–${r * 2 + 1}`, { fill: RC[f], size: 12 }));
            ar.push(G.arrow(f === 0 ? 100 + (r % 2) * 90 : 360 + (r % 2) * 90, 98, x + 62, 246, { color: RC[f], w: 2 }));
          }
          G.label(24, 346, "colour = which file the data came from", { size: 12 });
          G.label(24, 368, "each rank reads only the byte ranges of its own rows", { size: 12, color: "ink" });
          G.label(24, 390, "the layout is decided at load time, not at save time", { size: 12, color: "ink" });
          G.from(ar, { opacity: 0, stagger: 0.12, duration: 0.4 }); G.from(nw, { opacity: 0.3, stagger: 0.12, duration: 0.3 });
          G.caption("2 files in, 4 shards out: the metadata makes it possible");
        } },

      { rail: "safe commit", title: "A crash in the middle of a save must not look like a good checkpoint",
        body: `<p>Saving takes time, and the crash may strike <i>during</i> it. Imagine rank 3's node dies after ranks 0–2 finished writing but before its own shard was flushed. The checkpoint directory exists. Three of four files are there. A restart that trusts "the newest directory" would load a broken checkpoint.</p>
<p>DCP does not solve this; you do. The rule used in <code>platform/training/checkpoint.py</code>:</p>
<ol><li>Every rank writes its shard.</li>
<li>All ranks wait at a <b>barrier</b> (a point no rank passes until all have reached it), so every shard is on disk.</li>
<li>Only then does rank 0 write a tiny file, <code>COMMITTED</code>.</li></ol>
<p>On resume, only directories that contain <code>COMMITTED</code> count (<code>latest_committed</code>). A half-written checkpoint has no marker and is ignored; the job falls back to the previous one. Pruning follows the same care: keep the newest <code>keep</code> committed checkpoints, delete older ones, and never touch a newer, still-unfinished save.</p>
<p>Exercise 1 asks you to remove the barrier and explain what breaks: rank 0 can finish first and write <code>COMMITTED</code> while rank 3's shard is still missing, which is exactly the failure above.</p>`,
        check: { q: "Why does rank 0 write the COMMITTED marker only after a barrier?",
          options: ["Barriers make the files smaller", "So that a marker means every rank's shard is already on disk", "Because rank 0 is the fastest writer", "So other ranks can delete old checkpoints"], answer: 1,
          why: "Without the barrier, rank 0 could mark the checkpoint complete while another rank is still writing, or has died. With it, 'has a marker' is equivalent to 'every shard was written', so resume can trust the marker." },
        scene(G) {
          const lens = [190, 150, 220, 330];
          const panel = (y0, title, crash) => {
            G.text(24, y0 - 8, title, { size: 13 });
            const out = [];
            for (let r = 0; r < 4; r++) {
              const y = y0 + r * 26;
              G.label(24, y + 15, `rank ${r}`, { size: 11 });
              const w = crash && r === 3 ? 150 : lens[r];
              out.push(G.rect(80, y, w, 18, { fill: crash && r === 3 ? "hot" : RC[r], rx: 3 }));
            }
            return out;
          };
          const a = panel(60, "normal: every shard lands", false);
          G.line(410, 54, 410, 166, { color: "ink", dash: "5 4", w: 2 }); G.label(414, 70, "barrier", { size: 11, color: "ink" });
          const m1 = G.box(430, 100, 150, 30, "COMMITTED", { fill: "ok", size: 12 });
          G.label(430, 150, "written by rank 0", { size: 11 });
          const b = panel(222, "rank 3's node dies mid-write", true);
          G.text(238, 316, "✕", { size: 18, color: "hot" });
          G.box(430, 252, 150, 30, "no marker", { stroke: "hot", size: 12, color: "hot", dash: "4 3" });
          G.label(430, 302, "restart ignores this directory", { size: 11 });
          G.label(24, 380, "resume uses only directories that contain COMMITTED", { size: 12, color: "ink" });
          G.from([...a, ...b], { attr: { width: 0 }, stagger: 0.06, duration: 0.5 }); G.from(m1, { opacity: 0, delay: 0.6, duration: 0.3 });
          G.caption("the marker turns \"visible\" into \"complete\"");
        }, legend: [["ok", "committed"], ["hot", "failed write"]] },

      { rail: "restart loop", title: "Recovery is just: start the job again",
        body: `<p>With committed checkpoints, recovery needs no special logic. The training script does the same thing at every start: look for the newest committed checkpoint; if there is one, load it and continue from the step after it; if not, start from scratch.</p>
<p>So whoever launches the job only has to <b>run it again</b> when it dies. The three common launchers differ in who does that:</p>
<ul><li><b>torchrun:</b> its elastic agent starts the processes; <code>--max-restarts</code> restarts the group after a failure.</li>
<li><b>Ray Train on KubeRay:</b> Ray workers run in pods; <code>FailureConfig(max_failures)</code> restarts the worker group, and KubeRay replaces lost pods.</li>
<li><b>Slurm:</b> <code>srun</code> starts the processes; <code>--requeue</code> resubmits the job.</li></ul>
<p>Every launcher sets up the same thing: a <b>rendezvous</b> (the point where all processes meet to agree on who is rank 0, 1, 2…) and the ranks. The course's primary path is Ray on EKS (it reuses the P3 cluster); Slurm on ParallelCluster is the alternative. Both run the same <code>platform/training/train.py</code>.</p>`,
        scene(G) {
          const b1 = G.box(60, 50, 280, 46, "launcher starts all ranks", { fill: "blue", size: 13 });
          const b2 = G.box(60, 120, 280, 46, "find newest COMMITTED step", { fill: "k", size: 13 });
          const b3 = G.box(60, 190, 280, 46, "load shards, continue at step + 1", { fill: "q", size: 13 });
          const b4 = G.box(60, 260, 280, 46, "train; save + commit every N steps", { fill: "ok", size: 13 });
          [[96, 120], [166, 190], [236, 260]].forEach(([a, b]) => G.arrow(200, a + 2, 200, b - 2, { color: "muted", w: 2 }));
          const cr = G.box(430, 190, 160, 56, "crash", { fill: "hot", size: 14 });
          G.arrow(342, 283, 430, 232, { color: "hot", w: 2 });
          G.path("M 510 188 C 510 120, 440 73, 344 73", { color: "hot", w: 2, arrow: true });
          G.text(356, 160, "launcher", { size: 12, color: "hot" }); G.text(356, 176, "restarts it", { size: 12, color: "hot" });
          G.label(430, 262, "process · pod · node", { size: 11 });
          G.label(60, 340, "torchrun --max-restarts · Ray FailureConfig · Slurm --requeue", { size: 12, color: "ink" });
          G.label(60, 362, "the script never needs to know why it restarted", { size: 12 });
          G.from([b1, b2, b3, b4, cr], { opacity: 0, y: 8, stagger: 0.12, duration: 0.35 }); G.pulse(cr, { repeat: 4 });
          G.caption("one loop covers every kind of failure");
        }, legend: [["blue", "launcher"], ["hot", "failure"]] },

      { rail: "same data", title: "The data must not depend on the world size",
        body: `<p>Resuming is only correct if the run sees the same batches it would have seen. Here is a trap. If each rank draws its random batches with a seed based on <i>its rank number</i>, a restart on a different number of ranks changes who draws what, so the data order changes silently. The loss still falls, nothing crashes, and you can no longer reproduce or compare runs.</p>
<p>The fix, used in <code>train.batch()</code> in <code>platform/training/train.py</code>: fix the <b>global batch</b> and build every sequence from <code>(step, sequence index)</code>, never from the rank. Rank r just takes a slice of the indices.</p>
<div class="eq">global batch = 16 sequences (ids 0–15)
sequence i of step s: seed s × 1,000,003 + i
world 2: rank 0 takes ids 0–7, rank 1 takes 8–15
world 4: rank r takes ids 4r … 4r+3</div>
<p>The same 16 sequences appear at every world size. That is why exercise 2 can expect the loss after a 2 → 4 resume to match the uninterrupted run. (The global batch must divide evenly by the world size; the code asserts this.)</p>`,
        scene(G) {
          const grid = (y, n, title) => {
            G.text(24, y - 10, title, { size: 13 });
            const out = [];
            for (let i = 0; i < 16; i++) {
              const r = Math.floor((i * n) / 16);
              out.push(G.box(24 + i * 37, y, 33, 36, String(i), { fill: RC[r], size: 11, rx: 4 }));
            }
            return out;
          };
          const a = grid(70, 2, "world 2: ranks 0 and 1"), b = grid(170, 4, "world 4: ranks 0, 1, 2, 3");
          G.text(24, 260, "sequence i of step s is always built from", { size: 12, color: "ink" });
          G.text(24, 282, "seed = s × 1,000,003 + i      (never from the rank)", { size: 13, color: "ok" });
          G.label(24, 320, "same 16 sequences, only the owner changes", { size: 12, color: "ink" });
          G.label(24, 342, "rank-based seeds would change the data order after a resize", { size: 12, color: "hot" });
          G.from([...a, ...b], { opacity: 0, stagger: 0.02, duration: 0.25 });
          G.caption("resizing the job must not change which data is read");
        }, legend: [["k", "rank 0"], ["v", "rank 1"], ["q", "rank 2"], ["pink", "rank 3"]] },

      { rail: "RPO and RTO", title: "How much work is lost, and how long until training resumes",
        body: `<p>Two numbers describe how good your recovery is.</p>
<ul><li><b>RPO</b> (recovery point objective): how much work is lost, measured in steps or seconds since the last committed checkpoint.</li>
<li><b>RTO</b> (recovery time objective): how long the job is not training: detect the failure, get machines back, start processes, load the checkpoint, warm up.</li></ul>
<p>Worked example (the "predict first" from the module README). Checkpoint every 100 steps at 0.5 s per step; they commit after steps 99, 199, … The job crashes during step 1,337.</p>
<div class="eq">last commit: step 1,299  → resume at step 1,300
RPO = 1,337 − 1,300 = 37 steps = 37 × 0.5 s = 18.5 s
RTO = 20 s reschedule + 15 s load = 35 s
      (plus warm-up)
back where we were: 35 + 18.5 = 53.5 s</div>
<p>The repo's own check in <code>platform/training/rto.py</code> uses the same counting: the local drill in the walkthrough dies at step 37, resumes from 19, and reports an RPO of 37 − 20 = 17 steps. On average the RPO is half the checkpoint interval τ/2. The RTO depends on the failure: a crashed process restarts in seconds; a lost node waits for a replacement machine, which takes minutes (exercise 4 measures the real values on EKS).</p>`,
        check: { q: "Checkpoints commit after steps 99, 199, 299, … at 0.5 s per step. The job dies during step 450. What is the RPO?",
          options: ["50 steps = 25 s", "51 steps = 25.5 s", "150 steps = 75 s", "1 step"], answer: 0,
          why: "The last commit is after step 399, so training resumes at step 400. Steps 400 to 449 (50 steps) were done and are lost: 50 × 0.5 s = 25 s. (Step 450 itself was in progress.)" },
        scene(G) {
          const X = (s) => 40 + ((s - 1280) / 80) * 560;
          G.text(24, 34, "steps around the failure", { size: 13 });
          G.line(40, 110, 600, 110, { color: "line", w: 2 });
          [1280, 1300, 1320, 1340, 1360].forEach((s) => { G.line(X(s), 104, X(s), 116, { color: "muted" }); G.label(X(s), 136, String(s), { anchor: "middle", size: 10 }); });
          G.line(X(1300), 70, X(1300), 116, { color: "ok", w: 3 }); G.label(X(1300) - 6, 66, "commit after step 1,299", { anchor: "end", color: "ok" });
          G.line(X(1337), 70, X(1337), 116, { color: "hot", w: 3 }); G.label(X(1337) + 6, 66, "crash in step 1,337", { color: "hot" });
          const lost = G.rect(X(1300), 92, X(1337) - X(1300), 14, { fill: "hot", rx: 3, opacity: 0.8 });
          G.text(X(1318.5), 160, "RPO: 37 steps", { anchor: "middle", size: 13, color: "hot" });
          G.label(X(1318.5), 178, "= 18.5 s of work", { anchor: "middle", size: 12 });
          G.text(24, 230, "RTO (example values)", { size: 13 });
          const tot = 35, sc = 560 / 60;
          const a = G.rect(40, 246, 20 * sc / 1.0 * (1), 34, { fill: "v", rx: 3 }), b = G.rect(40 + 20 * sc, 246, 15 * sc, 34, { fill: "q", rx: 3 });
          G.text(40 + 10 * sc, 268, "reschedule 20 s", { anchor: "middle", size: 12, color: "bg" });
          G.text(40 + 27.5 * sc, 268, "load 15 s", { anchor: "middle", size: 12, color: "bg" });
          G.text(40 + tot * sc + 10, 268, "= 35 s", { size: 13 });
          G.label(40, 316, "plus 18.5 s to redo the lost steps: 53.5 s after the crash", { size: 12, color: "ink" });
          G.label(40, 338, "process crash: seconds · node loss: minutes (waiting for a new node)", { size: 12 });
          G.from([lost, a, b], { attr: { width: 0 }, stagger: 0.2, duration: 0.5 });
          G.caption("RPO is lost work; RTO is lost time");
        }, legend: [["hot", "lost work (RPO)"], ["v", "reschedule"], ["q", "load"], ["ok", "last commit"]] },

      { rail: "how often", title: "Saving too often wastes time; too rarely wastes work",
        body: `<p>How often should the job checkpoint? Two costs pull in opposite directions. Let C be the seconds training pauses to save, M the job's MTBF, and τ the interval between saves (in seconds of training).</p>
<ul><li><b>Saving costs C every τ:</b> the fraction of time lost to saving is about C / (τ + C). Larger τ makes it smaller.</li>
<li><b>Failing costs the work since the last save,</b> on average half an interval. Failures happen once per M, so the fraction lost is about (τ + C) / (2M). Larger τ makes it bigger.</li></ul>
<div class="eq">waste(τ) ≈ C/(τ+C) + (τ+C)/(2M) + R/M
(R = restart cost)
minimum at   τ* ≈ √(2 C M)   (Young, 1974)</div>
<p>Worked example, C = 30 s and M = 4 h = 14,400 s:</p>
<div class="eq">τ* = √(2 × 30 × 14,400) = √864,000 ≈ 930 s ≈ 15.5 min
waste(τ*) = 30/960 + 960/28,800 = 3.1% + 3.3% ≈ 6.5%</div>
<p>At the optimum the two costs are about equal. The curve is flat near τ*, so being off by 2× either way costs little; being off by 10× costs a lot. <code>platform/training/faults.py</code> has <code>young</code>, <code>daly</code> (a refinement), <code>waste_fraction</code> and a Monte-Carlo <code>simulate</code> that exercise 3 uses to check the formula.</p>`,
        check: { q: "C = 30 s and M = 4 hours. What is Young's optimal checkpoint interval?",
          options: ["about 15 minutes", "about 2 minutes", "about 4 hours", "about 30 seconds"], answer: 0,
          why: "τ* = √(2·C·M) = √(2 × 30 × 14,400) ≈ 930 s, about 15.5 minutes. Checking: 4 hours is the MTBF itself (far too rare), and 30 s is C (the job would spend half its time saving)." },
        scene(G) {
          const C = 30, M = 14400, lo = 1.3, hi = 4.2;                                  // log10 τ in seconds (20 s … 4.4 h)
          const X = (t) => 70 + (540 * (Math.log10(t) - lo)) / (hi - lo), ymax = 0.25, Y = (w) => 300 - (240 * Math.min(w, ymax)) / ymax;
          G.text(24, 34, "wasted fraction of time vs checkpoint interval τ", { size: 13 });
          G.axes(70, 60, 540, 240, {});
          [0, 0.1, 0.2].forEach((w) => { G.line(70, Y(w), 610, Y(w), { color: "line", w: 0.5 }); G.label(64, Y(w) + 4, (w * 100).toFixed(0) + "%", { anchor: "end", size: 10 }); });
          [[30, "30 s"], [300, "5 min"], [3600, "1 h"]].forEach(([t, l]) => G.label(X(t), 318, l, { anchor: "middle", size: 10 }));
          G.label(610, 336, "τ (log) →", { anchor: "end", size: 11 });
          const curve = (f) => { let d = ""; for (let e = lo; e <= hi + 1e-9; e += 0.03) { const t = 10 ** e; d += (d ? " L " : "M ") + X(t).toFixed(1) + " " + Y(f(t)).toFixed(1); } return d; };
          const p1 = G.path(curve((t) => C / (t + C)), { color: "q", w: 2, dash: "6 4" });
          const p2 = G.path(curve((t) => (t + C) / (2 * M)), { color: "v", w: 2, dash: "6 4" });
          const p3 = G.path(curve((t) => waste(t, C, M)), { color: "k", w: 3 });
          const ts = young(C, M);
          G.line(X(ts), 60, X(ts), 300, { color: "ok", dash: "4 4" });
          G.circle(X(ts), Y(waste(ts, C, M)), 5, { fill: "ok" });
          G.label(X(ts) + 8, 78, "τ* ≈ 930 s", { color: "ok", size: 12 }); G.label(X(ts) + 8, 94, "waste ≈ 6.5%", { color: "ok", size: 12 });
          G.label(80, 90, "saving cost C/(τ+C)", { color: "q", size: 11 }); G.label(400, 266, "recompute (τ+C)/2M", { color: "v", size: 11 });
          G.label(24, 366, "C = 30 s · M = 4 h (example values) · restart cost R = 0", { size: 12 });
          G.from([p1, p2, p3], { opacity: 0, duration: 0.6 });
          G.caption("two costs cross near the best interval");
        }, legend: [["q", "saving"], ["v", "lost work"], ["k", "total"], ["ok", "τ*"]] },

      { rail: "scale + async", title: "Bigger jobs save more often; async saves make it cheaper",
        body: `<p>Combine steps 1 and 9. A bigger job has a smaller M, so τ* = √(2CM) shrinks. With C = 30 s and the example node MTBF of 720 h:</p>
<div class="eq">N = 1:    M = 720 h   τ* = 208 min   waste 0.5%
N = 2:    M = 360 h   τ* = 147 min   waste 0.7%
N = 16:   M =  45 h   τ* =  52 min   waste 1.9%
N = 128:  M = 5.6 h   τ* =  18 min   waste 5.4%
(e.g. N = 1: √(2 × 30 × 2.59e6 s) = 12,471 s)</div>
<p>Note that waste still climbs with N: more machines, more interruptions, even at the best interval. One global interval for all job sizes is therefore a mistake.</p>
<p>The other lever is C. <b>Asynchronous checkpointing</b> (<code>async_save</code> in PyTorch DCP; check your version) copies the state from GPU to CPU memory, which is fast, and writes it to storage in a background thread while training continues. Training blocks only for the copy. If that cuts the blocking C from 30 s to 3 s at N = 128:</p>
<div class="eq">τ* = √(2 × 3 × 20,250) ≈ 349 s ≈ 5.8 min
waste ≈ 1.7%   (was 5.4%)</div>
<p>Cheaper saves mean you can afford to save more often, and lose less when you fail. The price is extra host memory for the copy and care about commit semantics: the marker may only be written when the background write has finished and all ranks agree (exercise 5). If the job dies while a save is in flight, resume must use the <i>previous</i> committed checkpoint.</p>`,
        check: { q: "Async checkpointing cuts the blocking cost C from 30 s to 3 s. What happens to the optimal interval τ*?",
          options: ["It grows, because saving is cheaper so you can wait longer", "It shrinks by about 3×, so you save more often", "It stays the same: τ* does not depend on C", "It becomes zero"], answer: 1,
          why: "τ* = √(2CM), so a 10× smaller C gives a √10 ≈ 3.2× smaller interval. When each save costs little, saving often is affordable, and the lost work per failure shrinks." },
        scene(G) {
          const C = 30, rows = [1, 2, 16, 128].map((n) => ({ lab: `N = ${n}`, t: young(C, (720 * 3600) / n), c: "k" }));
          rows.push({ lab: "N = 128, async C = 3 s", t: young(3, (720 * 3600) / 128), c: "ok" });
          G.text(24, 34, "best checkpoint interval τ* (minutes), C = 30 s", { size: 13 });
          const x0 = 200, sc = 330 / 210, out = [];
          rows.forEach((r, i) => {
            const y = 60 + i * 56;
            G.text(x0 - 10, y + 24, r.lab, { anchor: "end", size: 12 });
            out.push(G.rect(x0, y, (r.t / 60) * sc, 36, { fill: r.c, rx: 4 }));
            G.text(x0 + (r.t / 60) * sc + 8, y + 24, (r.t / 60).toFixed(r.t < 600 ? 1 : 0) + " min", { size: 13 });
          });
          G.label(24, 354, "example node MTBF 720 h (placeholder); job M = 720 h / N", { size: 12 });
          G.label(24, 376, "async: C = 3 s blocking → τ* about 3.2× shorter, waste 1.7% instead of 5.4%", { size: 12, color: "ink" });
          G.from(out, { attr: { width: 0 }, stagger: 0.1, duration: 0.5 });
          G.caption("the bigger the job, the more often it must save");
        }, legend: [["k", "τ*, C = 30 s"], ["ok", "async, C = 3 s"]] },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A job's MTBF is the node MTBF divided by N, so big jobs fail often and must tolerate it.</li>
<li>A checkpoint is the weights, optimizer state, step and LR position, and data position: everything the next step depends on.</li>
<li>DCP writes one shard per rank in parallel plus a metadata file, which also lets a load reshard to a different world size.</li>
<li>A <code>COMMITTED</code> marker written by rank 0 after a barrier makes "visible" mean "complete"; resume uses only committed checkpoints.</li>
<li>Recovery is restarting the job; the data stream must be keyed by (step, sequence), not by rank.</li>
<li>RPO is lost work, RTO is lost time; Young's τ* = √(2CM) balances saving against lost work, and async saves lower C.</li></ul>`,
    sim: {
      title: "How often should this job checkpoint?",
      intro: "Set the cost of one save (C), the MTBF of a single node, the number of nodes, the restart cost and a checkpoint interval. The chart plots the first-order waste model from faults.py against the interval: the dot is your interval, the green line is Young's optimum. The model has no Monte-Carlo noise; exercise 3 compares it with a simulation.",
      height: 300,
      controls: [
        { id: "C", label: "blocking cost of one save C", min: 1, max: 300, step: 1, value: 30, format: (v) => v + " s" },
        { id: "mtbf", label: "MTBF of one node, hours (example value, not measured)", min: 24, max: 2000, step: 12, value: 720, format: (v) => v + " h" },
        { id: "n", label: "nodes in the job", min: 1, max: 256, step: 1, value: 16 },
        { id: "R", label: "restart cost R (reschedule + load + warm-up)", min: 0, max: 1800, step: 10, value: 120, format: (v) => v + " s" },
        { id: "tau", label: "checkpoint interval τ", min: 1, max: 720, step: 1, value: 20, format: (v) => v + " min" },
      ],
      draw(G, v) {
        const C = v.C, M = (v.mtbf * 3600) / v.n, R = v.R, tau = v.tau * 60, ts = young(C, M);
        const lo = 1, hi = 4.6, X = (t) => 70 + (550 * (Math.log10(t) - lo)) / (hi - lo);
        let wmin = Infinity;
        for (let e = lo; e <= hi; e += 0.02) wmin = Math.min(wmin, waste(10 ** e, C, M, R));
        const ymax = Math.min(1, nice(Math.max(wmin * 4, 0.02))), Y = (w) => 250 - (210 * Math.min(w, ymax)) / ymax;
        G.axes(70, 40, 550, 210, {});
        [0.25, 0.5, 0.75, 1].forEach((f) => { G.line(70, Y(ymax * f), 620, Y(ymax * f), { color: "line", w: 0.5 }); G.label(64, Y(ymax * f) + 4, (ymax * f * 100).toFixed(ymax < 0.1 ? 1 : 0) + "%", { anchor: "end", size: 10 }); });
        [[10, "10 s"], [60, "1 min"], [600, "10 min"], [3600, "1 h"], [36000, "10 h"]].forEach(([t, l]) => G.label(X(t), 268, l, { anchor: "middle", size: 10 }));
        G.label(620, 288, "checkpoint interval τ (log) →", { anchor: "end", size: 11 }); G.label(70, 30, "wasted fraction of time", { size: 11 });
        let d = ""; for (let e = lo; e <= hi + 1e-9; e += 0.02) { const t = 10 ** e; d += (d ? " L " : "M ") + X(t).toFixed(1) + " " + Y(waste(t, C, M, R)).toFixed(1); }
        G.path(d, { color: "k", w: 3 });
        if (ts > 10 ** lo && ts < 10 ** hi) { G.line(X(ts), 40, X(ts), 250, { color: "ok", dash: "4 4" }); G.label(X(ts) + 4, 54, "τ*", { color: "ok", size: 11 }); G.circle(X(ts), Y(waste(ts, C, M, R)), 4, { fill: "ok" }); }
        if (tau >= 10 ** lo && tau <= 10 ** hi) G.circle(X(tau), Y(waste(tau, C, M, R)), 6, { fill: "hot" });
        const wt = waste(tau, C, M, R), ws = waste(ts, C, M, R);
        return [
          { title: "The job", rows: [["job MTBF  M = node / N", dur(M)], ["restart cost R", dur(R)]] },
          { title: "Best interval", rows: [["Young  √(2CM)", dur(ts)], ["Daly (refined)", dur(daly(C, M))], ["waste at Young's τ*", (ws * 100).toFixed(2) + "%"]] },
          { title: "Your interval", gauge: [[Math.min(1, wt)], [0]].slice(0, 1).map(([f]) => [f, wt > 1.5 * ws ? "hot" : "ok"]), gaugeText: `waste ${(wt * 100).toFixed(2)}%`,
            rows: [["τ", dur(tau)], ["average RPO ≈ τ/2", dur(tau / 2)], ["vs the best", ((wt / ws - 1) * 100).toFixed(0) + "% more waste"]],
            chip: [wt <= 1.15 * ws, wt <= 1.15 * ws ? "near the optimum" : "far from the optimum"],
            html: `<p class="note">First-order model C/(τ+C) + (τ+C)/(2M) + R/M from <code>faults.py</code>. It assumes failures are random with mean M, saves block training for C, and a failure loses half an interval on average. Real failures are not so tidy.</p>` },
        ];
      },
    },
    practice: {
      items: [
        { title: "DCP round trip and commit markers", tier: "T0 · medium", goal: "Save TinyLM and its AdamW state with DCP and load them back bitwise equal; then run test_checkpoint_logic.py (same folder) for the COMMITTED marker, latest-checkpoint discovery and pruning. Explain why the barrier comes first.",
          cmd: `uv run --extra torch pytest ${MOD}/exercises/test_dcp.py` },
        { title: "Resume with a different world size (2 → 4)", tier: "T0 · slow", goal: "Train on 2 ranks, resume on 4, and match the uninterrupted run's final loss within 0.2%. Then seed the data by rank and watch it fail.",
          cmd: `uv run --extra torch pytest -m slow ${MOD}/exercises/test_reshard.py` },
        { title: "Optimal checkpoint interval", tier: "T0 · medium", goal: "Check Young's formula against the waste model and the Monte-Carlo simulator at 0.25× to 4× τ*; then plug in measured C and an estimated M for 16 and 1,024 GPUs.",
          cmd: `uv run --extra torch pytest ${MOD}/exercises/test_interval.py` },
        { title: "Project #9: checkpointed training on EKS with an RTO/RPO table", tier: "T3 · hard", goal: "Run #9 as a RayJob on EKS (4 GPUs), inject a process crash, a pod loss and a node loss, and fill the RTO/RPO table. Local rehearsal on CPU first.",
          cmd: `bash ${MOD}/examples/04_fault_inject.sh local` },
        { title: "Async checkpointing and its overhead", tier: "T0 + T3 · hard", goal: "Add an --async-ckpt option to train.py with one save in flight, commit only after the write finishes and all ranks agree, measure blocking time per checkpoint, and recompute τ*.",
          cmd: `torchrun --nproc-per-node 2 platform/training/train.py --steps 60 --ckpt-every 20 --ckpt /tmp/ckpt   # add --async-ckpt once you implement it` },
      ],
      labs: [
        { label: "DDP on CPU with gloo: replicas stay identical because gradients are all-reduced (<code>torchrun --nproc-per-node 2 …</code>)", path: `${MOD}/examples/01_ddp_cpu.py` },
        { label: "A DCP round trip of model and optimizer state, and the files it writes", path: `${MOD}/examples/02_dcp_save_load.py` },
        { label: "Primary T3 path: the job as a RayJob on EKS", path: `${MOD}/examples/03_ray/README.md` },
        { label: "Alternative T3 path: Slurm on ParallelCluster", path: `${MOD}/examples/03_slurm/README.md` },
        { label: "Kill a rank or a pod mid-run, resume, and print the RPO/RTO table (<code>local</code> or <code>ray</code>)", path: `${MOD}/examples/04_fault_inject.sh` },
        { label: "#9 itself: the FSDP2 training script with automatic resume and fault injection", path: "platform/training/train.py" },
        { label: "Commit marker, latest-checkpoint discovery and pruning", path: "platform/training/checkpoint.py" },
        { label: "Young/Daly interval, waste model and failure simulator", path: "platform/training/faults.py" },
        { label: "RPO and RTO from the logs of a failed and a resumed run", path: "platform/training/rto.py" },
        { label: "AWS guide: RayJob on EKS with cost, auto-stop and teardown", path: `${MOD}/aws.md` },
      ],
    },
  });
})();
