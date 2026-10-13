/* P6.3 — The paged KV block manager: slots and block tables, ref counts, allocation, copy-on-write, chained block hashes,
   cached-free LRU. Mirrors platform/engine/v1/s2s_engine/block_manager.py. All worked numbers were run on that class. */
(function () {
  const F = S2S.fmt;
  const cdiv = (a, b) => Math.ceil(a / b);
  const cell = (G, x, y, w, h, t, color, o = {}) => G.box(x, y, w, h, t, Object.assign({ fill: color, size: 11, rx: 3 }, o));
  const KV_PER_TOKEN = 131072; // Llama-3-8B bf16: 32 layers x 2 (K,V) x 8 KV heads x 128 x 2 bytes (P1.2)

  S2S.lesson({
    id: "p6-3", n: "P6.3", title: "The paged KV block manager",
    subtitle: "Engine internals · first principles · T0 pure Python, T2 for the paged kernel",
    kicker: "Lesson · ≈ 45 min",
    headline: "A block manager is an allocator with reference counts",
    intro: `<p>P2.3 used PagedAttention as a user: KV memory is cut into blocks and each sequence has a block table. This lesson builds the part that makes it work: the <b>block manager</b>, a plain Python object that hands out block ids and keeps the books. It holds no tensors, only integers. You will see how a token position becomes a memory slot, why the books need <i>reference counts</i>, how a forked sequence copies a block only when it must, and how a chain of hashes lets a new request reuse KV another request computed. The reference is <code>platform/engine/v1/s2s_engine/block_manager.py</code>, which you re-implement in the exercises.</p>`,
    facts: ["9 steps", "5 checkpoints", "1 simulator", "6 exercises"],
    legend: [["k", "sequence A"], ["v", "sequence B / child"], ["ok", "shared / cached"], ["hot", "copy / evict / waste"], ["w", "free"]],
    prev: "p6-2", next: "p6-4",
    steps: [
      { rail: "the pool", title: "One big array, handed out in blocks",
        body: `<p>The engine reserves KV memory once, at start-up (P2.1): one array per layer with <code>num_blocks × block_size</code> <b>slots</b>. A slot holds the K and V of one token. Nothing is allocated or freed on the GPU after that. "Allocating" only means: <i>this block id now belongs to that sequence</i>.</p>
<p>The block manager owns that bookkeeping. A sequence's <b>block table</b> is a Python list of block ids. Entry <i>i</i> is the physical block that holds the sequence's tokens <code>i·bs … i·bs + bs − 1</code> (<code>bs</code> = <code>block_size</code>). The list grows as the sequence grows, and the blocks it names can be anywhere in the pool.</p>
<div class="eq">num_blocks = 8, block_size = 4 → 32 slots
A: 9 tokens → ceil(9/4) = 3 blocks
   table [5, 0, 3]
B: 6 tokens → ceil(6/4) = 2 blocks
   table [1, 6]</div>
<p>Why not give each sequence one contiguous range? You do not know a sequence's final length, and ranges of different sizes leave gaps that nothing fits into. With equal-sized blocks, any free block fits any sequence, so a gap can never be too small. This is a page table for GPU memory, the same trick an operating system uses for RAM (P0.2).</p>`,
        scene(G) {
          G.text(24, 34, "the pool: 8 blocks × 4 slots (32 slots)", { size: 13 });
          for (let j = 0; j < 8; j++) {
            const x = 24 + j * 75;
            G.label(x + 30, 56, "block " + j, { size: 10, anchor: "middle" });
            for (let s = 0; s < 4; s++) G.rect(x + s * 17, 64, 15, 34, { stroke: "line", rx: 2 });
          }
          const fills = [];
          [[[5, 0, 3], 9, "k"], [[1, 6], 6, "v"]].forEach(([tbl, len, col]) => {
            for (let p = 0; p < len; p++) { const b = tbl[Math.floor(p / 4)], s = p % 4; fills.push(G.rect(24 + b * 75 + s * 17, 64, 15, 34, { fill: col, rx: 2 })); }
          });
          G.label(24, 118, "outline = empty slot · block 3 holds 1 token of A, block 6 holds 2 of B", { size: 11 });
          G.text(24, 168, "block tables: logical block → physical block", { size: 13 });
          [["A", [5, 0, 3], "k", 190, "9 tokens → 3 blocks, last one 1/4 full"], ["B", [1, 6], "v", 270, "6 tokens → 2 blocks, last one 2/4 full"]].forEach(([n, tbl, col, y, note]) => {
            G.text(24, y + 22, n, { size: 15, weight: 700, color: col });
            tbl.forEach((b, i) => { G.box(60 + i * 64, y, 56, 34, "L" + i + "→" + b, { stroke: col, color: col, size: 12 }); });
            G.text(60 + 3 * 64 + 20, y + 22, "table = [" + tbl.join(", ") + "]", { size: 14, color: col, weight: 700 });
            G.label(60, y + 54, note, { size: 11 });
          });
          G.text(24, 380, "no tensors here: the manager moves integers", { size: 14, color: "ink" });
          G.from(fills, { opacity: 0, stagger: 0.03, duration: 0.2 });
          G.caption("a sequence is a list of block ids; the pool never moves");
        } },

      { rail: "slot mapping", title: "From a token position to a memory slot",
        body: `<p>Attention needs two things from the table. When the model computes a token's K and V it must <b>write</b> them somewhere, and when it attends it must <b>read</b> all earlier K and V. Both use one formula, <code>BlockManager.slot()</code>:</p>
<div class="eq">slot(pos) = block_table[pos // bs] * bs + pos % bs</div>
<p><code>pos // bs</code> picks the logical block, <code>pos % bs</code> is the offset inside it, and the table translates logical to physical. Worked example, <code>bs = 16</code>, table <code>[7, 2, 9, 4, 0, 5, 1]</code>, position 37:</p>
<div class="eq">37 // 16 = 2     → block_table[2] = 9
37 % 16  = 5     → offset 5
slot = 9 * 16 + 5 = 149</div>
<p>The runner keeps the pool as a flat array indexed by slot. <code>NumpyPagedRunner</code> (<code>model_runner.py</code>) is the whole idea in a few lines: the v0 forward pass with <code>self.k[l, pos] = k</code> replaced by <code>self.k[l, slot] = k</code>, and the contiguous read replaced by a gather over <code>[slot(p) for p in range(pos + 1)]</code>. <code>test_numpy_engine.py</code> checks that this paged engine produces the same tokens as the contiguous v0 model, even with chunking and preemption.</p>`,
        check: { q: "Block size 8, block table [3, 0, 6]. Which slot holds position 19?",
          options: ["6 * 8 + 3 = 51", "19 + 3 = 22", "0 * 8 + 3 = 3", "3 * 8 + 19 = 43"], answer: 0,
          why: "19 // 8 = 2, so logical block 2, and block_table[2] = 6. The offset is 19 % 8 = 3. The slot is 6 * 8 + 3 = 51. Using the block index directly as the table entry (answer 3) or forgetting the table (answers 2 and 4) are the usual slips." },
        scene(G) {
          const tbl = [7, 2, 9, 4, 0, 5, 1];
          G.text(24, 32, "block size 16 · position 37", { size: 13 });
          G.label(24, 52, "logical blocks of the sequence (token positions)", { size: 11 });
          const lg = tbl.map((b, i) => G.box(24 + i * 84, 58, 78, 34, `${i * 16}–${i * 16 + 15}`, { fill: i === 2 ? "k" : null, stroke: "k", color: i === 2 ? undefined : "k", size: 11 }));
          G.label(24, 118, "block table", { size: 11 });
          tbl.forEach((b, i) => G.box(24 + i * 84, 124, 78, 30, String(b), { stroke: i === 2 ? "v" : "line", color: i === 2 ? "v" : "muted", size: 14 }));
          G.arrow(24 + 2 * 84 + 39, 94, 24 + 2 * 84 + 39, 122, { color: "k" });
          G.label(24, 190, "physical blocks (slot = block * 16 + offset)", { size: 11 });
          const ph = [];
          for (let j = 0; j < 10; j++) ph.push(G.box(24 + j * 61, 196, 56, 34, String(j), { fill: j === 9 ? "v" : null, stroke: tbl.indexOf(j) >= 0 ? "k" : "line", color: j === 9 ? undefined : tbl.indexOf(j) >= 0 ? "k" : "muted", size: 12 }));
          G.arrow(24 + 2 * 84 + 39, 156, 24 + 9 * 61 + 28, 194, { color: "v", w: 2 });
          G.text(24, 276, "37 // 16 = 2      → table[2] = 9", { size: 14, color: "ink" });
          G.text(24, 302, "37 % 16  = 5      → offset 5", { size: 14, color: "ink" });
          G.text(24, 334, "slot = 9 * 16 + 5 = 149", { size: 16, color: "v", weight: 700 });
          G.label(24, 366, "same lookup for writing a new token's K/V and for reading the old ones", { size: 11 });
          G.from(ph, { opacity: 0, stagger: 0.03, duration: 0.2 });
          G.caption("divide, look up, multiply, add");
        } },

      { rail: "waste", title: "What blocks cost: at most block_size − 1 slots each",
        body: `<p>A sequence of <i>n</i> tokens holds <code>ceil(n / bs)</code> blocks, so only its last block can be partly empty. The waste is <code>ceil(n / bs) · bs − n</code>, between 0 and <code>bs − 1</code> slots, and about <code>(bs − 1) / 2</code> on average. This is <b>internal fragmentation</b>. There is no <b>external</b> fragmentation: every free block is the same size, so a request never fails for lack of a big enough gap.</p>
<div class="eq">bs = 16, a 100-token sequence
  ceil(100/16) = 7 blocks = 112 slots
  waste = 112 - 100 = 12 slots
Llama-3-8B bf16, 128 KiB per token (P1.2):
  12 * 128 KiB = 1.5 MiB</div>
<p>Small blocks waste less, so why not <code>bs = 1</code>? Two costs rise as blocks shrink: the block table gets longer (one entry per token at <code>bs = 1</code>, so more bookkeeping and more table lookups), and the GPU kernel reads K/V in smaller, less regular pieces (P5.8). Prefix reuse, in step 7, also works in whole blocks, so smaller blocks help there too. <code>bench/utilization.py</code> replays a fixed trace of 60 requests that share a 150-token system prompt through the engine with the fake model and a pool of 8,192 slots:</p>
<table><tr><th>block size</th><th>slot utilization</th><th>prefix-hit rate</th></tr>
<tr><td>1</td><td>1.000</td><td>0.668</td></tr><tr><td>8</td><td>0.986</td><td>0.640</td></tr><tr><td>16</td><td>0.969</td><td>0.640</td></tr><tr><td>32</td><td>0.939</td><td>0.569</td></tr></table>
<p>Utilization is used token slots divided by allocated slots, averaged over steps. These numbers came from running the script (seed 0); they describe that synthetic trace, not your workload.</p>`,
        scene(G) {
          const rows = [[1, 1.0, 0.668], [8, 0.986, 0.64], [16, 0.969, 0.64], [32, 0.939, 0.569]];
          G.text(24, 32, "bench/utilization.py · 60 requests, 150-token shared prompt", { size: 13 });
          G.text(24, 62, "slot utilization", { size: 12, color: "k" });
          G.text(24, 222, "prefix-hit rate", { size: 12, color: "ok" });
          const a = [], b = [];
          rows.forEach(([bs, u, h], i) => {
            const y = 74 + i * 34, y2 = 234 + i * 34;
            G.label(24, y + 17, "bs " + bs, { size: 11 }); G.label(24, y2 + 17, "bs " + bs, { size: 11 });
            a.push(G.rect(70, y, 480 * u, 24, { fill: "k", rx: 3 })); G.text(70 + 480 * u + 6, y + 17, u.toFixed(3), { size: 12 });
            b.push(G.rect(70, y2, 480 * h, 24, { fill: "ok", rx: 3 })); G.text(70 + 480 * h + 6, y2 + 17, h.toFixed(3), { size: 12 });
          });
          G.label(24, 396, "bigger blocks: more waste and fewer prefix hits; the cost of small ones is GPU-side", { size: 11 });
          G.from([...a, ...b], { attr: { width: 0 }, stagger: 0.08, duration: 0.4 });
          G.caption("both numbers fall as the block grows");
        } },

      { rail: "ref counts", title: "Who owns a block? Reference counts and three states",
        body: `<p>Blocks can be <b>shared</b>: two block tables may name the same physical block (prefix reuse and forks, both coming). Then "free it when the sequence finishes" is wrong, because the other sequence still needs it. The manager keeps a <b>reference count</b> per block: <code>ref[b]</code>, the number of block tables that contain <code>b</code>. A block goes back to the pool only when its count reaches 0.</p>
<p>There is one more wrinkle. A finished request's full blocks hold KV that a future request with the same prefix could reuse, so freeing them straight away throws value away. So a block with <code>ref == 0</code> is in one of two places:</p>
<table><tr><th>state</th><th>ref</th><th>kept in</th><th>meaning</th></tr>
<tr><td>in use</td><td>≥ 1</td><td>some block tables</td><td>ref = number of tables naming it</td></tr>
<tr><td>cached-free</td><td>0</td><td><code>cached_free</code> (LRU)</td><td>holds a full, hashed block; reusable by a prefix match until its slot is needed</td></tr>
<tr><td>free</td><td>0</td><td><code>free</code> (deque)</td><td>no reusable content</td></tr></table>
<p><code>num_free</code> counts <b>both</b> kinds of ref-0 block, because either can be handed out. <code>_release(b)</code> decrements the count and, at 0, sends the block to <code>cached_free</code> if it has a hash, otherwise to <code>free</code>. A decrement below 0 is a double free and trips an assert.</p>`,
        check: { q: "A finished request's hashed blocks have ref 0. Why do they go to cached_free instead of the plain free list?",
          options: ["The free list is full", "So a later request with the same prefix can still match them; they are handed out for new data only when nothing else is free", "Cached-free blocks cannot be overwritten"], answer: 1,
          why: "A block in cached_free still holds valid KV and keeps its hash entry, so match_prefix can find it. It also counts as free capacity: _take() evicts the least recently freed cached block when the plain free list is empty. Putting it on free would discard the hash and make prefix hits stop working after the first request finishes." },
        scene(G) {
          const st = (x, y, w, name, sub, col) => { G.box(x, y, w, 62, "", { stroke: col, rx: 12 }); G.text(x + w / 2, y + 28, name, { anchor: "middle", size: 15, weight: 700, color: col }); G.label(x + w / 2, y + 48, sub, { anchor: "middle", size: 11 }); };
          st(24, 44, 180, "in use", "ref ≥ 1", "k");
          st(230, 44, 180, "cached-free", "ref 0 · hashed · LRU", "ok");
          st(436, 44, 180, "free", "ref 0 · no content", "w");
          G.text(24, 150, "moves between the states", { size: 13, weight: 700 });
          [["k", "ok", "in use → cached-free", "last table lets go (ref 0) and the block has a hash"],
           ["k", "w", "in use → free", "ref 0 and no hash (a partial block)"],
           ["ok", "k", "cached-free → in use", "a prefix match takes it (ref 0 → 1)"],
           ["w", "k", "free → in use", "_take() hands it out first"],
           ["ok", "k", "cached-free → in use", "_take() evicts the oldest, only if free is empty; hash deleted"]].forEach(([c1, c2, t, d], i) => {
            const y = 176 + i * 40;
            G.rect(24, y, 10, 26, { fill: c1, rx: 2 }); G.arrow(38, y + 13, 58, y + 13, { color: "ink" }); G.rect(62, y, 10, 26, { fill: c2, rx: 2 });
            G.text(84, y + 12, t, { size: 12, weight: 700, color: "ink" }); G.label(84, y + 26, d, { size: 11 });
          });
          G.text(24, 392, "num_free = len(free) + len(cached_free)", { size: 13, color: "ink" });
          G.caption("ref 0 does not mean empty: it means nobody is using it");
        } },

      { rail: "allocate", title: "Allocation: count first, then take",
        body: `<p>The scheduler (P6.2) must know before it builds a chunk whether the blocks exist. So there are two calls. <code>can_allocate(seq, n)</code> answers; <code>allocate(seq, n)</code> does it. Both reason about <code>seq.num_computed + n</code>, the number of slots the table must cover after the chunk:</p>
<div class="eq">need = blocks_for(num_computed + n)
       - len(block_table)
fits = max(0, need) + cow &lt;= num_free
  (cow = 1 if a copy-on-write is pending)
blocks_for(t) = ceil(t / bs)</div>
<div class="eq">bs = 4. New sequence, chunk of 10:
  blocks_for(10) = 3, table empty
  -> need 3 (12 slots, 2 spare)
Decode with num_computed 10 (n = 1):
  blocks_for(11) = 3, table has 3
  -> need 0 (slot 10 is free)
Decode with num_computed 12:
  blocks_for(13) = 4 -> need 1</div>
<p>So a decode step needs a new block only once every <code>bs</code> tokens. <code>allocate</code> then calls <code>_take()</code> for each missing block. <code>_take()</code> prefers the plain <code>free</code> list. Only when it is empty does it evict the least recently freed <code>cached_free</code> block, removing that block's hash from both lookup tables, and counting it in <code>stats["evicted_cached"]</code>. If both are empty it raises <code>NoFreeBlocks</code>.</p>
<p>The check-then-act split is deliberate. The scheduler calls <code>can_allocate</code> while deciding who to preempt (P6.2), and a failed <code>allocate</code> must not leave a half-updated table, so it checks first and raises before touching anything.</p>`,
        scene(G) {
          G.text(24, 32, "block size 4 · a sequence's life", { size: 13 });
          const stages = [["prefill 10", 10, 3, "need 3"], ["decode → 11", 11, 3, "need 0"], ["decode → 12", 12, 3, "need 0"], ["decode → 13", 13, 4, "need 1"]];
          const all = [];
          stages.forEach(([t, len, nb, need], r) => {
            const y = 56 + r * 76;
            G.label(24, y + 12, t, { size: 12, color: "ink" });
            G.label(24, y + 30, need, { size: 12, color: need === "need 0" ? "muted" : "hot" });
            for (let b = 0; b < nb; b++) {
              G.rect(130 + b * 116, y, 108, 44, { stroke: "line", rx: 5 });
              for (let s = 0; s < 4; s++) {
                const p = b * 4 + s;
                if (p < len) all.push(G.rect(134 + b * 116 + s * 26, y + 4, 24, 36, { fill: p === len - 1 && r > 0 ? "v" : "k", rx: 3 }));
              }
            }
          });
          G.label(24, 370, "cyan = tokens with K/V · amber = the new token · outline = one block of 4 slots", { size: 11 });
          G.label(24, 392, "a new block is needed only when the token lands in a slot past the end of the table", { size: 11, color: "ink" });
          G.from(all, { opacity: 0, stagger: 0.02, duration: 0.15 });
          G.caption("blocks_for(num_computed + n) vs len(block_table)");
        } },

      { rail: "fork and CoW", title: "Fork and copy-on-write: share until someone writes",
        body: `<p>Some requests ask for several outputs from one prompt (parallel sampling, <code>n = 4</code>) or explore several continuations (beam search). All branches start with identical KV. Copying it per branch would waste memory and time. <code>fork(parent, child)</code> instead gives the child a copy of the parent's <i>table</i> and raises every block's ref count by one. No KV moves.</p>
<p>Full blocks never change again, so sharing them is free forever. The <b>last, partial</b> block is different: both sequences will write their next token into it, and they will write different tokens. The first one to append gets a private copy, done in <code>allocate</code>:</p>
<div class="eq">_needs_cow(seq) = last block is partial
                  AND ref[last] &gt; 1
allocate: dst = _take()
          copy_ops.append((src, dst))
          _release(src)      # ref[src] -= 1
          block_table[-1] = dst</div>
<p>I ran this on the reference (bs = 4, 8 blocks). The parent has 6 tokens: table <code>[0, 1]</code>, block 1 holds 2. After <code>fork</code> both tables are <code>[0, 1]</code> and <code>ref = [2, 2]</code>. When the child appends, <code>_needs_cow</code> is true (6 % 4 = 2 ≠ 0 and ref[1] = 2), so the child takes block 2, queues <code>(1, 2)</code>, and ends with table <code>[0, 2]</code>, ref = <code>[2, 1, 1]</code>. Block 0, full, stays shared. The parent then appends: ref[1] = 1, so no copy.</p>
<p>The queued <code>(src, dst)</code> pairs are only bookkeeping. The runner executes them on the GPU through <code>copy_blocks</code> at the start of <code>execute()</code>, <b>before</b> the forward pass. Copy after, and the child's copy would already contain its parent's new token.</p>`,
        check: { q: "A parent with a partial last block is forked into 3 children (4 sequences share it, ref = 4), and every sequence then appends one token. How many block copies happen?",
          options: ["4", "3", "1", "0"], answer: 1,
          why: "Each sequence in turn finds the block shared and takes a private copy, which drops the ref count: 4 → 3 → 2 → 1. The last sequence sees ref == 1, so it owns the block and writes in place. Three copies; I confirmed cow_copies == 3 on the reference. The copies are for the first decode step only: afterwards each sequence has its own partial block." },
        scene(G) {
          const trow = (x, y, ids, col, shared) => ids.map((b, i) => G.box(x + i * 60, y, 52, 32, "blk " + b, { fill: col, size: 11, rx: 4 }));
          G.text(24, 30, "bs 4 · parent has 6 tokens", { size: 13 });
          // before
          G.label(24, 54, "after fork: tables identical, ref = 2, no KV moved", { size: 11, color: "ink" });
          G.label(24, 78, "parent", { size: 11 }); trow(80, 64, [0, 1], "k");
          G.label(24, 120, "child", { size: 11 }); trow(80, 106, [0, 1], "v");
          const sh = G.rect(76, 60, 62, 82, { stroke: "ok", rx: 6, dash: "4 3", sw: 2 });
          G.label(220, 84, "block 0 full · block 1 has 2 of 4", { size: 11 });
          G.label(220, 104, "ref[0] = 2   ref[1] = 2", { size: 11, color: "ok" });
          G.line(24, 172, 616, 172, { color: "line", dash: "3 4" });
          G.label(24, 194, "child appends a token: partial AND ref > 1 → copy", { size: 12, color: "ink" });
          G.label(24, 220, "parent", { size: 11 }); trow(80, 206, [0, 1], "k");
          G.label(24, 262, "child", { size: 11 }); const cc = [G.box(80, 248, 52, 32, "blk 0", { fill: "v", size: 11, rx: 4 }), G.box(140, 248, 52, 32, "blk 2", { fill: "v", size: 11, rx: 4 })];
          const arr = G.path("M 166 206 C 230 206 230 264 196 264", { color: "hot", w: 2, arrow: true, dash: "5 3" });
          G.label(240, 232, "copy_ops: (1 → 2)", { size: 12, color: "hot" });
          G.label(240, 252, "runner copies the KV BEFORE the forward pass", { size: 11 });
          G.label(24, 316, "ref[0] = 2   ref[1] = 1   ref[2] = 1", { size: 12, color: "ink" });
          G.label(24, 338, "parent appends later: ref[1] = 1 → writes in place, no copy", { size: 12 });
          G.text(24, 380, "n = 4 sharing a partial block → 3 copies (the last one owns it)", { size: 13, color: "hot" });
          G.pulse(arr, { repeat: 4 });
          G.from(cc, { opacity: 0, duration: 0.4, delay: 0.3 });
          G.caption("sharing is a pointer; copying happens only on the first write");
        } },

      { rail: "block hashes", title: "Reusing KV: a hash that covers the whole prefix",
        body: `<p>Requests often start with the same tokens (a system prompt, a chat history). Their KV is identical because a token's K and V depend only on the tokens up to it (P1.2, the causal mask). The engine can skip prefill for those tokens if it can <b>find</b> blocks that already hold them. It finds them by name: a hash.</p>
<p>The hash must identify the block's tokens <i>and everything before them</i>, because the KV in the block depends on all of it. So each hash is <b>chained</b> on the previous one:</p>
<div class="eq">h_0 = H("",  tokens[0:bs])
h_i = H(h_{i−1}, tokens[i*bs:(i+1)*bs])</div>
<p><code>chain_hash</code> uses sha256, not Python's <code>hash()</code>, so values agree across processes. Three rules follow:</p>
<ul><li><b>Only full blocks are hashed.</b> A partial block is still being written, so its content (and hash) would change under you. <code>commit(seq)</code> registers hashes after each forward pass for the blocks that just became full, for prompt <i>and</i> generated tokens.</li>
<li><b>Matching stops at the first miss.</b> Block 3 can only match if blocks 0 to 2 matched, which the chain guarantees.</li>
<li><b>Leave at least one prompt token uncomputed.</b> The model needs logits at the last prompt position to produce the first output token. <code>match_prefix</code> only looks at <code>(len(prompt) − 1) // bs</code> blocks.</li></ul>
<div class="eq">bs = 16. A: 100 tokens, finished
  -> 6 full blocks hashed (96 tokens)
B: same first 70 tokens, then different
  usable = (100 - 1) // 16 = 6 blocks to look at
  blocks 0..3 match = 64 tokens
  block 4 has 6 shared + 10 other tokens: miss
  skipped = 64
C: exactly A's first 64 tokens
  usable = (64 - 1) // 16 = 3
  skipped = 48, not 64</div>
<p>I ran all three on the reference: 64 and 48. A radix tree (P6.4) can match to the exact token, 70 in B's case; blocks give up the partial 6 tokens in exchange for a simple hash table.</p>`,
        check: { q: "Block size 16. Prompt A (100 tokens) has finished. Prompt B has the same first 70 tokens, then diverges. How many tokens of B's prefill are skipped?",
          options: ["70", "64", "80", "96"], answer: 1,
          why: "Only full blocks match: 70 // 16 = 4 blocks = 64 tokens. Block 4 holds tokens 64 to 79, which differ from token 70 on, so its chained hash is different. A token-level radix tree would reuse all 70." },
        scene(G) {
          G.text(24, 32, "chained hashes · bs 16 · A (100 tokens) vs B (shares 70)", { size: 13 });
          const xs = [0, 1, 2, 3, 4, 5].map((i) => 24 + i * 100);
          G.label(24, 56, "A", { size: 13, color: "ink" });
          const hA = xs.map((x, i) => G.box(x + 14, 62, 86, 34, "h" + i, { fill: i < 4 ? "ok" : "k", size: 12 }));
          xs.slice(0, 5).forEach((x, i) => G.arrow(x + 100, 79, x + 114, 79, { color: "ink" }));
          G.label(24, 118, "each hash covers its own 16 tokens AND the previous hash", { size: 11 });
          G.label(24, 154, "B", { size: 13, color: "ink" });
          const hB = xs.map((x, i) => G.box(x + 14, 160, 86, 34, i < 4 ? "= h" + i : "h'" + i, { fill: i < 4 ? "ok" : null, stroke: i < 4 ? undefined : "v", color: i < 4 ? undefined : "v", size: 12 }));
          G.line(24 + 4 * 100 + 8, 150, 24 + 4 * 100 + 8, 210, { color: "hot", w: 2 });
          G.label(24 + 4 * 100 + 14, 224, "first miss: stop", { size: 11, color: "hot" });
          G.label(24, 240, "tokens 0–63 reused (4 blocks)", { size: 12, color: "ok" });
          G.label(24, 262, "block 4 (tokens 64–79) differs from token 70 on → new hash, stop", { size: 11 });
          G.text(24, 316, "usable = (len − 1) // bs   always ≥ 1 token left to compute", { size: 13, color: "ink" });
          G.text(24, 344, "skipped = 4 * 16 = 64 tokens", { size: 15, color: "ok", weight: 700 });
          G.label(24, 372, "if B were exactly A's first 64 tokens: (64 − 1) // 16 = 3 blocks → 48 skipped", { size: 11 });
          G.from([...hB], { opacity: 0, x: -10, stagger: 0.08, duration: 0.25 });
          G.caption("a hit means this block and everything before it match");
        } },

      { rail: "eviction order", title: "Evicting cached blocks: release the tail first",
        body: `<p>Cached-free blocks stay matchable until their slot is needed. Then <code>_take()</code> evicts the <b>least recently freed</b> one (<code>cached_free</code> is an <code>OrderedDict</code>; <code>popitem(last=False)</code> takes the oldest entry) and deletes its hash. Which block is oldest depends on the order <code>free_seq</code> releases them. It releases the table <b>in reverse, tail first</b>.</p>
<p>Why? Matching walks from block 0 and stops at the first miss. So a cached prefix is only as long as its first missing block. If the head were evicted first, the whole chain would be unreachable even though the later blocks are still in memory. Evicting the tail first loses only the end of the chain.</p>
<div class="eq">pool of 7 blocks, bs 16. A (100 tokens) fills
all 7, then finishes. Blocks 0..5 are hashed:
  cached_free = [5, 4, 3, 2, 1, 0]
  block 6 is partial, never hashed: free list
X arrives: 32 new tokens, needs 2 blocks
  takes block 6 (free first), then evicts
  block 5 (the oldest cached one)
A' arrives with A's prompt: blocks 0..4 match
  = 80 tokens skipped
Head-first release would have evicted block 0:
  0 tokens skipped</div>
<p>I ran the tail-first case on the reference and got 80. The same logic is why a block-level cache is an LRU of <i>blocks</i> while a radix tree (P6.4) evicts <i>leaves</i>: both protect the shared head.</p>
<p>One more detail in <code>commit</code>: if two sequences compute the same content into two different blocks at the same time, the second block is not registered, so a hash maps to one block id. Both blocks stay valid for their own sequences.</p>`,
        check: { q: "Why does free_seq release a sequence's blocks tail first?",
          options: ["The tail blocks are the largest", "Eviction takes the oldest freed block, so the head of the chain, which every prefix match needs, is evicted last", "Releasing the head first would double-free it"], answer: 1,
          why: "A prefix match walks from block 0 and stops at the first miss. If the head were evicted, the surviving tail blocks could never be matched. With tail-first release the head is freed last, so it is evicted last, and a long cached prefix shrinks from the end instead of vanishing." },
        scene(G) {
          G.text(24, 32, "A (100 tokens) finished: cached_free, oldest first", { size: 13 });
          const order = [5, 4, 3, 2, 1, 0];
          G.label(24, 52, "evicted first →", { size: 11, color: "hot" });
          const bx = order.map((b, i) => G.box(24 + i * 76, 62, 70, 42, "blk " + b, { fill: "ok", size: 13, rx: 5 }));
          G.label(24 + 6 * 76 + 4, 80, "blk 6", { size: 11 }); G.label(24 + 6 * 76 + 4, 94, "free (partial)", { size: 10 });
          G.label(24, 132, "X needs 2 blocks: takes blk 6 (free), then evicts the oldest cached block", { size: 12, color: "ink" });
          const ev = G.rect(24 - 2, 60, 74, 46, { stroke: "hot", rx: 6, sw: 2, dash: "4 3" });
          G.text(24 + 35, 124, "", { size: 1 });
          G.text(24, 188, "A' arrives with the same prompt", { size: 13 });
          const m = [0, 1, 2, 3, 4, 5].map((b, i) => G.box(24 + i * 76, 204, 70, 42, i < 5 ? "blk " + b : "blk 5 ✗", { fill: i < 5 ? "ok" : null, stroke: i < 5 ? undefined : "hot", color: i < 5 ? undefined : "hot", size: 13, rx: 5 }));
          G.label(24, 268, "match walks blocks 0..4 → 80 tokens skipped, stops at the evicted block 5", { size: 12, color: "ok" });
          G.line(24, 292, 616, 292, { color: "line", dash: "3 4" });
          G.text(24, 320, "head-first release would have evicted blk 0 first:", { size: 13, color: "hot" });
          G.label(24, 344, "block 0 gone → the first lookup misses → 0 tokens skipped,", { size: 12 });
          G.label(24, 362, "although blocks 1..5 are still sitting in memory, unreachable", { size: 12 });
          G.pulse(ev, { repeat: 5 });
          G.from(m, { opacity: 0, x: -8, stagger: 0.07, duration: 0.2 });
          G.caption("free the tail first so the head of the chain survives longest");
        } },

      { rail: "invariants + GPU", title: "Trusting the books: invariants, and the hand-off to the GPU",
        body: `<p>Bookkeeping with sharing, caching and eviction is easy to get wrong in ways that only show up as garbage text later. The manager ships its own auditor, <code>check_invariants(live)</code>:</p>
<div class="eq">1. ref[b] == block-table occurrences of b
2. each block is in exactly one of
   {in use, free, cached-free}
3. free and cached_free are disjoint
4. hash_of and block_of agree both ways</div>
<p>The exercise tests call it after <b>every</b> operation of a random sequence: 30 random pools, 200 operations each (admit, append a token, fork, free). Random sequences find the corner cases you would not write by hand, such as a fork of a fork or an eviction in the middle of a match. This is a <i>property test</i>: it does not check a particular answer, only that the invariants hold.</p>
<p>The GPU never sees Python lists. Per step the runner builds arrays from them:</p>
<ul><li><b>slot mapping</b>: for every token being computed, <code>slot(pos)</code>; the K/V write goes there (<code>write</code> slots in <code>TorchRunner</code>);</li>
<li><b>block tables</b>: for decode, a padded <code>[B, max_blocks]</code> matrix plus <code>context_lens [B]</code>; the paged attention kernel walks each row;</li>
<li><b>copy_ops</b>: <code>(src, dst)</code> pairs applied before anything else.</li></ul>
<p>Today <code>TorchRunner.decode</code> gathers each sequence's slots into a dense tensor and runs ordinary attention. The paged kernel from P5.8 (exercise 4 here) reads through the table directly and skips that copy.</p>`,
        scene(G) {
          G.box(24, 50, 270, 214, "", { stroke: "ink", rx: 12 });
          G.text(36, 76, "CPU · BlockManager", { size: 14, weight: 700 });
          ["block_table[seq] = [7, 2, 9, …]", "ref[b], free, cached_free", "hash_of, block_of", "copy_ops = [(1, 2)]"].forEach((t, i) => G.label(40, 104 + i * 26, t, { size: 12, color: "ink" }));
          G.label(40, 220, "integers only", { size: 11, color: "k" });
          const ar = [["slot mapping [T]", "ok", 90], ["block_tables [B, max_blocks]", "k", 138], ["context_lens [B]", "k", 170], ["copy_ops", "hot", 214]];
          const out = [];
          ar.forEach(([t, c, y]) => { out.push(G.arrow(296, y, 346, y, { color: c, w: 2 })); });
          G.box(350, 50, 270, 214, "", { stroke: "w", rx: 12, fill: "w", boxOpacity: 0.25 });
          G.text(362, 76, "GPU · runner", { size: 14, weight: 700 });
          ["copy blocks first (copy_blocks)", "write K/V at the slots", "attention reads via tables", "(paged kernel, P5.8)"].forEach((t, i) => G.label(366, 104 + i * 26, t, { size: 12, color: "ink" }));
          G.label(366, 220, "the pool: [layers, slots, …]", { size: 11, color: "w" });
          G.text(24, 310, "after every random operation:", { size: 13, weight: 700 });
          ["ref[b] == table occurrences of b", "each block in exactly one state", "hash_of ↔ block_of agree"].forEach((t, i) => G.label(34, 334 + i * 22, "✓ " + t, { size: 12, color: "ok" }));
          G.from(out, { opacity: 0, x: -10, stagger: 0.1, duration: 0.3 });
          G.caption("Python keeps the books; the GPU only sees arrays of slots");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>The pool is fixed; a sequence is a list of block ids. <code>slot(pos) = block_table[pos // bs] * bs + pos % bs</code>, used for both writing and reading K/V.</li>
<li>Blocks waste at most <code>bs − 1</code> slots per sequence and cause no external fragmentation. Larger blocks waste more and match fewer prefixes.</li>
<li>Reference counts make sharing safe. A ref-0 block is free, or cached-free if it holds a hashed prefix that a later request may match.</li>
<li><code>can_allocate</code> counts <code>blocks_for(num_computed + n) − len(table)</code> (plus one for a pending copy) against <code>num_free</code>; <code>_take</code> prefers free blocks and evicts the oldest cached block only when it must.</li>
<li>Fork shares tables; the first write into a shared partial block copies it, and the runner executes the copy before the forward pass.</li>
<li>Chained hashes of full blocks find reusable prefixes; keep one token uncomputed, release the tail first, and test the books with random operations and <code>check_invariants</code>.</li></ul>`,
    sim: {
      title: "Block size: waste against prefix reuse",
      intro: "Pick a block size and a request. The chart is the waste of one sequence at every length near yours: a sawtooth between 0 and block_size − 1 slots, with your length marked. The panels turn it into memory for N concurrent sequences (Llama-3-8B bf16, 128 KiB of KV per token) and tell you how much of the prompt a cached shared prefix saves.",
      height: 280,
      controls: [
        { id: "bs", label: "block size (tokens)", type: "select", value: 16, options: [[1, "1"], [4, "4"], [8, "8"], [16, "16"], [32, "32"], [64, "64"]] },
        { id: "len", label: "sequence length (tokens)", min: 1, max: 2000, step: 1, value: 300 },
        { id: "n", label: "concurrent sequences", min: 1, max: 256, step: 1, value: 64 },
        { id: "shared", label: "prefix shared with a cached request (tokens)", min: 0, max: 2000, step: 10, value: 150 },
      ],
      draw(G, v) {
        const bs = +v.bs, L = v.len, N = v.n;
        const blocks = cdiv(L, bs), waste = blocks * bs - L;
        const sh = Math.min(v.shared, L);
        const usable = Math.floor((L - 1) / bs), hitBlocks = Math.min(Math.floor(sh / bs), usable), hit = hitBlocks * bs;
        const span = Math.max(4 * bs, 16), w0 = Math.max(0, Math.floor(L / bs) * bs - 2 * bs);
        const x0 = 50, W = 570, H = 190, base = 230, bw = W / span, top = Math.max(bs - 1, 1);
        G.axes(x0, base - H, W, H);
        G.label(10, 16, "wasted slots in the last block, by sequence length (token n → ceil(n/bs)*bs − n)", { size: 11 });
        for (let i = 0; i < span; i++) {
          const n = w0 + i + 1, wv = cdiv(n, bs) * bs - n, h = (H * wv) / top;
          G.rect(x0 + i * bw, base - h, Math.max(1, bw - 0.4), Math.max(h, 0.5), { fill: n === L ? "hot" : "k", rx: 0 });
        }
        G.label(x0, base - H - 4, `your length ${L}: ${waste} wasted slots`, { size: 11, color: "hot" });
        G.label(x0, base + 18, "length " + (w0 + 1), { size: 10 }); G.label(x0 + W, base + 18, "length " + (w0 + span), { size: 10, anchor: "end" });
        G.label(x0 - 6, base - H + 4, String(bs - 1), { size: 10, anchor: "end" }); G.label(x0 - 6, base + 4, "0", { size: 10, anchor: "end" });
        G.line(x0, base - H, x0 + W, base - H, { color: "line", dash: "3 4" });
        G.label(x0 + W, base - H - 4, "max = bs − 1 = " + (bs - 1), { size: 10, anchor: "end" });
        G.label(10, 270, "red bar: your length · average waste over random lengths ≈ (bs − 1) / 2 = " + ((bs - 1) / 2).toFixed(1) + " slots", { size: 10 });
        const fleet = N * waste * KV_PER_TOKEN;
        const bad = waste / (blocks * bs);
        return [
          { title: "One sequence", rows: [["blocks held", String(blocks)], ["slots allocated", F.num(blocks * bs, 0)], ["wasted slots", String(waste)], ["utilization", (100 * L / (blocks * bs)).toFixed(1) + " %"], ["block table length", String(blocks) + " entries"]],
            gauge: [[1 - bad, "k"], [bad, "hot"]], gaugeText: `${waste} of ${blocks * bs} slots empty` },
          { title: `${N} sequences like this`, rows: [["KV in the empty slots", F.bytes(fleet)], ["KV actually used", F.bytes(N * L * KV_PER_TOKEN)]],
            html: `<p class="note">Llama-3-8B bf16: 32 layers × 2 × 8 KV heads × 128 × 2 bytes = 128 KiB per token (P1.2). Every sequence has the same length here, so this is the worst case for the waste column; real lengths spread out and average about (bs − 1) / 2 per sequence.</p>` },
          { title: "Prefix reuse", rows: [["usable blocks to match", String(usable)], ["blocks reused", String(hitBlocks)], ["tokens skipped", F.num(hit, 0)], ["share of the prompt", (100 * hit / L).toFixed(0) + " %"], ["shared tokens not reusable", String(sh - hit)]],
            chip: [hit === sh, hit === sh ? "the whole shared prefix is reused" : `${sh - hit} shared tokens re-computed (partial block or last-token rule)`] },
        ];
      },
      note: "A model of the block manager's rules, not a benchmark: waste = ceil(n/bs)*bs − n; prefix hit = min(shared // bs, (len − 1) // bs) * bs, the rules in match_prefix. It ignores LRU eviction, sharing between concurrent sequences and the kernel-side cost of small blocks.",
    },
    practice: {
      intro: "Exercises 1 to 3 and 5 are T0: pure Python, no GPU. Implement the manager in exercises/my_block_manager.py; the tests fail until you do, and S2S_SOLUTIONS=1 runs them against the reference in platform/engine/v1. Exercise 4 is T2 (one GPU, see the P5.8 paged-decode exercise).",
      items: [
        { title: "Block-table invariants under random operations", tier: "T0 · medium", goal: "Implement can_allocate, allocate (no CoW yet), free_seq and fork; check_invariants must hold after 200 random operations on 30 random pools. Then find a case where head-first release loses a hit that tail-first keeps.",
          cmd: "uv run pytest course/P6-engine-internals/P6.3-paged-kv-block-manager/exercises -q -k 'random or slot or no_free'" },
        { title: "Copy-on-write on fork", tier: "T0 · medium", goal: "Take a new block, queue (src, dst), release the source, update the table, and count the extra block in can_allocate. Say why the runner must copy before the forward pass.",
          cmd: "uv run pytest course/P6-engine-internals/P6.3-paged-kv-block-manager/exercises -q -k cow" },
        { title: "Block-hash prefix reuse, and how vLLM does it", tier: "T0 · hard", goal: "Implement commit and match_prefix with chained sha256 hashes, the one-token rule and LRU eviction of cached-free blocks. Then read vLLM's block hash at the pinned version and compare in three sentences.",
          cmd: "uv run pytest course/P6-engine-internals/P6.3-paged-kv-block-manager/exercises -q -k prefix" },
        { title: "Paged decode attention on the GPU", tier: "T2 · hard", goal: "Replace the gather-based attention inside TorchRunner.decode with your P5.8 paged kernel and compare step times. Results go in the exercise's table (TODO run-on: g6.xlarge).",
          cmd: "cat course/P6-engine-internals/P6.3-paged-kv-block-manager/exercises/04-paged-kernel.md" },
        { title: "Fragmentation and utilization under a trace", tier: "T0 · hard", goal: "Extend bench/utilization.py to replay a loadgen JSONL trace; report mean and p99 utilization, hit rate and preemptions for block sizes 1, 8, 16, 32, and argue for one.",
          cmd: "uv run python course/P6-engine-internals/P6.3-paged-kv-block-manager/bench/utilization.py" },
        { title: "The reference's tests", tier: "T0", goal: "Run the block manager and the paged NumPy engine tests (paged output must equal the contiguous v0 model token for token).",
          cmd: "uv run pytest platform/engine/v1/tests/test_block_manager.py platform/engine/v1/tests/test_numpy_engine.py -q" },
      ],
      labs: [
        { label: "The reference block manager you are re-implementing", path: "platform/engine/v1/s2s_engine/block_manager.py" },
        { label: "Runners: slot lookups, copy_blocks, the gather in decode", path: "platform/engine/v1/s2s_engine/model_runner.py" },
        { label: "Block size vs utilization and prefix-hit rate", path: "course/P6-engine-internals/P6.3-paged-kv-block-manager/bench/utilization.py" },
        { label: "Animation: paged attention (from P2.3)", path: "animations/p2-paged-attention.html" },
        { label: "Quiz", path: "course/P6-engine-internals/P6.3-paged-kv-block-manager/quiz.md" },
        { label: "Read, don't copy: vLLM kv_cache_manager.py, block_pool.py, kv_cache_utils.py at the pin (re-verify paths)", path: "vllm/v1/core/kv_cache_manager.py" },
        { label: "Paper: Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention (SOSP '23)", path: "arXiv 2309.06180" },
      ],
    },
  });
})();
