/* P6.4 — Prefix caching with a radix tree: compressed trie of token runs + KV slots, match with edge split, insert,
   ref-count locks to the root, LRU leaf eviction, radix vs block hashing, cache-aware scheduling.
   Mirrors platform/engine/v1/s2s_engine/radix_cache.py; worked numbers were run on that class. */
(function () {
  const F = S2S.fmt;
  const cell = (G, x, y, w, h, t, color, o = {}) => G.box(x, y, w, h, t, Object.assign({ fill: color, size: 11, rx: 3 }, o));
  // an edge of the radix tree: a box with the token run on top and KV slots below
  const edge = (G, x, y, toks, slots, o = {}) => {
    const w = Math.max(40, toks.length * 13 + 14), h = 46;
    const g = G.rect(x, y, w, h, { fill: o.fill || null, stroke: o.stroke || "line", rx: 6, sw: o.sw || 1.5, dash: o.dash, opacity: o.fill ? 0.9 : 1 });
    G.text(x + w / 2, y + 19, toks, { anchor: "middle", size: 13, weight: 700, color: o.fill ? "bg" : o.color || "ink" });
    if (slots) G.label(x + w / 2, y + 36, slots, { anchor: "middle", size: 9, color: o.fill ? "bg" : "muted" });
    return { g, w, h };
  };

  S2S.lesson({
    id: "p6-4", n: "P6.4", title: "Prefix caching with a radix tree",
    subtitle: "Engine internals · first principles · T0 pure Python",
    kicker: "Lesson · ≈ 40 min",
    headline: "Find the longest cached prefix in one walk down a tree",
    intro: `<p>P6.3 reused KV by hashing <i>full blocks</i>. That works, but it only matches in steps of the block size and it keeps no picture of how cached prompts relate to each other. SGLang uses a different structure, a <b>radix tree</b> over token ids, and matches to the exact token. This lesson builds it from the problem: how do you find, among thousands of cached sequences, the one that shares the longest prefix with a new prompt? You will see why a trie answers that, why it must be <i>compressed</i>, how a match splits an edge, why in-use KV is locked all the way to the root, and which leaves may be evicted. The reference is <code>platform/engine/v1/s2s_engine/radix_cache.py</code>, which you re-implement in the exercises.</p>`,
    facts: ["9 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "cached tokens"], ["ok", "matched / shared"], ["v", "new / inserted"], ["hot", "split / evict / miss"], ["q", "locked"]],
    prev: "p6-3", next: "p6-5",
    steps: [
      { rail: "the problem", title: "Which cached sequence shares the longest prefix?",
        body: `<p>A new request arrives with a prompt. Some of its first tokens may already have K/V in the cache, computed for an earlier request (a system prompt, a chat history). Every token you can reuse is a token you do not prefill. So the engine asks: <b>what is the longest prefix of this prompt that some cached sequence also starts with?</b></p>
<p>The obvious answer compares the prompt with every stored sequence and keeps the best:</p>
<div class="eq">stored: SYSTEMhello   SYSTEMbye   weather
prompt: SYSTEMhi
common prefix with each:  7    6    0
best = 7 tokens (S Y S T E M h)</div>
<p>The cost is (number of stored sequences) × (prefix length), per request, and it grows as the cache fills. It also stores "SYSTEM" three times. Both problems have the same cause: the sequences share a beginning, and the comparison throws that fact away. Merge the shared beginnings once, and a lookup becomes a single walk that stops at the first disagreement.</p>
<p>(The letters stand for token ids, so you can read the examples. <code>test_radix_cache.py</code> uses integers.)</p>`,
        scene(G) {
          const seqs = [["SYSTEMhello", 7], ["SYSTEMbye", 6], ["weather", 0]], prompt = "SYSTEMhi";
          G.text(24, 32, "naive: compare the prompt with every stored sequence", { size: 13 });
          G.label(24, 56, "prompt", { size: 11 });
          prompt.split("").forEach((ch, i) => cell(G, 24 + i * 30, 62, 26, 28, ch, "v", { size: 13 }));
          const marks = [];
          seqs.forEach(([s, m], r) => {
            const y = 128 + r * 70;
            G.label(24, y - 6, "stored " + (r + 1), { size: 11 });
            s.split("").forEach((ch, i) => marks.push(i < m ? cell(G, 24 + i * 30, y, 26, 28, ch, "ok", { size: 13 }) : G.box(24 + i * 30, y, 26, 28, ch, { stroke: i === m ? "hot" : "line", color: i === m ? "hot" : "muted", size: 13, rx: 3 })));
            G.text(24 + s.length * 30 + 12, y + 20, m + " in common", { size: 13, color: m === 7 ? "ok" : "muted", weight: m === 7 ? 700 : 400 });
          });
          G.text(24, 352, "cost per request: N sequences × prefix length", { size: 13, color: "hot" });
          G.label(24, 376, "and 'SYSTEM' is stored three times", { size: 12 });
          G.from(marks, { opacity: 0, stagger: 0.02, duration: 0.15 });
          G.caption("the sequences share a beginning; compare it once, not N times");
        } },

      { rail: "trie", title: "A trie stores each shared beginning once",
        body: `<p>A <b>trie</b> (prefix tree) is a tree where each node is one token and a path from the root spells a sequence. Sequences that start the same way share the same path until they differ:</p>
<div class="eq">SYSTEMhello  SYSTEMhi  SYSTEMbye
28 tokens as three lists
15 nodes as a trie  (S Y S T E M, h, e l l o,
                     i, b y e)</div>
<p>Matching a prompt is a walk from the root: take the child whose token equals the next prompt token; stop when there is none. The walk length is the match length, and the work is proportional to it, not to the number of stored sequences.</p>
<p>The trie answers the question, but it is wasteful for this job. It has one node per token, each with a child table, and most nodes have exactly one child (a long chain of S, Y, S, T, E, M). Every token also needs a KV slot, and a per-token node is a lot of Python object for one token. The next step fixes this.</p>`,
        scene(G) {
          const cw = 24, gap = 4, st = cw + gap;
          G.text(24, 32, "one node per token; shared beginnings stored once", { size: 13 });
          const chain = "SYSTEM".split("");
          const dy = 40, nodes = [];
          chain.forEach((ch, i) => nodes.push(cell(G, 24 + i * st, 130 + dy, cw, cw, ch, "ok", { size: 12 })));
          const x1 = 24 + 6 * st;
          nodes.push(cell(G, x1, 70 + dy, cw, cw, "h", "k", { size: 12 }));
          "ello".split("").forEach((ch, i) => nodes.push(cell(G, x1 + (i + 1) * st, 40 + dy, cw, cw, ch, "k", { size: 12 })));
          nodes.push(cell(G, x1 + st, 100 + dy, cw, cw, "i", "k", { size: 12 }));
          "bye".split("").forEach((ch, i) => nodes.push(cell(G, x1 + i * st, 168 + dy, cw, cw, ch, "k", { size: 12 })));
          const lines = [];
          lines.push(G.line(24 + 5 * st + cw, 142 + dy, x1, 82 + dy, { color: "muted", w: 1.5 }));
          lines.push(G.line(24 + 5 * st + cw, 142 + dy, x1, 180 + dy, { color: "muted", w: 1.5 }));
          lines.push(G.line(x1 + cw, 82 + dy, x1 + st, 52 + dy, { color: "muted", w: 1.5 }));
          lines.push(G.line(x1 + cw, 82 + dy, x1 + st, 112 + dy, { color: "muted", w: 1.5 }));
          G.label(24, 158, "root →", { size: 10 });
          G.label(24, 262, "green: shared by all three sequences", { size: 11, color: "ok" });
          G.text(24, 300, "28 tokens as three lists  →  15 nodes", { size: 14, color: "ink" });
          G.text(24, 328, "lookup = one walk, cost ∝ match length", { size: 14, color: "ink" });
          G.label(24, 360, "but: most nodes have one child, and a Python object per token", { size: 12 });
          G.label(24, 380, "is heavy. Next: merge each unbranching run into one edge.", { size: 12 });
          G.from(nodes, { opacity: 0, stagger: 0.04, duration: 0.2 });
          G.caption("a path from the root spells a cached sequence");
        } },

      { rail: "radix edges", title: "Compress the chains: edges carry runs of tokens and their KV slots",
        body: `<p>A <b>radix tree</b> (compressed trie) merges every chain of single-child nodes into one <b>edge</b> holding a <i>run</i> of tokens. A node now appears only where sequences <b>branch</b> or end. In <code>radix_cache.py</code> each <code>Node</code> holds:</p>
<div class="eq">key         tokens on the edge parent -> node
value       KV slot of each token (same length)
children    {first token of child's key: child}
ref         number of requests using this path
last_access logical clock, for LRU</div>
<p>Children are keyed by the <b>first token</b> of their edge. Two edges leaving one node can never start with the same token (they would be one edge), so that single token picks the edge. The <code>value</code> field is what makes this a KV cache: following the path from the root, concatenating the <code>value</code>s gives the slot of every token of the prefix, which is exactly what the runner needs to attend to it.</p>
<div class="eq">tree after the three sequences (slots 0..):
root -- SYSTEM  (6 tokens)
          |-- h  (1)  -- ello (4)
          |         \\-- i    (1)
          \\-- bye (3)       15 tokens, 5 edges</div>
<p>The same 15 tokens now cost 5 nodes instead of 15. The tree holds the <b>token ids</b> and the <b>slots</b>, not the K/V tensors; those stay in the pool.</p>`,
        scene(G) {
          G.text(24, 32, "radix tree: 5 edges instead of 15 nodes", { size: 13 });
          G.label(24, 62, "each edge: token run (top) · KV slot of each token (bottom)", { size: 11 });
          const e = [];
          e.push(edge(G, 24, 176, "SYSTEM", "slots 0–5", { fill: "ok" }));
          e.push(edge(G, 24 + 100 + 30, 126, "h", "6", { fill: "k" }));
          e.push(edge(G, 24 + 100 + 30 + 40 + 30, 88, "ello", "7–10", { fill: "k" }));
          e.push(edge(G, 24 + 100 + 30 + 40 + 30, 156, "i", "11", { fill: "k" }));
          e.push(edge(G, 24 + 100 + 30, 246, "bye", "12–14", { fill: "k" }));
          const ln = [];
          ln.push(G.line(24 + 100, 194, 24 + 130, 149, { color: "muted", w: 1.5 }));
          ln.push(G.line(24 + 100, 194, 24 + 130, 269, { color: "muted", w: 1.5 }));
          ln.push(G.line(24 + 170, 149, 24 + 200, 111, { color: "muted", w: 1.5 }));
          ln.push(G.line(24 + 170, 149, 24 + 200, 179, { color: "muted", w: 1.5 }));
          G.label(24, 168, "root", { size: 10 });
          G.text(24, 330, "children keyed by the FIRST token of the edge:", { size: 12, color: "ink" });
          G.label(24, 348, "node SYSTEM.children = { 'h': ..., 'b': ... }", { size: 12 });
          G.label(24, 366, "concatenating the slots along a path = KV slots of that prefix", { size: 12 });
          G.label(24, 384, "the tree stores ids and slots; the K/V tensors stay in the pool", { size: 12 });
          G.from(ln, { opacity: 0, duration: 0.3, stagger: 0.1 });
          G.caption("nodes appear only where sequences branch or end");
        } },

      { rail: "match + split", title: "Matching: walk down, and split an edge on a partial match",
        body: `<p><code>match_prefix(tokens)</code> walks from the root. At each node it looks up the child by the next token, then counts how many tokens of that child's edge agree with the prompt (<code>_common</code>). If the whole edge agrees, it steps into the child and continues. If only the first <i>n</i> tokens agree, the match ends <i>inside</i> an edge.</p>
<p>Inside an edge there is no node to point at, and the engine will want one: it must lock the matched prefix (step 6) and may later hang a new branch there. So <code>_split</code> cuts the edge in two:</p>
<div class="eq">before:  root -- SYSTEMhello  (slots 0..10)
match "SYSTEMhi": agree on 7 tokens, edge has 11
after:   root -- SYSTEMh (0..6) -- ello (7..10)
returns: (7, slots [0..6], node SYSTEMh)</div>
<p>Splitting loses no information: the two new edges hold the same tokens and slots as the old one. The new upper node inherits the old node's <code>ref</code> and <code>last_access</code>. One detail that is easy to get wrong: the <i>parent's</i> child table is keyed by the first token <code>S</code>, which is unchanged, so it must now point at the <b>upper</b> node, and the upper node must list the lower one under its own first token <code>e</code>.</p>
<p><code>match_prefix</code> returns the matched length, the slots of the matched tokens, and the last node. The request then starts prefill at position 7 with those 7 slots already in its table. It also refreshes <code>last_access</code> along the way, so matched edges look recently used.</p>`,
        check: { q: "The tree holds the single sequence 'SYSTEMhello'. You call match_prefix on 'SYSTEMhi'. What happens?",
          options: ["Returns 0: the sequences are different", "Returns 7 and splits the edge into 'SYSTEMh' and 'ello'; nothing new is stored", "Returns 8, since both have 8 or more tokens", "Inserts 'i' as a new leaf"], answer: 1,
          why: "The first 7 tokens (S Y S T E M h) agree, then 'e' meets 'i'. The match ends inside the edge, so the edge is split to give the match a node. Nothing is inserted: new tokens only enter the tree when the request later calls insert. This is the 'quiz 1' case, and I ran it on the reference: it returns 7 and the tree size stays 11." },
        scene(G) {
          G.text(24, 32, "match 'SYSTEMhi' against 'SYSTEMhello'", { size: 13 });
          G.label(24, 56, "before: one edge, 11 tokens", { size: 11 });
          const word = "SYSTEMhello";
          word.split("").forEach((ch, i) => cell(G, 24 + i * 38, 64, 34, 34, ch, i < 7 ? "ok" : "k", { size: 14 }));
          G.label(24, 116, "prompt:", { size: 11 });
          "SYSTEMhi".split("").forEach((ch, i) => cell(G, 24 + i * 38, 124, 34, 34, ch, i < 7 ? "ok" : "v", { size: 14 }));
          G.line(24 + 7 * 38 - 2, 56, 24 + 7 * 38 - 2, 166, { color: "hot", w: 2, dash: "4 3" });
          G.text(24 + 7 * 38 + 8, 182, "e ≠ i: the match ends inside the edge", { size: 11, color: "hot" });
          G.label(24, 214, "after the split (no tokens or slots lost):", { size: 11 });
          const a = edge(G, 24, 228, "SYSTEMh", "slots 0–6", { fill: "ok" });
          const b = edge(G, 24 + a.w + 36, 228, "ello", "7–10", { fill: "k" });
          G.arrow(24 + a.w + 2, 251, 24 + a.w + 34, 251, { color: "ink", w: 2 });
          G.label(24, 300, "returns (7, slots 0..6, node 'SYSTEMh')", { size: 12, color: "ink" });
          G.label(24, 322, "root.children['S'] now points at the UPPER node;", { size: 12 });
          G.label(24, 342, "upper.children['e'] points at the lower one", { size: 12 });
          G.text(24, 386, "the request prefills from position 7", { size: 14, color: "ok", weight: 700 });
          G.from([a.g, b.g], { opacity: 0, y: 8, duration: 0.4, stagger: 0.2 });
          G.caption("a partial match splits an edge so the match ends at a node");
        } },

      { rail: "insert", title: "Insert: share what exists, hang the rest as a leaf",
        body: `<p>When a request finishes prefill (and again when it finishes), its tokens and the slots holding their K/V are put in the tree with <code>insert(tokens, values)</code>. Insert is a match that keeps going: walk down as far as the tree agrees (splitting an edge if it ends mid-edge), then hang the leftover tokens as one new leaf.</p>
<p>It returns <b>how many tokens were already cached</b>. That number matters for memory. The request computed its own K/V for <i>all</i> its tokens into its own slots, even the ones the tree already had. The tree keeps its slots for the shared part, so the request's duplicates are now garbage, and the <b>caller must free them</b>.</p>
<div class="eq">tree holds SYSTEMhello (slots 0..10), then
insert("SYSTEMhi", slots 100..107):
  walk: 7 tokens agree  (split at 7)
  new leaf "i" with slot 107   tree size 11 -> 12
  returns 7
caller frees slots 100..106   (duplicates)
caller keeps slot 107     (the tree owns it)</div>
<p>Forget the free and every request that hits the cache leaks as many slots as it matched. That is the last item in "common mistakes" in the module README, and it is invisible in the output: the engine just runs out of KV memory sooner.</p>
<p>Note the tree has a single copy of each prefix. Once "SYSTEM" is in it, any number of requests extend it without storing it again.</p>`,
        scene(G) {
          G.text(24, 32, "insert('SYSTEMhi', slots 100..107)", { size: 13 });
          G.label(24, 56, "request's own slots", { size: 11 });
          "SYSTEMhi".split("").forEach((ch, i) => cell(G, 24 + i * 56, 64, 52, 44, ch + " · " + (100 + i), i < 7 ? "hot" : "v", { size: 11 }));
          G.arrow(24 + 3.5 * 56, 112, 24 + 3.5 * 56, 134, { color: "hot", w: 2 });
          G.text(24, 154, "7 duplicates (100..106): the tree already has this KV → caller frees them", { size: 12, color: "hot" });
          G.label(24, 190, "after insert: the tree keeps slots 0–6 for the shared part", { size: 11, color: "ok" });
          const a = edge(G, 24, 226, "SYSTEMh", "slots 0–6", { fill: "ok" });
          const b = edge(G, 24 + a.w + 30, 200, "ello", "7–10", { fill: "k" });
          const c = edge(G, 24 + a.w + 30, 256, "i", "107", { fill: "v" });
          G.line(24 + a.w, 249, 24 + a.w + 30, 223, { color: "muted", w: 1.5 }); G.line(24 + a.w, 249, 24 + a.w + 30, 279, { color: "muted", w: 1.5 });
          G.text(24, 340, "returns 7 = already cached", { size: 14, color: "ink" });
          G.label(24, 362, "tree size 11 → 12: only the one new token is added", { size: 12 });
          G.label(24, 382, "forget to free the duplicates → one KV leak per hit", { size: 12, color: "hot" });
          G.from([c.g], { opacity: 0, x: -10, duration: 0.4 });
          G.caption("insert returns the shared length; free your duplicates");
        } },

      { rail: "locks", title: "Lock the whole path while a request is using it",
        body: `<p>A running request reads the K/V of its matched prefix on every step. The tree must not give those slots away. So the request <b>locks</b> its last node: <code>lock(node)</code> adds one to <code>ref</code> on that node <b>and every ancestor up to the root</b>. <code>unlock</code> subtracts one the same way when the request ends. A node is evictable only if its <code>ref</code> is 0.</p>
<p>Why all the way up? A node's K/V was computed <i>attending to its ancestors' tokens</i>. If an ancestor were evicted, its descendant would still hold KV but could no longer be reached from the root, and a path with a hole is no longer a cached sequence. Counting the whole path makes "a node is in use" and "an ancestor is in use" the same thing.</p>
<div class="eq">two requests use SYSTEMhello and SYSTEMbye
lock(hello): SYSTEM 1, h 1, ello 1
lock(bye):   SYSTEM 2, bye 1
unlock(hello): SYSTEM 1, h 0, ello 0
SYSTEM stays locked until BOTH have finished</div>
<p>The cache also keeps <code>locked</code>, the number of tokens with <code>ref &gt; 0</code>, so <code>evictable() = size − locked</code> says how much it could free right now. In the 15-token tree above, locking only <code>SYSTEMhello</code> gives <code>locked = 11</code> (SYSTEM 6 + h 1 + ello 4) and <code>evictable = 4</code> (i and bye). The reference returns exactly that.</p>`,
        check: { q: "Why does lock(node) increment the ref count of every ancestor up to the root, not only the node?",
          options: ["It makes the tree faster to search", "Evicting an ancestor would make the node unreachable and break the prefix its KV depends on, so the whole path must stay while it is in use", "Ancestors are always leaves"], answer: 1,
          why: "A cached sequence is a path from the root. The node's K/V was computed by attending to every token above it, so the path above it is part of what it means. Ref counts on all ancestors mean an ancestor cannot be evicted (or even become a leaf) while any descendant is in use. Quiz question 3 asks the same thing." },
        scene(G) {
          G.text(24, 32, "ref counts along the path (request using SYSTEMhello)", { size: 13 });
          const a = edge(G, 24, 150, "SYSTEM", "ref 1", { fill: "q" });
          const h = edge(G, 154, 100, "h", "ref 1", { fill: "q" });
          const el = edge(G, 224, 60, "ello", "ref 1", { fill: "q" });
          const i = edge(G, 224, 140, "i", "ref 0", { stroke: "line" });
          const b = edge(G, 154, 220, "bye", "ref 0", { stroke: "line" });
          G.line(124, 168, 154, 123, { color: "muted", w: 1.5 }); G.line(124, 168, 154, 243, { color: "muted", w: 1.5 });
          G.line(194, 123, 224, 83, { color: "muted", w: 1.5 }); G.line(194, 123, 224, 163, { color: "muted", w: 1.5 });
          G.label(24, 142, "root", { size: 10 });
          G.label(310, 90, "← the request's last node", { size: 11, color: "q" });
          G.label(300, 164, "unlocked leaf: evictable", { size: 11 });
          G.label(250, 244, "unlocked leaf: evictable", { size: 11 });
          G.text(24, 316, "locked = 6 + 1 + 4 = 11 tokens", { size: 14, color: "q" });
          G.text(24, 342, "evictable = size − locked = 15 − 11 = 4", { size: 14, color: "ok" });
          G.label(24, 374, "lock walks to the root; unlock walks back and subtracts", { size: 12 });
          G.pulse([a.g, h.g, el.g], { lo: 0.6, repeat: 4 });
          G.caption("a node is in use if it, or anything below it, is in use");
        } },

      { rail: "eviction", title: "Eviction: least recently used, unlocked leaves only",
        body: `<p>The pool is finite, so the tree must give slots back. <code>evict(n)</code> frees <b>at least</b> <i>n</i> tokens by removing whole leaves. Which ones?</p>
<ul><li><b>Leaves only.</b> An internal node's tokens are a prefix of its children's sequences, so removing it would orphan them (same reason as locks).</li>
<li><b>Unlocked only</b> (<code>ref == 0</code>): never take KV out from under a running request.</li>
<li><b>Least recently used first</b>: the leaf with the smallest <code>last_access</code>, a logical clock bumped on every match and insert.</li></ul>
<p>It uses a min-heap of eligible leaves. When a leaf is removed and its parent has no children left and is unlocked, the parent has just become a leaf, so it joins the heap. Eviction can therefore eat a whole branch from the tips inward, but never from the middle.</p>
<div class="eq">tree: SYSTEM -- h -- ello   (accessed 11, locked)
                \\-- i      (accessed 5)
         \\-- bye            (accessed 8)
evict(1): oldest unlocked leaf is "i": 1
evict(2): "i" (1) then "bye" (3) -> frees 4 >= 2
evict(100): only 4 are evictable -> frees 4, stops
unlock, then evict(100): ello, then h, then
  SYSTEM, each once its children are gone = 11</div>
<p>I ran these calls on the reference and got 1, 4, 4 and 11 tokens. Note the granularity: <code>evict(2)</code> freed 4, because a leaf goes whole. The returned list is the freed <i>slots</i>; the caller hands them back to the KV pool. In #0 v1 the block manager (P6.3) still owns KV memory and uses block hashes for the engine's prefix reuse; the radix cache is a standalone module you test on its own, and exercise 2 asks how you would make it the owner.</p>`,
        check: { q: "Why can evict() only remove leaves?",
          options: ["Leaves hold the most data", "An internal node's tokens are a prefix of its children's sequences; their KV was computed attending to it, so removing it would orphan them", "Internal nodes are always locked"], answer: 1,
          why: "A path from the root is a cached sequence. If you remove a node in the middle, the nodes below it no longer have a path, so their KV cannot be matched and the slots leak. Evicting leaves first keeps every remaining path intact; a parent becomes evictable only after its last child is gone." },
        scene(G) {
          G.text(24, 32, "evict: LRU among unlocked leaves", { size: 13 });
          const a = edge(G, 24, 160, "SYSTEM", "acc 9", { fill: "q" });
          const h = edge(G, 154, 104, "h", "acc 10", { fill: "q" });
          const el = edge(G, 224, 64, "ello", "acc 11", { fill: "q" });
          const i = edge(G, 224, 140, "i", "acc 5", { fill: "hot" });
          const b = edge(G, 154, 220, "bye", "acc 8", { fill: "hot" });
          G.line(124, 178, 154, 127, { color: "muted", w: 1.5 }); G.line(124, 178, 154, 243, { color: "muted", w: 1.5 });
          G.line(194, 127, 224, 87, { color: "muted", w: 1.5 }); G.line(194, 127, 224, 163, { color: "muted", w: 1.5 });
          G.text(300, 168, "①", { size: 20, color: "hot" }); G.text(240, 246, "②", { size: 20, color: "hot" });
          G.label(330, 168, "'i': oldest (acc 5) → evicted first", { size: 11, color: "hot" });
          G.label(290, 246, "'bye' (acc 8) → second", { size: 11, color: "hot" });
          G.label(300, 88, "locked: never", { size: 11, color: "q" });
          G.text(24, 320, "evict(1) → 1 token · evict(2) → 4 · evict(100) → 4", { size: 13, color: "ink" });
          G.label(24, 344, "a leaf goes whole, so you may get more than you asked for", { size: 12 });
          G.label(24, 366, "unlock everything: ello, then h, then SYSTEM follow (11 tokens)", { size: 12 });
          G.pulse([i.g, b.g], { lo: 0.5, repeat: 4 });
          G.caption("tips first, oldest first, never a locked path");
        } },

      { rail: "radix vs blocks", title: "Radix tree vs block hashes: what the extra structure buys",
        body: `<p>Both schemes find KV another request computed. They differ in granularity. A block hash (P6.3) matches whole blocks. A radix tree matches any number of tokens.</p>
<table><tr><th></th><th>block hash (vLLM)</th><th>radix tree (SGLang)</th></tr>
<tr><td>match unit</td><td>full blocks</td><td>any token</td></tr>
<tr><td>lookup</td><td>one hash probe per block</td><td>one tree walk</td></tr>
<tr><td>evicts</td><td>a block (LRU of freed blocks)</td><td>a leaf's token run (LRU of unlocked leaves)</td></tr>
<tr><td>state</td><td>hash tables, stable across processes</td><td>a tree in one process</td></tr></table>
<p>The gain per request is at most <code>bs − 1</code> tokens, the partial block at the end of the shared prefix:</p>
<div class="eq">block size 16, shared prefix S, prompt long enough
block hash skips  floor(S / 16) * 16
radix skips       S
S =  50:  48 vs  50  (2 tokens, 4 %)
S = 500: 496 vs 500  (4 tokens;  0.8 %)</div>
<p>So the radix tree matters most for short shared prefixes, and for long ones the difference is a few tokens. <code>bench/hit_rate.py</code> (seed 0, fake model) measured two synthetic traces, as the fraction of prompt tokens served from cache with unlimited capacity:</p>
<table><tr><th>trace</th><th>prompt tokens</th><th>block hash (bs 16)</th><th>radix</th></tr>
<tr><td>multi-turn chat</td><td>29,230</td><td>0.682</td><td>0.709</td></tr>
<tr><td>few-shot (shared header 30 to 50 tokens)</td><td>12,176</td><td>0.784</td><td>0.819</td></tr></table>
<p>Radix is never lower and the gap is a few points. It is a property of these synthetic traces, not of your traffic: exercise 3 has you add a trace and a capacity limit. Whether a few points justify a second data structure depends on how short your shared prefixes are.</p>`,
        check: { q: "Block size 16. 1,000 requests share a 50-token system prompt, then diverge. Roughly how many prefix tokens per request do block hashing and a radix tree reuse?",
          options: ["50 and 50", "48 and 50", "32 and 50", "64 and 50"], answer: 1,
          why: "Block hashing matches full blocks only: 50 // 16 = 3 blocks = 48 tokens. The radix tree matches to the token: 50. The difference is at most bs − 1 = 15 tokens, and with a 500-token prefix it shrinks to 4 (496 vs 500)." },
        scene(G) {
          G.text(24, 32, "tokens reused per request, block size 16", { size: 13 });
          const rows = [["S = 50", 48, 50], ["S = 500", 496, 500]];
          const out = [];
          rows.forEach(([lab, bl, rd], r) => {
            const y = 60 + r * 100, sc = 500 / (r ? 500 : 50);
            G.text(24, y + 14, lab, { size: 12, weight: 700 });
            out.push(G.rect(24, y + 22, bl * sc / 1.0 * (r ? 1 : 1), 22, { fill: "w", rx: 3 }));
            G.label(24 + bl * sc + 6, y + 38, "block hash " + bl, { size: 11, color: "ink" });
            out.push(G.rect(24, y + 50, rd * sc, 22, { fill: "ok", rx: 3 }));
            G.label(24 + rd * sc + 6, y + 66, "radix " + rd, { size: 11, color: "ok" });
          });
          G.label(24, 254, "(each bar pair scaled to its own row)", { size: 10 });
          G.text(24, 276, "bench/hit_rate.py (hit fraction of prompt tokens)", { size: 12 });
          [["multi-turn", 0.682, 0.709], ["few-shot", 0.784, 0.819]].forEach(([n, a, b], i) => {
            const y = 290 + i * 54;
            G.label(24, y + 12, n, { size: 11 });
            out.push(G.rect(110, y, 400 * a, 18, { fill: "w", rx: 3 })); G.label(110 + 400 * a + 6, y + 13, a.toFixed(3), { size: 11 });
            out.push(G.rect(110, y + 22, 400 * b, 18, { fill: "ok", rx: 3 })); G.label(110 + 400 * b + 6, y + 35, b.toFixed(3) + " radix", { size: 11, color: "ok" });
          });
          G.from(out, { attr: { width: 0 }, stagger: 0.08, duration: 0.4 });
          G.caption("radix wins by at most block_size − 1 tokens per request");
        } },

      { rail: "cache-aware", title: "Cache-aware scheduling: serve the longest cached prefix first",
        body: `<p>A cache hit is only worth something if the cached KV is still there when you need it. Eviction is driven by the order requests run in, so order is a lever.</p>
<p>Take a queue of three waiting requests. Two share a 1,000-token system prompt that is cached; one has an unrelated 1,020-token prompt. The cache holds about 1,100 tokens. An <b>illustration</b>, with these assumptions: every request adds 20 unique tokens, and the two sharing requests finish before the unrelated one is admitted when they go first:</p>
<div class="eq">FCFS: unrelated request first
  it needs 1,020 slots, the cache has 1,100,
  so the unlocked shared prompt is evicted
  then each shared request recomputes 1,020
  tokens computed = 1,020 + 1,020 + 1,020 = 3,060
Longest-prefix-first (LPM): shared requests first
  each computes 20 (their unique suffix)
  the unrelated one computes 1,020
  tokens computed = 20 + 20 + 1,020 = 1,060</div>
<p>SGLang's <b>longest-prefix-match</b> policy does this: among waiting requests, prefer the one with the longest cached prefix (found with <code>match_prefix</code>), and lock its prefix so it cannot be evicted before use. The risk is <b>starvation</b>: a request with no cached prefix always comes last, and if shared-prefix requests keep arriving it never runs. P6.2's aging bounds that: effective priority = priority + waited // aging_steps. Exercise 4 measures total tokens computed, mean and max wait for FCFS, LPM, and LPM + aging on a trace with a cache of 25 % of the working set.</p>`,
        check: { q: "What is the main risk of longest-prefix-first scheduling, and how do you bound it?",
          options: ["It computes more tokens in total; use a bigger cache", "Requests with no cached prefix can wait forever; add aging so every request's wait is bounded", "It breaks the radix tree's invariants; lock fewer nodes"], answer: 1,
          why: "LPM saves compute by reordering, and reordering can starve whoever is last. If shared-prefix requests keep arriving, the unrelated request is always behind them. Aging (P6.2) turns waiting time into priority, so the wait is bounded again." },
        scene(G) {
          const lane = (y, title, items) => {
            G.text(24, y, title, { size: 13, weight: 700 });
            let x = 24; const out = [];
            items.forEach(([t, n, c]) => { const w = (n / 3060) * 570 + 6; out.push(cell(G, x, y + 12, Math.max(w, 30), 40, t, c, { size: 11, rx: 4 })); if (n > 100) G.label(x + 2, y + 66, n + " tok", { size: 11 }); x += Math.max(w, 30) + 3; });
            return out;
          };
          G.text(24, 26, "illustration: cache ≈ 1,100 tokens, shared prompt 1,000", { size: 12 });
          const a = lane(56, "FCFS · computed 3,060", [["unrelated", 1020, "hot"], ["shared 1", 1020, "w"], ["shared 2", 1020, "w"]]);
          G.label(24, 268, "s1, s2: 20 tokens each (their unique suffix)", { size: 11 });
          const b = lane(176, "LPM · computed 1,060", [["unrelated", 1020, "hot"], ["s1", 20, "ok"], ["s2", 20, "ok"]].sort((x, y) => (x[1] - y[1])));
          G.label(24, 300, "FCFS: the unrelated prompt evicts the shared one,", { size: 12, color: "ink" });
          G.label(24, 320, "so both shared requests recompute it", { size: 12 });
          G.label(24, 346, "LPM: shared requests first, their prefix is locked and hit", { size: 12, color: "ink" });
          G.text(24, 386, "cost: the unrelated request waits → add aging", { size: 13, color: "hot" });
          G.from([...a, ...b], { opacity: 0, x: -10, stagger: 0.08, duration: 0.25 });
          G.caption("order decides what is still cached when you need it");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Finding the longest cached prefix by comparing with every sequence costs N × length. A trie merges shared beginnings so one walk finds it; a radix tree merges unbranching chains into edges of token runs plus their KV slots.</li>
<li><code>match_prefix</code> walks while tokens agree and <b>splits</b> an edge on a partial match, so the match ends at a node. <code>insert</code> shares the existing prefix, hangs the rest as a leaf, and returns how many tokens were already cached; the caller must free its duplicate slots.</li>
<li><code>lock</code> adds a reference on the node and every ancestor to the root; <code>evict</code> removes least-recently-used unlocked <i>leaves</i> only, and a parent joins the heap once its last child is gone.</li>
<li>Against block hashing (P6.3) the tree gains at most <code>bs − 1</code> tokens per request: 48 vs 50 for a 50-token prefix, 496 vs 500 for 500. On the synthetic traces in <code>bench/hit_rate.py</code> it is 0.709 vs 0.682 and 0.819 vs 0.784.</li>
<li>Serving the longest cached prefix first saves recompute but can starve unmatched requests; bound the wait with aging from P6.2.</li></ul>`,
    sim: {
      title: "Tokens reused: radix tree against block hashing",
      intro: "N requests share a prefix of S tokens and then diverge; each prompt has P tokens. The first request fills the cache, every later one reuses what the scheme can match. Both rules keep one prompt token uncomputed, as the reference does. The chart sweeps S from 0 to P for your block size: the grey staircase is block hashing, the green line is the radix tree, the red marker is your S.",
      height: 280,
      controls: [
        { id: "bs", label: "block size (tokens)", type: "select", value: 16, options: [[1, "1"], [4, "4"], [8, "8"], [16, "16"], [32, "32"], [64, "64"]] },
        { id: "P", label: "prompt length P (tokens)", min: 16, max: 1024, step: 8, value: 128 },
        { id: "S", label: "shared prefix S (tokens)", min: 0, max: 1024, step: 1, value: 50 },
        { id: "N", label: "requests sharing it (N)", min: 2, max: 1000, step: 1, value: 200 },
      ],
      draw(G, v) {
        const bs = +v.bs, P = v.P, S = Math.min(v.S, P), N = v.N;
        const hitBlock = (s) => Math.min(Math.floor(s / bs), Math.floor((P - 1) / bs)) * bs;
        const hitRadix = (s) => Math.min(s, P - 1);
        const x0 = 50, W = 570, H = 190, base = 230, X = (s) => x0 + (W * s) / P, Y = (t) => base - (H * t) / P;
        G.axes(x0, base - H, W, H);
        G.label(10, 16, "prefix tokens reused per request, against shared-prefix length S", { size: 11 });
        let d = "";
        for (let s = 0; s <= P; s++) d += `${s ? "L" : "M"} ${X(s)} ${Y(hitBlock(s))} `;
        G.path(d, { color: "w", w: 2.5 });
        let r = "";
        for (let s = 0; s <= P; s++) r += `${s ? "L" : "M"} ${X(s)} ${Y(hitRadix(s))} `;
        G.path(r, { color: "ok", w: 2 });
        G.line(X(S), base, X(S), Y(hitRadix(S)), { color: "hot", dash: "3 3" });
        G.circle(X(S), Y(hitRadix(S)), 4, { fill: "ok" }); G.circle(X(S), Y(hitBlock(S)), 4, { fill: "hot" });
        G.label(x0, base + 18, "S = 0", { size: 10 }); G.label(x0 + W, base + 18, "S = " + P, { size: 10, anchor: "end" });
        G.label(x0 - 6, base - H + 4, String(P), { size: 10, anchor: "end" });
        G.label(10, 270, "green: radix (any token) · grey: block hash (full blocks) · the gap is never more than bs − 1", { size: 10 });
        const a = hitBlock(S), b = hitRadix(S), gap = b - a;
        const tot = N * P, hitsB = (N - 1) * a, hitsR = (N - 1) * b;
        return [
          { title: "One request", rows: [["radix reuses", F.num(b, 0)], ["block hash reuses", F.num(a, 0)], ["radix advantage", `${gap} tokens`], ["blocks matched", `${Math.min(Math.floor(S / bs), Math.floor((P - 1) / bs))}`]],
            chip: [gap < bs, `gap ${gap} ≤ bs − 1 = ${bs - 1}`] },
          { title: `${N} requests`, rows: [["prompt tokens", F.num(tot, 0)], ["hit rate, block hash", (100 * hitsB / tot).toFixed(1) + " %"], ["hit rate, radix", (100 * hitsR / tot).toFixed(1) + " %"], ["extra tokens saved by radix", F.num(hitsR - hitsB, 0)]],
            html: `<p class="note">The first request misses (nothing is cached yet), so hit rates top out below 100 % even for a fully shared prefix.</p>` },
        ];
      },
      note: "A model of the matching rules only: unlimited capacity, one shared prefix, no eviction and no divergence inside a block other than at S. Real hit rates depend on your traffic (bench/hit_rate.py gives 0.682 vs 0.709 on its multi-turn trace).",
    },
    practice: {
      intro: "Everything here is T0: pure Python, no GPU. Implement the cache in exercises/my_radix.py; the tests fail until you do, and S2S_SOLUTIONS=1 runs them against the reference in platform/engine/v1.",
      items: [
        { title: "Insert, match, split against a naive trie", tier: "T0 · medium", goal: "Implement match_prefix (with edge split) and insert. 50 random trees check the match length against a brute-force longest common prefix, the returned slots, insert's already-cached length and the tree invariants.",
          cmd: "uv run pytest course/P6-engine-internals/P6.4-prefix-cache-radix-tree/exercises -q -k 'match or split'" },
        { title: "LRU eviction that respects locks", tier: "T0 · medium", goal: "Implement lock, unlock and evict: only unlocked leaves, least recent first, and a parent that loses its last child becomes a candidate. Then sketch how the radix cache could own KV memory at block granularity.",
          cmd: "uv run pytest course/P6-engine-internals/P6.4-prefix-cache-radix-tree/exercises -q -k evict" },
        { title: "Hit rate on real workload shapes", tier: "T0 · medium", goal: "Add a third trace and a capacity limit to the bench; fill the table of block-hash vs radix hit rates, including a 25 % capacity row.",
          cmd: "uv run python course/P6-engine-internals/P6.4-prefix-cache-radix-tree/bench/hit_rate.py" },
        { title: "Cache-aware (longest-prefix-first) scheduling", tier: "T0 · hard", goal: "Among equal-priority waiting requests admit the one with the longest cached prefix. Report tokens computed and mean/max wait for FCFS, LPM, LPM + aging and your best variant.",
          cmd: "cat course/P6-engine-internals/P6.4-prefix-cache-radix-tree/exercises/04-cache-aware.md" },
        { title: "The reference's tests", tier: "T0", goal: "Run the radix cache tests of the engine (random operations against the naive trie, split, LRU, locks).",
          cmd: "uv run pytest platform/engine/v1/tests/test_radix_cache.py -q" },
      ],
      labs: [
        { label: "The reference radix cache you are re-implementing", path: "platform/engine/v1/s2s_engine/radix_cache.py" },
        { label: "Block hash vs radix hit rates on two traces", path: "course/P6-engine-internals/P6.4-prefix-cache-radix-tree/bench/hit_rate.py" },
        { label: "Block-hash prefix reuse (P6.3), the other scheme", path: "platform/engine/v1/s2s_engine/block_manager.py" },
        { label: "Animation: radix tree operations", path: "animations/p6-radix-tree.html" },
        { label: "Quiz", path: "course/P6-engine-internals/P6.4-prefix-cache-radix-tree/quiz.md" },
        { label: "Read, don't copy: SGLang radix_cache.py and schedule_policy.py at the pin (re-verify paths)", path: "python/sglang/srt/mem_cache/radix_cache.py" },
        { label: "Paper: Zheng et al., SGLang: Efficient Execution of Structured Language Model Programs, section 3 (arXiv 2312.07104)", path: "arXiv 2312.07104" },
      ],
    },
  });
})();
