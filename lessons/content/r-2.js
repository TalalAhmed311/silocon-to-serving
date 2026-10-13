/* R.2 — Memory: stack, heap and ownership. Where variables live, the three classic bugs, ASan, RAII, unique_ptr, copy vs move. */
(function () {
  // ---------- simulator programs: each step is a full snapshot ----------
  // vars: [name, value]  value "B1" = pointer to block B1, "null", "?" or plain text
  const PROGRAMS = {
    correct: {
      title: "correct: allocate, use, free",
      code: ["int main() {", "  int* p = new int[4];", "  p[0] = 7;", "  helper();", "  delete[] p;", "  p = nullptr;", "}"],
      steps: [
        { line: 0, frames: [["main", [["p", "?"]]]], heap: [], note: "main starts. Its frame holds p, not yet set." },
        { line: 1, frames: [["main", [["p", "B1"]]]], heap: [["B1", "int[4] 16 B", "live"]], note: "new int[4] takes 16 bytes from the heap. p holds its address. p is the owner." },
        { line: 2, frames: [["main", [["p", "B1"]]]], heap: [["B1", "int[4] 16 B", "live"]], note: "Writing p[0] stays inside the block. Safe." },
        { line: 3, frames: [["main", [["p", "B1"]]], ["helper", [["x", "1"]]]], heap: [["B1", "int[4] 16 B", "live"]], note: "A call pushes a new frame on the stack. helper's local x lives there." },
        { line: 3, frames: [["main", [["p", "B1"]]]], heap: [["B1", "int[4] 16 B", "live"]], note: "helper returns: its frame is popped. x is gone. The heap block is untouched." },
        { line: 4, frames: [["main", [["p", "B1"]]]], heap: [["B1", "int[4] 16 B", "freed"]], dangling: ["p"], note: "delete[] p gives the block back. p still holds the old address: it now dangles." },
        { line: 5, frames: [["main", [["p", "null"]]]], heap: [["B1", "int[4] 16 B", "freed"]], note: "p = nullptr removes the dangling address. Good habit." },
        { line: 6, frames: [], heap: [["B1", "int[4] 16 B", "freed"]], ok: "main returns: every block was freed exactly once. No bug.", note: "The last frame is popped. Nothing leaked." },
      ],
    },
    leak: {
      title: "leak: the only pointer is overwritten",
      code: ["int main() {", "  int* p = new int[4];", "  p = new int[8];", "  delete[] p;", "}"],
      steps: [
        { line: 0, frames: [["main", [["p", "?"]]]], heap: [], note: "main starts." },
        { line: 1, frames: [["main", [["p", "B1"]]]], heap: [["B1", "int[4] 16 B", "live"]], note: "B1 is allocated. p is its only pointer." },
        { line: 2, frames: [["main", [["p", "B2"]]]], heap: [["B1", "int[4] 16 B", "leak"], ["B2", "int[8] 32 B", "live"]], flag: "LEAK: nothing points to B1 any more. Nobody can free it (16 B lost).", note: "new int[8] makes B2 and p is overwritten. The address of B1 is gone." },
        { line: 3, frames: [["main", [["p", "B2"]]]], heap: [["B1", "int[4] 16 B", "leak"], ["B2", "int[8] 32 B", "freed"]], flag: "B1 is still lost. delete[] p freed only B2.", note: "delete[] p frees whatever p points at now: B2." },
        { line: 4, frames: [], heap: [["B1", "int[4] 16 B", "leak"], ["B2", "int[8] 32 B", "freed"]], flag: "At exit: 16 bytes in 1 block leaked (LeakSanitizer reports this).", note: "main returns. B1 stays allocated until the process ends." },
      ],
    },
    uaf: {
      title: "use-after-free: a second pointer outlives the block",
      code: ["int main() {", "  int* p = new int[4];", "  int* q = p;", "  delete[] p;", "  q[0] = 1;", "}"],
      steps: [
        { line: 0, frames: [["main", [["p", "?"], ["q", "?"]]]], heap: [], note: "main starts." },
        { line: 1, frames: [["main", [["p", "B1"], ["q", "?"]]]], heap: [["B1", "int[4] 16 B", "live"]], note: "B1 is allocated. p owns it." },
        { line: 2, frames: [["main", [["p", "B1"], ["q", "B1"]]]], heap: [["B1", "int[4] 16 B", "live"]], note: "q is a copy of the address. Two pointers, one block. Who frees it?" },
        { line: 3, frames: [["main", [["p", "B1"], ["q", "B1"]]]], heap: [["B1", "int[4] 16 B", "freed"]], dangling: ["p", "q"], note: "delete[] p frees B1. p and q both dangle now, but only p's owner knows." },
        { line: 4, frames: [["main", [["p", "B1"], ["q", "B1"]]]], heap: [["B1", "int[4] 16 B", "freed"]], dangling: ["p", "q"], flag: "USE-AFTER-FREE: q[0] = 1 writes into freed memory (ASan: heap-use-after-free).", note: "The write goes to memory the allocator may already have given to someone else." },
        { line: 5, frames: [], heap: [["B1", "int[4] 16 B", "freed"]], flag: "The bug already happened at the previous step. Returning does not undo it.", note: "main returns." },
      ],
    },
  };
  const STATE_COL = { live: "ok", freed: "muted", leak: "hot" };

  S2S.lesson({
    id: "r-2", n: "R.2", title: "Memory: stack, heap and ownership",
    subtitle: "Readiness · first principles · T0, any laptop",
    kicker: "Lesson · ≈ 30 min reading + 4 h hands-on",
    headline: "Who frees this?",
    intro: `<p><b>Already comfortable with the stack and the heap, <code>new</code>/<code>delete</code>, leaks, <code>unique_ptr</code> and move? Answer the checkpoints. If you get them all right, skip to R.3.</b></p>
<p>Python frees memory for you. C++ does not, and most of the bugs you will meet in inference engines are memory bugs: a buffer freed twice, a pointer used after its block was released, a leak that grows until the server falls over. This lesson builds the picture that makes those bugs obvious: where every variable lives, how long it lives, and who is responsible for freeing it.</p>
<p>It assumes R.1 (pointers, references, constructors and destructors). Toolchain: <code>g++</code> or <code>clang++</code> 11 or newer, CMake 3.20 or newer.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "stack"], ["v", "heap"], ["q", "globals"], ["ok", "live / owned"], ["hot", "bug"]],
    prev: "r-1", next: "r-3",
    steps: [
      { rail: "address space", title: "A process has four regions of memory",
        body: `<p>When your program runs, the operating system gives it its own range of addresses. (They are <i>virtual</i> addresses; P0.2 explains the translation.) The range is divided into four regions with different jobs:</p>
<table><tr><th>region</th><th>holds</th><th>lives until</th><th>freed by</th></tr>
<tr><td><b>code</b></td><td>your compiled functions</td><td>the program ends</td><td>OS</td></tr>
<tr><td><b>globals</b></td><td>global and <code>static</code> variables</td><td>the program ends</td><td>OS</td></tr>
<tr><td><b>stack</b></td><td>locals and call frames</td><td>the function returns</td><td>automatic</td></tr>
<tr><td><b>heap</b></td><td>blocks from <code>new</code> and <code>malloc</code></td><td>you free them</td><td><b>you</b></td></tr></table>
<p>The stack and the heap start far apart and grow toward each other: the stack down, the heap up. Which region a value lives in decides <b>how long it lives and who must free it</b>. The rest of this lesson is that one question.</p>`,
        scene(G) {
          G.label(24, 34, "high addresses", { size: 12 });
          const seg = [
            ["stack  (grows down)", 50, 70, "k", 0.85], ["unused: a huge gap", 120, 56, "line", 0.3],
            ["heap  (grows up)", 176, 70, "v", 0.85], ["globals / statics", 246, 50, "q", 0.85], ["code", 296, 50, "w", 0.7],
          ];
          const boxes = seg.map(([n, y, h, c, o]) => { const r = G.rect(50, y, 250, h - 2, { fill: c, rx: 6, opacity: o, dash: c === "line" ? "5 4" : undefined, stroke: c === "line" ? "muted" : undefined }); G.text(175, y + h / 2 + 3, n, { anchor: "middle", size: 13, weight: 700, color: c === "line" ? "muted" : "bg" }); return r; });
          G.label(24, 376, "low addresses", { size: 12 });
          G.arrow(34, 56, 34, 112, { color: "k", w: 2.5 }); G.arrow(34, 224, 34, 180, { color: "v", w: 2.5 });
          const ex = [["local variables, one frame per call", 85], ["", 148], ["new / malloc: size chosen at run time", 211], ["global and static variables", 276], ["main(), mean(), every function", 321]];
          ex.forEach(([t, y]) => t && G.text(318, y, t, { size: 12 }));
          G.text(24, 408, "stack: freed for you  ·  heap: freed by you", { size: 13, color: "ok" });
          G.from(boxes, { opacity: 0, x: -16, stagger: 0.1, duration: 0.3 });
          G.caption("four regions: the stack and the heap decide most memory bugs");
        } },

      { rail: "the stack", title: "The stack: one frame per call, freed on return",
        body: `<p>Every function call needs room for its parameters, its local variables and the place to return to. That room is a <b>stack frame</b>. A call <b>pushes</b> a frame on top of the stack. A return <b>pops</b> it. A single register, the stack pointer, marks the top.</p>
<div class="eq">int parse(int x) { int y = x * 2; return y; }
int load(int n)  { int a = parse(n); return a + 1; }
int main()       { int r = load(3); }</div>
<p>While <code>parse</code> runs, three frames exist: <code>main</code>, <code>load</code>, <code>parse</code>. When <code>parse</code> returns, its frame is popped and <code>y</code> stops existing. You never write a <code>free</code>. That is why locals are cheap and safe, and why a <code>for</code> loop's variables vanish after the loop.</p>
<p>The price: the stack is small. Linux gives the main thread 8 MiB by default (<code>ulimit -s</code> prints it in KiB: 8192 here). A local <code>float big[10000000]</code> is 10,000,000 × 4 = 40,000,000 bytes, so it overflows the stack and the program crashes. Big data goes on the heap.</p>`,
        scene(G) {
          const cols = [24, 232, 440], titles = ["while parse runs", "parse has returned", "load has returned"];
          const names = [["main", "r"], ["load", "n = 3, a"], ["parse", "x = 3, y = 6"]];
          const tops = [];
          cols.forEach((x, c) => {
            G.text(x, 70, titles[c], { size: 13, weight: 700 });
            const live = 3 - c;
            names.forEach(([n, v], i) => {
              const y = 330 - i * 70;
              if (i < live) {
                G.rect(x, y, 176, 60, { fill: "k", opacity: 0.25, stroke: "k", rx: 6 });
                G.text(x + 10, y + 24, n, { size: 13, weight: 700 }); G.text(x + 10, y + 44, v, { size: 12, color: "muted" });
                if (i === live - 1) tops.push(G.text(x + 126, y + 24, "◄ top", { size: 12, color: "ok" }));
              } else {
                G.rect(x, y, 176, 60, { stroke: "muted", dash: "4 4", rx: 6, opacity: 0.7 });
                G.text(x + 10, y + 34, n + ": popped", { size: 12, color: "muted" });
              }
            });
          });
          G.label(24, 408, "dashed = the frame is gone; its bytes will be reused by the next call", { size: 12 });
          G.from(tops, { opacity: 0, duration: 0.6, delay: 0.3 });
          G.caption("call = push a frame · return = pop it · nothing to free by hand");
        } },

      { rail: "dangling", title: "Returning a pointer to a local leaves it dangling",
        body: `<p>A local variable dies when its function returns. A pointer to it does not. If you return that pointer, the caller holds the address of a stack slot that no longer belongs to anyone:</p>
<div class="eq">int* f() {
  int x = 42;
  return &amp;x;     // x dies at the closing brace
}
int* p = f();   // p dangles from this line on
g();            // g's frame reuses the same bytes
int v = *p;     // ??? undefined behaviour</div>
<p>This is a <b>dangling pointer</b>. Right after <code>f</code> returns, the 42 usually still sits in the old bytes, so the code <i>seems</i> to work. Then the next function call pushes its own frame on the same bytes and overwrites it. The program changes behaviour when an unrelated line is added. Compilers warn about the simplest form (<code>-Wall</code>: "address of local variable returned"). Exercise 2 is this bug.</p>
<p>The fix is to return the <b>value</b>. A copy of the number, or a <code>std::vector</code> that owns its own storage, outlives the frame.</p>`,
        check: { q: "After the code above, *p is read once f has returned and g has been called. What does the language guarantee?",
          options: ["It reads 42, because nobody cleared the memory", "It reads whatever g left there, but never crashes", "Nothing: it is undefined behaviour, so it may read 42, read a different number, or crash"], answer: 2,
          why: "The stack slot belongs to no variable once f returns. Reading it is undefined behaviour. 42 may survive for a while, which is why the bug can hide, but g's frame is free to overwrite it, and the compiler is free to do anything else." },
        scene(G) {
          const cols = [24, 232, 440], tt = ["1  inside f", "2  f has returned", "3  g has been called"];
          cols.forEach((x, c) => G.text(x, 60, tt[c], { size: 13, weight: 700 }));
          // main frames
          cols.forEach((x) => { G.rect(x, 300, 176, 70, { fill: "k", opacity: 0.25, stroke: "k", rx: 6 }); G.text(x + 10, 322, "main", { size: 13, weight: 700 }); });
          G.text(24 + 10, 346, "p: not set yet", { size: 12, color: "muted" });
          G.text(232 + 10, 346, "p = &x  (dangling)", { size: 12, color: "hot" });
          G.text(440 + 10, 346, "p = &x  (dangling)", { size: 12, color: "hot" });
          // slot above main
          G.rect(24, 210, 176, 70, { fill: "k", opacity: 0.25, stroke: "k", rx: 6 }); G.text(34, 232, "f", { size: 13, weight: 700 }); G.text(34, 260, "x = 42", { size: 13 });
          G.rect(232, 210, 176, 70, { stroke: "muted", dash: "4 4", rx: 6 }); G.text(242, 232, "(f popped)", { size: 12, color: "muted" }); const s2 = G.text(242, 260, "42 still there", { size: 13, color: "v" });
          G.rect(440, 210, 176, 70, { fill: "v", opacity: 0.25, stroke: "v", rx: 6 }); G.text(450, 232, "g", { size: 13, weight: 700 }); const s3 = G.text(450, 260, "k = 7", { size: 13 });
          G.arrow(380, 298, 380, 284, { color: "hot", w: 2.5 }); G.arrow(588, 298, 588, 284, { color: "hot", w: 2.5 });
          G.text(24, 130, "p = f();", { size: 13 }); G.text(232, 120, "*p  →  42   (lucky)", { size: 13, color: "v" }); G.text(440, 120, "*p  →  7   (overwritten)", { size: 13, color: "hot" });
          G.label(232, 144, "works by accident", { size: 11 }); G.label(440, 144, "g reused the same bytes", { size: 11 });
          G.text(24, 408, "the pointer outlived the variable it points to", { size: 13, color: "hot" });
          G.from([s2, s3], { opacity: 0, delay: 0.3, stagger: 0.3, duration: 0.5 });
          G.caption("the address survives the return; the variable does not");
        } },

      { rail: "the heap", title: "The heap: memory whose lifetime you control",
        body: `<p>The stack frees itself, but it is small and tied to function calls. When you need a big array, or data that must outlive the function that made it, you ask the <b>heap</b> allocator for a block. It returns the block's address. The block stays yours until you give it back.</p>
<div class="eq">float* w = new float[4096];   // 4096 × 4 = 16 KiB
w[0] = 1.5f;
delete[] w;                    // give it back

int* one = new int(5);
delete one;</div>
<p>Rules that keep it correct:</p>
<ul><li>Every <code>new</code> needs exactly one matching <code>delete</code>; every <code>new[]</code> needs <code>delete[]</code>. Mixing them is a bug.</li>
<li>C code uses <code>malloc</code> and <code>free</code> instead. They pair with each other, never with <code>new</code> and <code>delete</code>.</li>
<li><code>new int[4]</code> leaves the values uninitialised (garbage). <code>new int[4]()</code> zeroes them.</li></ul>
<p>The pointer on the stack and the block on the heap are two separate things. The pointer dies with its frame. The block does not. If the pointer is lost before <code>delete</code>, nobody can ever free the block. Next: the three ways that goes wrong.</p>`,
        scene(G) {
          G.label(24, 40, "stack (pointers)", { size: 12 }); G.label(380, 40, "heap (blocks)", { size: 12 });
          const ptr = [["float* w", 60], ["int* ids", 130]];
          const blk = [["float[4096]  16 KiB", 54, "ok"], ["int[100]  400 B", 124, "ok"]];
          ptr.forEach(([n, y]) => G.box(24, y, 150, 40, n, { stroke: "q", color: "q" }));
          const bs = blk.map(([n, y, c]) => G.box(380, y, 236, 40, n, { fill: c }));
          G.arrow(176, 80, 376, 74, { color: "q", w: 2 }); G.arrow(176, 150, 376, 144, { color: "q", w: 2 });
          G.text(24, 232, "every block has exactly one owner who frees it", { size: 13, color: "ok" });
          G.label(24, 270, "matching pairs", { size: 12 });
          [["new", "delete"], ["new[]", "delete[]"], ["malloc", "free"]].forEach(([a, b], i) => { G.box(24 + i * 200, 282, 84, 30, a, { fill: "v", size: 12 }); G.text(112 + i * 200 + 4, 302, "↔", { size: 14, color: "muted" }); G.box(122 + i * 200 + 10, 282, 84, 30, b, { fill: "ok", size: 12 }); });
          G.box(24, 340, 590, 36, "new[] + delete   ·   malloc + delete   ·   new + free   =  bug", { stroke: "hot", color: "hot", size: 12 });
          G.from(bs, { opacity: 0, x: 20, stagger: 0.2, duration: 0.3 });
          G.caption("the pointer lives on the stack; the block lives until it is freed");
        } },

      { rail: "three bugs", title: "The three classic bugs: leak, double free, use-after-free",
        body: `<p>Every heap bug is a mistake about <i>when</i> a block is freed:</p>
<div class="eq">// 1. LEAK: freed zero times
int* p = new int[4];
p = nullptr;          // the address is lost

// 2. DOUBLE FREE: freed twice
delete[] p;
delete[] p;           // allocator corrupted

// 3. USE-AFTER-FREE: used after the free
delete[] p;
p[0] = 1;             // memory belongs to someone else</div>
<p>A <b>leak</b> is slow: memory grows until the machine runs out. A server that leaks a few kilobytes per request dies after a few million requests. A <b>double free</b> corrupts the allocator's own bookkeeping, so the crash often appears later and somewhere unrelated. A <b>use-after-free</b> reads or writes memory the allocator may already have handed to another part of the program. It can corrupt unrelated data, or become a security hole.</p>
<p>All three come from the same cause: it was not clear <i>who owns</i> the block. Steps 7 and 8 fix the cause. Step 6 gives you a tool to find these bugs. The simulator animates all three.</p>`,
        check: { q: "int* p = new int(5); int* q = p; delete p; std::cout << *q;   Which bug is this?",
          options: ["Memory leak", "Double free", "Use-after-free", "No bug: q is a separate pointer"], answer: 2,
          why: "delete p frees the block. q still holds the same address, so *q reads a freed block. The block was freed once (not a double free) and was freed (not a leak). The root cause is that two pointers shared one block with no agreement about who owns it." },
        scene(G) {
          const lanes = [["leak", 100, [["alloc", "ok"], ["use", "ok"], ["pointer lost", "hot"]], "never freed: memory grows"], ["double free", 200, [["alloc", "ok"], ["free", "ok"], ["free again", "hot"]], "allocator state corrupted"], ["use after free", 300, [["alloc", "ok"], ["free", "ok"], ["read / write", "hot"]], "touches someone else's memory"]];
          G.line(150, 40, 600, 40, { color: "muted", w: 1.5, arrow: true }); G.label(556, 32, "time", { size: 11 });
          const all = [];
          lanes.forEach(([name, y, evs, eff]) => {
            G.text(24, y + 5, name, { size: 13, weight: 700 });
            G.line(150, y, 590, y, { color: "line", w: 3 });
            evs.forEach(([lab, col], i) => {
              const x = 190 + i * 160;
              all.push(G.circle(x, y, 13, { fill: col }));
              G.text(x, y - 24, lab, { anchor: "middle", size: 12, color: col === "hot" ? "hot" : "ink" });
            });
            G.text(150, y + 40, "→ " + eff, { size: 12, color: "hot" });
          });
          G.text(24, 408, "red marks the step where the bug happens", { size: 13, color: "hot" });
          G.from(all, { opacity: 0, scale: 0, transformOrigin: "center", stagger: 0.1, duration: 0.25 });
          G.caption("three timelines, one cause: unclear ownership");
        } },

      { rail: "AddressSanitizer", title: "AddressSanitizer turns silent bugs into exact reports",
        body: `<p>These bugs often print nothing. The program may pass its tests. <b>AddressSanitizer</b> (ASan) is a compiler option that checks every memory access, and puts poisoned "red zones" around each heap block, so one-byte overruns are caught. It also notices frees of freed blocks, and (on Linux) reports leaks when the program exits.</p>
<div class="eq">g++ -std=c++20 -g -O0 \\
    -fsanitize=address,undefined \\
    -fno-omit-frame-pointer main.cpp -o app
./app</div>
<p><code>-g</code> adds line numbers to the report. <code>undefined</code> adds UBSan, which flags signed overflow, bad shifts and similar. ASan slows the program down (the Clang docs say about 2x), so use it for tests, never for benchmarks. Here is a report shaped like a real one, for a loop that wrote one element too many:</p>
<div class="eq">==4242==ERROR: AddressSanitizer: heap-buffer-overflow
WRITE of size 4 at 0x502000000020 thread T0
    #0 make_ramp(unsigned long) impl.hpp:15
    #1 ramp_of_four test.cpp:5
0x502000000020 is located 0 bytes after
  16-byte region [0x502000000010,0x502000000020)
allocated by thread T0 here:
    #0 operator new[](unsigned long)
    #1 make_ramp(unsigned long) impl.hpp:14</div>
<p>Read it top to bottom: the <b>kind</b> of error, whether it was a read or a write, the first frame in <i>your</i> code (the line to fix), how far from the block the access was, and who allocated the block. Exercise 4 gives you a real one.</p>`,
        check: { q: "ASan says: \"located 8 bytes after 16-byte region\" for a buffer of 4 ints. Which index was accessed?",
          options: ["4", "5", "6", "8"], answer: 2,
          why: "16 bytes is 4 ints (indexes 0 to 3). Index 4 starts exactly at the end of the block: 0 bytes after. 8 bytes after the end is 2 ints further, so index 6. (Real Address numbers vary; the 'bytes after' number is what you convert.)" },
        scene(G) {
          const lines = [
            "==4242==ERROR: AddressSanitizer: heap-buffer-overflow",
            "WRITE of size 4 at 0x502000000020 thread T0",
            "    #0 make_ramp(unsigned long) impl.hpp:15",
            "    #1 ramp_of_four test.cpp:5",
            "0x502000000020 is located 0 bytes after",
            "  16-byte region [0x502000000010,0x502000000020)",
            "allocated by thread T0 here:",
            "    #1 make_ramp(unsigned long) impl.hpp:14",
          ];
          G.rect(16, 36, 608, 190, { fill: "line", opacity: 0.25, rx: 8 });
          const L = lines.map((t, i) => G.text(26, 62 + i * 22, t, { size: 11.5, color: i === 0 ? "hot" : "ink" }));
          const items = [["1", "kind of bug", 0], ["2", "read or write, size", 1], ["3", "your line: fix here", 2], ["4", "distance: off by one", 4], ["5", "who allocated it", 7]];
          items.forEach(([n, t, i]) => { G.circle(448, 58 + i * 22, 9, { fill: "v" }); G.text(448, 62 + i * 22, n, { anchor: "middle", size: 11, color: "bg", weight: 700 }); G.text(464, 62 + i * 22, t, { size: 11, color: "v" }); });
          G.label(24, 260, "the buffer the report describes (4 ints = 16 bytes)", { size: 12 });
          const cells = [0, 1, 2, 3].map((i) => G.box(24 + i * 74, 274, 68, 40, `[${i}]`, { fill: "k", size: 13 }));
          const bad = G.box(24 + 4 * 74, 274, 68, 40, "[4]", { stroke: "hot", color: "hot", dash: "4 3", size: 13 });
          G.text(24 + 4 * 74 + 34, 340, "0 bytes after", { anchor: "middle", size: 12, color: "hot" });
          G.text(24 + 6 * 74 + 34, 340, "8 bytes after = [6]", { anchor: "middle", size: 12, color: "muted" });
          G.text(24, 392, "g++ -g -fsanitize=address,undefined ...", { size: 13, color: "ok" });
          G.from(L, { opacity: 0, x: -8, stagger: 0.07, duration: 0.2 }); G.pulse(bad, { repeat: 5 });
          G.caption("read a report in order: kind, access, your line, distance, allocation");
        } },

      { rail: "ownership · RAII", title: "One owner frees; RAII ties the free to a destructor",
        body: `<p>The cure is a rule: <b>every heap block has exactly one owner</b>, and the owner frees it. Other pointers may <i>use</i> the block, but they never free it, and they must not outlive the owner.</p>
<p>Writing <code>delete</code> by hand is fragile, because a function can be left in several ways: an early <code>return</code>, an exception, or the normal end. You must remember the <code>delete</code> on every path.</p>
<p>C++ has a better tool: the destructor (R.1, step 8) runs at the closing brace on <i>every</i> path. So make an object own the block, allocate in its constructor and free in its destructor. This is <b>RAII</b>, "resource acquisition is initialisation":</p>
<div class="eq">class FloatBuffer {
  float* p_;
 public:
  explicit FloatBuffer(size_t n) : p_(new float[n]) {}
  ~FloatBuffer() { delete[] p_; }
};

void work() {
  FloatBuffer b(1024);
  if (bad()) return;     // ~FloatBuffer runs here
  risky();               // ...and if this throws
}                        // ...and here</div>
<p>The free can no longer be forgotten, because it is not a separate line to write. Exercise 3 builds this class.</p>`,
        scene(G) {
          const draw = (y0, title, tcol, fixed) => {
            G.text(24, y0, title, { size: 13, weight: 700, color: tcol });
            G.box(24, y0 + 22, 130, 84, "", { stroke: "ink" });
            G.text(36, y0 + 52, "p = new ...", { size: 12 }); G.text(36, y0 + 78, "work()", { size: 12 });
            [["early return", 0], ["exception", 1], ["normal end", 2]].forEach(([n, i]) => {
              const y = y0 + 20 + i * 32;
              G.arrow(156, y0 + 64, 250, y + 12, { color: "muted", w: 1.5 });
              G.box(252, y, 120, 26, n, { stroke: "muted", size: 12 });
              const frees = fixed || i === 2;
              G.arrow(374, y + 13, 424, y + 13, { color: frees ? "ok" : "hot", w: 2 });
              G.box(426, y, 190, 26, frees ? (fixed ? "~FloatBuffer frees" : "delete[] p") : "skipped: LEAK", { fill: frees ? "ok" : "hot", size: 12 });
            });
          };
          draw(44, "manual delete at the end of the function", "hot", false);
          draw(212, "RAII: the destructor runs on every exit", "ok", true);
          G.text(24, 392, "3 exits, 1 delete  vs  3 exits, 3 destructor calls", { size: 13 });
          G.caption("a destructor cannot be skipped by an early return");
        } },

      { rail: "smart pointers", title: "unique_ptr is the sole owner; shared_ptr counts owners",
        body: `<p>You rarely write your own RAII class. The standard library has ready-made owners, in <code>&lt;memory&gt;</code>.</p>
<p><b><code>std::unique_ptr&lt;T&gt;</code></b> is the sole owner of one block. It is the size of one pointer (8 bytes), adds no counting, and frees the block when it dies. It cannot be copied (two owners would double-free), only <b>moved</b>:</p>
<div class="eq">auto a = std::make_unique&lt;float[]&gt;(1024);
a[0] = 1.0f;
auto b = std::move(a);   // b owns it now, a is null
// auto c = b;           // compile error: copy deleted</div>
<p><code>std::make_unique</code> allocates and wraps in one step. This is the default choice.</p>
<p><b><code>std::shared_ptr&lt;T&gt;</code></b> allows several owners. A hidden <b>reference count</b> says how many; the block is freed when it reaches 0. It is typically 16 bytes (a pointer to the block and a pointer to the counter). Copying one updates a shared counter with an atomic operation. That is fine now and then, but costly in a tight loop or across threads, and it makes it hard to say when anything is freed. Inference engines keep it off hot paths: prefer <code>unique_ptr</code>, and share only when lifetimes truly overlap, such as a cached prompt prefix used by several requests.</p>
<p>A function that just <i>uses</i> a block takes a reference or a plain pointer, not a smart pointer. A raw pointer then means "borrowing", never "owning".</p>`,
        check: { q: "auto a = std::make_unique<int>(1); auto b = a;   What happens?",
          options: ["b and a both own the int; it is freed twice", "A compile error: unique_ptr cannot be copied", "b gets a copy of the int; a keeps the original"], answer: 1,
          why: "unique_ptr's copy constructor is deleted, so the compiler rejects the line. That is the point: it makes two owners impossible. To transfer ownership you write auto b = std::move(a), which leaves a null." },
        scene(G) {
          G.text(24, 44, "unique_ptr: move transfers", { size: 13, weight: 700 });
          const a0 = G.box(24, 70, 96, 36, "a", { stroke: "q", color: "q" }), b0 = G.box(24, 130, 96, 36, "b", { stroke: "q", color: "q" });
          G.rect(190, 90, 110, 44, { fill: "v", rx: 6, opacity: 0.9 }); G.text(245, 117, "float[1024]", { anchor: "middle", size: 12, color: "bg", weight: 700 });
          G.arrow(122, 148, 186, 118, { color: "ok", w: 2.5 });
          G.line(122, 88, 150, 88, { color: "hot", w: 2, dash: "4 3" }); G.text(154, 92, "null", { size: 12, color: "hot" });
          G.label(24, 196, "b = std::move(a)", { size: 12, color: "ink" }); G.label(24, 214, "one owner, always", { size: 12 });
          G.text(340, 44, "shared_ptr: counted owners", { size: 13, weight: 700 });
          G.box(340, 70, 90, 36, "s1", { stroke: "q", color: "q" }); G.box(340, 130, 90, 36, "s2", { stroke: "q", color: "q" });
          G.rect(520, 76, 96, 44, { fill: "v", rx: 6, opacity: 0.9 }); G.text(568, 103, "block", { anchor: "middle", size: 12, color: "bg", weight: 700 });
          const cb = G.box(520, 140, 96, 36, "refs = 2", { stroke: "ok", color: "ok", size: 13 });
          G.arrow(432, 88, 516, 96, { color: "q", w: 2 }); G.arrow(432, 148, 516, 108, { color: "q", w: 2 });
          G.label(340, 196, "freed when refs reaches 0", { size: 12 });
          G.text(24, 270, "unique_ptr:  8 bytes, no counter, free at scope end", { size: 13, color: "ok" });
          G.text(24, 296, "shared_ptr: ~16 bytes, atomic count on every copy", { size: 13, color: "v" });
          G.text(24, 340, "default: unique_ptr + make_unique", { size: 14, weight: 700 });
          G.text(24, 366, "borrowing: pass T& or T*, never free it", { size: 13 });
          G.from(cb, { opacity: 0, delay: 0.4, duration: 0.4 });
          G.caption("sole owner by default; shared only when lifetimes truly overlap");
        } },

      { rail: "copy vs move", title: "Copying a vector copies its buffer; moving hands over the pointer",
        body: `<p>A <code>std::vector</code> is a small header (pointer, size, capacity) that <i>owns</i> a heap buffer. That decides what copying and moving mean.</p>
<div class="eq">std::vector&lt;float&gt; a(1000000);   // 4,000,000 B buffer

auto b = a;              // COPY: allocate a new 4 MB
                         // buffer and copy every float
auto c = std::move(a);   // MOVE: c takes a's pointer
                         // (24-byte header), a is empty</div>
<p>A <b>copy</b> makes a second owner of a second block: a fresh allocation plus 1,000,000 × 4 = 4,000,000 bytes copied. A <b>move</b> gives the one block to a new owner: three words change hands and the old object is left empty (null pointer, size 0). No allocation, no element copied. After <code>std::move(a)</code> do not use <code>a</code> except to assign to it or destroy it.</p>
<p>That is why returning a vector from a function is cheap (it is moved, or built in place), why <code>unique_ptr</code> can be moved but not copied, and why <code>FloatBuffer</code> in exercise 3 deletes copying and implements moving. P0.1 measures the difference on a 64 MB buffer and counts copies and moves in code. This is the bridge to its move section.</p>`,
        check: { q: "v holds 1,000,000 floats. You call consume(std::move(v)) where consume takes a std::vector<float> by value. How many bytes of float data are copied?",
          options: ["4,000,000", "24", "0"], answer: 2,
          why: "The by-value parameter is move-constructed from v: it takes over v's heap buffer. Only the 24-byte header changes hands. No float is copied. Without std::move the same call would copy 4,000,000 bytes." },
        scene(G) {
          G.text(24, 40, "copy:  auto b = a;", { size: 13, weight: 700, color: "hot" });
          const hdr = (y, lab, col) => G.box(24, y, 150, 34, lab, { stroke: col, color: col, size: 12 });
          hdr(54, "a: ptr, size, cap", "q"); hdr(104, "b: ptr, size, cap", "q");
          G.rect(330, 52, 280, 36, { fill: "v", rx: 6, opacity: 0.9 }); G.text(470, 75, "block A  4,000,000 B", { anchor: "middle", size: 12, color: "bg", weight: 700 });
          const nb = G.rect(330, 102, 280, 36, { fill: "hot", rx: 6, opacity: 0.9 }); G.text(470, 125, "block B  4,000,000 B  (new copy)", { anchor: "middle", size: 12, color: "bg", weight: 700 });
          G.arrow(176, 71, 326, 70, { color: "q", w: 2 }); G.arrow(176, 121, 326, 120, { color: "q", w: 2 });
          G.text(24, 176, "1 allocation + 4,000,000 bytes copied", { size: 13, color: "hot" });
          G.text(24, 236, "move:  auto c = std::move(a);", { size: 13, weight: 700, color: "ok" });
          G.box(24, 250, 150, 34, "a: null, size 0", { stroke: "muted", color: "muted", size: 12 }); G.box(24, 300, 150, 34, "c: ptr, size, cap", { stroke: "q", color: "q", size: 12 });
          G.rect(330, 290, 280, 36, { fill: "v", rx: 6, opacity: 0.9 }); G.text(470, 313, "block A  4,000,000 B  (same)", { anchor: "middle", size: 12, color: "bg", weight: 700 });
          const mv = G.arrow(176, 317, 326, 310, { color: "ok", w: 2.5 });
          G.text(24, 372, "0 allocations, 0 floats copied, 24 bytes moved", { size: 13, color: "ok" });
          G.from(nb, { scaleX: 0, transformOrigin: "left center", duration: 0.8 }); G.from(mv, { opacity: 0, delay: 0.4, duration: 0.4 });
          G.caption("a move changes who owns the block; the block itself stays put");
        } },

      { rail: "alignment", title: "Alignment, briefly: addresses that are multiples",
        body: `<p>A type's <b>alignment</b> is the number its address must be a multiple of. <code>alignof(double)</code> is 8: a <code>double</code> lives at an address divisible by 8.</p>
<p>The CPU reads memory in 64-byte cache lines (P0.1 covers why). A value that straddles two lines costs two reads, so you sometimes ask for stricter alignment yourself: <code>alignas(64) struct Slot { ... };</code> starts every <code>Slot</code> on a line boundary. Since C++17, <code>new</code> honours <code>alignas</code>.</p>
<p>That is all you need for now. P0.1 section 2 has the full treatment: padding, field order and why struct sizes are not the sum of their fields.</p>
<p><b>Where this leads.</b> You can now say where a variable lives, when it dies and who frees it. P0.1 builds a move-only tensor on exactly this, and P0.2 explains what the addresses really mean.</p>`,
        scene(G) {
          const px = 4.5, x0 = 24;
          const lines = (y) => { G.rect(x0, y, 64 * px - 2, 40, { stroke: "muted", rx: 4 }); G.rect(x0 + 64 * px, y, 64 * px - 2, 40, { stroke: "muted", rx: 4 }); G.label(x0 + 4, y - 8, "cache line 0", { size: 11 }); G.label(x0 + 64 * px + 4, y - 8, "cache line 1", { size: 11 }); };
          lines(70); lines(190);
          const a = G.rect(x0, 74, 64 * px - 6, 32, { fill: "ok", rx: 3, opacity: 0.9 });
          G.text(x0 + 32 * px, 95, "alignas(64) Slot: 64 B", { anchor: "middle", size: 12, color: "bg", weight: 700 });
          G.text(x0, 134, "starts at a multiple of 64: touches 1 line", { size: 12, color: "ok" });
          const b = G.rect(x0 + 32 * px, 194, 64 * px - 6, 32, { fill: "hot", rx: 3, opacity: 0.9 });
          G.text(x0 + 64 * px, 215, "same 64 B, address % 64 = 32", { anchor: "middle", size: 12, color: "bg", weight: 700 });
          G.text(x0, 254, "starts mid-line: touches 2 lines", { size: 12, color: "hot" });
          G.text(24, 316, "alignof(double) == 8  ·  address % 8 == 0", { size: 13 });
          G.text(24, 342, "alignas(64) struct Slot { ... };", { size: 13 });
          G.label(24, 380, "full treatment: P0.1 section 2", { size: 12 });
          G.from([a, b], { opacity: 0, scaleX: 0.3, transformOrigin: "left center", stagger: 0.3, duration: 0.4 });
          G.caption("alignment decides how many cache lines a value touches");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A process has code, globals, a stack and a heap. The stack frees itself on return; the heap is freed by you.</li>
<li>A pointer to a local dangles once its function returns. Return values, not addresses of locals.</li>
<li><code>new</code> pairs with <code>delete</code>, <code>new[]</code> with <code>delete[]</code>, <code>malloc</code> with <code>free</code>. A block freed zero times is a leak, twice a double free, and touched after the free a use-after-free.</li>
<li><code>-fsanitize=address,undefined -g</code> turns those silent bugs into reports. Read the kind, the access, your line and the distance.</li>
<li>Exactly one owner frees. RAII ties the free to a destructor, so no exit path can skip it. <code>std::unique_ptr</code> and <code>std::make_unique</code> are the default; <code>shared_ptr</code> counts owners and costs more.</li>
<li>Copying a vector copies its buffer; moving hands over the pointer, with no allocation and no element copied.</li></ul>
<h3>Go further</h3><ul>
<li><a href="https://www.learncpp.com/">learncpp.com</a>: the chapters on dynamic memory, smart pointers and move semantics.</li>
<li><a href="https://cs50.harvard.edu/x/">CS50x</a>, Week 4 "Memory": stack, heap and <code>malloc</code> with debugging tools.</li>
<li><a href="https://clang.llvm.org/docs/AddressSanitizer.html">Clang AddressSanitizer docs</a>: the full list of what it detects and its flags.</li>
<li><a href="https://en.cppreference.com/w/">cppreference</a>: <code>unique_ptr</code>, <code>shared_ptr</code>, <code>make_unique</code> and <code>std::move</code> exactly.</li></ul>
<p>Next, R.3 covers the terminal, git and the build tools you will use to run all of this. These links are UNVERIFIED until checked from a networked machine.</p>`,
    sim: {
      title: "Stack and heap timeline",
      intro: "Pick a small program and step through it. The left column is the code, the middle is the stack, the right is the heap. The simulator flags the bug on the line where it happens. Drag the step slider, or press Run to play the program.",
      height: 318,
      controls: [
        { id: "prog", label: "program", type: "select", value: "uaf", options: [["correct", "correct: allocate, use, free"], ["leak", "leak: pointer overwritten"], ["uaf", "use-after-free: second pointer"]] },
        { id: "step", label: "step", min: 0, max: 7, step: 1, value: 4 },
      ],
      run: { label: "Run the program", frames: 40, ms: 70 },
      draw(G, v, t) {
        const P = PROGRAMS[v.prog], n = P.steps.length;
        const idx = t < 1 ? Math.min(n - 1, Math.floor(t * n)) : Math.min(n - 1, v.step);
        const S = P.steps[idx];
        // code
        G.label(10, 16, "code", { size: 11 });
        P.code.forEach((ln, i) => {
          if (i === S.line) G.rect(6, 22 + i * 22, 232, 20, { fill: S.flag ? "hot" : "q", opacity: 0.3, rx: 3 });
          G.text(12, 37 + i * 22, ln, { size: 11.5 });
        });
        // stack
        G.label(262, 16, "stack", { size: 11 });
        const slotY = [];
        let y = 286;
        const varPos = {};
        S.frames.forEach(([name, vars]) => {
          const h = 24 + vars.length * 18;
          y -= h;
          G.rect(256, y, 150, h - 4, { fill: "k", opacity: 0.22, stroke: "k", rx: 5 });
          G.text(264, y + 16, name, { size: 12, weight: 700 });
          vars.forEach(([vn, val], j) => {
            const vy = y + 34 + j * 18;
            const isPtr = /^B\d/.test(val);
            const dangling = (S.dangling || []).includes(vn);
            const txt = isPtr ? `${vn} → ${val}` : val === "null" ? `${vn} = nullptr` : `${vn} = ${val}`;
            G.text(264, vy, txt, { size: 11.5, color: dangling ? "hot" : "ink" });
            if (isPtr) varPos[vn] = [406, vy - 4, val, dangling];
          });
          y -= 6;
        });
        // heap
        G.label(440, 16, "heap", { size: 11 });
        const blockY = {};
        S.heap.forEach(([id, lab, st], i) => {
          const by = 24 + i * 46; blockY[id] = by + 17;
          const col = STATE_COL[st];
          if (st === "live") G.rect(440, by, 186, 34, { fill: "ok", opacity: 0.85, rx: 5 });
          else G.rect(440, by, 186, 34, { stroke: col, dash: "5 4", rx: 5, sw: 2 });
          G.text(450, by + 21, `${id}  ${lab}`, { size: 11.5, color: st === "live" ? "bg" : col, weight: 700 });
          if (st !== "live") G.text(618, by + 21, st === "freed" ? "freed" : "LEAKED", { anchor: "end", size: 10, color: col });
        });
        Object.values(varPos).forEach(([x, vy, id, dang]) => {
          if (blockY[id] === undefined) return;
          G.arrow(x, vy, 436, blockY[id], { color: dang ? "hot" : "q", w: 1.8, dash: dang ? "4 3" : undefined });
        });
        // banner
        const bug = S.flag, good = S.ok;
        if (bug) G.text(10, 306, "⚠ " + bug, { size: 12, color: "hot", weight: 700 });
        else if (good) G.text(10, 306, "✓ " + good, { size: 12, color: "ok", weight: 700 });
        else G.text(10, 306, `step ${idx + 1} of ${n}`, { size: 12, color: "muted" });
        // counters
        const liveBytes = S.heap.filter((h) => h[2] === "live").reduce((a, h) => a + +h[1].match(/(\d+) B$/)[1], 0);
        const leakBytes = S.heap.filter((h) => h[2] === "leak").reduce((a, h) => a + +h[1].match(/(\d+) B$/)[1], 0);
        const ok = !bug;
        return [
          { title: "This step", html: `<p class="note"><b>Line:</b> <code>${P.code[S.line].trim().replace(/</g, "&lt;")}</code></p><p class="note">${S.note}</p>`, chip: [ok, ok ? "no bug yet" : "bug flagged: see the red message"] },
          { title: "Counters", rows: [["stack frames", String(S.frames.length)], ["live heap bytes", `${liveBytes} B`], ["leaked bytes", `${leakBytes} B`], ["freed blocks", String(S.heap.filter((h) => h[2] === "freed").length)]] },
          { title: "Model", html: `<p class="note">Addresses and exact frame sizes are left out on purpose. The point is the order of events and who owns each block. A real run under AddressSanitizer reports the flagged lines with a stack trace.</p>` },
        ];
      },
    },
    practice: {
      intro: `Four exercises, each tested with CTest and built with AddressSanitizer and UBSan in a Debug configuration, so memory bugs fail loudly. Run from the repository root. Add <code>-DS2S_USE_SOLUTIONS=ON</code> to the configure step to build the reference solutions first and confirm your toolchain works. The starters compile; the tests fail until you fix them.`,
      items: [
        { title: "Fix the leak", tier: "T0 · easy", goal: "An early-return path skips a delete[] and leaks. Replace the raw array with std::unique_ptr.",
          cmd: "cmake -S course/R-readiness/R.2-memory-management/exercises -B /tmp/r2-ex && cmake --build /tmp/r2-ex -j && ctest --test-dir /tmp/r2-ex -R 01-fix-the-leak --output-on-failure" },
        { title: "Dangling return", tier: "T0 · easy", goal: "A function returns a pointer to a local array. Change the API to return the values by value (a std::vector).",
          cmd: "ctest --test-dir /tmp/r2-ex -R 02-dangling-return --output-on-failure" },
        { title: "An owning FloatBuffer", tier: "T0 · medium", goal: "Constructor allocates, destructor frees, copy deleted, move implemented. The test counts live allocations through a static counter.",
          cmd: "ctest --test-dir /tmp/r2-ex -R 03-owning-buffer --output-on-failure" },
        { title: "Read a sanitizer report", tier: "T0 · easy", goal: "A planted heap-buffer-overflow. Paste ASan's first line into a comment in impl.hpp, then fix the loop bound.",
          cmd: "ctest --test-dir /tmp/r2-ex -R 04-sanitizer-read --output-on-failure" },
      ],
      labs: [
        { label: "Self-check questions", path: "course/R-readiness/R.2-memory-management/quiz.md" },
        { label: "Module README with the exercise table and further reading", path: "course/R-readiness/R.2-memory-management/README.md" },
        { label: "Next: P0.1 uses these ideas for strides, alignment and a move-only Tensor", path: "course/P0-systems-primer/P0.1-c-cpp-for-systems/README.md" },
      ],
    },
  });
})();
