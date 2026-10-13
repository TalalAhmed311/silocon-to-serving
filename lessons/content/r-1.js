/* R.1 — C and C++ from zero. Everything P0.1 silently assumes: compiling, types, pointers, references, headers, classes, UB. */
(function () {
  const HEX = (v) => "0x" + v.toString(16);
  // a run of n byte cells starting at (x, y); cw px per byte
  const byteRow = (G, x, y, n, cw, fill, h = 28) => {
    const out = [];
    for (let i = 0; i < n; i++) out.push(G.rect(x + i * cw, y, cw - 2, h, { fill, rx: 3, opacity: 0.9 }));
    return out;
  };
  const TYPE_SIZE = { char: 1, int: 4, float: 4, double: 8, vec3: 12 };
  const TYPE_COL = { char: "q", int: "k", float: "k", double: "v", vec3: "ok" };

  S2S.lesson({
    id: "r-1", n: "R.1", title: "C and C++ from zero",
    subtitle: "Readiness · first principles · T0, any laptop",
    kicker: "Lesson · ≈ 30 min reading + 4 h hands-on",
    headline: "What does the compiler actually do with this?",
    intro: `<p><b>Already comfortable with pointers, references, <code>const</code>, headers and destructors? Answer the checkpoints. If you get them all right, skip to R.2.</b></p>
<p>You can write a loop and a function in Python or JavaScript. This lesson gives you everything P0.1 quietly assumes about C and C++: how source code becomes a program, how big things are, what a pointer is, how arguments are passed, how a project splits into files, and what a class does when its scope ends. Each step is one idea.</p>
<p>The dark panel redraws for each step. Toolchain: <code>g++</code> or <code>clang++</code> 11 or newer, and CMake 3.20 or newer.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "4 exercises"],
    legend: [["k", "int / float"], ["v", "double"], ["q", "pointer / char"], ["ok", "ok"], ["hot", "problem"]],
    next: "r-2",
    steps: [
      { rail: "build", title: "C++ needs a build step: compile each file, then link",
        body: `<p>Python reads your source and runs it line by line. C++ does not. A <b>compiler</b> first translates each source file into machine code for your CPU. A second tool, the <b>linker</b>, joins the pieces into one program you can run.</p>
<div class="eq">g++ -c main.cpp    -o main.o      # compile
g++ -c mathlib.cpp -o mathlib.o   # compile
g++ main.o mathlib.o -o app       # link
./app</div>
<p>An <b>object file</b> (<code>.o</code>) holds machine code for one source file. It is not runnable yet. If <code>main.cpp</code> calls <code>square()</code>, then <code>main.o</code> only says "I need a function called <code>square</code>; address unknown". <code>mathlib.o</code> says "I define <code>square</code>". The linker matches the two and fills in the real address.</p>
<p>Two practical results. Compiling one file at a time means that after an edit you rebuild only that file (R.3 shows this). And there are two families of error: <b>compile errors</b> (the compiler rejects one file) and <b>link errors</b> such as <code>undefined reference to 'square'</code> (a promised function never turned up).</p>`,
        check: { q: "main.cpp calls square(), which is defined in mathlib.cpp. You run: g++ main.cpp -o app. What happens?",
          options: ["It runs: g++ finds mathlib.cpp on its own", "A link error: undefined reference to square", "A compile error in main.cpp, because square is unknown"], answer: 1,
          why: "If main.cpp includes a header that declares square, compiling main.cpp succeeds. The call is left as a placeholder. The linker then finds no definition, because mathlib.cpp was never given to g++. Run g++ main.cpp mathlib.cpp -o app instead." },
        scene(G) {
          const src = [G.box(24, 70, 130, 44, "main.cpp", { stroke: "ink" }), G.box(24, 200, 130, 44, "mathlib.cpp", { stroke: "ink" })];
          const obj = [G.box(230, 70, 120, 44, "main.o", { fill: "k" }), G.box(230, 200, 120, 44, "mathlib.o", { fill: "k" })];
          G.arrow(156, 92, 226, 92, { color: "muted", w: 2 }); G.arrow(156, 222, 226, 222, { color: "muted", w: 2 });
          G.label(160, 82, "g++ -c", { size: 11 }); G.label(160, 212, "g++ -c", { size: 11 });
          const lk = G.box(400, 130, 84, 44, "linker", { fill: "q" });
          G.arrow(352, 96, 398, 146, { color: "muted", w: 2 }); G.arrow(352, 218, 398, 160, { color: "muted", w: 2 });
          const app = G.box(520, 130, 96, 44, "app", { fill: "ok" });
          G.arrow(486, 152, 516, 152, { color: "muted", w: 2 });
          G.box(396, 220, 92, 30, "libstdc++", { stroke: "muted", color: "muted", size: 11 });
          G.arrow(442, 218, 442, 178, { color: "muted", dash: "4 3" });
          G.text(24, 140, "compile", { size: 12, color: "muted" }); G.text(24, 158, "one file at a time", { size: 12, color: "muted" });
          G.text(24, 300, "main.o:     calls square()   address ?", { size: 13, color: "hot" });
          G.text(24, 324, "mathlib.o:  defines square() at offset 0x40", { size: 13, color: "k" });
          G.text(24, 348, "linker:     patches the call with the real address", { size: 13, color: "ok" });
          G.text(24, 388, "no definition found  →  undefined reference", { size: 13, color: "hot" });
          G.from([...src, ...obj], { opacity: 0, x: -14, stagger: 0.12, duration: 0.3 }); G.from([lk, app], { opacity: 0, delay: 0.6, stagger: 0.2, duration: 0.3 });
          G.caption("compile each file to an object, then link the objects into one program");
        } },

      { rail: "sizes", title: "Every type has a fixed size in bytes",
        body: `<p>In Python an integer can grow as big as you like. In C++ every type has a <b>fixed size</b> chosen at compile time. Memory is a line of numbered bytes (P0.1 builds on this), so the size says how many bytes a value occupies.</p>
<table><tr><th>type</th><th>bytes</th><th>holds</th></tr>
<tr><td><code>char</code></td><td>1</td><td>one byte, often a letter</td></tr>
<tr><td><code>int</code></td><td>4</td><td>whole number, about ±2.1 billion</td></tr>
<tr><td><code>float</code></td><td>4</td><td>decimal number, about 7 digits</td></tr>
<tr><td><code>double</code></td><td>8</td><td>decimal number, about 15 digits</td></tr>
<tr><td>any pointer</td><td>8</td><td>an address (64-bit machine)</td></tr></table>
<p>These are the sizes on 64-bit Linux, macOS and Windows. Ask the compiler instead of trusting a table: <code>sizeof(int)</code> is evaluated at compile time. An array is its elements back to back, so <code>int a[10]</code> takes <code>10 × 4 = 40</code> bytes. A model with 8 billion <code>float</code> weights needs <code>8e9 × 4 = 32 GB</code> for them. Sizes turn into memory bills.</p>`,
        scene(G) {
          const rows = [["char", 1, "q"], ["int", 4, "k"], ["float", 4, "k"], ["double", 8, "v"], ["int*  (pointer)", 8, "q"]];
          const all = [];
          rows.forEach(([n, b, col], r) => {
            const y = 50 + r * 54;
            G.text(24, y + 19, n, { size: 13 });
            all.push(...byteRow(G, 168, y, b, 22, col, 28));
            G.text(168 + b * 22 + 8, y + 19, `${b} B`, { size: 13, color: "muted" });
          });
          G.text(24, 340, "sizeof(int[10])        = 10 × 4 = 40 bytes", { size: 13 });
          G.text(24, 366, "sizeof(struct{float x,y;}) = 4 + 4 = 8 bytes", { size: 13 });
          G.label(24, 396, "one cell = one byte · sizes shown are for a 64-bit machine", { size: 12 });
          G.from(all, { opacity: 0, scaleX: 0.2, transformOrigin: "left center", stagger: 0.015, duration: 0.2 });
          G.caption("a type is a promise about how many bytes a value takes");
        } },

      { rail: "addresses", title: "A variable lives at an address; a pointer stores one",
        body: `<p>Each variable occupies bytes, and the first byte has an <b>address</b>: a number. The operator <code>&amp;x</code> means "the address of <code>x</code>". A <b>pointer</b> is a variable whose value is an address. Its type says what lives there.</p>
<div class="eq">int x = 7;
int* p = &amp;x;      // p holds the address of x
*p = 9;           // follow the arrow: x is now 9
int y = *p;       // read through it: y is 9</div>
<p><code>*p</code> means "the thing at the address in <code>p</code>". The star appears twice with two jobs: in <code>int* p</code> it is part of the <i>type</i> ("pointer to int"), and in <code>*p = 9</code> it <i>follows</i> the pointer. The pointer <code>p</code> is itself a variable: it takes 8 bytes and has its own address.</p>
<p>A pointer that points at nothing holds <code>nullptr</code>. Following it crashes. Always know whether a pointer can be null before you use <code>*</code>.</p>
<div class="analogy"><b>Picture it</b>A house number is not the house. <code>x</code> is the house; <code>p</code> is a piece of paper with its number written on it.</div>`,
        scene(G) {
          G.label(24, 40, "memory near 0x1000   (addresses are illustrative)", { size: 12 });
          byteRow(G, 100, 62, 12, 22, "line", 34);
          const x = G.rect(100 + 4 * 22, 62, 4 * 22 - 2, 34, { fill: "k", rx: 3 });
          G.text(100 + 6 * 22, 85, "x = 7", { anchor: "middle", size: 13, color: "bg", weight: 700 });
          G.label(204, 112, "0x1004", { size: 11 });
          G.label(24, 200, "memory near 0x2000", { size: 12 });
          byteRow(G, 100, 222, 12, 22, "line", 34);
          const p = G.rect(100 + 2 * 22, 222, 8 * 22 - 2, 34, { fill: "q", rx: 3 });
          G.text(100 + 6 * 22, 245, "p = 0x1004", { anchor: "middle", size: 13, color: "bg", weight: 700 });
          G.label(100 + 2 * 22, 272, "0x2002   (8 bytes)", { size: 11 });
          const ar = G.path("M 232 220 C 232 170, 188 150, 188 100", { color: "q", w: 2.5, arrow: true });
          G.text(24, 330, "int* p = &x;    p holds the address of x", { size: 13 });
          G.text(24, 354, "*p = 9;         the arrow is followed: x = 9", { size: 13, color: "ok" });
          G.text(24, 378, "p = nullptr;    points at nothing: never use *p", { size: 13, color: "hot" });
          G.from([p, ar], { opacity: 0, duration: 0.5, delay: 0.2 }); G.pulse(x, { repeat: 4 });
          G.caption("a pointer is a number that says where the data is");
        } },

      { rail: "arrays", title: "Arrays are contiguous, and p + i steps by whole elements",
        body: `<p>An array is its elements one after another, with no gaps. If <code>int a[5]</code> starts at address <code>0x1000</code>, then <code>a[0]</code> is at <code>0x1000</code>, <code>a[1]</code> at <code>0x1004</code>, and so on, because an <code>int</code> is 4 bytes.</p>
<p>The name <code>a</code> converts to a pointer to its first element. Adding to a pointer is <b>pointer arithmetic</b>, and it counts in elements, not bytes:</p>
<div class="eq">p + i  =  address in p + i × sizeof(element)

int a[5] at 0x1000:
  a + 3 = 0x1000 + 3 × 4 = 0x100c
  a[i]  is defined as  *(a + i)</div>
<p>There is no hidden table. Indexing is just this arithmetic. That is why C++ does not check bounds: <code>a[7]</code> computes <code>0x1000 + 28</code> and reads whatever is there. The simulator lets you try other types and indexes.</p>`,
        check: { q: "double* p points at address 0x2000. What is the address of p + 3?",
          options: ["0x2003", "0x2018", "0x2030", "0x2008"], answer: 1,
          why: "A double is 8 bytes, so p + 3 moves 3 × 8 = 24 bytes. 24 is 0x18 in hex, so the address is 0x2018. Adding 3 to the address as a plain number (0x2003) would land in the middle of the first double." },
        scene(G) {
          const vals = [10, 20, 30, 40, 50];
          const cells = vals.map((v, i) => G.box(40 + i * 100, 110, 94, 50, String(v), { fill: i === 3 ? "ok" : "k", size: 15 }));
          vals.forEach((v, i) => {
            G.text(87 + i * 100, 98, `a[${i}]`, { anchor: "middle", size: 13 });
            G.label(87 + i * 100, 184, HEX(0x1000 + 4 * i), { anchor: "middle", size: 12 });
            G.label(87 + i * 100, 202, `+${4 * i}`, { anchor: "middle", size: 12, color: "ink" });
          });
          G.label(24, 230, "byte offset from the start", { size: 11 });
          const mk = G.box(325, 250, 124, 32, "a + 3", { stroke: "ok", color: "ok", size: 14 });
          const ar = G.arrow(387, 248, 387, 212, { color: "ok", w: 2.5 });
          G.text(24, 330, "a + 3  =  0x1000 + 3 × 4  =  0x100c", { size: 14 });
          G.text(24, 358, "*(a + 3)  is  a[3]  =  40", { size: 14, color: "ok" });
          G.text(24, 386, "a + 7  =  0x101c: past the end, not yours", { size: 13, color: "hot" });
          G.from(cells, { opacity: 0, y: 12, stagger: 0.08, duration: 0.25 }); G.from([mk, ar], { opacity: 0, delay: 0.5, duration: 0.3 });
          G.caption("pointer arithmetic counts elements; the compiler multiplies by the size");
        } },

      { rail: "references", title: "Three ways to pass an argument: value, reference, pointer",
        body: `<p>When you call a function, what does it receive? In Python every argument is a reference to an object. In C++ you choose.</p>
<div class="eq">void f(int  x) { x += 1; }    // value
void g(int&amp; x) { x += 1; }    // reference
void h(int* x) { *x += 1; }   // pointer

int n = 5;
f(n);    // n is still 5
g(n);    // n is 6
h(&amp;n);   // n is 7</div>
<p>A <b>reference</b> (<code>int&amp;</code>) is an alias: a second name for an existing variable. It cannot be null and cannot be re-pointed at something else later. It is usually what you want. A <b>pointer</b> is a real variable holding an address. It can be null, and can be re-pointed or moved with arithmetic. Use a pointer when "no object" is a valid answer or when you walk an array.</p>
<p>By value, the callee works on a private copy, so the caller's variable is safe. By reference or pointer, the callee works on the original.</p>`,
        check: { q: "int n = 5; f(n); g(n); h(&n); with f by value, g by reference, h by pointer, each adding 1. What is n afterwards?",
          options: ["5", "6", "7", "8"], answer: 2,
          why: "f changes only its own copy, so n stays 5. g changes n itself (6). h follows the pointer to n and changes it (7). Two of the three calls modify the caller's variable." },
        scene(G) {
          const cols = [16, 224, 432];
          const heads = ["f(int x)", "g(int& x)", "h(int* x)"];
          const kinds = ["by value", "by reference", "by pointer"];
          cols.forEach((x, c) => {
            G.text(x, 44, heads[c], { size: 14, weight: 700 });
            G.label(x, 62, kinds[c], { size: 12 });
            G.box(x, 84, 110, 40, c === 0 ? "n = 5" : "n = 6", { fill: c === 0 ? "k" : "ok", size: 14 });
          });
          // value: a separate copy
          const cp = G.box(16, 190, 110, 40, "x = 6 (copy)", { fill: "hot", size: 12 });
          G.arrow(71, 186, 71, 128, { color: "muted", dash: "4 3" }); G.label(80, 160, "copied in", { size: 11 });
          // reference: tag stuck to n
          G.box(304, 74, 44, 20, "x", { fill: "bg", stroke: "ok", color: "ok", size: 12, dash: "3 3" });
          G.label(224, 170, "x is the same", { size: 12 }); G.label(224, 188, "variable, a", { size: 12 }); G.label(224, 206, "second name", { size: 12 });
          // pointer
          const pt = G.box(432, 190, 110, 40, "x = &n", { fill: "q", size: 13 });
          G.arrow(487, 186, 487, 128, { color: "q", w: 2.5 });
          const res = ["n stays 5", "n becomes 6", "n becomes 6"];
          cols.forEach((x, c) => G.text(x, 290, res[c], { size: 13, color: c === 0 ? "hot" : "ok", weight: 700 }));
          const cost = ["copies 4 B", "copies nothing", "copies 8 B (address)"];
          cols.forEach((x, c) => G.label(x, 312, cost[c], { size: 12 }));
          G.text(24, 372, "after f, g, h  (each adds 1, run in order):  n = 7", { size: 13 });
          G.from([cp, pt], { opacity: 0, scale: 0.7, transformOrigin: "center", stagger: 0.2, duration: 0.3 });
          G.caption("the copy is private; reference and pointer reach the original");
        } },

      { rail: "const", title: "const T& is the default way to pass big things",
        body: `<p>Copying a big object is slow and often pointless. A <code>std::vector&lt;float&gt;</code> of one million floats owns about 4 MB (<code>1,000,000 × 4 = 4,000,000</code> bytes). Passing it <i>by value</i> copies all of it. Passing it <i>by reference</i> copies an 8-byte address.</p>
<p>But a reference lets the function change your data. The word <b><code>const</code></b> forbids that, and the compiler enforces it:</p>
<div class="eq">float sum(const std::vector&lt;float&gt;&amp; v) {
  v[0] = 1;       // compile error: v is read-only
  float s = 0;
  for (float x : v) s += x;
  return s;
}</div>
<p>The rule of thumb: <b>small, cheap types</b> (<code>int</code>, <code>float</code>, pointers) go <b>by value</b>. <b>Big types you only read</b> go as <code>const T&amp;</code>. Types you must modify go as <code>T&amp;</code>. The same <code>const</code> after a method's parameter list (step 8) says "this method does not modify the object".</p>
<details><summary>Go deeper: where const goes next to a *</summary><p><code>const int* p</code> means "pointer to a read-only int": you can move <code>p</code> but not write through it. <code>int* const p</code> means "a read-only pointer to an int": you can write through it but not move it. Read right to left.</p></details>`,
        scene(G) {
          G.text(24, 44, "void f(std::vector<float> v)       by value", { size: 13 });
          const big = G.rect(24, 60, 590, 38, { fill: "hot", rx: 4, opacity: 0.85 });
          G.text(319, 85, "copies 4,000,000 bytes (1M floats × 4 B)", { anchor: "middle", size: 13, color: "bg", weight: 700 });
          G.text(24, 142, "void f(const std::vector<float>& v)", { size: 13 });
          G.rect(24, 158, 8, 38, { fill: "ok", rx: 2 });
          G.text(46, 183, "passes an 8-byte address", { size: 13, color: "ok" });
          G.label(24, 214, "bars are not to scale: 8 B is 500,000 times smaller", { size: 11 });
          G.box(24, 258, 590, 44, "v[0] = 1;     error: assignment of read-only location", { stroke: "hot", color: "hot", size: 13 });
          G.text(24, 340, "const T&  = read-only, no copy      ← default for big inputs", { size: 13, color: "ok" });
          G.text(24, 364, "T&        = read-write, no copy", { size: 13 });
          G.text(24, 388, "T         = copy; fine for int, float, pointers", { size: 13 });
          G.from(big, { scaleX: 0, transformOrigin: "left center", duration: 0.7 });
          G.caption("const& gives speed of a pointer with a promise not to write");
        } },

      { rail: "headers", title: "Headers declare, source files define, #include pastes",
        body: `<p>A real project has many files. File A needs to call a function that lives in file B. The compiler reads one file at a time, so it needs to be told the function exists. That is a <b>declaration</b>: the name, parameters and return type, with no body.</p>
<div class="eq">// stats.hpp          declaration (the promise)
double mean(const double* a, size_t n);

// stats.cpp          definition (the code)
#include "stats.hpp"
double mean(const double* a, size_t n) { ... }

// main.cpp           caller
#include "stats.hpp"
... mean(data, 4) ...</div>
<p><code>#include "stats.hpp"</code> is plain <b>text paste</b>: the preprocessor copies the header's text into the file before compiling. Put declarations in <code>.hpp</code> files so every caller sees the same promise. Put definitions in exactly one <code>.cpp</code>.</p>
<p>The <b>one-definition rule</b>: every function can be defined once in the whole program. Zero definitions is an <code>undefined reference</code> at link time. Two definitions is a <code>multiple definition</code> error. That is why a function defined inside a header (so every includer gets a copy) must be marked <code>inline</code>. Start each header with <code>#pragma once</code> so including it twice in one file is harmless. Exercise 4 makes you split a module this way.</p>`,
        scene(G) {
          G.box(24, 50, 190, 78, "", { stroke: "k" });
          G.text(36, 74, "stats.hpp", { size: 13, weight: 700 }); G.text(36, 98, "double mean(...);", { size: 12, color: "k" }); G.label(36, 116, "declaration: a promise", { size: 11 });
          G.box(24, 290, 190, 78, "", { stroke: "ok" });
          G.text(36, 314, "stats.cpp", { size: 13, weight: 700 }); G.text(36, 338, "double mean(...) {…}", { size: 12, color: "ok" }); G.label(36, 356, "definition: the code", { size: 11 });
          G.box(250, 150, 190, 108, "", { stroke: "ink" });
          G.text(262, 174, "main.cpp", { size: 13, weight: 700 }); G.text(262, 198, "#include \"stats.hpp\"", { size: 12, color: "muted" });
          const pasted = G.text(262, 222, "double mean(...);", { size: 12, color: "k" }); G.text(262, 244, "... mean(data, 4) ...", { size: 12 });
          const a1 = G.arrow(214, 128, 300, 146, { color: "k", w: 2 }); G.label(222, 122, "#include = paste", { size: 11, color: "k" });
          const a2 = G.arrow(120, 130, 120, 286, { color: "k", dash: "4 3" }); G.label(128, 210, "same promise", { size: 11 });
          G.box(490, 150, 126, 40, "main.o", { fill: "k" }); G.box(490, 310, 126, 40, "stats.o", { fill: "ok" });
          G.label(456, 212, "calls mean,", { size: 11 }); G.label(456, 228, "address ?", { size: 11 });
          G.arrow(553, 194, 553, 306, { color: "q", w: 2.5 }); G.label(562, 256, "linker", { size: 12, color: "q" });
          G.text(24, 408, "mean defined 0×: undefined reference · 2×: multiple definition", { size: 12, color: "hot" });
          G.from([pasted, a1, a2], { opacity: 0, stagger: 0.25, duration: 0.4 });
          G.caption("declare in the header, define once in a .cpp, let the linker join them");
        } },

      { rail: "classes", title: "A class bundles data and functions; its destructor runs at the closing brace",
        body: `<p>A <b><code>struct</code></b> groups fields into one type. C++ lets it also carry functions (<b>methods</b>), which is what makes it a class. (<code>class</code> and <code>struct</code> differ only in default access: <code>class</code> members are private until you write <code>public:</code>.)</p>
<div class="eq">struct Vec2 {
  float x, y;
  Vec2(float a, float b) : x(a), y(b) {}
  float length() const {
    return std::sqrt(x*x + y*y);
  }
};
const Vec2 v(3, 4);
v.length();       // 5</div>
<p>The <b>constructor</b> has the type's name and runs when an object is created. The part after the colon is a <b>member initializer list</b>. A <code>const</code> after the parameter list promises the method does not change the object, which lets you call it on a <code>const Vec2</code>. <code>Vec2</code> is exactly two floats: <code>sizeof(Vec2) == 8</code>. Methods cost no space.</p>
<p>The <b>destructor</b>, written <code>~Name()</code>, runs automatically when the object's scope ends at the closing brace. Objects are destroyed in reverse order of creation. Nothing in Python works like this, and it is the idea the whole of R.2 is built on.</p>
<div class="eq">void run() {
  Logger a("a");   // ctor a
  Logger b("b");   // ctor b
}                  // dtor b, then dtor a</div>`,
        check: { q: "For the run() above, in which order are the constructors and destructors called?",
          options: ["ctor a, ctor b, dtor a, dtor b", "ctor a, ctor b, dtor b, dtor a", "ctor b, ctor a, dtor a, dtor b"], answer: 1,
          why: "Objects are created in the order the lines run (a, then b) and destroyed in the reverse order (b, then a), like unstacking plates. Later objects may depend on earlier ones, so they must go first." },
        scene(G) {
          G.line(110, 60, 110, 232, { color: "muted", dash: "4 4" }); G.line(590, 60, 590, 232, { color: "muted", dash: "4 4" });
          G.text(110, 52, "{", { anchor: "middle", size: 18 }); G.text(590, 52, "}", { anchor: "middle", size: 18 });
          const la = G.rect(150, 90, 410, 30, { fill: "k", opacity: 0.35, rx: 4 }); G.text(172, 110, "Logger a", { size: 13 });
          const lb = G.rect(250, 150, 240, 30, { fill: "v", opacity: 0.35, rx: 4 }); G.text(272, 170, "Logger b", { size: 13 });
          const dots = [[150, 105, "ok", "1"], [250, 165, "ok", "2"], [490, 165, "hot", "3"], [560, 105, "hot", "4"]];
          dots.forEach(([x, y, col, n]) => { G.circle(x, y, 11, { fill: col }); G.text(x, y + 4, n, { anchor: "middle", size: 12, color: "bg", weight: 700 }); });
          G.label(110, 252, "start of scope", { anchor: "middle", size: 11 }); G.label(590, 252, "end of scope", { anchor: "middle", size: 11 });
          G.text(24, 292, "1  ctor a     2  ctor b", { size: 13, color: "ok" });
          G.text(24, 316, "3  dtor b     4  dtor a   (reverse order)", { size: 13, color: "hot" });
          G.label(24, 354, "layout of struct Vec2 { float x, y; }", { size: 12 });
          const vx = G.box(24, 366, 80, 34, "x", { fill: "k" }), vy = G.box(108, 366, 80, 34, "y", { fill: "k" });
          G.text(204, 388, "= 8 bytes, methods take no space", { size: 12, color: "muted" });
          G.from([la, lb], { opacity: 0, scaleX: 0, transformOrigin: "left center", stagger: 0.3, duration: 0.5 }); G.from([vx, vy], { opacity: 0, delay: 0.6, stagger: 0.1, duration: 0.2 });
          G.caption("the destructor is called for you at the closing brace");
        } },

      { rail: "std library", title: "The standard library you will meet first",
        body: `<p>You rarely allocate arrays by hand. The standard library gives you containers:</p>
<div class="eq">#include &lt;vector&gt;
#include &lt;string&gt;
#include &lt;array&gt;

std::vector&lt;float&gt; v = {1, 2, 3};
v.push_back(4);             // grows: size 4
std::array&lt;int, 4&gt; a = {1, 2, 3, 4};
std::string s = "tokens";

for (float x : v) total += x;
for (auto&amp; x : v) x *= 2;   // &amp; edits in place</div>
<ul><li><code>std::vector&lt;T&gt;</code> is a growable array. It is a small header (a pointer, a size and a capacity: three 8-byte values, 24 bytes) plus a separate block of elements on the heap. It frees that block itself when it dies.</li>
<li><code>std::array&lt;T, N&gt;</code> keeps all <code>N</code> elements inside the object. <code>std::array&lt;int, 4&gt;</code> is exactly 16 bytes and needs no heap.</li>
<li><code>std::string</code> is a vector of characters that also keeps a terminating zero.</li></ul>
<p><b>Range-for</b> (<code>for (x : container)</code>) replaces the index loop whenever you do not need the index. Use <code>auto</code> to let the compiler fill in the type, and add <code>&amp;</code> or <code>const&amp;</code> to avoid copying each element. <code>v.size()</code> returns a <code>size_t</code>, an unsigned type, so mixing it with negative numbers is a classic bug. The reference site is cppreference.com.</p>`,
        scene(G) {
          G.text(24, 40, "std::vector<float> v  (3 elements, room for 4)", { size: 13 });
          G.label(24, 62, "the object: 24 bytes", { size: 11 });
          const hd = [G.box(24, 70, 80, 36, "ptr", { fill: "q" }), G.box(108, 70, 80, 36, "size 3", { fill: "k" }), G.box(192, 70, 80, 36, "cap 4", { fill: "k" })];
          G.label(300, 62, "heap block (owned by v)", { size: 11 });
          const el = [1, 2, 3].map((n, i) => G.box(300 + i * 70, 70, 64, 36, String(n), { fill: "ok", size: 14 }));
          G.box(300 + 3 * 70, 70, 64, 36, "spare", { stroke: "muted", color: "muted", dash: "4 3", size: 11 });
          G.path("M 64 108 C 64 160, 270 160, 300 108", { color: "q", w: 2, arrow: true });
          G.text(24, 214, "std::array<int, 4> a  (16 bytes, inline, no heap)", { size: 13 });
          const ar = [1, 2, 3, 4].map((n, i) => G.box(24 + i * 70, 226, 64, 36, String(n), { fill: "k", size: 14 }));
          G.text(24, 316, "for (float x : v) total += x;     // read each", { size: 13 });
          G.text(24, 340, "for (auto& x : v) x *= 2;         // change each", { size: 13 });
          G.text(24, 364, "for (const auto& s : names) ...   // no copies", { size: 13 });
          G.from([...hd, ...el, ...ar], { opacity: 0, y: 10, stagger: 0.05, duration: 0.25 });
          G.caption("vector = small header + heap block; array = elements inline");
        } },

      { rail: "undefined behaviour", title: "Undefined behaviour: the compiler will not catch it",
        body: `<p>Python checks every index and every variable. C++ does not. If you break a rule of the language, the result is <b>undefined behaviour</b> (UB): the standard places no limit on what happens. It may crash, print garbage, or work perfectly today and fail after you change the compiler flags. The compiler is allowed to assume UB never happens, so it can optimise on that assumption.</p>
<div class="eq">int a[4] = {1, 2, 3, 4};
int v = a[4];        // UB: one past the end

int x;
printf("%d", x);      // UB: never set</div>
<p>Neither line is a compile error. <code>-Wall -Wextra</code> catches some simple cases. For the rest, the compiler can add run-time checks for you. These are <b>sanitizers</b>: <code>-fsanitize=address,undefined -g</code> stops the program at the exact line of an out-of-bounds access or an overflow, and names the problem. That is why every C++ exercise in this course after R.1 builds with them. R.2 shows how to read their reports.</p>
<p>Habits that keep you safe: initialise every variable, prefer <code>std::vector</code> and <code>std::array</code> with range-for, and run tests under sanitizers.</p>`,
        check: { q: "int a[4]; reading a[4] compiles with no error. What is guaranteed to happen at run time?",
          options: ["It always crashes", "It always reads 0", "Nothing is guaranteed: it may crash, read garbage, or seem to work"], answer: 2,
          why: "Reading past the end is undefined behaviour. The language promises nothing, so the same program can behave differently on another machine or compiler setting. A sanitizer turns it into a certain, reported failure." },
        scene(G) {
          const ok = [1, 2, 3, 4].map((n, i) => G.box(40 + i * 82, 80, 76, 44, String(n), { fill: "ok", size: 15 }));
          [0, 1, 2, 3].forEach((i) => G.text(78 + i * 82, 70, `a[${i}]`, { anchor: "middle", size: 12 }));
          const bad = G.box(40 + 4 * 82, 80, 76, 44, "?", { stroke: "hot", color: "hot", dash: "4 3", size: 18 });
          G.text(78 + 4 * 82, 70, "a[4]", { anchor: "middle", size: 12, color: "hot" });
          G.box(40 + 5 * 82, 80, 76, 44, "?", { stroke: "line", color: "muted", dash: "4 3", size: 18 });
          G.label(40, 146, "the array ends here  |  someone else's memory", { size: 12 });
          G.arrow(78 + 4 * 82 - 30, 190, 78 + 4 * 82 - 10, 130, { color: "hot", w: 2.5 });
          ["crash", "garbage value", "looks fine today"].forEach((t, i) => G.box(40 + i * 190, 206, 176, 36, t, { stroke: ["hot", "v", "muted"][i], color: ["hot", "v", "muted"][i], size: 13 }));
          G.label(40, 262, "same source line: any of these, on any run", { size: 12 });
          G.box(40, 304, 176, 40, "int x;   (never set)", { stroke: "muted", size: 12 });
          G.box(230, 304, 66, 40, "?????", { fill: "v", size: 13 });
          G.text(312, 330, "reading x = whatever was in memory", { size: 12 });
          G.text(24, 388, "-Wall: warns about some.   -fsanitize=address,undefined: finds them at run time", { size: 12, color: "ok" });
          G.from(ok, { opacity: 0, stagger: 0.08, duration: 0.2 }); G.pulse(bad, { repeat: 6 });
          G.caption("C++ trusts you: a bug is silent unless you add checks");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>C++ compiles each <code>.cpp</code> into an object file, then the linker joins the objects. Compile errors and link errors are different things.</li>
<li>Types have fixed sizes (<code>char</code> 1, <code>int</code> and <code>float</code> 4, <code>double</code> and pointers 8), and <code>sizeof</code> tells you.</li>
<li>A pointer is an address with a type. <code>&amp;x</code> takes it, <code>*p</code> follows it, and <code>p + i</code> moves by <code>i × sizeof(T)</code> bytes. <code>a[i]</code> is <code>*(a + i)</code>.</li>
<li>Arguments go by value (copy), reference (alias) or pointer (address). Big read-only inputs go as <code>const T&amp;</code>.</li>
<li>Headers declare, one <code>.cpp</code> defines, <code>#include</code> pastes text, and a definition may exist only once.</li>
<li>A class has a constructor at creation and a destructor at the closing brace. <code>std::vector</code>, <code>std::array</code> and range-for cover most array work.</li>
<li>Out-of-bounds access and uninitialised reads are undefined behaviour. The compiler will not stop you, so you test with sanitizers.</li></ul>
<h3>Go further</h3><ul>
<li><a href="https://www.learncpp.com/">learncpp.com</a>: the chapters on pointers, references, <code>const</code> and classes, for slower, longer explanations.</li>
<li><a href="https://en.cppreference.com/w/">cppreference</a>: look up any function or container exactly.</li>
<li><a href="https://cs50.harvard.edu/x/">CS50x</a>, Week 4 "Memory": a gentler video route to pointers.</li></ul>
<p>Next, R.2 follows the memory itself: the stack, the heap, and who is responsible for freeing what. All links are listed as UNVERIFIED in SOURCES.md until checked from a networked machine.</p>`,
    sim: {
      title: "Pointer arithmetic explorer",
      intro: "Pick an element type, an array length, a base address and an index i. The picture shows where <code>p + i</code> lands. Dashed boxes are outside the array: forming that address is easy, but reading or writing it is undefined behaviour.",
      height: 290,
      controls: [
        { id: "type", label: "element type", type: "select", value: "int", options: [["char", "char (1 B)"], ["int", "int (4 B)"], ["float", "float (4 B)"], ["double", "double (8 B)"], ["vec3", "struct of 3 floats (12 B)"]] },
        { id: "base", label: "base address p", type: "select", value: 4096, options: [[4096, "0x1000"], [8192, "0x2000"], [65536, "0x10000"]] },
        { id: "n", label: "array length n", min: 3, max: 8, step: 1, value: 6 },
        { id: "i", label: "index i in p + i", min: -2, max: 10, step: 1, value: 3 },
      ],
      draw(G, v) {
        const s = TYPE_SIZE[v.type], col = TYPE_COL[v.type], n = v.n, i = v.i, base = v.base;
        const lo = Math.min(0, i * s), hi = Math.max(n * s, (i + 1) * s), D = hi - lo;
        const sc = 600 / D, x0 = 20, X = (b) => x0 + (b - lo) * sc;
        const addr = base + i * s;
        G.label(16, 18, `${v.type} is ${s} byte${s > 1 ? "s" : ""} · array of ${n} = ${n * s} bytes from ${HEX(base)}`, { size: 12 });
        const k0 = Math.floor(lo / s), k1 = Math.ceil(hi / s) - 1;
        for (let k = k0; k <= k1; k++) {
          const inside = k >= 0 && k < n, w = s * sc;
          if (inside) G.rect(X(k * s), 56, w - 2, 50, { fill: col, rx: 3, opacity: k === i ? 1 : 0.55 });
          else G.rect(X(k * s), 56, w - 2, 50, { stroke: "hot", dash: "4 3", rx: 3, opacity: 0.8 });
          if (w >= 18) G.text(X(k * s) + (w - 2) / 2, 86, String(k), { anchor: "middle", size: Math.min(14, w / 2.2), color: inside ? "bg" : "hot", weight: 700 });
          if (w >= 36 || k % 2 === 0) G.label(X(k * s) + 2, 48, `+${k * s}`, { size: 10 });
        }
        // markers
        const px = X(0) + 1, qx = X(i * s) + 1;
        G.arrow(px, 160, px, 112, { color: "ink", w: 2 });
        G.text(px, 176, i === 0 ? "p = p + 0" : "p", { anchor: "middle", size: 13 });
        if (i !== 0) {
          G.arrow(qx, 200, qx, 112, { color: i >= 0 && i < n ? "ok" : "hot", w: 2.5 });
          G.text(qx, 216, `p + ${i}`, { anchor: "middle", size: 13, color: i >= 0 && i < n ? "ok" : "hot", weight: 700 });
        }
        G.text(16, 250, `address = ${HEX(base)} + ${i} × ${s} = ${HEX(addr)}`, { size: 13 });
        G.label(16, 272, "byte offsets above the boxes are measured from p", { size: 11 });
        const inb = i >= 0 && i < n;
        const where = inb ? `element ${i}` : i === n ? "one past the end" : i < 0 ? "before the array" : "past the end";
        const msg = inb ? "inside the array: safe to read and write" : i === n ? "one past the end: may compare, never read" : "outside the array: undefined behaviour if used";
        return [
          { title: "Where p + i lands", rows: [["sizeof(T)", `${s} B`], ["bytes stepped (i × sizeof)", `${i * s} B`], ["address of p + i", HEX(addr)], ["lands on", where], ["(p + i) − p", `${i} element${Math.abs(i) === 1 ? "" : "s"}`]], chip: [inb, msg] },
          { title: "Remember", html: `<p class="note"><code>p + i</code> counts in <b>elements</b>. The compiler multiplies by <code>sizeof(T)</code>. Adding <code>i</code> to the address as a plain number would be wrong for any type bigger than <code>char</code>. The base addresses here are multiples of 64, so they are aligned for every type. Real addresses look like <code>0x7ffd3a2c1b40</code>.</p>` },
        ];
      },
    },
    practice: {
      intro: `Four exercises, each tested with CTest. Run from the repository root. Add <code>-DS2S_USE_SOLUTIONS=ON</code> to the configure step to build the reference solutions and confirm your toolchain works before you start. The starters compile but the tests fail until you write the code.`,
      items: [
        { title: "Swap by pointer and by reference", tier: "T0 · easy", goal: "Write swap_by_pointer(int*, int*) and swap_by_ref(int&, int&). Same job, two calling styles.",
          cmd: "cmake -S course/R-readiness/R.1-cpp-from-zero/exercises -B /tmp/r1-ex && cmake --build /tmp/r1-ex -j && ctest --test-dir /tmp/r1-ex -R 01-swap --output-on-failure" },
        { title: "Sum and max with pointer arithmetic", tier: "T0 · easy", goal: "Sum and maximum of a raw float array by walking a pointer, with no a[i]. Handle n == 0.",
          cmd: "ctest --test-dir /tmp/r1-ex -R 02-array-sum --output-on-failure" },
        { title: "A Vec2 class", tier: "T0 · easy", goal: "A struct with constructors, length(), operator+ and a const method that works on a const object.",
          cmd: "ctest --test-dir /tmp/r1-ex -R 03-vec2-class --output-on-failure" },
        { title: "Split a module into .hpp and .cpp", tier: "T0 · medium", goal: "Fill in stats.cpp so it matches the declarations in stats.hpp; it builds as a library the test links. Then provoke the one-definition error on purpose.",
          cmd: "ctest --test-dir /tmp/r1-ex -R 04-split-header --output-on-failure" },
      ],
      labs: [
        { label: "Self-check questions", path: "course/R-readiness/R.1-cpp-from-zero/quiz.md" },
        { label: "Module README with the exercise table and further reading", path: "course/R-readiness/R.1-cpp-from-zero/README.md" },
        { label: "Next: P0.1 builds strides, RAII and sanitizers on these ideas", path: "course/P0-systems-primer/P0.1-c-cpp-for-systems/README.md" },
      ],
    },
  });
})();
