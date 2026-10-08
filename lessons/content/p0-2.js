/* P0.2 — Virtual memory and mmap. From "why addresses are fake" to loading a model with zero copies. */
(function () {
  const F = S2S.fmt;
  const KiB = 1024, MiB = 1024 * 1024, GiB = 1024 * MiB;

  S2S.lesson({
    id: "p0-2", n: "P0.2", title: "Virtual memory and mmap",
    subtitle: "Systems primer · first principles · T0, any laptop",
    kicker: "Lesson · ≈ 45 min",
    headline: "Every address you have ever printed was a lie",
    intro: `<p>In P0.1 an address was a number on a line of bytes. That line is an illusion that the CPU and the operating system build for each program. This lesson takes the illusion apart: pages, page tables, the TLB that makes translation fast, and the page faults that let a 16 GB model file "load" in a microsecond.</p>
<p>It matters because a serving system's cold start is mostly moving weights: <code>disk → page cache → host memory → GPU</code>. Every arrow in that chain is a virtual-memory idea, and by the end you will load a model file with one <code>mmap</code> call and no copies.</p>`,
    facts: ["10 steps", "4 checkpoints", "1 simulator", "4 exercises"],
    legend: [["q", "virtual"], ["k", "physical"], ["ok", "hit / present"], ["hot", "miss / fault"], ["w", "kernel / disk"]],
    prev: "p0-1", next: "p0-3",
    steps: [
      { rail: "why virtual", title: "Two programs, one address",
        body: `<p>Run two programs at once. Both were compiled to keep a variable at address <code>0x4000</code>. If addresses went straight to RAM chips, they would overwrite each other, and any bug in one program could scribble over the other or over the operating system.</p>
<p>The fix is a layer of indirection. Every load and store a program issues uses a <b>virtual address</b> (VA). A piece of the CPU called the <b>MMU</b> (memory management unit) translates it to a <b>physical address</b> (PA) in RAM, using a table that the operating system keeps <i>per process</i>.</p>
<p>So <code>0x4000</code> in process A and <code>0x4000</code> in process B are different bytes of RAM. Each process sees a private, clean address space; the OS decides where things really live, and can refuse translations that a process has no right to.</p>
<div class="analogy"><b>Picture it</b>Two hotels that both have a "room 12". The room number only means something together with the hotel. The page table is the hotel's register that says which actual building and floor room 12 is.</div>`,
        scene(G) {
          G.text(24, 40, "virtual (what each program sees)", { size: 13, color: "q" });
          G.text(420, 40, "physical RAM", { size: 13, color: "k" });
          G.box(24, 70, 220, 120, "", { stroke: "q" }); G.text(36, 92, "process A", { size: 13 });
          G.box(56, 120, 150, 40, "0x4000: x = 7", { fill: "q", size: 12 });
          G.box(24, 230, 220, 120, "", { stroke: "q" }); G.text(36, 252, "process B", { size: 13 });
          G.box(56, 280, 150, 40, "0x4000: x = 99", { fill: "q", size: 12 });
          const frames = [];
          for (let f = 0; f < 8; f++) frames.push(G.box(420, 60 + f * 42, 200, 36, `frame ${f}`, { fill: f === 2 || f === 6 ? "k" : "line", size: 12 }));
          const a = G.arrow(208, 140, 416, 162, { color: "ink", w: 2 });
          const b = G.arrow(208, 300, 416, 330, { color: "ink", w: 2 });
          G.box(276, 128, 112, 28, "A's table", { stroke: "muted", color: "muted", size: 11 });
          G.box(276, 290, 112, 28, "B's table", { stroke: "muted", color: "muted", size: 11 });
          G.label(24, 396, "same virtual address, different physical frames", { size: 13, color: "ink" });
          G.from([a, b], { opacity: 0, stagger: 0.4, duration: 0.5 });
          G.caption("the MMU translates every access through the process's own table");
        } },

      { rail: "pages", title: "Translate in pages, not bytes",
        body: `<p>A table entry per byte would be bigger than the memory it describes. So memory is managed in fixed-size blocks called <b>pages</b>: 4 KiB (4,096 bytes) on x86-64 and Linux/arm64, 16 KiB on Apple silicon. The same-size blocks of physical RAM are called <b>frames</b>.</p>
<p>Because 4,096 = 2¹², the low 12 bits of an address say <i>where inside the page</i> (the <b>offset</b>), and the remaining bits say <i>which page</i> (the <b>virtual page number</b>, VPN). Only the page number is translated; the offset is copied through unchanged.</p>
<div class="eq">VA      = 0x12345678
offset  = VA &amp; 0xFFF   = 0x678     (low 12 bits)
VPN     = VA &gt;&gt; 12     = 0x12345
table[0x12345] → frame 0x0abcd
PA      = 0x0abcd &lt;&lt; 12 | 0x678 = 0x0abcd678</div>
<p>One translation now covers 4,096 bytes. That is the whole trick, and the reason page size will matter so much later.</p>`,
        scene(G) {
          G.text(24, 44, "virtual address 0x12345678", { size: 14, color: "q" });
          const vpn = G.box(24, 60, 380, 50, "page number 0x12345", { fill: "q", size: 14 });
          const off = G.box(410, 60, 200, 50, "offset 0x678", { fill: "ok", size: 14 });
          G.label(24, 128, "bits 31 … 12", { size: 11 }); G.label(410, 128, "bits 11 … 0 (4 KiB = 2¹²)", { size: 11 });
          G.box(120, 170, 200, 70, "", { stroke: "muted" }); G.text(220, 198, "page table", { anchor: "middle", size: 13 });
          G.text(220, 222, "0x12345 → 0x0abcd", { anchor: "middle", size: 12, color: "k" });
          G.arrow(214, 112, 214, 166, { color: "q", w: 2 });
          G.arrow(510, 112, 510, 296, { color: "ok", w: 2, dash: "5 4" });
          G.label(520, 210, "copied", { size: 11 }); G.label(520, 226, "unchanged", { size: 11 });
          G.arrow(214, 242, 214, 296, { color: "k", w: 2 });
          const fr = G.box(24, 300, 380, 50, "frame 0x0abcd", { fill: "k", size: 14 });
          G.box(410, 300, 200, 50, "offset 0x678", { fill: "ok", size: 14 });
          G.text(24, 386, "physical address 0x0abcd678", { size: 14, color: "k" });
          G.from(fr, { opacity: 0, y: -30, duration: 0.6, delay: 0.3 });
          G.caption("only the page number is translated; the offset passes straight through");
        } },

      { rail: "page table", title: "A tree of small tables",
        body: `<p>x86-64 uses 48-bit virtual addresses. With 4 KiB pages that leaves 36 bits of page number: 2³⁶ ≈ 69 billion pages. A flat table with one 8-byte entry each would be 512 GiB, per process. Impossible.</p>
<p>Most of that space is never used, so the table is a <b>tree</b>. The 36 bits are cut into four 9-bit indices. Each level is a 4 KiB page holding 512 entries of 8 bytes (2⁹ = 512), and each entry points to the next level's table. Unused branches simply don't exist.</p>
<div class="eq">| 47..39 | 38..30 | 29..21 | 20..12 | 11..0  |
|  PML4  |  PDPT  |   PD   |   PT   | offset |

VA 0x7f3a12345678 →
  PML4 254 · PDPT 232 · PD 145 · PT 325 · offset 0x678</div>
<p>The price: a full translation is <b>four dependent memory reads</b> (each level's address comes from the previous one) before the load you actually wanted can start. That is called a <b>page walk</b>. <code>examples/01_page_walk.py</code> splits addresses this way so you can check the example by hand.</p>`,
        check: { q: "With 4 KiB pages and 4-level paging, how many page-table reads does a translation need in the worst case, before the real load?",
          options: ["1", "4", "9", "512"], answer: 1,
          why: "One read per level: PML4, PDPT, PD, PT. Each read's result is the address of the next table, so they cannot overlap. Caches often serve some of them, which is why the average is better." },
        scene(G) {
          const parts = [["PML4", "254"], ["PDPT", "232"], ["PD", "145"], ["PT", "325"], ["offset", "0x678"]];
          parts.forEach(([n, v], i) => { G.box(24 + i * 120, 44, 112, 46, `${n} ${v}`, { fill: i < 4 ? "q" : "ok", size: 12 }); });
          G.label(24, 108, "9 + 9 + 9 + 9 index bits, 12 offset bits", { size: 12 });
          const tabs = [];
          for (let l = 0; l < 4; l++) {
            const x = 24 + l * 150, y = 136 + l * 30;
            const t = G.group();
            G.rect(x, y, 110, 136, { stroke: "muted", rx: 6, parent: t });
            for (let e = 0; e < 6; e++) G.rect(x + 8, y + 10 + e * 20, 94, 14, { fill: e === 3 ? "q" : "line", rx: 3, opacity: e === 3 ? 0.95 : 0.5, parent: t });
            G.text(x + 55, y + 152, ["PML4", "PDPT", "PD", "PT"][l] + " (512)", { anchor: "middle", size: 11, color: "muted", parent: t });
            tabs.push(t);
            if (l < 3) G.arrow(x + 102, y + 77, x + 148, y + 102, { color: "q", w: 2 });
          }
          G.arrow(24 + 3 * 150 + 102, 136 + 90 + 77, 604, 370, { color: "k", w: 2 });
          G.text(600, 388, "frame", { anchor: "end", size: 12, color: "k" });
          G.label(24, 408, "read 1 → read 2 → read 3 → read 4 → data", { size: 13, color: "hot" });
          G.from(tabs, { opacity: 0, stagger: 0.35, duration: 0.3 });
          G.caption("four dependent reads per translation: the page walk");
        } },

      { rail: "TLB", title: "The TLB: a cache of translations",
        body: `<p>Four extra reads on every load would make every program several times slower. But programs reuse the same pages over and over, so the CPU keeps recent translations in a small, very fast cache called the <b>TLB</b> (translation lookaside buffer).</p>
<ul><li><b>TLB hit</b>: the page number is in the TLB. Translation costs essentially nothing.</li>
<li><b>TLB miss</b>: the hardware does the page walk, then stores the result in the TLB, evicting an older entry.</li></ul>
<p>A TLB holds from tens to a couple of thousand entries, depending on the level and the CPU. Reading a 1 GiB array front to back touches 262,144 pages, but each page's 4,096 bytes are read one after another, so there is one miss per page and thousands of hits in between: a hit rate above 99.9%. Jumping randomly across that same gigabyte misses on almost every access. <code>01_page_walk.py</code> runs exactly these patterns through a 64-entry TLB.</p>`,
        scene(G) {
          G.box(24, 60, 140, 46, "VA page 0x12345", { fill: "q", size: 12 });
          G.rect(220, 40, 180, 200, { stroke: "ok", rx: 8 }); G.text(310, 34, "TLB (small, fast)", { anchor: "middle", size: 12, color: "ok" });
          const rows = ["0x00401 → 0x1a2b3", "0x12345 → 0x0abcd", "0x7ff00 → 0x33c01", "0x00402 → 0x1a2b4", "…"];
          rows.forEach((r, i) => G.box(232, 54 + i * 36, 156, 28, r, { fill: i === 1 ? "ok" : undefined, stroke: i === 1 ? undefined : "line", color: i === 1 ? "bg" : "muted", size: 11 }));
          const hit = G.arrow(166, 83, 228, 104, { color: "ok", w: 2 });
          G.text(420, 104, "hit: ≈ free", { size: 14, color: "ok" });
          G.box(24, 300, 140, 46, "VA page 0x55555", { fill: "q", size: 12 });
          G.arrow(166, 322, 226, 230, { color: "hot", w: 2, dash: "4 3" });
          G.text(176, 282, "miss", { size: 13, color: "hot" });
          const walk = [];
          for (let l = 0; l < 4; l++) walk.push(G.box(240 + l * 92, 320, 80, 40, ["PML4", "PDPT", "PD", "PT"][l], { stroke: "hot", color: "hot", size: 12 }));
          for (let l = 0; l < 3; l++) G.arrow(320 + l * 92, 340, 330 + l * 92, 340, { color: "hot" });
          G.label(240, 384, "page walk: 4 dependent reads, then fill the TLB", { size: 12, color: "hot" });
          G.pulse(hit, { repeat: 4 }); G.from(walk, { opacity: 0, stagger: 0.25, duration: 0.25, delay: 0.4 });
          G.caption("most accesses hit; a miss pays for the walk once per page");
        } },

      { rail: "reach · huge pages", title: "TLB reach, and why huge pages exist",
        body: `<p>A TLB with <i>E</i> entries can cover at most <i>E</i> pages at once. Multiply by the page size and you get its <b>reach</b>: how much memory you can touch in any order without missing.</p>
<div class="eq">reach = entries × page size

1536 entries × 4 KiB = 6 MiB
1536 entries × 2 MiB = 3 GiB     (512× more)</div>
<p>(1536 is an example L2 TLB size for the arithmetic; exercise 1 asks you to look up your own CPU's.) Llama-3-8B's weights in bf16 are about 16 GB. In 4 KiB pages that is about 3.9 million pages, roughly 2,500 times what the example TLB can hold. Each decode step reads all of them.</p>
<p>The fix is <b>huge pages</b>: 2 MiB pages, where the walk stops one level early (the PD entry points straight at a 2 MiB frame and the low 21 bits are the offset). The same 16 GB is then about 7,630 pages. On Linux, <code>madvise(p, len, MADV_HUGEPAGE)</code> asks for them (<code>examples/03_hugepages.c</code>). GPU drivers and big-memory servers rely on them for exactly this reason.</p>`,
        check: { q: "A TLB has 64 entries. Your program touches 1 MiB of memory in random order. With 4 KiB pages, does it fit in the TLB's reach?",
          options: ["Yes: 64 × 4 KiB = 1 MiB is exactly the working set", "No: reach is 64 × 4 KiB = 256 KiB, a quarter of the working set", "No: reach is 64 bytes"], answer: 1,
          why: "64 × 4 KiB = 256 KiB. 1 MiB needs 256 pages, so at any moment only 64 of 256 translations are cached and about 3 out of 4 random accesses miss. With 2 MiB pages, one entry covers it all." },
        scene(G) {
          G.text(24, 40, "same 16 MiB of memory, two page sizes", { size: 13 });
          G.text(24, 74, "4 KiB pages: 4,096 pages", { size: 13, color: "hot" });
          const small = G.grid(24, 86, 16, 64, 9, 8, (r, c) => (r * 64 + c < 16 ? "ok" : "hot"), { gap: 1, rx: 0, opacity: 0.85 });
          G.label(24, 232, "each cell = 4 pages · green = what a 64-entry TLB covers (256 KiB)", { size: 11 });
          G.text(24, 276, "2 MiB pages: 8 pages", { size: 13, color: "ok" });
          const big = G.grid(24, 288, 1, 8, 72, 60, () => "ok", { gap: 4, rx: 4 });
          G.label(24, 368, "8 entries cover everything; the same 64-entry TLB reaches 128 MiB", { size: 12 });
          G.text(24, 404, "reach = entries × page size", { size: 15, color: "ink" });
          G.from(big, { opacity: 0, scale: 0.5, transformOrigin: "center", stagger: 0.08, duration: 0.3 });
          G.caption("bigger pages: one TLB entry covers 512× more memory");
        } },

      { rail: "page faults", title: "mmap: map now, load on first touch",
        body: `<p><code>mmap(file)</code> asks the OS to make a file appear as a range of memory. Surprisingly, it does not read the file. It only creates page-table entries marked <b>not present</b>, and returns in microseconds even for 16 GB.</p>
<p>The first time the program touches one of those pages, the MMU finds "not present" and raises a <b>page fault</b>: the CPU stops the program and jumps into the kernel. The kernel finds the file data for that page, puts it in a frame, fills in the page-table entry, and resumes the program at the same instruction, which now succeeds. The program never notices, except for the time it took.</p>
<p>This is <b>demand paging</b>: data arrives only when needed. It explains a classic benchmarking mistake. Timing <code>mmap()</code> alone measures almost nothing; the cost moved to the first touch. Always time the touch.</p>
<div class="eq">char* p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
// nothing read yet
sum += p[0];        // page fault → kernel loads page 0
sum += p[1];        // same page: no fault
sum += p[4096];     // new page: another fault</div>`,
        scene(G) {
          G.text(24, 40, "a mapped file: 24 pages, all 'not present' after mmap()", { size: 13 });
          const cells = [];
          for (let i = 0; i < 24; i++) cells.push(G.rect(24 + (i % 12) * 50, 60 + Math.floor(i / 12) * 54, 44, 44, { fill: "line", rx: 4, opacity: 0.5 }));
          const touched = [0, 1, 2, 3, 4, 5, 6];
          const hits = touched.map((i) => G.rect(24 + (i % 12) * 50, 60, 44, 44, { fill: "ok", rx: 4 }));
          G.text(24 + 7 * 50 + 22, 88, "✕", { anchor: "middle", size: 18, color: "hot" });
          G.arrow(24 + 7 * 50 + 22, 106, 24 + 7 * 50 + 22, 196, { color: "hot", w: 2 });
          G.box(250, 200, 250, 60, "", { fill: "w", boxOpacity: 0.7 });
          G.text(375, 224, "kernel: page fault handler", { anchor: "middle", size: 13 });
          G.text(375, 246, "find data → map frame → resume", { anchor: "middle", size: 12, color: "muted" });
          G.label(24, 300, "green = touched and now mapped · red ✕ = first touch of page 7", { size: 12 });
          G.text(24, 340, "mmap() of 16 GB: microseconds (no data moved)", { size: 13, color: "ok" });
          G.text(24, 366, "first touch of each page: a trip into the kernel", { size: 13, color: "hot" });
          G.from(hits, { opacity: 0, stagger: 0.18, duration: 0.2 });
          G.caption("demand paging: the cost moves from mmap() to the first touch");
        } },

      { rail: "page cache", title: "Minor faults, major faults, and the page cache",
        body: `<p>Where does the kernel get the page from? Linux keeps recently used file data in otherwise idle RAM, called the <b>page cache</b>. So a fault has two flavours:</p>
<ul><li><b>Minor fault</b>: the page is already in the page cache. The kernel just points the page-table entry at it. Roughly a microsecond.</li>
<li><b>Major fault</b>: the page must be read from disk first. That costs a storage read, tens to hundreds of microseconds on an SSD.</li></ul>
<p>Counting faults is plain arithmetic. Touch one byte in each page of a 1 GiB file that is already cached:</p>
<div class="eq">1 GiB ÷ 4 KiB = 262,144 pages → 262,144 minor faults?

Linux "fault-around" maps the aligned 64 KiB group
(16 pages) around each faulting page:
262,144 ÷ 16 = 16,384 faults</div>
<p>A second pass over the same mapping takes no faults at all. <code>examples/02_first_touch.c</code> reads the counters from <code>getrusage()</code> so you can see each case. To see major faults on Linux you must first empty the page cache, which needs root: <code>sync; echo 3 &gt; /proc/sys/vm/drop_caches</code>.</p>`,
        check: { q: "One byte per 4 KiB page over a 1 MiB cached file, with 16-page fault-around. How many faults?",
          options: ["256", "16", "1", "4,096"], answer: 1,
          why: "1 MiB is 256 pages. Each fault maps its whole aligned group of 16 pages, so only the first touch in each group faults: 256 ÷ 16 = 16. The order of the touches doesn't change that. Exercise 2 tests exactly this." },
        scene(G) {
          G.box(24, 60, 150, 70, "disk / SSD", { fill: "w", size: 13 });
          G.box(244, 60, 170, 70, "page cache (RAM)", { fill: "k", size: 13 });
          G.box(470, 60, 150, 70, "your mapping", { stroke: "q", color: "q", size: 13 });
          const maj = G.arrow(176, 95, 240, 95, { color: "hot", w: 3 });
          const min = G.arrow(416, 95, 466, 95, { color: "ok", w: 3 });
          G.text(208, 156, "major fault", { anchor: "middle", size: 12, color: "hot" }); G.text(208, 172, "read from disk", { anchor: "middle", size: 11, color: "muted" });
          G.text(441, 156, "minor fault", { anchor: "middle", size: 12, color: "ok" }); G.text(441, 172, "just map it", { anchor: "middle", size: 11, color: "muted" });
          G.text(24, 224, "1 MiB = 256 pages, fault-around = 16 pages", { size: 13 });
          const g = G.grid(24, 240, 4, 64, 9, 22, (r, c) => ((r * 64 + c) % 16 === 0 ? "hot" : "ok"), { gap: 1, rx: 1, opacity: 0.9 });
          G.label(24, 350, "red = a page that faults · green = mapped by fault-around", { size: 12 });
          G.text(24, 384, "256 pages ÷ 16 per group = 16 faults", { size: 14, color: "ink" });
          G.pulse(maj, { repeat: 3 }); G.from(g, { opacity: 0, stagger: 0.002, duration: 0.1 });
          G.caption("one fault maps a whole aligned group of pages");
        } },

      { rail: "mmap vs read", title: "mmap vs read: zero copies, one shared copy",
        body: `<p>The classic way to load a file is <code>read()</code> into a buffer you allocated. The kernel first brings the data into the page cache, then <b>copies</b> it into your buffer. The data now sits in RAM twice.</p>
<p>With <code>mmap</code>, your pointers point straight at the page-cache pages. No copy, and no second buffer.</p>
<table><tr><th></th><th><code>read()</code></th><th><code>mmap()</code></th></tr>
<tr><td>copies</td><td>page cache → your buffer</td><td>none</td></tr>
<tr><td>RAM used</td><td>buffer + page cache</td><td>page cache only</td></tr>
<tr><td>data arrives</td><td>at the call</td><td>on first touch</td></tr></table>
<p>The bonus: two processes that map the same model file read-only share <b>one</b> physical copy, because both page tables point at the same page-cache frames. That is how llama.cpp lets several processes use one model without duplicating 16 GB. <code>bench/mmap_bench.c</code> compares <code>read()</code>, <code>mmap</code> + touch, and <code>mmap</code> + <code>MAP_POPULATE</code> (which pre-faults the whole range up front).</p>`,
        scene(G) {
          G.text(24, 40, "read(): two copies in RAM", { size: 13, color: "hot" });
          G.box(24, 56, 170, 56, "page cache", { fill: "k", size: 13 });
          G.box(260, 56, 170, 56, "your buffer", { fill: "hot", size: 13 });
          const cp = G.arrow(196, 84, 256, 84, { color: "hot", w: 3 }); G.label(200, 128, "memcpy by the kernel", { size: 11 });
          G.text(24, 196, "mmap(): two processes, one copy", { size: 13, color: "ok" });
          G.box(24, 216, 150, 50, "process A", { stroke: "q", color: "q", size: 13 });
          G.box(24, 300, 150, 50, "process B", { stroke: "q", color: "q", size: 13 });
          G.box(320, 248, 220, 70, "page cache: model file", { fill: "k", size: 13 });
          const a = G.arrow(176, 241, 316, 270, { color: "q", w: 2 }); const b = G.arrow(176, 325, 316, 300, { color: "q", w: 2 });
          G.label(24, 396, "both page tables point at the same frames", { size: 13, color: "ink" });
          G.from([a, b], { opacity: 0, stagger: 0.3, duration: 0.4 }); G.pulse(cp, { repeat: 3 });
          G.caption("mmap hands out pointers into the page cache itself");
        } },

      { rail: "pinned · DMA", title: "Pinned memory: pages that promise not to move",
        body: `<p>The OS is free to move a page around: write it out to swap, migrate it, or remap it, as long as it fixes the page table. Your program never notices, because it only sees virtual addresses. Such memory is called <b>pageable</b>.</p>
<p>A <b>DMA engine</b> (direct memory access: the GPU's copy engine, an NVMe or network card) is different. It copies between devices and RAM on its own, using <i>physical</i> addresses, without the CPU. If the OS moved a page mid-transfer, the device would read the wrong bytes. So DMA needs <b>pinned</b> (page-locked) memory: pages the OS promises to keep in place.</p>
<p>That is why copying from ordinary memory to a GPU is slower than it looks. The driver first copies your data into its own pinned <b>staging buffer</b> with the CPU, then lets the DMA engine take it from there: an extra copy, and the transfer can't fully overlap with other work. Allocating with <code>cudaMallocHost</code> (or pinning with <code>cudaHostRegister</code>) removes the extra copy. On a laptop without a GPU, <code>mlock()</code> is the closest analogue (<code>examples/04_mlock_pinned.c</code>); P5.1 measures the real thing.</p>`,
        scene(G) {
          G.text(24, 40, "pageable source", { size: 13, color: "hot" });
          G.box(24, 56, 150, 56, "your buffer", { stroke: "hot", color: "hot", size: 13 });
          G.box(250, 56, 150, 56, "pinned staging", { fill: "k", size: 13 });
          G.box(470, 56, 150, 56, "GPU memory", { fill: "w", size: 13 });
          G.arrow(176, 84, 246, 84, { color: "hot", w: 3 }); G.label(176, 132, "CPU memcpy (extra)", { size: 11, color: "hot" });
          const d1 = G.arrow(402, 84, 466, 84, { color: "ok", w: 3 }); G.label(410, 132, "DMA", { size: 11, color: "ok" });
          G.text(24, 216, "pinned source", { size: 13, color: "ok" });
          G.box(24, 232, 220, 56, "cudaMallocHost buffer", { fill: "k", size: 13 });
          G.box(470, 232, 150, 56, "GPU memory", { fill: "w", size: 13 });
          const d2 = G.arrow(246, 260, 466, 260, { color: "ok", w: 3 }); G.label(330, 306, "DMA directly", { size: 11, color: "ok" });
          G.text(24, 368, "DMA reads physical addresses: pages must not move", { size: 13 });
          G.from([d1, d2], { opacity: 0, stagger: 0.3, duration: 0.4 });
          G.caption("pageable memory costs an extra staging copy before every DMA");
        } },

      { rail: "safetensors", title: "A file format built for mmap",
        body: `<p>Put it together: to load a model with zero copies, the file's bytes must already be exactly the bytes the program wants in memory. That is the design of <b>safetensors</b>:</p>
<div class="eq">[ 8 bytes: little-endian u64 N ]
[ N bytes: JSON header           ]
[ raw tensor bytes …             ]</div>
<p>The header maps each tensor name to its dtype, shape and byte range inside the data section, for example <code>{"dtype":"BF16", "shape":[128256,4096], "data_offsets":[b, e]}</code> for Llama-3-8B's embedding table. Check the arithmetic: 128,256 × 4,096 × 2 bytes = 1,050,673,152 bytes, so <code>e − b</code> must equal exactly that.</p>
<p>A loader maps the file once and hands out <code>base + 8 + N + b</code> as the tensor's pointer. Nothing is parsed or copied except the header; pages fault in as the forward pass first touches each layer.</p>
<p>But the header is untrusted input. A truncated download or a malicious file can claim offsets past the end of the file. Exercise 3 makes the loader check every field: header size, dtype, <code>begin ≤ end ≤ data size</code>, <code>end − begin = numel × dtype size</code> (with overflow-checked multiplication), no overlaps, and natural alignment. It becomes the loader of the P0.5 engine. (Format: the README of <code>huggingface/safetensors@e246a256</code>.)</p>`,
        check: { q: "A safetensors header claims a tensor's data_offsets are [0, 2⁴⁰] in a 1 MB file. What should the loader do?",
          options: ["Map it anyway: pages only fault when touched", "Reject the file before forming any pointer from those offsets", "Clamp the end offset to the file size"], answer: 1,
          why: "The end offset is past the data section, so any pointer into that range would read outside the mapping. Untrusted offsets must be checked against the file size, and the byte length against numel × dtype size, before use." },
        scene(G) {
          G.text(24, 40, "model.safetensors", { size: 13 });
          G.box(24, 56, 70, 56, "N", { fill: "v", size: 13 });
          G.box(96, 56, 150, 56, "JSON header", { fill: "q", size: 13 });
          const t = [["embed", 130], ["layer 0 …", 140], ["lm_head", 100]];
          let x = 248; const segs = [];
          t.forEach(([n, w], i) => { segs.push(G.box(x, 56, w - 4, 56, n, { fill: i === 0 ? "k" : "w", size: 12 })); x += w; });
          G.label(24, 130, "8 B", { size: 11 }); G.label(96, 130, "N bytes", { size: 11 }); G.label(420, 130, "raw data, little-endian", { size: 11 });
          G.path("M 170 114 C 170 170, 300 170, 300 116", { color: "q", dash: "4 4", arrow: true });
          G.label(140, 184, "\"data_offsets\": [b, e]", { size: 12, color: "q" });
          G.text(24, 240, "ptr = base + 8 + N + b", { size: 15, color: "ok" });
          G.text(24, 270, "no parse, no copy: pages fault in on first use", { size: 13 });
          G.text(24, 320, "validate before trusting:", { size: 13, color: "hot" });
          G.label(24, 344, "dtype known · b ≤ e ≤ data size · e − b = numel × dtype size", { size: 12, color: "ink" });
          G.label(24, 366, "no overlaps · offset aligned to dtype size", { size: 12, color: "ink" });
          G.from(segs, { opacity: 0, x: -12, stagger: 0.12, duration: 0.3 });
          G.caption("the file's bytes are the tensors' bytes: one mmap loads the model");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>Programs use virtual addresses; the MMU translates each page through per-process page tables, a 4-level tree on x86-64.</li>
<li>A full translation is four dependent reads, so the TLB caches translations. Its reach is entries × page size, and huge pages raise it 512×.</li>
<li><code>mmap</code> only creates not-present entries. Pages arrive by page fault on first touch: minor from the page cache, major from disk. Fault-around maps groups of pages at once.</li>
<li><code>mmap</code> avoids the copy that <code>read()</code> makes and lets processes share one copy. DMA needs pinned pages, so pageable transfers pay a staging copy.</li>
<li>safetensors is laid out so a model loads with one <code>mmap</code>, as long as you validate the header.</li></ul>
<p>Next, P0.3 puts several threads on those pages at once, and shows how caches make them interfere.</p>`,
    sim: {
      title: "TLB reach and page faults",
      intro: "Pick how much memory a program touches, the page size and the TLB size. The chart shows the fraction of random accesses that miss the TLB as the working set grows, for each page size; the dot is your configuration. The panels count pages, TLB reach and the faults needed to touch every page once.",
      height: 280,
      controls: [
        { id: "ws", label: "working set (log₂ MiB: 0 = 1 MiB … 15 = 32 GiB)", min: 0, max: 15, step: 1, value: 14, format: (v) => F.bytes(2 ** v * MiB) },
        { id: "pg", label: "page size", type: "select", value: 4096, options: [[4096, "4 KiB (x86-64, Linux arm64)"], [16384, "16 KiB (Apple silicon)"], [2097152, "2 MiB (huge page)"]] },
        { id: "e", label: "TLB entries (example value: check your CPU)", min: 64, max: 4096, step: 64, value: 1536 },
        { id: "fa", label: "fault-around (pages mapped per fault)", type: "select", value: 16, options: [[1, "1 (off)"], [4, "4"], [16, "16 (Linux file default)"]] },
      ],
      draw(G, v) {
        const ws = 2 ** v.ws * MiB, pg = v.pg, E = v.e;
        const miss = (w, p) => Math.max(0, 1 - (E * p) / w);
        const X = (lg) => 50 + (lg / 15) * 560, Y = (m) => 230 - m * 190;
        G.axes(50, 40, 570, 190, { ylabel: "TLB miss rate, random access" });
        G.label(620, 272, "working set (log scale) →", { anchor: "end", size: 11 });
        [0, 0.5, 1].forEach((m) => { G.label(44, Y(m) + 4, `${m * 100}%`, { anchor: "end", size: 10 }); G.line(50, Y(m), 620, Y(m), { color: "line", dash: "2 4", w: 1 }); });
        [0, 5, 10, 15].forEach((lg) => G.label(X(lg), 248, F.bytes(2 ** lg * MiB), { anchor: "middle", size: 10 }));
        const curves = [[4096, "hot", "4 KiB"], [16384, "v", "16 KiB"], [2097152, "ok", "2 MiB"]];
        curves.forEach(([p, col, name], ci) => {
          let d = "";
          for (let s = 0; s <= 150; s++) { const lg = (s / 150) * 15; d += (s ? " L " : "M ") + X(lg).toFixed(1) + " " + Y(miss(2 ** lg * MiB, p)).toFixed(1); }
          G.path(d, { color: col, w: p === pg ? 3 : 1.5, opacity: p === pg ? 1 : 0.6 });
          G.rect(50 + ci * 90, 263, 10, 10, { fill: col, rx: 2 }); G.label(64 + ci * 90, 272, name, { size: 11 });
        });
        G.circle(X(v.ws), Y(miss(ws, pg)), 6, { fill: "ink" });
        const pages = Math.ceil(ws / pg), reach = E * pg, m = miss(ws, pg);
        const faults = Math.ceil(pages / Math.max(1, v.fa));
        return [
          { title: "Your configuration", rows: [["working set", F.bytes(ws)], ["pages", F.num(pages)], ["TLB reach", F.bytes(reach)], ["random-access miss rate", (100 * m).toFixed(1) + "%"]],
            gauge: [[Math.min(1, reach / ws), "ok"], [Math.max(0, 1 - reach / ws), "hot"]], gaugeText: "share of the working set covered by the TLB", chip: [reach >= ws, reach >= ws ? "fits in TLB reach" : "exceeds TLB reach"] },
          { title: "First touch of every page", rows: [["faults without fault-around", F.num(pages)], ["faults with fault-around", F.num(faults)], ["pages vs 2 MiB pages", F.num(pages) + " vs " + F.num(Math.ceil(ws / (2 * MiB)))]] },
          { title: "Model", html: `<p class="note">Uniform random access with a fully associative TLB: the chance that the next page is cached is entries ÷ pages. Sequential access misses only once per page. Real CPUs have several TLB levels and page-walk caches, so measure with <code>examples/01_page_walk.py</code> and <code>03_hugepages.c</code>. Fault-around applies to file mappings; huge pages are not used for ordinary file mappings on most systems.</p>` },
        ];
      },
    },
    practice: {
      intro: `Exercises 1 and 2 are Python: run them from the repository root. Exercises 3 and 4 are C++: run their commands from the module folder <code>course/P0-systems-primer/P0.2-virtual-memory-and-mmap</code>. <code>S2S_SOLUTIONS=1</code> (Python) or <code>-DS2S_USE_SOLUTIONS=ON</code> (CMake) runs the same tests against the reference solutions.`,
      items: [
        { title: "TLB reach calculator", tier: "T0 · easy", goal: "Write tlb_reach, pages_needed (rounding up), fits_in_tlb and walks_saved_by_hugepages, then look up your own CPU's L2 TLB size.",
          cmd: "uv run pytest course/P0-systems-primer/P0.2-virtual-memory-and-mmap/exercises/01-tlb-reach" },
        { title: "Predict page faults", tier: "T0 · easy", goal: "Count first-touch faults for an address stream with optional fault-around. Predict the three test cases before running.",
          cmd: "uv run pytest course/P0-systems-primer/P0.2-virtual-memory-and-mmap/exercises/02-predict-faults" },
        { title: "Harden the safetensors loader", tier: "T0 · medium", goal: "Reject six kinds of malformed file (header size, dtype, ranges, length, overlap, alignment) and count parameters of a valid one.",
          cmd: "uv run python exercises/03-safetensors-loader/make_test_files.py build/st-test && cmake -S exercises -B build/ex && cmake --build build/ex -j && ctest --test-dir build/ex -R 03 --output-on-failure" },
        { title: "Prefetch with MADV_WILLNEED", tier: "T0 · hard", goal: "Map a file and ask the kernel to start reading it in the background; checksums must match read(), and the bench shows the cold-load gain.",
          cmd: "ctest --test-dir build/ex -R 04 --output-on-failure" },
      ],
      labs: [
        { label: "Examples: page-walk simulator, first touch, huge pages, mlock, safetensors mmap", path: "course/P0-systems-primer/P0.2-virtual-memory-and-mmap/examples/" },
        { label: "Benchmark: read() vs mmap vs MAP_POPULATE, cold and warm", path: "course/P0-systems-primer/P0.2-virtual-memory-and-mmap/bench/mmap_bench.c" },
        { label: "The loader the exercises harden", path: "course/P0-systems-primer/P0.2-virtual-memory-and-mmap/examples/safetensors_view.hpp" },
        { label: "Animation: VA to page table to TLB to frame, with first-touch faults", path: "animations/p0-vm-page-walk.html" },
        { label: "Self-check questions", path: "course/P0-systems-primer/P0.2-virtual-memory-and-mmap/quiz.md" },
      ],
    },
  });
})();
