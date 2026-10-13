/* R.3 — The terminal, git and building code. A Readiness lesson: teaches and self-checks. */
(function () {
  // The project used by step 7 and the simulator: 2 headers, 4 sources, 4 object files, 1 executable.
  const FILES = [["util.hpp", "h"], ["tensor.hpp", "h"], ["main.cpp", "s"], ["load.cpp", "s"], ["kernel.cpp", "s"], ["log.cpp", "s"]];
  const OBJS = [["main.o", 2, [0]], ["load.o", 3, [0, 1]], ["kernel.o", 4, [1]], ["log.o", 5, []]]; // [name, source row, header rows it includes]

  // Draw the dependency graph. changed: array of 6 booleans. stage: 0 = just edited, 1 = compiling, 2 = linked.
  function graph(G, changed, stage, o) {
    const y0 = o.y0, step = o.step, h = o.h;
    const fy = (i) => y0 + i * step;
    const objChanged = OBJS.map(([, s, hs]) => changed[s] || hs.some((x) => changed[x]));
    const anyObj = objChanged.some(Boolean);
    G.label(16, y0 - 12, "source files", { size: 11 });
    G.label(250, y0 - 12, "object files", { size: 11 });
    G.label(484, y0 - 12, "executable", { size: 11 });
    // edges first so boxes sit on top
    OBJS.forEach(([, s, hs], k) => {
      const oy = fy(s) + h / 2;
      [s, ...hs].forEach((src) => {
        G.line(166, fy(src) + h / 2, 250, oy, { color: changed[src] ? "v" : "line", w: changed[src] ? 2 : 1 });
      });
      G.line(380, oy, 484, fy(3) + h / 2 + 8, { color: stage >= 1 && objChanged[k] ? "hot" : "line", w: stage >= 1 && objChanged[k] ? 2 : 1, arrow: true });
    });
    FILES.forEach(([n, kind], i) => {
      G.box(16, fy(i), 150, h, n, { fill: changed[i] ? "v" : null, stroke: changed[i] ? "v" : "line", size: 12, dash: kind === "h" ? "4 3" : undefined });
    });
    OBJS.forEach(([n], k) => {
      const hot = stage >= 1 && objChanged[k];
      G.box(250, fy(OBJS[k][1]), 130, h, n, { fill: hot ? "hot" : "ok", size: 12 });
    });
    const link = stage >= 2 && anyObj;
    G.box(484, fy(3) + 8, 140, h + 4, "app", { fill: link ? "hot" : anyObj ? null : "ok", stroke: anyObj ? "hot" : "line", size: 13, dash: anyObj && !link ? "4 3" : undefined });
    return { objChanged, compiles: objChanged.filter(Boolean).length, link: anyObj };
  }

  const none = [false, false, false, false, false, false];

  S2S.lesson({
    id: "r-3", n: "R.3", title: "The terminal, git and building code",
    subtitle: "Readiness · first principles · T0, any laptop",
    kicker: "Lesson · ≈ 35 min",
    headline: "Every command in this course is typed here",
    intro: `<p><b>Already comfortable with the shell, git and CMake? Answer the checkpoints. If you get them all right, skip to R.4.</b></p>
<p>Building a kernel, running a benchmark, starting a server and tearing down a cloud stack all happen in a terminal. When one fails, the exit code and the error text are your only clues. This lesson teaches the ten pieces you need: the shell, paths, pipes, processes and environment variables, permissions, git, build systems, CMake, the course's Python tools, and how you will reach a cloud machine. Scroll: the dark panel redraws for each step.</p>`,
    facts: ["10 steps", "5 checkpoints", "1 simulator", "3 exercises"],
    legend: [["k", "program"], ["v", "input / edited"], ["ok", "ok / up to date"], ["hot", "failure / rebuilt"]],
    prev: "r-2", next: "r-4",
    steps: [
      { rail: "the shell", title: "The shell runs programs and gets an exit code back",
        body: `<p>A <b>shell</b> (bash, zsh) reads one line, splits it into words, and starts the first word as a program. The other words are its <b>arguments</b>.</p>
<div class="eq">ls -l course
^^ program   ^^ ^^^^^^ arguments</div>
<p>When a program ends it hands back an <b>exit code</b>, a number from 0 to 255. By convention <b>0 means success</b> and anything else means failure. The shell keeps the last one in <code>$?</code>.</p>
<div class="eq">$ ls course
$ echo $?
0
$ ls no-such-folder
ls: cannot access 'no-such-folder'
$ echo $?
2</div>
<p><code>&amp;&amp;</code> runs the next command only if the last one returned 0. CTest, pytest and make all report pass or fail this way.</p>`,
        check: { q: "You run <code>cmake --build build</code> and then <code>echo $?</code> prints 1. What happened?",
          options: ["The build succeeded", "The build failed", "The build is still running"], answer: 1,
          why: "Exit code 0 is success. Any other number means the program reported a failure. The error text above it tells you why; the code only tells you that." },
        scene(G) {
          G.label(24, 40, "you type one line", { size: 13 });
          G.box(24, 56, 70, 38, "ls", { fill: "k", size: 14 });
          G.box(104, 56, 70, 38, "-l", { fill: "v", size: 14 });
          G.box(184, 56, 110, 38, "course", { fill: "v", size: 14 });
          G.label(24, 114, "program", { size: 12 }); G.label(104, 114, "argument", { size: 12 }); G.label(184, 114, "argument", { size: 12 });
          G.arrow(160, 124, 160, 168, { color: "ink", w: 2 });
          G.box(24, 170, 270, 46, "the shell starts a process", { stroke: "ink", size: 13 });
          G.arrow(100, 218, 100, 262, { color: "ok", w: 2 }); G.arrow(220, 218, 440, 262, { color: "hot", w: 2 });
          const a = G.box(24, 264, 250, 52, "exit code 0 = success", { fill: "ok", size: 14 });
          const b = G.box(350, 264, 270, 52, "exit code 2 = failure", { fill: "hot", size: 14 });
          G.text(24, 346, "echo $?     → shows the last exit code", { size: 13 });
          G.text(24, 372, "make && ./app    runs ./app only after exit code 0", { size: 13 });
          G.from([a, b], { opacity: 0, y: 10, stagger: 0.2, duration: 0.4 });
          G.caption("the exit code is the program's one-number verdict");
        } },

      { rail: "paths", title: "Paths say where a file is: absolute or relative",
        body: `<p>Files live in a tree of folders. An <b>absolute path</b> starts at the root <code>/</code> and works from anywhere. A <b>relative path</b> starts at the folder you are in (the <i>current directory</i>, printed by <code>pwd</code>).</p>
<p><code>.</code> means "this folder", <code>..</code> the parent, <code>~</code> your home folder.</p>
<table><tr><th>command</th><th>does</th></tr>
<tr><td><code>cd course</code></td><td>go into a folder</td></tr>
<tr><td><code>ls -l</code></td><td>list files with sizes</td></tr>
<tr><td><code>cat f</code> / <code>less f</code></td><td>print a file / page through it (<code>q</code> quits)</td></tr>
<tr><td><code>grep -rn "word" .</code></td><td>search inside files, recursively</td></tr>
<tr><td><code>find . -name "*.md"</code></td><td>search by file name</td></tr></table>`,
        scene(G) {
          const nodes = [["/", 40, 30], ["home", 40, 84], ["you", 40, 138], ["repo", 40, 192]];
          nodes.forEach(([n, x, y], i) => { if (i) G.line(85, y - 20, 85, y, { color: "line", w: 2 }); });
          nodes.forEach(([n, x, y], i) => G.box(x, y, 90, 34, n, { fill: i === 3 ? "v" : null, stroke: "ink", size: 13 }));
          G.line(130, 209, 200, 209, { color: "line", w: 2 }); G.line(130, 209, 200, 263, { color: "line", w: 2 });
          G.box(200, 192, 100, 34, "README.md", { stroke: "ink", size: 12 });
          G.box(200, 246, 100, 34, "course", { stroke: "ink", size: 13 });
          G.line(250, 280, 250, 300, { color: "line", w: 2 });
          G.box(190, 300, 120, 34, "R-readiness", { stroke: "ink", size: 12 });
          G.text(332, 214, "← pwd: you are here", { size: 13, color: "v" });
          G.text(24, 372, "absolute:  /home/you/repo/course/R-readiness", { size: 13 });
          G.text(24, 398, "relative:  course/R-readiness      (from repo)", { size: 13, color: "ok" });
          G.caption("an absolute path never changes; a relative one depends on where you stand");
        } },

      { rail: "pipes", title: "Pipes join small programs; redirection sends output to files",
        body: `<p>Every program has three streams: <b>stdin</b> (input), <b>stdout</b> (normal output, stream 1) and <b>stderr</b> (error messages, stream 2).</p>
<ul><li><code>a | b</code>: a's stdout becomes b's stdin.</li>
<li><code>a &gt; f</code>: write stdout to file f (overwrite). <code>&gt;&gt;</code> appends.</li>
<li><code>2&gt;&amp;1</code>: send stderr to the same place as stdout.</li></ul>
<div class="eq">grep -rF "TODO(run" course | wc -l | sort</div>
<p>Pipes carry only stdout. Error messages skip the pipe and reach your screen, unless you redirect them.</p>`,
        check: { q: "<code>./build/app &gt; out.txt</code> runs and prints an error message on your screen, not in out.txt. Why?",
          options: ["Errors travel on stderr, and only stdout was redirected", "out.txt is full", "Errors are never written to the terminal"], answer: 0,
          why: "The > redirects stream 1 only. Add 2>&1 to send stream 2 (stderr) to the file as well." },
        scene(G) {
          const p = [["grep -r", 16], ["wc -l", 245], ["sort", 474]];
          const bx = p.map(([n, x]) => G.box(x, 60, 150, 44, n, { fill: "k", size: 14 }));
          G.arrow(166, 82, 245, 82, { color: "ink", w: 2 }); G.arrow(395, 82, 474, 82, { color: "ink", w: 2 });
          G.label(172, 74, "stdout", { size: 10 }); G.label(401, 74, "stdout", { size: 10 });
          G.box(520, 170, 104, 40, "out.txt", { stroke: "ok", color: "ok", size: 13 });
          G.arrow(580, 104, 580, 170, { color: "ok", w: 2 }); G.label(526, 142, "> out.txt", { size: 11, color: "ok" });
          G.box(16, 288, 608, 44, "terminal (your screen)", { stroke: "hot", color: "hot", size: 13 });
          [[91, 104], [320, 104], [500, 104]].forEach(([x, y]) => G.arrow(x, y, x, 288, { color: "hot", w: 2, dash: "4 3" }));
          G.label(104, 200, "stderr skips the pipes", { size: 12, color: "hot" });
          G.text(24, 372, "cmd > out.txt 2>&1     both streams into the file", { size: 13 });
          G.from(bx, { opacity: 0, x: -20, stagger: 0.18, duration: 0.4 });
          G.caption("three processes, two pipes; errors take their own path");
        } },

      { rail: "processes", title: "Processes inherit environment variables",
        body: `<p>A running program is a <b>process</b> with an id (PID). <code>ps aux</code> lists them, <code>top</code> shows what is busy, <code>kill PID</code> asks one to stop (<code>kill -9</code> forces it), <code>Ctrl-C</code> stops the foreground one.</p>
<p>Each process carries <b>environment variables</b>: named strings that its children inherit as a copy.</p>
<div class="eq">export S2S_SOLUTIONS=1       # this shell
S2S_SOLUTIONS=1 uv run pytest   # one command</div>
<p>The course's test helper <code>load_impl</code> (<code>course/common/python/s2s/exercise.py</code>) reads <code>S2S_SOLUTIONS</code>. If it is <code>1</code>, a test loads the reference solution; otherwise it loads your starter. One variable switches between "check my work" and "check the answer key".</p>`,
        scene(G) {
          const sh = G.box(24, 40, 230, 110, "", { stroke: "ink" });
          G.text(36, 66, "your shell", { size: 14, weight: 700 }); G.label(36, 86, "PID 4182", { size: 11 });
          G.box(36, 98, 206, 36, "S2S_SOLUTIONS=1", { fill: "v", size: 13 });
          G.arrow(254, 95, 360, 80, { color: "ink", w: 2 }); G.label(268, 72, "starts", { size: 11 });
          G.box(360, 40, 260, 84, "", { stroke: "ink" });
          G.text(372, 66, "uv run pytest ...", { size: 14, weight: 700 }); G.label(372, 86, "PID 5120 (child)", { size: 11 });
          G.box(372, 94, 236, 24, "S2S_SOLUTIONS=1 (copy)", { fill: "v", size: 12 });
          G.text(24, 196, "load_impl reads the variable:", { size: 14 });
          const a = G.box(24, 214, 280, 74, "", { stroke: "ok" }); G.text(36, 242, "= 1  → exercises/solutions/", { size: 13, color: "ok" }); G.label(36, 268, "the reference answer", { size: 12 });
          const b = G.box(340, 214, 280, 74, "", { stroke: "k" }); G.text(352, 242, "unset → exercises/01-name/", { size: 13, color: "k" }); G.label(352, 268, "your starter code", { size: 12 });
          G.text(24, 340, "a child gets a copy: changing it there does not", { size: 13 }); G.text(24, 362, "change the parent shell.", { size: 13 });
          G.from([a, b], { opacity: 0, y: 10, stagger: 0.2, duration: 0.4 });
          G.caption("an environment variable travels from parent to child");
        } },

      { rail: "permissions", title: "Permissions, and when you need sudo",
        body: `<p>Every file has an owner and three permission sets: <b>r</b>ead, <b>w</b>rite, e<b>x</b>ecute for the owner, the group and everyone else. <code>ls -l</code> shows them as <code>-rwxr-xr-x</code>. A script needs the x bit to run directly (<code>chmod +x run.sh</code>); otherwise run it as <code>bash run.sh</code>.</p>
<p>Normal users cannot read the CPU's or GPU's low-level counters, or change system settings. <code>sudo cmd</code> runs <i>one command</i> as the administrator. Profilers such as <code>perf</code> and Nsight Compute (<code>ncu</code>) sometimes need it because they read counters the kernel protects.</p>
<p>Use <code>sudo</code> only for the single command that needs it. Never <code>sudo cmake</code> or <code>sudo pip</code>: root-owned files break later builds.</p>`,
        scene(G) {
          [["-", "file", 24, 40], ["rwx", "owner", 84, 90], ["r-x", "group", 184, 90], ["r-x", "everyone", 284, 90]].forEach(([s, l, x, w]) => {
            G.box(x, 50, w, 40, s, { fill: l === "owner" ? "ok" : l === "file" ? null : "k", stroke: "ink", size: 16 });
            G.label(x + 4, 110, l, { size: 11 });
          });
          G.text(24, 150, "chmod +x run.sh   adds x so it can run", { size: 13 });
          G.box(24, 200, 330, 46, "perf stat ./app   (normal user)", { stroke: "ink", size: 13 });
          G.arrow(100, 248, 100, 296, { color: "hot", w: 2 }); G.text(112, 278, "denied", { color: "hot", size: 13 });
          G.box(24, 298, 600, 56, "kernel: protected hardware counters", { stroke: "hot", color: "hot", size: 13 });
          G.box(390, 200, 234, 46, "sudo perf stat ./app", { fill: "v", size: 13 });
          G.arrow(500, 248, 500, 296, { color: "ok", w: 2 }); G.text(512, 278, "allowed", { color: "ok", size: 13 });
          G.caption("sudo lifts the lock for one command only");
        } },

      { rail: "git", title: "git saves snapshots in four places",
        body: `<p>Git keeps your project's history as <b>commits</b>: snapshots with a message. Your edits move through places:</p>
<div class="eq">working tree --add--> staging --commit--> repo
repo --push--> remote</div>
<p>The <b>working tree</b> is the files you edit. The <b>staging area</b> is the list of changes the next commit will hold. A <b>branch</b> is a movable label on a commit; <code>main</code> is the usual trunk.</p>
<div class="eq">git clone URL        copy a repo
git status           what changed
git add f            stage f
git commit -m "msg"  save a commit
git log --oneline --graph
git diff             unstaged changes
git switch -c name   new branch (or: checkout -b)
git push             send commits to the remote</div>`,
        check: { q: "You edited <code>a.py</code> and ran <code>git add a.py</code>, nothing else. Is the change saved in history?",
          options: ["Yes, add saves it", "No, it is only staged until you commit", "Yes, but only on the remote"], answer: 1,
          why: "git add copies the change into the staging area, the draft of the next commit. Only git commit writes it into history, and only git push sends it to the remote." },
        scene(G) {
          const names = [["working tree", 16], ["staging area", 176], ["repository", 336], ["remote", 496]];
          names.forEach(([n, x], i) => G.box(x, 70, 128, 50, n, { fill: i === 0 ? "v" : i === 3 ? "w" : "k", size: 13 }));
          [["git add", 144], ["git commit", 304], ["git push", 464]].forEach(([n, x]) => { G.arrow(x, 95, x + 32, 95, { color: "ink", w: 2 }); G.label(x - 20, 62, n, { size: 11 }); });
          G.label(16, 180, "inside the repository: commits are a chain", { size: 13 });
          const pts = [[60, 250], [150, 250], [240, 250]];
          G.line(60, 250, 240, 250, { color: "ink", w: 2 }); G.line(150, 250, 240, 330, { color: "ink", w: 2 });
          pts.forEach(([x, y]) => G.circle(x, y, 14, { fill: "k" }));
          G.circle(240, 330, 14, { fill: "q" });
          G.box(262, 232, 64, 32, "main", { fill: "ok", size: 12 });
          G.box(262, 314, 90, 32, "feature", { fill: "q", size: 12 });
          G.label(40, 290, "older → newer", { size: 11 });
          G.text(24, 392, "a branch is a label on a commit, not a copy", { size: 13 });
          G.caption("add stages, commit saves, push shares");
        } },

      { rail: "build systems", title: "A build system rebuilds only what changed",
        body: `<p>From R.1: each <code>.cpp</code> is compiled to an <b>object file</b>, then the <b>linker</b> joins objects into the executable. In a big project you must not recompile everything after a one-line edit.</p>
<p>A build system stores the <b>dependency graph</b>: an object depends on its <code>.cpp</code> and every header it includes; the executable depends on all objects. After an edit it rebuilds only what lies downstream.</p>
<p><b>Make</b> reads rules you write by hand. <b>CMake</b> is a generator: you describe targets and it writes the Make or Ninja files.</p>
<p>In the picture, <code>util.hpp</code> changed. <code>main.cpp</code> and <code>load.cpp</code> include it, so two objects rebuild and the link runs again. The other two stay green.</p>`,
        check: { q: "In the project shown, only <code>log.cpp</code> (which includes no headers) is edited. How many compile steps run?",
          options: ["0", "1 compile, then the link", "4 compiles, then the link"], answer: 1,
          why: "Only log.o depends on log.cpp, so one compile runs. The executable depends on log.o, so it is re-linked. The other three objects are up to date." },
        scene(G) {
          graph(G, [true, false, false, false, false, false], 2, { y0: 52, step: 56, h: 40 });
          G.text(24, 408, "util.hpp edited → main.o, load.o recompile → link", { size: 13 });
          G.caption("red = rebuilt, green = reused");
        } },

      { rail: "cmake", title: "CMake in practice: configure, build, test",
        body: `<p>You describe <b>targets</b> in <code>CMakeLists.txt</code>:</p>
<div class="eq">add_library(mathlib mathlib.cpp)
target_include_directories(mathlib PUBLIC .)
add_executable(app main.cpp)
target_link_libraries(app PRIVATE mathlib)
enable_testing()
add_test(NAME mathlib COMMAND test_mathlib)</div>
<p>Then three commands:</p>
<div class="eq">cmake -S . -B build        # configure
cmake --build build -j      # compile + link
ctest --test-dir build --output-on-failure</div>
<p><code>-S</code> is the source folder, <code>-B</code> the build folder (keep it outside git). <code>Release</code> optimises (<code>-O3</code>): use it for timing. <code>Debug</code> keeps symbols and skips optimisation: use it for debuggers and sanitizers. Forgetting <code>target_link_libraries</code> gives a linker error, <code>undefined reference</code>. Exercise 1 makes you meet it.</p>`,
        scene(G) {
          const rows = [
            ["CMakeLists.txt", "cmake -S . -B build", "build/ (Makefiles)"],
            ["sources + build/", "cmake --build build -j", "app, test_mathlib"],
            ["test_mathlib", "ctest --test-dir build", "exit 0 = pass"],
          ];
          rows.forEach(([a, c, o], i) => {
            const y = 40 + i * 100;
            G.box(16, y, 150, 56, a, { stroke: "ink", size: 12 });
            G.box(212, y, 210, 56, c, { fill: "k", size: 12 });
            G.box(474, y, 150, 56, o, { stroke: i === 2 ? "ok" : "ink", color: i === 2 ? "ok" : undefined, size: 12 });
            G.arrow(166, y + 28, 212, y + 28, { color: "ink", w: 2 }); G.arrow(422, y + 28, 474, y + 28, { color: "ink", w: 2 });
          });
          G.label(24, 350, "-S source folder    -B build folder", { size: 12 });
          G.text(24, 380, "Release: -O3 (timing)    Debug: -g, no optimisation", { size: 13 });
          G.caption("configure once, build often, test every time");
        } },

      { rail: "python tools", title: "The course's Python tools: uv and pytest",
        body: `<p>The course pins every Python dependency in <code>pyproject.toml</code> and <code>uv.lock</code>. <b>uv</b> is the tool that installs them and runs commands inside that environment.</p>
<div class="eq">uv sync                  # once: create .venv
uv run pytest &lt;path&gt;     # your starters
S2S_SOLUTIONS=1 uv run pytest &lt;path&gt;   # answers</div>
<p><code>uv run</code> means "run this inside the project's environment", so you never activate anything by hand. Without <code>S2S_SOLUTIONS</code>, your starters fail with <code>NotImplementedError</code> until you write them. That is the starting line, not a broken install.</p>
<p>The three local checks the course uses: <code>python tools/check_links.py</code>, <code>bash tools/ci_terraform.sh</code> and <code>S2S_SOLUTIONS=1 uv run pytest</code>.</p>`,
        scene(G) {
          G.box(16, 50, 190, 56, "pyproject.toml + uv.lock", { stroke: "ink", size: 11 });
          G.arrow(206, 78, 262, 78, { color: "ink", w: 2 });
          G.box(262, 50, 140, 56, "uv sync", { fill: "k", size: 14 });
          G.arrow(402, 78, 454, 78, { color: "ink", w: 2 });
          G.box(454, 50, 170, 56, ".venv/ (packages)", { stroke: "ok", color: "ok", size: 12 });
          G.box(16, 170, 250, 50, "uv run pytest path", { fill: "k", size: 14 });
          G.arrow(266, 195, 330, 195, { color: "ink", w: 2 });
          G.box(330, 170, 294, 50, "test_x.py → load_impl(...)", { stroke: "ink", size: 13 });
          G.arrow(420, 220, 200, 290, { color: "ink", w: 2 }); G.arrow(520, 220, 500, 290, { color: "ink", w: 2 });
          G.box(24, 290, 250, 50, "S2S_SOLUTIONS=1 → solutions/", { fill: "ok", size: 12 });
          G.box(340, 290, 284, 50, "unset → your starter (TODO)", { fill: "hot", size: 12 });
          G.text(24, 388, "starter fails with NotImplementedError: expected", { size: 13 });
          G.caption("same test, two implementations, one variable");
        } },

      { rail: "ssh vs ssm", title: "Reaching a cloud machine: SSH opens a port, SSM does not",
        body: `<p>To work on a rented machine you need a shell on it. <b>SSH</b> listens on a network <b>port</b> (22) and lets in anyone holding a key. That port is a door that must stay open, and the internet probes open doors all day.</p>
<p><b>Session Manager (SSM)</b> works the other way round. A small agent on the machine <i>dials out</i> to the cloud provider's API. You log in to the API with your own identity, and it relays your session. No inbound port is ever opened.</p>
<p>This course uses <b>SSM only</b> and never opens port 22. R.7 sets up the account that makes this work, and P2.1 is the first time you use it. You do not need to do anything with it yet.</p>`,
        check: { q: "Why does the course prefer SSM to SSH for its cloud machines?",
          options: ["SSM is faster", "No inbound port has to be open to the internet", "SSM needs no account"], answer: 1,
          why: "With SSM the instance connects out to the provider's API and you authenticate to that API. There is no open port 22 to probe or brute-force. Speed is not the reason, and you still need an account." },
        scene(G) {
          G.text(16, 36, "SSH", { size: 15, weight: 700 });
          G.box(16, 50, 100, 50, "laptop", { stroke: "ink", size: 13 });
          G.arrow(116, 75, 190, 75, { color: "hot", w: 2 });
          G.box(190, 50, 250, 50, "port 22 open to the internet", { fill: "hot", size: 12 });
          G.arrow(440, 75, 514, 75, { color: "hot", w: 2 });
          G.box(514, 50, 110, 50, "instance", { stroke: "ink", size: 13 });
          G.label(190, 122, "a door that stays open", { size: 12, color: "hot" });
          G.text(16, 216, "SSM", { size: 15, weight: 700 });
          G.box(16, 230, 100, 50, "laptop", { stroke: "ink", size: 13 });
          G.arrow(116, 255, 190, 255, { color: "ok", w: 2 });
          G.box(190, 230, 250, 50, "provider API (HTTPS + identity)", { fill: "ok", size: 12 });
          G.arrow(514, 255, 440, 255, { color: "ok", w: 2, dash: "4 3" });
          G.box(514, 230, 110, 50, "instance", { stroke: "ink", size: 13 });
          G.label(190, 302, "the agent dials out; no inbound port", { size: 12, color: "ok" });
          G.text(16, 380, "R.7 prepares the account; P2.1 is the first use", { size: 13 });
          G.caption("SSM: the machine calls out, nobody calls in");
        } },
    ],
    outro: `<h2>What you now know</h2><ul>
<li>A shell runs a program with arguments and gets an exit code back: 0 is success. <code>$?</code> shows it.</li>
<li>Pipes carry stdout between programs; <code>&gt;</code> and <code>2&gt;&amp;1</code> redirect streams. Children inherit environment variables such as <code>S2S_SOLUTIONS</code>.</li>
<li>git has a working tree, a staging area and a history of commits. A branch is a label on a commit.</li>
<li>A build system keeps a dependency graph and rebuilds only what is downstream of a change. CMake drives that with <code>-S</code>, <code>-B</code>, <code>--build</code> and <code>ctest</code>.</li>
<li><code>uv run pytest</code> runs tests in the pinned environment. The course uses SSM, never an open SSH port.</li></ul>
<h2>Go further</h2><ul>
<li>MIT, <i>The Missing Semester of Your CS Education</i>: <a href="https://missing.csail.mit.edu/">missing.csail.mit.edu</a></li>
<li><i>Pro Git</i>: <a href="https://git-scm.com/book/en/v2">git-scm.com/book/en/v2</a></li>
<li>The official CMake tutorial: <a href="https://cmake.org/cmake/help/latest/guide/tutorial/index.html">cmake.org tutorial</a></li></ul>
<p>These links are unverified from the build environment. Next, R.4 covers the Python and NumPy the course uses.</p>`,
    sim: {
      title: "Incremental build",
      intro: "Mark which files you edited. The graph shows which object files are recompiled (red) and whether the executable is relinked. Green means up to date: the build system reuses it. Press Run to watch the order: edit, compile, link. Headers touch every file that includes them.",
      height: 330,
      controls: FILES.map(([n], i) => ({ id: "f_" + n.replace(/\W/g, "_"), label: n, type: "select", value: i === 0 ? 1 : 0, options: [[0, "unchanged"], [1, "edited"]] })),
      run: { label: "Run the build", frames: 24, ms: 90 },
      draw(G, v, t) {
        const changed = FILES.map(([n]) => !!+v["f_" + n.replace(/\W/g, "_")]);
        const stage = t < 0.34 ? 0 : t < 0.67 ? 1 : 2;
        const r = graph(G, changed, stage, { y0: 40, step: 46, h: 34 });
        const rows = [["files edited", String(changed.filter(Boolean).length)], ["compiles (of 4)", String(r.compiles)], ["links (of 1)", r.link ? "1" : "0"], ["steps in a clean build", "5"]];
        const total = r.compiles + (r.link ? 1 : 0);
        return [
          { title: "What the build does", rows,
            gauge: [[total / 5, total === 5 ? "hot" : "ok"]], gaugeText: `${total} of 5 steps`,
            chip: [total < 5, total === 0 ? "nothing to do" : total === 5 ? "full rebuild" : "partial rebuild"] },
          { title: "Why", html: `<p class="note">Each object depends on its own .cpp and the headers it includes: main.o on util.hpp; load.o on util.hpp and tensor.hpp; kernel.o on tensor.hpp; log.o on none. The executable depends on every object. Counts are steps, not time: real compile times differ per file.</p>` },
        ];
      },
    },
    practice: {
      intro: `Run these from the repository root. Add <code>S2S_SOLUTIONS=1</code> in front to check the reference solutions.`,
      items: [
        { title: "Finish a CMake project", tier: "T0 · easy", goal: "Add a library target, link it into the app and register the test. First meet the linker error, then fix it.",
          cmd: "uv run pytest course/R-readiness/R.3-terminal-git-build/exercises/01-cmake-project" },
        { title: "A shell pipeline", tier: "T0 · easy", goal: "Write pipeline.sh: count TODO(run markers per top-level folder, print 'folder count', largest first.",
          cmd: "uv run pytest course/R-readiness/R.3-terminal-git-build/exercises/02-shell-pipeline" },
        { title: "A git checklist", tier: "T0 · easy", goal: "In a scratch repo under /tmp, commit, branch, create a merge conflict, resolve it and match the log graph. No test.",
          cmd: "less course/R-readiness/R.3-terminal-git-build/exercises/03-git-checklist/README.md" },
      ],
      labs: [
        { label: "Module README: all ten sections with commands", path: "course/R-readiness/R.3-terminal-git-build/README.md" },
        { label: "Self-check questions", path: "course/R-readiness/R.3-terminal-git-build/quiz.md" },
        { label: "The load_impl helper that reads S2S_SOLUTIONS", path: "course/common/python/s2s/exercise.py" },
      ],
    },
  });
})();
