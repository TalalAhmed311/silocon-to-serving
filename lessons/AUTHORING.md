# Writing a lesson

Every lesson is one file, `lessons/content/<id>.js`, that calls `S2S.lesson({...})` once. The shared shell (`assets/lesson.js`, `assets/lesson.css`) builds the page: top bar, **Learn** (progress rail · scrolling story · pinned animated stage), **Simulate** and **Practice**. `content/p1-2.js` is the reference lesson: read it before writing another.

After adding or renaming a content file:

```bash
python tools/build_lessons.py                                   # writes lessons/<id>.html + assets/available.js
PW=$(npm root -g)/playwright node tools/check_lessons.js <id>   # visits every step, sim and practice; screenshots in /tmp/lesson-shots/<id>/
```

The `id`s and titles come from `assets/catalog.js`; the `n` and `title` in the content file must match the catalog entry.

## The teaching bar (from design/REVAMP.md)

- **First principles.** Assume the learner knows programming and basic maths, and the lessons before this one in the catalog, nothing else. Introduce every term in plain words the first time it appears.
- **One idea per step.** 7–11 steps per lesson. Each step: the problem or question first, then the intuition, then the mechanism, then a worked example with real numbers.
- **Every step has a scene.** The scene must *show* the step's idea (a mechanism, a data flow, a before/after, a count), not decorate it. If you can't draw it, the step is probably two ideas or not concrete enough.
- **Real numbers.** Use real model configs (Llama-3-8B: 32 layers, hidden 4096, 32 query heads, 8 KV heads, head_dim 128, vocab 128,256, ≈8.0B params), real code from this repo, real commands. Never invent benchmark results. GPU specs and prices: either cite a source in the text or say "example value, check the spec sheet".
- **3–5 checkpoints** (multiple choice) on the steps that carry the lesson's key ideas. `why` must explain the answer, not restate it.
- **Analogies** in a `<div class="analogy"><b>Picture it</b>…</div>` block, at most one per step, only where they genuinely help.
- **Tie to the repo.** Name the file and function where the concept lives (`platform/...`, the module's `examples/`).
- **Plain style.** Short sentences. Active voice. No filler phrases. No emoji.
- **Licences.** Don't copy text from LeetGPU problems (CC BY-NC-ND) or long excerpts from papers or other projects. Explain in your own words and link.

## The lesson object

```js
S2S.lesson({
  id: "p1-2", n: "P1.2", title: "Prefill, decode and the KV cache",   // must match catalog.js
  subtitle: "Inference fundamentals · first principles · T0, no GPU needed",
  kicker: "Lesson · ≈ 40 min",
  headline: "The model that refused to read twice",   // h1 above the story: a hook, not the title again
  intro: "<p>…what this lesson does and why it matters…</p>",
  facts: ["10 steps", "4 checkpoints", "1 simulator", "4 exercises"],
  legend: [["k", "key"], ["v", "value"]],             // default stage legend: [colorName, label]
  prev: "p1-1", next: "p1-3",                          // neighbouring lesson ids from the catalog (omit at the ends)
  steps: [ { rail, title, body, scene(G), check?, legend? }, … ],
  outro: "<h2>What you now know</h2><ul><li>…</li></ul>",
  sim: { title, intro, controls: [...], run?: { label, frames, ms }, draw(G, values, t) → panels, height?, note? },
  practice: { intro?, items: [{ title, tier, goal, cmd }], labs: [{ label, path }] },
});
```

### Steps

- `rail`: 1–3 word label for the progress rail and stage header (lowercase).
- `title`: the step heading (a claim or question, not a noun).
- `body`: HTML. Allowed: `<p>`, `<b>`, `<i>`, `<code>`, `<ul>/<ol>`, `<table>`, `<div class="eq">` (pre-formatted maths or code, keep lines ≤ 60 chars), `<div class="analogy">`, `<details><summary>Go deeper</summary>…</details>` for optional maths.
- `check`: `{ q, options: [..], answer: index, why }`.
- `scene(G)`: draws on a **640 × 450** dark stage (keep content inside x 16–624, y 16–420; the caption sits at y 436).

### The drawing API `G`

Color names (stage and simulator use the same names): `ink`, `muted`, `line`, `bg`, `k` (cyan), `v` (amber), `q` (violet), `hot` (red), `ok` (green), `w` (grey, "weights"), `blue`, `pink`. A literal `#rrggbb` also works but breaks theming in the simulator, so prefer names.

| call | draws |
|---|---|
| `G.rect(x, y, w, h, {fill, stroke, rx, dash, opacity, sw})` | rectangle |
| `G.box(x, y, w, h, label, {fill, stroke, color, size, rx, dash})` | rectangle with centred label (text auto-contrasts on a fill) |
| `G.row(x, y, labels, {w, h, gap, fill})` | a row of boxes (`fill` may be `(i) => color`) |
| `G.grid(x, y, rows, cols, cw, ch, (r, c) => color \| null, {gap, rx})` | cell grid (matrices, memory, tiles) |
| `G.bars(x, baselineY, values, {w, gap, h, max, fill})` | bar chart (`fill` may be `(i, v) => color`) |
| `G.text(x, y, s, {size, color, anchor, weight})` · `G.label(...)` (small, muted) | text |
| `G.line(x1, y1, x2, y2, {color, w, dash, arrow})` · `G.arrow(...)` | lines and arrows |
| `G.path(d, {color, fill, w, dash, arrow})` · `G.circle(cx, cy, r, {fill, stroke})` | paths, circles |
| `G.axes(x, y, w, h, {xlabel, ylabel})` | chart axes |
| `G.caption(s)` | the stage's bottom caption: one line saying what to notice |
| `G.group({opacity})` | `<g>` to animate several shapes together (pass `{parent: g}` to shapes) |
| `G.from(targets, gsapVars)` · `G.to` · `G.fromTo` | GSAP entrance/motion (skipped automatically for reduced motion or offline) |
| `G.pulse(target, {lo, hi, d, repeat})` | blink to draw attention |
| `G.after(ms, fn)` · `G.every(ms, fn)` | timed or looping updates; cleared automatically when the step changes |

Animation guidance: one deliberate motion per scene that demonstrates the mechanism (things flowing, appearing in order, a counter growing). The scene must also read correctly as a still picture, because reduced-motion users and screenshots see only the final frame. For a looping demonstration use `G.every(ms, () => { G.clear(); draw(state++) })` only if you redraw everything inside; prefer `G.from` with `stagger`.

### Simulator

The **Simulate** tab is for the lesson's central quantitative idea: the learner changes inputs and sees the consequence. Every lesson that has a formula or a trade-off should have one.

- `controls`: `{ id, label, min, max, step, value, format? }` for sliders; `{ id, label, type: "select", value, options: [[value, text], …] }` for dropdowns. Values arrive in `draw` as `values[id]` (numbers when numeric).
- `draw(G, values, t)` draws on a **640 × height** chart (default 260; light/dark themed) and returns panels for the side column: `[{ title, rows: [[label, value], …], gauge: [[fraction, color], …], gaugeText, chip: [ok, text], html }]`.
- `run: { label, frames, ms }` adds a Run button that calls `draw` with `t` going 0 → 1 (animate a process); without `run`, `t` is always 1.
- Use `S2S.fmt.bytes / num / si / ms` for numbers.
- Be honest: label models as models ("lower bound", "assumes …") in a `note` or panel `html`.

### Practice

List the module's existing exercises (folder names under the module's `exercises/`) with a one-line goal in plain words and the exact command, usually `uv run pytest <module>/exercises/<NN-name>` (for CUDA exercises: the `cmake`/`ctest` command from the module README). `labs` lists examples, benches, AWS guides and platform code to explore, as repo paths.

## Done means

- `node --check lessons/content/<id>.js` passes and `tools/check_lessons.js <id>` reports `ok`.
- Every step's screenshot in `/tmp/lesson-shots/<id>/` shows a readable scene with no overlapping or clipped text.
- The facts are correct and the numbers check out (do the arithmetic).
