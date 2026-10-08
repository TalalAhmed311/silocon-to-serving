# Course revamp plan

**Status:** decided, not started (2026-10-07). This file is the brief for the next working session. Read it before touching any course content.

## Why

The first build (Stages 1–5, merged in PR #1) produced a course that tells the learner **what to study**: each module lists objectives, gives short notes, then sends them to papers, docs and labs. Almost none of it **teaches** the concepts, and only 20 animations cover about 50 modules.

The owner wants a course that **explains every concept itself, from first principles, with a visual explainer for each concept**. The practice material (labs, exercises, tests, `platform/` code, AWS guides) stays.

## Decisions made with the owner

| Topic | Decision |
|---|---|
| Scope | Revamp **everything, start to end**: P0–P6, Capstone and Lane B, not just P1.2 |
| Depth | **First principles.** Assume only programming and basic maths. Build every idea from the problem it solves |
| Format | A **custom HTML lesson app** (not MkDocs Markdown pages) |
| Style | **Hybrid of prototypes B and C**: a scroll-driven story with a pinned animated diagram, inside an app shell with Learn / Simulate / Practice tabs |
| Practice | **Keep** the existing exercises, tests, labs, `platform/` projects and AWS guides as the Practice half of each lesson |
| Motion | SVG + vanilla JS, with **GSAP** for tweens. **Remotion** only for optional short video intros, never for the interactive explainers |
| Pilot | **P1.2 (prefill, decode and the KV cache)** is the reference lesson. Finish it fully, then apply the template to every module |

## The approved template: `design/prototypes/d-hybrid.html`

Open it in a browser. Published preview: https://claude.ai/artifact/CLhwqXVCJ4igMcHGjguroJ (private to the owner).

Every lesson page has:

1. **Top bar:** lesson title, tier, a progress meter (checkpoints answered) and three tabs: **Learn · Simulate · Practice**.
2. **Learn tab**, in three columns:
   - **Left: lesson path rail.** One entry per story step. It marks seen and completed steps, and clicking an entry jumps to that step.
   - **Middle: the story.** 6–10 steps, each one idea, written from first principles: the problem, then the intuition, then the mechanism, then the formula with real numbers. **Checkpoint quizzes** go right after the steps they test, with instant feedback that explains the answer.
   - **Right: pinned stage.** A dark SVG diagram that redraws and animates for each step as it scrolls into view (IntersectionObserver + GSAP), with a Replay button and a legend.
   - **Phone layout:** the stage pins to the top and the story scrolls beneath it.
3. **Simulate tab:** an interactive simulator for the lesson's central idea, with live numbers in side panels. For P1.2 that is model, dtype, context, output and users, a work-per-step chart, a GPU memory gauge, and the KV-cache maths.
4. **Practice tab:** the module's existing exercises, each with a one-line goal and a copyable command (`uv run pytest …`). Labs and AWS guides link from here.
5. **Light and dark themes** through CSS tokens, `prefers-reduced-motion` respected, no external hosts except Google Fonts and cdnjs (GSAP 3.12.5).

The other prototypes in `design/prototypes/` were the alternatives shown to the owner. Keep them for reference:

- `a-essay.html`: explorable essay. Its inline figures and "Check yourself" ideas can be reused inside story steps.
- `b-scrolly.html`: pure scrollytelling. Its 8 stage scenes became the P1.2 Learn tab.
- `c-lab.html`: pure app lab. Its simulator and machine panel became the Simulate tab.

## Content rules for every lesson

- One concept per story step. Before writing a step, name the single idea it teaches and the picture that shows it.
- Every step has a diagram state on the stage. A concept without a visual is not done.
- Start from the problem ("why does this exist?"), then intuition, then mechanism, then numbers.
- Work examples with **real model numbers** (for example Llama-3-8B: 32 layers, 8 KV heads, head_dim 128 → 128 KiB of KV per token in bf16). Keep the existing rules: no invented benchmark numbers, GPU specs and prices marked UNVERIFIED until cited, versions pinned.
- Tie each lesson to the repo's own code where it exists (for example `self.k_cache[l, pos] = k` in `platform/engine/v0/reference/llama_numpy.py`).
- 3–5 checkpoints per lesson, each explaining *why* the answer is right.
- Don't copy LeetGPU text (CC BY-NC-ND). Link to it only.

## Proposed implementation plan (confirm with the owner before starting)

1. **Lesson framework.** Extract the shell from `d-hybrid.html` into shared assets (`lesson.css`, `lesson.js`: tabs, rail, scroll observer, checkpoints, progress, theme). Each lesson then supplies only its content: steps, scenes, quizzes, simulator and exercises. A lesson could be one HTML file per module, or a data file (JSON/JS) plus scenes, rendered by the shared shell.
2. **Course home and navigation.** A landing page with the phase map, progress across lessons (per-browser `localStorage`), and links into each lesson.
3. **Finish P1.2** to full depth as the reference lesson, then get the owner's review.
4. **Roll out phase by phase:** P0 → P1 → P2 → … → P6 → Capstone, then Lane B (L1–L6 concept lessons, with the LeetGPU maps as Practice). About 150–200 concepts in total. Fold the existing 20 animations into the matching lessons' stages.
5. **Hosting:** decide with the owner, either GitHub Pages (needs a `gh-pages` publish from their machine) or published artifacts.
6. **Retire or redirect** the MkDocs site once the new lessons replace it, or keep MkDocs only for reference pages (SOURCES, GAPS, GLOSSARY).

## State of the repo when this was written

- `master` includes the whole first build (PR #1 merged). CI was removed at the owner's request; checks are local scripts (`tools/check_links.py`, `tools/ci_terraform.sh`, `S2S_SOLUTIONS=1 uv run pytest`).
- Nothing in the repo has been executed except the MkDocs build (which passes `--strict`) and the prototype pages in a headless browser.
- The owner originally said not to run or download things. For the revamp they asked to **see** output, so rendering pages in a browser and publishing previews is fine. Ask before running anything heavy.
