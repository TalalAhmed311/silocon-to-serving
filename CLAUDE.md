# CLAUDE.md

## Read first: the course is being revamped

Before changing any course content, read **[design/REVAMP.md](design/REVAMP.md)**. In short:

- The current lessons (Markdown + MkDocs) tell the learner *what* to study. The owner wants every concept **taught from first principles with a visual explainer**.
- The approved lesson template is **`design/prototypes/d-hybrid.html`**: a scroll-driven story with a pinned animated diagram, inside an app shell with **Learn · Simulate · Practice** tabs.
- Scope is the whole course, P0 → Capstone plus Lane B. The pilot reference lesson is **P1.2 (KV cache)**.
- The existing exercises, tests, labs, `platform/` code and AWS guides stay as the Practice half.

## Working rules from the owner

- Discuss plans before large implementations; the owner wants to agree on direction first.
- No invented numbers: GPU specs and prices stay `UNVERIFIED` until cited, and versions stay pinned. Don't copy LeetGPU text (CC BY-NC-ND); link only.
- No CI pipeline (it was removed on request). Local checks: `python tools/check_links.py`, `bash tools/ci_terraform.sh`, `S2S_SOLUTIONS=1 uv run pytest`.
- Secrets never go in the repo. AWS guides state cost, auto-stop and teardown, and expose no unauthenticated public endpoints.
