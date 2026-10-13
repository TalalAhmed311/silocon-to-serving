# Readiness and ordering fixes: build plan

**Status:** approved by the owner on 2026-10-13 ("yes build and fix this"). This is the brief the writers follow.

## The problems this fixes

1. **No stated prerequisites.** P0.1 says "you can write a loop in some language", then expects RAII, move semantics, templates, CMake and sanitizers within about 6 hours. Pointers, the stack and heap, compiling and linking are never taught.
2. **CUDA basics are in the wrong place.** Lane B L1 has learners write kernels from week 1, before the GPU is explained (P1.4, week 5). P5 "deep dive" doesn't list Lane B as a prerequisite, and re-teaches kernels, grids and grid-stride loops. Toolchain setup, the host/device split, a complete first program and debugging are not taught anywhere.
3. **Skills used before they're taught.**
   - P2.1 (week 7) runs Terraform on AWS; Terraform and AWS are taught in P3.2 (week 14).
   - P3.1 assumes Docker; P3.3 assumes Kubernetes basics.
   - P1.3 and P2.7 assume HTTP and server-sent events.
4. **No phase entry checks.** No phase says "you're ready when…".

## What gets built

| New | id | Module folder | Writer |
|---|---|---|---|
| R.1 C and C++ from zero | `r-1` | `course/R-readiness/R.1-cpp-from-zero` | A |
| R.2 Memory: stack, heap and ownership | `r-2` | `course/R-readiness/R.2-memory-management` | A |
| R.3 The terminal, git and building code | `r-3` | `course/R-readiness/R.3-terminal-git-build` | B |
| R.4 Python and NumPy for this course | `r-4` | `course/R-readiness/R.4-python-and-numpy` | B |
| R.5 The maths and ML you need | `r-5` | `course/R-readiness/R.5-maths-and-ml-basics` | C |
| R.6 HTTP and networking basics | `r-6` | `course/R-readiness/R.6-http-and-networking` | C |
| R.7 A safe cloud account | `r-7` | `course/R-readiness/R.7-cloud-account-and-cost-safety` | D |
| P3.0 Containers and Kubernetes from zero | `p3-0` | `course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics` | D |
| P1.6 Your first CUDA program | `p1-6` | `course/P1-inference-fundamentals/P1.6-first-cuda-program` | E |
| CUDA re-sequencing (P5, Lane B edits) | — | see Writer E | E |
| Readiness syllabus, phase entry checklists, SOURCES, GLOSSARY | — | see Writer F | F |
| Catalog, home banner, README, PROGRESS, nav, prev/next links, final checks | — | integration | lead |

`lessons/assets/catalog.js` already has the new entries. Each lesson's `n` and `title` must match it exactly.

## Rules for every writer

- Read `CLAUDE.md`, `design/REVAMP.md` and `lessons/AUTHORING.md` first. Then read `lessons/content/p1-2.js` (the reference lesson) and `lessons/content/p0-1.js` (the closest in tone to Readiness).
- **Edit only the files your section names.** Other writers are working in parallel. Never touch `catalog.js`, `home.js`, `index.html`, `README.md`, `PROGRESS.md`, `SOURCES.md`, `GLOSSARY.md` or `site/mkdocs.yml` unless your section lists it.
- **Don't commit or push.** The lead reviews and commits.
- **Assume no prior knowledge beyond the lessons before this one in the catalog.** Readiness assumes only "can write a loop and a function in some language" and school maths.
- **Readiness lessons are teaching lessons and a self-check at the same time.** Each one opens its `intro` with: "Already comfortable with X? Answer the checkpoints. If you get them all right, skip to <next>." Each one ends its `outro` with a **"Go further"** list of the linked resources from the table below. Link only; never copy.
- **No invented numbers.** Prices, GPU specs and free-tier limits are marked "example value, check the current price page" or UNVERIFIED. Versions stay pinned where the repo pins them.
- **Every module folder gets:** `README.md` (same shape as `course/P0-systems-primer/P0.1-c-cpp-for-systems/README.md`: the header table with Tier/Time/Prerequisites/You will build, objectives, numbered sections, exercises list, sources), `quiz.md` (6–8 `<details>` questions, same format as P0.1's), and `exercises/` with starters, `solutions/` and tests, unless your section says otherwise.
- **Exercise conventions (follow exactly):**
  - **Python:** `exercises/<NN-name>/<module>.py` (starter with `TODO`s that raise `NotImplementedError`), `exercises/solutions/<NN-name>/<module>.py`, `exercises/<NN-name>/test_<module>.py` using `from s2s.exercise import load_impl` then `m = load_impl(__file__, "<module>")`, plus `exercises/<NN-name>/README.md` with a one-paragraph goal. Verify with `S2S_SOLUTIONS=1 uv run pytest <module folder>`, which must pass.
  - **C++:** copy the pattern of `course/P0-systems-primer/P0.1-c-cpp-for-systems/exercises/CMakeLists.txt`: `exercises/<NN-name>/{README.md,impl.hpp,test.cpp}`, `exercises/solutions/<NN-name>/impl.hpp`, and an `s2s_exercise()` CMake function with `S2S_USE_SOLUTIONS`. Tests use `course/common/include/check.hpp` if it fits, or plain `assert`-style checks that `return 1` on failure. Verify with `cmake -S <ex> -B /tmp/<name> -DS2S_USE_SOLUTIONS=ON && cmake --build /tmp/<name> -j && ctest --test-dir /tmp/<name> --output-on-failure`, which must pass. Also confirm the starters **compile** (configure without the flag and build; tests may fail).
  - **CUDA:** there is no GPU or `nvcc` here. Follow `course/P5-cuda-deep-dive/CMakeLists.txt` and `P5.1-execution-model/exercises/01-vector-add`, and mark every run `TODO(run-on: T2)` the way existing modules do.
- **Lesson verification (required before you report done):**
  ```bash
  node --check lessons/content/<id>.js
  python tools/build_lessons.py
  PW=$(npm root -g)/playwright node tools/check_lessons.js <id>
  ```
  `check_lessons.js` must print `ok`. Then **open the screenshots** in `/tmp/lesson-shots/<id>/` with the Read tool and fix any clipped, overlapping or unreadable scenes. Also check `sim.png` and `practice.png`.
- **Size:** 8–10 steps, 4–5 checkpoints, one simulator, and a Practice tab listing your exercises with exact commands.
- **Report back:** files created or changed, verification output, anything you couldn't verify, and any external link you used that isn't in the table below.

## Shared resource list (link only; all UNVERIFIED from this build environment)

| Topic | Resource | URL |
|---|---|---|
| C++ basics | learncpp.com (chapters on pointers, references, dynamic allocation, classes; name the topics, not chapter numbers) | https://www.learncpp.com/ |
| C++ reference | cppreference | https://en.cppreference.com/w/ |
| Memory | CS50x, Week 4 "Memory" | https://cs50.harvard.edu/x/ |
| Sanitizers | Clang AddressSanitizer docs | https://clang.llvm.org/docs/AddressSanitizer.html |
| Terminal, shell, git | MIT "The Missing Semester of Your CS Education" | https://missing.csail.mit.edu/ |
| git | Pro Git (book) | https://git-scm.com/book/en/v2 |
| CMake | Official CMake tutorial | https://cmake.org/cmake/help/latest/guide/tutorial/index.html |
| Python | The Python tutorial | https://docs.python.org/3/tutorial/ |
| NumPy | NumPy: the absolute basics for beginners | https://numpy.org/doc/stable/user/absolute_beginners.html |
| Linear algebra | 3Blue1Brown, Essence of linear algebra | https://www.3blue1brown.com/topics/linear-algebra |
| Neural networks | 3Blue1Brown, Neural networks | https://www.3blue1brown.com/topics/neural-networks |
| Neural networks (code) | Karpathy, Neural Networks: Zero to Hero | https://karpathy.ai/zero-to-hero.html |
| HTTP | MDN, An overview of HTTP | https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview |
| SSE | MDN, Using server-sent events | https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events |
| AWS budgets | AWS Budgets user guide | https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html |
| AWS sign-in | IAM Identity Center user guide | https://docs.aws.amazon.com/singlesignon/latest/userguide/what-is.html |
| AWS access | Systems Manager Session Manager | https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html |
| Terraform | HashiCorp, Get started with Terraform on AWS | https://developer.hashicorp.com/terraform/tutorials/aws-get-started |
| Docker | Docker, Get started | https://docs.docker.com/get-started/ |
| Kubernetes | Kubernetes basics tutorial | https://kubernetes.io/docs/tutorials/kubernetes-basics/ |
| kind | kind quick start | https://kind.sigs.k8s.io/docs/user/quick-start/ |
| CUDA intro | NVIDIA blog, An Even Easier Introduction to CUDA | https://developer.nvidia.com/blog/even-easier-introduction-cuda/ |
| CUDA reference | CUDA C++ Programming Guide | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ |
| CUDA debugging | Compute Sanitizer docs | https://docs.nvidia.com/compute-sanitizer/ |
| No-GPU warm-up | srush/GPU-Puzzles | https://github.com/srush/GPU-Puzzles |
| Free notebook GPU | Google Colab | https://colab.research.google.com/ |
| Browser GPU | LeetGPU (problems are CC BY-NC-ND: link only) | https://leetgpu.com/ |

---

## Writer A: R.1 and R.2 (C++ and memory)

Files: `course/R-readiness/R.1-cpp-from-zero/**`, `course/R-readiness/R.2-memory-management/**`, `lessons/content/r-1.js`, `lessons/content/r-2.js`.
Tier T0. Toolchain: `g++`/`clang++` ≥ 11, CMake ≥ 3.20. Links: r-1 `next: "r-2"`, no `prev`. r-2 `prev: "r-1"`, `next: "r-3"`.

**R.1 C and C++ from zero** (≈ 30 min reading + 4 h hands-on). The goal is to give a Python or JavaScript programmer everything P0.1 silently assumes.
Steps, one idea each:
1. Source → compiler → object file → linker → executable. Why C++ needs a build step (scene: the pipeline, with `g++ -c` and the link).
2. Types have sizes: `char` 1, `int` 4, `float` 4, `double` 8, pointer 8 on 64-bit (scene: boxes of bytes).
3. Variables live at addresses. `&x` and a pointer `int* p = &x`; `*p` reads through it (scene: memory line with an arrow).
4. Arrays are contiguous; `a[i]` is `*(a + i)`; pointer arithmetic steps by `sizeof(T)` (scene: array with byte offsets).
5. References: an alias that can't be null or re-seated. Pass by value vs by reference vs by pointer (scene: three calls, which copies).
6. `const` and why `const T&` is the default way to pass big things.
7. Functions, headers and the one-definition rule: declaration in `.hpp`, definition in `.cpp`, `#include` is text paste.
8. `struct`/`class`: fields, methods, constructors, destructors (scene: object lifetime timeline, constructor at `{`, destructor at `}`).
9. The standard library you'll meet first: `std::vector`, `std::string`, `std::array`, range-for.
10. Undefined behaviour in one picture: out-of-bounds read, uninitialised variable. Why the compiler won't save you, and why P0.1 uses sanitizers.
Simulator: "pointer arithmetic explorer". Pick a type (char/int/float/double/struct of 12 bytes), a base address and an index, and see the computed address, bytes stepped and which elements a `p + i` lands on.
Exercises (C++, ctest):
- `01-swap`: `swap_by_pointer(int*, int*)` and `swap_by_ref(int&, int&)`.
- `02-array-sum`: sum and max of a raw array `const float* a, size_t n` with pointer arithmetic, no indexing. Handles `n == 0`.
- `03-vec2-class`: `struct Vec2` with a constructor, `length()`, `operator+` and a `const` method. The test checks values.
- `04-split-header`: fill in a two-file `stats.hpp`/`stats.cpp` (declaration vs definition), built as a library target that the test links.

**R.2 Memory: stack, heap and ownership** (≈ 30 min + 4 h).
Steps:
1. A process's memory: code, globals, heap, stack (scene: address-space map).
2. The stack: one frame per call, freed automatically on return (scene: frames pushing and popping).
3. Returning a pointer to a local is a dangling pointer. Why it "works" sometimes.
4. The heap: `new`/`delete`, `new[]`/`delete[]`, `malloc`/`free`. Lifetime is yours (scene: heap blocks with owners).
5. The three classic bugs: leak, double free, use-after-free (scene: each as a timeline).
6. AddressSanitizer: compile with `-fsanitize=address -g` and read the report (show a real-shaped report, written by you).
7. Ownership: exactly one owner frees. RAII: tie the free to a destructor.
8. `std::unique_ptr` (sole owner, move-only) and `std::make_unique`; `std::shared_ptr` with a reference count, and why engines rarely want it on hot paths.
9. Copy vs move in one picture: copying a vector copies its heap buffer; moving hands over the pointer. This is the bridge to P0.1's move section.
10. Alignment and `alignas` in one sentence each: a teaser for P0.1, not a full treatment.
Simulator: "stack and heap timeline". Step through a small program (choose one of 3 preset programs: correct, leak, use-after-free). It shows frames and heap blocks over time and flags the bug at the step it happens.
Exercises (C++, ctest; build with `-fsanitize=address,undefined` in a Debug configuration, like P0.1's exercise 5):
- `01-fix-the-leak`: a function that leaks on an early-return path. Fix it with `std::unique_ptr`.
- `02-dangling-return`: a function returns a pointer to a local. Change the API to return by value.
- `03-owning-buffer`: a tiny `FloatBuffer` class: constructor allocates, destructor frees, copy deleted, move implemented. The test counts live allocations through a static counter.
- `04-sanitizer-read`: a planted heap-buffer-overflow. The README asks the learner to paste ASan's first line into a comment, then fix it.

## Writer B: R.3 and R.4 (terminal, git, build; Python and NumPy)

Files: `course/R-readiness/R.3-terminal-git-build/**`, `course/R-readiness/R.4-python-and-numpy/**`, `lessons/content/r-3.js`, `lessons/content/r-4.js`.
Links: r-3 `prev: "r-2"`, `next: "r-4"`; r-4 `prev: "r-3"`, `next: "r-5"`.

**R.3 The terminal, git and building code** (T0, ≈ 30 min + 3 h).
Steps:
1. The shell runs programs: command, arguments, exit code (`$?`).
2. Files and paths: absolute vs relative, `cd`, `ls`, `cat`, `less`, `grep`, `find`.
3. Pipes and redirection: `|`, `>`, `2>&1`. Scene: three processes joined by pipes.
4. Processes and environment variables: `ps`, `top`, `kill`, `export S2S_SOLUTIONS=1` (and why the course's tests read it).
5. Permissions and `sudo` in one step: why `perf` and `ncu` sometimes need it.
6. git's model: working tree → staging → commit → branch; `clone`, `status`, `add`, `commit`, `log`, `diff`, `checkout -b`, `push`.
7. Make vs CMake: what a build system does (dependency graph, only rebuild what changed). Scene: a dependency graph lighting up after one file changes.
8. CMake in practice: `cmake -S . -B build`, `cmake --build build -j`, `ctest --test-dir build`. Targets, `Release` vs `Debug`.
9. The course's Python toolchain: `uv sync`, `uv run pytest <path>`, and `S2S_SOLUTIONS=1`.
10. SSH vs SSM in one step (the course uses SSM only); a forward pointer to R.7.
Simulator: "incremental build". The learner toggles which source files changed in a 6-file project (2 headers, 4 sources) and sees which object files and the executable rebuild, with a count.
Exercises:
- `01-cmake-project` (C++): a starter folder with `main.cpp`, `mathlib.cpp`, `mathlib.hpp` and an **incomplete** `CMakeLists.txt` (`TODO`: add a library target, link it, register the test). The README gives the exact commands. The solution is a complete CMakeLists. Verification: configure, build and ctest against the solution.
- `02-shell-pipeline` (Python test): the learner writes `pipeline.sh` that counts the number of `TODO(run` markers per top-level folder of the repo and prints `folder count`, sorted. The pytest runs it with `subprocess` (`bash`) on a small fixture tree the test creates in `tmp_path` and checks the output. Put the starter and solution `.sh` in the usual places, and load the script path the same way `load_impl` picks a folder (read `S2S_SOLUTIONS`).
- `03-git-checklist`: Markdown only, no test. A sequence of git tasks on a scratch repo (init, commit, branch, merge conflict, resolve), with the expected `git log --oneline --graph` shape shown.

**R.4 Python and NumPy for this course** (T0, ≈ 30 min + 3 h).
Steps:
1. Why the course uses Python next to C++: reference implementations and tests (point at `platform/engine/v0/reference/llama_numpy.py`).
2. Python essentials in one step: functions, dicts, list comprehensions, `dataclass`, type hints as documentation.
3. An `ndarray` is a typed, flat buffer plus a shape: the same idea as P0.1's strides (scene: buffer and shape).
4. dtype decides bytes: float32 4, float16 2, int8 1. `a.nbytes`. Real number: one Llama-3-8B layer's `q_proj` is 4096×4096 → 64 MiB in fp32, 32 MiB in bf16.
5. Indexing, slicing and views vs copies (`a[:, 0]` is a view; `.copy()`).
6. Reshape and transpose change strides, not data (`a.strides`).
7. Broadcasting rules, with a scene aligning shapes from the right.
8. Vectorise: replace a Python loop with one array expression. Measure with `time.perf_counter` (no invented speed-ups: say "measure it").
9. Matrix multiply: `@`, shapes `(m, k) @ (k, n) → (m, n)`, and 2·m·k·n FLOPs.
10. pytest in one step: `assert`, `np.testing.assert_allclose(rtol, atol)`, and why float comparisons need tolerances.
Simulator: "broadcasting checker". Two shape inputs (selects for up to 3 dimensions each). It shows the right-aligned shapes, which dimensions stretch, the result shape or the error, and the result's element count and bytes for a chosen dtype.
Exercises (Python):
- `01-shapes-and-bytes`: `nbytes(shape, dtype)` and `matmul_flops(m, k, n)`.
- `02-views`: return a view, not a copy, for row/column/transposed selections. The test checks `np.shares_memory`.
- `03-broadcast`: `normalize_rows(x)` (subtract row mean, divide by row std) without loops.
- `04-vectorize`: rewrite a given loop-based pairwise-distance function into a broadcast version that matches within `rtol=1e-6`.

## Writer C: R.5 and R.6 (maths and ML; HTTP)

Files: `course/R-readiness/R.5-maths-and-ml-basics/**`, `course/R-readiness/R.6-http-and-networking/**`, `lessons/content/r-5.js`, `lessons/content/r-6.js`.
Links: r-5 `prev: "r-4"`, `next: "r-6"`; r-6 `prev: "r-5"`, `next: "r-7"`.

**R.5 The maths and ML you need** (T0, ≈ 40 min + 3 h). The goal is to make P1.1 (the transformer forward pass) readable. Don't teach the transformer itself; P1.1 does.
Steps:
1. Vectors as lists of numbers; dot product as "how aligned" (scene: two arrows and the number).
2. Matrix × vector = many dot products. FLOPs = 2 per multiply-add (scene: a row sweeping a vector).
3. Matrix × matrix, shapes and cost 2·m·k·n.
4. exp and log: why exp turns scores into positive weights; logs turn products into sums.
5. Softmax, and the numerically stable version (subtract the max). Scene: scores → exp → normalise.
6. A neuron and a layer: `y = activation(W x + b)`; ReLU and SiLU as curves.
7. A network is layers stacked; parameters are the numbers in W and b. Count them for a small 2-layer MLP (real arithmetic).
8. Training vs inference in one step: training changes the weights; inference only reads them (this course is about inference).
9. Language models: text → tokens → ids → embedding vectors → … → scores over the vocabulary → next token. Use Llama-3-8B's vocab of 128,256 as the real number.
10. Sampling in one step: greedy vs temperature (pointer to P0.5 and P2).
Simulator: "softmax and temperature". Five editable logits (sliders) and a temperature slider. It shows the probability bars, the entropy, and the naive vs stable computation (and a flag when the naive version overflows in float32 for large logits).
Exercises (Python):
- `01-dot-and-matvec`: implement with loops, compare against `@`.
- `02-softmax`: stable softmax over the last axis. The test includes logits like `[1000, 1001]`.
- `03-mlp-forward`: a 2-layer MLP forward pass with SiLU, given weights. The test also checks `param_count`.
- `04-next-token`: given logits and a vocab list, return the greedy token and the top-k tokens with probabilities.

**R.6 HTTP and networking basics** (T0, ≈ 30 min + 3 h). It must cover what P1.3 (TTFT/ITL, SSE) and P2.7 (gateway) assume.
Steps:
1. Client and server, IP address and port. `localhost:8000` (scene: two boxes, one socket).
2. TCP in one step: a reliable byte stream with a connection setup round trip. Latency vs bandwidth.
3. DNS and TLS in one step each, as the costs that sit before your first byte.
4. An HTTP request: method, path, headers, body; a response: status code, headers, body. Real `curl -v` shape.
5. JSON APIs: the OpenAI-style `/v1/chat/completions` request body the course uses (model, messages, max_tokens, stream).
6. Status codes that matter for serving: 200, 400, 401, 404, 429, 500, 503, and what a gateway does with each.
7. Streaming with server-sent events: `data: …\n\n` chunks; why streaming makes time-to-first-token visible (scene: tokens arriving over time with TTFT and inter-token gaps marked).
8. Timeouts, retries and idempotency: when retrying is safe.
9. Concurrency: many open connections, and why a server needs async I/O or many workers (forward pointer to P2).
10. Hands-on tools: `curl`, `curl -N` for streams, and FastAPI's `TestClient` (used by the course's gateway tests).
Simulator: "request timeline". Inputs: network round-trip time, TLS on/off, time to first token, tokens out, inter-token latency, and streaming on/off. It draws the timeline (connect, TLS, request, TTFT, token stream) and reports when the user sees the first token and the full answer in each mode.
Exercises (Python):
- `01-parse-sse`: parse a raw SSE byte stream into events (handles multi-line `data:`, comments, `[DONE]`).
- `02-ttft-itl`: from a list of chunk arrival timestamps, compute TTFT, mean and p90 inter-token latency.
- `03-tiny-api`: a FastAPI app with `POST /v1/echo` that validates a JSON body with pydantic and returns 400/422 on bad input. Tested with `fastapi.testclient.TestClient`.
- `04-retry-policy`: `should_retry(status, attempt, idempotent)` and an exponential backoff with jitter bound. Test the table of cases.

## Writer D: R.7 and P3.0 (cloud account; containers and Kubernetes)

Files: `course/R-readiness/R.7-cloud-account-and-cost-safety/**`, `course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics/**`, `lessons/content/r-7.js`, `lessons/content/p3-0.js`.
Links: r-7 `prev: "r-6"`, `next: "p0-1"`. p3-0 `prev: "p2-7"`, `next: "p3-1"`.
Read `infra/aws/README.md`, `infra/aws/guardrails/`, `infra/aws/single-node/README.md` and `course/P2-serving-engines/P2.1-first-serve-with-vllm/aws.md` first: R.7 must prepare the learner to run exactly that P2.1 flow.

**R.7 A safe cloud account** (T0 to read; the account setup is free but **the learner must do it before P2.1**). ≈ 30 min + 2 h.
Steps:
1. Why the cloud: GPUs by the hour vs buying one. The bill is the risk. Show the cost formula `hours × hourly price`, with the price as an "example value, check the price page".
2. Root user vs everyday identity: lock the root user with MFA and use IAM Identity Center (SSO) for daily work.
3. Regions and availability zones; GPU quotas ("vCPU limits") and why the first GPU launch often fails until a quota increase.
4. Budgets and alarms: an AWS Budget with an email alert. Point at the repo's `infra/aws/guardrails` stack.
5. What costs money when "off": a stopped instance still pays for EBS; snapshots, NAT gateways, elastic IPs and load balancers bill while idle.
6. Access without open SSH: Session Manager (SSM), and why the course never opens port 22.
7. Terraform in one step: you describe resources; `plan` shows the diff; `apply` makes it; `destroy` removes it. State is the record. (P3.2 teaches it properly; here you only run a given stack.)
8. The course's stack lifecycle: `make up`, `make ssm`, `make down`, and the idle auto-stop. Scene: the lifecycle as a loop with the cost meter running only between up and down.
9. Teardown checklist: confirm nothing is left (the console's billing page, `terraform state list` empty).
10. Secrets: never commit keys; `aws configure sso`, environment variables, and `.gitignore` for `terraform.tfvars`.
Simulator: "what will this cost?". Inputs: hourly price (a free number labelled "enter the current price for your region"), hours per session, sessions per week, weeks, EBS GB, and a "forgot to tear down for N days" slider. It shows the total, how much of it comes from the forgotten days, and a gauge against a budget input.
Exercises:
- `01-cost-calc` (Python): `session_cost`, `idle_cost(ebs_gb, price_per_gb_month, days)` and `budget_ok(...)`. Tests use made-up example prices in the test only, clearly named `EXAMPLE_PRICE`.
- `02-account-checklist` (Markdown, no test): a checkbox list (root MFA, SSO user, budget alarm, region chosen, GPU quota requested, `aws sts get-caller-identity` works via SSO, `infra/aws/guardrails` applied).
- `03-tfvars-hygiene` (Python): `check_tfvars(text)` flags secrets-looking values (AKIA… keys, `password =`), and `gitignore_covers(lines, path)` checks that `terraform.tfvars` is ignored.

**P3.0 Containers and Kubernetes from zero** (T0 with Docker and kind; ≈ 30 min + 5 h). It must make P3.1 (GPU containers) and P3.3 (Kubernetes and the GPU Operator) readable.
Steps:
1. "Works on my machine": why packaging the whole user space helps.
2. An image is stacked read-only layers; a container is a process with a writable layer on top (scene: layers).
3. A Dockerfile: `FROM`, `RUN`, `COPY`, `CMD`; layer caching and why order matters (scene: a changed line invalidating the layers below it).
4. Isolation: namespaces and cgroups in one step. A container is not a VM.
5. Ports, volumes and environment: `-p 8000:8000`, `-v`, `-e`.
6. Registries and tags vs digests (pin by digest, as the course's `env/` does).
7. Why one machine isn't enough: Kubernetes as a desired-state loop (scene: desired vs actual and a controller reconciling).
8. Pods, Deployments and Services; labels and selectors.
9. `kubectl apply/get/describe/logs`, and kind for a laptop cluster (`platform/deploy/kind`).
10. Requests and limits, and how a GPU is "just another resource" (`nvidia.com/gpu: 1`), a forward pointer to P3.3.
Simulator: "layer cache". The learner chooses which Dockerfile line changed in an 8-line Dockerfile (base, apt, uv sync, copy code, …) and sees which layers rebuild, with example sizes labelled as examples. A toggle reorders the Dockerfile ("copy code before dependencies") to show the cache miss.
Exercises:
- `01-dockerfile-lint` (Python): `lint(dockerfile_text)` returns findings: unpinned `FROM` (no digest or tag), `latest` tag, `COPY . .` before dependency install, running as root (no `USER`). Read `course/P3-deployment-and-infra/P3.1-gpu-containers` first so this is the beginner version of its linter, not a duplicate.
- `02-deployment-yaml` (Python, pyyaml): the learner writes `deployment.yaml` (Deployment and Service for a tiny HTTP echo app with 2 replicas, a readiness probe, and resource requests and limits). The test parses it and checks the fields and that the label selectors match.
- `03-kind-walkthrough` (Markdown, no test): create a kind cluster, apply the YAML, `kubectl get pods`, `port-forward`, curl it, delete the cluster.

## Writer E: P1.6 and the CUDA re-sequencing

Files you own:
- `course/P1-inference-fundamentals/P1.6-first-cuda-program/**`
- `lessons/content/p1-6.js`
- edits to `course/P1-inference-fundamentals/syllabus.md` (add a P1.6 row and section)
- edits to `course/P5-cuda-deep-dive/syllabus.md`, `course/P5-cuda-deep-dive/P5.1-execution-model/README.md` and `lessons/content/p5-1.js`
- edits to `laneB-cuda/README.md`, `laneB-cuda/L1-launch-and-indexing/syllabus.md`, `laneB-cuda/L2-memory-and-shared-memory/syllabus.md`, `laneB-cuda/L3-reductions-and-scans/syllabus.md` and `lessons/content/l1.js`

Links: p1-6 `prev: "p1-5"`, `next: "p2-1"`. **Also change `next` in `lessons/content/p1-5.js` to `"p1-6"` and `prev` in `lessons/content/p2-1.js` to `"p1-6"`** (you own those two one-line edits).

**P1.6 Your first CUDA program** (T1 LeetGPU or Colab for the kernel; T2 for the full host program; ≈ 30 min + 5 h). Prerequisites: P1.4 (GPU architecture), R.1–R.2. It sits right after P1.4/P1.5, so a learner meets the GPU's structure first and writes kernels second.
Steps:
1. Host and device: two processors, two memories, connected by PCIe. The CPU drives; the GPU runs what it's given (scene: CPU+RAM, PCIe, GPU+HBM).
2. Where to get a GPU: LeetGPU in the browser (kernel only), Colab (notebook GPU), or the course's AWS `g4dn.xlarge` (P5's `aws.md`). Check with `nvidia-smi`. Prices are example values.
3. The toolchain: `nvcc` compiles `.cu` files; host code goes to the C++ compiler, device code to PTX/SASS. `-arch=sm_75`. Compute capability in one sentence.
4. A kernel is a function marked `__global__` that every thread runs; `threadIdx`, `blockIdx` and `blockDim` (keep it to 1D here).
5. The five-step program: allocate device memory (`cudaMalloc`) → copy in (`cudaMemcpy` H2D) → launch `<<<grid, block>>>` → copy out (D2H) → free. Scene: the five steps as arrows across PCIe.
6. Ceil-div and the bounds check: launching more threads than elements, safely.
7. Launches are asynchronous; errors arrive late. `cudaGetLastError()` after the launch, `cudaDeviceSynchronize()` before reading results, and a `CUDA_CHECK` macro (point at `laneB-cuda/harness/include/s2s_cuda.cuh`).
8. Debugging: `printf` from a kernel (with its caveats), and `compute-sanitizer` catching an out-of-bounds write.
9. Timing it honestly with CUDA events, and why the first launch is slow (context creation). Bytes moved ÷ time = GB/s; compare with the P1.4 roofline. No invented numbers.
10. Where to go next: Lane B L1 (indexing in 2D and 3D, many problems) runs alongside P2–P3; P5 is the performance deep dive and assumes both.
Simulator: "the transfer tax". Inputs: N elements, dtype bytes, PCIe bandwidth (labelled "example value, check your platform"), GPU memory bandwidth (example value) and kernel passes over the data. It shows the time split between H2D copy, kernel and D2H copy, and when the copies dominate. Note: a model, not a measurement.
Exercises (CUDA, T2, using the P5 pattern and `laneB-cuda/harness/include/s2s_cuda.cuh`). Write an `exercises/CMakeLists.txt` modelled on `course/P5-cuda-deep-dive/CMakeLists.txt`, scoped to this module:
- `01-hello-device`: print each thread's global index from a 2-block × 4-thread launch; the test captures nothing, it just checks the launch succeeds with `CUDA_CHECK` (a smoke test).
- `02-vector-add-full`: the full five-step host program for `c = a + b` for any N, including ceil-div and the bounds check. The test compares with a CPU loop for N in {1, 1000, 1<<20, 1<<20 + 3}.
- `03-error-checks`: a planted bug (a launch with 2048 threads per block, and a missing sync before reading). The learner adds checks so the test sees the error reported instead of silent garbage.
- `04-sanitizer` (Markdown, no test): run `compute-sanitizer` on a provided out-of-bounds kernel and explain its report.
Mark every run `TODO(run-on: T2)`. Add an `aws.md` that points at the P5 `aws.md` single-GPU flow instead of duplicating it.

**The re-sequencing edits:**
- `course/P5-cuda-deep-dive/syllabus.md`: retitle to "P5 — CUDA performance deep dive". Add a "Before you start P5" block: P1.6 done, Lane B L1–L3 exit checks passed, Nsight-ready AWS access (P5 `aws.md`). Keep the folder name.
- `P5.1-execution-model/README.md`: prerequisites gain P1.6 and Lane B L1–L3. Add one sentence at the top of section 1 saying it recaps P1.6/L1 briefly and then goes deeper (SM scheduling, occupancy, divergence, streams).
- `lessons/content/p5-1.js`: say in the `intro` that the first three steps recap P1.6 and L1 quickly. Tighten those three step bodies to a recap (keep their scenes) and make sure the later steps carry the weight. Keep the step count ≥ 8.
- `laneB-cuda/README.md`: new week ranges. Weeks 1–5: an optional **L0 warm-up** with no GPU (srush/GPU-Puzzles in a notebook, linked only). L1 weeks 6–10 (after P1.6), L2 weeks 11–16, L3 weeks 17–24; L4–L6 unchanged. Add one line: "Do P1.6 before L1."
- The L1/L2/L3 `syllabus.md` headers: update their week ranges to match, and add "Prerequisite: P1.6" to L1.
- `lessons/content/l1.js`: add P1.6 as a prerequisite in the intro, and remove anything that contradicts the new ordering.

## Writer F: Readiness syllabus, phase entry checklists, SOURCES, GLOSSARY

Files you own: `course/R-readiness/syllabus.md`, `course/R-readiness/README.md`, all `course/P*/syllabus.md` and `course/capstone/syllabus.md` (**except** the P1 and P5 syllabi, which Writer E owns; send Writer E's phases' checklist text in your report so the lead can paste it), `SOURCES.md`, `GLOSSARY.md`, `GAPS.md`.

1. `course/R-readiness/README.md`: **"Start here"**. It holds:
   - The disclaimer: what the course assumes (a loop and a function in some language; school maths), and what it expects you to learn first if you lack it.
   - A **15-minute self-check**: 2–3 concrete questions per R module, with answers in `<details>`, plus "If you missed any, do R.N" routing.
   - A table of R.1–R.7: what each covers, time, and the external resource to use if you prefer a longer course (from the shared list).
   - A note that R.7 must be done before P2.1 (the first cloud run), even if you skip the rest.
2. `course/R-readiness/syllabus.md`: same shape as the other phase syllabi (header line, why this phase, the module table, a short section per module with objectives, exercises and sources). Use the step outlines in this plan for objectives.
3. **"Ready for this phase?"** block at the top of every phase syllabus you own (P0, P2, P3, P4, P6, Capstone): 4–6 checkbox items naming specific prior modules or exit checks, and "If not, go to …". P0 points at R.1–R.3. P2 points at P1 and R.6–R.7. P3 points at P3.0, R.7 and P2.1. P3's syllabus also gains the P3.0 row and section (use Writer D's outline in this plan).
4. `SOURCES.md`: a new subsection "4.x Readiness and bridge resources" listing every row of the shared resource table with status **UNVERIFIED** (egress-blocked) and "Used by". `GAPS.md` §A gains one row pointing at that subsection.
5. `GLOSSARY.md`: add the new terms the R modules, P1.6 and P3.0 introduce (about 30: e.g. linker, object file, stack frame, heap, dangling pointer, RAII, ASan, exit code, pipe, environment variable, staging area, broadcasting, dtype, softmax, logit, token id, SSE, TTFT, status code, idempotent, IAM Identity Center, budget alarm, Session Manager, image layer, cgroup, Pod, Deployment, Service, host/device, `cudaMemcpy`, compute-sanitizer). Match the file's existing format and sort order; don't duplicate terms already there.

## Lead: integration (after the writers report)

- Review every writer's diff, re-run all checks, fix contradictions.
- Change `prev` in `lessons/content/p0-1.js` to `"r-7"`, `next` in `lessons/content/p2-7.js` to `"p3-0"`, and `prev` in `lessons/content/p3-1.js` to `"p3-0"`.
- Home page: a "New here? Start with Readiness" banner (`lessons/index.html`, `lessons/assets/home.js`, `lessons/assets/lesson.css`).
- `README.md` course map (add the R row, P1.6, P3.0, the P5 rename and the Lane B weeks), `PROGRESS.md` (a week 0 row, P1.6 in week 6, P3.0 in week 13, Lane B weeks) and `site/mkdocs.yml` nav.
- Prerequisite lines: P0.1 README → R.1–R.3; P2.1 README and `aws.md` → R.6, R.7; P3.1 → P3.0; P3.2 gains a line saying "P2.1 already used this stack; now you learn how it works".
- Final checks: `python tools/build_lessons.py`, `node tools/check_lessons.js` (all lessons), `python tools/check_links.py`, `S2S_SOLUTIONS=1 uv run pytest course/R-readiness course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics`, and the C++ ctest runs.
