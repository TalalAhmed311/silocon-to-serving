# R — Readiness: the skills the course assumes

**Week 0 (before P0) · Lane A ≈ 26 h, skip what you already know · Tier T0 (laptop, no GPU, no spend) · Lane B: not started**

**Why this phase exists.** Later phases quietly rely on skills that are never taught there: pointers and the heap, a build system, NumPy, the maths behind softmax, HTTP, and a safe cloud account. R covers them in seven short modules. Each one is a lesson and a self-check at once. If you can answer its checkpoints, skip to the next module. Start with the [15-minute self-check](README.md) to see which ones you need.

**What it assumes.** You can write a loop and a function in some language, and you remember school maths. Nothing else.

**One hard rule.** [R.7](R.7-cloud-account-and-cost-safety/README.md) must be done before P2.1, the first lesson that starts a billed GPU.

| Module | Time (read + hands-on) | Exercises | Simulator |
|---|---|---|---|
| R.1 C and C++ from zero | 0.5 h + 4 h | swap, array sum, `Vec2`, split header | **pointer arithmetic explorer** |
| R.2 Memory: stack, heap and ownership | 0.5 h + 4 h | fix a leak, dangling return, owning buffer, read an ASan report | **stack and heap timeline** |
| R.3 The terminal, git and building code | 0.5 h + 3 h | CMake project, shell pipeline, git checklist | **incremental build** |
| R.4 Python and NumPy for this course | 0.5 h + 3 h | shapes and bytes, views, broadcast, vectorise | **broadcasting checker** |
| R.5 The maths and ML you need | 0.7 h + 3 h | dot and matvec, softmax, MLP forward, next token | **softmax and temperature** |
| R.6 HTTP and networking basics | 0.5 h + 3 h | parse SSE, TTFT and ITL, tiny API, retry policy | **request timeline** |
| R.7 A safe cloud account | 0.5 h + 2 h | cost calculator, account checklist, tfvars hygiene | **what will this cost?** |

Times are estimates until the exercises have been run (see [GAPS.md](../../GAPS.md) section C).

---

## R.1 — C and C++ from zero

**Objectives.** After this module you can:

1. Explain the path source, compiler, object file, linker, executable, and why C++ needs a build step.
2. Say the size of `char`, `int`, `float`, `double` and a pointer on a 64-bit machine.
3. Take an address with `&x`, hold it in a pointer and read through it with `*p`.
4. Explain that `a[i]` is `*(a + i)` and that pointer arithmetic steps by `sizeof(T)`.
5. Choose between pass by value, by reference and by pointer, and use `const T&` for big arguments.
6. Split code into a header (declaration) and a source file (definition), and say what `#include` really does.
7. Write a small `struct` or `class` with a constructor, a destructor and a `const` method.
8. Use `std::vector`, `std::string`, `std::array` and range-for, and describe undefined behaviour with two examples.

**Exercises** (C++, ctest). (1) `swap_by_pointer` and `swap_by_ref` · (2) sum and max of a raw array with pointer arithmetic, including `n == 0` · (3) a `Vec2` class with `length()`, `operator+` and a `const` method · (4) split `stats.hpp` and `stats.cpp` into a library the test links.

**Sources.** learncpp.com (pointers, references, dynamic allocation, classes) · cppreference. UNVERIFIED.

**Go up to:** R.2.

## R.2 — Memory: stack, heap and ownership

**Objectives.**

1. Draw a process's memory: code, globals, heap and stack.
2. Explain stack frames, and why returning a pointer to a local is a dangling pointer.
3. Use `new`/`delete`, `new[]`/`delete[]` and `malloc`/`free`, and say who owns each block.
4. Recognise a leak, a double free and a use-after-free.
5. Build with `-fsanitize=address -g` and read the first lines of an AddressSanitizer report.
6. Apply RAII: tie a free to a destructor, with exactly one owner.
7. Use `std::unique_ptr`, say when `std::shared_ptr` is the wrong tool on a hot path, and explain copy vs move for a vector's heap buffer.

**Exercises** (C++, ctest, built with `-fsanitize=address,undefined`). (1) Fix a leak on an early-return path with `std::unique_ptr` · (2) replace a returned pointer to a local with a return by value · (3) a `FloatBuffer` class with a destructor, deleted copy and implemented move, tested by counting live allocations · (4) read ASan's report for a planted heap overflow, paste its first line into a comment and fix it.

**Sources.** CS50x Week 4 (Memory) · Clang AddressSanitizer docs · learncpp. UNVERIFIED.

**Go up to:** P0.1, which treats alignment and move semantics in depth.

## R.3 — The terminal, git and building code

**Objectives.**

1. Run programs from a shell and read their exit codes (`$?`).
2. Move around files and search them: `cd`, `ls`, `cat`, `less`, `grep`, `find`.
3. Join commands with pipes and redirect streams (`|`, `>`, `2>&1`).
4. Inspect and stop processes, and set environment variables such as `S2S_SOLUTIONS=1`.
5. Explain why some profilers need `sudo`.
6. Use git's model (working tree, staging, commit, branch) and the commands `clone`, `status`, `add`, `commit`, `log`, `diff`, `checkout -b`, `push`.
7. Explain what a build system does, and configure, build and test a CMake project.
8. Use the course's Python toolchain: `uv sync`, `uv run pytest <path>`.
9. Say how SSH differs from SSM. The course uses SSM only.

**Exercises.** (1) Finish an incomplete `CMakeLists.txt`: library target, link, test (C++) · (2) write `pipeline.sh` that counts `TODO(run` markers per top-level folder (pytest runs it) · (3) a git checklist on a scratch repo, ending in a resolved merge conflict (Markdown, no test).

**Sources.** The Missing Semester · Pro Git · the official CMake tutorial. UNVERIFIED.

**Go up to:** R.4, then P0.1's CMake section.

## R.4 — Python and NumPy for this course

**Objectives.**

1. Explain why the course pairs Python reference code with C++ and CUDA.
2. Write small Python functions with dicts, comprehensions, `dataclass` and type hints.
3. Describe an `ndarray` as a typed flat buffer plus a shape, and compute its bytes from shape and dtype.
4. Tell a view from a copy, and show it with `np.shares_memory`.
5. Explain that reshape and transpose change strides, not data.
6. Predict the result shape of a broadcast, or the error.
7. Replace a Python loop with one array expression, and measure it with `time.perf_counter`.
8. Give the shapes and FLOPs of a matrix multiply, and compare floats with `np.testing.assert_allclose`.

**Exercises** (Python). (1) `nbytes(shape, dtype)` and `matmul_flops(m, k, n)` · (2) return views, checked with `np.shares_memory` · (3) `normalize_rows` without loops · (4) rewrite a loop-based pairwise distance as a broadcast, matching within `rtol=1e-6`.

**Sources.** The Python tutorial · NumPy absolute basics. UNVERIFIED.

**Go up to:** R.5; P0.1 (strides); P0.5 (the NumPy reference engine).

## R.5 — The maths and ML you need

**Objectives.**

1. Read a dot product as "how aligned", and a matrix times a vector as many dot products.
2. Count FLOPs: 2 per multiply-add, and 2·m·k·n for a matrix multiply.
3. Use exp and log, and explain why exp turns scores into positive weights.
4. Compute softmax, and its numerically stable form (subtract the max).
5. Write a neuron and a layer as `y = activation(Wx + b)`, and count a small MLP's parameters.
6. Say what training changes and what inference only reads.
7. Follow text to tokens, ids, embeddings and scores over the vocabulary, to the next token.
8. Describe greedy decoding and temperature sampling.

This module does not teach the transformer. [P1.1](../P1-inference-fundamentals/P1.1-transformer-forward-pass/README.md) does.

**Exercises** (Python). (1) dot product and matvec with loops, compared with `@` · (2) stable softmax over the last axis, tested on logits like `[1000, 1001]` · (3) a 2-layer MLP forward pass with SiLU, plus `param_count` · (4) greedy token and top-k tokens with probabilities.

**Sources.** 3Blue1Brown, Essence of linear algebra and Neural networks · Karpathy, Neural Networks: Zero to Hero. UNVERIFIED.

**Go up to:** P1.1; P0.5 (sampling).

## R.6 — HTTP and networking basics

**Objectives.**

1. Explain client and server, IP address, port and `localhost:8000`.
2. Describe TCP as a reliable byte stream with a setup round trip, and separate latency from bandwidth.
3. Say what DNS and TLS cost before the first byte.
4. Read an HTTP request and response, and a `curl -v` trace.
5. Write an OpenAI-style `/v1/chat/completions` request body.
6. Say what 200, 400, 401, 404, 429, 500 and 503 mean for a serving gateway.
7. Read a server-sent events stream, and mark TTFT and inter-token gaps on it.
8. Decide when a retry is safe (idempotency), with backoff and jitter.
9. Explain why a server needs async I/O or many workers for many open connections.
10. Use `curl`, `curl -N` and FastAPI's `TestClient`.

**Exercises** (Python). (1) parse a raw SSE byte stream, including multi-line `data:`, comments and `[DONE]` · (2) TTFT, mean ITL and p90 ITL from chunk timestamps · (3) a FastAPI `POST /v1/echo` that validates its body and returns 400 or 422 on bad input · (4) `should_retry(status, attempt, idempotent)` and a bounded backoff with jitter.

**Sources.** MDN, An overview of HTTP · MDN, Using server-sent events. UNVERIFIED.

**Go up to:** R.7; P1.3 (TTFT and ITL); P2.7 (the gateway).

## R.7 — A safe cloud account

**Objectives.**

1. Estimate a cloud bill as hours × hourly price, using an example value and checking the current price page.
2. Lock the root user with MFA and use IAM Identity Center (SSO) for daily work.
3. Choose a region, and request a GPU vCPU quota before you need it.
4. Create an AWS Budget with an email alert, and apply the repo's `infra/aws/guardrails` stack.
5. List what bills while "off": disks of stopped instances, snapshots, NAT gateways, Elastic IPs, load balancers.
6. Reach an instance through Session Manager without opening port 22.
7. Read `terraform plan`, `apply` and `destroy`, and say what state is. P3.2 teaches Terraform properly.
8. Run the course's stack lifecycle (`make up`, `make ssm`, `make down`, idle auto-stop) and finish with a teardown checklist.
9. Keep secrets out of git: SSO login, environment variables and a `.gitignore` for `terraform.tfvars`.

**Exercises.** (1) `session_cost`, `idle_cost` and `budget_ok` (Python; example prices appear only in the tests) · (2) an account checklist (Markdown, no test) · (3) `check_tfvars` and `gitignore_covers` (Python).

**Sources.** AWS Budgets user guide · IAM Identity Center user guide · Systems Manager Session Manager · HashiCorp, Get started with Terraform on AWS. UNVERIFIED. `infra/aws/README.md` and `infra/aws/guardrails/` in this repo.

**Go up to:** P0.1 (or straight to P2.1 if you only need the cloud part).
