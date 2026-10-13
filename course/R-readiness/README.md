# Start here: readiness for the course

This course teaches LLM inference from the silicon up. It does not teach programming. This page tells you what it assumes, lets you check yourself in 15 minutes, and sends you to the right readiness module (R.1 to R.7) for anything you miss.

## What the course assumes

- You can write a loop and a function in some language (Python, JavaScript, Java, anything).
- You remember school maths: algebra, powers, a bit of graphs. You do not need calculus.
- You have a Linux or macOS machine, or WSL on Windows, where you can install tools and open a terminal.

Everything else is taught, here or in the phases. If you have never met pointers, the stack and heap, a build system, NumPy, HTTP or a cloud account, do the matching R module before P0. Each R module opens with the same offer: if you can answer its checkpoints, skip to the next one.

## 15-minute self-check

Answer from memory, then open the answer. Two or three questions per module. If you miss any question in a module, do that module.

### R.1 C and C++ from zero

1. On a 64-bit machine, what is `sizeof(double)`, and what is the size of a pointer?
2. With `int a[4]; int* p = a;`, how many bytes past `a` does `p + 3` point?
3. Why do we usually pass a big object as `const T&` instead of `T`?

<details><summary>Answers</summary>

1. 8 bytes for `double`, and 8 bytes for a pointer on a 64-bit machine.
2. 12 bytes. `p + 3` steps three elements, and each `int` is 4 bytes.
3. `T` copies the whole object on every call. `const T&` passes an alias with no copy, and `const` stops the function from changing it.

</details>

If you missed any: do [R.1](R.1-cpp-from-zero/README.md).

### R.2 Memory: stack, heap and ownership

1. A function returns a pointer to one of its local variables. What is wrong?
2. Name the three classic heap bugs.
3. When does a `std::unique_ptr` free what it owns?

<details><summary>Answers</summary>

1. The local lives in the function's stack frame, and that frame is gone after the return. The pointer is dangling. Using it is undefined behaviour, even if it seems to work.
2. A leak (never freed), a double free (freed twice) and use-after-free (used after being freed).
3. In its destructor, when the owner goes out of scope. That is RAII: the free is tied to the object's lifetime.

</details>

If you missed any: do [R.2](R.2-memory-management/README.md).

### R.3 The terminal, git and building code

1. How do you see the exit code of the last command in a shell?
2. What does `./run 2>&1 | tee log.txt` do?
3. In git, which area holds the changes that will go into the next commit?

<details><summary>Answers</summary>

1. `echo $?`. Zero means success.
2. It sends the program's error output (stream 2) to the same place as its normal output (stream 1), then pipes both into `tee`, which prints them and also writes them to `log.txt`.
3. The staging area (the index), filled by `git add`.

</details>

If you missed any: do [R.3](R.3-terminal-git-build/README.md).

### R.4 Python and NumPy for this course

1. How many bytes does a float32 array of shape `(1000, 1000)` hold?
2. Is `a[:, 0]` a view or a copy?
3. What is the result shape of adding arrays of shape `(3, 1)` and `(1, 4)`?

<details><summary>Answers</summary>

1. 1,000 × 1,000 × 4 = 4,000,000 bytes.
2. A view. It shares memory with `a`. Use `.copy()` if you need independent data.
3. `(3, 4)`. Shapes are aligned from the right, and each size-1 dimension stretches.

</details>

If you missed any: do [R.4](R.4-python-and-numpy/README.md).

### R.5 The maths and ML you need

1. How many FLOPs does a matrix multiply of `(m, k)` by `(k, n)` take?
2. Why do we subtract the largest logit before computing softmax?
3. Does inference change a model's weights?

<details><summary>Answers</summary>

1. 2·m·k·n. Each output element is k multiply-adds, and each multiply-add counts as 2 FLOPs.
2. It stops `exp` overflowing for large logits. The result is unchanged, because the same factor cancels in the ratio.
3. No. Training changes the weights. Inference only reads them. This course is about inference.

</details>

If you missed any: do [R.5](R.5-maths-and-ml-basics/README.md).

### R.6 HTTP and networking basics

1. What does status code 429 mean, and what should a client do?
2. What does one server-sent event look like on the wire?
3. What is TTFT?

<details><summary>Answers</summary>

1. Too many requests. The client should wait and retry later, ideally with backoff and jitter.
2. A line starting `data: ` followed by the payload, ended by a blank line (`\n\n`).
3. Time to first token: from sending the request to receiving the first output token. Connection setup, queueing and prefill all count toward it.

</details>

If you missed any: do [R.6](R.6-http-and-networking/README.md).

### R.7 A safe cloud account

1. You stop a cloud GPU instance. Do you still pay for anything?
2. Why does this course never open port 22?
3. What does `terraform plan` do?

<details><summary>Answers</summary>

1. Yes. The disk (EBS volume) and other attached resources keep billing. Snapshots, NAT gateways, Elastic IPs and load balancers also bill while idle. Tear the stack down with `make down`.
2. An open SSH port is an attack surface. The course reaches instances through Session Manager (SSM), which needs no inbound port.
3. It shows the changes `apply` would make, and changes nothing.

</details>

If you missed any: do [R.7](R.7-cloud-account-and-cost-safety/README.md).

## The modules

All seven are Tier T0: a laptop, no GPU, no spend.

| Module | What it covers | Time (read + hands-on) | If you prefer a longer course |
|---|---|---|---|
| [R.1 C and C++ from zero](R.1-cpp-from-zero/README.md) | Compile and link, type sizes, pointers, arrays, references, `const`, headers, classes | 0.5 h + 4 h | [learncpp.com](https://www.learncpp.com/) (pointers, references, dynamic allocation, classes), [cppreference](https://en.cppreference.com/w/) |
| [R.2 Memory: stack, heap and ownership](R.2-memory-management/README.md) | Stack frames, `new`/`delete`, leaks, dangling pointers, AddressSanitizer, RAII, `unique_ptr`, copy vs move | 0.5 h + 4 h | [CS50x Week 4, Memory](https://cs50.harvard.edu/x/), [Clang AddressSanitizer docs](https://clang.llvm.org/docs/AddressSanitizer.html) |
| [R.3 The terminal, git and building code](R.3-terminal-git-build/README.md) | Shell, pipes, processes, git's model, Make vs CMake, `uv` and `pytest` | 0.5 h + 3 h | [The Missing Semester](https://missing.csail.mit.edu/), [Pro Git](https://git-scm.com/book/en/v2), [CMake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html) |
| [R.4 Python and NumPy for this course](R.4-python-and-numpy/README.md) | `ndarray`, dtype and bytes, views, strides, broadcasting, vectorising, `pytest` tolerances | 0.5 h + 3 h | [The Python tutorial](https://docs.python.org/3/tutorial/), [NumPy absolute basics](https://numpy.org/doc/stable/user/absolute_beginners.html) |
| [R.5 The maths and ML you need](R.5-maths-and-ml-basics/README.md) | Dot products, matmul cost, exp/log, softmax, layers, tokens to logits, sampling | 0.7 h + 3 h | [3Blue1Brown linear algebra](https://www.3blue1brown.com/topics/linear-algebra), [3Blue1Brown neural networks](https://www.3blue1brown.com/topics/neural-networks), [Karpathy, Zero to Hero](https://karpathy.ai/zero-to-hero.html) |
| [R.6 HTTP and networking basics](R.6-http-and-networking/README.md) | TCP, DNS, TLS, requests, status codes, JSON APIs, SSE, retries, `curl` | 0.5 h + 3 h | [MDN, An overview of HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview), [MDN, server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) |
| [R.7 A safe cloud account](R.7-cloud-account-and-cost-safety/README.md) | Root MFA, SSO, budgets, quotas, hidden costs, SSM, Terraform basics, teardown | 0.5 h + 2 h | [AWS Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html), [IAM Identity Center](https://docs.aws.amazon.com/singlesignon/latest/userguide/what-is.html), [Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html), [Terraform on AWS](https://developer.hashicorp.com/terraform/tutorials/aws-get-started) |

The external links were not opened when this page was written (see [SOURCES.md](../../SOURCES.md) section 4.1). Open each one before relying on it.

## R.7 comes before the first cloud run

Even if you skip every other readiness module, do [R.7](R.7-cloud-account-and-cost-safety/README.md) before [P2.1](../P2-serving-engines/P2.1-first-serve-with-vllm/README.md). P2.1 is the first lesson that starts a billed GPU. R.7 sets up the account, the budget alarm and the teardown habit that keep that safe.

## Where this leads

When you pass the self-check, start [P0](../P0-systems-primer/syllabus.md). The full list of modules, with objectives, is in the [readiness syllabus](syllabus.md).
