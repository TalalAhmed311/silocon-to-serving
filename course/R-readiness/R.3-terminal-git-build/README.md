# R.3 — The terminal, git and building code

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) Linux or macOS, no GPU. On Windows use WSL2. |
| **Time** | ≈30 min reading + ≈3 h hands-on |
| **Prerequisites** | None beyond "can write a loop and a function". R.1 and R.2 help but are not required. |
| **You will build** | A finished CMake project, a shell pipeline, a scratch git repo with a resolved merge conflict |

## Learning objectives

By the end of this module you can:

1. Run a program from a shell, pass it arguments and read its **exit code**.
2. Move around the file system and search it with `cd`, `ls`, `cat`, `less`, `grep` and `find`.
3. Join programs with pipes and redirect their output and errors.
4. Read and set **environment variables** (why the course's tests read `S2S_SOLUTIONS`), list and stop processes, and say when `sudo` is needed.
5. Use git's four places (working tree, staging area, commits, branches) to save work, branch, merge and resolve a conflict.
6. Explain what a build system does, drive CMake (`-S`, `-B`, `--build`, `ctest`), and run the course's Python tests with `uv`.

## Why this matters for inference engineering

Every command in this course is typed into a terminal: building a kernel, running a benchmark, starting a server, tearing down a cloud stack. If a command fails, the exit code and the error text are your only clues. Git is how you keep your exercise work safe. A build system is why a 200-file C++ project does not recompile everything after a one-line change.

---

## 1. The shell runs programs

A **shell** (bash, zsh) reads a line, splits it into words, and runs the first word as a program with the rest as **arguments**.

```bash
ls -l course        # program: ls   arguments: -l  course
echo $?             # prints the exit code of the last command
```

When a program ends it returns an **exit code**, a number from 0 to 255. By convention `0` means success and anything else means failure. `$?` holds the last one. `&&` runs the next command only after success; `||` only after failure. CTest, `pytest` and `make` all report through exit codes.

## 2. Files and paths

A path is **absolute** if it starts with `/` (`/home/you/repo/README.md`) and **relative** otherwise (`course/R-readiness`), measured from the current directory. `.` is the current folder, `..` the parent, `~` your home.

| Command | Does |
|---|---|
| `pwd` | print the current directory |
| `cd path` | change directory |
| `ls -l` | list files with sizes |
| `cat file` | print a whole file |
| `less file` | page through it (`q` quits, `/word` searches) |
| `grep -rn "word" dir` | search inside files, recursively, with line numbers |
| `find dir -name "*.md"` | find files by name |

## 3. Pipes and redirection

Every program has three streams: **stdin** (input), **stdout** (output, stream 1) and **stderr** (errors, stream 2).

```bash
grep -rn "TODO(run" course | wc -l       # | sends stdout of one program into stdin of the next
./build/app > out.txt                     # > writes stdout to a file (overwrites)
./build/app >> out.txt                    # >> appends
./build/app > out.txt 2>&1                # 2>&1 sends errors to the same place as stdout
```

Small programs that each do one job, joined by pipes, are how you answer most questions about a repo. Exercise 2 is one.

## 4. Processes and environment variables

A running program is a **process** with an id (PID). `ps aux` lists them, `top` shows what uses the CPU, `kill PID` asks one to stop (`kill -9 PID` forces it). `Ctrl-C` stops the foreground program.

An **environment variable** is a named string every process inherits from its parent. `export NAME=value` sets it for the current shell and everything started from it. `NAME=value command` sets it for that one command.

```bash
export S2S_SOLUTIONS=1
uv run pytest course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/exercises/01-kv-size
```

The course's Python helper `load_impl` (in `course/common/python/s2s/exercise.py`) reads `S2S_SOLUTIONS`. If it is `1` the test loads the reference solution; otherwise it loads your starter. One variable switches between "check my work" and "check the answer key".

## 5. Permissions and `sudo`

Every file has an owner and read, write and execute bits (`ls -l` shows `-rwxr-xr-x`). A script needs the execute bit to run directly (`chmod +x run.sh`), or you can run it as `bash run.sh`.

Ordinary users cannot read other processes' hardware counters or change system settings. `sudo command` runs one command as the administrator. Profilers such as `perf` and Nsight Compute (`ncu`) sometimes need it, because they read counters the kernel protects. Rule: use `sudo` for the one command that needs it, never for building code, and never to "fix" a permission error you do not understand.

## 6. git's model

Git keeps your project's history as a chain of **commits**, each a snapshot with a message. Changes move through four places:

```
working tree  --git add-->  staging area  --git commit-->  repository (commits)
                                                              |  git push --> remote
```

| Command | Does |
|---|---|
| `git clone URL` | copy a remote repo to your machine |
| `git status` | what changed, what is staged |
| `git add file` | stage a change |
| `git commit -m "msg"` | save the staged changes as a commit |
| `git log --oneline --graph` | history |
| `git diff` / `git diff --staged` | unstaged / staged changes |
| `git switch -c name` (or `git checkout -b name`) | new branch and switch to it |
| `git merge name` | bring a branch's commits into the current one |
| `git push` | send your commits to the remote |

A **branch** is a movable label on a commit. Exercise 3 walks the whole cycle, including a merge conflict.

## 7. What a build system does

Compiling C++ takes two steps (R.1): each `.cpp` becomes an **object file**, then the **linker** joins object files into an executable. In a project of many files you do not want to recompile everything after one edit. A build system keeps a **dependency graph**: the executable depends on object files, each object depends on its `.cpp` and the headers it includes. After an edit it rebuilds only the targets downstream of what changed.

**Make** reads rules by hand. **CMake** is a generator: you describe targets in `CMakeLists.txt`, and it writes the Makefiles (or Ninja files) for you.

## 8. CMake in practice

```cmake
add_library(mathlib mathlib.cpp)                            # a target
target_include_directories(mathlib PUBLIC .)                # PUBLIC: users get it too
add_executable(app main.cpp)
target_link_libraries(app PRIVATE mathlib)                  # app uses mathlib
enable_testing()
add_test(NAME mathlib COMMAND test_mathlib)                 # CTest runs it, exit 0 = pass
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # configure: source dir -S, build dir -B
cmake --build build -j                            # build, using all cores
ctest --test-dir build --output-on-failure        # run the registered tests
```

`Release` turns on optimisation (`-O3`). `Debug` keeps symbols and disables optimisation, which is better for sanitizers and debuggers. Never time a Debug build. Keep build folders out of the repo (the course uses `build/` and `/tmp`).

## 9. The course's Python toolchain

```bash
uv sync                                  # create the virtual env and install pinned dependencies (uv.lock)
uv run pytest course/R-readiness/R.4-python-and-numpy   # run tests inside that env
S2S_SOLUTIONS=1 uv run pytest <path>     # run the same tests against the reference solutions
```

`uv run` means "run this command inside the project's environment", so you never activate anything by hand.

## 10. SSH versus SSM

To work on a cloud machine you need a shell on it. **SSH** opens a network port (22) on the machine and logs in with a key. **Session Manager (SSM)** reaches the machine through the cloud provider's API instead, with no open port. This course uses **SSM only** and never opens port 22. R.7 sets up the account and P2.1 is the first time you use it.

---

## Exercises

```bash
uv run pytest course/R-readiness/R.3-terminal-git-build                # your versions
S2S_SOLUTIONS=1 uv run pytest course/R-readiness/R.3-terminal-git-build  # the reference
```

| # | Exercise | Difficulty | Check |
|---|---|---|---|
| 1 | [Finish a CMake project](exercises/01-cmake-project/README.md) | easy | pytest configures, builds and runs ctest |
| 2 | [A shell pipeline](exercises/02-shell-pipeline/README.md) | easy | pytest runs your script on a fixture tree |
| 3 | [A git checklist](exercises/03-git-checklist/README.md) | easy | none: match the `git log --graph` shape |

## Common mistakes

- **Forgetting that `>` overwrites.** Use `>>` to append.
- **Piping and losing errors.** `|` carries stdout only; add `2>&1` to include stderr.
- **Committing build folders.** Add them to `.gitignore`.
- **Reading a conflict as a failure.** Conflict markers are git asking you a question. Edit, `git add`, `git commit`.
- **Running `sudo cmake`.** Root-owned build files break later builds.

## Go further

- MIT, *The Missing Semester of Your CS Education* (shell, scripting, version control): https://missing.csail.mit.edu/
- *Pro Git* (the book): https://git-scm.com/book/en/v2
- The official CMake tutorial: https://cmake.org/cmake/help/latest/guide/tutorial/index.html

All three links are UNVERIFIED from the build environment (no outbound access).

## Next

[R.4 — Python and NumPy for this course](../R.4-python-and-numpy/README.md)
