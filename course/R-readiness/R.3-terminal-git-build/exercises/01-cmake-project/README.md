# Exercise 1 — Finish a CMake project (easy)

The folder holds a tiny project: `mathlib.hpp`/`mathlib.cpp` (three functions), `main.cpp` (a program that uses them) and `test_mathlib.cpp` (a test). `CMakeLists.txt` is incomplete. Fill in the three `TODO`s:

1. a **library target** `mathlib` with its folder as a public include directory,
2. link `app` to it,
3. register `test_mathlib` with CTest.

```bash
cd course/R-readiness/R.3-terminal-git-build/exercises/01-cmake-project
cmake -S . -B /tmp/r3-build                      # configure: reads CMakeLists.txt, writes the build system
cmake --build /tmp/r3-build -j                   # build: compile what is out of date, then link
ctest --test-dir /tmp/r3-build --output-on-failure
/tmp/r3-build/app                                # run the program
```

Before you start, run the build once. It fails at the **linker** with `undefined reference to 'add(int, int)'`. `main.cpp` has the declaration but nothing compiled the definition in `mathlib.cpp`. That error is the point of the exercise.

After you finish, edit `mathlib.cpp` and rebuild. Only `mathlib.cpp` recompiles, then the link steps run again. Edit `mathlib.hpp` and everything that includes it recompiles.

**Check from pytest** (configures and builds in a temporary folder):

```bash
uv run pytest course/R-readiness/R.3-terminal-git-build/exercises/01-cmake-project
S2S_SOLUTIONS=1 uv run pytest course/R-readiness/R.3-terminal-git-build/exercises/01-cmake-project   # the reference
```
