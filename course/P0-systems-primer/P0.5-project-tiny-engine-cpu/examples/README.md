# P0.5 examples

The engine is a project, so its code lives in the portfolio repo, [`platform/engine/v0/`](../../../../platform/engine/v0/README.md). The examples for this module are:

| # | File | Run | Hardware |
|---|---|---|---|
| 01 | [`platform/engine/v0/reference/llama_numpy.py`](../../../../platform/engine/v0/reference/llama_numpy.py): the reference forward pass | `uv run python platform/engine/v0/reference/llama_numpy.py --model build/tiny-llama` | T0 |
| 02 | [`platform/engine/v0/src/`](../../../../platform/engine/v0/src): the C++ engine | `cmake -S platform/engine/v0 -B build/engine-v0 && cmake --build build/engine-v0 -j` | T0 |
| 03 | [`03_sampling_demo.py`](03_sampling_demo.py): what temperature and top-p do to a distribution | `uv run python course/P0-systems-primer/P0.5-project-tiny-engine-cpu/examples/03_sampling_demo.py` | T0 |
