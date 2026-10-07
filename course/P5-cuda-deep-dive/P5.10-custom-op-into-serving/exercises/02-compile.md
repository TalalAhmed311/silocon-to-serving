# Exercise 2: make it `torch.compile`-safe (T0)

`test_torch_compile_fullgraph_cpu` compiles a tiny "block" that calls the op and then uses the mutated tensor. `fullgraph=True` turns any graph break into an error.

1. Make it pass (it should as shipped), then look at what Dynamo produced: `TORCH_LOGS=graph_code uv run --extra torch pytest … -k compile -s`. Find your op as a single node, with the mutation handled through a functionalized call.
2. Replace the custom op with a plain Python function doing the same math. Does it still compile with `fullgraph=True`? What does the graph look like now, and why is that a problem for a kernel you wanted to keep opaque?
3. On a GPU, wrap the block in `torch.cuda.graphs.make_graphed_callables` (or capture with `torch.cuda.CUDAGraph`) and confirm replay gives the same result. The JIT-built extension must not allocate during capture.
