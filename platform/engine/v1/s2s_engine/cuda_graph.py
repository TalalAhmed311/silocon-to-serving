"""cuda_graph.py — capture TorchRunner.decode once per padded batch size and replay it (P6.6).

Decode at small batch is launch-bound: dozens of kernels per layer, each a few µs of work. A CUDA graph records the
whole launch sequence once; replay is one launch. The rules it imposes (all honoured by TorchRunner.decode):
  * static shapes  → pad the batch up to the next captured size (1, 2, 4, 8, 16, 32, …) and the block table to
                     max_blocks_per_seq;
  * static addresses → inputs are copied INTO persistent buffers; outputs are read OUT of a persistent buffer;
  * no host sync / data-dependent control flow inside → masks, not Python ifs.
Padded rows decode position 0 of block 0 — harmless garbage written to slot 0, which is why block 0 is reserved
(engine.py never hands it to a sequence when graphs are enabled).
TODO(run-on: T2): measure eager vs graph ITL at B ∈ {1, 8, 32} → exercise 4's table.
"""
from __future__ import annotations

import bisect


class DecodeGraphs:
    def __init__(self, runner, batch_sizes=(1, 2, 4, 8, 16, 32, 64)):
        import torch
        self.torch, self.runner = torch, runner
        self.sizes = sorted(batch_sizes)
        self.graphs, self.bufs = {}, {}
        pool = None
        for B in reversed(self.sizes):                       # largest first: smaller graphs reuse its memory pool
            dev = runner.device
            buf = {"tok": torch.zeros(B, dtype=torch.long, device=dev),
                   "pos": torch.zeros(B, dtype=torch.long, device=dev),
                   "tables": torch.zeros((B, runner.max_blocks), dtype=torch.long, device=dev)}
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):                        # warm-up outside capture (allocator, lazy init)
                for _ in range(2):
                    runner.decode(buf["tok"], buf["pos"], buf["tables"])
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                buf["out"] = runner.decode(buf["tok"], buf["pos"], buf["tables"])
            pool = g.pool()
            self.graphs[B], self.bufs[B] = g, buf

    def run(self, tok, pos, tables):
        n = tok.shape[0]
        i = bisect.bisect_left(self.sizes, n)
        if i == len(self.sizes):                              # bigger than anything captured: eager
            return self.runner.decode(tok, pos, tables)
        B = self.sizes[i]
        buf = self.bufs[B]
        buf["tok"].zero_(); buf["pos"].zero_(); buf["tables"].zero_()
        buf["tok"][:n].copy_(tok); buf["pos"][:n].copy_(pos); buf["tables"][:n].copy_(tables)
        self.graphs[B].replay()
        return buf["out"][:n]


def capture_decode(runner, batch_sizes=(1, 2, 4, 8, 16, 32, 64)) -> DecodeGraphs:
    runner.graphs = None
    graphs = DecodeGraphs(runner, batch_sizes)
    runner.graphs = graphs
    return graphs
