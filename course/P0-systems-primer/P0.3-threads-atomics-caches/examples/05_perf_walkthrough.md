# Reading `perf` output (Linux)

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-load-misses \
  ./build/examples/02_false_sharing 4
```

| Counter | What it tells you | False-sharing signature |
|---|---|---|
| `instructions / cycles` (IPC) | how busy the core is; below ~1 means it is stalled | IPC collapses in the packed run |
| `cache-misses` | last-level misses, roughly | rises for the packed run even though the data is tiny |
| `L1-dcache-load-misses` | L1 misses | very high relative to the loads: the line keeps getting invalidated |

**Find the culprit line directly.** `perf c2c record ./app; perf c2c report --stdio` lists cache lines with "HITM" (hit-modified) events. These are loads that found the line modified in *another* core's cache. A packed counter array shows up as one line with many HITMs from several CPUs.

**Flame graphs.** `perf record -g ./app`, then `perf script | stackcollapse-perf.pl | flamegraph.pl > out.svg` (scripts from Brendan Gregg's FlameGraph repository). The width of each box is the share of samples in that call stack.

**No permission?** Set `sudo sysctl kernel.perf_event_paranoid=1`, or run as root. On cloud VMs some hardware counters are unavailable, and `perf stat` prints `<not supported>`.

**macOS.** Instruments → *CPU Counters* (L1/L2 misses) and *Time Profiler*.
