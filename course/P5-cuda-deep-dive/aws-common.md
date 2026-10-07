# P5 on AWS: the shared setup for every module

| | |
|---|---|
| Instances | **`g4dn.xlarge`** (1× T4, sm_75) for P5.1–P5.6 basics. **`g6.xlarge`** (1× L4, sm_89) for bf16/fp8, `cp.async` and `mma.sync` (P5.7–P5.8). Both via [`infra/aws/single-node`](../../infra/aws/single-node/README.md) |
| Cost | look up both on-demand prices in your region: `g4dn.xlarge ____ $/h`, `g6.xlarge ____ $/h`. Spot (`use_spot = true`) is fine: every step can be rerun. Each module's `aws.md` gives its hours |
| Toolchain | the Deep Learning AMI ships CUDA (check `nvcc --version` ≥ 12.4) and Nsight Systems/Compute (`nsys`, `ncu`). Install CMake ≥ 3.24 if the AMI's is older: `pip install cmake` |
| Safety | no inbound ports. Work in the SSM session (`make ssm`). Copy reports back with `make` targets or base64 |

## Build once per session

```bash
git clone https://github.com/talalahmed311/silocon-to-serving && cd silocon-to-serving
cmake -S course/P5-cuda-deep-dive -B build/p5 -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build/p5 -j
ctest --test-dir build/p5 --output-on-failure          # your starters: most tests FAIL until you implement them
cmake -S course/P5-cuda-deep-dive -B build/p5sol -DCMAKE_CUDA_ARCHITECTURES=native -DS2S_USE_SOLUTIONS=ON && \
  cmake --build build/p5sol -j && ctest --test-dir build/p5sol   # the references: should all pass
cmake -S platform/kernels -B build/d4 -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build/d4 -j && ctest --test-dir build/d4
```

## Nsight Compute counters (`ERR_NVGPUCTRPERM`)

`ncu` needs access to GPU performance counters, which are admin-only by default on Linux:

- quick: `sudo $(which ncu) …` (keep `PATH` and the binary path).
- persistent (needs a reboot or a driver reload): `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/ncu.conf`, then reboot the instance. This is the documented knob (NVIDIA "ERR_NVGPUCTRPERM" page, UNVERIFIED wording). It's acceptable on a single-user lab box, never on a shared host.

Copy reports back for the desktop UI: `ncu -o rung5 --set full ./build/p5/…` writes `rung5.ncu-rep`. `tar czf - *.ncu-rep | base64` in the SSM session, then decode locally. `.ncu-rep` and `.nsys-rep` are git-ignored.

## Teardown

```bash
make down && ../scripts/cost.sh       # in infra/aws/single-node
```

**Auto-stop:** the idle watchdog stops the instance after 30 idle minutes, and the CloudWatch backstop after 3 h. `make down` deletes the disk.
