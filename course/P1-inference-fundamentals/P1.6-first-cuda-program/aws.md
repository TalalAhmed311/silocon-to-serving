# P1.6 on AWS

Use the single-GPU flow in the shared P5 guide: [`../../P5-cuda-deep-dive/aws-common.md`](../../P5-cuda-deep-dive/aws-common.md). It covers the instance (`g4dn.xlarge`, 1× T4, sm_75), cost, the Deep Learning AMI's CUDA and `nvcc`, SSM access with no open ports, the idle auto-stop, and teardown (`make down`). Do not duplicate it here.

This module needs about 1 to 2 instance-hours (an estimate). Look up the current on-demand price for your region before you start: hours × hourly price is the cost.

Differences from P5:

- Skip the Nsight counter setup (`ncu`). This module uses `compute-sanitizer`, which needs no special permission.
- Build only this module's folder:

```bash
cmake -S course/P1-inference-fundamentals/P1.6-first-cuda-program/exercises -B build/p1.6 -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build/p1.6 -j
ctest --test-dir build/p1.6 --output-on-failure        # TODO(run-on: T2): starters fail until you implement them
./build/p1.6/p1.6_01_device_info && ./build/p1.6/p1.6_02_transfer_tax
compute-sanitizer ./build/p1.6/p1.6_04-sanitizer-oob
```

No GPU budget? Do the kernel parts on [LeetGPU](https://leetgpu.com/) or in a Colab GPU runtime and come back for the sanitizer and the timing example. Teardown and auto-stop: see the shared page.
