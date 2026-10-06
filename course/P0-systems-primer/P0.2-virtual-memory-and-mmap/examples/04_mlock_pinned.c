/* 04_mlock_pinned.c — page-locking memory: the CPU-side half of "pinned memory".
 * Run:      ./build/examples/04_mlock_pinned [MiB]   (default 256)
 * Expected: mlock succeeds (or fails with ENOMEM/EPERM if RLIMIT_MEMLOCK is small — the program prints the limit);
 *           memcpy speed from locked vs unlocked memory is about the same on a CPU.
 * Lesson:   pinning is not about CPU copy speed. It guarantees the physical pages stay put, which a DMA engine
 *           (GPU copy engine, NIC, NVMe) requires. cudaMallocHost = allocate + pin + register with the driver.
 *           The GPU measurement (pinned vs pageable cudaMemcpy) is P5.1 example 04.
 * Hardware: T0.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static double copy_gbs(void *dst, const void *src, size_t len) {
  memcpy(dst, src, len);                      /* warm-up */
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    double t0 = now_ms();
    memcpy(dst, src, len);
    double t = now_ms() - t0;
    if (t < best) best = t;
  }
  return len / 1e9 / (best / 1e3);
}

int main(int argc, char **argv) {
  size_t len = (size_t)(argc > 1 ? atol(argv[1]) : 256) << 20;
  struct rlimit rl;
  getrlimit(RLIMIT_MEMLOCK, &rl);
  printf("RLIMIT_MEMLOCK soft = %llu bytes\n", (unsigned long long)rl.rlim_cur);

  char *a = aligned_alloc(4096, len), *b = aligned_alloc(4096, len), *dst = aligned_alloc(4096, len);
  memset(a, 1, len); memset(b, 2, len); memset(dst, 0, len);

  int locked = mlock(b, len) == 0;
  if (!locked) printf("mlock failed: %s (raise the limit with `ulimit -l` or run with CAP_IPC_LOCK)\n", strerror(errno));

  printf("| source | memcpy GB/s |\n|---|---|\n");
  printf("| pageable | %.2f |\n", copy_gbs(dst, a, len));
  printf("| %s | %.2f |\n", locked ? "locked (mlock)" : "locked (FAILED, still pageable)", copy_gbs(dst, b, len));
  if (locked) munlock(b, len);
  free(a); free(b); free(dst);
  return 0;
}
