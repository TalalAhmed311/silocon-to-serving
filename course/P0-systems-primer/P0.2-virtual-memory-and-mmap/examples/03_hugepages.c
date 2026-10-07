/* 03_hugepages.c — random access over a large buffer with 4 KiB pages vs transparent huge pages (THP).
 * Run:      ./build/examples/03_hugepages [MiB]   (default 2048)
 * Expected: on Linux with THP enabled ("madvise" or "always" in /sys/kernel/mm/transparent_hugepage/enabled),
 *           the MADV_HUGEPAGE run is faster for random access because far fewer TLB misses occur.
 *           On macOS the hint does not exist; the program reports that and runs the 4 KiB variant only.
 * Hardware: T0.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* xorshift: a cheap PRNG so the RNG itself doesn't dominate the timing. */
static inline uint64_t next(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

static double run(size_t len, int huge) {
  uint8_t *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { perror("mmap"); exit(1); }
#ifdef MADV_HUGEPAGE
  if (huge) madvise(p, len, MADV_HUGEPAGE);  /* a hint: the kernel may or may not honor it */
#endif
  memset(p, 1, len);                          /* fault everything in before timing */
  uint64_t s = 88172645463325252ull, sum = 0;
  const size_t n = 20u * 1000 * 1000;
  double t0 = now_ms();
  for (size_t i = 0; i < n; ++i) sum += p[next(&s) % len];
  double t1 = now_ms();
  munmap(p, len);
  if (sum == 42) puts("");                    /* keep `sum` alive */
  return (t1 - t0) * 1e6 / n;                 /* ns per access */
}

int main(int argc, char **argv) {
  size_t len = (size_t)(argc > 1 ? atol(argv[1]) : 2048) << 20;
  printf("| pages | ns per random access |\n|---|---|\n");
  printf("| 4 KiB | %.1f |\n", run(len, 0));
#ifdef MADV_HUGEPAGE
  printf("| THP (MADV_HUGEPAGE) | %.1f |\n", run(len, 1));
  printf("\nCheck AnonHugePages in /proc/meminfo during the run to confirm THP was used.\n");
#else
  printf("| THP | not available on this OS |\n");
#endif
  return 0;
}
