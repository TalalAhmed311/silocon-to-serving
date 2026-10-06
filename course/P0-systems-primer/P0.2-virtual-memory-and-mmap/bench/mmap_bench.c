/* mmap_bench.c — load a weight-sized file three ways: read(), mmap+touch, mmap+MAP_POPULATE (Linux).
 * Run:      ./build/bench/mmap_bench [path] [MiB]     (default /tmp/s2s-1g.bin 1024)
 *           As root it first drops the page cache once and reports a "cold" row for each method.
 * Output:   Markdown table + results/mmap.json. "% of peak" is relative to the warm read() row.
 * Hardware: T0.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static int drop_caches(void) {           /* returns 1 if it worked (needs root on Linux) */
  sync();
  FILE *f = fopen("/proc/sys/vm/drop_caches", "w");
  if (!f) return 0;
  int ok = fputs("3\n", f) >= 0;
  fclose(f);
  return ok;
}

static uint64_t touch(const volatile uint8_t *p, size_t len) {
  uint64_t s = 0;
  for (size_t o = 0; o < len; o += 4096) s += p[o];
  return s;
}

static double m_read(const char *path, size_t len) {
  int fd = open(path, O_RDONLY);
  char *buf = malloc(len);
  double t0 = now_ms();
  size_t got = 0;
  while (got < len) { ssize_t r = read(fd, buf + got, len - got); if (r <= 0) break; got += (size_t)r; }
  double t = now_ms() - t0;
  free(buf); close(fd);
  return t;
}

static double m_mmap(const char *path, size_t len, int populate) {
  int fd = open(path, O_RDONLY);
  int flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
  if (populate) flags |= MAP_POPULATE;   /* pre-fault every page inside mmap() itself */
#else
  (void)populate;
#endif
  double t0 = now_ms();
  uint8_t *p = mmap(NULL, len, PROT_READ, flags, fd, 0);
  volatile uint64_t s = touch(p, len);
  double t = now_ms() - t0;              /* time mmap + touch: mmap alone would be misleadingly cheap */
  (void)s;
  munmap(p, len); close(fd);
  return t;
}

static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "/tmp/s2s-1g.bin";
  size_t len = (size_t)(argc > 2 ? atol(argv[2]) : 1024) << 20;
  struct stat st;
  if (stat(path, &st) != 0 || (size_t)st.st_size < len) {
    fprintf(stderr, "create the file first: ./build/examples/02_first_touch %s %zu\n", path, len >> 20);
    return 1;
  }
  const char *names[3] = {"read()", "mmap + touch", "mmap + MAP_POPULATE"};
  double cold[3] = {0}, med[3], p90[3];
  int have_cold = 1;
  for (int m = 0; m < 3; ++m) {
    if (have_cold && drop_caches()) cold[m] = m == 0 ? m_read(path, len) : m_mmap(path, len, m == 2);
    else have_cold = 0;
    double t[9];
    for (int r = 0; r < 9; ++r) t[r] = m == 0 ? m_read(path, len) : m_mmap(path, len, m == 2);  /* warm */
    qsort(t, 9, sizeof t[0], cmp);
    med[m] = t[4]; p90[m] = t[8];
  }
  const double gb = len / 1e9, peak = gb / (med[0] / 1e3);
  printf("| file size | variant | median ms | p90 ms | GB/s | %% of peak |\n|---|---|---|---|---|---|\n");
  FILE *js = (mkdir("results", 0755), fopen("results/mmap.json", "w"));
  fprintf(js, "{\"bench\": \"mmap\", \"unit\": \"GB/s\", \"rows\": [");
  for (int m = 0; m < 3; ++m) {
    double rate = gb / (med[m] / 1e3);
    printf("| %zu MiB | %s (warm) | %.1f | %.1f | %.2f | %.0f |\n", len >> 20, names[m], med[m], p90[m], rate, 100 * rate / peak);
    if (have_cold) printf("| %zu MiB | %s (cold) | %.1f | — | %.2f | %.0f |\n", len >> 20, names[m], cold[m], gb / (cold[m] / 1e3), 100 * gb / (cold[m] / 1e3) / peak);
    fprintf(js, "%s{\"label\": \"%s warm\", \"size\": %zu, \"median_ms\": %.3f, \"p90_ms\": %.3f, \"rate\": %.3f, \"pct_peak\": %.1f}",
            m ? ", " : "", names[m], len, med[m], p90[m], rate, 100 * rate / peak);
  }
  fprintf(js, "]}\n");
  fclose(js);
  if (!have_cold) printf("\n(no cold rows: dropping the page cache needs root on Linux)\n");
  return 0;
}
