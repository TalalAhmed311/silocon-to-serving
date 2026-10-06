/* 02_first_touch.c — what mmap really costs: page faults on first touch, nothing on the second.
 * Run:      ./build/examples/02_first_touch [path] [MiB]   (default /tmp/s2s-1g.bin 1024; creates the file)
 * Expected: pass 1 (new mapping): many minor faults; pass 2 (same mapping): ~0 faults and much faster.
 *           Run once as root after `sync; echo 3 > /proc/sys/vm/drop_caches` to see major faults (disk reads).
 * Hardware: T0 (Linux or macOS).
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void faults(long *minor, long *major) {
  struct rusage u;
  getrusage(RUSAGE_SELF, &u);
  *minor = u.ru_minflt;
  *major = u.ru_majflt;
}

/* Touch one byte per page: the cheapest way to force every page to be mapped. */
static uint64_t touch(const volatile uint8_t *p, size_t len, size_t page) {
  uint64_t sum = 0;
  for (size_t off = 0; off < len; off += page) sum += p[off];
  return sum;
}

static void make_file(const char *path, size_t bytes) {
  struct stat st;
  if (stat(path, &st) == 0 && (size_t)st.st_size >= bytes) return;
  FILE *f = fopen(path, "wb");
  if (!f) { perror("fopen"); exit(1); }
  static uint8_t chunk[1 << 20];
  for (size_t i = 0; i < sizeof chunk; ++i) chunk[i] = (uint8_t)i;
  for (size_t w = 0; w < bytes; w += sizeof chunk) fwrite(chunk, 1, sizeof chunk, f);
  fclose(f);
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "/tmp/s2s-1g.bin";
  size_t mib = argc > 2 ? strtoul(argv[2], 0, 10) : 1024;
  size_t len = mib << 20, page = (size_t)sysconf(_SC_PAGESIZE);
  make_file(path, len);

  int fd = open(path, O_RDONLY);
  if (fd < 0) { perror("open"); return 1; }

  printf("page size = %zu bytes, file = %zu MiB, pages = %zu\n\n", page, mib, len / page);
  printf("| pass | ms | minor faults | major faults | GB/s (touched bytes = whole file) |\n|---|---|---|---|---|\n");

  for (int round = 0; round < 2; ++round) {
    /* A fresh mapping each round: round 0 may hit disk (if cache was dropped), round 1 hits the page cache. */
    uint8_t *p = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    for (int pass = 0; pass < 2; ++pass) {
      long mn0, mj0, mn1, mj1;
      faults(&mn0, &mj0);
      double t0 = now_ms();
      volatile uint64_t s = touch(p, len, page);
      double t1 = now_ms();
      faults(&mn1, &mj1);
      (void)s;
      printf("| mapping %d, pass %d | %.1f | %ld | %ld | %.2f |\n", round + 1, pass + 1, t1 - t0, mn1 - mn0, mj1 - mj0,
             (double)len / 1e9 / ((t1 - t0) / 1e3));
    }
    munmap(p, len);
  }
  close(fd);
  printf("\nPass 2 of each mapping has ~0 faults: the page-table entries already exist.\n");
  printf("If 'minor faults' < pages, the kernel mapped several pages per fault (fault-around).\n");
  return 0;
}
