/* 01_proc_vs_thread.c — cost of creating (and joining) a process vs a thread.
 * Run:      ./build/examples/01_proc_vs_thread [N]   (default 2000)
 * Expected: thread create+join is roughly 10x cheaper than fork+waitpid. Exact µs depend on your machine.
 * Hardware: T0 (Linux/macOS).
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void *noop(void *arg) { return arg; }

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 2000;
  double t0 = now_us();
  for (int i = 0; i < n; ++i) {
    pid_t p = fork();
    if (p == 0) _exit(0);          /* child: exit immediately, no atexit handlers */
    waitpid(p, NULL, 0);
  }
  double t1 = now_us();
  for (int i = 0; i < n; ++i) {
    pthread_t th;
    pthread_create(&th, NULL, noop, NULL);
    pthread_join(th, NULL);
  }
  double t2 = now_us();
  printf("| primitive | µs per create+join |\n|---|---|\n");
  printf("| fork + waitpid | %.1f |\n| pthread_create + join | %.1f |\n", (t1 - t0) / n, (t2 - t1) / n);
  return 0;
}
