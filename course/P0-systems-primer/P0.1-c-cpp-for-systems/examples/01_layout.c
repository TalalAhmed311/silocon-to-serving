/* 01_layout.c — struct padding and alignment, checked by the compiler.
 * Run:      ./build/examples/01_layout
 * Expected: Bad sizeof=24, Good sizeof=16, Line sizeof=64 on x86-64 / arm64 (LP64 or LLP64 ABIs).
 * Hardware: T0.
 */
#include <stdalign.h>
#include <stddef.h>
#include <stdio.h>

struct Bad {   /* 1 + 7 (pad so `value` is 8-aligned) + 8 + 1 + 7 (tail pad to a multiple of 8) */
  char tag;
  double value;
  char flag;
};

struct Good {  /* largest alignment first: 8 + 1 + 1 + 6 tail pad */
  double value;
  char tag;
  char flag;
};

/* A counter that owns a whole cache line, so another core's counter can never share it (P0.3). */
struct Line {
  alignas(64) long count;
};

/* Fail the build, not the run, if an assumption is wrong on some exotic ABI. */
_Static_assert(sizeof(struct Good) <= sizeof(struct Bad), "reordering must not grow the struct");
_Static_assert(sizeof(struct Line) == 64, "alignas(64) pads the struct to one cache line");

int main(void) {
  printf("Bad   sizeof=%zu offsets: tag=%zu value=%zu flag=%zu\n", sizeof(struct Bad),
         offsetof(struct Bad, tag), offsetof(struct Bad, value), offsetof(struct Bad, flag));
  printf("Good  sizeof=%zu offsets: value=%zu tag=%zu flag=%zu\n", sizeof(struct Good),
         offsetof(struct Good, value), offsetof(struct Good, tag), offsetof(struct Good, flag));
  printf("Line  sizeof=%zu (alignas(64))\n", sizeof(struct Line));

  /* An array of N structs costs N * sizeof — padding is paid N times. */
  size_t n = 1000000;
  printf("1M records: Bad=%zu KB, Good=%zu KB\n", n * sizeof(struct Bad) / 1024, n * sizeof(struct Good) / 1024);
  return 0;
}
