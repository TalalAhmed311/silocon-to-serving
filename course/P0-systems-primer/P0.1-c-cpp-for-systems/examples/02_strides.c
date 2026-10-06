/* 02_strides.c — one buffer, three views, chosen only by strides.
 * Run:      ./build/examples/02_strides
 * Expected: the three printed matrices; the transposed view is the row-major one flipped.
 * Hardware: T0.
 */
#include <stdio.h>

typedef struct {
  const float *data;
  int rows, cols;
  long stride0, stride1; /* elements to step for +1 row, +1 column */
} View;

static float at(View v, int i, int j) { return v.data[i * v.stride0 + j * v.stride1]; }

static void print(const char *name, View v) {
  printf("%s (%dx%d, strides=%ld,%ld)\n", name, v.rows, v.cols, v.stride0, v.stride1);
  for (int i = 0; i < v.rows; ++i) {
    for (int j = 0; j < v.cols; ++j) printf("%5.0f", at(v, i, j));
    printf("\n");
  }
}

int main(void) {
  enum { R = 2, C = 3 };
  float buf[R * C] = {0, 1, 2, 3, 4, 5}; /* the bytes never change below */

  View row_major = {buf, R, C, C, 1};
  View col_major = {buf, R, C, 1, R};   /* same bytes interpreted column-major */
  View transposed = {buf, C, R, 1, C};  /* transpose of row_major: swap shape and strides */

  print("row-major", row_major);
  print("column-major", col_major);
  print("transpose of row-major", transposed);

  /* Walking `transposed` row by row jumps C floats per step: on a real 4096-wide matrix that is
   * 16 KB per step, a new cache line every access. That is the cost .contiguous() pays once. */
  return 0;
}
