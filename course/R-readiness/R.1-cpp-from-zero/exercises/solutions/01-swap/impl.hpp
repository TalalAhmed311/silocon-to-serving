#pragma once

inline void swap_by_pointer(int* a, int* b) {
  int tmp = *a;
  *a = *b;
  *b = tmp;
}

inline void swap_by_ref(int& a, int& b) {
  int tmp = a;
  a = b;
  b = tmp;
}
