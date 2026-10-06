// Exercise 1 starter. TODO: reorder the fields so sizeof(RequestRecord) == 32.
#pragma once
#include <cstdint>

struct RequestRecord {
  bool streaming;          // 1 byte
  uint64_t request_id;     // 8
  uint16_t priority;       // 2
  double arrival_time_s;   // 8
  uint8_t state;           // 1
  uint32_t prompt_tokens;  // 4
  uint32_t output_tokens;  // 4
};
