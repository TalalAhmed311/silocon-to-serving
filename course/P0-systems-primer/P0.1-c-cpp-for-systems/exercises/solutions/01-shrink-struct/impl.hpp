// Exercise 1 solution: largest alignment first. 8 + 8 + 4 + 4 + 2 + 1 + 1 = 28, padded to 32 (multiple of 8).
#pragma once
#include <cstdint>

struct RequestRecord {
  uint64_t request_id;
  double arrival_time_s;
  uint32_t prompt_tokens;
  uint32_t output_tokens;
  uint16_t priority;
  uint8_t state;
  bool streaming;
};
