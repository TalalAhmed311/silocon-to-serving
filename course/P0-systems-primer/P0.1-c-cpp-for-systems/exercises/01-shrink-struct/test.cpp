#include <cstddef>
#include <type_traits>

#include <s2s/check.hpp>
#include "impl.hpp"

// Every field must still exist with its original type.
static_assert(std::is_same_v<decltype(RequestRecord::request_id), uint64_t>);
static_assert(std::is_same_v<decltype(RequestRecord::arrival_time_s), double>);
static_assert(std::is_same_v<decltype(RequestRecord::prompt_tokens), uint32_t>);
static_assert(std::is_same_v<decltype(RequestRecord::output_tokens), uint32_t>);
static_assert(std::is_same_v<decltype(RequestRecord::priority), uint16_t>);
static_assert(std::is_same_v<decltype(RequestRecord::state), uint8_t>);
static_assert(std::is_same_v<decltype(RequestRecord::streaming), bool>);

S2S_TEST(size_is_32) { CHECK_EQ(sizeof(RequestRecord), size_t(32)); }

S2S_TEST(alignment_is_8) { CHECK_EQ(alignof(RequestRecord), size_t(8)); }

S2S_TEST(fields_round_trip) {
  RequestRecord r{};
  r.request_id = 42; r.arrival_time_s = 1.5; r.prompt_tokens = 100; r.output_tokens = 7;
  r.priority = 3; r.state = 2; r.streaming = true;
  CHECK(r.request_id == 42 && r.arrival_time_s == 1.5 && r.prompt_tokens == 100);
  CHECK(r.output_tokens == 7 && r.priority == 3 && r.state == 2 && r.streaming);
}

S2S_TEST_MAIN()
