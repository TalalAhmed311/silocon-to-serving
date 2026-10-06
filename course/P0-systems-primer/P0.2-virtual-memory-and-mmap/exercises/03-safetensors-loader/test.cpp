// Run make_test_files.py first; the directory is passed as argv[1] (CMake does this for you).
#include <string>

#include <s2s/check.hpp>
#include "validate.hpp"

static std::string dir = "build/st-test";

static bool rejected(const std::string& name) {
  try {
    st::File f(dir + "/" + name);
    st::validate(f);
    return false;
  } catch (const std::exception&) {
    return true;  // either the constructor or validate() refused it: both count as safe
  }
}

S2S_TEST(good_file_counts_params) {
  st::File f(dir + "/good.safetensors");
  CHECK_EQ(st::validate(f), int64_t(16 + 8 + 3));
}
S2S_TEST(rejects_bad_header_len) { CHECK(rejected("bad_header_len.safetensors")); }
S2S_TEST(rejects_bad_dtype) { CHECK(rejected("bad_dtype.safetensors")); }
S2S_TEST(rejects_past_end) { CHECK(rejected("bad_past_end.safetensors")); }
S2S_TEST(rejects_size_mismatch) { CHECK(rejected("bad_size_mismatch.safetensors")); }
S2S_TEST(rejects_overlap) { CHECK(rejected("bad_overlap.safetensors")); }
S2S_TEST(rejects_misaligned) { CHECK(rejected("bad_misaligned.safetensors")); }

int main(int argc, char** argv) {
  if (argc > 1) dir = argv[1];
  return s2s::run_all();
}
