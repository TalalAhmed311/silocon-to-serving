#include <cstdio>
#include <vector>

#include <s2s/check.hpp>
#include "prefetch.hpp"

static const char* kPath = "prefetch_test.bin";

static uint64_t checksum(const uint8_t* p, size_t n) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a
  for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
  return h;
}

static uint64_t read_checksum() {
  std::FILE* f = std::fopen(kPath, "rb");
  std::vector<uint8_t> buf(1 << 22);
  size_t n = std::fread(buf.data(), 1, buf.size(), f);
  std::fclose(f);
  return checksum(buf.data(), n);
}

S2S_TEST(prefetch_matches_read) {
  {
    std::FILE* f = std::fopen(kPath, "wb");
    for (int i = 0; i < (1 << 22); ++i) std::fputc((i * 131) & 0xFF, f);
    std::fclose(f);
  }
  const uint64_t want = read_checksum();
  for (bool pf : {false, true}) {
    Mapping m = map_with_prefetch(kPath, pf);
    CHECK_EQ(m.len, size_t(1 << 22));
    CHECK_EQ(checksum(m.p, m.len), want);
  }
  std::remove(kPath);
}

S2S_TEST_MAIN()
