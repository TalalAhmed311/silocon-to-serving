// Exercise 4 starter.
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <string>

struct Mapping {
  const uint8_t* p = nullptr;
  size_t len = 0;
  int fd = -1;
  ~Mapping() { if (p) munmap(const_cast<uint8_t*>(p), len); if (fd >= 0) close(fd); }
  Mapping() = default;
  Mapping(Mapping&& o) noexcept : p(o.p), len(o.len), fd(o.fd) { o.p = nullptr; o.fd = -1; }
  Mapping(const Mapping&) = delete;
};

inline Mapping map_with_prefetch(const std::string& path, bool prefetch) {
  Mapping m;
  (void)path; (void)prefetch;
  // TODO: open, fstat, mmap read-only; if prefetch, madvise(MADV_WILLNEED)
  throw std::runtime_error("not implemented");
  return m;
}
