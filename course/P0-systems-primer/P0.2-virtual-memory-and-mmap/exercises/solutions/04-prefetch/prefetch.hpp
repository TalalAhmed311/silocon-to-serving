// Exercise 4 solution.
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
  m.fd = ::open(path.c_str(), O_RDONLY);
  if (m.fd < 0) throw std::runtime_error("open " + path);
  struct stat s{};
  ::fstat(m.fd, &s);
  m.len = size_t(s.st_size);
  void* p = ::mmap(nullptr, m.len, PROT_READ, MAP_PRIVATE, m.fd, 0);
  if (p == MAP_FAILED) throw std::runtime_error("mmap");
  m.p = static_cast<const uint8_t*>(p);
  // WILLNEED starts asynchronous readahead into the page cache and returns immediately, so the
  // caller can overlap other setup (tokenizer load, allocation) with the disk reads.
  if (prefetch) ::madvise(p, m.len, MADV_WILLNEED);
  return m;
}
