// safetensors_view.hpp — a minimal, zero-copy safetensors reader (example quality; exercise 3 hardens it).
// Format: [u64 little-endian header_len][header_len bytes of JSON][data]. Offsets in the JSON are relative to data.
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace st {

struct TensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;  // relative to the data section
};

// RAII owner of one read-only mapping (P0.1): the destructor unmaps and closes.
class MappedFile {
 public:
  explicit MappedFile(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("open failed: " + path);
    struct stat s{};
    ::fstat(fd_, &s);
    size_ = size_t(s.st_size);
    if (size_ == 0) throw std::runtime_error("empty file");
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
    base_ = static_cast<const uint8_t*>(p);
  }
  ~MappedFile() {
    if (base_) ::munmap(const_cast<uint8_t*>(base_), size_);
    if (fd_ >= 0) ::close(fd_);
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  const uint8_t* data() const { return base_; }
  size_t size() const { return size_; }

 private:
  int fd_ = -1;
  size_t size_ = 0;
  const uint8_t* base_ = nullptr;
};

// --- a tiny JSON parser that understands exactly what safetensors headers contain ---
class Json {
 public:
  explicit Json(const char* s, size_t n) : p_(s), e_(s + n) {}
  void ws() { while (p_ < e_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_; }
  bool eat(char c) { ws(); if (p_ < e_ && *p_ == c) { ++p_; return true; } return false; }
  void expect(char c) { if (!eat(c)) throw std::runtime_error(std::string("json: expected '") + c + "'"); }
  std::string str() {
    expect('"');
    std::string out;
    while (p_ < e_ && *p_ != '"') {
      if (*p_ == '\\' && p_ + 1 < e_) ++p_;  // keep escaped char (names never need more than this)
      out += *p_++;
    }
    expect('"');
    return out;
  }
  int64_t integer() {
    ws();
    bool neg = eat('-');
    if (p_ >= e_ || *p_ < '0' || *p_ > '9') throw std::runtime_error("json: expected integer");
    int64_t v = 0;
    while (p_ < e_ && *p_ >= '0' && *p_ <= '9') v = v * 10 + (*p_++ - '0');
    return neg ? -v : v;
  }
  std::vector<int64_t> int_array() {
    std::vector<int64_t> v;
    expect('[');
    if (eat(']')) return v;
    do { v.push_back(integer()); } while (eat(','));
    expect(']');
    return v;
  }
  void skip_value() {  // used for __metadata__ (an object of strings)
    ws();
    if (eat('{')) { if (eat('}')) return; do { str(); expect(':'); skip_value(); } while (eat(',')); expect('}'); }
    else if (p_ < e_ && *p_ == '"') str();
    else if (eat('[')) { if (eat(']')) return; do { skip_value(); } while (eat(',')); expect(']'); }
    else integer();
  }
  bool at_end() { ws(); return p_ == e_; }

 private:
  const char* p_;
  const char* e_;
};

inline std::map<std::string, TensorInfo> parse_header(const char* s, size_t n) {
  std::map<std::string, TensorInfo> out;
  Json j(s, n);
  j.expect('{');
  if (j.eat('}')) return out;
  do {
    std::string name = j.str();
    j.expect(':');
    if (name == "__metadata__") { j.skip_value(); continue; }
    TensorInfo t;
    j.expect('{');
    do {
      std::string key = j.str();
      j.expect(':');
      if (key == "dtype") t.dtype = j.str();
      else if (key == "shape") t.shape = j.int_array();
      else if (key == "data_offsets") {
        auto o = j.int_array();
        if (o.size() != 2) throw std::runtime_error("data_offsets must have 2 entries");
        t.begin = uint64_t(o[0]); t.end = uint64_t(o[1]);
      } else j.skip_value();
    } while (j.eat(','));
    j.expect('}');
    out.emplace(std::move(name), std::move(t));
  } while (j.eat(','));
  j.expect('}');
  return out;
}

class File {
 public:
  explicit File(const std::string& path) : map_(path) {
    if (map_.size() < 8) throw std::runtime_error("file too small");
    uint64_t n = 0;
    std::memcpy(&n, map_.data(), 8);  // little-endian on every platform we target
    header_len_ = n;
    // Never let the parser read past the mapping, even if the header length is a lie.
    // (Exercise 3's validate() reports the lie; here we only stay memory-safe.)
    const uint64_t avail = map_.size() - 8;
    tensors_ = parse_header(reinterpret_cast<const char*>(map_.data() + 8), size_t(n < avail ? n : avail));
  }
  uint64_t file_size() const { return map_.size(); }
  uint64_t header_len() const { return header_len_; }
  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
  // Zero-copy: a pointer straight into the page cache.
  const uint8_t* bytes(const TensorInfo& t) const { return map_.data() + 8 + header_len_ + t.begin; }

 private:
  MappedFile map_;
  uint64_t header_len_ = 0;
  std::map<std::string, TensorInfo> tensors_;
};

}  // namespace st
