// Exercise 3 solution.
#pragma once
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../examples/safetensors_view.hpp"

namespace st {

inline int64_t dtype_size(const std::string& d) {
  static const std::map<std::string, int64_t> sizes = {{"F64", 8}, {"I64", 8}, {"F32", 4}, {"I32", 4}, {"F16", 2},
                                                       {"BF16", 2}, {"I16", 2}, {"I8", 1}, {"U8", 1}, {"BOOL", 1}};
  auto it = sizes.find(d);
  if (it == sizes.end()) throw std::runtime_error("unknown dtype " + d);
  return it->second;
}

// Shape products come from an untrusted file: a shape like [2^40, 2^40] must not wrap around.
inline int64_t checked_numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d < 0) throw std::runtime_error("negative dimension");
    if (d != 0 && n > std::numeric_limits<int64_t>::max() / d) throw std::runtime_error("shape overflow");
    n *= d;
  }
  return n;
}

inline int64_t validate(const File& f) {
  const uint64_t file_size = f.file_size(), header_len = f.header_len();
  if (header_len > file_size - 8) throw std::runtime_error("header length past end of file");  // check 1
  const uint64_t data_size = file_size - 8 - header_len;
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  int64_t total = 0;
  for (auto& [name, t] : f.tensors()) {
    const int64_t es = dtype_size(t.dtype);                                                       // check 2
    if (t.end < t.begin || t.end > data_size) throw std::runtime_error(name + ": range outside data");  // 3
    const int64_t n = checked_numel(t.shape);
    if (n > std::numeric_limits<int64_t>::max() / es || uint64_t(n * es) != t.end - t.begin)
      throw std::runtime_error(name + ": byte size does not match shape");                       // check 4
    if ((8 + header_len + t.begin) % uint64_t(es) != 0) throw std::runtime_error(name + ": misaligned");  // 6
    if (t.end > t.begin) ranges.emplace_back(t.begin, t.end);
    total += n;
  }
  std::sort(ranges.begin(), ranges.end());
  for (size_t i = 1; i < ranges.size(); ++i)
    if (ranges[i].first < ranges[i - 1].second) throw std::runtime_error("overlapping tensors");  // check 5
  return total;
}

}  // namespace st
