// Reads named float32/int32 tensors from the fixture files written by make_fixtures.py.
#pragma once
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "safetensors_view.hpp"

struct Fixtures {
  st::File f;
  explicit Fixtures(const std::string& path) : f(path) {}
  template <class T>
  std::vector<T> get(const std::string& name) const {
    auto it = f.tensors().find(name);
    if (it == f.tensors().end()) throw std::runtime_error("fixture missing: " + name + " (re-run make_fixtures.py)");
    std::vector<T> v((it->second.end - it->second.begin) / sizeof(T));
    std::memcpy(v.data(), f.bytes(it->second), v.size() * sizeof(T));
    return v;
  }
};
