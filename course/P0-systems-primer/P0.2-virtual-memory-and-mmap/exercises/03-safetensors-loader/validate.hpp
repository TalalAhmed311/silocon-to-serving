// Exercise 3 starter. Implement validate(); throw std::runtime_error on any of the 6 problems in the README.
#pragma once
#include <stdexcept>

#include "../../examples/safetensors_view.hpp"

namespace st {

inline int64_t validate(const File& f) {
  (void)f;
  // TODO: checks 1-6, then return the total number of parameters.
  throw std::runtime_error("not implemented");
}

}  // namespace st
