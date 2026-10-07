// Exercise 4 starter (hard): fused cross-entropy. loss[r] = logsumexp(logits[r, :]) − logits[r, target[r]].
// One read of the logits row (online max + sum), the target logit gathered in the same pass. rows × vocab fp32.
#pragma once
#include <cuda_runtime.h>

inline void cross_entropy(const float* logits, const int* target, float* loss, int rows, int vocab) {
  (void)logits; (void)target; (void)loss; (void)rows; (void)vocab;   // TODO
}
