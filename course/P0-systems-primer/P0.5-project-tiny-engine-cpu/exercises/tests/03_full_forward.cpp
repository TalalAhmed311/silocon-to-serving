#include <cmath>

#include <s2s/check.hpp>
#include "engine.hpp"
#include "fixtures.hpp"

static std::string dir;

S2S_TEST(teacher_forced_logits) {
  Fixtures fx(dir + "/forward.safetensors");
  auto ids = fx.get<int32_t>("tf_ids");
  auto want = fx.get<float>("tf_logits");
  engine::Engine eng(dir + "/tiny", 2);
  const int V = eng.config().vocab_size;
  for (size_t p = 0; p < ids.size(); ++p) {
    const float* lg = eng.forward(ids[p], int(p));
    CHECK_ALLCLOSE(lg, want.data() + p * size_t(V), size_t(V), 1e-4, 1e-4);
  }
}

S2S_TEST(greedy_tokens) {
  Fixtures fx(dir + "/forward.safetensors");
  auto prompt = fx.get<int32_t>("greedy_prompt"), want = fx.get<int32_t>("greedy_tokens");
  auto gaps = fx.get<float>("greedy_gaps");
  engine::Engine eng(dir + "/tiny", 2);
  std::vector<int> toks(prompt.begin(), prompt.end());
  for (int pos = 0; pos + 1 < int(want.size()); ++pos) {
    const float* lg = eng.forward(toks[size_t(pos)], pos);
    if (pos + 1 >= int(prompt.size())) {
      toks.push_back(ops::argmax(lg, eng.config().vocab_size));
      if (toks.back() != want[size_t(pos + 1)]) {
        // Only acceptable at a genuine near-tie in the reference (fp32 summation-order noise).
        CHECK(gaps[size_t(pos)] < 1e-4f);
        return;
      }
    }
  }
}

int main(int argc, char** argv) { dir = argc > 1 ? argv[1] : "build/p05-fixtures"; return s2s::run_all(); }
