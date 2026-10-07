#include <s2s/check.hpp>
#include <ops.hpp>
#include "fixtures.hpp"

static std::string dir;

S2S_TEST(every_position_matches_numpy) {
  Fixtures fx(dir + "/ops.safetensors");
  auto q = fx.get<float>("attn_q"), k = fx.get<float>("attn_k"), v = fx.get<float>("attn_v"), want = fx.get<float>("attn_out");
  const int T = 10, H = 4, KVH = 2, HD = 16;
  std::vector<float> out(H * HD), scratch(T);
  for (int pos = 0; pos < T; ++pos) {
    ops::attention(q.data() + pos * H * HD, k.data(), v.data(), out.data(), pos, H, KVH, HD, scratch.data());
    CHECK_ALLCLOSE(out.data(), want.data() + pos * H * HD, size_t(H * HD), 1e-5, 1e-5);
  }
}

S2S_TEST(pos0_returns_v0) {  // one key: softmax weight 1, so out = V[0] for the matching KV head
  std::vector<float> q(2 * 4, 1.0f), k(4, 0.5f), v = {1, 2, 3, 4}, out(8), s(1);
  ops::attention(q.data(), k.data(), v.data(), out.data(), 0, 2, 1, 4, s.data());
  std::vector<float> want = {1, 2, 3, 4, 1, 2, 3, 4};
  CHECK_ALLCLOSE(out.data(), want.data(), 8, 0.0, 1e-6);
}

int main(int argc, char** argv) { dir = argc > 1 ? argv[1] : "build/p05-fixtures"; return s2s::run_all(); }
