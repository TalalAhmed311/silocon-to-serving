#include <s2s/check.hpp>
#include <ops.hpp>
#include "fixtures.hpp"

static std::string dir;

S2S_TEST(rmsnorm_matches_numpy) {
  Fixtures fx(dir + "/ops.safetensors");
  auto x = fx.get<float>("rmsnorm_x"), w = fx.get<float>("rmsnorm_w"), want = fx.get<float>("rmsnorm_out");
  std::vector<float> got(x.size());
  for (int r = 0; r < 4; ++r) ops::rmsnorm(got.data() + r * 64, x.data() + r * 64, w.data(), 64, 1e-5f);
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-5, 1e-5);
}

S2S_TEST(rmsnorm_in_place) {  // the engine calls rmsnorm(x, x, ...) for the final norm
  Fixtures fx(dir + "/ops.safetensors");
  auto x = fx.get<float>("rmsnorm_x"), w = fx.get<float>("rmsnorm_w"), want = fx.get<float>("rmsnorm_out");
  ops::rmsnorm(x.data(), x.data(), w.data(), 64, 1e-5f);
  CHECK_ALLCLOSE(x.data(), want.data(), 64, 1e-5, 1e-5);
}

S2S_TEST(rope_matches_numpy) {
  Fixtures fx(dir + "/ops.safetensors");
  auto pos = fx.get<int32_t>("rope_pos");
  auto q = fx.get<float>("rope_in"), want = fx.get<float>("rope_out");
  for (size_t i = 0; i < pos.size(); ++i) ops::rope(q.data() + i * 64, 4, 16, pos[i], 10000.0f);
  CHECK_ALLCLOSE(q.data(), want.data(), want.size(), 1e-5, 1e-5);
}

S2S_TEST(rope_pos0_is_identity) {
  std::vector<float> v(32), orig(32);
  for (int i = 0; i < 32; ++i) v[size_t(i)] = orig[size_t(i)] = float(i) - 10.0f;
  ops::rope(v.data(), 2, 16, 0, 10000.0f);
  CHECK_ALLCLOSE(v.data(), orig.data(), 32, 0.0, 1e-6);
}

int main(int argc, char** argv) { dir = argc > 1 ? argv[1] : "build/p05-fixtures"; return s2s::run_all(); }
