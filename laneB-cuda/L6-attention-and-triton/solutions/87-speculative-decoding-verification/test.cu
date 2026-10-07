#include <algorithm>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static std::vector<float> rand_dist(size_t V, unsigned seed, float peak_at, int peak_idx) {
  auto w = s2s::random_vec<float>(V, 0.0, 1.0, seed);
  if (peak_idx >= 0) w[size_t(peak_idx)] += peak_at;
  double s = 0;
  for (float x : w) s += x;
  for (float& x : w) x = float(x / s);
  return w;
}

static int ref_sample(const std::vector<double>& w, double u) {
  double tot = 0;
  for (double x : w) tot += x;
  double c = 0;
  for (size_t j = 0; j < w.size(); ++j) { c += w[j]; if (w[j] > 0 && c > u * tot) return int(j); }
  for (size_t j = w.size(); j-- > 0;) if (w[j] > 0) return int(j);
  return 0;
}

S2S_TEST(accept_reject_resample) {
  const int B = 6, k = 4, V = 1000;
  std::vector<int> draft(size_t(B) * k);
  std::vector<float> qp, pp, ua(size_t(B) * k), us(size_t(B));
  auto u_all = s2s::random_vec<float>(size_t(B) * (k + 1), 0.0, 0.999, 77);
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < k; ++i) {
      draft[size_t(b) * k + i] = (b * 37 + i * 101) % V;
      auto q = rand_dist(V, unsigned(1000 + b * 10 + i), b % 2 ? 50.f : 0.f, draft[size_t(b) * k + i]);
      qp.insert(qp.end(), q.begin(), q.end());
    }
    for (int i = 0; i <= k; ++i) {
      const int pk = i < k ? draft[size_t(b) * k + i] : -1;
      auto p = rand_dist(V, unsigned(2000 + b * 10 + i), b < 3 ? 30.f : 0.f, pk);   // b < 3: target agrees → accepts
      pp.insert(pp.end(), p.begin(), p.end());
    }
    for (int i = 0; i < k; ++i) ua[size_t(b) * k + i] = u_all[size_t(b) * (k + 1) + i];
    us[size_t(b)] = u_all[size_t(b) * (k + 1) + k];
  }
  // CPU reference of the same rule
  std::vector<int> want_tok(size_t(B) * (k + 1), -1), want_n(size_t(B));
  for (int b = 0; b < B; ++b) {
    int n = 0;
    for (; n < k; ++n) {
      const int x = draft[size_t(b) * k + n];
      const double p = pp[(size_t(b) * (k + 1) + n) * V + x], q = qp[(size_t(b) * k + n) * V + x];
      if (!(ua[size_t(b) * k + n] < std::min(1.0, q > 0 ? p / q : 1.0))) break;
      want_tok[size_t(b) * (k + 1) + n] = x;
    }
    std::vector<double> w(size_t(V));
    for (int j = 0; j < V; ++j) {
      const double p = pp[(size_t(b) * (k + 1) + n) * V + j];
      w[size_t(j)] = n < k ? std::max(0.0, p - double(qp[(size_t(b) * k + n) * V + j])) : p;
    }
    want_tok[size_t(b) * (k + 1) + n] = ref_sample(w, us[size_t(b)]);
    want_n[size_t(b)] = n + 1;
  }
  s2s::DeviceBuffer<int> dd(draft), tok(size_t(B) * (k + 1)), nout(size_t(B));
  s2s::DeviceBuffer<float> dq(qp), dp(pp), dua(ua), dus(us);
  std::vector<int> neg(size_t(B) * (k + 1), -1);
  tok.upload(neg);
  solve(dd.get(), dq.get(), dp.get(), dua.get(), dus.get(), tok.get(), nout.get(), B, k, V);
  CUDA_CHECK_LAUNCH();
  auto gt = tok.download(), gn = nout.download();
  for (int b = 0; b < B; ++b) {
    CHECK_EQ(gn[size_t(b)], want_n[size_t(b)]);
    for (int i = 0; i < want_n[size_t(b)]; ++i) CHECK_EQ(gt[size_t(b) * (k + 1) + i], want_tok[size_t(b) * (k + 1) + i]);
  }
}

S2S_TEST_MAIN()
