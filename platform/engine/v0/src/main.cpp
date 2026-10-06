// main.cpp — CLI for #0 v0.
// Usage: s2s-engine --model DIR [--prompt-ids "1 2 3"] [--steps 64] [--temp 0] [--top-p 0.9] [--seed 0]
//                   [--threads N] [--max-seq N] [--dump-logits FILE] [--teacher-forced "ids..."]
// Prints the generated ids on one line, then timing to stderr: prefill tok/s, decode tok/s.
// --teacher-forced feeds the given ids one by one and writes every step's logits (for tests vs the NumPy reference).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "engine.hpp"

static std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> v;
  std::istringstream in(s);
  for (int x; in >> x;) v.push_back(x);
  return v;
}

int main(int argc, char** argv) {
  std::string model, prompt = "1", dump, teacher;
  int steps = 32, threads = 0, max_seq = 0;
  float temp = 0.0f, top_p = 0.9f;
  unsigned seed = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--model") model = v;
    else if (k == "--prompt-ids") prompt = v;
    else if (k == "--steps") steps = std::atoi(v.c_str());
    else if (k == "--temp") temp = float(std::atof(v.c_str()));
    else if (k == "--top-p") top_p = float(std::atof(v.c_str()));
    else if (k == "--seed") seed = unsigned(std::atoi(v.c_str()));
    else if (k == "--threads") threads = std::atoi(v.c_str());
    else if (k == "--max-seq") max_seq = std::atoi(v.c_str());
    else if (k == "--dump-logits") dump = v;
    else if (k == "--teacher-forced") teacher = v;
    else { std::fprintf(stderr, "unknown flag %s\n", k.c_str()); return 2; }
  }
  if (model.empty()) { std::fprintf(stderr, "usage: %s --model DIR [...]\n", argv[0]); return 2; }
  engine::Engine eng(model, unsigned(threads), max_seq);
  const int V = eng.config().vocab_size;

  if (!teacher.empty()) {
    auto ids = parse_ids(teacher);
    std::ofstream out(dump, std::ios::binary);  // raw float32 [len(ids), vocab]
    for (size_t pos = 0; pos < ids.size(); ++pos) {
      const float* lg = eng.forward(ids[pos], int(pos));
      out.write(reinterpret_cast<const char*>(lg), std::streamsize(sizeof(float) * V));
    }
    return 0;
  }

  auto toks = parse_ids(prompt);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now(), t_first = t0;
  const size_t n_prompt = toks.size();
  for (int pos = 0; pos < int(n_prompt) + steps - 1; ++pos) {
    const float* lg = eng.forward(toks[size_t(pos)], pos);
    if (pos == int(n_prompt) - 1) t_first = clk::now();  // first generated token's logits are ready: TTFT
    if (pos >= int(n_prompt) - 1) toks.push_back(ops::sample_top_p(lg, V, temp, top_p, uni(rng)));
  }
  auto t1 = clk::now();
  for (size_t i = 0; i < toks.size(); ++i) std::printf("%s%d", i ? " " : "", toks[i]);
  std::printf("\n");
  const double prefill_s = std::chrono::duration<double>(t_first - t0).count();
  const double decode_s = std::chrono::duration<double>(t1 - t_first).count();
  std::fprintf(stderr, "weights %.1f MiB, KV cache %.1f MiB | prefill %zu tok in %.3f s (%.1f tok/s) | decode %d tok in %.3f s (%.1f tok/s)\n",
               eng.weight_bytes() / 1048576.0, eng.kv_cache_bytes() / 1048576.0, n_prompt, prefill_s,
               n_prompt / prefill_s, steps - 1, decode_s, (steps - 1) / decode_s);
  return 0;
}
