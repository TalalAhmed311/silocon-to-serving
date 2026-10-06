// config.hpp — read the handful of LlamaConfig fields the engine needs from an HF config.json.
#pragma once
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace engine {

struct Config {
  int dim = 0, hidden_dim = 0, n_layers = 0, n_heads = 0, n_kv_heads = 0, vocab_size = 0, max_seq_len = 0;
  float rope_theta = 10000.0f, norm_eps = 1e-5f;
  bool tie_embeddings = false;
  int head_dim() const { return dim / n_heads; }
  int kv_dim() const { return n_kv_heads * head_dim(); }
};

// Finds `"key": <value>` in flat JSON text. Enough for config.json; not a general JSON parser.
inline std::string json_raw(const std::string& s, const std::string& key) {
  const std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return "";
  p = s.find(':', p + pat.size());
  if (p == std::string::npos) return "";
  ++p;
  while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\t')) ++p;
  size_t e = p;
  while (e < s.size() && s[e] != ',' && s[e] != '}' && s[e] != '\n') ++e;
  std::string v = s.substr(p, e - p);
  while (!v.empty() && (v.back() == ' ' || v.back() == '\r')) v.pop_back();
  return v;
}

inline Config load_config(const std::string& path, int max_seq_override = 0) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string s = ss.str();
  auto num = [&](const char* k, double def) { auto v = json_raw(s, k); return v.empty() || v == "null" ? def : std::stod(v); };
  Config c;
  c.dim = int(num("hidden_size", 0));
  c.hidden_dim = int(num("intermediate_size", 0));
  c.n_layers = int(num("num_hidden_layers", 0));
  c.n_heads = int(num("num_attention_heads", 0));
  c.n_kv_heads = int(num("num_key_value_heads", c.n_heads));
  c.vocab_size = int(num("vocab_size", 0));
  c.max_seq_len = int(num("max_position_embeddings", 2048));
  c.rope_theta = float(num("rope_theta", 10000.0));
  c.norm_eps = float(num("rms_norm_eps", 1e-5));
  c.tie_embeddings = json_raw(s, "tie_word_embeddings") == "true";
  if (max_seq_override > 0 && max_seq_override < c.max_seq_len) c.max_seq_len = max_seq_override;
  if (!c.dim || !c.n_layers || !c.n_heads || !c.vocab_size || c.dim % c.n_heads || c.n_heads % c.n_kv_heads)
    throw std::runtime_error("config.json is missing fields or has inconsistent head counts");
  return c;
}

}  // namespace engine
