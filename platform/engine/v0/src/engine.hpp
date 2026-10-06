// engine.hpp — #0 v0: a Llama-architecture decoder on the CPU, weights mmap'd zero-copy from safetensors.
// One call to forward(token, pos) = one decode step. The prompt is fed one token at a time (simple, and the
// right shape for teaching: prefill as a batched GEMM comes in P1.2 and P6).
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include <ops.hpp>  // angle brackets: -I order decides, so S2S_OPS_DIR can override src/ops.hpp
#include "safetensors_view.hpp"

namespace engine {

struct LayerWeights {
  const float *attn_norm, *wq, *wk, *wv, *wo, *mlp_norm, *w_gate, *w_up, *w_down;
};

class Engine {
 public:
  Engine(const std::string& model_dir, unsigned threads = 0, int max_seq_override = 0)
      : cfg_(load_config(model_dir + "/config.json", max_seq_override)),
        file_(model_dir + "/model.safetensors"),
        pool_(threads ? threads : std::max(1u, std::thread::hardware_concurrency())) {
    const int d = cfg_.dim, kv = cfg_.kv_dim(), hd = cfg_.head_dim();
    embed_ = tensor("model.embed_tokens.weight", {cfg_.vocab_size, d});
    for (int l = 0; l < cfg_.n_layers; ++l) {
      const std::string p = "model.layers." + std::to_string(l) + ".";
      layers_.push_back({tensor(p + "input_layernorm.weight", {d}),
                         tensor(p + "self_attn.q_proj.weight", {cfg_.n_heads * hd, d}),
                         tensor(p + "self_attn.k_proj.weight", {kv, d}),
                         tensor(p + "self_attn.v_proj.weight", {kv, d}),
                         tensor(p + "self_attn.o_proj.weight", {d, cfg_.n_heads * hd}),
                         tensor(p + "post_attention_layernorm.weight", {d}),
                         tensor(p + "mlp.gate_proj.weight", {cfg_.hidden_dim, d}),
                         tensor(p + "mlp.up_proj.weight", {cfg_.hidden_dim, d}),
                         tensor(p + "mlp.down_proj.weight", {d, cfg_.hidden_dim})});
    }
    final_norm_ = tensor("model.norm.weight", {d});
    lm_head_ = cfg_.tie_embeddings ? embed_ : tensor("lm_head.weight", {cfg_.vocab_size, d});
    // Activations and the KV cache are the only memory we allocate; weights stay in the page cache.
    x_.resize(size_t(d)); xb_.resize(size_t(d)); q_.resize(size_t(d)); att_out_.resize(size_t(d));
    hb_.resize(size_t(cfg_.hidden_dim)); hb2_.resize(size_t(cfg_.hidden_dim));
    logits_.resize(size_t(cfg_.vocab_size)); scratch_.resize(size_t(cfg_.max_seq_len));
    k_cache_.assign(size_t(cfg_.n_layers) * cfg_.max_seq_len * kv, 0.0f);
    v_cache_.assign(k_cache_.size(), 0.0f);
  }

  const Config& config() const { return cfg_; }
  size_t kv_cache_bytes() const { return 2 * k_cache_.size() * sizeof(float); }
  size_t weight_bytes() const { return weight_bytes_; }

  // One decode step. Returns logits [vocab]; the pointer is valid until the next call.
  const float* forward(int token, int pos) {
    if (pos >= cfg_.max_seq_len) throw std::runtime_error("pos >= max_seq_len (raise --max-seq)");
    const int d = cfg_.dim, hd = cfg_.head_dim(), kv = cfg_.kv_dim();
    std::copy(embed_ + size_t(token) * d, embed_ + size_t(token + 1) * d, x_.begin());
    for (int l = 0; l < cfg_.n_layers; ++l) {
      const LayerWeights& w = layers_[size_t(l)];
      float* kc = k_cache_.data() + size_t(l) * cfg_.max_seq_len * kv;
      float* vc = v_cache_.data() + size_t(l) * cfg_.max_seq_len * kv;
      // --- attention block ---
      ops::rmsnorm(xb_.data(), x_.data(), w.attn_norm, d, cfg_.norm_eps);
      ops::matvec(w.wq, xb_.data(), q_.data(), cfg_.n_heads * hd, d, pool_);
      ops::matvec(w.wk, xb_.data(), kc + size_t(pos) * kv, kv, d, pool_);  // write K/V straight into the cache
      ops::matvec(w.wv, xb_.data(), vc + size_t(pos) * kv, kv, d, pool_);
      ops::rope(q_.data(), cfg_.n_heads, hd, pos, cfg_.rope_theta);
      ops::rope(kc + size_t(pos) * kv, cfg_.n_kv_heads, hd, pos, cfg_.rope_theta);
      ops::attention(q_.data(), kc, vc, att_out_.data(), pos, cfg_.n_heads, cfg_.n_kv_heads, hd, scratch_.data());
      ops::matvec(w.wo, att_out_.data(), xb_.data(), d, cfg_.n_heads * hd, pool_);
      for (int i = 0; i < d; ++i) x_[size_t(i)] += xb_[size_t(i)];  // residual
      // --- SwiGLU MLP block ---
      ops::rmsnorm(xb_.data(), x_.data(), w.mlp_norm, d, cfg_.norm_eps);
      ops::matvec(w.w_gate, xb_.data(), hb_.data(), cfg_.hidden_dim, d, pool_);
      ops::matvec(w.w_up, xb_.data(), hb2_.data(), cfg_.hidden_dim, d, pool_);
      for (int i = 0; i < cfg_.hidden_dim; ++i) hb_[size_t(i)] = ops::silu(hb_[size_t(i)]) * hb2_[size_t(i)];
      ops::matvec(w.w_down, hb_.data(), xb_.data(), d, cfg_.hidden_dim, pool_);
      for (int i = 0; i < d; ++i) x_[size_t(i)] += xb_[size_t(i)];
    }
    ops::rmsnorm(x_.data(), x_.data(), final_norm_, d, cfg_.norm_eps);
    ops::matvec(lm_head_, x_.data(), logits_.data(), cfg_.vocab_size, d, pool_);
    return logits_.data();
  }

 private:
  const float* tensor(const std::string& name, std::vector<int64_t> shape) {
    auto it = file_.tensors().find(name);
    if (it == file_.tensors().end()) throw std::runtime_error("missing tensor " + name);
    const auto& t = it->second;
    if (t.dtype != "F32") throw std::runtime_error(name + ": dtype " + t.dtype + " (v0 needs F32; use tools/convert_hf_model.py)");
    if (t.shape != shape) throw std::runtime_error(name + ": unexpected shape");
    const uint8_t* p = file_.bytes(t);
    if (reinterpret_cast<uintptr_t>(p) % alignof(float)) throw std::runtime_error(name + ": misaligned");
    weight_bytes_ += t.end - t.begin;
    return reinterpret_cast<const float*>(p);  // zero-copy: points into the mmap'd file (P0.2)
  }

  Config cfg_;
  st::File file_;
  s2s::ThreadPool pool_;
  size_t weight_bytes_ = 0;
  const float *embed_ = nullptr, *final_norm_ = nullptr, *lm_head_ = nullptr;
  std::vector<LayerWeights> layers_;
  std::vector<float> x_, xb_, q_, att_out_, hb_, hb2_, logits_, scratch_, k_cache_, v_cache_;
};

}  // namespace engine
