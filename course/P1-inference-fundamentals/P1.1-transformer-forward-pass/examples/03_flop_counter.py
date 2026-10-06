"""03_flop_counter.py — parameters, FLOPs/token and bytes/token from a config (no weights needed).

Run:      uv run python course/P1-inference-fundamentals/P1.1-transformer-forward-pass/examples/03_flop_counter.py --preset llama3-8b
          uv run python .../03_flop_counter.py --config path/to/config.json
Expected: parameter breakdown, then FLOPs/token at several context lengths with the attention share.
Hardware: T0. Presets are the author's recollection of the published configs: UNVERIFIED until you check
          them against the real config.json (exercise 1 does that).
"""
import argparse
import json

PRESETS = {  # UNVERIFIED — check each against the model's config.json on the Hub.
    "llama3-8b": dict(hidden_size=4096, intermediate_size=14336, num_hidden_layers=32, num_attention_heads=32,
                      num_key_value_heads=8, vocab_size=128256, tie_word_embeddings=False),
    "llama3-70b": dict(hidden_size=8192, intermediate_size=28672, num_hidden_layers=80, num_attention_heads=64,
                       num_key_value_heads=8, vocab_size=128256, tie_word_embeddings=False),
}


def params(c: dict) -> dict:
    d, L, H, Hkv, f, V = (c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"],
                          c.get("num_key_value_heads", c["num_attention_heads"]), c["intermediate_size"], c["vocab_size"])
    h = d // H
    per = {"q_proj": d * H * h, "k_proj": d * Hkv * h, "v_proj": d * Hkv * h, "o_proj": H * h * d,
           "mlp (gate+up+down)": 3 * d * f, "norms": 2 * d}
    out = {k: v * L for k, v in per.items()}
    out["embed_tokens"] = V * d
    out["lm_head"] = 0 if c.get("tie_word_embeddings") else V * d
    out["final norm"] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS)
    ap.add_argument("--config")
    ap.add_argument("--bytes-per-param", type=float, default=2.0)
    a = ap.parse_args()
    c = json.load(open(a.config)) if a.config else PRESETS[a.preset or "llama3-8b"]
    p = params(c)
    total = sum(p.values())
    print("| tensor group | params | % |\n|---|---|---|")
    for k, v in p.items():
        print(f"| {k} | {v / 1e6:,.1f} M | {100 * v / total:.1f} |")
    print(f"| **total** | **{total / 1e9:.3f} B** | |")
    matrices = total - p["embed_tokens"] - p["norms"] - p["final norm"]  # embedding is a lookup, not a matvec
    d, L = c["hidden_size"], c["num_hidden_layers"]
    print(f"\nweights read per token at {a.bytes_per_param} B/param: {total * a.bytes_per_param / 1e9:.1f} GB")
    print("\n| context t | weight FLOPs | attention FLOPs | attention share |\n|---|---|---|---|")
    for t in (1_000, 8_000, 32_000, 128_000):
        wf, af = 2 * matrices, 4 * d * L * t
        print(f"| {t:,} | {wf / 1e9:.1f} G | {af / 1e9:.1f} G | {100 * af / (wf + af):.0f}% |")


if __name__ == "__main__":
    main()
