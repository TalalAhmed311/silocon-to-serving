"""P1.5 starter: your capacity calculator. Mirror the formulas in the lesson; keep assumptions as parameters."""

DTYPE_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}


def total_params(cfg: dict) -> int:
    raise NotImplementedError


def active_params(cfg: dict) -> int:
    raise NotImplementedError


def weights_bytes(cfg: dict, dtype: str, tp: int = 1) -> float:
    raise NotImplementedError


def kv_bytes_per_token(cfg: dict, kv_dtype: str, tp: int = 1) -> float:
    raise NotImplementedError


def max_sequences(cfg: dict, gpu: dict, weight_dtype: str, kv_dtype: str, context: int,
                  mem_util: float = 0.9, activation_gb: float = 1.0, tp: int = 1) -> int:
    raise NotImplementedError


def parse_vllm_kv_tokens(log_text: str) -> int | None:
    raise NotImplementedError
