from __future__ import annotations


def tp_comm_seconds_per_step(model, tp, tokens, link_gbs, alpha_us=0.0, act_bytes=2.0):
    if tp <= 1:
        return 0.0
    size = tokens * model.hidden * act_bytes
    one = 2 * (tp - 1) * alpha_us * 1e-6 + 2 * (tp - 1) / tp * size / (link_gbs * 1e9)
    return 2 * model.layers * one
