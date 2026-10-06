"""Exercise 3 solution."""


def _good(r, ttft_slo, tpot_slo):
    return r["ttft"] <= ttft_slo and (r["tpot"] is None or r["tpot"] <= tpot_slo)


def goodput(requests, ttft_slo, tpot_slo, window_s):
    return sum(_good(r, ttft_slo, tpot_slo) for r in requests) / window_s


def slo_attainment(requests, ttft_slo, tpot_slo):
    return sum(_good(r, ttft_slo, tpot_slo) for r in requests) / len(requests) if requests else 0.0
