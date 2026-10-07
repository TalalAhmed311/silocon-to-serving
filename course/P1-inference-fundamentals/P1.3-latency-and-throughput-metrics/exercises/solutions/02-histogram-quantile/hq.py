"""Exercise 2 solution: the same interpolation rule as Prometheus' histogram_quantile()."""
import bisect
import math


def observe_into_buckets(values, les):
    les = sorted(les) + [math.inf]
    counts = [0] * len(les)
    for v in values:
        counts[bisect.bisect_left(les, v)] += 1     # first le >= v  (buckets are "less than or equal")
    cum, out = 0, []
    for le, c in zip(les, counts):
        cum += c
        out.append((le, cum))
    return out


def histogram_quantile(q, buckets):
    total = buckets[-1][1]
    if total == 0:
        return math.nan
    rank = q * total
    prev_le, prev_c = 0.0, 0
    for le, c in buckets:
        if c >= rank:
            if math.isinf(le):
                return prev_le                       # Prometheus returns the highest finite bound
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return prev_le
