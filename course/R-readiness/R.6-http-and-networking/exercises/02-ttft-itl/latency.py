"""Exercise 2: time-to-first-token and inter-token latency from timestamps."""


def stream_metrics(request_sent: float, arrivals: list[float]) -> dict:
    """Metrics for one streamed answer. All times are in seconds.

    request_sent: when the client sent the request.
    arrivals:     when each token chunk arrived, in order.

    Return {"ttft": ..., "itl_mean": ..., "itl_p90": ..., "e2e": ...} where
      ttft     = first arrival - request_sent
      itl_*    = statistics of the gaps between consecutive arrivals;
                 itl_p90 uses numpy's default (linear) percentile.
                 With a single chunk there are no gaps: use None for both.
      e2e      = last arrival - request_sent
    Raise ValueError if `arrivals` is empty.
    """
    # TODO
    raise NotImplementedError
