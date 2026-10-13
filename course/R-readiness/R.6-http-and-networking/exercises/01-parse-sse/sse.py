"""Exercise 1: parse a raw server-sent-events byte stream."""


def parse_sse(raw: bytes) -> list[dict]:
    """Return the events in `raw` as [{"event": str, "data": str}, ...].

    Rules (a subset of the SSE format that LLM servers use):
      - Lines end in \\n or \\r\\n. A blank line ends an event.
      - "data: x" adds x to the event's data. Strip ONE space after the colon.
        Several data lines in one event are joined with "\\n".
      - "event: name" sets the event name. The default name is "message".
      - A line starting with ":" is a comment (a keep-alive). Ignore it.
      - An event with no data lines is not delivered.
      - The data "[DONE]" ends the stream: do not return it or anything after it.
      - Bytes after the last blank line are an unfinished event: drop them.
    """
    # TODO: decode as UTF-8, split into lines, and build events line by line.
    raise NotImplementedError
