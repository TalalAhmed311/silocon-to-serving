"""Exercise 1 (solution): parse a raw server-sent-events byte stream."""


def parse_sse(raw: bytes) -> list[dict]:
    events: list[dict] = []
    name = "message"
    data: list[str] = []
    for line in raw.decode("utf-8").replace("\r\n", "\n").split("\n")[:-1]:
        if line == "":                                   # blank line: dispatch
            if data:
                payload = "\n".join(data)
                if payload == "[DONE]":
                    return events
                events.append({"event": name, "data": payload})
            name, data = "message", []
        elif line.startswith(":"):
            continue
        else:
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "event":
                name = value
    return events
