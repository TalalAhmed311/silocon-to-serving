from s2s.exercise import load_impl

m = load_impl(__file__, "sse")


def test_two_simple_events():
    raw = b'data: {"t": "Hel"}\n\ndata: {"t": "lo"}\n\n'
    assert m.parse_sse(raw) == [{"event": "message", "data": '{"t": "Hel"}'}, {"event": "message", "data": '{"t": "lo"}'}]


def test_done_ends_the_stream():
    raw = b"data: a\n\ndata: [DONE]\n\ndata: ignored\n\n"
    assert m.parse_sse(raw) == [{"event": "message", "data": "a"}]


def test_multiline_data_is_joined_with_newline():
    assert m.parse_sse(b"data: line one\ndata: line two\n\n") == [{"event": "message", "data": "line one\nline two"}]


def test_comments_are_ignored():
    raw = b": keep-alive\n\ndata: x\n\n: another\n\n"
    assert m.parse_sse(raw) == [{"event": "message", "data": "x"}]


def test_event_name_and_crlf():
    raw = b"event: usage\r\ndata: 42\r\n\r\ndata: y\r\n\r\n"
    assert m.parse_sse(raw) == [{"event": "usage", "data": "42"}, {"event": "message", "data": "y"}]


def test_only_one_leading_space_is_stripped():
    assert m.parse_sse(b"data:  two spaces\n\ndata:none\n\n") == [
        {"event": "message", "data": " two spaces"}, {"event": "message", "data": "none"}]


def test_unfinished_event_is_dropped():
    assert m.parse_sse(b"data: done\n\ndata: cut off mid") == [{"event": "message", "data": "done"}]


def test_empty_and_utf8():
    assert m.parse_sse(b"") == []
    assert m.parse_sse("data: café\n\n".encode()) == [{"event": "message", "data": "café"}]
