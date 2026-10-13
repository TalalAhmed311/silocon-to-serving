from fastapi.testclient import TestClient

from s2s.exercise import load_impl

m = load_impl(__file__, "app")
client = TestClient(m.app)


def test_ok():
    r = client.post("/v1/echo", json={"text": "hi", "repeat": 2})
    assert r.status_code == 200
    assert r.json() == {"text": "hi hi"}


def test_default_repeat_is_one():
    r = client.post("/v1/echo", json={"text": "hello"})
    assert r.status_code == 200 and r.json() == {"text": "hello"}


def test_invalid_json_is_400():
    r = client.post("/v1/echo", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json() == {"error": "invalid JSON"}


def test_wrong_type_is_422():
    r = client.post("/v1/echo", json={"text": 123})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid body"
    assert isinstance(r.json()["detail"], list) and r.json()["detail"]


def test_missing_field_is_422():
    assert client.post("/v1/echo", json={"repeat": 2}).status_code == 422


def test_repeat_out_of_range_is_422():
    assert client.post("/v1/echo", json={"text": "x", "repeat": 0}).status_code == 422
    assert client.post("/v1/echo", json={"text": "x", "repeat": 6}).status_code == 422


def test_blank_text_is_400():
    r = client.post("/v1/echo", json={"text": "   "})
    assert r.status_code == 400
    assert r.json() == {"error": "text must not be empty"}


def test_get_is_405():
    assert client.get("/v1/echo").status_code == 405


def test_unknown_path_is_404():
    assert client.post("/v1/nope", json={"text": "x"}).status_code == 404
