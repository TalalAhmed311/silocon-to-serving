from s2s.exercise import load_impl

m = load_impl(__file__, "dockerlint")


def codes(text):
    return sorted({f.split()[0] for f in m.lint(text)})


GOOD = """FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .
USER 10001
CMD ["python", "-m", "app"]
"""


def test_good_file_has_no_findings():
    assert m.lint(GOOD) == []


def test_d1_no_tag_and_line_number():
    out = m.lint("FROM python\nUSER 1000\n")
    assert out and out[0].startswith("D1 line 1")
    assert codes("FROM ghcr.io/org/app\nUSER 1000\n") == ["D1"]


def test_registry_port_is_not_a_tag():
    assert codes("FROM localhost:5000/app\nUSER 1000\n") == ["D1"]


def test_d2_latest():
    assert codes("FROM python:latest\nUSER 1000\n") == ["D2"]


def test_digest_counts_as_pinned_even_without_tag():
    assert m.lint("FROM python@sha256:" + "a" * 64 + "\nUSER 1000\n") == []


def test_d3_copy_all_before_install():
    text = "FROM python:3.12-slim\nCOPY . .\nRUN pip install -r requirements.txt\nUSER 1000\n"
    out = m.lint(text)
    assert [f.split()[0] for f in out] == ["D3"]
    assert out[0].startswith("D3 line 2")


def test_d3_handles_line_continuation_and_apt():
    text = "FROM ubuntu:24.04\nCOPY . /src\nRUN apt-get update \\\n && apt-get install -y curl\nUSER 1000\n"
    assert codes(text) == ["D3"]


def test_copy_all_without_a_later_install_is_fine():
    assert m.lint("FROM python:3.12-slim\nRUN pip install flask\nCOPY . .\nUSER 1000\n") == []


def test_d4_missing_or_root_user():
    assert codes("FROM python:3.12-slim\nRUN true\n") == ["D4"]
    assert codes("FROM python:3.12-slim\nUSER root\n") == ["D4"]
    assert codes("FROM python:3.12-slim\nUSER 0\n") == ["D4"]


def test_multistage_only_last_stage_needs_user_and_stage_names_are_not_images():
    text = """FROM python:3.12-slim AS build
COPY req.txt .
RUN pip install -r req.txt
FROM build AS test
FROM python:3.12-slim
COPY --from=build /opt /opt
USER app
"""
    assert m.lint(text) == []


def test_comments_and_blank_lines_are_skipped():
    text = "# FROM python\n\nFROM python:3.12\n# COPY . .\nUSER 1\n"
    assert m.lint(text) == []


def test_several_findings_at_once():
    text = "FROM python:latest\nCOPY . .\nRUN uv sync\n"
    assert codes(text) == ["D2", "D3", "D4"]
