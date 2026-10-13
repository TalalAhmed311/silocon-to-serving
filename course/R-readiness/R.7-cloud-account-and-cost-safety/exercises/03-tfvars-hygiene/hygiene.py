"""Exercise 3 starter: keep secrets out of terraform.tfvars and keep that file out of git."""
from __future__ import annotations


def check_tfvars(text: str) -> list[str]:
    """Return one finding per suspicious line, formatted "line N: <reason>".

    Flag: an AWS access key id (AKIA or ASIA followed by 16 upper-case letters or digits), an
    assignment to a name containing password, secret or token whose value is a non-empty
    string literal, and any aws_secret_access_key assignment. Do not flag comments (#, //),
    empty values, or harmless values such as an email address or a region.
    """
    raise NotImplementedError


def gitignore_covers(lines: list[str], path: str) -> bool:
    """True if a .gitignore made of `lines` ignores `path` (a repo-relative file path).

    Support: blank lines and # comments, exact names, `*` wildcards inside one name,
    patterns without a slash matching in any directory, and `!` negation (the last matching
    line wins).
    """
    raise NotImplementedError
