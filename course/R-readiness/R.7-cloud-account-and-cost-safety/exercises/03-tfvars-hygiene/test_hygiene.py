from s2s.exercise import load_impl

m = load_impl(__file__, "hygiene")

# AWS's documented example key id, split so that secret scanners do not trip on this file.
FAKE_KEY = "AKIA" + "IOSFODNN7EXAMPLE"

SAFE = '''# cp terraform.tfvars.example terraform.tfvars
owner              = "your-handle"
budget_email       = "you@example.com"
region             = "us-east-1"
instance_type      = "g6.xlarge"
monthly_budget_usd = 100
'''


def test_safe_file_has_no_findings():
    assert m.check_tfvars(SAFE) == []


def test_flags_access_key_id_with_line_number():
    out = m.check_tfvars(f'owner = "me"\naccess_key = "{FAKE_KEY}"\n')
    assert len(out) == 1 and out[0].startswith("line 2:")


def test_flags_password_and_token_literals():
    out = m.check_tfvars('db_password = "hunter2"\nhf_token = "abc"\n')
    assert [f.split(":")[0] for f in out] == ["line 1", "line 2"]


def test_flags_secret_access_key():
    assert m.check_tfvars('aws_secret_access_key = "x"\n')


def test_ignores_comments_and_empty_values():
    assert m.check_tfvars('# password = "example"\n// token = "x"\npassword = ""\n') == []


def test_does_not_flag_names_without_literals():
    assert m.check_tfvars("token_ttl_seconds = 3600\nsecret_name = var.name\n") == []


def test_gitignore_exact_and_wildcard():
    assert m.gitignore_covers(["terraform.tfvars"], "terraform.tfvars")
    assert m.gitignore_covers(["*.tfvars"], "terraform.tfvars")
    assert m.gitignore_covers(["*.tfvars"], "infra/aws/single-node/terraform.tfvars")


def test_gitignore_does_not_cover_the_example_file():
    assert not m.gitignore_covers(["terraform.tfvars"], "terraform.tfvars.example")
    assert not m.gitignore_covers(["*.tfvars"], "terraform.tfvars.example")


def test_gitignore_negation_last_match_wins():
    lines = ["*.tfvars", "!example.tfvars"]
    assert not m.gitignore_covers(lines, "example.tfvars")
    assert m.gitignore_covers(lines, "prod.tfvars")
    assert not m.gitignore_covers(["terraform.tfvars", "!terraform.tfvars"], "terraform.tfvars")


def test_gitignore_comments_blank_lines_and_directories():
    assert not m.gitignore_covers(["# terraform.tfvars", ""], "terraform.tfvars")
    assert m.gitignore_covers([".terraform/"], ".terraform/providers/x")
    assert m.gitignore_covers(["/terraform.tfvars"], "terraform.tfvars")
    assert not m.gitignore_covers(["/terraform.tfvars"], "sub/terraform.tfvars")
    assert not m.gitignore_covers([], "terraform.tfvars")
