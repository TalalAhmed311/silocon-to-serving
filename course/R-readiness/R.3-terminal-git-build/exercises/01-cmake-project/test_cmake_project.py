"""Configures, builds and runs ctest for the CMake project in a temporary build directory."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "solutions" / HERE.name if os.environ.get("S2S_SOLUTIONS") == "1" else HERE


def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake not installed")
def test_configure_build_and_ctest(tmp_path):
    build = tmp_path / "build"
    r = run("cmake", "-S", str(SRC), "-B", str(build))
    assert r.returncode == 0, r.stdout + r.stderr
    r = run("cmake", "--build", str(build), "-j")
    assert r.returncode == 0, "build failed (undefined reference means a target is not linked):\n" + r.stdout + r.stderr
    r = run("ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error")
    assert r.returncode == 0, "ctest found no passing test named `mathlib`:\n" + r.stdout + r.stderr
    assert "mathlib" in r.stdout


@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake not installed")
def test_app_runs(tmp_path):
    build = tmp_path / "build"
    assert run("cmake", "-S", str(SRC), "-B", str(build)).returncode == 0
    assert run("cmake", "--build", str(build), "-j").returncode == 0
    r = run(str(build / "app"))
    assert r.returncode == 0
    assert "fib(10)    = 55" in r.stdout
