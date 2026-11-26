"""Shared test helpers. The tests run a program through its CLI, the way a CodeCrafters tester would.

By default they test the last reference solution in solutions/python/. To test your own code:

    PROGRAM_DIR=path/to/your/code pytest tests/test_stage_05.py      # or: make test STAGE=05 DIR=path/to/your/code
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CORPUS = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n"
    "First Citizen:\nYou are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n\n"
    "Second Citizen:\nWould you proceed especially against Caius Marcius?\n\nAll:\nAgainst him first.\n\n"
) * 40

# a model small enough to train in seconds on a laptop CPU
SMALL = ["--layers", "2", "--n-embd", "64", "--expert-hidden", "32", "--block-size", "32", "--batch-size", "16"]


def program_dir() -> Path:
    if os.environ.get("PROGRAM_DIR"):
        return Path(os.environ["PROGRAM_DIR"]).resolve()
    return sorted(p for p in (ROOT / "solutions" / "python").iterdir() if p.is_dir())[-1] / "code"


def run(args, cwd, timeout=900) -> str:
    """Run ./your_program.sh with args in cwd; fail with the program's output if it exits non-zero."""
    env = dict(os.environ, PYTHON=os.environ.get("PYTHON", sys.executable))
    proc = subprocess.run(
        [str(program_dir() / "your_program.sh"), *map(str, args)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, f"exit code {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc.stdout


def line_value(out: str, prefix: str) -> str:
    for line in out.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()[len(prefix) :].strip()
    raise AssertionError(f"no line starting with {prefix!r} in output:\n{out}")


def ints(s: str) -> list[int]:
    return [int(v) for v in s.strip("[]").split(",") if v.strip()]


def load_json(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text(CORPUS)
    return path


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """A small MoE trained for 300 steps (a few seconds), shared by stages 14 and 15."""
    d = tmp_path_factory.mktemp("trained")
    (d / "corpus.txt").write_text(CORPUS)
    run(["compare", d / "corpus.txt", "--steps", "300", "--variants", "moe-bias", *SMALL, "--out-dir", d], cwd=d)
    return d
