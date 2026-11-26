"""Stage 03: train-dense. The dense baseline's loss falls well below guessing."""

import math

from conftest import CORPUS, SMALL, line_value, run


def test_train_dense(corpus, tmp_path):
    args = ["train-dense", corpus, "--steps", "200", *SMALL, "--sample-len", "40", "--prompt", "All:"]
    out = run([*args, "--out-dir", tmp_path], cwd=tmp_path)
    val = float(line_value(out, "final val loss:").split()[0])
    assert val < 0.5 * math.log(len(set(CORPUS))), f"val loss {val} is not far enough below guessing"
    assert (tmp_path / "dense.pt").exists()
    assert "sample (greedy):" in out
