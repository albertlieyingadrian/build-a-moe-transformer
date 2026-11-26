"""Stage 02: forward. A dense GPT that knows nothing scores about ln V, and nobody reads the future."""

import math

from conftest import CORPUS, line_value, run


def test_forward(corpus, tmp_path):
    out = run(["forward", corpus, "First Citizen"], cwd=tmp_path)
    V = len(set(CORPUS))
    assert line_value(out, "ids shape:") == "(1, 13)"
    assert line_value(out, "logits shape:") == f"(1, 13, {V})"
    assert abs(float(line_value(out, "loss at initialisation:")) - math.log(V)) < 0.2
    assert abs(float(line_value(out, "ln(vocab size):")) - math.log(V)) < 1e-3
    assert int(line_value(out, "parameters:")) > 0
    assert float(line_value(out, "change at earlier positions after editing the last token:")) == 0.0
