"""Stage 09: bias-balance. A bias that only steers the choice evens the load without touching the weights."""

from conftest import line_value, run


def test_bias_balance(tmp_path):
    out = run(["bias-balance", "--steps", "200"], cwd=tmp_path)
    before, after = (float(v) for v in line_value(out, "max load:").split("->"))
    assert before > 1.6 and after < 1.4 and after < before
    assert line_value(out, "gate weights ignore the bias:") == "True"
    assert line_value(out, "bias is trained by gradients:") == "False"


def test_zero_rate_changes_nothing(tmp_path):
    out = run(["bias-balance", "--steps", "100", "--rate", "0"], cwd=tmp_path)
    before, after = (float(v) for v in line_value(out, "max load:").split("->"))
    assert after > 1.6, "with rate 0 the bias never moves, so the load stays lopsided"
