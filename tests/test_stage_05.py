"""Stage 05: router. Softmax over experts, top-k distinct choices, weights that sum to one."""

import re

from conftest import ints, line_value, run


def test_router(tmp_path):
    out = run(["router", "--tokens", "10", "--experts", "6", "--top-k", "3"], cwd=tmp_path)
    rows = re.findall(r"token \d+: experts (\[.*?\]) weights (\[.*?\])", out)
    assert len(rows) == 10
    for experts, weights in rows:
        e = ints(experts)
        w = [float(v) for v in weights.strip("[]").split(",")]
        assert len(e) == 3 and len(set(e)) == 3 and all(0 <= i < 6 for i in e)
        assert abs(sum(w) - 1) < 0.02 and w == sorted(w, reverse=True)
    for line in (
        "probabilities sum to 1:",
        "top-k weights sum to 1:",
        "experts per token distinct:",
        "chosen experts are the most probable:",
    ):
        assert line_value(out, line) == "True", line
    slots = ints(line_value(out, "slots per expert:"))
    assert len(slots) == 6 and sum(slots) == 30


def test_top1_keeps_the_raw_probability(tmp_path):
    out = run(["router", "--tokens", "5", "--experts", "4", "--top-k", "1"], cwd=tmp_path)
    weights = [float(w) for w in re.findall(r"weights \[(.*?)\]", out)]
    assert all(0 < w < 1 for w in weights), "with top-1 the weight is the router's probability, not 1.0"
