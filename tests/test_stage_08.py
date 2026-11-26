"""Stage 08: aux-loss. 1.0 when routing is even, E / k when it collapses, and its gradient spreads the load."""

from conftest import line_value, run


def test_aux_loss(tmp_path):
    out = run(["aux-loss", "--experts", "8", "--top-k", "2"], cwd=tmp_path)
    assert abs(float(line_value(out, "aux loss, even routing:")) - 1.0) < 1e-3
    assert abs(float(line_value(out, "aux loss, collapsed onto 2 experts:").split()[0]) - 4.0) < 1e-3
    assert line_value(out, "router gets a gradient:") == "True"
    a, b = (float(v) for v in line_value(out, "after 100 steps on the aux loss alone:").split("->"))
    assert b < a and b < 1.2
    s0, s1 = (float(v) for v in line_value(out, "expert 0's share of slots:").split()[:3:2])
    assert s1 < s0 and s1 < 0.25


def test_aux_loss_four_experts_top1(tmp_path):
    out = run(["aux-loss", "--experts", "4", "--top-k", "1"], cwd=tmp_path)
    assert abs(float(line_value(out, "aux loss, collapsed onto 1 experts:").split()[0]) - 4.0) < 1e-3
