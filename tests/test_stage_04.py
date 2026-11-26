"""Stage 04: expert. A SwiGLU expert has 3·d·h weights and a closed gate lets nothing through."""

from conftest import line_value, run


def test_expert(tmp_path):
    out = run(["expert", "--d", "48", "--h", "96"], cwd=tmp_path)
    assert line_value(out, "expert parameters:").split()[0] == str(3 * 48 * 96)
    assert int(line_value(out, "a GELU MLP (two matrices) with the same parameters has hidden width:")) == 144
    assert line_value(out, "output shape:") == "(5, 48)"
    assert line_value(out, "silu(-3), silu(0), silu(3):").split() == ["-0.142", "0.000", "2.858"]
    assert float(line_value(out, "output with the gate closed (gate weights 0):")) == 0.0
