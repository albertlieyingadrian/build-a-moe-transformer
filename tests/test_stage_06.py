"""Stage 06: dispatch. The batched MoE layer gives the same answer as a token-by-token loop."""

from conftest import ints, line_value, run


def test_dispatch(tmp_path):
    out = run(["dispatch", "--tokens", "40", "--experts", "5", "--top-k", "2"], cwd=tmp_path)
    counts = ints(line_value(out, "tokens per expert:"))
    assert len(counts) == 5 and sum(counts) == 80
    assert line_value(out, "total slots:").split()[0] == "80"
    assert float(line_value(out, "max difference from the token-by-token reference:")) < 1e-5
    assert line_value(out, "one expert, top-1: the layer is just that expert:") == "True"
