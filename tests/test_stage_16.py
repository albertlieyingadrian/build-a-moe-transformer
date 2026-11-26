"""Stage 16: collapse. Detect a collapsed router, and watch the two balancing methods pull it back."""

from conftest import SMALL, line_value, load_json, run


def test_collapse(corpus, tmp_path):
    out = run(["collapse", corpus, "--steps", "200", *SMALL, "--out-dir", tmp_path], cwd=tmp_path)
    assert line_value(out, "detector on an even layer:") == "ok"
    flags = line_value(out, "detector on a collapsed layer:")
    assert "entropy" in flags and "max load" in flags and "3 dead" in flags
    r = load_json(tmp_path / "collapse_s0.json")
    assert r["none-skew"]["worst_max_load"] > 3.0, "with a head start and no balancing, expert 0 keeps its lead"
    assert r["bias-skew"]["worst_max_load"] < 2.0, "the bias should pull the load back towards even"
    assert r["bias-skew"]["worst_max_load"] < r["aux-skew"]["worst_max_load"] + 0.3
    assert r["aux-skew"]["worst_max_load"] < r["none-skew"]["worst_max_load"]
    assert r["none-skew"]["first_flagged_step"] == 0
    assert (tmp_path / "collapse_s0.png").exists()
