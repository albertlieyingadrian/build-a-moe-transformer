"""Stage 13: compare. Dense and MoE at the same active parameters both learn; the MoE stores more weights."""

import math

from conftest import CORPUS, SMALL, load_json, run


def test_compare(corpus, tmp_path):
    run(
        ["compare", corpus, "--steps", "250", "--variants", "dense,moe-bias,dense-wide", *SMALL, "--out-dir", tmp_path],
        cwd=tmp_path,
    )
    r = load_json(tmp_path / "compare_s0.json")
    dense, moe, wide = r["dense"], r["moe-bias"], r["dense-wide"]
    assert abs(moe["active"] - dense["active"]) / dense["active"] < 0.03, "active parameters must match"
    assert moe["params"] > 1.8 * dense["params"]
    assert abs(wide["params"] - moe["params"]) / moe["params"] < 0.03, "dense-wide matches the MoE's total"
    guess = math.log(len(set(CORPUS)))
    for name, v in r.items():
        assert v["val"] < 0.5 * guess, f"{name} did not learn: {v['val']}"
    assert moe["val"] < dense["val"] + 0.15, "the MoE should be no worse than dense at equal active parameters"
    assert moe["max_load"] < 2.0
    assert (tmp_path / "compare_s0.png").exists() and (tmp_path / "moe-bias_s0.pt").exists()
