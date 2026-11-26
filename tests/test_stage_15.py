"""Stage 15: ablate. Switching off an expert hurts the tokens it served more than the rest."""

from conftest import line_value, load_json, run


def test_ablate(trained):
    out = run(
        ["ablate", trained / "corpus.txt", "--ckpt", trained / "moe-bias_s0.pt", "--out-dir", trained], cwd=trained
    )
    rows = load_json(trained / "ablate.json")
    assert len(rows) == 2 * (8 + 1)
    hits, total = (
        int(v) for v in line_value(out, "ablations that hurt an expert's own tokens more than the rest:").split(" of ")
    )
    assert total == 16 and hits >= 12
    routed = [r for r in rows if r["expert"] != "shared"]
    assert sum(r["delta"] for r in routed) / len(routed) > 0, "removing an expert should cost something on average"
