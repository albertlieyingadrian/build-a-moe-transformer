"""Stage 11: fine-grained. Splitting experts keeps the weights and multiplies the combinations."""

import math

from conftest import run


def test_fine_grained(tmp_path):
    out = run(
        ["fine-grained", "--d", "32", "--h", "64", "--experts", "8", "--top-k", "2", "--split", "4"], cwd=tmp_path
    )
    rows = [line.split() for line in out.splitlines() if line.strip() and line.split()[0].isdigit()]
    assert len(rows) == 2
    (m1, e1, k1, s1, h1, w1, r1, a1, c1), (m4, e4, k4, s4, h4, w4, r4, a4, c4) = ([int(v) for v in r] for r in rows)
    assert (e1, k1, h1) == (8, 2, 64) and (e4, k4, s4, h4) == (32, 8, 4, 16)
    assert w1 == w4 == 9 * 3 * 32 * 64, "the expert weights are the same, only cut finer"
    assert r4 == 4 * r1
    assert a4 - a1 == r4 - r1, "active weights differ only by the bigger router"
    assert c1 == math.comb(8, 2) and c4 == math.comb(32, 8)
