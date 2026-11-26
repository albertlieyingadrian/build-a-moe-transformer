"""Stage 07: capacity. Over-full experts drop tokens; dropped tokens get zero from the MoE layer."""

import math

from conftest import ints, line_value, run


def test_capacity_drops(tmp_path):
    out = run(
        ["capacity", "--tokens", "64", "--experts", "8", "--top-k", "1", "--capacity-factor", "1.0", "--skew", "1.5"],
        cwd=tmp_path,
    )
    cap = int(line_value(out, "capacity per expert:"))
    assert cap == math.ceil(1.0 * 64 * 1 / 8)
    wanted, kept = ints(line_value(out, "wanted:")), ints(line_value(out, "kept:"))
    assert sum(wanted) == 64
    assert all(k == min(w, cap) for w, k in zip(wanted, kept, strict=True))
    dropped = int(line_value(out, "dropped slots:").split()[0])
    assert dropped == sum(wanted) - sum(kept) > 0
    assert int(line_value(out, "tokens with every slot dropped:")) > 0
    assert float(line_value(out, "their MoE output (the residual carries them on unchanged):")) == 0.0


def test_no_limit_no_drops(tmp_path):
    out = run(["capacity", "--tokens", "64", "--capacity-factor", "0", "--skew", "1.5"], cwd=tmp_path)
    assert int(line_value(out, "dropped slots:").split()[0]) == 0


def test_bigger_factor_fewer_drops(tmp_path):
    drops = []
    for cf in ("1.0", "1.5", "2.5"):
        out = run(["capacity", "--tokens", "64", "--capacity-factor", cf, "--skew", "1.5"], cwd=tmp_path)
        drops.append(int(line_value(out, "dropped slots:").split()[0]))
    assert drops[0] >= drops[1] >= drops[2]
