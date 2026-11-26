"""Stage 12: mla. A small cached latent, a causal mask, and RoPE that only sees distance."""

import re

from conftest import line_value, run


def test_mla(tmp_path):
    out = run(["mla", "--d", "96", "--heads", "4", "--kv-rank", "24", "--rope-dim", "12"], cwd=tmp_path)
    mha, mla = (int(v) for v in re.findall(r"(?:MHA|MLA) (\d+)", line_value(out, "cache floats per token per layer:")))
    assert mha == 2 * 96 and mla == 24 + 12
    assert float(line_value(out, "MLA weight on future positions:")) == 0.0
    assert float(line_value(out, "change at earlier positions after editing the last one:")) == 0.0
    assert line_value(out, "RoPE score depends only on the distance:") == "True"
