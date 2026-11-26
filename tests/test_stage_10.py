"""Stage 10: shared-expert. Every token uses it; the layer is the shared part plus the routed part."""

import re

from conftest import line_value, run


def test_shared_expert(tmp_path):
    out = run(["shared-expert", "--d", "32", "--h", "16", "--experts", "6", "--top-k", "2"], cwd=tmp_path)
    routed, shared, router, total = (int(v) for v in re.findall(r"\d+", line_value(out, "parameters:")))
    per = 3 * 32 * 16
    assert routed == 6 * per and shared == per and router == 32 * 6 and total == routed + shared + router
    assert int(line_value(out, "active per token:").split("=")[-1]) == 2 * per + per + router
    assert line_value(out, "routed experts off: the layer is the shared expert:") == "True"
    assert line_value(out, "layer = shared part + routed part:") == "True"
