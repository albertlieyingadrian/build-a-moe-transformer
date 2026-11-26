"""Stage 17 (extension): mtp. A second head that predicts the token after next, for free at inference."""

import math

from conftest import CORPUS, SMALL, line_value, run


def test_mtp(corpus, tmp_path):
    out = run(["mtp", corpus, "--steps", "200", *SMALL, "--out-dir", tmp_path], cwd=tmp_path)
    guess = math.log(len(set(CORPUS)))
    assert float(line_value(out, "MTP val loss on token t+2:").split()[0]) < 0.5 * guess
    assert line_value(out, "active parameters unchanged by MTP:") == "True"
    assert int(line_value(out, "MTP module parameters (training only):")) > 0
    assert 0 <= float(line_value(out, "MTP agrees with the main model's next-step guess:").split()[0]) <= 1
    assert abs(float(line_value(out, "main val loss change with MTP:"))) < 0.3
    assert (tmp_path / "mtp_s0.json").exists()
