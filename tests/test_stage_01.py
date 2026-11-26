"""Stage 01: tokenize. Text becomes ids and back, and batches are windows with targets shifted by one."""

from conftest import CORPUS, ints, line_value, run


def test_tokenize(corpus, tmp_path):
    out = run(["tokenize", corpus, "Speak, all?"], cwd=tmp_path)
    assert int(line_value(out, "corpus characters:")) == len(CORPUS)
    assert int(line_value(out, "vocab size:")) == len(set(CORPUS))
    chars = sorted(set(CORPUS))
    assert ints(line_value(out, "ids:")) == [chars.index(c) for c in "Speak, all?"]
    assert line_value(out, "decoded:") == "Speak, all?"
    assert line_value(out, "round trip ok:") == "True"
    tr, va = (int(v) for v in line_value(out, "train / val characters:").split("/"))
    assert tr + va == len(CORPUS) and abs(va / len(CORPUS) - 0.1) < 0.01
    assert line_value(out, "batch shape:") == "(4, 8)"
    assert line_value(out, "targets are the inputs shifted by one:") == "True"
