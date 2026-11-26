"""Stage 14: route-lab. Which tokens go to which expert, against a shuffled control."""

from conftest import load_json, run


def test_route_lab(trained):
    out = run(
        ["route-lab", trained / "corpus.txt", "--ckpt", trained / "moe-bias_s0.pt", "--out-dir", trained], cwd=trained
    )
    rep = load_json(trained / "route_lab.json")
    assert len(rep) == 2
    for layer in rep:
        s = layer["specialisation"]
        assert 0 <= s["shuffled_control"] < 0.05
        assert s["normalised"] > s["shuffled_control"] + 0.1, "routing should depend on the token"
        assert len(layer["class_table"]) == 6 and len(layer["class_table"][0]) == 8
        for row in layer["class_table"]:
            assert abs(sum(row) - 1) < 1e-3 or sum(row) == 0
        assert abs(sum(layer["load"]["shares"]) - 1) < 1e-2
    assert (trained / "routing_heatmap.png").exists()
    assert "expert 7:" in out
