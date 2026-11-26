"""Tables and figures for the README from runs/seed*/ (written by ./run_all.sh). Prints every number it plots.

python scripts/summarize.py
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = sorted((ROOT / "runs").glob("seed*"))
ASSETS = ROOT / "assets"
VARIANTS = ["dense", "dense-wide", "moe-none", "moe-aux", "moe-bias", "moe-bias-fine"]


def load(name):
    return [
        json.loads((d / name.format(s=d.name[4:])).read_text())
        for d in RUNS
        if (d / name.format(s=d.name[4:])).exists()
    ]


def ms(xs):
    return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"


def main():
    ASSETS.mkdir(exist_ok=True)
    comp = load("compare_s{s}.json")
    print(f"seeds: {len(comp)}")
    print(
        "\n| variant | total params | active params | val loss, seeds "
        + " / ".join(d.name[4:] for d in RUNS)
        + " | mean ± sd | max load (mean of layers) |"
    )
    print("|---|---|---|---|---|---|")
    for v in VARIANTS:
        vals = [c[v]["val"] for c in comp]
        loads = [c[v]["max_load"] for c in comp if c[v]["max_load"] is not None]
        ml = " / ".join(f"{x:.2f}" for x in loads) if loads else "-"
        print(
            f"| `{v}` | {comp[0][v]['params']:,} | {comp[0][v]['active']:,} | "
            + " / ".join(f"{x:.3f}" for x in vals)
            + f" | {ms(vals)} | {ml} |"
        )
    print("\npaired difference from dense (same seed, same validation windows):")
    for v in VARIANTS[1:]:
        d = [c[v]["val"] - c["dense"]["val"] for c in comp]
        print(f"  {v:14s} " + " ".join(f"{x:+.3f}" for x in d) + f"   mean {statistics.mean(d):+.3f}")
    print("\nseconds per run (8-vCPU container, 4 threads):")
    for v in VARIANTS:
        print(f"  {v:14s} " + " ".join(f"{c[v]['seconds']:.0f}" for c in comp))

    steps = [r["step"] for r in comp[0]["dense"]["history"]]
    print(
        "\npaired gap to dense over training (mean over seeds; worst seed in brackets; negative = better than dense):"
    )
    print("step  " + "  ".join(f"{v:>20s}" for v in VARIANTS[1:]))
    gaps = {}
    for v in VARIANTS[1:]:
        gaps[v] = [
            [c[v]["history"][i]["val"] - c["dense"]["history"][i]["val"] for c in comp] for i in range(len(steps))
        ]
    for i, st in enumerate(steps):
        print(
            f"{st:4d}  "
            + "  ".join(f"{statistics.mean(gaps[v][i]):+.3f} [{max(gaps[v][i]):+.3f}]".rjust(20) for v in VARIANTS[1:])
        )

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
    for i, v in enumerate(VARIANTS):
        vals = [c[v]["val"] for c in comp]
        color = "#d9622b" if v.startswith("moe") else "#7a7a7a"
        ax.bar(i, statistics.mean(vals), color=color, alpha=0.35, width=0.6)
        ax.scatter([i] * len(vals), vals, color=color, zorder=3, s=14)
    ax.set_xticks(range(len(VARIANTS)), [v.replace("moe-", "moe-\n") for v in VARIANTS], fontsize=7)
    lo = min(c[v]["val"] for c in comp for v in VARIANTS)
    ax.set_ylim(lo - 0.04, lo + 0.08)
    ax.set_ylabel("validation loss at step 3,000")
    ax.set_title(
        f"final loss: dots are seeds ({len(comp)}), bars their mean\n(y axis cut; all within a few hundredths)",
        fontsize=8,
    )
    colors = {
        "dense-wide": "#7a7a7a",
        "moe-none": "#e9a77f",
        "moe-aux": "#3b6fb6",
        "moe-bias": "#d9622b",
        "moe-bias-fine": "#8c3b12",
    }
    for v, colr in colors.items():
        ax2.plot(steps[1:], [statistics.mean(g) for g in gaps[v][1:]], "-o", ms=3, color=colr, label=v)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("step")
    ax2.set_ylabel("val loss minus dense's\n(same seed, same text; below 0 = better)")
    ax2.set_title("paired gap to dense with the same active parameters, mean of seeds", fontsize=8)
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(ASSETS / "compare.png", dpi=150)
    plt.close(fig)

    routes = load("route_lab.json")
    print("\ntoken -> expert, normalised MI per layer (moe-bias): trained | untrained | shuffled")
    for r in routes:
        print(
            "  "
            + "  ".join(
                f"{la['specialisation']['normalised']:.2f}|{la['specialisation']['untrained_model']:.2f}|"
                f"{la['specialisation']['shuffled_control']:.3f}"
                for la in r
            )
        )
    none_routes = [json.loads((d / "none" / "route_lab.json").read_text()) for d in RUNS if (d / "none").exists()]
    print("load shares, moe-none, worst layer per seed (max load):")
    for r in none_routes:
        worst = max(r, key=lambda la: la["load"]["max_load"])
        print(
            f"  layer {worst['layer']}: {worst['load']['shares']}  max load {worst['load']['max_load']:.2f}, "
            f"dead {worst['load']['dead']}"
        )

    abl = load("ablate.json")
    print("\nablation (moe-bias): mean loss change, routed expert off | shared expert off | own > rest")
    for rows in abl:
        routed = [r for r in rows if r["expert"] != "shared"]
        shared = [r for r in rows if r["expert"] == "shared"]
        own = sum(r["own_tokens"] > r["other_tokens"] for r in routed)
        l0 = [r["delta"] for r in routed if r["layer"] == 0]
        later = [r["delta"] for r in routed if r["layer"] > 0]
        print(
            f"  {statistics.mean(r['delta'] for r in routed):+.3f} | {statistics.mean(r['delta'] for r in shared):+.3f}"
            f" | {own}/{len(routed)} | layer 0 routed {statistics.mean(l0):+.3f}, "
            f"later layers {statistics.mean(later):+.3f}"
            f" | own tokens {statistics.mean(r['own_tokens'] for r in routed):+.3f}, rest "
            f"{statistics.mean(r['other_tokens'] for r in routed):+.3f}"
        )

    col = load("collapse_s{s}.json")
    print("\ncollapse: final worst-layer max load (val loss) per seed")
    for name in ("none", "none-skew", "aux-skew", "bias-skew"):
        print(
            f"  {name:10s} "
            + "  ".join(f"{c[name]['worst_max_load']:.2f} ({c[name]['val']:.3f})" for c in col)
            + "   first flagged step: "
            + " ".join(str(c[name]["first_flagged_step"]) for c in col)
        )
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = {"none": "#7a7a7a", "none-skew": "#222222", "aux-skew": "#3b6fb6", "bias-skew": "#d9622b"}
    for name, colr in colors.items():
        for j, c in enumerate(col):
            h = c[name]["history"]
            ax.plot(
                [r["step"] for r in h],
                [max(la["max_load"] for la in r["layers"]) for r in h],
                "-o",
                ms=2.5,
                color=colr,
                alpha=0.8,
                label=name if j == 0 else None,
            )
    ax.axhline(1, color="grey", lw=0.8, ls=":")
    ax.set_ylim(0, 4.3)
    ax.set_xlabel("step")
    ax.set_ylabel("max load, worst layer\n(1 = even, 4 = worst for top-2)")
    ax.set_title(f"expert 0 given a fixed head start (skew 3) in every router; {len(col)} seeds", fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "collapse.png", dpi=150)
    plt.close(fig)

    mtp = load("mtp_s{s}.json")
    print("\nMTP: main val without | with | MTP loss t+2 | MTP acc t+2 | agreement")
    for m in mtp:
        print(
            f"  {m['0']['val']:.3f} | {m['1']['val']:.3f} | {m['mtp_val']:.3f} | {m['mtp_acc']:.3f} | "
            f"{m['mtp_agree']:.3f}"
        )
    if RUNS:
        src = RUNS[0] / "routing_heatmap.png"
        if src.exists():
            (ASSETS / "routing_heatmap.png").write_bytes(src.read_bytes())
    print(f"\nwrote {ASSETS / 'compare.png'}, {ASSETS / 'collapse.png'}, {ASSETS / 'routing_heatmap.png'}")


if __name__ == "__main__":
    main()
