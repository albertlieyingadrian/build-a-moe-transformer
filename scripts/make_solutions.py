"""Regenerate the staged reference solutions from the last one.

solutions/python/17-mtp/code is written by hand. For every earlier stage N this copies only the modules
that stage needs and removes from app/main.py the command functions (and their helpers) that belong to later stages,
so each solution is exactly "the program after stage N".

    python scripts/make_solutions.py
"""

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS = ROOT / "solutions" / "python"
LAST = 17

# the first stage that needs each module
MODULES = {
    "main.py": 1,
    "tokenizer.py": 1,
    "model.py": 2,
    "train.py": 3,
    "interp.py": 14,
}
# helpers in main.py that are not commands, and the first stage that uses them
HELPERS = {
    "load_text": 2,
    "model_cfg": 3,
    "save_ckpt": 3,
    "variant_cfg": 13,
    "plot_curves": 13,
    "load_model": 14,
    "plot_collapse": 16,
}


def stage_slugs():
    text = (ROOT / "course-definition.yml").read_text().split("\nstages:\n", 1)[1]
    return re.findall(r'^  - slug: "([\w-]+)"', text, flags=re.M)


def trim_main(src: str, n: int) -> str:
    commands = {fn: int(st) for _, st, fn in re.findall(r'"([\w-]+)": \((\d+), (cmd_\w+)\)', src)}
    stage_of = {**commands, **HELPERS}
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and stage_of.get(node.name, 0) > n:
            start = node.lineno - 1
            while start > 0 and lines[start - 1].lstrip().startswith("# ---"):  # a section banner above it
                start -= 1
            drop.append((start, node.end_lineno))
    for start, end in sorted(drop, reverse=True):
        del lines[start:end]
        while start < len(lines) - 1 and lines[start].strip() == "" and lines[start - 1].strip() == "":
            del lines[start]
    out = "".join(lines)
    out = "\n".join(
        line
        for line in out.split("\n")
        if not (m := re.match(r'\s+"[\w-]+": \((\d+), cmd_\w+\),', line)) or int(m.group(1)) <= n
    )
    out = re.sub(r"MAX_STAGE = \d+", f"MAX_STAGE = {n}", out)
    return re.sub(r"\n{4,}", "\n\n\n", out)


def main():
    slugs = stage_slugs()
    last = SOLUTIONS / f"{LAST:02d}-{slugs[LAST - 1]}" / "code"
    for d in SOLUTIONS.iterdir():
        if d.name != last.parent.name:
            shutil.rmtree(d)
    for n in range(1, LAST):
        code = SOLUTIONS / f"{n:02d}-{slugs[n - 1]}" / "code"
        (code / "app").mkdir(parents=True)
        shutil.copy2(last / "your_program.sh", code / "your_program.sh")
        for module, first in MODULES.items():
            if first <= n:
                text = (last / "app" / module).read_text()
                if module == "main.py":
                    text = trim_main(text, n)
                (code / "app" / module).write_text(text)
        print(f"wrote {code.relative_to(ROOT)}")
    # trimming main.py can leave imports that only later stages use; let ruff remove them
    staged = [str(d) for d in SOLUTIONS.iterdir() if d.name != last.parent.name]
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--fix", "--select", "F401,I", "--quiet", *staged], check=True
    )
    subprocess.run([sys.executable, "-m", "ruff", "format", "--quiet", *staged], check=True)


if __name__ == "__main__":
    main()
